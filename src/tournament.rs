use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use lash::direct::{
    DirectJsonSchema, DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole,
};
use lash::tools::{
    ToolCall, ToolContext, ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult,
};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::ObliqTools;

const BATCH_SIZE: usize = 20;
// Tournament's recall@top_k is structurally bounded by top_k / pool_size:
// docs eliminated in round 1 land in the appended tail, past every doc that
// survived to round 2+. Preserve a recall reservoir before tournament:
// canonical RRF top-300 plus each pool's top slice, deduped, capped at 600.
// This keeps single-pool hard hits alive without letting every noisy tail in.
const RRF_TOP_N: usize = 300;
const PER_POOL_RESERVOIR: usize = 75;
const MAX_TOURNAMENT_INPUT: usize = 600;
const KEEP_PER_BATCH: usize = 8;
const RRF_K: f64 = 60.0;
const DEFAULT_TOP_K: usize = 100;
const MAX_DOC_CHARS: usize = 6000;
const MAX_JUDGE_DOC_CHARS: usize = 2500;
const MAX_JUDGE_CANDIDATES: usize = 50;
const MAX_PARALLEL_BATCHES: usize = 8;
const SHUFFLE_SEED: u64 = 0x0B11_9EBE_7C8B_AD00;
const DIRECT_RERANK_VARIANT: &str = "low";

pub struct CandidateJudgeProvider {
    obliq: Arc<ObliqTools>,
    description: String,
}

impl CandidateJudgeProvider {
    pub fn new(obliq: Arc<ObliqTools>, description: String) -> Self {
        Self { obliq, description }
    }

    pub fn tool_definition() -> ToolDefinition {
        ToolDefinition::raw(
            "judge_candidates",
            "Calibrate retrieval against a verifier predicate. Provide retrieved document ids \
             plus the latent pattern you are testing. Returns likely positives, \
             surface-similar distractors, unclear cases, surface bait to avoid, a refined \
             predicate, and follow-up queries for the next retrieval pass.",
            json!({
                "type": "object",
                "properties": {
                    "verifier_predicate": { "type": "string", "minLength": 1 },
                    "candidate_doc_ids": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 },
                        "minItems": 1,
                        "maxItems": MAX_JUDGE_CANDIDATES
                    },
                    "surface_bait": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 },
                        "maxItems": 20,
                        "default": []
                    }
                },
                "required": ["verifier_predicate", "candidate_doc_ids"],
                "additionalProperties": false
            }),
            json!({
                "type": "object",
                "properties": {
                    "positive_ids": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 }
                    },
                    "distractor_ids": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 }
                    },
                    "unclear_ids": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 }
                    },
                    "surface_bait": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 }
                    },
                    "refined_predicate": { "type": "string" },
                    "next_queries": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 }
                    }
                },
                "required": [
                    "positive_ids",
                    "distractor_ids",
                    "unclear_ids",
                    "surface_bait",
                    "refined_predicate",
                    "next_queries"
                ],
                "additionalProperties": false
            }),
        )
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    async fn judge(&self, args: &Value, context: &ToolContext) -> Result<Value, String> {
        let verifier_predicate = args
            .get("verifier_predicate")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "missing required parameter: verifier_predicate".to_string())?
            .to_string();
        let candidate_doc_ids = args
            .get("candidate_doc_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| "missing required parameter: candidate_doc_ids".to_string())?;
        let mut candidates = Vec::new();
        let mut seen = BTreeSet::new();
        for value in candidate_doc_ids {
            let Some(id) = value.as_str().map(str::trim).filter(|id| !id.is_empty()) else {
                continue;
            };
            if seen.insert(id.to_string()) {
                candidates.push(id.to_string());
            }
        }
        if candidates.is_empty() {
            return Err("judge_candidates needs at least one candidate id".to_string());
        }
        if candidates.len() > MAX_JUDGE_CANDIDATES {
            candidates.truncate(MAX_JUDGE_CANDIDATES);
        }

        let surface_bait = string_array(args.get("surface_bait"), 20);
        let docs = self
            .obliq
            .fetch_doc_texts(&candidates)
            .await
            .map_err(|err| format!("failed to fetch doc texts: {err}"))?;
        let session_model = context
            .session_model()
            .await
            .map_err(|err| format!("failed to read session model: {err}"))?;

        let raw = judge_candidate_batch(
            context.clone(),
            &session_model.model,
            Some(DIRECT_RERANK_VARIANT),
            context.session_id(),
            context.tool_call_id().map(str::to_string),
            &self.description,
            &verifier_predicate,
            &surface_bait,
            &docs,
            &candidates,
        )
        .await?;

        Ok(clean_judge_output(raw, &candidates, &verifier_predicate))
    }
}

#[async_trait]
impl ToolProvider for CandidateJudgeProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![Self::tool_definition()]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "judge_candidates" => match self.judge(call.args, call.context).await {
                Ok(value) => ToolResult::ok(value),
                Err(err) => ToolResult::err(json!(err)),
            },
            other => ToolResult::err_fmt(format_args!("unknown tool: {other}")),
        }
    }
}

pub struct TournamentRerankProvider {
    obliq: Arc<ObliqTools>,
    description: String,
}

impl TournamentRerankProvider {
    pub fn new(obliq: Arc<ObliqTools>, description: String) -> Self {
        Self { obliq, description }
    }

    pub fn tool_definition() -> ToolDefinition {
        ToolDefinition::raw(
            "tournament_rerank",
            "Listwise tournament reranker. Takes labeled candidate pools (one per channel \
             or probe-set you ran). Each pool uses the same `matches` array returned by \
             `search` and `discover_docs`. The reranker extracts `doc_id` in match order, \
             builds a deduped recall reservoir from canonical Reciprocal Rank Fusion \
             (k=60) plus each pool's top candidates, caps that reservoir at 600, then \
             runs the listwise tournament: \
             shuffles into batches of 20, ranks each batch under the dataset's relevance \
             description, promotes the top 8 of each batch to the next round, and keeps \
             eliminated tails in elimination-depth order. Returns the merged ranking \
             truncated to top_k. Pass each retrieval result set as its own pool — the merge \
             is deterministic and won't drop single-channel hits.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "minLength": 1 },
                    "candidate_pools": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": { "type": "string", "minLength": 1 },
                                "matches": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "rank": { "type": "integer", "minimum": 1 },
                                            "doc_id": { "type": "string", "minLength": 1 },
                                            "score": { "type": "number" },
                                            "text": { "type": "string" },
                                            "metadata": { "type": "object", "additionalProperties": true }
                                        },
                                        "required": ["rank", "doc_id", "score", "text", "metadata"],
                                        "additionalProperties": false
                                    },
                                    "minItems": 1
                                }
                            },
                            "required": ["label", "matches"],
                            "additionalProperties": false
                        },
                        "minItems": 1,
                        "maxItems": 16
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    }
                },
                "required": ["query", "candidate_pools"],
                "additionalProperties": false
            }),
            json!({
                "type": "object",
                "properties": {
                    "ranked_doc_ids": {
                        "type": "array",
                        "items": { "type": "string", "minLength": 1 }
                    }
                },
                "required": ["ranked_doc_ids"],
                "additionalProperties": false
            }),
        )
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    async fn rerank(&self, args: &Value, context: &ToolContext) -> Result<Value, String> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "missing required parameter: query".to_string())?
            .to_string();
        let pools = args
            .get("candidate_pools")
            .and_then(Value::as_array)
            .ok_or_else(|| "missing required parameter: candidate_pools".to_string())?;
        let mut candidates =
            recall_reservoir(pools, RRF_TOP_N, PER_POOL_RESERVOIR, MAX_TOURNAMENT_INPUT);
        if candidates.len() < 2 {
            return Err(format!(
                "tournament_rerank needs >= 2 unique candidates across pools, got {}",
                candidates.len()
            ));
        }
        candidates.truncate(MAX_TOURNAMENT_INPUT);
        let top_k = args
            .get("top_k")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_TOP_K)
            .min(candidates.len());

        let docs = self
            .obliq
            .fetch_doc_texts(&candidates)
            .await
            .map_err(|err| format!("failed to fetch doc texts: {err}"))?;

        let session_model = context
            .session_model()
            .await
            .map_err(|err| format!("failed to read session model: {err}"))?;

        let ranked = run_tournament(
            context.clone(),
            &session_model.model,
            Some(DIRECT_RERANK_VARIANT),
            context.session_id(),
            context.tool_call_id().map(str::to_string),
            &self.description,
            &query,
            &docs,
            candidates,
            BATCH_SIZE,
            KEEP_PER_BATCH,
        )
        .await?;

        let truncated: Vec<String> = ranked.into_iter().take(top_k).collect();
        Ok(json!({ "ranked_doc_ids": truncated }))
    }
}

#[async_trait]
impl ToolProvider for TournamentRerankProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![Self::tool_definition()]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "tournament_rerank" => match self.rerank(call.args, call.context).await {
                Ok(value) => ToolResult::ok(value),
                Err(err) => ToolResult::err(json!(err)),
            },
            other => ToolResult::err_fmt(format_args!("unknown tool: {other}")),
        }
    }
}

fn recall_reservoir(
    pools: &[Value],
    rrf_top_n: usize,
    per_pool_top_n: usize,
    max_candidates: usize,
) -> Vec<String> {
    let ranked = rrf_merge(pools, rrf_top_n);
    let mut out =
        Vec::with_capacity(max_candidates.min(ranked.len() + pools.len() * per_pool_top_n));
    let mut seen = BTreeSet::new();
    for id in ranked {
        push_unique_candidate(&mut out, &mut seen, id, max_candidates);
    }
    for pool in pools {
        for id in pool_doc_ids(pool).into_iter().take(per_pool_top_n) {
            push_unique_candidate(&mut out, &mut seen, id, max_candidates);
            if out.len() >= max_candidates {
                return out;
            }
        }
    }
    out
}

fn push_unique_candidate(
    out: &mut Vec<String>,
    seen: &mut BTreeSet<String>,
    id: String,
    max_candidates: usize,
) {
    if out.len() < max_candidates && seen.insert(id.clone()) {
        out.push(id);
    }
}

fn rrf_merge(pools: &[Value], top_n: usize) -> Vec<String> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut first_seen_order: Vec<String> = Vec::new();
    let mut first_seen: BTreeSet<String> = BTreeSet::new();
    for pool in pools {
        for (rank, id) in pool_doc_ids(pool).into_iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f64);
            if first_seen.insert(id.clone()) {
                first_seen_order.push(id);
            }
        }
    }
    let mut ranked: Vec<(String, f64, usize)> = first_seen_order
        .into_iter()
        .map(|id| {
            let score = *scores.get(&id).unwrap_or(&0.0);
            let order = scores.len();
            (id, score, order)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .take(top_n)
        .map(|(id, _, _)| id)
        .collect()
}

fn pool_doc_ids(pool: &Value) -> Vec<String> {
    let Some(matches) = pool.get("matches").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut ids = Vec::with_capacity(matches.len());
    let mut seen = BTreeSet::new();
    for value in matches {
        let Some(trimmed) = value.get("doc_id").and_then(Value::as_str).map(str::trim) else {
            continue;
        };
        if !trimmed.is_empty() && seen.insert(trimmed.to_string()) {
            ids.push(trimmed.to_string());
        }
    }
    ids
}

#[expect(
    clippy::too_many_arguments,
    reason = "candidate judging needs the session context plus immutable calibration inputs"
)]
async fn judge_candidate_batch(
    context: ToolContext,
    model: &str,
    model_variant: Option<&str>,
    session_id: &str,
    originating_tool_call_id: Option<String>,
    description: &str,
    verifier_predicate: &str,
    surface_bait: &[String],
    docs: &HashMap<String, String>,
    candidates: &[String],
) -> Result<Value, String> {
    let mut user = String::new();
    user.push_str("Benchmark relevance description:\n");
    user.push_str(description);
    user.push_str("\n\nVerifier predicate:\n");
    user.push_str(verifier_predicate);
    if !surface_bait.is_empty() {
        user.push_str("\n\nSurface bait already noticed:\n");
        for bait in surface_bait {
            user.push_str("- ");
            user.push_str(bait);
            user.push('\n');
        }
    }
    user.push_str("\n\nDocuments to judge (doc_id followed by text):\n\n");
    for id in candidates {
        user.push_str("=== ");
        user.push_str(id);
        user.push_str(" ===\n");
        let text = docs.get(id).map(String::as_str).unwrap_or("(missing)");
        user.push_str(&truncate_for_prompt(text, MAX_JUDGE_DOC_CHARS));
        user.push_str("\n\n");
    }
    user.push_str(
        "Return JSON only. Classify supplied ids as positive_ids, distractor_ids, or \
         unclear_ids. A positive satisfies the verifier predicate; a distractor shares \
         misleading surface cues but not the latent pattern. Also return surface_bait \
         terms or themes to avoid, a concise refined_predicate, and up to 8 next_queries \
         that search for attribute carriers rather than repeating the query wording.",
    );

    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "required": [
            "positive_ids",
            "distractor_ids",
            "unclear_ids",
            "surface_bait",
            "refined_predicate",
            "next_queries"
        ],
        "properties": {
            "positive_ids": {
                "type": "array",
                "items": { "type": "string" }
            },
            "distractor_ids": {
                "type": "array",
                "items": { "type": "string" }
            },
            "unclear_ids": {
                "type": "array",
                "items": { "type": "string" }
            },
            "surface_bait": {
                "type": "array",
                "items": { "type": "string" },
                "maxItems": 20
            },
            "refined_predicate": { "type": "string" },
            "next_queries": {
                "type": "array",
                "items": { "type": "string" },
                "maxItems": 8
            }
        }
    });

    let request = DirectRequest {
        model: model.to_string(),
        model_variant: model_variant.map(str::to_string),
        messages: vec![
            DirectMessage {
                role: DirectRole::System,
                parts: vec![DirectPart::Text(
                    "You calibrate retrieval for an analogy benchmark. Judge whether each \
                     document matches the latent predicate, not whether it reuses the query's \
                     wording. Return JSON only."
                        .to_string(),
                )],
            },
            DirectMessage {
                role: DirectRole::User,
                parts: vec![DirectPart::Text(user)],
            },
        ],
        attachments: Vec::new(),
        output: DirectOutputSpec::JsonSchema(DirectJsonSchema {
            name: "judge_candidates_batch".to_string(),
            schema,
            strict: true,
        }),
        stream_events: None,
        session_id: Some(format!("{session_id}-judge")),
        originating_tool_call_id,
    };

    let completion = context
        .direct_completion(request, "judge_candidates")
        .await
        .map_err(|err| format!("direct_completion failed: {err}"))?;
    let text = completion.text.trim();
    serde_json::from_str(text)
        .map_err(|err| format!("malformed JSON from candidate judge: {err}; raw=`{text}`"))
}

fn clean_judge_output(raw: Value, supplied_ids: &[String], fallback_predicate: &str) -> Value {
    let supplied: BTreeSet<String> = supplied_ids.iter().cloned().collect();
    let mut used = BTreeSet::new();
    let positive_ids = clean_id_array(raw.get("positive_ids"), &supplied, &mut used);
    let distractor_ids = clean_id_array(raw.get("distractor_ids"), &supplied, &mut used);
    let mut unclear_ids = clean_id_array(raw.get("unclear_ids"), &supplied, &mut used);
    for id in supplied_ids {
        if used.insert(id.clone()) {
            unclear_ids.push(id.clone());
        }
    }

    let refined_predicate = raw
        .get("refined_predicate")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback_predicate)
        .to_string();

    json!({
        "positive_ids": positive_ids,
        "distractor_ids": distractor_ids,
        "unclear_ids": unclear_ids,
        "surface_bait": string_array(raw.get("surface_bait"), 20),
        "refined_predicate": refined_predicate,
        "next_queries": string_array(raw.get("next_queries"), 8)
    })
}

fn clean_id_array(
    value: Option<&Value>,
    supplied: &BTreeSet<String>,
    used: &mut BTreeSet<String>,
) -> Vec<String> {
    let Some(array) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in array {
        let Some(id) = item.as_str().map(str::trim).filter(|id| !id.is_empty()) else {
            continue;
        };
        if supplied.contains(id) && used.insert(id.to_string()) {
            out.push(id.to_string());
        }
    }
    out
}

fn string_array(value: Option<&Value>, max_items: usize) -> Vec<String> {
    let Some(array) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for item in array {
        let Some(text) = item.as_str().map(str::trim).filter(|text| !text.is_empty()) else {
            continue;
        };
        if seen.insert(text.to_string()) {
            out.push(text.to_string());
        }
        if out.len() >= max_items {
            break;
        }
    }
    out
}

fn truncate_for_prompt(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut truncated: String = text.chars().take(max_chars).collect();
    truncated.push_str("\n[...truncated]");
    truncated
}

#[expect(
    clippy::too_many_arguments,
    reason = "listwise benchmark orchestration passes independent run inputs explicitly"
)]
async fn run_tournament(
    context: ToolContext,
    model: &str,
    model_variant: Option<&str>,
    session_id: &str,
    originating_tool_call_id: Option<String>,
    description: &str,
    query: &str,
    docs: &HashMap<String, String>,
    candidates: Vec<String>,
    batch: usize,
    keep: usize,
) -> Result<Vec<String>, String> {
    let mut survivors = candidates;
    let mut tails: Vec<Vec<String>> = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(SHUFFLE_SEED);
    let semaphore = Arc::new(Semaphore::new(MAX_PARALLEL_BATCHES));

    while survivors.len() > batch {
        survivors.shuffle(&mut rng);
        let mut batches: Vec<Vec<String>> = survivors
            .chunks(batch)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut tasks: JoinSet<(usize, Result<Vec<String>, String>)> = JoinSet::new();
        for (idx, batch_ids) in batches.drain(..).enumerate() {
            let context = context.clone();
            let model = model.to_string();
            let model_variant = model_variant.map(str::to_string);
            let session_id = session_id.to_string();
            let description = description.to_string();
            let query = query.to_string();
            let docs = docs.clone();
            let semaphore = semaphore.clone();
            let originating = originating_tool_call_id.clone();
            tasks.spawn(async move {
                let _permit = match semaphore.acquire_owned().await {
                    Ok(permit) => permit,
                    Err(err) => return (idx, Err(format!("semaphore closed: {err}"))),
                };
                let result = rank_batch(
                    context,
                    &model,
                    model_variant.as_deref(),
                    &session_id,
                    originating,
                    &description,
                    &query,
                    &docs,
                    batch_ids,
                )
                .await;
                (idx, result)
            });
        }

        let mut completed: Vec<(usize, Result<Vec<String>, String>)> = Vec::new();
        while let Some(joined) = tasks.join_next().await {
            let pair = joined.map_err(|err| format!("rerank task join failed: {err}"))?;
            completed.push(pair);
        }
        completed.sort_by_key(|(idx, _)| *idx);

        let mut next = Vec::new();
        let mut tail = Vec::new();
        for (_, res) in completed {
            let ranked = res?;
            let split = ranked.len().min(keep);
            next.extend(ranked[..split].iter().cloned());
            tail.extend(ranked[split..].iter().cloned());
        }
        tails.push(tail);
        survivors = next;
    }

    let final_ranked = rank_batch(
        context,
        model,
        model_variant,
        session_id,
        originating_tool_call_id.clone(),
        description,
        query,
        docs,
        survivors,
    )
    .await?;

    let mut out =
        Vec::with_capacity(final_ranked.len() + tails.iter().map(Vec::len).sum::<usize>());
    out.extend(final_ranked);
    for tail in tails.into_iter().rev() {
        out.extend(tail);
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "direct batch completion needs the session context plus immutable ranking inputs"
)]
async fn rank_batch(
    context: ToolContext,
    model: &str,
    model_variant: Option<&str>,
    session_id: &str,
    originating_tool_call_id: Option<String>,
    description: &str,
    query: &str,
    docs: &HashMap<String, String>,
    batch_ids: Vec<String>,
) -> Result<Vec<String>, String> {
    if batch_ids.len() <= 1 {
        return Ok(batch_ids);
    }
    let n = batch_ids.len();

    let mut user = String::new();
    user.push_str("Relevance description:\n");
    user.push_str(description);
    user.push_str("\n\nQuery:\n");
    user.push_str(query);
    user.push_str("\n\nDocuments to rank (doc_id followed by text):\n\n");
    for id in &batch_ids {
        user.push_str("=== ");
        user.push_str(id);
        user.push_str(" ===\n");
        let text = docs.get(id).map(String::as_str).unwrap_or("(missing)");
        if text.len() > MAX_DOC_CHARS {
            user.push_str(&text[..MAX_DOC_CHARS]);
            user.push_str("\n[...truncated]");
        } else {
            user.push_str(text);
        }
        user.push_str("\n\n");
    }
    user.push_str(&format!(
        "Return JSON {{\"ranked\": [<doc_id>, ...]}} with EXACTLY {n} entries: every \
         supplied doc_id appears once, ordered from MOST to LEAST relevant under the \
         relevance description above. Do not omit, repeat, or invent IDs.",
    ));

    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["ranked"],
        "properties": {
            "ranked": {
                "type": "array",
                "items": { "type": "string" },
                "minItems": n,
                "maxItems": n
            }
        }
    });

    let request = DirectRequest {
        model: model.to_string(),
        model_variant: model_variant.map(str::to_string),
        messages: vec![
            DirectMessage {
                role: DirectRole::System,
                parts: vec![DirectPart::Text(
                    "You are a relevance reranker. Rank the supplied documents from most to \
                     least relevant to the query under the relevance description. Return \
                     JSON only."
                        .to_string(),
                )],
            },
            DirectMessage {
                role: DirectRole::User,
                parts: vec![DirectPart::Text(user)],
            },
        ],
        attachments: Vec::new(),
        output: DirectOutputSpec::JsonSchema(DirectJsonSchema {
            name: "tournament_rerank_batch".to_string(),
            schema,
            strict: true,
        }),
        stream_events: None,
        session_id: Some(format!("{session_id}-tr")),
        originating_tool_call_id,
    };

    let completion = context
        .direct_completion(request, "tournament_rerank")
        .await
        .map_err(|err| format!("direct_completion failed: {err}"))?;

    let text = completion.text.trim();
    let value: Value = serde_json::from_str(text)
        .map_err(|err| format!("malformed JSON from reranker: {err}; raw=`{text}`"))?;
    let ranked_array = value
        .get("ranked")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("reranker output missing 'ranked' array: {text}"))?;
    let ranked_ids: Vec<String> = ranked_array
        .iter()
        .filter_map(|item| item.as_str().map(String::from))
        .collect();

    let supplied: BTreeSet<String> = batch_ids.iter().cloned().collect();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut clean = Vec::with_capacity(batch_ids.len());
    for id in &ranked_ids {
        if supplied.contains(id) && seen.insert(id.clone()) {
            clean.push(id.clone());
        }
    }
    for id in &batch_ids {
        if seen.insert(id.clone()) {
            clean.push(id.clone());
        }
    }
    Ok(clean)
}
