use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use lash::{
    DirectJsonSchema, DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole,
    SessionPolicy, ToolDefinition, ToolExecutionContext, ToolExecutionMode, ToolHookHost,
    ToolProvider, ToolResult,
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
// survived to round 2+. With pool=864, keep=5 → only 220/864 docs reach
// rank 100. Cap the input pool at 300 instead of growing keep — this gives
// every candidate a real shot at top-100 (300/300 = full coverage with
// some tail-of-tail spillover). Keep=8 then gives gold ~40% per-round
// survival probability instead of 25%.
const MAX_TOURNAMENT_INPUT: usize = 300;
const KEEP_PER_BATCH: usize = 8;
const DEFAULT_TOP_K: usize = 100;
const MAX_DOC_CHARS: usize = 6000;
const MAX_PARALLEL_BATCHES: usize = 8;
const SHUFFLE_SEED: u64 = 0x0B11_9EBE_7C8B_AD00;

pub struct TournamentRerankProvider {
    obliq: Arc<ObliqTools>,
    description: String,
}

impl TournamentRerankProvider {
    pub fn new(obliq: Arc<ObliqTools>, description: String) -> Self {
        Self { obliq, description }
    }

    pub fn tool_definition() -> ToolDefinition {
        ToolDefinition::new(
            "tournament_rerank",
            "Listwise tournament reranker. Internally truncates candidate_doc_ids to the \
             first 300 entries; pass them in your preferred prefilter order (RRF rank \
             across channels recommended). Anything past index 300 is dropped silently. \
             Shuffles into batches of 20, has the LLM rank each batch under the dataset's \
             relevance description, promotes the top 8 of each batch to the next round, \
             and keeps the eliminated tails in elimination-depth order. Returns the \
             merged ranking truncated to top_k. Use this to produce the final ranking \
             instead of hand-ordering ids.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "candidate_doc_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 2,
                        "maxItems": 2000
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    }
                },
                "required": ["query", "candidate_doc_ids"],
                "additionalProperties": false
            }),
            json!({
                "type": "object",
                "properties": {
                    "ranked_doc_ids": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["ranked_doc_ids"],
                "additionalProperties": true
            }),
        )
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    async fn rerank(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "missing required parameter: query".to_string())?
            .to_string();
        let raw = args
            .get("candidate_doc_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| "missing required parameter: candidate_doc_ids".to_string())?;

        let mut seen = BTreeSet::new();
        let mut candidates: Vec<String> = Vec::new();
        for value in raw {
            if let Some(text) = value.as_str() {
                let trimmed = text.trim();
                if !trimmed.is_empty() && seen.insert(trimmed.to_string()) {
                    candidates.push(trimmed.to_string());
                }
            }
        }
        if candidates.len() < 2 {
            return Err(format!(
                "tournament_rerank needs >= 2 unique candidate_doc_ids, got {}",
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

        let snapshot = context
            .host
            .snapshot_session(&context.session_id)
            .await
            .map_err(|err| format!("failed to snapshot session: {err}"))?;
        let policy = snapshot.policy;

        let ranked = run_tournament(
            &context.host,
            &policy,
            &context.session_id,
            context.tool_call_id.clone(),
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

    async fn execute(&self, name: &str, _args: &Value) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{name}` requires session context and cannot run without it"
        ))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "tournament_rerank" => match self.rerank(args, context).await {
                Ok(value) => ToolResult::ok(value),
                Err(err) => ToolResult::err(json!(err)),
            },
            other => ToolResult::err_fmt(format_args!("unknown tool: {other}")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
        _progress: Option<&lash::ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

async fn run_tournament(
    host: &Arc<dyn ToolHookHost>,
    policy: &SessionPolicy,
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
        let mut batches: Vec<Vec<String>> =
            survivors.chunks(batch).map(|chunk| chunk.to_vec()).collect();

        let mut tasks: JoinSet<(usize, Result<Vec<String>, String>)> = JoinSet::new();
        for (idx, batch_ids) in batches.drain(..).enumerate() {
            let host = host.clone();
            let policy = policy.clone();
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
                    host,
                    &policy,
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
        host.clone(),
        policy,
        session_id,
        originating_tool_call_id.clone(),
        description,
        query,
        docs,
        survivors,
    )
    .await?;

    let mut out = Vec::with_capacity(final_ranked.len() + tails.iter().map(Vec::len).sum::<usize>());
    out.extend(final_ranked);
    for tail in tails.into_iter().rev() {
        out.extend(tail);
    }
    Ok(out)
}

async fn rank_batch(
    host: Arc<dyn ToolHookHost>,
    policy: &SessionPolicy,
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
        model: policy.model.clone(),
        model_variant: policy.model_variant.clone(),
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

    let completion = host
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
