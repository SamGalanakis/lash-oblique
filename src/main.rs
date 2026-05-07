use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use clap::{Parser, Subcommand};
use lash::plugin::{PluginSpec, StaticPluginFactory};
use lash::provider::LashConfig;
use lash::{
    ExecutionMode, JsonlTraceSink, PluginFactory, RuntimePersistence, SessionPolicy,
    SessionStoreCreateRequest, SessionStoreFactory, ToolDefinition, ToolExecutionMode,
    ToolProvider, ToolResult, TraceContext, TraceLevel, TraceSink,
};
use lash_embed::{Input, LashCore, ModePreset, ModeTurnOptions};
use lash_export::{ExportFormat, load_session_from_paths, render};
use lash_rlm_types::RlmTermination;
use lash_sqlite_store::Store;
use lash_subagents::{
    CapabilityField, CapabilityOptionalField, CapabilityRecursion, CapabilityRegistry,
    CapabilitySpec, CapabilityToolSurface, LocalSubagentHost, StaticCapability, SubagentHost,
    SubagentsPluginFactory,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

const DEFAULT_DATA_DIR: &str = ".benchmarks/obliq/data";
const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";
const DEFAULT_COLLECTION: &str = "obliq_math";
const DEFAULT_MODEL: &str = "gpt-5.5";
const DEFAULT_VARIANT: &str = "medium";
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const SUBAGENT_CAPABILITY: &str = "explore";

#[derive(Parser, Debug)]
#[command(name = "lash-oblique")]
#[command(about = "Run OBLIQ-Bench math queries through Lash RLM with Qdrant retrieval tools.")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    RunMath {
        #[arg(long)]
        query_id: String,
        #[arg(long, default_value = DEFAULT_DATA_DIR)]
        data_dir: PathBuf,
        #[arg(long, default_value = DEFAULT_QDRANT_URL)]
        qdrant_url: String,
        #[arg(long, default_value = DEFAULT_COLLECTION)]
        collection: String,
        #[arg(long)]
        provider_id: Option<String>,
        #[arg(long, default_value = DEFAULT_MODEL)]
        model: String,
        #[arg(long, default_value = DEFAULT_VARIANT)]
        variant: String,
        #[arg(long, default_value_t = DEFAULT_MAX_CONTEXT_TOKENS)]
        max_context_tokens: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    EvalMath {
        #[arg(long)]
        query_id: String,
        #[arg(long)]
        submission: PathBuf,
        #[arg(long, default_value = DEFAULT_DATA_DIR)]
        data_dir: PathBuf,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct QueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RankedSubmission {
    ranked_doc_ids: Vec<String>,
}

#[derive(Debug)]
struct SanitizedSubmission {
    ranked_doc_ids: Vec<String>,
    removed_doc_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RunOutput {
    query_id: String,
    query: String,
    raw_ranked_doc_ids: Vec<String>,
    ranked_doc_ids: Vec<String>,
    excluded_doc_ids: Vec<String>,
    removed_doc_ids: Vec<String>,
    metrics: Option<Metrics>,
    tool_calls: usize,
    final_text: String,
    errors: Vec<String>,
    artifacts: RunArtifacts,
}

#[derive(Debug, Clone, Serialize)]
struct RunArtifacts {
    output_json: String,
    session_db: String,
    trace_jsonl: String,
    trace_html: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct Metrics {
    ndcg_at_10: f64,
    ndcg_at_50: f64,
    recall_at_10: f64,
    recall_at_50: f64,
    recall_at_100: f64,
    gold_count: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let args = Args::parse();
    match args.command {
        Command::RunMath {
            query_id,
            data_dir,
            qdrant_url,
            collection,
            provider_id,
            model,
            variant,
            max_context_tokens,
            output,
        } => {
            let query = load_math_query(&data_dir, &query_id)?;
            let excluded = load_excluded_doc_ids(&data_dir, &query_id)?;
            let stats = load_math_stats(&data_dir).unwrap_or_default();
            let provider = resolve_provider(provider_id.as_deref())?;
            let artifacts = RunArtifacts::from_output(output.as_ref(), &query_id);
            artifacts.create_parent_dirs()?;
            let store = Arc::new(
                Store::open(Path::new(&artifacts.session_db))
                    .with_context(|| format!("open {}", artifacts.session_db))?,
            );
            let tools = Arc::new(ObliqTools {
                script: PathBuf::from("scripts/query_math_qdrant.py"),
                python: python_bin(),
                data_dir,
                qdrant_url,
                collection,
                excluded_doc_ids: excluded.clone(),
                late_available: false,
            });
            let late_available = tools.probe_late_available().await.unwrap_or(false);
            let tools = Arc::new(ObliqTools {
                late_available,
                ..(*tools).clone()
            });
            let core = build_core(
                provider,
                model,
                variant,
                max_context_tokens,
                tools.clone(),
                store.clone() as Arc<dyn RuntimePersistence>,
                PathBuf::from(&artifacts.trace_jsonl),
                &query,
            )?;
            let session = core
                .session(format!("obliq-math-{}", uuid::Uuid::new_v4()))
                .rlm()
                .open()
                .await
                .context("open Lash RLM session")?;
            let schema = output_schema();
            let turn = session
                .turn(Input::text(run_prompt(&query, &stats, &excluded, late_available)))
                .mode_turn_options(
                    ModeTurnOptions::typed(
                        ExecutionMode::new("rlm"),
                        RlmTermination::Finish {
                            schema: Some(schema),
                            include_submit_prompt: true,
                        },
                    )
                    .map_err(anyhow::Error::msg)?,
                )
                .run()
                .await
                .context("run OBLIQ math query")?;

            let submitted = match &turn.outcome {
                lash::TurnOutcome::Finished(lash::TurnFinish::Submission { value, .. }) => {
                    parse_submission(value.clone())?
                }
                other => bail!(
                    "RLM did not submit ranked_doc_ids: outcome={other:?} errors={:?} text={}",
                    turn.errors,
                    turn.final_text
                ),
            };
            let raw_ranked_doc_ids = submitted.ranked_doc_ids;
            let sanitized = sanitize_ranked_doc_ids(raw_ranked_doc_ids.clone(), &excluded);
            let metrics = load_qrels(&tools.data_dir, &query.id)
                .ok()
                .map(|gold| score_submission(&sanitized.ranked_doc_ids, &gold));
            if let Err(err) = export_single_session_html(
                Path::new(&artifacts.session_db),
                Path::new(&artifacts.trace_jsonl),
                Path::new(&artifacts.trace_html),
            ) {
                eprintln!("warn: failed to render trace html: {err:#}");
            }
            let run_output = RunOutput {
                query_id: query.id,
                query: query.text,
                raw_ranked_doc_ids,
                ranked_doc_ids: sanitized.ranked_doc_ids,
                excluded_doc_ids: excluded.iter().cloned().collect(),
                removed_doc_ids: sanitized.removed_doc_ids,
                metrics,
                tool_calls: turn.tool_calls.len(),
                final_text: turn.final_text,
                errors: turn.errors.into_iter().map(|issue| issue.message).collect(),
                artifacts: artifacts.clone(),
            };
            let json = serde_json::to_string_pretty(&run_output)?;
            if let Some(output) = output {
                fs::write(&output, format!("{json}\n"))
                    .with_context(|| format!("write {}", output.display()))?;
            } else {
                println!("{json}");
            }
        }
        Command::EvalMath {
            query_id,
            submission,
            data_dir,
        } => {
            let raw = fs::read_to_string(&submission)
                .with_context(|| format!("read {}", submission.display()))?;
            let value: Value = serde_json::from_str(&raw)
                .with_context(|| format!("parse {}", submission.display()))?;
            let submitted = parse_submission(value)?;
            let excluded = load_excluded_doc_ids(&data_dir, &query_id)?;
            let sanitized = sanitize_ranked_doc_ids(submitted.ranked_doc_ids, &excluded);
            let gold = load_qrels(&data_dir, &query_id)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&score_submission(&sanitized.ranked_doc_ids, &gold))?
            );
        }
    }
    Ok(())
}

fn build_core(
    provider: lash::ProviderHandle,
    model: String,
    variant: String,
    max_context_tokens: usize,
    obliq_tools: Arc<ObliqTools>,
    store: Arc<dyn RuntimePersistence>,
    trace_path: PathBuf,
    query: &QueryRow,
) -> anyhow::Result<LashCore> {
    let policy = SessionPolicy {
        provider: provider.clone(),
        model: model.clone(),
        model_variant: Some(variant.clone()),
        max_context_tokens: Some(max_context_tokens),
        execution_mode: ExecutionMode::new("rlm"),
        standard_context_approach: None,
        ..SessionPolicy::default()
    };
    let tool_surface = rlm_tool_surface(obliq_tools.clone());
    let list_async = Arc::new(ListAsyncHandlesTool);
    let subagents = Arc::new(SubagentsPluginFactory::new(
        policy,
        Arc::new(
            CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
                SUBAGENT_CAPABILITY,
                CapabilitySpec {
                    model: CapabilityField::Inherit,
                    model_variant: CapabilityOptionalField::Inherit,
                    execution_mode: CapabilityField::Inherit,
                    tool_surface: CapabilityToolSurface::Explicit(tool_surface),
                    recursion: CapabilityRecursion::Inherit,
                },
            ))),
        ),
        Arc::new(LocalSubagentHost::default()) as Arc<dyn SubagentHost>,
    ));

    LashCore::builder()
        .install_mode(ModePreset::rlm())
        .provider(provider)
        .model(model)
        .model_variant(variant)
        .max_context_tokens(max_context_tokens)
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .trace_sink(Some(Arc::new(JsonlTraceSink::new(trace_path)) as Arc<dyn TraceSink>))
        .trace_level(TraceLevel::Extended)
        .trace_context(trace_context_for_query(query))
        .tools(obliq_tools)
        .plugin(Arc::new(
            lash::BuiltinToolResultProjectionPluginFactory::default(),
        ))
        .plugin(Arc::new(lash_llm_tools::LlmToolsPluginFactory))
        .plugin(Arc::new(StaticPluginFactory::new(
            "obliq_async_handles",
            PluginSpec::new().with_tool_provider(list_async),
        )) as Arc<dyn PluginFactory>)
        .plugin(subagents as Arc<dyn PluginFactory>)
        .build()
        .map_err(anyhow::Error::msg)
}

#[derive(Clone)]
struct ReusableStoreFactory {
    store: Arc<dyn RuntimePersistence>,
}

impl SessionStoreFactory for ReusableStoreFactory {
    fn create_store(
        &self,
        _request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        Ok(Arc::clone(&self.store))
    }
}

fn trace_context_for_query(query: &QueryRow) -> TraceContext {
    let mut metadata = BTreeMap::new();
    metadata.insert("benchmark".to_string(), json!("obliq-bench"));
    metadata.insert("subset".to_string(), json!("math"));
    TraceContext {
        run_id: Some(format!("obliq-math-{}", query.id)),
        example_id: Some(query.id.clone()),
        split: Some("math".to_string()),
        metadata,
        ..TraceContext::default()
    }
}

fn rlm_tool_surface(obliq_tools: Arc<ObliqTools>) -> Vec<ToolDefinition> {
    let mut tools = obliq_tools.definitions();
    tools.push(lash_llm_tools::llm_query_tool_definition());
    tools.push(lash_mode_rlm::continue_as_tool_definition());
    tools.push(list_async_handles_tool_definition());
    tools.push(lash_subagents::spawn_agent_tool_definition(&[
        SUBAGENT_CAPABILITY.to_string(),
    ]));
    tools
}

#[derive(Debug, Clone, Default)]
struct MathStats {
    corpus_docs: usize,
}

fn run_prompt(
    query: &QueryRow,
    stats: &MathStats,
    excluded: &BTreeSet<String>,
    late_available: bool,
) -> String {
    let excluded_text = if excluded.is_empty() {
        "No document IDs are excluded for this query.".to_string()
    } else {
        format!(
            "Do not submit these excluded document IDs for this query: {}",
            excluded.iter().cloned().collect::<Vec<_>>().join(", ")
        )
    };
    let late_tool_text = if late_available {
        "- `late_search`: late-interaction retrieval. Use it for reranking semantic candidates when fine-grained token interactions matter.\n"
    } else {
        ""
    };
    let hybrid_text = if late_available {
        "- `hybrid_search`: runs multiple probes and fuses BM25/dense results with late-interaction evidence. This is usually the best first broad candidate generator."
    } else {
        "- `hybrid_search`: runs multiple probes and fuses BM25/dense results. This is usually the best first broad candidate generator."
    };
    format!(
        r#"You are running one query from OBLIQ-Bench.

Benchmark context:
- OBLIQ-Bench tests oblique retrieval: the query and relevant documents often do not share surface terms.
- The task asks for documents with the same latent relevance pattern, "aha moment", reasoning strategy, stance, failure mode, scenario, or abstract relation as the query, even when topic, notation, and vocabulary differ.
- The retrieval unit is a whole corpus document. Do not invent document IDs; every submitted ID must come from tool results.
- Excluded source/near-duplicate documents are invalid submissions. Retrieval tools hide excluded IDs when possible; never submit an excluded ID if you see one elsewhere.
- Evaluation uses ranked retrieval metrics: NDCG@10, NDCG@50, Recall@10, Recall@50, and Recall@100.

Current query id: `{}`

{}

Current query text:
{}

Available tools:
- `bm25_search`: lexical sparse retrieval. Good for exact terms, named entities, distinctive phrases, and structural vocabulary.
- `dense_search`: dense semantic retrieval. Good for meaning-level similarity when surface terms differ.
- `hybrid_search`: {}
- `discover_docs`: after inspecting candidates, use positive-vs-negative document pairs to guide a target query.
{}
- `fetch_docs`: inspect full text for selected candidate IDs.
- `spawn_agent`: dispatch independent retrieval theories in parallel. Use this for real fanout, not just commentary.
- `llm_query`: use as a direct LLM classifier/filter/verifier over passages you already retrieved. It cannot search or use tools; it only judges supplied inputs.

Professional searcher protocol:
- Use Bates-style named search tactics. Treat every search as a deliberate move, not as one magic query:
  - Monitoring tactics: track which hypothesis buckets, retrieval channels, and latent fingerprints are covered or missing.
  - File-structure tactics: vary where evidence comes from by using BM25, dense, hybrid, fetched full text, subagent scouts, and discovery search.
  - Formulation tactics: reformulate the query as mechanism fingerprints, false-friend tests, scenario templates, role relations, and cross-domain analogies.
  - Term tactics: vary terminology, notation, synonyms, antonyms, abstraction level, and generic relation language; include probes that deliberately avoid the query's surface words.
- Use berrypicking. Let the query evolve after each useful hit. Accumulate small pieces of evidence from many searches instead of trying to build one perfect query up front.
- Use systematic-review search guardrails. Translate the problem into separate concept blocks, check for missing synonyms and variants, validate each strategy against fetched seed examples, and run an independent peer-review step with `llm_query` before final pruning.
- Use pearl-growing/snowballing through `discover_docs`. Once you identify likely analogues, grow outward from their shared features and from positive-vs-negative examples. Do not only rerank the original query neighborhood.

Surface-anchor audit:
- Before searching, identify the query's obvious surface anchors: exact terms, named entities, notation, domain labels, headline operations, and distinctive phrasing.
- Surface anchors are useful for one retrieval branch, but they are also the easiest trap. Do not let every scout and query family reuse them.
- Create at least three non-surface hypotheses by asking:
  - What roles are present if all names and domain words are removed?
  - What relation connects the main actors or objects?
  - What constraint is doing the work?
  - What changes, stays invariant, is transferred, or is ruled out?
  - What is the smallest abstract scenario that would still have the same "aha"?
  - What would this pattern look like in a different domain with different vocabulary?
  - What false friends share the surface words but miss the hidden relation?
- Run one surface-anchor search, then run separate abstraction/analogy searches that avoid the strongest surface anchors and use only roles, relations, constraints, and outcomes.
- When assigning subagents, give at least one scout an anti-literal task: search for candidates using no obvious surface anchors from the query, only the latent relation and outcome.

Examples of strong retrieval behavior:
- Example 1: after reading the query, split it into several independent hypotheses before searching. One hypothesis keeps the query's exact terms and entities. Another rewrites the query as roles and relations without surface vocabulary. Another asks what different domain could express the same hidden structure. Another searches for false friends that should be excluded. Dispatch those hypotheses through different search channels or scouts, then merge their candidate IDs before judging.
- Example 2: after fetching a few candidates, label some as likely positives and some as false friends based on the latent pattern, not surface overlap. Use those pairs with `discover_docs` to grow the pool toward the hidden relevance target. Then use `llm_query` on the merged pool to audit coverage, reject false friends, and rerank, while retaining plausible tail candidates for recall.
- Example 3: if the query mentions a named process, operation, or setting, first search those terms directly. Then deliberately rephrase without them: "a stronger condition on a derived object forces a property of the original", "a transformed candidate still satisfies the same criterion", "self-equivalences force a canonical form", or "a local rule determines a global structure". These are examples of the shape of rephrasing, not fixed templates.
- Example 4: if early results are all near the surface wording, pause and ask what is missing from the search space: action, uniqueness, obstruction, symmetry, transfer, extremality, classification, reconstruction, conservation, equivalence, impossibility, or stability. Use whichever of those roles actually fit the query to create new search probes.

When to use `spawn_agent` vs `llm_query`:
- Use `spawn_agent` for independent search work: a different relevance theory, vocabulary family, domain transfer, or search route that can run without waiting for your local work. Give each scout a focused retrieval hypothesis and ask it to return candidate IDs plus why they fit or fail the latent pattern.
- Use local search tools directly when you already know the next probes and only need retrieval results.
- Use `llm_query` only after you have data in hand. It is for judging supplied documents, comparing latent patterns, spotting false friends, auditing the search, or reranking a candidate pool. It cannot search, fetch, or repair a thin candidate pool.
- Do not substitute `llm_query` for search. Do not substitute subagents for verification. Search/scout first, fetch evidence, then use `llm_query` to judge and audit.

Subagent scouting phase:
- You must use `spawn_agent` before final ranking unless there are fewer than 100 searchable corpus documents.
- Spawn independent scouts whose jobs are to produce candidate sets matching different retrieval criteria or latent-relevance theories. They should not all search the same words.
- Give each scout a clear criterion, such as:
  - surface-anchor scout: exact terms, entities, phrases, notation, or structural vocabulary from the query;
  - abstraction scout: the query rewritten as roles, relations, mechanisms, constraints, and conclusion without surface terms;
  - analogy scout: a different domain or setting where the same latent relation could appear with different vocabulary;
  - counterexample scout: likely false friends that share surface terms but not the intended latent pattern;
  - pearl-growing scout: expand outward from promising positives and contrast against negatives.
- Merge all scout candidate sets into the main pool. A scout result is useful only if its IDs are considered alongside local BM25/dense/hybrid results.

Research rubric:
1. Search iteratively, not one-shot. Start broad, inspect results, update hypotheses, and search again. A good run should show the query evolving after useful hits.
2. Separate the query into concepts: setting, actors or objects, constraints, relationships, desired outcome, latent move, and likely false friends. Search these separately and in combinations.
3. Use named search moves:
   - coverage check: what parts of the concept space have not been searched?
   - channel shift: try BM25, dense, hybrid, fetched full text, discovery, or subagent scouts;
   - reformulation: rewrite the latent pattern in different language and at different abstraction levels;
   - vocabulary shift: use synonyms, adjacent terms, generic relation language, and queries that avoid the surface words.
4. Preserve candidates before judging. Build a broad candidate pool from all search routes. Do not let strong candidates disappear because they came from a tail rank, a weird query, or only one channel.
5. Run the subagent scouting phase. Use parallel scouts to widen the pool across different criteria, then merge their candidates with local retrieval results.
6. Fetch enough evidence before pruning. If a candidate appears high in any search, appears across multiple searches, or has a preview matching a non-surface relevance pattern, inspect it or keep it alive until there is a clear pattern-level reason to reject it.
7. Verify by latent pattern. Use `llm_query` over fetched documents or candidate batches to compare the abstract relation or reasoning pattern. Judge whether the candidate matches the query's hidden relevance target, not whether it shares topic words.
8. Grow from pearls. Once a likely analogue is found, use it as a seed: search near its shared features, contrast it with false friends, and use `discover_docs` from positive/negative examples.
9. Audit before final ranking:
   - Did you over-focus on the query's surface vocabulary?
   - Did you miss synonym families or adjacent pattern templates?
   - Did you use the retrieval channels that could help?
   - Did a high-signal candidate disappear without being inspected or rejected?
   - Is the final ranking drawn from the broad pool, not from a small hand-picked subset?
10. Rank for recall first. The top ranks should be the best latent-pattern matches. The tail should preserve plausible candidates from diverse search routes, not near-duplicates of one lexical theme.
11. Submit exactly 100 unique, non-excluded corpus document IDs, ranked best first.

Operational constraints:
- The benchmark rewards surfacing oblique analogues. Basic lexical/semantic calls are not enough.
- Your search should look creative and hypothesis-driven in the trace: multiple theories, parallel dispatch, merge, verify, and iterate.
- Do not rush from first-page retrieval results to a final answer. The common failure mode is finding the right document in a broad search and then losing it during manual pruning.
- The second common failure mode is letting `llm_query` rank only a small hand-pruned preview batch. Do not do that.
- Do not submit before using `fetch_docs` and at least one `llm_query` verification/reranking call.
- Do not drop high-signal candidates silently. If a document appears high in any search or appears across multiple independent searches, it must be inspected, retained, or rejected for a pattern-level reason before final ranking.
- If a tool returns many candidates, keep them in variables and print only small summaries.

Your final submission must be:
```lashlang
submit {{ ranked_doc_ids: [/* exactly 100 strings */] }}
```

Do not submit explanations in the object. If you have fewer than 100 high-confidence candidates, fill the tail with the best remaining retrieved candidates in ranked order. Do not stop at fewer than 100 IDs.

Dataset note:
- You are searching the OBLIQ-Bench math subset: a corpus of whole mathematical problems where relevance is defined by shared proof strategy or "aha moment", not shared surface vocabulary.
- The searchable math corpus has {} documents."#,
        query.id,
        excluded_text,
        query.text,
        hybrid_text.strip_prefix("- `hybrid_search`: ").unwrap_or(hybrid_text),
        late_tool_text,
        stats.corpus_docs
    )
}

fn output_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["ranked_doc_ids"],
        "properties": {
            "ranked_doc_ids": {
                "type": "array",
                "minItems": 100,
                "maxItems": 100,
                "items": { "type": "string" }
            }
        }
    })
}

fn parse_submission(value: Value) -> anyhow::Result<RankedSubmission> {
    if value.get("ranked_doc_ids").is_some() {
        return serde_json::from_value(value).context("parse ranked_doc_ids submission");
    }
    if let Some(ids) = value
        .get("submission")
        .and_then(|v| v.get("ranked_doc_ids"))
    {
        return Ok(RankedSubmission {
            ranked_doc_ids: serde_json::from_value(ids.clone())?,
        });
    }
    bail!("submission did not contain ranked_doc_ids")
}

#[derive(Clone)]
struct ObliqTools {
    script: PathBuf,
    python: PathBuf,
    data_dir: PathBuf,
    qdrant_url: String,
    collection: String,
    excluded_doc_ids: BTreeSet<String>,
    late_available: bool,
}

#[async_trait]
impl ToolProvider for ObliqTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let hybrid_description = if self.late_available {
            "Run hybrid retrieval for several probes, combining lexical and dense evidence with late-interaction evidence. This is usually the best broad candidate generator. Returns `{ matches: [...] }` ordered best-first by fused relevance; each match has `rank`, `doc_id`, `score`, `text_preview`, and `metadata`. Default `limit` is 100."
        } else {
            "Run hybrid retrieval for several probes, combining lexical and dense evidence. This is usually the best broad candidate generator. Returns `{ matches: [...] }` ordered best-first by fused relevance; each match has `rank`, `doc_id`, `score`, `text_preview`, and `metadata`. Default `limit` is 100."
        };
        let mut tools = vec![
            obliq_tool(
                "fetch_docs",
                "Fetch OBLIQ corpus documents by document id.",
                json!({
                    "type": "object",
                    "properties": {
                        "doc_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "minItems": 1,
                            "maxItems": 100
                        }
                    },
                    "required": ["doc_ids"],
                    "additionalProperties": false
                }),
            ),
            obliq_tool(
                "bm25_search",
                "Search the OBLIQ corpus with lexical BM25 retrieval. Returns `{ matches: [...] }` ordered best-first by BM25 relevance; each match has `rank`, `doc_id`, `score`, `text_preview`, and `metadata`. Default `limit` is 100.",
                search_schema(),
            ),
            obliq_tool(
                "dense_search",
                "Search the OBLIQ corpus with dense semantic retrieval. Returns `{ matches: [...] }` ordered best-first by dense relevance; each match has `rank`, `doc_id`, `score`, `text_preview`, and `metadata`. Default `limit` is 100.",
                search_schema(),
            ),
            obliq_tool(
                "discover_docs",
                "Use example-guided discovery: provide a target query plus positive-vs-negative document pairs to guide retrieval toward the latent pattern you want and away from false friends. Use this after `fetch_docs` when you can identify examples of matching and non-matching relevance patterns. Returns `{ matches: [...] }` ordered best-first; each match has `rank`, `doc_id`, `score`, `text_preview`, and `metadata`. Default `limit` is 100.",
                json!({
                    "type": "object",
                    "properties": {
                        "target_query": { "type": "string" },
                        "context_pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "positive_doc_id": { "type": "string" },
                                    "negative_doc_id": { "type": "string" }
                                },
                                "required": ["positive_doc_id", "negative_doc_id"],
                                "additionalProperties": false
                            },
                            "minItems": 1,
                            "maxItems": 20
                        },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 100 }
                    },
                    "required": ["target_query", "context_pairs"],
                    "additionalProperties": false
                }),
            ),
            obliq_tool(
                "hybrid_search",
                hybrid_description,
                json!({
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": { "type": "string" },
                            "minItems": 1,
                            "maxItems": 12
                        },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 100 },
                        "candidate_pool": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 1000 }
                    },
                    "required": ["queries"],
                    "additionalProperties": false
                }),
            ),
        ];
        if self.late_available {
            tools.push(obliq_tool(
                "late_search",
                "Search the OBLIQ corpus with late-interaction reranking. Returns `{ matches: [...] }` ordered best-first by late-interaction score. Default `limit` is 100.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "candidate_pool": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 1000 },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 100 }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
            ));
        }
        tools
    }

    async fn execute(&self, name: &str, args: &Value) -> ToolResult {
        match self.call_script(name, args).await {
            Ok(value) => ToolResult::ok(value),
            Err(error) => ToolResult::err_fmt(error),
        }
    }
}

impl ObliqTools {
    async fn probe_late_available(&self) -> anyhow::Result<bool> {
        let stats = self.call_script_raw("corpus_stats", &json!({})).await?;
        Ok(stats
            .get("late_available")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    async fn call_script(&self, op: &str, args: &Value) -> anyhow::Result<Value> {
        let value = self.call_script_raw(op, args).await?;
        Ok(self.filter_excluded(value))
    }

    async fn call_script_raw(&self, op: &str, args: &Value) -> anyhow::Result<Value> {
        let mut child = tokio::process::Command::new(&self.python)
            .arg(&self.script)
            .arg("--op")
            .arg(op)
            .arg("--data-dir")
            .arg(&self.data_dir)
            .arg("--qdrant-url")
            .arg(&self.qdrant_url)
            .arg("--collection")
            .arg(&self.collection)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawn {}", self.script.display()))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(serde_json::to_vec(args)?.as_slice())
                .await?;
        }
        let output = child.wait_with_output().await?;
        if !output.status.success() {
            bail!(
                "{} failed: {}",
                self.script.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let value = serde_json::from_slice(&output.stdout)
            .with_context(|| format!("parse {} output", self.script.display()))?;
        Ok(value)
    }

    fn filter_excluded(&self, mut value: Value) -> Value {
        if self.excluded_doc_ids.is_empty() {
            return value;
        }
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "excluded_doc_ids_hidden".to_string(),
                json!(self.excluded_doc_ids.len()),
            );
            if let Some(matches) = object.get_mut("matches").and_then(Value::as_array_mut) {
                matches.retain(|item| {
                    item.get("doc_id")
                        .and_then(Value::as_str)
                        .map(|doc_id| !self.excluded_doc_ids.contains(doc_id))
                        .unwrap_or(true)
                });
                rerank_items(matches);
            }
            if let Some(docs) = object.get_mut("docs").and_then(Value::as_array_mut) {
                docs.retain(|item| {
                    item.get("doc_id")
                        .and_then(Value::as_str)
                        .map(|doc_id| !self.excluded_doc_ids.contains(doc_id))
                        .unwrap_or(true)
                });
            }
        }
        value
    }
}

impl RunArtifacts {
    fn from_output(output: Option<&PathBuf>, query_id: &str) -> Self {
        let output_json = output.cloned().unwrap_or_else(|| {
            PathBuf::from(format!(".benchmarks/obliq/runs/{query_id}.json"))
        });
        let session_db = output_json.with_extension("session.db");
        let trace_jsonl = output_json.with_extension("trace.jsonl");
        let trace_html = output_json.with_extension("trace.html");
        Self {
            output_json: output_json.display().to_string(),
            session_db: session_db.display().to_string(),
            trace_jsonl: trace_jsonl.display().to_string(),
            trace_html: trace_html.display().to_string(),
        }
    }

    fn create_parent_dirs(&self) -> anyhow::Result<()> {
        for path in [
            &self.output_json,
            &self.session_db,
            &self.trace_jsonl,
            &self.trace_html,
        ] {
            if let Some(parent) = Path::new(path).parent() {
                fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
            }
        }
        Ok(())
    }
}

fn sanitize_ranked_doc_ids(
    ranked_doc_ids: Vec<String>,
    excluded_doc_ids: &BTreeSet<String>,
) -> SanitizedSubmission {
    let mut seen = BTreeSet::new();
    let mut ranked = Vec::with_capacity(ranked_doc_ids.len());
    let mut removed = Vec::new();
    for doc_id in ranked_doc_ids {
        if excluded_doc_ids.contains(&doc_id) || !seen.insert(doc_id.clone()) {
            removed.push(doc_id);
            continue;
        }
        ranked.push(doc_id);
    }
    SanitizedSubmission {
        ranked_doc_ids: ranked,
        removed_doc_ids: removed,
    }
}

fn export_single_session_html(
    store_path: &Path,
    trace_path: &Path,
    html_path: &Path,
) -> anyhow::Result<()> {
    let session = load_session_from_paths(store_path, trace_path)?;
    let html = render(&session, ExportFormat::Html);
    fs::write(html_path, html).with_context(|| format!("write {}", html_path.display()))?;
    Ok(())
}

fn rerank_items(items: &mut [Value]) {
    for (index, item) in items.iter_mut().enumerate() {
        if let Some(object) = item.as_object_mut() {
            object.insert("rank".to_string(), json!(index + 1));
        }
    }
}

fn obliq_tool(name: &str, description: &str, input_schema: Value) -> ToolDefinition {
    ToolDefinition::new(
        name,
        description,
        input_schema,
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn search_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": { "type": "string" },
            "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 100 }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

struct ListAsyncHandlesTool;

#[async_trait]
impl ToolProvider for ListAsyncHandlesTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![list_async_handles_tool_definition()]
    }

    async fn execute(&self, name: &str, _args: &Value) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{name}` is handled by the RLM runtime and cannot run directly"
        ))
    }
}

fn list_async_handles_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "list_async_handles",
        "List live lashlang async handles only. Returns `{ monitor: { monitor_id: handle }, subagent: { name: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted.",
        ToolDefinition::default_input_schema(),
        json!({
            "type": "object",
            "properties": {
                "monitor": { "type": "object" },
                "subagent": { "type": "object" },
                "tool": { "type": "object" }
            },
            "required": ["monitor", "subagent", "tool"]
        }),
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn resolve_provider(provider_id: Option<&str>) -> anyhow::Result<lash::ProviderHandle> {
    lash_providers_builtin::register_all();
    let config_path = lash_home().join("config.json");
    let mut config = LashConfig::load(&config_path)
        .ok_or_else(|| anyhow::anyhow!("missing or invalid {}", config_path.display()))?;
    if let Some(provider_id) = provider_id {
        config
            .set_active_provider_kind(provider_id)
            .map_err(anyhow::Error::msg)?;
    }
    config.build_active_provider().map_err(anyhow::Error::msg)
}

fn python_bin() -> PathBuf {
    if let Some(value) = std::env::var_os("OBLIQ_PYTHON") {
        return PathBuf::from(value);
    }
    let venv = PathBuf::from(".venv/bin/python");
    if venv.exists() {
        venv
    } else {
        PathBuf::from("python3")
    }
}

fn lash_home() -> PathBuf {
    std::env::var_os("LASH_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".lash")))
        .unwrap_or_else(|| PathBuf::from(".lash"))
}

fn load_math_query(data_dir: &Path, query_id: &str) -> anyhow::Result<QueryRow> {
    let path = data_dir.join("math/queries.jsonl");
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<QueryRow>)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .find(|row| row.id == query_id)
        .ok_or_else(|| anyhow::anyhow!("query `{query_id}` not found in {}", path.display()))
}

fn load_math_stats(data_dir: &Path) -> anyhow::Result<MathStats> {
    let math = data_dir.join("math");
    let corpus_docs = count_nonempty_lines(&math.join("corpus.jsonl"))?;
    Ok(MathStats {
        corpus_docs,
    })
}

fn count_nonempty_lines(path: &Path) -> anyhow::Result<usize> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(raw.lines().filter(|line| !line.trim().is_empty()).count())
}

fn load_qrels(data_dir: &Path, query_id: &str) -> anyhow::Result<HashMap<String, f64>> {
    let path = data_dir.join("math/qrels.tsv");
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let mut gold = HashMap::new();
    for (index, line) in raw.lines().enumerate() {
        if index == 0 && line.starts_with("query-id") {
            continue;
        }
        let parts = line.split('\t').collect::<Vec<_>>();
        if parts.len() < 3 || parts[0] != query_id {
            continue;
        }
        let score = parts[2].parse::<f64>().unwrap_or(1.0);
        gold.insert(parts[1].to_string(), score);
    }
    Ok(gold)
}

fn load_excluded_doc_ids(data_dir: &Path, query_id: &str) -> anyhow::Result<BTreeSet<String>> {
    let path = data_dir.join("math/per_query_excluded_ids.json");
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let value: HashMap<String, Vec<String>> =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    Ok(value
        .get(query_id)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect())
}

fn score_submission(ranked: &[String], gold: &HashMap<String, f64>) -> Metrics {
    Metrics {
        ndcg_at_10: ndcg_at(ranked, gold, 10),
        ndcg_at_50: ndcg_at(ranked, gold, 50),
        recall_at_10: recall_at(ranked, gold, 10),
        recall_at_50: recall_at(ranked, gold, 50),
        recall_at_100: recall_at(ranked, gold, 100),
        gold_count: gold.len(),
    }
}

fn ndcg_at(ranked: &[String], gold: &HashMap<String, f64>, k: usize) -> f64 {
    let dcg = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(index, doc_id)| {
            let rel = gold.get(doc_id).copied().unwrap_or(0.0);
            if rel <= 0.0 {
                0.0
            } else {
                ((2.0_f64).powf(rel) - 1.0) / ((index + 2) as f64).log2()
            }
        })
        .sum::<f64>();
    let mut ideal = gold.values().copied().collect::<Vec<_>>();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = ideal
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(index, rel)| ((2.0_f64).powf(rel) - 1.0) / ((index + 2) as f64).log2())
        .sum::<f64>();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

fn recall_at(ranked: &[String], gold: &HashMap<String, f64>, k: usize) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    let seen = ranked
        .iter()
        .take(k)
        .filter(|doc_id| gold.contains_key(*doc_id))
        .collect::<BTreeSet<_>>()
        .len();
    seen as f64 / gold.len() as f64
}
