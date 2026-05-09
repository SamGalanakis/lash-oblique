mod tournament;

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
    ExecutionMode, JsonlTraceSink, PluginFactory, SessionPolicy, ToolDefinition, ToolExecutionMode,
    ToolResult,
};
use lash_embed::{
    Input, LashCore, ModeId, ModeTurnOptions, RuntimePersistence, SessionStoreCreateRequest,
    SessionStoreFactory, ToolProvider, TraceContext, TraceLevel, TraceSink, TurnOutcome,
};
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

use crate::tournament::TournamentRerankProvider;

const DEFAULT_DATA_DIR: &str = ".benchmarks/obliq/data";
const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";
const DEFAULT_COLLECTION: &str = "obliq_math";
const DEFAULT_MODEL: &str = "gpt-5.5";
const DEFAULT_VARIANT: &str = "medium";
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const SUBAGENT_CAPABILITY: &str = "explore";
/// Source files that affect agent behavior. Their bytes are baked in at
/// compile time and SHA'd to produce a stable cache key. Changing the
/// prompt, the tournament algorithm, or the python retrieval script
/// invalidates cached results automatically.
const AGENT_MAIN_SRC: &[u8] = include_bytes!("main.rs");
const AGENT_TOURNAMENT_SRC: &[u8] = include_bytes!("tournament.rs");
const AGENT_QDRANT_SCRIPT: &[u8] = include_bytes!("../scripts/query_math_qdrant.py");

const DEFAULT_MATH_DESCRIPTION: &str = "Queries are mathematical problems. Relevant documents \
are other math problems whose solution requires the SAME abstract proof strategy or \
'aha moment' / 'eureka insight', even when the surface topic, notation, vocabulary, or \
mathematical domain (e.g., algebra vs. analysis vs. number theory) differs entirely. \
Surface lexical or topical overlap does NOT indicate relevance; the latent reasoning move \
does.";

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
        #[arg(long, default_value = DEFAULT_MATH_DESCRIPTION)]
        description: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run many math queries concurrently. Each query gets its own
    /// session.db, trace.jsonl, and output JSON. Writes a batch
    /// summary at `<output_dir>/_batch_summary.json` at the end.
    RunMathBatch {
        /// Comma-separated query IDs. If omitted, runs every query in
        /// `<data_dir>/math/queries.jsonl`.
        #[arg(long)]
        query_ids: Option<String>,
        /// Limit to the first N queries (after the `query_ids` filter,
        /// if any). Useful for sampling.
        #[arg(long)]
        limit: Option<usize>,
        /// How many queries run concurrently. Each query may itself
        /// fan out (tournament_rerank uses up to 8 parallel batches),
        /// so total concurrent LLM calls ≈ concurrency × 8.
        #[arg(long, default_value_t = 3)]
        concurrency: usize,
        /// Skip queries whose output JSON already exists. On by
        /// default so a re-run resumes after a crash.
        #[arg(long, default_value_t = true)]
        skip_existing: bool,
        /// Where per-query outputs land.
        #[arg(long, default_value = ".benchmarks/obliq/runs")]
        output_dir: PathBuf,
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
        #[arg(long, default_value = DEFAULT_MATH_DESCRIPTION)]
        description: String,
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
    metrics: Option<MetricsBundle>,
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

/// Paper Table 3 reports every score as G/P: G uses initial qrels.tsv, P
/// uses post-pooled qrels_pool.tsv. We compute both so headline numbers
/// are directly comparable to the paper.
#[derive(Debug, Clone, Default, Serialize)]
struct MetricsBundle {
    gold: Metrics,
    pooled: Metrics,
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
            description,
            output,
        } => {
            let provider = resolve_provider(provider_id.as_deref())?;
            let params = RunMathParams {
                query_id: query_id.clone(),
                data_dir,
                qdrant_url,
                collection,
                model,
                variant,
                max_context_tokens,
                description,
                output_path: output.clone(),
            };
            let run_output = run_one_math_query(provider, params).await?;
            let json = serde_json::to_string_pretty(&run_output)?;
            if output.is_none() {
                println!("{json}");
            }
        }
        Command::RunMathBatch {
            query_ids,
            limit,
            concurrency,
            skip_existing,
            output_dir,
            data_dir,
            qdrant_url,
            collection,
            provider_id,
            model,
            variant,
            max_context_tokens,
            description,
        } => {
            run_math_batch(RunMathBatchParams {
                query_ids,
                limit,
                concurrency: concurrency.max(1),
                skip_existing,
                output_dir,
                data_dir,
                qdrant_url,
                collection,
                provider_id,
                model,
                variant,
                max_context_tokens,
                description,
            })
            .await?;
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
            let bundle = score_submission_bundle(&sanitized.ranked_doc_ids, &data_dir, &query_id)?;
            println!("{}", serde_json::to_string_pretty(&bundle)?);
        }
    }
    Ok(())
}

#[derive(Clone)]
struct RunMathParams {
    query_id: String,
    data_dir: PathBuf,
    qdrant_url: String,
    collection: String,
    model: String,
    variant: String,
    max_context_tokens: usize,
    description: String,
    /// If `Some`, output JSON is written to this path. If `None`, the
    /// caller is responsible for printing/handling the returned
    /// `RunOutput` themselves.
    output_path: Option<PathBuf>,
}

struct RunMathBatchParams {
    query_ids: Option<String>,
    limit: Option<usize>,
    concurrency: usize,
    skip_existing: bool,
    output_dir: PathBuf,
    data_dir: PathBuf,
    qdrant_url: String,
    collection: String,
    provider_id: Option<String>,
    model: String,
    variant: String,
    max_context_tokens: usize,
    description: String,
}

async fn run_one_math_query(
    provider: lash::ProviderHandle,
    params: RunMathParams,
) -> anyhow::Result<RunOutput> {
    let RunMathParams {
        query_id,
        data_dir,
        qdrant_url,
        collection,
        model,
        variant,
        max_context_tokens,
        description,
        output_path,
    } = params;
    let query = load_math_query(&data_dir, &query_id)?;
    let excluded = load_excluded_doc_ids(&data_dir, &query_id)?;
    let stats = load_math_stats(&data_dir).unwrap_or_default();
    let artifacts = RunArtifacts::from_output(output_path.as_ref(), &query_id);
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
        &description,
    )?;
    let session = core
        .session(format!("obliq-math-{}", uuid::Uuid::new_v4()))
        .rlm()
        .open()
        .await
        .context("open Lash RLM session")?;
    let schema = output_schema();
    let turn = session
        .turn(Input::text(run_prompt(
            &query,
            &stats,
            &excluded,
            late_available,
            &description,
        )))
        .mode_turn_options(
            ModeTurnOptions::typed(
                ExecutionMode::new("rlm"),
                RlmTermination::SubmitRequired {
                    schema: Some(schema),
                },
            )
            .map_err(anyhow::Error::msg)?,
        )
        .run()
        .await
        .context("run OBLIQ math query")?;

    let submitted = match &turn.outcome {
        TurnOutcome::Finished(lash::TurnFinish::Value { value, .. }) => {
            parse_submission(value.clone())?
        }
        other => bail!(
            "RLM did not submit ranked_doc_ids: outcome={other:?} errors={:?} text={}",
            turn.errors,
            turn.transcript.rendered_output()
        ),
    };
    let raw_ranked_doc_ids = submitted.ranked_doc_ids;
    let sanitized = sanitize_ranked_doc_ids(raw_ranked_doc_ids.clone(), &excluded);
    let metrics = score_submission_bundle(&sanitized.ranked_doc_ids, &tools.data_dir, &query.id).ok();
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
        tool_calls: turn.transcript.tool_calls.len(),
        final_text: turn.transcript.rendered_output(),
        errors: turn.errors.into_iter().map(|issue| issue.message).collect(),
        artifacts: artifacts.clone(),
    };
    if let Some(output) = output_path {
        let json = serde_json::to_string_pretty(&run_output)?;
        fs::write(&output, format!("{json}\n"))
            .with_context(|| format!("write {}", output.display()))?;
    }
    Ok(run_output)
}

/// 12-char hex prefix of SHA-256 over every source file that affects
/// agent behavior. Compile-time captured via `include_bytes!`, so dirty
/// edits invalidate the hash on the next `cargo build`.
fn agent_spec_hash() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(AGENT_MAIN_SRC);
    h.update(AGENT_TOURNAMENT_SRC);
    h.update(AGENT_QDRANT_SCRIPT);
    let out = h.finalize();
    let mut s = String::with_capacity(12);
    for b in &out[..6] {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Cache key combining agent source hash with the runtime config that
/// also changes agent behavior (model, variant, description).
fn config_hash(model: &str, variant: &str, description: &str) -> String {
    use sha2::{Digest, Sha256};
    let agent = agent_spec_hash();
    let mut h = Sha256::new();
    h.update(agent.as_bytes());
    h.update(b":");
    h.update(model.as_bytes());
    h.update(b":");
    h.update(variant.as_bytes());
    h.update(b":");
    h.update(description.as_bytes());
    let out = h.finalize();
    let mut s = String::with_capacity(12);
    for b in &out[..6] {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

async fn run_math_batch(params: RunMathBatchParams) -> anyhow::Result<()> {
    let RunMathBatchParams {
        query_ids,
        limit,
        concurrency,
        skip_existing,
        output_dir,
        data_dir,
        qdrant_url,
        collection,
        provider_id,
        model,
        variant,
        max_context_tokens,
        description,
    } = params;

    // Compute a stable cache key from agent source + runtime config.
    // Outputs land at `<output_dir>/<hash>/...` so unchanged-code re-runs
    // hit the existing files via `--skip-existing`, and any code/config
    // change automatically opens a fresh cache slot.
    let agent_hash = agent_spec_hash();
    let cfg_hash = config_hash(&model, &variant, &description);
    let scoped_output = output_dir.join(&cfg_hash);
    fs::create_dir_all(&scoped_output)
        .with_context(|| format!("create {}", scoped_output.display()))?;
    // Update a `_latest` pointer file so the dashboard can find the most
    // recent run for this output_dir without scanning.
    let _ = fs::write(
        output_dir.join("_latest"),
        format!("{cfg_hash}\n"),
    );
    // Persist a small manifest so the dashboard can show what generated
    // this run without parsing every per-query JSON.
    let manifest = json!({
        "agent_spec_hash": agent_hash,
        "config_hash": cfg_hash,
        "model": model,
        "variant": variant,
        "description": description,
    });
    let _ = fs::write(
        scoped_output.join("_manifest.json"),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    );
    let output_dir = scoped_output;

    // Resolve query list. Either explicit CSV or every id in queries.jsonl.
    let mut requested: Vec<String> = if let Some(ids) = query_ids.as_deref() {
        ids.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        load_all_math_query_ids(&data_dir)?
    };
    if let Some(n) = limit {
        requested.truncate(n);
    }
    let total_requested = requested.len();
    if total_requested == 0 {
        bail!("no queries to run");
    }

    // Filter out queries whose output already exists, if requested.
    let to_run: Vec<String> = if skip_existing {
        requested
            .into_iter()
            .filter(|qid| !output_dir.join(format!("{qid}.json")).exists())
            .collect()
    } else {
        requested
    };
    let skipped = total_requested - to_run.len();
    if to_run.is_empty() {
        eprintln!("[batch] all {total_requested} queries already have outputs; nothing to run");
        write_batch_summary(&output_dir, &description, &model, &variant, &[])?;
        return Ok(());
    }
    eprintln!(
        "[batch] running {} of {total_requested} queries (skipping {skipped} that already exist), concurrency={concurrency}",
        to_run.len(),
    );

    // Resolve provider once. ProviderHandle is Clone; share across tasks.
    let provider = resolve_provider(provider_id.as_deref())?;

    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut tasks: tokio::task::JoinSet<(String, anyhow::Result<RunOutput>)> =
        tokio::task::JoinSet::new();
    for qid in &to_run {
        let provider = provider.clone();
        let semaphore = semaphore.clone();
        let params = RunMathParams {
            query_id: qid.clone(),
            data_dir: data_dir.clone(),
            qdrant_url: qdrant_url.clone(),
            collection: collection.clone(),
            model: model.clone(),
            variant: variant.clone(),
            max_context_tokens,
            description: description.clone(),
            output_path: Some(output_dir.join(format!("{qid}.json"))),
        };
        let qid_owned = qid.clone();
        tasks.spawn(async move {
            let _permit = match semaphore.acquire_owned().await {
                Ok(p) => p,
                Err(err) => return (qid_owned, Err(anyhow::anyhow!("semaphore closed: {err}"))),
            };
            let result = run_one_math_query(provider, params).await;
            (qid_owned, result)
        });
    }

    let total_to_run = to_run.len();
    let mut completed: Vec<(String, anyhow::Result<RunOutput>)> = Vec::with_capacity(total_to_run);
    while let Some(joined) = tasks.join_next().await {
        let (qid, result) = joined.map_err(|err| anyhow::anyhow!("task join: {err}"))?;
        let n_done = completed.len() + 1;
        match &result {
            Ok(out) => match &out.metrics {
                Some(b) => eprintln!(
                    "[{n_done}/{total_to_run}] {qid} — G:NDCG@10={:.3} R@100={:.3} | P:NDCG@10={:.3} R@100={:.3} tools={}",
                    b.gold.ndcg_at_10, b.gold.recall_at_100,
                    b.pooled.ndcg_at_10, b.pooled.recall_at_100,
                    out.tool_calls
                ),
                None => eprintln!(
                    "[{n_done}/{total_to_run}] {qid} — submitted (no qrels) tools={}",
                    out.tool_calls
                ),
            },
            Err(err) => eprintln!("[{n_done}/{total_to_run}] {qid} — FAILED: {err:#}"),
        }
        completed.push((qid, result));
    }

    write_batch_summary(&output_dir, &description, &model, &variant, &completed)?;
    Ok(())
}

fn write_batch_summary(
    output_dir: &Path,
    description: &str,
    model: &str,
    variant: &str,
    completed: &[(String, anyhow::Result<RunOutput>)],
) -> anyhow::Result<()> {
    let mut total = 0usize;
    let mut ok = 0usize;
    let mut scored = 0usize;
    let mut sum_g = Metrics::default();
    let mut sum_p = Metrics::default();
    let mut sum_tool_calls = 0usize;
    let mut per_query: Vec<Value> = Vec::new();

    let accumulate = |sum: &mut Metrics, m: &Metrics| {
        sum.ndcg_at_10 += m.ndcg_at_10;
        sum.ndcg_at_50 += m.ndcg_at_50;
        sum.recall_at_10 += m.recall_at_10;
        sum.recall_at_50 += m.recall_at_50;
        sum.recall_at_100 += m.recall_at_100;
    };

    for (qid, result) in completed {
        total += 1;
        match result {
            Ok(out) => {
                ok += 1;
                sum_tool_calls += out.tool_calls;
                if let Some(b) = &out.metrics {
                    scored += 1;
                    accumulate(&mut sum_g, &b.gold);
                    accumulate(&mut sum_p, &b.pooled);
                }
                per_query.push(json!({
                    "query_id": qid,
                    "ok": true,
                    "tool_calls": out.tool_calls,
                    "metrics": out.metrics,
                }));
            }
            Err(err) => {
                per_query.push(json!({
                    "query_id": qid,
                    "ok": false,
                    "error": err.to_string(),
                }));
            }
        }
    }

    let n = scored as f64;
    let mean_of = |sum: &Metrics| {
        if scored == 0 {
            json!({})
        } else {
            json!({
                "ndcg_at_10": sum.ndcg_at_10 / n,
                "ndcg_at_50": sum.ndcg_at_50 / n,
                "recall_at_10": sum.recall_at_10 / n,
                "recall_at_50": sum.recall_at_50 / n,
                "recall_at_100": sum.recall_at_100 / n,
            })
        }
    };
    let summary = json!({
        "model": model,
        "variant": variant,
        "description": description,
        "total": total,
        "ok": ok,
        "failed": total - ok,
        "scored": scored,
        "mean_metrics": {
            "gold": mean_of(&sum_g),
            "pooled": mean_of(&sum_p),
        },
        "mean_tool_calls": if ok > 0 { sum_tool_calls as f64 / ok as f64 } else { 0.0 },
        "per_query": per_query,
    });
    let path = output_dir.join("_batch_summary.json");
    fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&summary)?))
        .with_context(|| format!("write {}", path.display()))?;
    let g_n10 = if scored > 0 { sum_g.ndcg_at_10 / n } else { 0.0 };
    let g_r100 = if scored > 0 { sum_g.recall_at_100 / n } else { 0.0 };
    let p_n10 = if scored > 0 { sum_p.ndcg_at_10 / n } else { 0.0 };
    let p_r100 = if scored > 0 { sum_p.recall_at_100 / n } else { 0.0 };
    eprintln!(
        "[batch] summary written to {}: ok={ok}/{total} | G: NDCG@10={g_n10:.3} R@100={g_r100:.3} | P: NDCG@10={p_n10:.3} R@100={p_r100:.3}",
        path.display(),
    );
    Ok(())
}

fn load_all_math_query_ids(data_dir: &Path) -> anyhow::Result<Vec<String>> {
    let path = data_dir.join("math/queries.jsonl");
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: QueryRow = serde_json::from_str(line)
            .with_context(|| format!("parse query row in {}", path.display()))?;
        out.push(row.id);
    }
    Ok(out)
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
    description: &str,
) -> anyhow::Result<LashCore> {
    let subagent_policy = SessionPolicy {
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
        subagent_policy,
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
    let tournament: Arc<dyn ToolProvider> = Arc::new(TournamentRerankProvider::new(
        obliq_tools.clone(),
        description.to_string(),
    ));

    LashCore::rlm()
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(model)
        .model_variant(variant)
        .max_context_tokens(max_context_tokens)
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .trace_sink(Some(
            Arc::new(JsonlTraceSink::new(trace_path)) as Arc<dyn TraceSink>
        ))
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
        .plugin(Arc::new(StaticPluginFactory::new(
            "obliq_tournament_rerank",
            PluginSpec::new().with_tool_provider(tournament),
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
    description: &str,
) -> String {
    let excluded_text = if excluded.is_empty() {
        "No document IDs are excluded for this query.".to_string()
    } else {
        format!(
            "Do not submit these excluded document IDs for this query: {}",
            excluded.iter().cloned().collect::<Vec<_>>().join(", ")
        )
    };
    let late_tool_line = if late_available {
        "- `late_search(query, candidate_pool=1500, limit=200)`: late-interaction reranking; single-channel diversity probe.\n"
    } else {
        ""
    };
    format!(
        r#"You are running one query from a retrieval benchmark.

# Relevance description (what makes a document relevant)
{description}

# Query
id: `{query_id}`
{excluded_text}

text:
{query_text}

# Corpus
The searchable corpus has {corpus_docs} documents. Every submitted id must come from a tool result. Submit exactly 100 unique, non-excluded ids ranked best first.

# Vocabulary (used throughout this playbook)
- **surface features** — vocabulary, notation, topical labels present in the query text. A surface-only matcher (BM25, dense embedding) ranks by these.
- **structural features** / **schema** — the relational pattern that determines relevance per the description above: the objects involved and the relations that must hold between them. The schema is what makes two documents analogous even when their surface features differ.
- **surface-similar distractor** — a document that shares surface features with the query but does NOT share the schema. These are what surface-only retrieval ranks high; they are wrong.
- **schema-anchored query** — a phrasing of the schema that deliberately avoids the original surface vocabulary, so it can match documents whose surface features are different but whose structural relations are the same.

# Tool surface
- `hybrid_search(queries: [str], limit=300, candidate_pool=1500)`: BM25 + dense fused via RRF. Best broad candidate generator. Pass 4–6 probes per call.
- `bm25_search(query, limit=200)`: lexical only. Single-channel diversity probe.
- `dense_search(query, limit=200)`: dense only. Single-channel diversity probe.
- `discover_docs(target_query, context_pairs=[{{positive_doc_id, negative_doc_id}}], limit=300)`: example-anchored dense search. Use after Phase B' once you have schema-sharing positives AND named surface-similar distractors as negatives.
{late_tool_line}- `fetch_docs(doc_ids, up to 50)`: read full text. Required in Phase B'.
- `spawn_agent(...)`: full RLM session with the same tools; for parallel scout lines, each in a different surface domain.
- `llm_query(task, inputs, output)`: ONE direct LLM call against text you supply. CAN return structured JSON, extract many fields per doc, judge a batch of 30–50 docs in one shot. CANNOT search, fetch, or call tools.
- `tournament_rerank(query, candidate_doc_ids, top_k=100)`: listwise reranker. **Internally caps input at 300 — DO NOT pass more, the tail is dropped silently. Pass your top 300 by RRF rank.**

# Effort routing
- Think → main loop (free).
- Read → `fetch_docs` + main-loop reasoning.
- Look-at-text-and-judge → `llm_query`, batched.
- Run-its-own-search-loop → `spawn_agent`, in parallel.
- Rank → `tournament_rerank`. Submission must come from its output.

# Playbook — waterfall, escalate only when the audit says you must

## Phase A — articulate the schema (main loop, free)
1. From the relevance description and the query, write in your head 1–2 sentences:
   - "The relevance schema is: <objects> + <relations that must hold between them>." Concrete and structural. Avoid the original surface vocabulary of the query.
   - Name 2–3 surface features (tokens, topic labels) likely to appear in surface-similar distractors — i.e. tokens a surface-only matcher would over-weight even though they aren't part of the schema.
2. Write 4–6 SURFACE probes — paraphrases of the query in its own surface vocabulary. These exercise Tier 1.

## Phase B — Tier 1 cheap surface retrieval (1 call)
3. `hybrid_search(queries = your 4–6 surface probes, limit=300, candidate_pool=1500)`.

## Phase B' — audit (the load-bearing step; never skip)
4. `fetch_docs(doc_ids = top 8 of Phase B output)`.
5. In the main loop, for each of the top 8 candidates, classify and label:
   - (i) **shares the schema** — name in one phrase the structural correspondence (which objects, which relations).
   - (ii) **surface-similar distractor** — name the surface tokens that fooled the search (these are what schema-anchored probes must AVOID).
   - (iii) **unclear** — count as a non-hit for the decision below.
6. Decide:
   - **≥ 5 of top 8 are (i)** → Tier 1 was sufficient. Skip to Phase D using the Phase B pool's top 300.
   - **Otherwise** → Phase C (Tier 2).

## Phase C — Tier 2 schema-anchored widening (escalation)
7. Write 4–6 SCHEMA-ANCHORED probes:
   - Each phrases the schema as a complete-sentence query.
   - Each uses surface vocabulary from a DIFFERENT surface domain than the query's.
   - Each AVOIDS the distractor tokens you flagged in step 5.
8. `hybrid_search(queries = your 4–6 schema-anchored probes, limit=300, candidate_pool=1500)`.
9. If you have ≥ 1 schema-sharing positive AND ≥ 1 named distractor from Phase B', also run:
   `discover_docs(target_query = your best schema-anchored phrasing, context_pairs = 4–10 (positive, distractor) pairs, limit=300)`.
10. **Optional Tier 3** — only if step 8/9 is still thin or still surface-y when spot-checked:
    - `spawn_agent` × 2–4 in parallel, each with a distinct schema-anchored brief in a different surface domain.
    - One `bm25_search` and/or `dense_search` (limit=200) on a schema-anchored phrasing not yet covered.
11. Merge results from all channels (B + C steps). Take the **top 300 by RRF rank** across channels. This is the input to Phase D.

## Phase D — rerank (mandatory)
12. `tournament_rerank(query = the original query, optionally appended with a one-line schema description from step 1, candidate_doc_ids = your top-300 RRF-ranked merged pool, top_k=100)`.

## Submission
The output of step 12 is your submission, in order. Do not hand-reorder.

# Pool sizing summary
- hybrid_search:     limit 300, candidate_pool 1500, 4–6 probes/call
- bm25/dense:        limit 200, single-channel diversity only
- discover_docs:     limit 300, 4–10 (positive, distractor) pairs
- fetch_docs:        top 8 in Phase B' is mandatory; up to 50/call
- spawn_agent:       2–4 in parallel, each ~150 ids
- tournament_rerank: input ≤ 300 (HARD CAP, enforced internally), top_k 100
- llm_query batches: 30–50 items/call

# Hard rules
- Every submitted id comes from a tool result.
- Submission = `tournament_rerank` output. No hand-ordering, no skipping the rerank.
- Submit exactly 100 unique, non-excluded ids. If `tournament_rerank` returns fewer than 100 (because your input pool was small), fill the tail with the next-best candidates from your merged pool by RRF rank.
- `tournament_rerank` input MUST be your top 300 by RRF — passing more wastes time because the tail is dropped silently.

# Submission
```lashlang
submit {{ ranked_doc_ids: [/* exactly 100 strings */] }}
```
"#,
        description = description,
        query_id = query.id,
        excluded_text = excluded_text,
        query_text = query.text,
        corpus_docs = stats.corpus_docs,
        late_tool_line = late_tool_line,
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
    pub async fn fetch_doc_texts(
        &self,
        doc_ids: &[String],
    ) -> anyhow::Result<HashMap<String, String>> {
        let mut out = HashMap::new();
        if doc_ids.is_empty() {
            return Ok(out);
        }
        for chunk in doc_ids.chunks(100) {
            let value = self
                .call_script_raw("fetch_docs", &json!({ "doc_ids": chunk }))
                .await?;
            let Some(docs) = value.get("docs").and_then(Value::as_array) else {
                continue;
            };
            for doc in docs {
                let Some(doc_id) = doc.get("doc_id").and_then(Value::as_str) else {
                    continue;
                };
                let text = doc
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                out.insert(doc_id.to_string(), text);
            }
        }
        Ok(out)
    }

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
        let output_json = output
            .cloned()
            .unwrap_or_else(|| PathBuf::from(format!(".benchmarks/obliq/runs/{query_id}.json")));
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
                fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
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
    Ok(MathStats { corpus_docs })
}

fn count_nonempty_lines(path: &Path) -> anyhow::Result<usize> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(raw.lines().filter(|line| !line.trim().is_empty()).count())
}

fn load_qrels_file(
    data_dir: &Path,
    filename: &str,
    query_id: &str,
) -> anyhow::Result<HashMap<String, f64>> {
    let path = data_dir.join("math").join(filename);
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

fn load_qrels(data_dir: &Path, query_id: &str) -> anyhow::Result<HashMap<String, f64>> {
    load_qrels_file(data_dir, "qrels.tsv", query_id)
}

fn load_qrels_pool(data_dir: &Path, query_id: &str) -> anyhow::Result<HashMap<String, f64>> {
    load_qrels_file(data_dir, "qrels_pool.tsv", query_id)
}

fn score_submission_bundle(
    ranked: &[String],
    data_dir: &Path,
    query_id: &str,
) -> anyhow::Result<MetricsBundle> {
    let gold_qrels = load_qrels(data_dir, query_id)?;
    let pooled_qrels = load_qrels_pool(data_dir, query_id)?;
    Ok(MetricsBundle {
        gold: score_submission(ranked, &gold_qrels),
        pooled: score_submission(ranked, &pooled_qrels),
    })
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
