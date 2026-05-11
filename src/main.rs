mod tournament;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use clap::{Parser, Subcommand};
use lash::advanced::{ExecutionMode, ModeTurnOptions};
use lash::persistence::{RuntimePersistence, SessionStoreCreateRequest, SessionStoreFactory};
use lash::plugins::ToolOutputBudgetPluginFactory;
use lash::plugins::{PluginFactory, PluginSpec, StaticPluginFactory};
use lash::provider::{ProviderHandle, ProviderSpec, build_provider};
use lash::tools::{ToolCall, ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult};
use lash::tracing::{JsonlTraceSink, TraceContext, TraceLevel, TraceSink};
use lash::{LashCore, ModeId, ModePreset, SessionSpec, TurnInput, TurnOutput};
use lash_export::{ExportFormat, load_session_from_paths, render};
use lash_rlm_types::RlmTermination;
use lash_sqlite_store::Store;
use lash_subagents::{
    CapabilityRegistry, LocalSubagentHost, StaticCapability, SubagentHost, SubagentsPluginFactory,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

use crate::tournament::TournamentRerankProvider;

const DEFAULT_DATA_DIR: &str = ".benchmarks/obliq/data";
const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";
const DEFAULT_COLLECTION: &str = "obliq_analogues";
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

const DEFAULT_SUBSETS: &str = "math";
const DEFAULT_DESCRIPTION: &str = "Queries and documents come from an OBLIQ-Bench analogue subset. Relevant documents share the SAME abstract strategy, structure, or latent pattern as the query, even when surface topic, vocabulary, or formatting differs. Surface lexical overlap alone does NOT indicate relevance.";
const MATH_REP10_TASKS: &[&str] = &[
    "q02193", "q01486", "q01066", "q02834", "q01757", "q02979", "q01298", "q00844", "q00847",
    "q01488",
];
const WRITING_REP10_TASKS: &[&str] = &[
    "1", "182", "306", "487", "708", "951", "1727", "1987", "2105", "2213",
];
const NAMED_TASK_SUBSETS: &[NamedTaskSubset] = &[
    NamedTaskSubset {
        name: "math-rep10",
        subset: "math",
        task_ids: MATH_REP10_TASKS,
    },
    NamedTaskSubset {
        name: "writing-rep10",
        subset: "writing",
        task_ids: WRITING_REP10_TASKS,
    },
];

struct NamedTaskSubset {
    name: &'static str,
    subset: &'static str,
    task_ids: &'static [&'static str],
}

#[derive(Parser, Debug)]
#[command(name = "lash-oblique")]
#[command(about = "Run OBLIQ-Bench analogue tasks through Lash RLM with Qdrant retrieval tools.")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(name = "run-task", alias = "run-math")]
    RunMath {
        #[arg(long, default_value = "math")]
        subset: String,
        #[arg(long, alias = "query-id")]
        task_id: String,
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
        #[arg(long, default_value = DEFAULT_DESCRIPTION)]
        description: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run many tasks concurrently. Each task gets its own
    /// session.db, trace.jsonl, and output JSON. Writes a batch
    /// summary at `<output_dir>/_batch_summary.json` at the end.
    #[command(name = "run-batch", alias = "run-math-batch")]
    RunMathBatch {
        /// Comma-separated task IDs. Use `subset/id` to disambiguate; bare IDs
        /// are resolved through the discovered task index.
        #[arg(long, alias = "query-ids")]
        tasks: Option<String>,
        /// Comma-separated dataset subsets or named subsets to include when
        /// `--tasks` is omitted. Named subsets: math-rep10, writing-rep10,
        /// rep20.
        #[arg(long, default_value = DEFAULT_SUBSETS)]
        subsets: String,
        /// Limit to the first N tasks (after the `--tasks` filter,
        /// if any). Useful for sampling.
        #[arg(long)]
        limit: Option<usize>,
        /// How many tasks run concurrently. Each task may itself
        /// fan out (tournament_rerank uses up to 8 parallel batches),
        /// so total concurrent LLM calls ≈ concurrency × 8.
        #[arg(long, default_value_t = 3)]
        concurrency: usize,
        /// Skip tasks whose output JSON already exists. On by
        /// default so a re-run resumes after a crash.
        #[arg(long, default_value_t = true)]
        skip_existing: bool,
        /// Where per-task outputs land.
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
        #[arg(long, default_value = DEFAULT_DESCRIPTION)]
        description: String,
    },
    #[command(name = "eval-task", alias = "eval-math")]
    EvalMath {
        #[arg(long, default_value = "math")]
        subset: String,
        #[arg(long, alias = "query-id")]
        task_id: String,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct TaskRef {
    subset: String,
    task_id: String,
}

impl TaskRef {
    fn new(subset: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            subset: subset.into(),
            task_id: task_id.into(),
        }
    }

    fn label(&self) -> String {
        format!("{}/{}", self.subset, self.task_id)
    }
}

#[derive(Debug, Clone)]
struct Task {
    reference: TaskRef,
    query: QueryRow,
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
    subset: String,
    task_id: String,
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
            subset,
            task_id,
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
                task: TaskRef::new(subset, task_id),
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
            tasks,
            subsets,
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
                tasks,
                subsets,
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
            subset,
            task_id,
            submission,
            data_dir,
        } => {
            let raw = fs::read_to_string(&submission)
                .with_context(|| format!("read {}", submission.display()))?;
            let value: Value = serde_json::from_str(&raw)
                .with_context(|| format!("parse {}", submission.display()))?;
            let submitted = parse_submission(value)?;
            let task = TaskRef::new(subset, task_id);
            let excluded = load_excluded_doc_ids(&data_dir, &task)?;
            let sanitized = sanitize_ranked_doc_ids(submitted.ranked_doc_ids, &excluded);
            let bundle = score_submission_bundle(&sanitized.ranked_doc_ids, &data_dir, &task)?;
            println!("{}", serde_json::to_string_pretty(&bundle)?);
        }
    }
    Ok(())
}

#[derive(Clone)]
struct RunMathParams {
    task: TaskRef,
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
    tasks: Option<String>,
    subsets: String,
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
    provider: ProviderHandle,
    params: RunMathParams,
) -> anyhow::Result<RunOutput> {
    let RunMathParams {
        task,
        data_dir,
        qdrant_url,
        collection,
        model,
        variant,
        max_context_tokens,
        description,
        output_path,
    } = params;
    let loaded = load_task(&data_dir, &task)?;
    let query = loaded.query;
    let excluded = load_excluded_doc_ids(&data_dir, &task)?;
    let subset_stats = load_subset_stats(&data_dir, &task.subset).unwrap_or_default();
    let artifacts = RunArtifacts::from_output(output_path.as_ref(), &task);
    artifacts.create_parent_dirs()?;
    let store = Arc::new(
        Store::open(Path::new(&artifacts.session_db))
            .with_context(|| format!("open {}", artifacts.session_db))?,
    );
    let tools = Arc::new(ObliqTools {
        script: PathBuf::from("scripts/query_math_qdrant.py"),
        data_dir,
        qdrant_url,
        collection,
        subset: task.subset.clone(),
        excluded_doc_ids: excluded.clone(),
        late_available: false,
    });
    let qdrant_stats = tools
        .preflight_qdrant()
        .await
        .with_context(|| format!("preflight retrieval for {}", task.label()))?;
    let late_available = qdrant_stats
        .get("late_available")
        .and_then(Value::as_bool)
        .unwrap_or(false);
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
        &loaded.reference,
        &query,
        &description,
    )?;
    let session = core
        .session(format!(
            "obliq-{}-{}-{}",
            task.subset,
            task.task_id,
            uuid::Uuid::new_v4()
        ))
        .rlm()
        .open()
        .await
        .context("open Lash RLM session")?;
    let schema = output_schema();
    let turn = session
        .turn(TurnInput::text(run_prompt(
            &query,
            &subset_stats,
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
        .context("run OBLIQ task")?;

    let submitted = match turn.submitted_value() {
        Some(value) => parse_submission(value.clone())?,
        other => bail!(
            "RLM did not submit ranked_doc_ids: submitted_value={other:?} outcome={:?} errors={:?} text={}",
            turn.result.outcome,
            turn.result.errors,
            terminal_turn_text(&turn)
        ),
    };
    let raw_ranked_doc_ids = submitted.ranked_doc_ids;
    let sanitized = sanitize_ranked_doc_ids(raw_ranked_doc_ids.clone(), &excluded);
    let metrics = score_submission_bundle(&sanitized.ranked_doc_ids, &tools.data_dir, &task).ok();
    if let Err(err) = export_single_session_html(
        Path::new(&artifacts.session_db),
        Path::new(&artifacts.trace_jsonl),
        Path::new(&artifacts.trace_html),
    ) {
        eprintln!("warn: failed to render trace html: {err:#}");
    }
    let run_output = RunOutput {
        subset: task.subset,
        task_id: query.id.clone(),
        query_id: query.id,
        query: query.text,
        raw_ranked_doc_ids,
        ranked_doc_ids: sanitized.ranked_doc_ids,
        excluded_doc_ids: excluded.iter().cloned().collect(),
        removed_doc_ids: sanitized.removed_doc_ids,
        metrics,
        tool_calls: turn.result.tool_calls.len(),
        final_text: terminal_turn_text(&turn),
        errors: turn
            .result
            .errors
            .into_iter()
            .map(|issue| issue.message)
            .collect(),
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
/// also changes agent behavior or selected task scope.
fn config_hash(model: &str, variant: &str, description: &str, selection: &str) -> String {
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
    h.update(b":");
    h.update(selection.as_bytes());
    let out = h.finalize();
    let mut s = String::with_capacity(12);
    for b in &out[..6] {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

async fn run_math_batch(params: RunMathBatchParams) -> anyhow::Result<()> {
    let RunMathBatchParams {
        tasks,
        subsets,
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

    let selection_key = tasks
        .as_ref()
        .map(|tasks| format!("tasks:{tasks}"))
        .unwrap_or_else(|| format!("subsets:{subsets}"));

    // Compute a stable cache key from agent source + runtime config.
    // Outputs land at `<output_dir>/<hash>/...` so unchanged-code re-runs
    // hit the existing files via `--skip-existing`, and any code/config
    // change automatically opens a fresh cache slot.
    let agent_hash = agent_spec_hash();
    let cfg_hash = config_hash(&model, &variant, &description, &selection_key);
    let scoped_output = output_dir.join(&cfg_hash);
    fs::create_dir_all(&scoped_output)
        .with_context(|| format!("create {}", scoped_output.display()))?;
    // Update a `_latest` pointer file so the dashboard can find the most
    // recent run for this output_dir without scanning.
    let _ = fs::write(output_dir.join("_latest"), format!("{cfg_hash}\n"));
    // Persist a small manifest so the dashboard can show what generated
    // this run without parsing every per-task JSON.
    let manifest = json!({
        "agent_spec_hash": agent_hash,
        "config_hash": cfg_hash,
        "model": model,
        "variant": variant,
        "description": description,
        "selection": selection_key,
        "subsets": subsets,
        "tasks": tasks,
    });
    let _ = fs::write(
        scoped_output.join("_manifest.json"),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    );
    let output_dir = scoped_output;

    let task_index = load_task_index(&data_dir)?;
    let mut requested: Vec<TaskRef> = if let Some(tasks) = tasks.as_deref() {
        parse_task_list(tasks, &task_index)?
    } else {
        let selected_subsets = parse_csv(&subsets);
        tasks_for_subsets(&task_index, &selected_subsets)?
    };
    if let Some(n) = limit {
        requested.truncate(n);
    }
    let total_requested = requested.len();
    if total_requested == 0 {
        bail!("no tasks to run");
    }

    // Filter out tasks whose output already exists, if requested.
    let to_run: Vec<TaskRef> = if skip_existing {
        requested
            .into_iter()
            .filter(|task| !RunArtifacts::output_path_in_dir(&output_dir, task).exists())
            .collect()
    } else {
        requested
    };
    let skipped = total_requested - to_run.len();
    if to_run.is_empty() {
        eprintln!("[batch] all {total_requested} tasks already have outputs; nothing to run");
        write_batch_summary(&output_dir, &description, &model, &variant, &[])?;
        return Ok(());
    }
    eprintln!(
        "[batch] running {} of {total_requested} tasks (skipping {skipped} that already exist), concurrency={concurrency}",
        to_run.len(),
    );

    // Resolve provider once. ProviderHandle is Clone; share across tasks.
    let provider = resolve_provider(provider_id.as_deref())?;

    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut tasks_join: tokio::task::JoinSet<(TaskRef, anyhow::Result<RunOutput>)> =
        tokio::task::JoinSet::new();
    for task in &to_run {
        let provider = provider.clone();
        let semaphore = semaphore.clone();
        let params = RunMathParams {
            task: task.clone(),
            data_dir: data_dir.clone(),
            qdrant_url: qdrant_url.clone(),
            collection: collection.clone(),
            model: model.clone(),
            variant: variant.clone(),
            max_context_tokens,
            description: description.clone(),
            output_path: Some(RunArtifacts::output_path_in_dir(&output_dir, task)),
        };
        let task_owned = task.clone();
        tasks_join.spawn(async move {
            let _permit = match semaphore.acquire_owned().await {
                Ok(p) => p,
                Err(err) => return (task_owned, Err(anyhow::anyhow!("semaphore closed: {err}"))),
            };
            let result = run_one_math_query(provider, params).await;
            (task_owned, result)
        });
    }

    let total_to_run = to_run.len();
    let mut completed: Vec<(TaskRef, anyhow::Result<RunOutput>)> = Vec::with_capacity(total_to_run);
    while let Some(joined) = tasks_join.join_next().await {
        let (task, result) = joined.map_err(|err| anyhow::anyhow!("task join: {err}"))?;
        let n_done = completed.len() + 1;
        let label = task.label();
        match &result {
            Ok(out) => match &out.metrics {
                Some(b) => eprintln!(
                    "[{n_done}/{total_to_run}] {label} — G:NDCG@10={:.3} R@100={:.3} | P:NDCG@10={:.3} R@100={:.3} tools={}",
                    b.gold.ndcg_at_10,
                    b.gold.recall_at_100,
                    b.pooled.ndcg_at_10,
                    b.pooled.recall_at_100,
                    out.tool_calls
                ),
                None => eprintln!(
                    "[{n_done}/{total_to_run}] {label} — submitted (no qrels) tools={}",
                    out.tool_calls
                ),
            },
            Err(err) => eprintln!("[{n_done}/{total_to_run}] {label} — FAILED: {err:#}"),
        }
        completed.push((task, result));
    }

    let failed = completed
        .iter()
        .filter(|(_, result)| result.is_err())
        .count();
    write_batch_summary(&output_dir, &description, &model, &variant, &completed)?;
    if failed > 0 {
        bail!("batch failed: {failed} of {total_to_run} tasks failed");
    }
    Ok(())
}

fn write_batch_summary(
    output_dir: &Path,
    description: &str,
    model: &str,
    variant: &str,
    completed: &[(TaskRef, anyhow::Result<RunOutput>)],
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

    for (task, result) in completed {
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
                    "subset": task.subset,
                    "task_id": task.task_id,
                    "query_id": task.task_id,
                    "ok": true,
                    "tool_calls": out.tool_calls,
                    "metrics": out.metrics,
                }));
            }
            Err(err) => {
                per_query.push(json!({
                    "subset": task.subset,
                    "task_id": task.task_id,
                    "query_id": task.task_id,
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
        "per_task": per_query.clone(),
        "per_query": per_query,
    });
    let path = output_dir.join("_batch_summary.json");
    fs::write(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )
    .with_context(|| format!("write {}", path.display()))?;
    let g_n10 = if scored > 0 {
        sum_g.ndcg_at_10 / n
    } else {
        0.0
    };
    let g_r100 = if scored > 0 {
        sum_g.recall_at_100 / n
    } else {
        0.0
    };
    let p_n10 = if scored > 0 {
        sum_p.ndcg_at_10 / n
    } else {
        0.0
    };
    let p_r100 = if scored > 0 {
        sum_p.recall_at_100 / n
    } else {
        0.0
    };
    eprintln!(
        "[batch] summary written to {}: ok={ok}/{total} | G: NDCG@10={g_n10:.3} R@100={g_r100:.3} | P: NDCG@10={p_n10:.3} R@100={p_r100:.3}",
        path.display(),
    );
    Ok(())
}

type TaskIndex = BTreeMap<String, Vec<TaskRef>>;

fn load_task_index(data_dir: &Path) -> anyhow::Result<TaskIndex> {
    let mut index: TaskIndex = BTreeMap::new();
    for subset in discover_subsets(data_dir)? {
        let path = data_dir.join(&subset).join("queries.jsonl");
        let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            let row: QueryRow = serde_json::from_str(line)
                .with_context(|| format!("parse query row in {}", path.display()))?;
            index
                .entry(row.id.clone())
                .or_default()
                .push(TaskRef::new(subset.clone(), row.id));
        }
    }
    Ok(index)
}

fn discover_subsets(data_dir: &Path) -> anyhow::Result<Vec<String>> {
    let mut subsets = Vec::new();
    for entry in fs::read_dir(data_dir).with_context(|| format!("read {}", data_dir.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let subset = entry.file_name().to_string_lossy().to_string();
        if entry.path().join("queries.jsonl").exists() {
            subsets.push(subset);
        }
    }
    subsets.sort();
    Ok(subsets)
}

fn parse_csv(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_task_list(input: &str, index: &TaskIndex) -> anyhow::Result<Vec<TaskRef>> {
    parse_csv(input)
        .into_iter()
        .map(|token| resolve_task_token(&token, index))
        .collect()
}

fn resolve_task_token(token: &str, index: &TaskIndex) -> anyhow::Result<TaskRef> {
    if let Some((subset, task_id)) = token.split_once('/') {
        return Ok(TaskRef::new(subset.trim(), task_id.trim()));
    }
    let matches = index.get(token).cloned().unwrap_or_default();
    match matches.as_slice() {
        [task] => Ok(task.clone()),
        [] => bail!("task `{token}` was not found in any subset"),
        many => bail!(
            "task `{token}` is ambiguous across subsets: {}",
            many.iter()
                .map(TaskRef::label)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn tasks_for_subsets(index: &TaskIndex, subsets: &[String]) -> anyhow::Result<Vec<TaskRef>> {
    let discovered_subsets = discovered_subset_names(index);
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for selector in subsets {
        let mut selected = if selector == "rep20" {
            named_tasks("math-rep10", index)?
                .into_iter()
                .chain(named_tasks("writing-rep10", index)?)
                .collect()
        } else if named_task_subset(selector).is_some() {
            named_tasks(selector, index)?
        } else if discovered_subsets.contains(selector) {
            tasks_for_dataset_subset(index, selector)
        } else {
            bail!(
                "unknown subset selector `{selector}`; known dataset subsets: {}; named subsets: {}",
                discovered_subsets
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                named_subset_names().join(", ")
            );
        };
        for task in selected.drain(..) {
            if seen.insert(task.clone()) {
                out.push(task);
            }
        }
    }
    if out.is_empty() {
        bail!(
            "no tasks found for subset selectors: {}",
            subsets.join(", ")
        );
    }
    Ok(out)
}

fn tasks_for_dataset_subset(index: &TaskIndex, subset: &str) -> Vec<TaskRef> {
    let mut out = Vec::new();
    for tasks in index.values() {
        for task in tasks {
            if task.subset == subset {
                out.push(task.clone());
            }
        }
    }
    out.sort();
    out
}

fn named_tasks(name: &str, index: &TaskIndex) -> anyhow::Result<Vec<TaskRef>> {
    let named = named_task_subset(name).expect("checked by caller");
    let mut tasks = Vec::with_capacity(named.task_ids.len());
    for task_id in named.task_ids {
        let task = TaskRef::new(named.subset, *task_id);
        if !task_exists(index, &task) {
            bail!(
                "named subset `{}` references missing task `{}`",
                named.name,
                task.label()
            );
        }
        tasks.push(task);
    }
    Ok(tasks)
}

fn named_task_subset(name: &str) -> Option<&'static NamedTaskSubset> {
    NAMED_TASK_SUBSETS.iter().find(|subset| subset.name == name)
}

fn task_exists(index: &TaskIndex, task: &TaskRef) -> bool {
    index
        .get(&task.task_id)
        .map(|matches| matches.iter().any(|candidate| candidate == task))
        .unwrap_or(false)
}

fn discovered_subset_names(index: &TaskIndex) -> BTreeSet<String> {
    index
        .values()
        .flat_map(|tasks| tasks.iter().map(|task| task.subset.clone()))
        .collect()
}

fn named_subset_names() -> Vec<&'static str> {
    let mut names = NAMED_TASK_SUBSETS
        .iter()
        .map(|subset| subset.name)
        .collect::<Vec<_>>();
    names.push("rep20");
    names
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark setup keeps run-scoped Lash knobs explicit at the call site"
)]
fn build_core(
    provider: ProviderHandle,
    model: String,
    variant: String,
    max_context_tokens: usize,
    obliq_tools: Arc<ObliqTools>,
    store: Arc<dyn RuntimePersistence>,
    trace_path: PathBuf,
    task: &TaskRef,
    query: &QueryRow,
    description: &str,
) -> anyhow::Result<LashCore> {
    let subagent_spec = SessionSpec::inherit()
        .provider(provider.clone())
        .model(model.clone(), Some(variant.clone()))
        .max_context_tokens(max_context_tokens)
        .mode(ExecutionMode::new("rlm"));
    let list_async = Arc::new(ListAsyncHandlesTool);
    let registry = Arc::new(
        CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
            SUBAGENT_CAPABILITY,
            SessionSpec::inherit(),
        ))),
    );
    let subagents = Arc::new(
        SubagentsPluginFactory::new(
            registry,
            Arc::new(LocalSubagentHost::default()) as Arc<dyn SubagentHost>,
        )
        .with_session_spec(subagent_spec),
    );
    let tournament: Arc<dyn ToolProvider> = Arc::new(TournamentRerankProvider::new(
        obliq_tools.clone(),
        description.to_string(),
    ));

    LashCore::builder()
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(model, Some(variant))
        .max_context_tokens(max_context_tokens)
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .trace_sink(Some(
            Arc::new(JsonlTraceSink::new(trace_path)) as Arc<dyn TraceSink>
        ))
        .trace_level(TraceLevel::Extended)
        .trace_context(trace_context_for_query(task, query))
        .tools(obliq_tools)
        .plugin(Arc::new(ToolOutputBudgetPluginFactory::default()))
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

fn trace_context_for_query(task: &TaskRef, query: &QueryRow) -> TraceContext {
    let mut metadata = BTreeMap::new();
    metadata.insert("benchmark".to_string(), json!("obliq-bench"));
    metadata.insert("subset".to_string(), json!(task.subset));
    TraceContext {
        run_id: Some(format!("obliq-{}-{}", task.subset, query.id)),
        example_id: Some(query.id.clone()),
        split: Some(task.subset.clone()),
        metadata,
        ..TraceContext::default()
    }
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
    let late_note = if late_available {
        "The `late` search mode is available for optional single-channel diversity probes."
    } else {
        "The `late` search mode is not available; use `hybrid`, `bm25`, or `dense`."
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

# Strategy
1. State the hidden relevance schema in your own words: objects plus relations. Avoid the query's surface vocabulary.
2. Run one broad hybrid `search` using 4–6 surface probes, `limit=300`, `candidate_pool=1500`.
3. Read the top 8 returned texts inline. Classify each as schema match, surface-similar distractor, or unclear.
4. If top results are mostly distractors, run one schema-anchored hybrid `search` with 4–6 probes from different surface domains. Avoid the distractor vocabulary.
5. If you found at least one likely positive and one named distractor, run `discover_docs` with those positive/negative pairs.
6. Optional: add one single-channel diversity `search` if a distinct phrasing or channel is missing. {late_note}
7. Build `candidate_pools` with one pool per retrieval call. Use each call's ids in order. Do not pre-merge or truncate pools.
8. Call `tournament_rerank` once with `top_k=100`. Use its output as the submission order.

# Hard rules
- Every submitted id comes from a tool result.
- Use `tournament_rerank`; do not hand-rank the final list.
- Keep candidate pools separate; tournament handles RRF merge and the internal top-300 cap.
- Submit exactly 100 unique, non-excluded ids. If `tournament_rerank` returns fewer than 100, fill the tail with the next-best candidates from your raw pools (in any reasonable order).

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
        late_note = late_note,
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

fn terminal_turn_text(output: &TurnOutput) -> String {
    if let Some(value) = output.submitted_value() {
        return terminal_value_text(value);
    }
    if let Some((_tool_name, value)) = output.tool_value() {
        return terminal_value_text(value);
    }
    output.assistant_message().unwrap_or_default().to_string()
}

fn terminal_value_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

#[derive(Clone)]
struct ObliqTools {
    script: PathBuf,
    data_dir: PathBuf,
    qdrant_url: String,
    collection: String,
    subset: String,
    excluded_doc_ids: BTreeSet<String>,
    late_available: bool,
}

#[async_trait]
impl ToolProvider for ObliqTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let modes_clause = if self.late_available {
            "\"hybrid\" | \"bm25\" | \"dense\" | \"late\""
        } else {
            "\"hybrid\" | \"bm25\" | \"dense\""
        };
        let search_description = format!(
            "Search the OBLIQ corpus. `mode` selects the retrieval channel: {modes_clause}. \
             `hybrid` (default) does BM25+dense (and late, if available) fused via RRF over \
             all `queries` — pass 4–6 probes per call. Single-channel modes (`bm25` / \
             `dense` / `late`) take only `queries[0]`. Returns `{{ matches: [...] }}` \
             ordered best-first; each match has `rank`, `doc_id`, `score`, `text` (full \
             text inline), and `metadata`. Read top-N entries directly from the result — \
             no separate fetch step needed."
        );
        let mode_enum: Vec<Value> = if self.late_available {
            vec![
                json!("hybrid"),
                json!("bm25"),
                json!("dense"),
                json!("late"),
            ]
        } else {
            vec![json!("hybrid"), json!("bm25"), json!("dense")]
        };
        vec![
            obliq_tool(
                "search",
                &search_description,
                json!({
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": { "type": "string", "minLength": 1 },
                            "minItems": 1,
                            "maxItems": 12
                        },
                        "mode": { "type": "string", "enum": mode_enum, "default": "hybrid" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 300, "default": 300 },
                        "candidate_pool": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 1000 }
                    },
                    "required": ["queries"],
                    "additionalProperties": false
                }),
                search_response_schema(),
            ),
            obliq_tool(
                "discover_docs",
                "Example-guided discovery: provide a target query plus positive-vs-negative document pairs to bias retrieval toward the latent pattern you want and away from surface-similar distractors. Returns `{ matches: [...] }` ordered best-first; each match has `rank`, `doc_id`, `score`, `text`, and `metadata`.",
                json!({
                    "type": "object",
                    "properties": {
                        "target_query": { "type": "string", "minLength": 1 },
                        "context_pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "positive_doc_id": { "type": "string", "minLength": 1 },
                                    "negative_doc_id": { "type": "string", "minLength": 1 }
                                },
                                "required": ["positive_doc_id", "negative_doc_id"],
                                "additionalProperties": false
                            },
                            "minItems": 1,
                            "maxItems": 20
                        },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 300, "default": 300 }
                    },
                    "required": ["target_query", "context_pairs"],
                    "additionalProperties": false
                }),
                search_response_schema(),
            ),
        ]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let result = match call.name {
            "search" => self.dispatch_search(call.args).await,
            other => self.call_script(other, call.args).await,
        };
        match result {
            Ok(value) => ToolResult::ok(value),
            Err(error) => ToolResult::err_fmt(error),
        }
    }
}

impl ObliqTools {
    async fn dispatch_search(&self, args: &Value) -> anyhow::Result<Value> {
        let mode = args.get("mode").and_then(Value::as_str).unwrap_or("hybrid");
        let queries = args
            .get("queries")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("search needs `queries: [str]`"))?;
        if queries.is_empty() {
            bail!("search needs at least one query");
        }
        let limit = args.get("limit").cloned();
        let candidate_pool = args.get("candidate_pool").cloned();
        let (op, payload) = match mode {
            "hybrid" => {
                let mut p = json!({ "queries": queries });
                if let Some(v) = limit {
                    p["limit"] = v;
                }
                if let Some(v) = candidate_pool {
                    p["candidate_pool"] = v;
                }
                ("hybrid_search", p)
            }
            "bm25" | "dense" => {
                let q = queries[0].as_str().unwrap_or_default();
                let mut p = json!({ "query": q });
                if let Some(v) = limit {
                    p["limit"] = v;
                }
                if mode == "bm25" {
                    ("bm25_search", p)
                } else {
                    ("dense_search", p)
                }
            }
            "late" => {
                if !self.late_available {
                    bail!("search mode `late` is not available on this collection");
                }
                let q = queries[0].as_str().unwrap_or_default();
                let mut p = json!({ "query": q });
                if let Some(v) = limit {
                    p["limit"] = v;
                }
                if let Some(v) = candidate_pool {
                    p["candidate_pool"] = v;
                }
                ("late_search", p)
            }
            other => bail!("unknown search mode: {other}"),
        };
        self.call_script(op, &payload).await
    }

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

    async fn preflight_qdrant(&self) -> anyhow::Result<Value> {
        let stats = self.call_script_raw("corpus_stats", &json!({})).await?;
        let points = stats
            .get("points_count")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        if points == 0 {
            bail!(
                "Qdrant collection `{}` at `{}` has no points; run `scripts/setup_math.sh --recreate` or pass the correct `--collection`",
                self.collection,
                self.qdrant_url
            );
        }
        Ok(stats)
    }

    async fn call_script(&self, op: &str, args: &Value) -> anyhow::Result<Value> {
        let value = self.call_script_raw(op, args).await?;
        Ok(self.filter_excluded(value))
    }

    async fn call_script_raw(&self, op: &str, args: &Value) -> anyhow::Result<Value> {
        let mut child = tokio::process::Command::new(&self.script)
            .arg("--op")
            .arg(op)
            .arg("--data-dir")
            .arg(&self.data_dir)
            .arg("--qdrant-url")
            .arg(&self.qdrant_url)
            .arg("--collection")
            .arg(&self.collection)
            .arg("--subset")
            .arg(&self.subset)
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
    fn from_output(output: Option<&PathBuf>, task: &TaskRef) -> Self {
        let output_json = output
            .cloned()
            .unwrap_or_else(|| Self::output_path_in_dir(Path::new(".benchmarks/obliq/runs"), task));
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

    fn output_path_in_dir(output_dir: &Path, task: &TaskRef) -> PathBuf {
        output_dir
            .join(&task.subset)
            .join(format!("{}.json", task.task_id))
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

fn obliq_tool(
    name: &str,
    description: &str,
    input_schema: Value,
    output_schema: Value,
) -> ToolDefinition {
    ToolDefinition::raw(name, description, input_schema, output_schema)
        .with_execution_mode(ToolExecutionMode::Parallel)
}

fn search_match_item_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "rank": { "type": "integer", "minimum": 1 },
            "doc_id": { "type": "string" },
            "score": { "type": "number" },
            "text": { "type": "string" },
            "metadata": { "type": "object", "additionalProperties": true }
        },
        "required": ["rank", "doc_id", "score", "text", "metadata"],
        "additionalProperties": false
    })
}

fn search_response_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": search_match_item_schema()
            },
            "excluded_doc_ids_hidden": { "type": "integer", "minimum": 0 }
        },
        "required": ["matches"],
        "additionalProperties": false
    })
}

struct ListAsyncHandlesTool;

#[async_trait]
impl ToolProvider for ListAsyncHandlesTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![list_async_handles_tool_definition()]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{}` is handled by the RLM runtime and cannot run directly",
            call.name
        ))
    }
}

fn list_async_handles_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "list_async_handles",
        "List live lashlang async handles only. Returns `{ monitor: { monitor_id: handle }, subagent: { name: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted.",
        ToolDefinition::default_input_schema(),
        json!({
            "type": "object",
            "properties": {
                "monitor": { "type": "object", "additionalProperties": true },
                "subagent": { "type": "object", "additionalProperties": true },
                "tool": { "type": "object", "additionalProperties": true }
            },
            "required": ["monitor", "subagent", "tool"],
            "additionalProperties": false
        }),
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn resolve_provider(provider_id: Option<&str>) -> anyhow::Result<ProviderHandle> {
    lash_providers_builtin::register_all();
    let config_path = lash_home().join("config.json");
    let config = load_provider_config(&config_path)?;
    let provider_kind = provider_id.unwrap_or(&config.active_provider);
    let spec = config
        .providers
        .get(provider_kind)
        .ok_or_else(|| anyhow::anyhow!("provider `{provider_kind}` is not configured"))?;
    build_provider(spec)
        .map(ProviderHandle::new)
        .map_err(anyhow::Error::msg)
}

#[derive(Debug, Deserialize)]
struct ProviderConfigFile {
    active_provider: String,
    providers: BTreeMap<String, ProviderSpec>,
}

fn load_provider_config(path: &Path) -> anyhow::Result<ProviderConfigFile> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let config: ProviderConfigFile =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    if !config.providers.contains_key(&config.active_provider) {
        bail!(
            "{} points to missing active provider `{}`",
            path.display(),
            config.active_provider
        );
    }
    Ok(config)
}

fn lash_home() -> PathBuf {
    std::env::var_os("LASH_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".lash")))
        .unwrap_or_else(|| PathBuf::from(".lash"))
}

fn load_task(data_dir: &Path, task: &TaskRef) -> anyhow::Result<Task> {
    let path = data_dir.join(&task.subset).join("queries.jsonl");
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let query = raw
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<QueryRow>)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .find(|row| row.id == task.task_id)
        .ok_or_else(|| {
            anyhow::anyhow!("task `{}` not found in {}", task.label(), path.display())
        })?;
    Ok(Task {
        reference: task.clone(),
        query,
    })
}

fn load_subset_stats(data_dir: &Path, subset: &str) -> anyhow::Result<MathStats> {
    let subset_dir = data_dir.join(subset);
    let corpus_docs = count_nonempty_lines(&subset_dir.join("corpus.jsonl"))?;
    Ok(MathStats { corpus_docs })
}

fn count_nonempty_lines(path: &Path) -> anyhow::Result<usize> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(raw.lines().filter(|line| !line.trim().is_empty()).count())
}

fn load_qrels_file(
    data_dir: &Path,
    filename: &str,
    task: &TaskRef,
) -> anyhow::Result<HashMap<String, f64>> {
    let path = data_dir.join(&task.subset).join(filename);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let mut gold = HashMap::new();
    for (index, line) in raw.lines().enumerate() {
        if index == 0 && line.starts_with("query-id") {
            continue;
        }
        let parts = line.split('\t').collect::<Vec<_>>();
        if parts.len() < 3 || parts[0] != task.task_id {
            continue;
        }
        let score = parts[2].parse::<f64>().unwrap_or(1.0);
        gold.insert(parts[1].to_string(), score);
    }
    Ok(gold)
}

fn load_qrels(data_dir: &Path, task: &TaskRef) -> anyhow::Result<HashMap<String, f64>> {
    load_qrels_file(data_dir, "qrels.tsv", task)
}

fn load_qrels_pool(data_dir: &Path, task: &TaskRef) -> anyhow::Result<HashMap<String, f64>> {
    load_qrels_file(data_dir, "qrels_pool.tsv", task)
}

fn score_submission_bundle(
    ranked: &[String],
    data_dir: &Path,
    task: &TaskRef,
) -> anyhow::Result<MetricsBundle> {
    let gold_qrels = load_qrels(data_dir, task)?;
    let pooled_qrels = load_qrels_pool(data_dir, task)?;
    let pooled_qrels = if pooled_qrels.is_empty() {
        gold_qrels.clone()
    } else {
        pooled_qrels
    };
    Ok(MetricsBundle {
        gold: score_submission(ranked, &gold_qrels),
        pooled: score_submission(ranked, &pooled_qrels),
    })
}

fn load_excluded_doc_ids(data_dir: &Path, task: &TaskRef) -> anyhow::Result<BTreeSet<String>> {
    let path = data_dir
        .join(&task.subset)
        .join("per_query_excluded_ids.json");
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let value: HashMap<String, Vec<String>> =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    Ok(value
        .get(&task.task_id)
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
