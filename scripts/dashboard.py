#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate a self-contained HTML dashboard for one OBLIQ-Bench batch run.

Reads:
  <run_dir>/_manifest.json        run config (model, variant, description, hashes)
  <run_dir>/_batch_summary.json   aggregate metrics + per-task rollup
  <run_dir>/<subset>/<task_id>.json per-task results
  <run_dir>/<subset>/<task_id>.trace.jsonl optional, for token/cost roll-up

Writes:
  <run_dir>/dashboard.html        single self-contained file

Usage:
  ./scripts/dashboard.py [run_dir]                    # uses uv shebang
  uv run scripts/dashboard.py                         # auto-pick from _latest
  uv run scripts/dashboard.py --base .benchmarks/obliq/runs --open

The dashboard's visual language matches lash-export/html_assets/style.css —
sodium / chalk / lichen on form-deep, monospace, no AI slop.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
from html import escape

# Per-million-token pricing in USD. Mirrors lash-export/src/html.rs's
# model_pricing_per_million() — keep in sync if you add models there.
PRICING = {
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-4.1": (2.50, 0.50, 10.0),
    "gpt-4o": (2.50, 0.50, 10.0),
    "claude-opus": (15.0, 1.50, 75.0),
    "claude-sonnet": (3.0, 0.30, 15.0),
    "claude-3-7": (3.0, 0.30, 15.0),
    "claude-haiku": (0.80, 0.08, 4.0),
    "gemini-2": (1.25, 0.30, 10.0),
}
DEFAULT_PRICING = (3.0, 0.30, 15.0)


def price_for(model: str | None) -> tuple[float, float, float]:
    if not model:
        return DEFAULT_PRICING
    m = model.lower()
    for prefix, p in PRICING.items():
        if m.startswith(prefix):
            return p
    return DEFAULT_PRICING


def estimate_cost(usage: dict, model: str | None) -> float:
    in_per_m, cached_per_m, out_per_m = price_for(model)
    inp = max(0, int(usage.get("input_tokens", 0) or 0))
    cached = max(0, int(usage.get("cached_input_tokens", 0) or 0))
    out = max(0, int(usage.get("output_tokens", 0) or 0))
    billed = max(0, inp - cached)
    return (
        billed * in_per_m / 1_000_000.0
        + cached * cached_per_m / 1_000_000.0
        + out * out_per_m / 1_000_000.0
    )


def aggregate_trace(trace_path: pathlib.Path, model: str | None) -> dict:
    if not trace_path.exists():
        return {}
    in_t = out_t = cached_t = reason_t = 0
    n_completed = n_failed = n_started = 0
    cost = 0.0
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("type")
            if t == "llm_call_started":
                n_started += 1
            elif t == "llm_call_failed":
                n_failed += 1
            elif t == "llm_call_completed":
                n_completed += 1
                u = e.get("usage") or {}
                in_t += int(u.get("input_tokens", 0) or 0)
                out_t += int(u.get("output_tokens", 0) or 0)
                cached_t += int(u.get("cached_input_tokens", 0) or 0)
                reason_t += int(u.get("reasoning_tokens", 0) or 0)
                cost += estimate_cost(u, model)
    return {
        "llm_started": n_started,
        "llm_completed": n_completed,
        "llm_failed": n_failed,
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cached_input_tokens": cached_t,
        "reasoning_tokens": reason_t,
        "est_cost_usd": cost,
    }


def find_run_dir(base: pathlib.Path, explicit: pathlib.Path | None) -> pathlib.Path:
    if explicit is not None:
        return explicit.resolve()
    latest = base / "_latest"
    if latest.exists():
        cfg_hash = latest.read_text().strip()
        return (base / cfg_hash).resolve()
    # Fallback: pick the dir with the most-recent _batch_summary.json
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "_batch_summary.json").exists()),
        key=lambda p: (p / "_batch_summary.json").stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        sys.exit(f"no run dirs with _batch_summary.json under {base}")
    return candidates[0].resolve()


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>obliq dashboard · {model}/{variant}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Azeret+Mono:wght@400;500;700;800&family=Chivo+Mono:wght@300;400;500;700&family=Spectral:wght@400;500;600&display=swap');
:root {{
  --form: #0e0d0b; --form-deep: #080807; --form-raised: #141412;
  --ash: #2a2a28; --ash-light: #3a3a34; --ash-mid: #4a4a44; --ash-text: #5a5a50;
  --chalk-dim: #7a7a70; --chalk-mid: #c8c4b8; --chalk: #e8e4d0;
  --sodium: #e8a33c; --sodium-deep: #b67c1f;
  --lichen: #8a9e6c; --error: #cc4444; --error-deep: #7a2727; --info: #8ca0b4;
  --line: rgba(232, 228, 208, 0.10);
  --line-strong: rgba(232, 163, 60, 0.28);
  --font-display: "Azeret Mono", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;
  --font-ui: "Chivo Mono", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;
  --font-body: "Spectral", Charter, "Iowan Old Style", Cambria, Georgia, serif;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; min-height: 100vh; }}
body {{
  color: var(--chalk);
  background:
    radial-gradient(circle at top left, rgba(232, 163, 60, 0.10), transparent 28rem),
    radial-gradient(circle at bottom right, rgba(138, 158, 108, 0.06), transparent 26rem),
    linear-gradient(180deg, var(--form-deep), var(--form));
  font-family: var(--font-ui);
  overflow-x: hidden;
}}
body::before {{
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: repeating-linear-gradient(-58deg, transparent 0, transparent 96px,
    rgba(232, 163, 60, 0.025) 96px, rgba(232, 163, 60, 0.025) 97px);
  opacity: 0.7;
}}
.page {{ position: relative; z-index: 1; max-width: 1640px; margin: 0 auto; padding: 28px 32px 48px; }}
.eyebrow {{
  font-family: var(--font-display); font-size: 11px; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--sodium); margin-bottom: 6px;
}}
.eyebrow .slash {{ color: var(--chalk-dim); }}
h1 {{
  font-family: var(--font-display); font-weight: 700; font-size: 28px;
  color: var(--chalk); margin: 0 0 18px; letter-spacing: -0.01em;
}}
.hero-meta {{
  display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: 12px; color: var(--chalk-mid);
  border: 1px solid var(--line); padding: 10px 14px; background: rgba(20,20,18,0.6);
  margin-bottom: 16px;
}}
.meta-row {{ display: inline-flex; gap: 6px; align-items: baseline; }}
.meta-key {{
  text-transform: uppercase; letter-spacing: 0.06em; font-size: 10px; color: var(--chalk-dim);
}}
.meta-val {{ color: var(--chalk); font-weight: 500; }}
.meta-val.hash {{
  color: var(--sodium); font-family: var(--font-display);
  cursor: pointer; border-bottom: 1px dotted transparent;
  transition: border-color 120ms;
}}
.meta-val.hash:hover {{ border-bottom-color: var(--sodium-deep); }}
.meta-val.hash.copied {{ color: var(--lichen); }}
.description {{
  font-family: var(--font-body); font-size: 14px; color: var(--chalk-mid);
  border-left: 2px solid var(--sodium-deep); padding: 8px 16px; margin: 0 0 24px;
  background: rgba(232, 163, 60, 0.04);
}}
.metric-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px; margin: 0 0 28px;
}}
.metric-card {{
  border: 1px solid var(--line); padding: 14px 16px; background: rgba(20,20,18,0.6);
}}
.metric-key {{
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--chalk-dim); margin-bottom: 4px;
}}
.metric-val {{
  font-family: var(--font-display); font-size: 26px; font-weight: 700; color: var(--chalk);
  line-height: 1;
}}
.metric-val.cost {{ color: var(--sodium); }}
.metric-val.ok {{ color: var(--lichen); }}
.metric-val.warn {{ color: var(--sodium-deep); }}
.metric-val.bad {{ color: var(--error); }}
.metric-val.zero {{ color: var(--ash-text); }}
.metric-key {{ cursor: help; }}
.metric-key[title]:hover {{ color: var(--chalk-mid); }}
.metric-sub {{ font-size: 12px; color: var(--chalk-dim); margin-top: 6px; }}
section {{ margin: 28px 0; }}
section h2 {{
  font-family: var(--font-display); font-size: 13px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--chalk);
  margin: 0 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line-strong);
}}
.histogram {{
  display: flex; align-items: end; gap: 4px; height: 140px;
  padding: 12px 14px; border: 1px solid var(--line); background: rgba(20,20,18,0.6);
}}
.hist-bar {{
  flex: 1; background: var(--sodium-deep);
  border-top: 2px solid var(--sodium); position: relative;
  transition: filter 120ms;
}}
.hist-bar.empty {{
  background: rgba(232, 163, 60, 0.04);
  border-top: 1px solid rgba(232, 163, 60, 0.10);
  height: 1px !important; align-self: end;
}}
.hist-bar:hover {{ filter: brightness(1.4); }}
.hist-bar-count {{
  position: absolute; top: -16px; left: 0; right: 0; text-align: center;
  font-size: 11px; color: var(--chalk);
}}
.histogram-axis {{
  display: flex; justify-content: space-between;
  font-family: var(--font-display); font-size: 11px; color: var(--chalk-dim);
  margin: 8px 14px 0; letter-spacing: 0.04em;
  font-variant-numeric: tabular-nums;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
thead th {{
  text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line-strong);
  font-family: var(--font-display); font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--chalk-dim); cursor: pointer; user-select: none;
  position: sticky; top: 0; background: var(--form-deep);
}}
thead th:hover {{ color: var(--sodium); }}
thead th.sorted {{ color: var(--sodium); }}
thead th.sorted::after {{ content: " ▾"; }}
thead th.sorted-asc::after {{ content: " ▴"; }}
tbody td {{
  padding: 6px 10px; border-bottom: 1px solid var(--line);
}}
tbody tr:hover td {{ background: rgba(232, 163, 60, 0.04); }}
.qid {{ font-family: var(--font-display); color: var(--sodium); }}
.qid a {{ color: inherit; text-decoration: none; border-bottom: 1px dotted var(--chalk-dim); }}
.qid a:hover {{ color: var(--chalk); border-bottom-color: var(--sodium); }}
tr.row-status-failed .qid {{ color: var(--error); }}
tr.row-status-failed .qid a {{ border-bottom-color: var(--error-deep); }}
.qid-untraced {{ color: var(--chalk-dim); }}
.num {{ font-family: var(--font-display); font-variant-numeric: tabular-nums; text-align: right; }}
.num-sub {{ display: block; font-size: 10px; color: var(--chalk-dim); font-weight: normal; }}
.num.zero {{ color: var(--ash-text); }}
.num.bad {{ color: var(--error); }}
.num.warn {{ color: var(--sodium-deep); }}
.num.ok {{ color: var(--lichen); }}
.row-status-failed td {{ color: var(--error); }}
.toolbar {{
  display: flex; gap: 12px; align-items: center; margin: 0 0 14px;
  font-size: 12px; color: var(--chalk-dim);
}}
.toolbar input {{
  background: var(--form-deep); color: var(--chalk); border: 1px solid var(--line);
  padding: 6px 10px; font-family: var(--font-ui); font-size: 12px; min-width: 220px;
}}
.toolbar input:focus {{ outline: none; border-color: var(--sodium); }}
footer {{
  margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--line);
  font-size: 12px; color: var(--chalk-dim);
}}
footer a {{
  color: var(--chalk-dim); text-decoration: none;
  border-bottom: 1px dotted var(--ash-mid);
}}
footer a:hover {{ color: var(--chalk); border-bottom-color: var(--sodium); }}
.run-path {{
  display: block; margin: 0 0 18px; font-size: 12px;
  color: var(--chalk-dim); word-break: break-all;
}}
.run-path .meta-key {{ margin-right: 6px; }}
.meta-val.danger {{ color: var(--error); }}
.failed-list {{
  margin-top: 8px; padding: 10px 14px; border-left: 2px solid var(--error);
  background: rgba(204,68,68,0.04); font-size: 12px; color: var(--chalk-mid);
}}
.failed-list li {{ margin: 2px 0; }}
.failed-list code {{ color: var(--error); font-size: 12px; }}
</style>
</head>
<body>
<div class="page">
  <div class="eyebrow">obliq <span class="slash">/</span> batch dashboard</div>
  <h1>{title}</h1>

  <div class="hero-meta">
    <span class="meta-row"><span class="meta-key">model</span><span class="meta-val">{model}/{variant}</span></span>
    <span class="meta-row"><span class="meta-key">config</span><span class="meta-val hash" title="{config_hash} (click to copy)" data-copy="{config_hash}">{config_hash}</span></span>
    <span class="meta-row"><span class="meta-key">agent</span><span class="meta-val hash" title="{agent_hash} (click to copy)" data-copy="{agent_hash}">{agent_hash}</span></span>
    <span class="meta-row"><span class="meta-key">tasks</span><span class="meta-val">{n_total}</span></span>
    <span class="meta-row"><span class="meta-key">scored</span><span class="meta-val">{n_scored}</span></span>
    <span class="meta-row"><span class="meta-key">failed</span><span class="meta-val{failed_class}">{n_failed}</span></span>
    <span class="meta-row"><span class="meta-key">when</span><span class="meta-val">{run_when}</span></span>
  </div>
  <div class="run-path"><span class="meta-key">run dir</span>{run_dir}</div>

  {description_block}

  <section>
    <h2>aggregate</h2>
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-key" title="{doc_ndcg10}">mean ndcg@10 P</div><div class="metric-val {cls_ndcg10}">{mean_ndcg10:.3f}</div><div class="metric-sub">G {g_ndcg10:.3f} · median {median_ndcg10:.3f} · σ {std_ndcg10:.3f}</div></div>
      <div class="metric-card"><div class="metric-key" title="{doc_ndcg50}">mean ndcg@50 P</div><div class="metric-val {cls_ndcg50}">{mean_ndcg50:.3f}</div><div class="metric-sub">G {g_ndcg50:.3f}</div></div>
      <div class="metric-card"><div class="metric-key" title="{doc_r10}">mean recall@10 P</div><div class="metric-val {cls_r10}">{mean_r10:.3f}</div><div class="metric-sub">G {g_r10:.3f}</div></div>
      <div class="metric-card"><div class="metric-key" title="{doc_r50}">mean recall@50 P</div><div class="metric-val {cls_r50}">{mean_r50:.3f}</div><div class="metric-sub">G {g_r50:.3f}</div></div>
      <div class="metric-card"><div class="metric-key" title="{doc_r100}">mean recall@100 P</div><div class="metric-val {cls_r100}">{mean_r100:.3f}</div><div class="metric-sub">G {g_r100:.3f}</div></div>
      <div class="metric-card"><div class="metric-key" title="{doc_tools}">mean tool calls</div><div class="metric-val">{mean_tools:.1f}</div></div>
      <div class="metric-card"><div class="metric-key" title="{doc_cost}">est cost · run</div><div class="metric-val cost">${total_cost:.2f}</div><div class="metric-sub">${mean_cost:.3f}/query · {total_calls} llm calls</div></div>
      <div class="metric-card"><div class="metric-key" title="{doc_tokens}">tokens (in/out)</div><div class="metric-val">{total_tokens_human}</div><div class="metric-sub">{cached_pct:.1f}% cached · reasoning {reasoning_human}</div></div>
    </div>
  </section>

  <section>
    <h2>ndcg@10 distribution</h2>
    <div class="histogram">{histogram_bars}</div>
    <div class="histogram-axis"><span>0.0</span><span>0.5</span><span>1.0</span></div>
  </section>

  {failed_section}

  <section>
    <h2>per-task results</h2>
    <div class="toolbar">
      <input id="q" type="search" placeholder="filter task…" autocomplete="off" spellcheck="false">
      <span id="qmeta"></span>
    </div>
    <table id="results">
      <thead><tr>
        <th data-sort="subset">subset</th>
        <th data-sort="qid">task_id</th>
        <th data-sort="num" data-key="ndcg10" title="P / G">NDCG@10</th>
        <th data-sort="num" data-key="ndcg50" title="P / G">NDCG@50</th>
        <th data-sort="num" data-key="r10" title="P / G">R@10</th>
        <th data-sort="num" data-key="r50" title="P / G">R@50</th>
        <th data-sort="num" data-key="r100" title="P / G">R@100</th>
        <th data-sort="num" data-key="tools">tools</th>
        <th data-sort="num" data-key="cost">$ est</th>
        <th data-sort="num" data-key="reason">reason tok</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>

  <footer>
    rendered by scripts/dashboard.py
    · <a href="https://github.com/openai/preparedness/tree/main/project/obliq-bench">about OBLIQ-Bench</a>
  </footer>
</div>

<script>
(function() {{
  const rows = Array.from(document.querySelectorAll('#results tbody tr'));
  const ths = Array.from(document.querySelectorAll('#results thead th'));
  const tbody = document.querySelector('#results tbody');
  const search = document.getElementById('q');
  const meta = document.getElementById('qmeta');
  let sortKey = 'ndcg10';
  let sortDir = 'desc';

  function applyFilter() {{
    const q = (search.value || '').trim().toLowerCase();
    let visible = 0;
    for (const row of rows) {{
      const qid = row.dataset.qid || '';
      const subset = row.dataset.subset || '';
      const task = row.dataset.task || '';
      const show = !q || qid.toLowerCase().includes(q) || subset.toLowerCase().includes(q) || task.toLowerCase().includes(q);
      row.style.display = show ? '' : 'none';
      if (show) visible++;
    }}
    meta.textContent = `${{visible}} / ${{rows.length}}`;
  }}

  function applySort() {{
    const key = sortKey, dir = sortDir;
    const sign = dir === 'asc' ? 1 : -1;
    const sorted = rows.slice().sort((a, b) => {{
      if (key === 'subset') return sign * a.dataset.subset.localeCompare(b.dataset.subset);
      if (key === 'qid') return sign * a.dataset.qid.localeCompare(b.dataset.qid);
      const av = parseFloat(a.dataset[key] || '0') || 0;
      const bv = parseFloat(b.dataset[key] || '0') || 0;
      return sign * (av - bv);
    }});
    sorted.forEach((r) => tbody.appendChild(r));
    ths.forEach((th) => {{
      th.classList.remove('sorted', 'sorted-asc');
      if (th.dataset.key === key || (key === 'qid' && th.dataset.sort === 'qid') || (key === 'subset' && th.dataset.sort === 'subset')) {{
        th.classList.add('sorted');
        if (dir === 'asc') th.classList.add('sorted-asc');
      }}
    }});
  }}

  ths.forEach((th) => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.key || (th.dataset.sort === 'qid' ? 'qid' : th.dataset.sort === 'subset' ? 'subset' : null);
      if (!key) return;
      if (sortKey === key) sortDir = sortDir === 'desc' ? 'asc' : 'desc';
      else {{ sortKey = key; sortDir = key === 'qid' || key === 'subset' ? 'asc' : 'desc'; }}
      applySort();
    }});
  }});

  search.addEventListener('input', applyFilter);
  applyFilter();
  applySort();

  document.querySelectorAll('.meta-val.hash[data-copy]').forEach((el) => {{
    el.addEventListener('click', async () => {{
      const text = el.dataset.copy || el.textContent;
      try {{ await navigator.clipboard.writeText(text); }} catch (e) {{ return; }}
      el.classList.add('copied');
      setTimeout(() => el.classList.remove('copied'), 900);
    }});
  }});
}})();
</script>
</body>
</html>
"""


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def score_class(v: float) -> str:
    """Map a 0..1 retrieval score to a semantic CSS class."""
    if v >= 0.5:
        return "ok"
    if v >= 0.2:
        return "warn"
    if v <= 0:
        return "zero"
    return "bad"


METRIC_DOCS = {
    "ndcg10": "Normalized Discounted Cumulative Gain at rank 10. Rewards relevant docs in the top 10, weighted by position.",
    "ndcg50": "Normalized Discounted Cumulative Gain at rank 50.",
    "r10":  "Recall at 10: fraction of gold-relevant docs retrieved in the top 10.",
    "r50":  "Recall at 50: fraction of gold-relevant docs retrieved in the top 50.",
    "r100": "Recall at 100: fraction of gold-relevant docs retrieved in the top 100 (the full submission).",
    "tools": "Mean number of retrieval tool calls per query.",
    "cost":  "Estimated USD cost across the run, derived from token usage and per-model pricing.",
    "tokens": "Total input/output tokens across all LLM calls in the run.",
}


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def entry_subset(entry: dict) -> str:
    return str(entry.get("subset") or "")


def entry_task_id(entry: dict) -> str:
    return str(entry.get("task_id") or entry.get("query_id") or "?")


def entry_task_label(entry: dict) -> str:
    subset = entry_subset(entry)
    task_id = entry_task_id(entry)
    return f"{subset}/{task_id}" if subset else task_id


def artifact_relpath(entry: dict, suffix: str) -> pathlib.Path | None:
    task_id = entry_task_id(entry)
    if not task_id or task_id == "?":
        return None
    subset = entry_subset(entry)
    if subset:
        return pathlib.Path(subset) / f"{task_id}{suffix}"
    return pathlib.Path(f"{task_id}{suffix}")


def render_histogram(values: list[float], bins: int | None = None) -> str:
    """Render an SVG-free flexbox histogram. Empty bins render as a 1px floor.

    Bin count adapts to sample size: small N gets coarser bins so the chart
    isn't dominated by empty cells.
    """
    if not values:
        return ""
    if bins is None:
        bins = max(5, min(20, len(values) * 2))
    counts = [0] * bins
    for v in values:
        idx = min(int(v * bins), bins - 1)
        if idx < 0:
            idx = 0
        counts[idx] += 1
    peak = max(counts) or 1
    bars = []
    for i, c in enumerate(counts):
        title = f"{i/bins:.2f}–{(i+1)/bins:.2f}: {c} tasks"
        if c == 0:
            bars.append(f'<div class="hist-bar empty" title="{title}"></div>')
            continue
        h = c / peak * 100
        bars.append(
            f'<div class="hist-bar" style="height: {h:.1f}%" title="{title}">'
            f'<span class="hist-bar-count">{c}</span></div>'
        )
    return "".join(bars)


def render_table_rows(per_query: list[dict]) -> str:
    rows = []
    for entry in per_query:
        subset = entry_subset(entry)
        task_id = entry_task_id(entry)
        label = entry_task_label(entry)
        ok = bool(entry.get("ok"))
        m = entry.get("metrics") or {}
        g = m.get("gold") or {}
        p = m.get("pooled") or {}
        # Pooled is the headline; gold shown in muted secondary cell.
        ndcg10 = float(p.get("ndcg_at_10") or 0)
        ndcg50 = float(p.get("ndcg_at_50") or 0)
        r10 = float(p.get("recall_at_10") or 0)
        r50 = float(p.get("recall_at_50") or 0)
        r100 = float(p.get("recall_at_100") or 0)
        g_ndcg10 = float(g.get("ndcg_at_10") or 0)
        g_ndcg50 = float(g.get("ndcg_at_50") or 0)
        g_r10 = float(g.get("recall_at_10") or 0)
        g_r50 = float(g.get("recall_at_50") or 0)
        g_r100 = float(g.get("recall_at_100") or 0)
        tools = int(entry.get("tool_calls") or 0)
        cost = float(entry.get("est_cost_usd") or 0)
        reason = int(entry.get("reasoning_tokens") or 0)
        trace_html = entry.get("trace_html_relpath")

        def numcls(v: float) -> str:
            return f"num {score_class(v)}"

        def cell(p_v: float, g_v: float) -> str:
            return (
                f'<td class="{numcls(p_v)}">{p_v:.3f}'
                f'<span class="num-sub">{g_v:.3f}</span></td>'
            )

        if trace_html:
            task_cell = f'<a href="{escape(trace_html)}">{escape(task_id)}</a>'
            qid_extra = ""
        else:
            task_cell = escape(task_id)
            qid_extra = " qid-untraced"
        tr_class_attr = '' if ok else ' class="row-status-failed"'
        rows.append(
            f'<tr{tr_class_attr} '
            f'data-subset="{escape(subset)}" data-task="{escape(task_id)}" '
            f'data-qid="{escape(label)}" '
            f'data-ndcg10="{ndcg10:.6f}" data-ndcg50="{ndcg50:.6f}" '
            f'data-r10="{r10:.6f}" data-r50="{r50:.6f}" data-r100="{r100:.6f}" '
            f'data-tools="{tools}" data-cost="{cost:.6f}" data-reason="{reason}">'
            f'<td>{escape(subset or "-")}</td>'
            f'<td class="qid{qid_extra}">{task_cell}</td>'
            f'{cell(ndcg10, g_ndcg10)}'
            f'{cell(ndcg50, g_ndcg50)}'
            f'{cell(r10, g_r10)}'
            f'{cell(r50, g_r50)}'
            f'{cell(r100, g_r100)}'
            f'<td class="num">{tools}</td>'
            f'<td class="num">${cost:.3f}</td>'
            f'<td class="num">{fmt_tokens(reason)}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def render_failed(per_query: list[dict]) -> str:
    failures = [e for e in per_query if not e.get("ok")]
    if not failures:
        return ""
    items = "\n".join(
        f'<li><code>{escape(entry_task_label(e))}</code>: {escape(str(e.get("error","unknown")))}</li>'
        for e in failures
    )
    return f"""
  <section>
    <h2>failed runs ({len(failures)})</h2>
    <ul class="failed-list">{items}</ul>
  </section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="path to a run dir; defaults to <base>/<_latest>")
    ap.add_argument("--base", default=".benchmarks/obliq/runs",
                    help="base output dir; default .benchmarks/obliq/runs")
    ap.add_argument("--open", action="store_true", help="xdg-open the dashboard at the end")
    args = ap.parse_args()

    base = pathlib.Path(args.base).resolve()
    run_dir = find_run_dir(base, pathlib.Path(args.run_dir).resolve() if args.run_dir else None)
    summary_path = run_dir / "_batch_summary.json"
    if not summary_path.exists():
        sys.exit(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text())

    manifest_path = run_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    model = summary.get("model") or manifest.get("model") or "?"
    variant = summary.get("variant") or manifest.get("variant") or ""
    description = summary.get("description") or manifest.get("description") or ""
    config_hash_v = manifest.get("config_hash") or run_dir.name
    agent_hash_v = manifest.get("agent_spec_hash") or "?"

    per_query = list(summary.get("per_task") or summary.get("per_query") or [])
    # Enrich with token totals + cost from individual trace.jsonl files.
    total_in = total_out = total_cached = total_reason = 0
    total_calls = 0
    total_cost = 0.0
    for entry in per_query:
        trace_relpath = artifact_relpath(entry, ".trace.jsonl")
        if trace_relpath is None:
            continue
        trace_path = run_dir / trace_relpath
        agg = aggregate_trace(trace_path, model)
        if agg:
            entry["est_cost_usd"] = agg["est_cost_usd"]
            entry["reasoning_tokens"] = agg["reasoning_tokens"]
            entry["llm_calls"] = agg["llm_completed"]
            total_in += agg["input_tokens"]
            total_out += agg["output_tokens"]
            total_cached += agg["cached_input_tokens"]
            total_reason += agg["reasoning_tokens"]
            total_calls += agg["llm_completed"]
            total_cost += agg["est_cost_usd"]
        # Wire up per-task trace HTML if it exists.
        trace_html_relpath = artifact_relpath(entry, ".trace.html")
        trace_html = run_dir / trace_html_relpath
        if trace_html.exists():
            entry["trace_html_relpath"] = trace_html_relpath.as_posix()

    ndcg10s = [
        float(((e.get("metrics") or {}).get("pooled") or {}).get("ndcg_at_10") or 0)
        for e in per_query
        if e.get("ok") and e.get("metrics")
    ]
    n_total = len(per_query)
    n_scored = len(ndcg10s)
    n_failed = sum(1 for e in per_query if not e.get("ok"))

    mean_cost = (total_cost / max(1, n_total - n_failed)) if (n_total - n_failed) else 0.0
    cached_pct = (total_cached * 100.0 / total_in) if total_in else 0.0
    subsets = sorted({entry_subset(entry) for entry in per_query if entry_subset(entry)})
    subset_label = ",".join(subsets) if subsets else "tasks"
    title = f"obliq · {subset_label} · {model}/{variant}".strip("/")

    mm = summary.get("mean_metrics") or {}
    mm_g = mm.get("gold") or {}
    mm_p = mm.get("pooled") or {}
    # Pooled is the paper's headline metric (Table 3 reports G/P; P is the
    # post-pool judgement). Show P in the metric cards for paper-aligned
    # comparison; per-task table shows both.
    mean_ndcg10 = float(mm_p.get("ndcg_at_10") or 0)
    mean_ndcg50 = float(mm_p.get("ndcg_at_50") or 0)
    mean_r10 = float(mm_p.get("recall_at_10") or 0)
    mean_r50 = float(mm_p.get("recall_at_50") or 0)
    mean_r100 = float(mm_p.get("recall_at_100") or 0)
    g_ndcg10 = float(mm_g.get("ndcg_at_10") or 0)
    g_ndcg50 = float(mm_g.get("ndcg_at_50") or 0)
    g_r10 = float(mm_g.get("recall_at_10") or 0)
    g_r50 = float(mm_g.get("recall_at_50") or 0)
    g_r100 = float(mm_g.get("recall_at_100") or 0)

    description_block = (
        f'<div class="description">{escape(description)}</div>' if description.strip() else ""
    )

    run_when = "?"
    try:
        ts = summary_path.stat().st_mtime
        run_when = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except OSError:
        pass

    html = HTML_TEMPLATE.format(
        title=escape(title),
        model=escape(model),
        variant=escape(variant),
        config_hash=escape(config_hash_v),
        agent_hash=escape(agent_hash_v),
        n_total=n_total,
        n_scored=n_scored,
        n_failed=n_failed,
        failed_class=" danger" if n_failed > 0 else "",
        run_dir=escape(str(run_dir)),
        run_when=escape(run_when),
        description_block=description_block,
        mean_ndcg10=mean_ndcg10,
        median_ndcg10=median(ndcg10s),
        std_ndcg10=std(ndcg10s),
        mean_ndcg50=mean_ndcg50,
        mean_r10=mean_r10,
        mean_r50=mean_r50,
        mean_r100=mean_r100,
        cls_ndcg10=score_class(mean_ndcg10),
        cls_ndcg50=score_class(mean_ndcg50),
        cls_r10=score_class(mean_r10),
        cls_r50=score_class(mean_r50),
        cls_r100=score_class(mean_r100),
        g_ndcg10=g_ndcg10,
        g_ndcg50=g_ndcg50,
        g_r10=g_r10,
        g_r50=g_r50,
        g_r100=g_r100,
        mean_tools=float(summary.get("mean_tool_calls") or 0),
        total_cost=total_cost,
        mean_cost=mean_cost,
        total_calls=total_calls,
        total_tokens_human=f"{fmt_tokens(total_in)}/{fmt_tokens(total_out)}",
        cached_pct=cached_pct,
        reasoning_human=fmt_tokens(total_reason),
        histogram_bars=render_histogram(ndcg10s),
        failed_section=render_failed(per_query),
        table_rows=render_table_rows(per_query),
        doc_ndcg10=escape(METRIC_DOCS["ndcg10"]),
        doc_ndcg50=escape(METRIC_DOCS["ndcg50"]),
        doc_r10=escape(METRIC_DOCS["r10"]),
        doc_r50=escape(METRIC_DOCS["r50"]),
        doc_r100=escape(METRIC_DOCS["r100"]),
        doc_tools=escape(METRIC_DOCS["tools"]),
        doc_cost=escape(METRIC_DOCS["cost"]),
        doc_tokens=escape(METRIC_DOCS["tokens"]),
    )

    out = run_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    if args.open:
        os.execvp("xdg-open", ["xdg-open", str(out)])


if __name__ == "__main__":
    main()
