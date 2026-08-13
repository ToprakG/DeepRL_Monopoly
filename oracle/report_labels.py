"""Generate a self-contained HTML report (charts + decoded action table) for
one oracle label_gen output.

Usage:
    python -m oracle.report_labels --labels oracle_labels/labels_seedX_Y.json

Writes ``<labels-stem>.report.html`` (or --output) next to the input by
default: seat win-rate/value chart, labels-per-game histogram, and a
decoded top-actions table (action ids resolved to names via
monopoly_game_engine.actions.action_to_description, with board square
names for property-indexed actions). Also writes a plain
``<stem>.actions.md`` table alongside for a quick non-HTML read.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from monopoly_game_engine.actions import action_to_description
from monopoly_game_engine.constants import BOARD

_SQ_RE = re.compile(r"sq=(-?\d+)")


def _decode_action(action_id: int) -> str:
    """action_to_description(), with sq=N property-index resolved to its board name."""
    raw = action_to_description(action_id)
    match = _SQ_RE.search(raw)
    if match:
        sq = int(match.group(1))
        name = BOARD.get(sq)
        if name:
            return _SQ_RE.sub(name, raw)
    return raw

TEMPLATE = """<title>{title}</title>
<style>
.viz-root {{
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --grid:           #e1e0d9;
  --axis:           #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --series-1:       #2a78d6;
  --series-2:       #eb6834;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary);
  background: var(--page);
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #898781; --grid: #2c2c2a;
    --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --text-muted: #898781; --grid: #2c2c2a;
  --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
.viz-root {{ min-height: 100%; padding: 32px 24px 64px; }}
.wrap {{ max-width: 1040px; margin: 0 auto; }}
h1 {{ font-size: 22px; font-weight: 650; margin: 0 0 4px; letter-spacing: -0.01em; }}
.subtitle {{ color: var(--text-secondary); font-size: 13.5px; margin: 0 0 28px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 28px; }}
.tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.tile .label {{ font-size: 11.5px; color: var(--text-secondary); margin-bottom: 6px; }}
.tile .value {{ font-size: 24px; font-weight: 650; font-variant-numeric: proportional-nums; }}
.tile .value small {{ font-size: 13px; font-weight: 500; color: var(--text-secondary); }}
.card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px 16px; margin-bottom: 20px; }}
.card h2 {{ font-size: 14.5px; font-weight: 650; margin: 0 0 2px; }}
.card .desc {{ font-size: 12.5px; color: var(--text-secondary); margin: 0 0 16px; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 760px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
svg {{ overflow: visible; display: block; }}
.axis-label {{ font-size: 10.5px; fill: var(--text-muted); }}
.gridline {{ stroke: var(--grid); stroke-width: 1; }}
.baseline {{ stroke: var(--axis); stroke-width: 1; }}
.ref-line {{ stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; }}
.ref-label {{ font-size: 9.5px; fill: var(--text-muted); }}
.bar {{ rx: 4; cursor: pointer; transition: opacity .12s; }}
.bar:hover, .bar.hover {{ opacity: 0.82; }}
.cat-label {{ font-size: 11px; fill: var(--text-secondary); text-anchor: middle; }}
.legend {{ display: flex; gap: 16px; margin-bottom: 14px; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); }}
.legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; flex: none; }}
.tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1);
  font-size: 11.5px; padding: 6px 9px; border-radius: 6px; opacity: 0; transform: translate(-50%, -100%);
  transition: opacity .08s; z-index: 10; white-space: nowrap; }}
.tooltip .tt-val {{ font-weight: 700; }}
.tooltip.show {{ opacity: 1; }}
.note {{ font-size: 12px; color: var(--text-secondary); border-top: 1px solid var(--border); margin-top: 8px; padding-top: 10px; line-height: 1.5; }}
table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
table.data-table th, table.data-table td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }}
table.data-table th:first-child, table.data-table td:first-child {{ text-align: left; }}
table.data-table th {{ color: var(--text-muted); font-weight: 600; font-size: 10.5px; text-transform: uppercase; letter-spacing: .02em; }}
details.table-toggle {{ margin-top: 24px; }}
details.table-toggle summary {{ cursor: pointer; font-size: 12.5px; color: var(--text-secondary); font-weight: 600; padding: 6px 0; }}
</style>
<div class="viz-root"><div class="wrap">
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
<div class="tiles">
  <div class="tile"><div class="label">Games</div><div class="value">{games}</div></div>
  <div class="tile"><div class="label">Labels</div><div class="value">{n_labels}</div></div>
  <div class="tile"><div class="label">Labels / game</div><div class="value">{mean_labels:.1f} <small>mean</small></div></div>
  <div class="tile"><div class="label">Truncated</div><div class="value">{truncated}</div></div>
  <div class="tile"><div class="label">Wall time</div><div class="value">{wall_min:.1f} <small>min</small></div></div>
  <div class="tile"><div class="label">Throughput</div><div class="value">{games_per_hour:.0f} <small>games/hr</small></div></div>
</div>
<div class="card">
  <h2>Win rate vs. mean backed-up value, by seat</h2>
  <p class="desc">Same 0&ndash;1 scale, grouped on one axis. Dashed line = 25% even-odds baseline for 4 players.</p>
  <div class="legend">
    <div class="legend-item"><span class="legend-swatch" style="background:var(--series-1)"></span>Win rate ({games} games)</div>
    <div class="legend-item"><span class="legend-swatch" style="background:var(--series-2)"></span>Mean backed-up value ({n_labels} labels)</div>
  </div>
  <svg id="chart-seats" width="100%" viewBox="0 0 760 260" preserveAspectRatio="xMinYMin meet"></svg>
</div>
<div class="grid2">
  <div class="card">
    <h2>Checkpoint labels per game</h2>
    <p class="desc">Distribution across {games} games.</p>
    <svg id="chart-hist" width="100%" viewBox="0 0 480 240" preserveAspectRatio="xMinYMin meet"></svg>
  </div>
  <div class="card">
    <h2>Most-selected actions (decoded)</h2>
    <p class="desc">Top {n_top_actions} of {n_unique_actions} unique actions.</p>
    <svg id="chart-actions" width="100%" viewBox="0 0 480 260" preserveAspectRatio="xMinYMin meet"></svg>
  </div>
</div>
<details class="table-toggle" open>
  <summary>Table view</summary>
  <div class="card" style="margin-top:8px;">
    <h2 style="margin-bottom:10px;">Seats</h2>
    <table class="data-table" id="table-seats">
      <thead><tr><th>Seat</th><th>Wins</th><th>Win rate</th><th>Mean value</th><th>Checkpoint labels</th></tr></thead>
      <tbody></tbody>
    </table>
    <h2 style="margin:20px 0 10px;">Decoded actions</h2>
    <table class="data-table" id="table-actions">
      <thead><tr><th>Action</th><th>ID</th><th>Count</th><th>Share</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</details>
</div></div>
<div class="tooltip" id="tooltip"><span class="tt-val"></span> <span class="tt-label"></span></div>
<script>
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) {{
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}}
const tooltip = document.getElementById("tooltip");
function showTooltip(evt, valueText, labelText) {{
  tooltip.querySelector(".tt-val").textContent = valueText;
  tooltip.querySelector(".tt-label").textContent = labelText;
  tooltip.style.left = evt.clientX + "px";
  tooltip.style.top = (evt.clientY - 10) + "px";
  tooltip.classList.add("show");
}}
function hideTooltip() {{ tooltip.classList.remove("show"); }}
function attachHover(node, valueText, labelText) {{
  node.addEventListener("pointermove", (e) => {{ node.classList.add("hover"); showTooltip(e, valueText, labelText); }});
  node.addEventListener("pointerleave", () => {{ node.classList.remove("hover"); hideTooltip(); }});
  node.setAttribute("tabindex", "0");
  node.addEventListener("focus", (e) => {{ node.classList.add("hover"); showTooltip(e, valueText, labelText); }});
  node.addEventListener("blur", () => {{ node.classList.remove("hover"); hideTooltip(); }});
}}

const seatData = {seat_data_json};
const games = {games};
const histBins = {hist_bins_json};
const histCounts = {hist_counts_json};
const topActions = {top_actions_json};
const totalLabels = {n_labels};

(function () {{
  const svg = document.getElementById("chart-seats");
  const W = 760, H = 260;
  const padL = 34, padR = 12, padT = 10, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const groupW = plotW / seatData.seats.length;
  const barW = 22, gap = 4;
  const yMax = Math.max(0.35, Math.max(...seatData.winRate, ...seatData.valueMean) * 1.15);
  const y = (v) => padT + plotH - (v / yMax) * plotH;
  const step = yMax > 0.3 ? 0.1 : 0.05;
  for (let v = 0; v <= yMax + 1e-9; v += step) {{
    el("line", {{ class: "gridline", x1: padL, x2: W - padR, y1: y(v), y2: y(v) }}, svg);
    el("text", {{ class: "axis-label", x: padL - 6, y: y(v) + 3, "text-anchor": "end" }}, svg).textContent = Math.round(v * 100) + "%";
  }}
  el("line", {{ class: "ref-line", x1: padL, x2: W - padR, y1: y(0.25), y2: y(0.25) }}, svg);
  el("text", {{ class: "ref-label", x: W - padR, y: y(0.25) - 4, "text-anchor": "end" }}, svg).textContent = "25% baseline";
  seatData.seats.forEach((seat, i) => {{
    const cx = padL + groupW * i + groupW / 2;
    const wr = seatData.winRate[i], vm = seatData.valueMean[i];
    const b1 = el("rect", {{ class: "bar", x: cx - gap/2 - barW, y: y(wr), width: barW, height: y(0) - y(wr), rx: 4, fill: "var(--series-1)" }}, svg);
    attachHover(b1, (wr * 100).toFixed(1) + "%", "Seat " + seat + " win rate");
    const b2 = el("rect", {{ class: "bar", x: cx + gap/2, y: y(vm), width: barW, height: y(0) - y(vm), rx: 4, fill: "var(--series-2)" }}, svg);
    attachHover(b2, vm.toFixed(3), "Seat " + seat + " mean value");
    el("text", {{ class: "cat-label", x: cx, y: y(0) + 18 }}, svg).textContent = "Seat " + seat;
  }});
  el("line", {{ class: "baseline", x1: padL, x2: W - padR, y1: y(0), y2: y(0) }}, svg);
}})();

(function () {{
  const svg = document.getElementById("chart-hist");
  const W = 480, H = 240;
  const padL = 30, padR = 8, padT = 10, padB = 40;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = histBins.length;
  const slot = plotW / n;
  const barW = Math.min(24, slot - 6);
  const yMax = Math.max(...histCounts, 1) * 1.15;
  const y = (v) => padT + plotH - (v / yMax) * plotH;
  [0, 0.25, 0.5, 0.75, 1].forEach((f) => {{
    const v = yMax * f;
    el("line", {{ class: "gridline", x1: padL, x2: W - padR, y1: y(v), y2: y(v) }}, svg);
    el("text", {{ class: "axis-label", x: padL - 6, y: y(v) + 3, "text-anchor": "end" }}, svg).textContent = Math.round(v);
  }});
  histBins.forEach((label, i) => {{
    const cx = padL + slot * i + slot / 2;
    const v = histCounts[i];
    const b = el("rect", {{ class: "bar", x: cx - barW/2, y: y(v), width: barW, height: y(0) - y(v), rx: 4, fill: "var(--series-1)" }}, svg);
    attachHover(b, v + " games", label + " labels");
    el("text", {{ class: "axis-label", x: cx, y: y(0) + 14, "text-anchor": "middle", transform: "rotate(-40 " + cx + " " + (y(0)+14) + ")" }}, svg).textContent = label;
  }});
  el("line", {{ class: "baseline", x1: padL, x2: W - padR, y1: y(0), y2: y(0) }}, svg);
}})();

(function () {{
  const svg = document.getElementById("chart-actions");
  const W = 480, H = 260;
  const padL = 40, padR = 8, padT = 10, padB = 56;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = topActions.length;
  const slot = plotW / n;
  const barW = Math.min(22, slot - 6);
  const maxCount = topActions.length ? topActions[0].count : 1;
  const yMax = maxCount * 1.1;
  const y = (v) => padT + plotH - (v / yMax) * plotH;
  const step = Math.ceil(maxCount / 4 / 100) * 100 || 1;
  for (let v = 0; v <= yMax; v += step) {{
    el("line", {{ class: "gridline", x1: padL, x2: W - padR, y1: y(v), y2: y(v) }}, svg);
    el("text", {{ class: "axis-label", x: padL - 6, y: y(v) + 3, "text-anchor": "end" }}, svg).textContent = v.toLocaleString();
  }}
  topActions.forEach((a, i) => {{
    const cx = padL + slot * i + slot / 2;
    const b = el("rect", {{ class: "bar", x: cx - barW/2, y: y(a.count), width: barW, height: y(0) - y(a.count), rx: 4, fill: "var(--series-1)" }}, svg);
    attachHover(b, a.count.toLocaleString() + " (" + (a.count / totalLabels * 100).toFixed(1) + "%)", a.name + " · id " + a.id);
    el("text", {{ class: "axis-label", x: cx, y: y(0) + 14, "text-anchor": "end", transform: "rotate(-50 " + cx + " " + (y(0)+14) + ")" }}, svg).textContent = a.name;
  }});
  el("line", {{ class: "baseline", x1: padL, x2: W - padR, y1: y(0), y2: y(0) }}, svg);
}})();

(function () {{
  const tbody = document.querySelector("#table-seats tbody");
  seatData.seats.forEach((seat, i) => {{
    const tr = document.createElement("tr");
    [
      "Seat " + seat,
      seatData.winCounts[i] + " / " + games,
      (seatData.winRate[i] * 100).toFixed(1) + "%",
      seatData.valueMean[i].toFixed(3),
      seatData.checkpointLabels[i].toLocaleString(),
    ].forEach((c) => {{ const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); }});
    tbody.appendChild(tr);
  }});
  const tbody2 = document.querySelector("#table-actions tbody");
  topActions.forEach((a) => {{
    const tr = document.createElement("tr");
    [a.name, String(a.id), a.count.toLocaleString(), (a.count / totalLabels * 100).toFixed(1) + "%"].forEach((c) => {{
      const td = document.createElement("td"); td.textContent = c; tr.appendChild(td);
    }});
    tbody2.appendChild(tr);
  }});
}})();
</script>
"""


def _histogram(nlabels: list[int], n_bins: int = 8) -> tuple[list[str], list[int]]:
    if not nlabels:
        return [], []
    lo, hi = min(nlabels), max(nlabels)
    if lo == hi:
        return [f"{lo}"], [len(nlabels)]
    edges = np.linspace(lo, hi + 1, n_bins + 1)
    counts, _ = np.histogram(nlabels, bins=edges)
    labels = [f"{int(edges[i])}-{int(edges[i+1]) - 1}" for i in range(n_bins)]
    return labels, counts.tolist()


def build_report(labels_json: Path, output: Path, top_n: int = 12) -> dict[str, Any]:
    meta = json.loads(labels_json.read_text(encoding="utf-8"))
    npz_path = labels_json.with_suffix(".npz")
    with np.load(npz_path) as d:
        actors = d["actors"]
        values = d["values"]
        selected_actions = d["selected_actions"]

    games_summaries = meta["game_summaries"]
    n_games = len(games_summaries)
    n_labels = int(meta["n_labels"])
    truncated = sum(1 for g in games_summaries if g.get("truncated"))
    winners = Counter(g["winner"] for g in games_summaries)
    num_seats = int(actors.max()) + 1 if len(actors) else 4

    win_counts = [winners.get(s, 0) for s in range(num_seats)]
    win_rate = [c / n_games if n_games else 0.0 for c in win_counts]
    value_mean = values.mean(axis=0).tolist() if len(values) else [0.0] * num_seats
    checkpoint_labels = [int((actors == s).sum()) for s in range(num_seats)]

    nlabels = [g["n_labels"] for g in games_summaries]
    hist_bins, hist_counts = _histogram(nlabels)

    action_counts = Counter(int(a) for a in selected_actions.tolist())
    top = action_counts.most_common(top_n)
    top_actions = [
        {"id": aid, "name": _decode_action(aid), "count": count}
        for aid, count in top
    ]

    wall_seconds = meta["throughput"]["wall_seconds"]
    games_per_hour = meta["throughput"]["games_per_hour"]
    mean_labels = meta["throughput"]["mean_labels_per_game"]
    seed_lo = min(g["seed"] for g in games_summaries) if games_summaries else meta.get("seed")
    seed_hi = max(g["seed"] for g in games_summaries) if games_summaries else meta.get("seed")

    html = TEMPLATE.format(
        title=f"Oracle label batch — seeds {seed_lo}–{seed_hi}",
        subtitle=(
            f"{n_games} games &middot; oracle.label_gen --calibrate &middot; "
            f"hybrid checkpoint labels (buy / build / trade / auction)"
        ),
        games=n_games,
        n_labels=n_labels,
        mean_labels=mean_labels,
        truncated=truncated,
        wall_min=wall_seconds / 60.0,
        games_per_hour=games_per_hour,
        n_top_actions=len(top_actions),
        n_unique_actions=len(action_counts),
        seat_data_json=json.dumps({
            "seats": list(range(num_seats)),
            "winCounts": win_counts,
            "winRate": win_rate,
            "valueMean": value_mean,
            "checkpointLabels": checkpoint_labels,
        }),
        hist_bins_json=json.dumps(hist_bins),
        hist_counts_json=json.dumps(hist_counts),
        top_actions_json=json.dumps(top_actions),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    actions_md_lines = [
        f"# Decoded actions — seeds {seed_lo}–{seed_hi}",
        "",
        f"{n_labels} labels, {len(action_counts)} unique actions, top {len(top_actions)} shown.",
        "",
        "| ID | Action | Count | Share |",
        "|---|---|---|---|",
    ]
    for a in top_actions:
        share = a["count"] / n_labels * 100 if n_labels else 0.0
        actions_md_lines.append(f"| {a['id']} | {a['name']} | {a['count']} | {share:.1f}% |")
    actions_md_path = output.with_name(output.stem.replace(".report", "") + ".actions.md")
    actions_md_path.write_text("\n".join(actions_md_lines) + "\n", encoding="utf-8")

    return {
        "html": output,
        "actions_md": actions_md_path,
        "games": n_games,
        "n_labels": n_labels,
        "win_rate": win_rate,
        "value_mean": value_mean,
        "top_actions": top_actions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a chart + decoded-action report for one label_gen output")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels_seedX_Y.json (meta)")
    parser.add_argument("--output", type=Path, default=None, help="Output .html path (default: <labels-stem>.report.html)")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args(argv)

    output = args.output or args.labels.with_name(args.labels.stem + ".report.html")
    result = build_report(args.labels, output, top_n=args.top)
    print(f"wrote {result['html']}")
    print(f"wrote {result['actions_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
