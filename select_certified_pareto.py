"""Select mean-value Pareto-optimal CNN candidates for measurement.

Uncertainty (std/lower/upper bounds) is intentionally ignored in Pareto selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mean_pareto_mask(maximize_values: np.ndarray, minimize_values: np.ndarray) -> np.ndarray:
    """Return a 2D mean-value Pareto mask in O(N log N).

    A competitor A dominates candidate B iff:
        A.maximize >= B.maximize
        A.minimize <= B.minimize
    and at least one inequality is strict.

    For Accuracy–Energy this means:
        A.accuracy_mean >= B.accuracy_mean
        A.energy_mean   <= B.energy_mean

    Exact duplicate points do NOT dominate one another because neither axis is
    strictly better. Therefore duplicate Pareto points are all retained.
    """
    maximize_values = np.asarray(maximize_values, dtype=float)
    minimize_values = np.asarray(minimize_values, dtype=float)

    if len(maximize_values) != len(minimize_values):
        raise ValueError("maximize_values and minimize_values must have the same length.")
    finite = np.isfinite(maximize_values) & np.isfinite(minimize_values)
    if not finite.all():
        bad = np.flatnonzero(~finite)[:10]
        raise ValueError(f"Non-finite mean values at indices {bad.tolist()}")

    n = len(maximize_values)
    order = np.argsort(minimize_values, kind="mergesort")
    is_pareto = np.ones(n, dtype=bool)

    # Best maximize value among strictly smaller minimize values.
    best_from_lower_minimize = -np.inf
    pos = 0

    while pos < n:
        group_start = pos
        current_min = minimize_values[order[pos]]
        while pos < n and minimize_values[order[pos]] == current_min:
            pos += 1
        group = order[group_start:pos]

        group_max = np.max(maximize_values[group])

        # Dominated by a point with strictly smaller minimize value, or by a
        # point at the same minimize value with strictly larger maximize value.
        dominated_from_left = best_from_lower_minimize >= maximize_values[group]
        dominated_within_group = group_max > maximize_values[group]
        is_pareto[group] = ~(dominated_from_left | dominated_within_group)

        best_from_lower_minimize = max(best_from_lower_minimize, group_max)

    return is_pareto


def mean_pareto_mask_bruteforce(maximize_values: np.ndarray, minimize_values: np.ndarray) -> np.ndarray:
    """Reference O(N^2) implementation for validation/tests."""
    maximize_values = np.asarray(maximize_values, dtype=float)
    minimize_values = np.asarray(minimize_values, dtype=float)
    n = len(maximize_values)
    out = np.ones(n, dtype=bool)
    for b in range(n):
        for a in range(n):
            if a == b:
                continue
            no_worse = (
                maximize_values[a] >= maximize_values[b]
                and minimize_values[a] <= minimize_values[b]
            )
            strict = (
                maximize_values[a] > maximize_values[b]
                or minimize_values[a] < minimize_values[b]
            )
            if no_worse and strict:
                out[b] = False
                break
    return out


def choose_mean_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    """Return the first available mean/prediction column from ``candidates``."""
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"No usable {label} column found. Tried: {', '.join(candidates)}")

def add_ranked(selected: list[int], reasons: dict[int, str], ordered: np.ndarray, count: int, reason: str) -> None:
    added = 0
    for index in ordered:
        index = int(index)
        if index not in reasons:
            selected.append(index)
            reasons[index] = reason
            added += 1
            if added == count:
                return


def structural_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric_columns = [column for column in frame if column.startswith("feature_") and not column.endswith("_pattern")]
    categorical_columns = [column for column in frame if column.endswith("_pattern")]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scale = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, 1.0)
    numeric = (numeric - numeric.median()) / scale
    categorical = pd.get_dummies(frame[categorical_columns].astype(str), dtype=float)
    return np.column_stack((numeric.to_numpy(float), categorical.to_numpy(float)))


def add_diverse(
    frame: pd.DataFrame,
    selected: list[int],
    reasons: dict[int, str],
    count: int,
    energy_mean: np.ndarray,
) -> None:
    """Add structurally diverse candidates without using uncertainty."""
    energy_cut = np.quantile(energy_mean, 0.75)
    param_cut = frame["parameter_count"].quantile(0.25)
    eligible = np.flatnonzero(
        (energy_mean <= energy_cut)
        & (frame["parameter_count"].to_numpy() >= param_cut)
    )
    eligible = np.array([index for index in eligible if int(index) not in reasons], dtype=int)
    if not count or not len(eligible):
        return
    matrix = structural_matrix(frame)
    if selected:
        chosen = np.array(selected, dtype=int)
        distances = np.sqrt(((matrix[eligible, None] - matrix[chosen]) ** 2).sum(axis=2)).min(axis=1)
    else:
        distances = np.full(len(eligible), np.inf)
    for _ in range(min(count, len(eligible))):
        position = (
            int(np.argmax(frame.iloc[eligible]["accuracy_energy_score"].to_numpy()))
            if np.isinf(distances).all()
            else int(np.argmax(distances))
        )
        choice = int(eligible[position])
        selected.append(choice)
        reasons[choice] = "structural_diversity"
        distances = np.minimum(distances, np.sqrt(((matrix[eligible] - matrix[choice]) ** 2).sum(axis=1)))
        distances[position] = -np.inf

def generate_pareto_html(full_frame: pd.DataFrame, selected_indices: list[int], reasons: dict[int, str], output_html: Path) -> None:
    """Generate an interactive tabbed HTML viewer."""
    records = []
    selected_set = set(selected_indices)

    for idx, row in full_frame.iterrows():
        is_acc_energy = bool(row.get("is_mean_accuracy_energy_pareto", False))
        is_acc_latency = bool(row.get("is_mean_accuracy_latency_pareto", False))
        is_cap_energy = bool(row.get("is_mean_capacity_energy_pareto", False))
        is_selected = idx in selected_set

        if not (is_acc_energy or is_acc_latency or is_cap_energy or is_selected):
            continue

        energy_j = float(row["_mean_energy_j"])
        latency_ms = float(row["_mean_latency_ms"])
        acc_pct = float(row["_mean_accuracy_percent"])
        reason_str = reasons.get(idx, "pareto_frontier")

        records.append({
            "id": str(row["candidate_id"]),
            "rank": selected_indices.index(idx) + 1 if is_selected else 0,
            "reason": reason_str,
            "depth": int(row["depth"]),
            "pool_count": int(row["pool_count"]),
            "pattern": str(row["pattern"]),
            "growth": str(row["growth_pattern"]),
            "pools": str(row["pools"]),
            "channels": str(row["channels"]),
            "parameters": int(row["parameter_count"]),
            # Keep full precision in the JSON so browser-side sorting never
            # uses rounded display values. Formatting is done only at render time.
            "accuracy": acc_pct,
            "energy_mj": energy_j * 1000.0,
            "latency_ms": latency_ms,
            "is_acc_energy_pareto": is_acc_energy,
            "is_acc_latency_pareto": is_acc_latency,
            "is_cap_energy_pareto": is_cap_energy,
            "is_selected": is_selected,
        })

    json_data = json.dumps(records, indent=2)

    html_content = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mean Pareto CNN Explorer</title>
  <style>
    :root {{ color-scheme: light; font-family: "Segoe UI", "Noto Sans KR", -apple-system, sans-serif; color: #172033; background: #f4f6fa; }}
    body {{ margin: 0; padding: 24px 16px; }}
    main {{ max-width: 1240px; margin: auto; background: #fff; border: 1px solid #dce2ee; border-radius: 14px; box-shadow: 0 10px 30px #17203314; overflow: hidden; }}
    header {{ padding: 28px 32px 22px; background: linear-gradient(135deg, #1b3860, #145980, #167a80); color: #fff; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; font-weight: 800; }}
    header p {{ margin: 0; color: #e1f3ff; line-height: 1.6; font-size: 14px; opacity: 0.95; }}
    .tabs-nav {{ display: flex; background: #ebf0f7; border-bottom: 1px solid #d8e0ed; padding: 6px 12px 0; gap: 6px; flex-wrap: wrap; }}
    .tab-btn {{ padding: 12px 20px; font-size: 14px; font-weight: 700; color: #4a5a73; background: transparent; border: none; border-radius: 8px 8px 0 0; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; gap: 8px; }}
    .tab-btn:hover {{ color: #145980; background: #ffffff66; }}
    .tab-btn.active {{ color: #1b3860; background: #ffffff; border-top: 3px solid #145980; box-shadow: 0 -2px 10px #0000000d; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 11px; background: #dce6f5; color: #1e3a66; font-weight: 800; }}
    .tab-btn.active .badge {{ background: #145980; color: #fff; }}
    .controls {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px 20px; padding: 16px 28px; background: #fafbfc; border-bottom: 1px solid #e7edf5; }}
    .control-group {{ display: flex; align-items: center; gap: 10px; }}
    label {{ font-size: 13px; font-weight: 700; color: #475467; }}
    select, input {{ font: inherit; font-size: 13px; border: 1px solid #c5d0e3; padding: 7px 12px; border-radius: 6px; background: #fff; outline: none; }}
    #search-box {{ width: 180px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
    th {{ background: #f1f5fa; color: #2d3b52; font-weight: 700; text-align: left; padding: 12px 14px; border-bottom: 2px solid #dde4f0; white-space: nowrap; }}
    td {{ padding: 12px 14px; border-bottom: 1px solid #e9eef5; vertical-align: middle; }}
    tbody tr:hover {{ background: #f4f8ff; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
    .pattern-tag {{ font-weight: 700; color: #0d4b75; background: #eef6fc; padding: 3px 8px; border-radius: 4px; font-size: 12px; }}
    .mono {{ font-family: "Cascadia Code", Consolas, monospace; font-size: 12px; color: #334155; }}
    .acc-highlight {{ color: #05603a; font-weight: 800; background: #ecfdf3; padding: 4px 8px; border-radius: 6px; display: inline-block; }}
    .energy-highlight {{ color: #b42318; font-weight: 800; background: #fef3f2; padding: 4px 8px; border-radius: 6px; display: inline-block; }}
    .lat-highlight {{ color: #b54708; font-weight: 700; background: #fffaeb; padding: 4px 8px; border-radius: 6px; display: inline-block; }}
    .reason-tag {{ display: inline-block; padding: 3px 9px; background: #eef2f6; color: #344054; border-radius: 99px; font-size: 11.5px; font-weight: 700; border: 1px solid #d0d5dd; }}
    .reason-tag.pareto {{ background: #e0f2fe; color: #0369a1; border-color: #bae6fd; }}
    footer {{ padding: 16px 28px; background: #f8fafc; border-top: 1px solid #e7edf5; color: #475467; font-size: 13px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
    .cond {{ font-family: Consolas, monospace; background: #f0f4fa; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>📈 Mean Pareto CNN Explorer</h1>
      <p>
        Mean-only 판정: <strong>불확실성(std/lower/upper)은 사용하지 않습니다.</strong><br>
        B는 <strong>A.Acc_mean ≥ B.Acc_mean AND A.Energy_mean ≤ B.Energy_mean</strong>인 A가 하나라도 있으면 탈락합니다. (적어도 한 축은 strict)
      </p>
    </header>

    <nav class="tabs-nav">
      <button class="tab-btn active" id="tab-acc-energy" onclick="setTab('acc-energy')">
        🎯 Mean Acc–Energy Pareto <span class="badge" id="badge-acc-energy">0</span>
      </button>
      <button class="tab-btn" id="tab-acc-latency" onclick="setTab('acc-latency')">
        ⚡ Mean Acc–Latency Pareto <span class="badge" id="badge-acc-latency">0</span>
      </button>
      <button class="tab-btn" id="tab-cap-energy" onclick="setTab('cap-energy')">
        📦 Mean Cap–Energy Pareto <span class="badge" id="badge-cap-energy">0</span>
      </button>
      <button class="tab-btn" id="tab-selected" onclick="setTab('selected')">
        📋 Selected Batch <span class="badge" id="badge-selected">0</span>
      </button>
    </nav>

    <section class="controls">
      <div class="control-group">
        <label for="sort-select">정렬 기준:</label>
        <select id="sort-select" onchange="render()">
          <option value="accuracy_desc">Accuracy ↑ (정확도 높은순)</option>
          <option value="energy_asc">Energy ↓ (에너지 적은순)</option>
          <option value="latency_asc">Latency ↓ (지연시간 짧은순)</option>
          <option value="parameters_desc">Parameters ↑ (파라미터 많은순)</option>
          <option value="id_asc">Candidate ID</option>
        </select>
      </div>
      <div class="control-group">
        <label for="pattern-select">채널 패턴 필터:</label>
        <select id="pattern-select" onchange="render()">
          <option value="all">전체 패턴 (All)</option>
          <option value="increasing">increasing</option>
          <option value="decreasing">decreasing</option>
          <option value="uniform">uniform</option>
          <option value="hourglass">hourglass</option>
          <option value="inverse_hourglass">inverse_hourglass</option>
        </select>
        <input type="text" id="search-box" placeholder="ID 검색..." oninput="render()">
      </div>
    </section>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Candidate ID</th>
            <th>Depth</th>
            <th>Pools</th>
            <th>Channel Pattern</th>
            <th>Pooling Layers</th>
            <th>Channels Sequence</th>
            <th class="num">Parameters</th>
            <th class="num">Pred. Accuracy</th>
            <th class="num">Pred. Energy</th>
            <th class="num">Pred. Latency</th>
            <th>Tag</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>

    <footer>
      <span id="footer-status">불러오는 중...</span>
      <span>Reject B if: <span class="cond">A.Acc_mean ≥ B.Acc_mean AND A.E_mean ≤ B.E_mean</span></span>
    </footer>
  </main>

  <script>
    const data = {json_data};
    let currentTab = "acc-energy";

    document.getElementById("badge-acc-energy").textContent = data.filter(d => d.is_acc_energy_pareto).length;
    document.getElementById("badge-acc-latency").textContent = data.filter(d => d.is_acc_latency_pareto).length;
    document.getElementById("badge-cap-energy").textContent = data.filter(d => d.is_cap_energy_pareto).length;
    document.getElementById("badge-selected").textContent = data.filter(d => d.is_selected).length;

    function setTab(t) {{
      currentTab = t;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.getElementById("tab-" + t).classList.add("active");
      const s = document.getElementById("sort-select");
      s.value = (t === "cap-energy") ? "parameters_desc" : "accuracy_desc";
      render();
    }}

    function getFilteredData() {{
      let filtered = [...data];
      if (currentTab === "acc-energy") filtered = filtered.filter(d => d.is_acc_energy_pareto);
      else if (currentTab === "acc-latency") filtered = filtered.filter(d => d.is_acc_latency_pareto);
      else if (currentTab === "cap-energy") filtered = filtered.filter(d => d.is_cap_energy_pareto);
      else if (currentTab === "selected") filtered = filtered.filter(d => d.is_selected);
      const pat = document.getElementById("pattern-select").value;
      if (pat !== "all") filtered = filtered.filter(d => d.pattern === pat);
      const q = document.getElementById("search-box").value.trim().toLowerCase();
      if (q) filtered = filtered.filter(d => d.id.toLowerCase().includes(q) || d.channels.includes(q));
      const sv = document.getElementById("sort-select").value;
      filtered.sort((a, b) => {{
        // Sort using the full-precision raw values stored in JSON.
        // Deterministic secondary keys make near-ties easier to inspect.
        if (sv === "accuracy_desc") {{
          const d = b.accuracy - a.accuracy;
          return d !== 0 ? d : a.energy_mj - b.energy_mj;
        }}
        if (sv === "energy_asc") {{
          const d = a.energy_mj - b.energy_mj;
          return d !== 0 ? d : b.accuracy - a.accuracy;
        }}
        if (sv === "latency_asc") {{
          const d = a.latency_ms - b.latency_ms;
          return d !== 0 ? d : b.accuracy - a.accuracy;
        }}
        if (sv === "parameters_desc") return b.parameters - a.parameters;
        return a.id.localeCompare(b.id);
      }});
      return filtered;
    }}

    function render() {{
      const rows = getFilteredData();
      const tbody = document.getElementById("rows");
      if (!rows.length) {{
        tbody.innerHTML = `<tr><td colspan="11" style="text-align:center;padding:30px;color:#888;">조건에 해당하는 구조가 없습니다.</td></tr>`;
        document.getElementById("footer-status").textContent = "표시 중: 0개";
        return;
      }}
      tbody.innerHTML = rows.map(r => {{
        const pt = r.reason.includes("pareto") ? "pareto" : "";
        return `<tr>
          <td><strong>${{r.id}}</strong></td>
          <td>${{r.depth}}</td>
          <td>${{r.pool_count}}</td>
          <td><span class="pattern-tag">${{r.pattern}}</span></td>
          <td class="mono">${{r.pools}}</td>
          <td class="mono">${{r.channels}}</td>
          <td class="num">${{r.parameters.toLocaleString()}}</td>
          <td class="num"><span class="acc-highlight">${{r.accuracy.toFixed(4)}}%</span></td>
          <td class="num"><span class="energy-highlight">${{r.energy_mj.toFixed(4)}} mJ</span></td>
          <td class="num"><span class="lat-highlight">${{r.latency_ms.toFixed(4)}} ms</span></td>
          <td><span class="reason-tag ${{pt}}">${{r.reason}}</span></td>
        </tr>`;
      }}).join("");
      document.getElementById("footer-status").textContent = `표시 중: ${{rows.length}}개 Mean Pareto 구조`;
    }}
    render();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")
    print(f"Generated Mean Pareto viewer: {output_html}")


def generate_interactive_explorer(_: pd.DataFrame, output_html: Path) -> None:
    """Create the lightweight explorer shell; its data is served at runtime."""
    template = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CNN Accuracy–Energy Explorer</title>
  <style>
    :root { font-family: "Segoe UI", "Noto Sans KR", sans-serif; color:#182235; background:#f4f7fb; }
    * { box-sizing:border-box; }
    body { margin:0; padding:24px; }
    main { max-width:1280px; margin:auto; background:#fff; border:1px solid #d8e0eb; border-radius:14px; overflow:hidden; box-shadow:0 10px 28px #17203312; }
    header { padding:24px 28px; background:linear-gradient(120deg,#123b67,#0d7888); color:#fff; }
    h1 { margin:0 0 6px; font-size:24px; } header p { margin:0; color:#e7f7fa; font-size:14px; }
    nav { display:flex; gap:6px; padding:10px 18px 0; border-bottom:1px solid #dfe6f0; background:#f8fafc; }
    button { font:inherit; cursor:pointer; } .tab { border:0; background:transparent; padding:11px 16px; color:#516076; font-weight:650; border-bottom:3px solid transparent; }
    .tab.active { color:#0c5f77; border-bottom-color:#0c7f91; }
    .panel { display:none; padding:22px 26px 26px; } .panel.active { display:block; }
    .note { margin:0 0 14px; color:#526176; font-size:13px; }
    .legend { display:flex; flex-wrap:wrap; gap:10px 16px; margin:0 0 12px; font-size:13px; }
    .legend span::before { content:""; display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px; vertical-align:-1px; background:var(--c); }
    .chart-actions { display:flex; justify-content:flex-end; margin:0 0 7px; }
    .chart-actions button { border:1px solid #9cb0c9; border-radius:6px; color:#164c6b; background:#fff; padding:5px 9px; font-size:13px; }
    .chart-wrap { position:relative; border:1px solid #d9e1ed; border-radius:8px; background:#fff; min-height:530px; }
    canvas { width:100%; height:530px; display:block; touch-action:none; }
    .tooltip { display:none; position:absolute; z-index:2; pointer-events:none; max-width:285px; padding:10px 12px; color:#f8fbff; background:#172235ed; border-radius:7px; font-size:12px; line-height:1.55; box-shadow:0 3px 12px #0004; }
    .toolbar { display:flex; flex-wrap:wrap; gap:12px 16px; align-items:end; margin-bottom:14px; }
    .control { display:grid; gap:5px; font-size:13px; font-weight:600; color:#45546a; }
    select, input { font:inherit; font-size:14px; min-height:34px; border:1px solid #bdc9da; border-radius:6px; padding:5px 8px; background:#fff; color:#182235; }
    .filters { border-top:1px solid #e3e9f2; padding-top:14px; margin:4px 0 14px; }
    .filters-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-weight:650; }
    .filter-row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:7px 0; }
    .filter-row select, .filter-row input { min-width:130px; }
    .add, .remove { border:1px solid #9cb0c9; border-radius:6px; color:#164c6b; background:#fff; padding:6px 10px; }
    .remove { color:#9a2d2d; border-color:#d6a7a7; } .status { color:#526176; font-size:13px; margin:10px 0 0; }
    @media (max-width:620px) { body{padding:10px;} .panel{padding:16px 12px;} header{padding:18px;} canvas{height:440px;} .chart-wrap{min-height:440px;} }
  </style>
</head>
<body>
<main>
  <header><h1>Accuracy–Energy Pareto Explorer</h1><p>평균 예측값 기준. Energy는 낮을수록, Accuracy는 높을수록 좋음.</p></header>
  <nav><button class="tab active" data-tab="pareto">Energy–Accuracy Pareto</button><button class="tab" data-tab="scatter">Custom scatter</button></nav>
  <section id="pareto" class="panel active">
    <p class="note">연한 점은 전체 50,000개 후보다. 색 점은 각 pattern 내부 Pareto, 검은 테두리는 전체 Accuracy–Energy Pareto다.</p>
    <div class="legend"><span style="--c:#8e99a9">All candidates</span><span style="--c:#222">Overall Pareto</span><span style="--c:#1f77b4">increasing</span><span style="--c:#e45756">decreasing</span><span style="--c:#54a24b">uniform</span><span style="--c:#b279a2">hourglass</span><span style="--c:#f2a541">inverse_hourglass</span></div>
    <div class="chart-actions"><input id="pareto-export-name" aria-label="Pareto PNG filename" value="accuracy-energy-pareto" spellcheck="false"><button type="button" data-export="pareto-canvas" data-name-input="pareto-export-name" data-filename="accuracy-energy-pareto">Export PNG</button><button type="button" data-reset="pareto-canvas">Reset view</button></div><p class="note">Wheel: both axes · Shift + wheel: X axis only · Alt + wheel: Y axis only</p>
    <div class="chart-wrap"><canvas id="pareto-canvas" aria-label="전체 및 pattern별 Accuracy-Energy Pareto 산점도"></canvas><div class="tooltip"></div></div>
  </section>
  <section id="scatter" class="panel">
    <div class="toolbar">
      <label class="control">X axis<select id="x-axis"></select></label>
      <label class="control">Y axis<select id="y-axis"></select></label>
    </div>
    <div class="filters"><div class="filters-head"><span>Constraints (AND)</span><button class="add" id="add-filter" type="button">+ Add constraint</button></div><div id="filter-list"></div></div>
    <div class="chart-wrap"><canvas id="custom-canvas" aria-label="조건 기반 사용자 산점도"></canvas><div class="tooltip"></div></div>
    <div class="chart-actions"><input id="custom-export-name" aria-label="Custom scatter PNG filename" value="custom-scatter" spellcheck="false"><button type="button" data-export="custom-canvas" data-name-input="custom-export-name" data-filename="custom-scatter">Export PNG</button><button type="button" data-reset="custom-canvas">Reset view</button></div><p class="note">Wheel: both axes · Shift + wheel: X axis only · Alt + wheel: Y axis only</p><p class="status" id="custom-status" aria-live="polite"></p>
  </section>
</main>
<script>
let DATA = [];
const PATTERN_COLORS = {increasing:'#1f77b4', decreasing:'#e45756', uniform:'#54a24b', hourglass:'#b279a2', inverse_hourglass:'#f2a541'};
const FIELDS = {
  accuracy:{label:'Predicted accuracy (%)', value:d=>d.accuracy}, energy:{label:'Predicted energy (mJ)', value:d=>d.energy}, latency:{label:'Predicted latency (ms)', value:d=>d.latency},
  parameters:{label:'Parameter count', value:d=>d.parameters}, depth:{label:'Depth', value:d=>d.depth}, poolCount:{label:'Pool count', value:d=>d.poolCount}, poolRatio:{label:'Pool ratio (pool_count / depth)', value:d=>d.poolRatio}
};
const FILTERS = {pattern:{label:'Pattern', type:'category', values:['increasing','decreasing','uniform','hourglass','inverse_hourglass']}, depth:{label:'Depth',type:'number'}, poolCount:{label:'Pool count',type:'number'}, poolRatio:{label:'Pool ratio',type:'number'}};
const CHART_VIEWS = new WeakMap();
function extent(values) { let lo=Math.min(...values), hi=Math.max(...values); const pad=(hi-lo||1)*0.06; return [lo-pad,hi+pad]; }
function fmt(key,v) { if(key==='parameters') return Math.round(v).toLocaleString(); if(key==='poolRatio') return v.toFixed(3); return Number(v).toFixed(key==='accuracy'||key==='energy'||key==='latency'?3:0); }
function tooltipText(d) { return `<b>${d.id}</b><br>pattern: ${d.pattern}<br>depth: ${d.depth} · pool count: ${d.poolCount} · ratio: ${d.poolRatio.toFixed(3)}<br>channels: ${d.channels}<br>parameters: ${d.parameters.toLocaleString()}<br>accuracy: ${d.accuracy.toFixed(3)}%<br>energy: ${d.energy.toFixed(4)} mJ<br>latency: ${d.latency.toFixed(4)} ms`; }
function drawAxes(ctx,w,h,box,xDomain,yDomain,xLabel,yLabel) {
  ctx.strokeStyle='#aab7c8'; ctx.fillStyle='#435269'; ctx.lineWidth=1; ctx.font='12px Segoe UI';
  ctx.strokeRect(box.l,box.t,box.w,box.h); ctx.textAlign='center'; ctx.fillText(xLabel,box.l+box.w/2,h-12); ctx.save(); ctx.translate(15,box.t+box.h/2); ctx.rotate(-Math.PI/2); ctx.fillText(yLabel,0,0); ctx.restore();
  for(let i=0;i<=5;i++){ const x=box.l+box.w*i/5, y=box.t+box.h*i/5; ctx.strokeStyle='#e6ebf2'; ctx.beginPath();ctx.moveTo(x,box.t);ctx.lineTo(x,box.t+box.h);ctx.stroke();ctx.beginPath();ctx.moveTo(box.l,y);ctx.lineTo(box.l+box.w,y);ctx.stroke(); ctx.fillStyle='#435269';ctx.textAlign='center';ctx.fillText(fmt('',xDomain[0]+(xDomain[1]-xDomain[0])*i/5),x,box.t+box.h+17);ctx.textAlign='right';ctx.fillText(fmt('',yDomain[1]-(yDomain[1]-yDomain[0])*i/5),box.l-7,y+4); }
}
function redrawSoon(canvas) { const state=CHART_VIEWS.get(canvas); if(state.frame)return; state.frame=requestAnimationFrame(()=>{state.frame=0;const r=state.render;setupChart(canvas,...r);}); }
function setupChart(canvas, rows, xKey, yKey, styleFn, statusId, statusText, overlayFn) {
  const wrap=canvas.parentElement, tip=wrap.querySelector('.tooltip'); const xs=rows.map(d=>FIELDS[xKey].value(d)),ys=rows.map(d=>FIELDS[yKey].value(d)); const defaultXd=extent(xs),defaultYd=extent(ys); let state=CHART_VIEWS.get(canvas); if(!state||state.xKey!==xKey||state.yKey!==yKey){state={xKey,yKey,xDomain:defaultXd,yDomain:defaultYd};CHART_VIEWS.set(canvas,state);} state.render=[rows,xKey,yKey,styleFn,statusId,statusText,overlayFn];
  const rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1,w=Math.max(360,rect.width),h=Math.max(360,rect.height);canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const box={l:70,t:20,w:w-95,h:h-72},xd=state.xDomain,yd=state.yDomain,sx=x=>box.l+(x-xd[0])/(xd[1]-xd[0])*box.w,sy=y=>box.t+box.h-(y-yd[0])/(yd[1]-yd[0])*box.h;state.box=box;
  drawAxes(ctx,w,h,box,xd,yd,FIELDS[xKey].label,FIELDS[yKey].label);ctx.save();ctx.beginPath();ctx.rect(box.l,box.t,box.w,box.h);ctx.clip();if(overlayFn)overlayFn(ctx,sx,sy);const grid=new Map(),cell=32; rows.forEach(d=>{const x=sx(FIELDS[xKey].value(d)),y=sy(FIELDS[yKey].value(d)),style=styleFn(d);if(!style)return;ctx.globalAlpha=style.alpha;ctx.fillStyle=style.fill;ctx.beginPath();ctx.arc(x,y,style.radius,0,Math.PI*2);ctx.fill();if(style.stroke){ctx.globalAlpha=1;ctx.strokeStyle=style.stroke;ctx.lineWidth=style.width||1.2;ctx.stroke();}if(style.hoverable!==false&&x>=box.l-12&&x<=box.l+box.w+12&&y>=box.t-12&&y<=box.t+box.h+12){const key=`${Math.floor(x/cell)}:${Math.floor(y/cell)}`;const bucket=grid.get(key)||[];bucket.push({d,x,y});grid.set(key,bucket);}});if(yKey==='accuracy'){const targetY=sy(85);if(targetY>=box.t&&targetY<=box.t+box.h){ctx.save();ctx.strokeStyle='#c3473c';ctx.fillStyle='#a9362f';ctx.lineWidth=1.7;ctx.globalAlpha=.9;ctx.setLineDash([7,5]);ctx.beginPath();ctx.moveTo(box.l,targetY);ctx.lineTo(box.l+box.w,targetY);ctx.stroke();ctx.setLineDash([]);ctx.font='12px Segoe UI';ctx.textAlign='right';ctx.fillText('Accuracy = 85%',box.l+box.w-8,targetY-6);ctx.restore();}}ctx.globalAlpha=1;ctx.restore();state.grid=grid;
  if(statusId)document.getElementById(statusId).textContent=statusText||`${rows.length.toLocaleString()} candidates shown`;
  const showTooltip=e=>{const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,cx=Math.floor(mx/cell),cy=Math.floor(my/cell);let best=null,bestDist=144;for(let gx=cx-1;gx<=cx+1;gx++)for(let gy=cy-1;gy<=cy+1;gy++)for(const p of state.grid.get(`${gx}:${gy}`)||[]){const dist=(p.x-mx)**2+(p.y-my)**2;if(dist<bestDist){best=p;bestDist=dist;}}if(!best){tip.style.display='none';return;}tip.innerHTML=tooltipText(best.d);tip.style.display='block';tip.style.left=Math.min(mx+14,wrap.clientWidth-295)+'px';tip.style.top=Math.max(5,my-12)+'px';};
  canvas.onmousemove=e=>{if(state.drag)return;state.hoverEvent=e;if(state.hoverFrame)return;state.hoverFrame=requestAnimationFrame(()=>{state.hoverFrame=0;showTooltip(state.hoverEvent);});};canvas.onmouseleave=()=>tip.style.display='none';canvas.onwheel=e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),fx=(e.clientX-r.left-box.l)/box.w,fy=(e.clientY-r.top-box.t)/box.h,scale=e.deltaY<0?.82:1.22,xCenter=xd[0]+(xd[1]-xd[0])*fx,yCenter=yd[1]-(yd[1]-yd[0])*fy,xSpan=(xd[1]-xd[0])*scale,ySpan=(yd[1]-yd[0])*scale,zoomX=!e.altKey,zoomY=!e.shiftKey;if(zoomX)state.xDomain=[xCenter-xSpan*fx,xCenter+xSpan*(1-fx)];if(zoomY)state.yDomain=[yCenter-ySpan*(1-fy),yCenter+ySpan*fy];redrawSoon(canvas);};canvas.onpointerdown=e=>{state.drag={x:e.clientX,y:e.clientY,xd:[...xd],yd:[...yd]};canvas.setPointerCapture(e.pointerId);tip.style.display='none';};canvas.onpointermove=e=>{const drag=state.drag;if(!drag)return;const dx=(e.clientX-drag.x)/box.w*(drag.xd[1]-drag.xd[0]),dy=(e.clientY-drag.y)/box.h*(drag.yd[1]-drag.yd[0]);state.xDomain=[drag.xd[0]-dx,drag.xd[1]-dx];state.yDomain=[drag.yd[0]+dy,drag.yd[1]+dy];redrawSoon(canvas);};canvas.onpointerup=canvas.onpointercancel=()=>{state.drag=null;};
}
function drawParetoLines(ctx,sx,sy){const series=[{color:'#202735',width:2.8,dash:[7,4],rows:DATA.filter(d=>d.globalPareto)},...Object.keys(PATTERN_COLORS).map(pattern=>({color:PATTERN_COLORS[pattern],width:1.8,dash:[],rows:DATA.filter(d=>d.pattern===pattern&&d.patternPareto)}))];for(const line of series){if(line.rows.length<2)continue;const frontier=[...line.rows].sort((a,b)=>a.energy-b.energy);ctx.save();ctx.strokeStyle=line.color;ctx.lineWidth=line.width;ctx.globalAlpha=.88;ctx.setLineDash(line.dash);ctx.beginPath();frontier.forEach((d,index)=>{const x=sx(d.energy),y=sy(d.accuracy);index?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.restore();}}
function drawPareto(){ setupChart(document.getElementById('pareto-canvas'),DATA,'energy','accuracy',d=>{ if(d.globalPareto)return {fill:'#ffffff',stroke:'#222',radius:4.3,width:2,alpha:1}; if(d.patternPareto)return {fill:PATTERN_COLORS[d.pattern],radius:3.4,alpha:.95}; return {fill:'#8e99a9',radius:1.25,alpha:.18}; },undefined,undefined,drawParetoLines); }
function options(){ return ['accuracy','energy','latency'].map(key=>`<option value="${key}">${FIELDS[key].label}</option>`).join(''); }
function makeFilter(){ const row=document.createElement('div');row.className='filter-row';row.innerHTML=`<select class="filter-field">${Object.entries(FILTERS).map(([k,f])=>`<option value="${k}">${f.label}</option>`).join('')}</select><select class="filter-op"></select><span class="filter-value"></span><button class="remove" type="button">Remove</button>`; row.querySelector('.filter-field').onchange=()=>refreshFilter(row);row.querySelector('.filter-op').onchange=drawCustom;row.querySelector('.remove').onclick=()=>{row.remove();drawCustom();};document.getElementById('filter-list').append(row);refreshFilter(row); }
function refreshFilter(row){const spec=FILTERS[row.querySelector('.filter-field').value],op=row.querySelector('.filter-op'),value=row.querySelector('.filter-value');op.innerHTML=spec.type==='category'?'<option value="eq">=</option>':'<option value="eq">=</option><option value="gte">≥</option><option value="lte">≤</option>'; value.innerHTML=spec.type==='category'?`<select>${spec.values.map(v=>`<option value="${v}">${v}</option>`).join('')}</select>`:'<input type="number" step="any" placeholder="value">';value.querySelector('select,input').oninput=drawCustom;drawCustom();}
function paretoFrontier(rows,xKey,yKey){const xMax=xKey==='accuracy',yMax=yKey==='accuracy',xValue=d=>FIELDS[xKey].value(d),yValue=d=>FIELDS[yKey].value(d),ordered=[...rows].sort((a,b)=>(xMax?-1:1)*(xValue(a)-xValue(b)));const frontier=[];let bestY=yMax?-Infinity:Infinity;for(let i=0;i<ordered.length;){let j=i,best=ordered[i];while(j<ordered.length&&xValue(ordered[j])===xValue(ordered[i])){if((yMax&&yValue(ordered[j])>yValue(best))||(!yMax&&yValue(ordered[j])<yValue(best)))best=ordered[j];j++;}const candidateY=yValue(best),improves=yMax?candidateY>bestY:candidateY<bestY;if(improves){frontier.push(best);bestY=candidateY;}i=j;}return frontier.sort((a,b)=>xValue(a)-xValue(b));}
function drawCustom(){const clauses=[...document.querySelectorAll('.filter-row')].map(row=>({key:row.querySelector('.filter-field').value,op:row.querySelector('.filter-op').value,value:row.querySelector('.filter-value select,.filter-value input').value}));const selected=DATA.filter(d=>clauses.every(c=>{const v=d[c.key],target=FILTERS[c.key].type==='number'?Number(c.value):c.value;if(c.value==='')return true;return c.op==='eq'?v===target:c.op==='gte'?v>=target:v<=target;}));const selectedIds=new Set(selected.map(d=>d.id)),hasConstraints=clauses.length>0,x=document.getElementById('x-axis').value,y=document.getElementById('y-axis').value,frontier=paretoFrontier(selected,x,y),frontierIds=new Set(frontier.map(d=>d.id)),meanX=selected.length?selected.reduce((sum,d)=>sum+FIELDS[x].value(d),0)/selected.length:NaN,meanY=selected.length?selected.reduce((sum,d)=>sum+FIELDS[y].value(d),0)/selected.length:NaN,status=selected.length?`${selected.length.toLocaleString()} selected / ${DATA.length.toLocaleString()} total · Mean X: ${fmt(x,meanX)} · Mean Y: ${fmt(y,meanY)} · Pareto: ${frontier.length}`:`0 selected / ${DATA.length.toLocaleString()} total`;
  setupChart(document.getElementById('custom-canvas'),DATA,x,y,d=>hasConstraints&&!selectedIds.has(d.id)?({fill:'#8e99a9',radius:1.2,alpha:.16,hoverable:false}):frontierIds.has(d.id)?({fill:'#ffffff',stroke:'#202735',radius:4.1,width:1.8,alpha:1}):({fill:PATTERN_COLORS[d.pattern],radius:2.2,alpha:.62}),'custom-status',status,(ctx,sx,sy)=>{if(frontier.length<2)return;ctx.save();ctx.strokeStyle='#202735';ctx.lineWidth=2.4;ctx.globalAlpha=.9;ctx.setLineDash([5,3]);ctx.beginPath();frontier.forEach((d,index)=>{const px=sx(FIELDS[x].value(d)),py=sy(FIELDS[y].value(d));index?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();ctx.restore();});}
function exportCanvas(canvas,filename){const base=(filename||'chart').trim().replace(/[\\/:*?"<>|]+/g,'_').replace(/\.png$/i,'')||'chart';canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`${base}.png`;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),0);},'image/png');}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active');if(b.dataset.tab==='pareto')setTimeout(drawPareto,0);else setTimeout(drawCustom,0);});
for(const id of ['x-axis','y-axis']){document.getElementById(id).innerHTML=options();document.getElementById(id).onchange=drawCustom;}document.getElementById('x-axis').value='energy';document.getElementById('y-axis').value='accuracy';document.getElementById('add-filter').onclick=makeFilter;document.querySelectorAll('[data-reset]').forEach(button=>button.onclick=()=>{CHART_VIEWS.delete(document.getElementById(button.dataset.reset));button.dataset.reset==='pareto-canvas'?drawPareto():drawCustom();});document.querySelectorAll('[data-export]').forEach(button=>button.onclick=()=>exportCanvas(document.getElementById(button.dataset.export),document.getElementById(button.dataset.nameInput).value||button.dataset.filename));new ResizeObserver(()=>{if(!DATA.length)return;if(document.getElementById('pareto').classList.contains('active'))drawPareto();else drawCustom();}).observe(document.querySelector('main'));
async function loadData(){try{const response=await fetch('/api/data');if(!response.ok)throw new Error(`HTTP ${response.status}`);DATA=await response.json();drawPareto();}catch(error){document.querySelector('#pareto .note').textContent='Run serve_pareto_explorer.py, then open http://127.0.0.1:8000/';console.error(error);}}loadData();
</script>
</body></html>'''
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(template, encoding="utf-8")
    print(f"Generated Accuracy-Energy explorer: {output_html}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("measurements/search/candidate_predictions.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("measurements/search/next_measurement_candidates.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("measurements/search/selection_summary.json"))
    parser.add_argument("--output-html", type=Path, default=Path("measurements/search/pareto_structures.html"))
    parser.add_argument("--accuracy-pareto-count", type=int, default=20)
    parser.add_argument("--high-accuracy-count", type=int, default=10)
    parser.add_argument("--low-energy-count", type=int, default=10)
    parser.add_argument("--uncertainty-count", type=int, default=0, help="Deprecated/ignored: uncertainty is not used")
    parser.add_argument("--diversity-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.accuracy_pareto_count, args.high_accuracy_count, args.low_energy_count, args.diversity_count) < 0:
        raise ValueError("Selection counts must be non-negative.")
    for path in (args.output_csv, args.summary_json, args.output_html):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}. Use --overwrite to replace it.")

    frame = pd.read_csv(args.predictions, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # MEAN VALUES ONLY. No std / lower / upper is used anywhere below.
    # Prefer ensemble means when they exist; otherwise fall back to the
    # existing point-prediction columns used by the original pipeline.
    # ------------------------------------------------------------------
    acc_col = choose_mean_column(
        frame,
        (
            "ensemble_accuracy_mean_percent",
            "ensemble_accuracy_mean",
            "predicted_accuracy_percent",
        ),
        "accuracy mean",
    )
    energy_col = choose_mean_column(
        frame,
        ("ensemble_energy_mean_j", "predicted_energy_j"),
        "energy mean",
    )
    latency_col = choose_mean_column(
        frame,
        ("ensemble_latency_mean_ms", "predicted_latency_ms"),
        "latency mean",
    )

    acc_mean = pd.to_numeric(frame[acc_col], errors="raise").to_numpy(float)
    energy_mean = pd.to_numeric(frame[energy_col], errors="raise").to_numpy(float)
    latency_mean = pd.to_numeric(frame[latency_col], errors="raise").to_numpy(float)
    param_count = pd.to_numeric(frame["parameter_count"], errors="raise").to_numpy(float)

    # Internal columns used consistently by selection + HTML output.
    frame["_mean_accuracy_percent"] = acc_mean
    frame["_mean_energy_j"] = energy_mean
    frame["_mean_latency_ms"] = latency_mean

    # Mean-value Pareto masks.
    # Accuracy: larger is better. Energy/latency: smaller is better.
    frame["is_mean_accuracy_energy_pareto"] = mean_pareto_mask(acc_mean, energy_mean)
    frame["is_mean_accuracy_latency_pareto"] = mean_pareto_mask(acc_mean, latency_mean)
    frame["is_mean_capacity_energy_pareto"] = mean_pareto_mask(param_count, energy_mean)

    # Mean-only balanced score used only for fallback/diversity ranking.
    # It is NOT the Pareto definition.
    acc_scale = np.median(np.abs(acc_mean)) or 1.0
    energy_scale = np.median(np.abs(energy_mean)) or 1.0
    frame["accuracy_energy_score"] = acc_mean / acc_scale - energy_mean / energy_scale

    selected: list[int] = []
    reasons: dict[int, str] = {}
    all_indices = frame.index.to_numpy()

    # 1. Mean Accuracy-Energy Pareto
    acc_mask = frame["is_mean_accuracy_energy_pareto"].to_numpy(bool)
    acc_pareto_df = frame.loc[acc_mask].copy()
    acc_pareto_df["_mean_sort_accuracy"] = acc_mean[acc_mask]
    acc_pareto_df = acc_pareto_df.sort_values("_mean_sort_accuracy", ascending=False)
    if len(acc_pareto_df):
        positions = np.linspace(
            0,
            len(acc_pareto_df) - 1,
            min(args.accuracy_pareto_count, len(acc_pareto_df)),
        ).round().astype(int)
        add_ranked(
            selected,
            reasons,
            acc_pareto_df.iloc[np.unique(positions)].index.to_numpy(),
            args.accuracy_pareto_count,
            "mean_acc_energy_pareto",
        )

    # 2. High mean Accuracy
    add_ranked(
        selected,
        reasons,
        all_indices[np.argsort(-acc_mean)],
        args.high_accuracy_count,
        "high_mean_accuracy",
    )

    # 3. Low mean Energy
    add_ranked(
        selected,
        reasons,
        all_indices[np.argsort(energy_mean)],
        args.low_energy_count,
        "low_mean_energy",
    )

    # 4. Structural diversity (still mean-only; no uncertainty involved)
    add_diverse(frame, selected, reasons, args.diversity_count, energy_mean)

    # 5. Fill using mean-only balanced score
    desired = (
        args.accuracy_pareto_count
        + args.high_accuracy_count
        + args.low_energy_count
        + args.diversity_count
    )
    if len(selected) < desired:
        add_ranked(
            selected,
            reasons,
            all_indices[np.argsort(-frame["accuracy_energy_score"].to_numpy())],
            desired - len(selected),
            "mean_balanced_fill",
        )

    output = frame.loc[selected].copy()
    output.insert(0, "selection_rank", range(1, len(output) + 1))
    output.insert(1, "selection_reason", [reasons[i] for i in selected])

    # Hide internal helper columns from exported CSV.
    output = output.drop(
        columns=["_mean_accuracy_percent", "_mean_energy_j", "_mean_latency_ms", "_mean_sort_accuracy"],
        errors="ignore",
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")

    summary = {
        "candidate_count": len(frame),
        "selected_count": len(output),
        "pareto_basis": "mean_only",
        "uncertainty_used": False,
        "accuracy_mean_column": acc_col,
        "energy_mean_column": energy_col,
        "latency_mean_column": latency_col,
        "mean_acc_energy_pareto_count": int(frame["is_mean_accuracy_energy_pareto"].sum()),
        "mean_accuracy_latency_pareto_count": int(frame["is_mean_accuracy_latency_pareto"].sum()),
        "mean_capacity_energy_pareto_count": int(frame["is_mean_capacity_energy_pareto"].sum()),
        "selection_reason_counts": output["selection_reason"].value_counts().to_dict(),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    generate_interactive_explorer(frame, args.output_html)

    print(f"Using mean columns: accuracy={acc_col}, energy={energy_col}, latency={latency_col}")
    print("Uncertainty: NOT USED")
    print(f"\nWrote {args.output_csv} ({len(output)} selected from {len(frame):,}).")
    print(f"  Mean Acc-Energy Pareto : {summary['mean_acc_energy_pareto_count']}")
    print(f"  Mean Acc-Latency Pareto: {summary['mean_accuracy_latency_pareto_count']}")
    print(f"  Mean Cap-Energy Pareto : {summary['mean_capacity_energy_pareto_count']}")
    print(f"  Selection reasons: {summary['selection_reason_counts']}")


if __name__ == "__main__":
    main()
