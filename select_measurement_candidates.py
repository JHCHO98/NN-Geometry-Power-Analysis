"""Select Strict Interval Pareto-optimal CNN candidates for measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def strict_pareto_mask(
    maximize_lcb: np.ndarray,
    maximize_ucb: np.ndarray,
    minimize_lcb: np.ndarray,
    minimize_ucb: np.ndarray,
) -> np.ndarray:
    """Strict Interval Pareto Dominance.

    B is dominated iff there exists A such that:
        A.maximize_LCB >= B.maximize_UCB  (A의 최악 >= B의 최선 : 무조건 A가 높음)
        AND
        A.minimize_UCB <= B.minimize_LCB  (A의 최악 <= B의 최선 : 무조건 A가 낮음)

    Both conditions must hold simultaneously (AND).
    If only one holds, the intervals overlap in that dimension → cannot dominate.
    """
    n = len(maximize_lcb)
    is_dominated = np.zeros(n, dtype=bool)

    for b in range(n):
        b_max_ucb = maximize_ucb[b]
        b_min_lcb = minimize_lcb[b]
        for a in range(n):
            if a == b:
                continue
            # A strictly dominates B if BOTH conditions hold
            if maximize_lcb[a] >= b_max_ucb and minimize_ucb[a] <= b_min_lcb:
                is_dominated[b] = True
                break

    return ~is_dominated


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


def add_diverse(frame: pd.DataFrame, selected: list[int], reasons: dict[int, str], count: int) -> None:
    eligible = np.flatnonzero(
        (frame["energy_upper_j"] <= frame["energy_upper_j"].quantile(0.75))
        & (frame["parameter_count"] >= frame["parameter_count"].quantile(0.25))
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
        position = int(np.argmin(frame.iloc[eligible]["accuracy_energy_score"].to_numpy())) if np.isinf(distances).all() else int(np.argmax(distances))
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
        is_acc_energy = bool(row.get("is_strict_accuracy_energy_pareto", False))
        is_acc_latency = bool(row.get("is_strict_accuracy_latency_pareto", False))
        is_cap_energy = bool(row.get("is_strict_capacity_energy_pareto", False))
        is_selected = idx in selected_set

        if not (is_acc_energy or is_acc_latency or is_cap_energy or is_selected):
            continue

        energy_j = float(row.get("predicted_energy_j", row.get("energy_upper_j", 0.0)))
        latency_ms = float(row.get("predicted_latency_ms", row.get("latency_upper_ms", 0.0)))
        acc_pct = float(row.get("predicted_accuracy_percent", 0.0))
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
            "accuracy": round(acc_pct, 2),
            "energy_mj": round(energy_j * 1000, 2),
            "latency_ms": round(latency_ms, 2),
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
  <title>Strict Interval Pareto CNN Explorer</title>
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
      <h1>🛡️ Strict Interval Pareto CNN Explorer</h1>
      <p>
        지배 조건: <strong>A.Acc_LCB ≥ B.Acc_UCB AND A.Energy_UCB ≤ B.Energy_LCB</strong><br>
        A의 비관적 정확도(하한)가 B의 낙관적 정확도(상한)보다 높고, A의 비관적 에너지(상한)가 B의 낙관적 에너지(하한)보다 낮아야 B를 엄격하게 지배합니다.
      </p>
    </header>

    <nav class="tabs-nav">
      <button class="tab-btn active" id="tab-acc-energy" onclick="setTab('acc-energy')">
        🎯 Strict Acc–Energy Pareto <span class="badge" id="badge-acc-energy">0</span>
      </button>
      <button class="tab-btn" id="tab-acc-latency" onclick="setTab('acc-latency')">
        ⚡ Strict Acc–Latency Pareto <span class="badge" id="badge-acc-latency">0</span>
      </button>
      <button class="tab-btn" id="tab-cap-energy" onclick="setTab('cap-energy')">
        📦 Strict Cap–Energy Pareto <span class="badge" id="badge-cap-energy">0</span>
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
      <span>Dominance: <span class="cond">A.Acc_LCB ≥ B.Acc_UCB AND A.E_UCB ≤ B.E_LCB</span></span>
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
        if (sv === "accuracy_desc") return b.accuracy - a.accuracy;
        if (sv === "energy_asc") return a.energy_mj - b.energy_mj;
        if (sv === "latency_asc") return a.latency_ms - b.latency_ms;
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
          <td class="num"><span class="acc-highlight">${{r.accuracy.toFixed(2)}}%</span></td>
          <td class="num"><span class="energy-highlight">${{r.energy_mj.toFixed(2)}} mJ</span></td>
          <td class="num"><span class="lat-highlight">${{r.latency_ms.toFixed(2)}} ms</span></td>
          <td><span class="reason-tag ${{pt}}">${{r.reason}}</span></td>
        </tr>`;
      }}).join("");
      document.getElementById("footer-status").textContent = `표시 중: ${{rows.length}}개 Strict Pareto 구조`;
    }}
    render();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")
    print(f"Generated Strict Interval Pareto viewer: {output_html}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("measurements/search/candidate_predictions.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("measurements/search/next_measurement_candidates.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("measurements/search/selection_summary.json"))
    parser.add_argument("--output-html", type=Path, default=Path("measurements/search/pareto_structures.html"))
    parser.add_argument("--accuracy-pareto-count", type=int, default=20)
    parser.add_argument("--high-accuracy-count", type=int, default=10)
    parser.add_argument("--low-energy-count", type=int, default=10)
    parser.add_argument("--uncertainty-count", type=int, default=10)
    parser.add_argument("--diversity-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.accuracy_pareto_count, args.high_accuracy_count, args.low_energy_count, args.uncertainty_count, args.diversity_count) < 0:
        raise ValueError("Selection counts must be non-negative.")
    for path in (args.output_csv, args.summary_json, args.output_html):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}. Use --overwrite to replace it.")

    frame = pd.read_csv(args.predictions, encoding="utf-8-sig")

    # Confidence bounds
    acc_lcb = frame["accuracy_lower_percent"].to_numpy() if "accuracy_lower_percent" in frame.columns else frame["predicted_accuracy_percent"].to_numpy()
    acc_ucb = frame["accuracy_upper_percent"].to_numpy() if "accuracy_upper_percent" in frame.columns else frame["predicted_accuracy_percent"].to_numpy()
    energy_lcb = frame["energy_lower_j"].to_numpy() if "energy_lower_j" in frame.columns else frame["predicted_energy_j"].to_numpy()
    energy_ucb = frame["energy_upper_j"].to_numpy() if "energy_upper_j" in frame.columns else frame["predicted_energy_j"].to_numpy()
    lat_lcb = frame["latency_lower_ms"].to_numpy() if "latency_lower_ms" in frame.columns else frame["predicted_latency_ms"].to_numpy()
    lat_ucb = frame["latency_upper_ms"].to_numpy() if "latency_upper_ms" in frame.columns else frame["predicted_latency_ms"].to_numpy()
    param_count = frame["parameter_count"].to_numpy()

    # Strict Interval Pareto masks
    frame["is_strict_accuracy_energy_pareto"] = strict_pareto_mask(acc_lcb, acc_ucb, energy_lcb, energy_ucb)
    frame["is_strict_accuracy_latency_pareto"] = strict_pareto_mask(acc_lcb, acc_ucb, lat_lcb, lat_ucb)
    # For capacity-energy: maximize param_count, minimize energy
    # Treat param_count as exact (no interval), compare against energy bounds
    frame["is_strict_capacity_energy_pareto"] = strict_pareto_mask(param_count, param_count, energy_lcb, energy_ucb)

    acc_scale = np.median(acc_lcb)
    energy_scale = np.median(energy_ucb)
    frame["accuracy_energy_score"] = acc_lcb / acc_scale - energy_ucb / energy_scale

    selected: list[int] = []
    reasons: dict[int, str] = {}
    all_indices = frame.index.to_numpy()

    # 1. Strict Accuracy-Energy Pareto
    acc_pareto_df = frame.loc[frame["is_strict_accuracy_energy_pareto"]].sort_values("accuracy_lower_percent", ascending=False)
    if len(acc_pareto_df):
        positions = np.linspace(0, len(acc_pareto_df) - 1, min(args.accuracy_pareto_count, len(acc_pareto_df))).round().astype(int)
        add_ranked(selected, reasons, acc_pareto_df.iloc[np.unique(positions)].index.to_numpy(), args.accuracy_pareto_count, "strict_acc_energy_pareto")

    # 2. High Accuracy
    add_ranked(selected, reasons, all_indices[np.argsort(-acc_lcb)], args.high_accuracy_count, "high_accuracy")

    # 3. Low Energy
    add_ranked(selected, reasons, all_indices[np.argsort(energy_ucb)], args.low_energy_count, "low_energy")

    # 4. Uncertainty
    if "combined_relative_uncertainty" in frame.columns:
        add_ranked(selected, reasons, all_indices[np.argsort(-frame["combined_relative_uncertainty"].to_numpy())], args.uncertainty_count, "high_uncertainty")

    # 5. Diversity
    add_diverse(frame, selected, reasons, args.diversity_count)

    # 6. Fill
    desired = args.accuracy_pareto_count + args.high_accuracy_count + args.low_energy_count + args.uncertainty_count + args.diversity_count
    if len(selected) < desired:
        add_ranked(selected, reasons, all_indices[np.argsort(-frame["accuracy_energy_score"].to_numpy())], desired - len(selected), "balanced_fill")

    output = frame.loc[selected].copy()
    output.insert(0, "selection_rank", range(1, len(output) + 1))
    output.insert(1, "selection_reason", [reasons[i] for i in selected])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")

    summary = {
        "candidate_count": len(frame),
        "selected_count": len(output),
        "strict_accuracy_energy_pareto_count": int(frame["is_strict_accuracy_energy_pareto"].sum()),
        "strict_accuracy_latency_pareto_count": int(frame["is_strict_accuracy_latency_pareto"].sum()),
        "strict_capacity_energy_pareto_count": int(frame["is_strict_capacity_energy_pareto"].sum()),
        "selection_reason_counts": output["selection_reason"].value_counts().to_dict(),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    generate_pareto_html(frame, selected, reasons, args.output_html)

    print(f"\nWrote {args.output_csv} ({len(output)} selected from {len(frame):,}).")
    print(f"  Strict Acc-Energy Pareto : {summary['strict_accuracy_energy_pareto_count']}")
    print(f"  Strict Acc-Latency Pareto: {summary['strict_accuracy_latency_pareto_count']}")
    print(f"  Strict Cap-Energy Pareto : {summary['strict_capacity_energy_pareto_count']}")
    print(f"  Selection reasons: {summary['selection_reason_counts']}")


if __name__ == "__main__":
    main()
