"""
build_graph_full.py
-------------------
Builds a full 3-level causal graph for the HAI Digital Twin.

Improvements over build_graph.py:
  - Includes phy_boiler.json (physical plant graph) for L1/L2 edges
  - Estimates lags per-pair using Time-Delay Mutual Information (TDMI)
    from the training CSVs instead of hardcoding lag=1

Levels:
  L0 : sensor_reading  → control_output    (DCS BFS through PLC blocks)
  L1 : control_output  → sensor_reading    (phy_boiler, 1 hop)
  L2 : control_output  → element → sensor  (phy_boiler, 2 hops)

Outputs (in outputs/causal_graph/):
  parents_full.json   — backward-compatible with causal_loss.py
  edges_full.csv
  summary_full.txt

Run:
  python 01_causal_graph/build_graph_full.py
"""

from __future__ import annotations
import ast, csv, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BOILER_DIR = Path("C:/Users/ahmma/Desktop/farah/boiler")
DATA_DIR   = Path("data/processed")
OUT_DIR    = Path("outputs/causal_graph")

TRAIN_CSVS  = [DATA_DIR / f"train{i}.csv" for i in range(1, 5)]
TDMI_SAMPLE = 50_000   # rows to sample for TDMI
LAG_MAX     = 30       # max lag in seconds
N_BINS      = 10       # histogram bins for MI estimation
MIN_STD     = 1e-4     # below this → treat column as constant, skip TDMI

# ── Dynamics → lag search range ───────────────────────────────────────────────
DYNAMICS_LAG_RANGE: dict[int, tuple[int, int]] = {
    1: (1,  5),   # fast: valves, pumps
    2: (5, 15),   # medium: level, pressure
    3: (15, 30),  # slow: temperature
}

# ── Sensor map (short DCS/PHY name → HAI column) ──────────────────────────────
# Only includes columns that exist in the 86-tag HAI 23.05 training CSVs.
# Corrections vs build_graph.py:
#   TIT02  → P1_TIT02  (not P1_TIT03 — build_graph.py has a bug)
#   TWIT03 → P1_TIT03  (not P3_TWIT03 — verified r=1.0 with training data)
#   PP04   → P1_PP04D  (not P4_PP04D  — verified r=0.9994)
# Removed (HAIEnd-only, not in HAI training CSVs):
#   LSL02  (P2_LSL02), LSH02 (P2_LSH02), LSH04 (P4_LSH04)
#   SOL02  (P2_SOL02D), SOL04 (P2_SOL04D)
SENSOR_MAP: dict[str, str] = {
    # Sensors
    "PIT01"  : "P1_PIT01",
    "PIT02"  : "P1_PIT02",
    "FT01"   : "P1_FT01",
    "FT02"   : "P1_FT02",
    "FT03"   : "P1_FT03",
    "LIT01"  : "P1_LIT01",
    "TIT01"  : "P1_TIT01",
    "TIT02"  : "P1_TIT02",    # corrected
    "TWIT03" : "P1_TIT03",    # corrected
    "LSH01"  : "P1_PIT01_HH",
    "LSL01"  : "P1_SOL01D",
    # Actuators
    "PCV01"  : "P1_PCV01D",
    "PCV02"  : "P1_PCV02D",
    "FCV01"  : "P1_FCV01D",
    "FCV02"  : "P1_FCV02D",
    "FCV03"  : "P1_FCV03D",
    "LCV01"  : "P1_LCV01D",
    "PP01"   : "P1_PP01AD",
    "PP01A"  : "P1_PP01AD",
    "PP01B"  : "P1_PP01BD",
    "PP02"   : "P1_PP02D",
    "PP04"   : "P1_PP04D",    # corrected
    "SOL01"  : "P1_SOL01D",
    "SOL03"  : "P1_SOL03D",
    "HT01"   : "P4_HT_PO",
}

# Reverse map: HAI column → short name (for phy_boiler lookup)
HAI_TO_SHORT: dict[str, str] = {}
for _short, _hai in SENSOR_MAP.items():
    if _hai not in HAI_TO_SHORT:
        HAI_TO_SHORT[_hai] = _short


# ── Section 1: DCS BFS (L0 edges) ─────────────────────────────────────────────

def _load_dcs_json(path: Path) -> dict:
    """Load a boiler DCS JSON (single-quoted Python dict) into a raw dict."""
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def _yellow_nodes(data: dict) -> set[str]:
    return {n["id"] for n in data["nodes"] if n.get("fillcolor") == "yellow"}


def _grey_nodes(data: dict) -> set[str]:
    return {n["id"] for n in data["nodes"] if n.get("fillcolor") == "lightgrey"}


def _adjacency(data: dict) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in data["links"]:
        adj[e["source"]].append(e["target"])
    return adj


def _bfs_reachable_yellow(src: str, adj: dict[str, list[str]],
                           yellow: set[str], grey: set[str]) -> set[str]:
    """BFS from src through PLC blocks; collect reachable yellow nodes."""
    visited: set[str] = {src}
    queue = [src]
    reached: set[str] = set()
    while queue:
        node = queue.pop(0)
        for nb in adj.get(node, []):
            if nb in visited:
                continue
            visited.add(nb)
            if nb in yellow:
                reached.add(nb)
            elif nb not in grey:
                queue.append(nb)
    return reached


def parse_dcs_graphs(
    boiler_dir: Path,
    sensor_map: dict[str, str],
) -> list[dict]:
    """
    Load all dcs_*.json, BFS through PLC blocks, return L0 edges.
    Each edge: {source_hai, target_hai, level, dynamics, via}
    """
    raw_edges: list[tuple[str, str, str]] = []  # (src_short, tgt_short, plc_stem)
    all_yellow: set[str] = set()

    for path in sorted(boiler_dir.glob("dcs_*.json")):
        data   = _load_dcs_json(path)
        yellow = _yellow_nodes(data)
        grey   = _grey_nodes(data)
        adj    = _adjacency(data)
        all_yellow.update(yellow)

        count = 0
        for src in yellow:
            for tgt in _bfs_reachable_yellow(src, adj, yellow, grey):
                if src != tgt:
                    raw_edges.append((src, tgt, path.stem))
                    count += 1
        print(f"  {path.stem:12s}: {len(yellow):3d} yellow | {count:3d} raw edges")

    print(f"  Total yellow nodes : {len(all_yellow)}")
    print(f"  Total raw L0 edges : {len(raw_edges)}")

    # Dedup by (source_short, target_short)
    seen: set[tuple[str, str]] = set()
    dedup: list[tuple[str, str, str]] = []
    for src, tgt, plc in raw_edges:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            dedup.append((src, tgt, plc))

    # Map to HAI columns
    edges: list[dict] = []
    for src, tgt, plc in dedup:
        src_hai = sensor_map.get(src)
        tgt_hai = sensor_map.get(tgt)
        if src_hai and tgt_hai and src_hai != tgt_hai:
            edges.append({
                "source_hai": src_hai,
                "target_hai": tgt_hai,
                "level"     : 0,
                "dynamics"  : 1,
                "via"       : plc,
            })

    # Final dedup by HAI pair
    final: list[dict] = []
    seen_hai: set[tuple[str, str]] = set()
    for e in edges:
        key = (e["source_hai"], e["target_hai"])
        if key not in seen_hai:
            seen_hai.add(key)
            final.append(e)

    print(f"  L0 edges (mapped)  : {len(final)}")
    return final


# ── Section 2: Physical graph ─────────────────────────────────────────────────

def parse_phy_graph(
    phy_path: Path,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, str]]:
    """
    Load phy_boiler.json.
    Returns:
      adj       : short_name → [(neighbor_short, dynamics_int), ...]
      node_role : short_name → "actuator" | "sensor" | "element"
    """
    data = json.loads(phy_path.read_text(encoding="utf-8"))

    node_role: dict[str, str] = {}
    for n in data["nodes"]:
        nid  = n["id"]
        in_t = str(n.get("in_tags",  "") or "").strip()
        out_t= str(n.get("out_tags", "") or "").strip()
        if in_t:
            node_role[nid] = "actuator"
        elif out_t:
            node_role[nid] = "sensor"
        else:
            node_role[nid] = "element"

    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for e in data["links"]:
        dyn = int(e.get("dynamics", 1)) if str(e.get("dynamics", "")).isdigit() else 1
        adj[e["source"]].append((e["target"], dyn))
        # phy_boiler is directed but physical causality is often bidirectional
        # (a sensor reading is caused by an actuator upstream); we keep direction
        # as-is since the JSON is already a directed causal graph

    print(f"  PHY nodes: {len(node_role)} "
          f"(actuators: {sum(1 for r in node_role.values() if r=='actuator')}, "
          f"sensors: {sum(1 for r in node_role.values() if r=='sensor')}, "
          f"elements: {sum(1 for r in node_role.values() if r=='element')})")
    print(f"  PHY edges: {sum(len(v) for v in adj.values())}")

    return dict(adj), node_role


# ── Section 3: Physical chains (L1 + L2) ─────────────────────────────────────

def build_physical_chains(
    l0_edges  : list[dict],
    phy_adj   : dict[str, list[tuple[str, int]]],
    sensor_map: dict[str, str],
    hai_to_short: dict[str, str],
) -> list[dict]:
    """
    Extend causal graph with physical propagation.

    L1: L0_actuator → direct phy_boiler neighbor sensor
    L2: L0_actuator → phy_boiler element → phy_boiler sensor (2 hops)
    """
    # Build reverse sensor_map (HAI → short) for all entries (not just first)
    short_to_hai = sensor_map  # alias for clarity

    # Collect unique L0 targets (actuators whose physical effects we want to trace)
    l0_targets: set[str] = {e["target_hai"] for e in l0_edges}

    new_edges: list[dict] = []
    seen: set[tuple[str, str]] = {(e["source_hai"], e["target_hai"]) for e in l0_edges}

    for act_hai in l0_targets:
        act_short = hai_to_short.get(act_hai)
        if not act_short or act_short not in phy_adj:
            continue

        # L1: direct physical neighbors
        for nb1_short, dyn1 in phy_adj[act_short]:
            nb1_hai = short_to_hai.get(nb1_short)
            if nb1_hai and nb1_hai != act_hai:
                key = (act_hai, nb1_hai)
                if key not in seen:
                    seen.add(key)
                    new_edges.append({
                        "source_hai": act_hai,
                        "target_hai": nb1_hai,
                        "level"     : 1,
                        "dynamics"  : dyn1,
                        "via"       : "phy_boiler",
                    })

            # L2: through element nodes
            elif nb1_hai is None:
                # nb1 is an element (tank, HEX, etc.) — go one more hop
                if nb1_short not in phy_adj:
                    continue
                for nb2_short, dyn2 in phy_adj[nb1_short]:
                    nb2_hai = short_to_hai.get(nb2_short)
                    if nb2_hai and nb2_hai != act_hai:
                        key = (act_hai, nb2_hai)
                        if key not in seen:
                            seen.add(key)
                            new_edges.append({
                                "source_hai": act_hai,
                                "target_hai": nb2_hai,
                                "level"     : 2,
                                "dynamics"  : max(dyn1, dyn2),
                                "via"       : f"phy_boiler:{nb1_short}",
                            })

    l1 = sum(1 for e in new_edges if e["level"] == 1)
    l2 = sum(1 for e in new_edges if e["level"] == 2)
    print(f"  L1 edges: {l1},  L2 edges: {l2}")
    return new_edges


# ── Section 4: TDMI lag estimation ───────────────────────────────────────────

def _load_columns(
    csv_paths  : list[Path],
    cols       : set[str],
    max_rows   : int,
    rng_seed   : int = 42,
) -> dict[str, np.ndarray]:
    """
    Stratified random sample across all training CSVs.

    Each file contributes max_rows // n_files rows sampled at a random
    offset within the file, so the sample spans the full temporal range
    of training (not just the first N rows). This reduces constant_fallback
    caused by stable setpoint periods at the start of each file.
    """
    rng = np.random.default_rng(rng_seed)
    n_files = len([p for p in csv_paths if p.exists()])
    rows_per_file = max(1, max_rows // max(n_files, 1))

    data: dict[str, list[float]] = {c: [] for c in cols}
    missing: set[str] = set()

    for path in csv_paths:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        # Count file rows efficiently
        with open(path, encoding="utf-8") as f:
            n_total = sum(1 for _ in f) - 1  # subtract header

        if n_total <= 0:
            continue

        # Random contiguous window — preserves time-series autocorrelation
        # while sampling from a different part of each file
        start = int(rng.integers(0, max(1, n_total - rows_per_file)))
        end   = start + rows_per_file

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i < start:
                    continue
                if i >= end:
                    break
                for c in cols:
                    if c in missing:
                        continue
                    v = row.get(c)
                    if v is None:
                        missing.add(c)
                        continue
                    try:
                        data[c].append(float(v))
                    except ValueError:
                        data[c].append(float("nan"))

    if missing:
        print(f"  WARNING: columns not found in CSVs: {sorted(missing)}")

    return {c: np.array(v, dtype=np.float64) for c, v in data.items() if v}


def _tdmi(x: np.ndarray, y: np.ndarray, lag: int, n_bins: int = 10) -> float:
    """
    Time-Delay Mutual Information between x[:-lag] and y[lag:].
    Uses histogram-based density estimation (pure numpy).
    Returns MI in nats, 0.0 on degenerate input.
    """
    if lag <= 0 or lag >= len(x):
        return 0.0
    xp = x[:-lag]
    yp = y[lag:]
    mask = ~(np.isnan(xp) | np.isnan(yp))
    n_valid = int(mask.sum())
    if n_valid < 2 * n_bins:
        return 0.0
    xv = xp[mask]
    yv = yp[mask]

    if xv.max() == xv.min() or yv.max() == yv.min():
        return 0.0

    x_edges = np.linspace(xv.min(), xv.max() + 1e-10, n_bins + 1)
    y_edges = np.linspace(yv.min(), yv.max() + 1e-10, n_bins + 1)

    xb = np.clip(np.digitize(xv, x_edges) - 1, 0, n_bins - 1)
    yb = np.clip(np.digitize(yv, y_edges) - 1, 0, n_bins - 1)

    joint = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(joint, (xb, yb), 1)
    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    pos = joint > 0
    mi = float(np.sum(joint[pos] * np.log(joint[pos] / (px * py)[pos])))
    return max(mi, 0.0)


def _best_lag(
    x       : np.ndarray,
    y       : np.ndarray,
    dynamics: int,
    n_bins  : int,
    min_std : float,
) -> tuple[int, str]:
    """Return (best_lag, lag_method)."""
    lo, hi = DYNAMICS_LAG_RANGE.get(dynamics, (1, LAG_MAX))
    midpoint = (lo + hi) // 2

    if np.nanstd(x) < min_std or np.nanstd(y) < min_std:
        return midpoint, "constant_fallback"

    lags   = list(range(lo, hi + 1))
    mi_vals = [_tdmi(x, y, lag, n_bins) for lag in lags]
    best_mi = max(mi_vals)

    if best_mi < 1e-5:
        return midpoint, "low_mi_fallback"

    return lags[int(np.argmax(mi_vals))], "tdmi"


def compute_tdmi_lags(
    all_edges  : list[dict],
    train_csvs : list[Path],
    sample_rows: int = TDMI_SAMPLE,
    n_bins     : int = N_BINS,
    min_std    : float = MIN_STD,
) -> list[dict]:
    """
    Estimate per-pair lag using TDMI. Mutates and returns all_edges with
    'lag' and 'lag_method' fields added.
    """
    # Collect all unique column names
    needed: set[str] = set()
    for e in all_edges:
        needed.add(e["source_hai"])
        needed.add(e["target_hai"])

    print(f"  Loading {sample_rows} rows for {len(needed)} unique columns ...")
    col_data = _load_columns(train_csvs, needed, sample_rows)
    print(f"  Columns loaded: {len(col_data)}")

    n_tdmi = n_const = n_lowmi = 0
    for e in all_edges:
        src = e["source_hai"]
        tgt = e["target_hai"]

        x = col_data.get(src)
        y = col_data.get(tgt)

        if x is None or y is None or len(x) == 0 or len(y) == 0:
            lo, hi = DYNAMICS_LAG_RANGE.get(e["dynamics"], (1, LAG_MAX))
            e["lag"]        = (lo + hi) // 2
            e["lag_method"] = "missing_column"
            n_const += 1
            continue

        lag, method = _best_lag(x, y, e["dynamics"], n_bins, min_std)
        e["lag"]        = lag
        e["lag_method"] = method

        if method == "tdmi":
            n_tdmi += 1
        elif method == "constant_fallback":
            n_const += 1
        else:
            n_lowmi += 1

    print(f"  Lag methods: tdmi={n_tdmi}, constant_fallback={n_const}, "
          f"low_mi_fallback={n_lowmi}")
    return all_edges


# ── Section 5: Write outputs ──────────────────────────────────────────────────

def write_outputs(
    all_edges: list[dict],
    out_dir  : Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── parents_full.json ─────────────────────────────────────────────────────
    parents: dict[str, list] = {}
    for e in all_edges:
        tgt = e["target_hai"]
        parents.setdefault(tgt, []).append({
            "parent"    : e["source_hai"],
            "lag"       : e["lag"],
            "level"     : e["level"],
            "dynamics"  : e["dynamics"],
            "via"       : e["via"],
            "lag_method": e["lag_method"],
        })

    with open(out_dir / "parents_full.json", "w") as f:
        json.dump(parents, f, indent=2)

    # ── edges_full.csv ────────────────────────────────────────────────────────
    fieldnames = ["source_hai", "target_hai", "level", "dynamics",
                  "lag", "lag_method", "via"]
    sorted_edges = sorted(all_edges, key=lambda e: (e["level"], e["source_hai"]))
    with open(out_dir / "edges_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted_edges)

    # ── summary_full.txt ──────────────────────────────────────────────────────
    l0 = [e for e in all_edges if e["level"] == 0]
    l1 = [e for e in all_edges if e["level"] == 1]
    l2 = [e for e in all_edges if e["level"] == 2]

    methods = defaultdict(int)
    for e in all_edges:
        methods[e.get("lag_method", "unknown")] += 1

    lines = [
        "=== HAI Digital Twin — Full Causal Graph ===",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Edge counts:",
        f"  L0 (DCS logic, sensor→actuator)     : {len(l0)}",
        f"  L1 (PHY direct, actuator→sensor)    : {len(l1)}",
        f"  L2 (PHY 2-hop, actuator→elem→sensor): {len(l2)}",
        f"  Total                                : {len(all_edges)}",
        "",
        "Lag estimation breakdown:",
    ]
    for method, cnt in sorted(methods.items()):
        lines.append(f"  {method:25s}: {cnt}")

    lines += ["", "=== Causal Parents per HAI Sensor ==="]
    for sensor, plist in sorted(parents.items()):
        for p in plist:
            lines.append(
                f"  {sensor:20s} <- {p['parent']:20s} "
                f"[L{p['level']}, dyn={p['dynamics']}, lag={p['lag']:2d}s, {p['lag_method']}]"
            )

    (out_dir / "summary_full.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nOutputs saved to {out_dir}/")
    print(f"  parents_full.json  ({len(parents)} target sensors, {len(all_edges)} edges)")
    print(f"  edges_full.csv")
    print(f"  summary_full.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("HAI Digital Twin — Full Causal Graph Extraction")
    print("=" * 60)

    # 1. L0 edges from DCS
    print("\n[1/5] Parsing DCS graphs (L0)...")
    l0_edges = parse_dcs_graphs(BOILER_DIR, SENSOR_MAP)

    # 2. Physical graph
    print("\n[2/5] Loading physical plant graph...")
    phy_adj, node_roles = parse_phy_graph(BOILER_DIR / "phy_boiler.json")

    # 3. L1 + L2 edges from physical graph
    print("\n[3/5] Building physical causal chains (L1, L2)...")
    l12_edges = build_physical_chains(l0_edges, phy_adj, SENSOR_MAP, HAI_TO_SHORT)

    all_edges = l0_edges + l12_edges
    print(f"  Total edges before lag estimation: {len(all_edges)}")

    # 4. TDMI lag estimation
    print(f"\n[4/5] Estimating lags via TDMI ({len(all_edges)} pairs)...")
    all_edges = compute_tdmi_lags(all_edges, TRAIN_CSVS)

    # 5. Write outputs
    print("\n[5/5] Writing outputs...")
    write_outputs(all_edges, OUT_DIR)

    print("\nDone.")
    print("\nTo use with causal_loss.py, update the path in train.py:")
    print('  causal = CausalLoss("outputs/causal_graph/parents_full.json", sensor_cols)')


if __name__ == "__main__":
    main()
