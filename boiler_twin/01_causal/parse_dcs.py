"""
parse_dcs.py — Parse boiler DCS + physical JSON files.

Outputs:
  LOOPS_DEF      : dict  — SP/PV/CV/CV_fb column names per loop
  ATTACK_TARGETS : dict  — yellow nodes (attack injection points) per loop
  CAUSAL_EDGES   : list  — (source_node, target_node) from phy_boiler physical graph
  CAUSAL_ADJ     : dict  — {pv_col: [cv_cols that causally affect it]}
"""

import ast
import json
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BOILER_DIR = Path(r"C:\Users\ahmma\Desktop\farah\boiler")

DCS_FILES = {
    "PC":    BOILER_DIR / "dcs_1001h.json",
    "LC":    BOILER_DIR / "dcs_1002h.json",
    "FC_TC": BOILER_DIR / "dcs_1003h.json",
    "HTR":   BOILER_DIR / "dcs_1004h.json",
    "EXT1":  BOILER_DIR / "dcs_1010h.json",
    "EXT2":  BOILER_DIR / "dcs_1011h.json",
    "CC":    BOILER_DIR / "dcs_1020h.json",
}
PHY_FILE = BOILER_DIR / "phy_boiler.json"

# ── Manual loop definitions (from DCS graph analysis + CSV column names) ──────
# Format: (sp, pv, cv, cv_fb)  — cv_fb=None if no feedback signal
LOOPS_DEF = {
    "PC":  {
        "sp":    "x1001_05_SETPOINT_OUT",
        "pv":    "P1_PIT01",
        "cv":    "P1_PCV01D",
        "cv_fb": "P1_PCV01Z",
    },
    "LC":  {
        "sp":    "x1002_07_SETPOINT_OUT",
        "pv":    "P1_LIT01",
        "cv":    "P1_LCV01D",
        "cv_fb": "P1_LCV01Z",
    },
    "FC":  {
        "sp":    "x1002_08_SETPOINT_OUT",
        "pv":    "P1_FT03Z",
        "cv":    "P1_FCV03D",
        "cv_fb": "P1_FCV03Z",
    },
    "TC":  {
        "sp":    "x1003_18_SETPOINT_OUT",
        "pv":    "P1_TIT01",
        "cv":    "P1_FCV01D",
        "cv_fb": "P1_FCV01Z",
    },
    "CC":  {
        "sp":    "P1_PP04SP",
        "pv":    "P1_TIT03",
        "cv":    "P1_PP04",
        "cv_fb": None,
    },
    # HTR — to be confirmed after inspecting dcs_1004h
    # "HTR": { "sp": "???", "pv": "???", "cv": "???", "cv_fb": None },
}

# All PVs and CVs
PV_COLS = [LOOPS_DEF[ln]["pv"] for ln in LOOPS_DEF]
CV_COLS = [LOOPS_DEF[ln]["cv"] for ln in LOOPS_DEF]

# Auxiliary plant inputs (not SP/PV/CV but physically relevant)
PLANT_AUX_COLS = [
    "P1_PP01AD", "P1_PP01BD", "P1_PP02D",
    "P1_FT01", "P1_FT01Z", "P1_FT02", "P1_FT02Z", "P1_FT03",
    "P1_PIT02", "P1_TIT02",
    "x1001_15_ASSIGN_OUT", "x1003_10_SETPOINT_OUT", "x1003_24_SUM_OUT",
    "GATEOPEN",
]


# ── Loader ────────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    """Load a file as standard JSON or Python-dict syntax depending on content."""
    with open(path, "r") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


# ── Attack targets ─────────────────────────────────────────────────────────────

def extract_attack_targets(loop_name: str, dcs_path: Path) -> list[str]:
    """
    Extract yellow-highlighted nodes from a DCS graph.
    These are the actuators/sensors that can be attack injection points.
    """
    data = _load(dcs_path)
    yellow_nodes = [
        node["id"]
        for node in data["nodes"]
        if node.get("fillcolor") == "yellow"
    ]
    return yellow_nodes


def build_attack_targets() -> dict[str, list[str]]:
    """Return {loop_name: [attack_target_nodes]} for all DCS files."""
    targets = {}
    for loop_name, path in DCS_FILES.items():
        if path.exists():
            targets[loop_name] = extract_attack_targets(loop_name, path)
        else:
            targets[loop_name] = []
    return targets


# ── Physical causal edges ─────────────────────────────────────────────────────

def extract_causal_edges() -> list[tuple[str, str]]:
    """
    Extract directed edges from phy_boiler.json.
    Returns list of (source_id, target_id) representing physical causality.
    """
    data = _load(PHY_FILE)
    edges = [
        (link["source"], link["target"])
        for link in data["links"]
    ]
    return edges


def build_causal_adjacency() -> dict[str, list[str]]:
    """
    Build {pv_col: [cv_cols]} mapping using node IDs from phy_boiler.
    """
    node_to_csv = {
        "PCV01": "P1_PCV01D",
        "PCV02": "P1_PCV02D",
        "LCV01": "P1_LCV01D",
        "FCV03": "P1_FCV03D",
        "FCV01": "P1_FCV01D",
        "FCV02": "P1_FCV02D",
        "PP01":  "P1_PP01AD",
        "PP02":  "P1_PP02D",
        "PP04":  "P1_PP04",
        "PIT01": "P1_PIT01",
        "PIT02": "P1_PIT02",
        "LIT01": "P1_LIT01",
        "FT01":  "P1_FT01",
        "FT03":  "P1_FT03Z",
        "TIT01": "P1_TIT01",
        "TIT02": "P1_TIT02",
        "TIT03": "P1_TIT03",
    }

    edges = extract_causal_edges()
    adjacency: dict[str, list[str]] = {pv: [] for pv in PV_COLS}

    for src_id, tgt_id in edges:
        src_csv = node_to_csv.get(src_id)
        tgt_csv = node_to_csv.get(tgt_id)

        # Skip if either side is not mapped to a CSV column
        if src_csv is None or tgt_csv is None:
            continue

        # Only consider edges where source is a CV and target is a PV
        if src_csv in CV_COLS and tgt_csv in PV_COLS:
            if src_csv not in adjacency[tgt_csv]:
                adjacency[tgt_csv].append(src_csv)

    return adjacency
    """
    Build {pv_col: [cv_cols]} mapping using node IDs from phy_boiler.

    Node ID → CSV column mapping (physical component → HAI signal name):
      PCV01 → P1_PCV01D    PIT01 → P1_PIT01
      LCV01 → P1_LCV01D    LIT01 → P1_LIT01
      FCV03 → P1_FCV03D    FT03  → P1_FT03Z
      FCV01 → P1_FCV01D    TIT01 → P1_TIT01
      PP04  → P1_PP04      TIT03 → P1_TIT03
    """
    node_to_csv = {
        "PCV01": "P1_PCV01D",
        "PCV02": "P1_PCV02D",
        "LCV01": "P1_LCV01D",
        "FCV03": "P1_FCV03D",
        "FCV01": "P1_FCV01D",
        "FCV02": "P1_FCV02D",
        "PP01":  "P1_PP01AD",
        "PP02":  "P1_PP02D",
        "PP04":  "P1_PP04",
        "PIT01": "P1_PIT01",
        "PIT02": "P1_PIT02",
        "LIT01": "P1_LIT01",
        "FT01":  "P1_FT01",
        "FT03":  "P1_FT03Z",
        "TIT01": "P1_TIT01",
        "TIT02": "P1_TIT02",
        "TIT03": "P1_TIT03",
    }

    edges = extract_causal_edges()
    adjacency: dict[str, list[str]] = {pv: [] for pv in PV_COLS}

    for src_id, tgt_id in edges:
        src_csv = node_to_csv.get(src_id)
        tgt_csv = node_to_csv.get(tgt_id)
        if src_csv in CV_COLS and tgt_csv in PV_COLS:
            if src_csv not in adjacency[tgt_csv]:
                adjacency[tgt_csv].append(src_csv)

    return adjacency


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LOOPS DEFINITION")
    print("=" * 60)
    for ln, defn in LOOPS_DEF.items():
        print(f"  {ln}: SP={defn['sp']}, PV={defn['pv']}, CV={defn['cv']}, CV_fb={defn['cv_fb']}")

    print("\n" + "=" * 60)
    print("ATTACK TARGETS (yellow nodes per DCS file)")
    print("=" * 60)
    attack_targets = build_attack_targets()
    for loop_name, nodes in attack_targets.items():
        print(f"  {loop_name}: {nodes}")

    print("\n" + "=" * 60)
    print("CAUSAL ADJACENCY (PV ← CVs from phy_boiler.json)")
    print("=" * 60)
    causal_adj = build_causal_adjacency()
    for pv, cvs in causal_adj.items():
        print(f"  {pv} ← {cvs if cvs else '(no direct CV edge found)'}")

    print("\n" + "=" * 60)
    print("PLANT INPUTS SUMMARY")
    print("=" * 60)
    print(f"  CVs  ({len(CV_COLS)}): {CV_COLS}")
    print(f"  AUX  ({len(PLANT_AUX_COLS)}): {PLANT_AUX_COLS}")
    print(f"  PVs  ({len(PV_COLS)}): {PV_COLS}")
