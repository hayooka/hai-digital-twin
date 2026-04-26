"""Select the best anomaly model from bench.json → outputs/classifier/winner.json.

Primary metric: AUROC. Tie-break: F1.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "classifier"

MODEL_PATHS = {"iforest": "iforest.joblib", "ocsvm": "ocsvm.joblib", "ae": "ae.pt"}


def main() -> int:
    bench = json.loads((OUT_DIR / "bench.json").read_text())
    results = bench["results"]
    # Sort: AUROC desc, then F1 desc
    results_sorted = sorted(
        results,
        key=lambda r: (
            r["auroc"] if r["auroc"] == r["auroc"] else -1.0,  # NaN-safe
            r["f1"],
        ),
        reverse=True,
    )
    top = results_sorted[0]
    winner = {
        "model": top["model"],
        "path": MODEL_PATHS[top["model"]],
        "threshold": top["threshold"],
        "auroc": top["auroc"],
        "f1": top["f1"],
    }
    (OUT_DIR / "winner.json").write_text(json.dumps(winner, indent=2))
    print(f"Winner: {winner['model']}  AUROC={winner['auroc']:.4f}  F1={winner['f1']:.4f}")
    print(f"Saved → {OUT_DIR / 'winner.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
