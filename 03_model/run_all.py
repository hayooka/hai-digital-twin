"""
run_all.py — Run all four training scripts sequentially, then compare results.

Usage:
    python 03_model/run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
PYTHON = sys.executable

SCRIPTS = [
    ("GRU",             ROOT / "03_model" / "train_gru.py"),
    ("LSTM",            ROOT / "03_model" / "train_lstm.py"),
    ("Transformer",     ROOT / "03_model" / "train_transformer.py"),
    ("Transformer-SS",  ROOT / "03_model" / "train_transformer_plant.py"),
]

def run(name: str, script: Path):
    print("\n" + "=" * 70)
    print(f"  RUNNING: {name}  ({script.name})")
    print("=" * 70)
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n  WARNING: {name} exited with code {result.returncode}")
    else:
        print(f"\n  DONE: {name}")


if __name__ == "__main__":
    for name, script in SCRIPTS:
        run(name, script)

    print("\n" + "=" * 70)
    print("  All models trained. Running comparison...")
    print("=" * 70)
    subprocess.run([PYTHON, str(ROOT / "03_model" / "compare_results.py")], cwd=str(ROOT))
