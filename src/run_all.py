"""RetainIQ - one command to reproduce everything from the raw CSV.

    python src/run_all.py
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STAGES = [
    ("Stage 1  clean      ", "clean.py"),
    ("Stage 2  see (EDA)  ", "eda.py"),
    ("Stage 3  predict    ", "train.py"),
    ("Stage 4  prioritise ", "prioritise.py"),
    ("Stage 5  pilot      ", "pilot_power.py"),
    ("Demo     unseen data", "demo_unseen.py"),
    ("Claims register     ", "claims_register.py"),
]

if __name__ == "__main__":
    t0 = time.time()
    for label, script in STAGES:
        print(f"\n{'=' * 70}\n{label}  ({script})\n{'=' * 70}", flush=True)
        t = time.time()
        runpy.run_path(str(HERE / script), run_name="__main__")
        print(f"-- {script} finished in {time.time() - t:.0f}s")
    print(f"\nRetainIQ pipeline complete in {time.time() - t0:.0f}s. Outputs in outputs/, reports in reports/.")
