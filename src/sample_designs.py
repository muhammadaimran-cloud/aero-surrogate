"""
sample_designs.py — build the Design of Experiments (DoE).

1. Latin Hypercube Sampling over the 5 shape variables (space-filling,
   far better coverage than a grid for the same number of runs).
2. Writes designs.csv  — one row per geometry (this is the ML input table;
   the CFD results columns get appended later by the API runner).
3. Writes geometries/run_XXX.stl for every design.

Usage:
    python3 sample_designs.py            # default 120 designs
    python3 sample_designs.py 8          # quick test with 8
"""

import csv
import sys
from pathlib import Path

from scipy.stats import qmc

from geometry import BOUNDS, build_mesh, frontal_area

SEED = 42          # fixed seed -> reproducible dataset (say this in interviews)


def sample(n):
    names = list(BOUNDS)
    lo = [BOUNDS[k][0] for k in names]
    hi = [BOUNDS[k][1] for k in names]
    lhs = qmc.LatinHypercube(d=len(names), seed=SEED)
    X = qmc.scale(lhs.random(n), lo, hi)
    return [dict(zip(names, row)) for row in X]


def main(n=120):
    out = Path("geometries")
    out.mkdir(exist_ok=True)
    designs = sample(n)

    with open("designs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run_id", *BOUNDS, "frontal_area_m2", "stl_file"])
        for i, p in enumerate(designs):
            rid = f"run_{i:03d}"
            stl = out / f"{rid}.stl"
            build_mesh(p).export(stl)
            wr.writerow([rid,
                         *[f"{p[k]:.5f}" for k in BOUNDS],
                         f"{frontal_area(p):.5f}", stl])
            print(f"{rid}  ok")

    print(f"\n{n} designs -> designs.csv + {out}/")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
