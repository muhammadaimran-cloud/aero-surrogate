"""
mesh_check.py — mesh sensitivity study.

Reruns run_000 with one refinement level finer everywhere (surface,
features, wake box), then compares Cd against the coarse-mesh value
already in results.csv.

If the two Cd values agree within a few percent, the coarse mesh is
validated for the whole 120-run campaign (and you have a defensible
"mesh independence" paragraph for the project writeup).

Usage:  python3 mesh_check.py          (needs run_cfd_local.py next to it)
"""

import csv
import time
from pathlib import Path

import run_cfd_local as rc

# one level finer: surface (3 4)->(4 5), feature 4->5, wake box 2->3
rc.FILES["system/snappyHexMeshDict"] = (
    rc.FILES["system/snappyHexMeshDict"]
    .replace("level (3 4);", "level (4 5);")
    .replace("levels ((1e15 2))", "levels ((1e15 3))")
    .replace("level 4;", "level 5;"))

NP = 6

d = rc.load_designs()[0]
rc.check_docker()
case = rc.CASES_DIR / "mesh_check"

print("running run_000 on the FINE mesh (expect noticeably longer)…")
t0 = time.time()
rc.write_case(case, Path(d["stl_file"]), float(d["frontal_area_m2"]), NP, rc.ITERS)
rc.run_case(case, NP)
cd_fine, cl_fine = rc.parse_coeffs(case)
mins = (time.time() - t0) / 60
print(f"\nfine mesh:   Cd={cd_fine:.4f}  Cl={cl_fine:.4f}   ({mins:.1f} min)")

cd_coarse = None
if Path(rc.RESULTS_CSV).exists():
    for r in csv.DictReader(open(rc.RESULTS_CSV)):
        if r["run_id"] == d["run_id"]:
            cd_coarse = float(r["Cd"])
            print(f"coarse mesh: Cd={cd_coarse:.4f}  Cl={float(r['Cl']):.4f}")
if cd_coarse:
    diff = 100 * abs(cd_fine - cd_coarse) / cd_fine
    print(f"\nCd difference: {diff:.1f}%")
    print("verdict:", "coarse mesh OK — run the batch"
          if diff < 5 else
          "meaningful mesh dependence — do not run the batch on this mesh")
