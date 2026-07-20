"""
mesh_check2.py — second rung of the mesh ladder.

Runs run_000 on a VERY FINE mesh (surface one level beyond mesh_check's
fine mesh) and compares Cd against the fine result still sitting in
cases/mesh_check/. If fine vs very-fine agree within ~5%, the FINE mesh
is validated as the production mesh for the 120-run batch.

NOTE: this is the heaviest run of the project (several million cells).
If your Mac has 16 GB RAM, first open Docker Desktop -> Settings ->
Resources and give Docker at least 10 GB memory (12-16 GB if you have
24/32 GB). Expect 30-60+ min.

Usage:  python3 mesh_check2.py     (needs run_cfd_local.py next to it)
"""

import time
from pathlib import Path

import run_cfd_local as rc

# very fine: surface (5 6), features 6, wake box 3, higher cell cap
rc.FILES["system/snappyHexMeshDict"] = (
    rc.FILES["system/snappyHexMeshDict"]
    .replace("level (3 4);", "level (5 6);")
    .replace("levels ((1e15 2))", "levels ((1e15 3))")
    .replace("level 4;", "level 6;")
    .replace("maxGlobalCells      8000000;", "maxGlobalCells      16000000;"))

NP = 8

d = rc.load_designs()[0]
rc.check_docker()
case = rc.CASES_DIR / "mesh_check_vfine"

print("running run_000 on the VERY FINE mesh (30-60+ min, be patient)…")
t0 = time.time()
rc.write_case(case, Path(d["stl_file"]), float(d["frontal_area_m2"]), NP, rc.ITERS)
rc.run_case(case, NP)
cd_vf, cl_vf = rc.parse_coeffs(case)
print(f"\nvery fine: Cd={cd_vf:.4f}  Cl={cl_vf:.4f}   "
      f"({(time.time()-t0)/60:.1f} min)")

fine_case = rc.CASES_DIR / "mesh_check"
try:
    cd_f, cl_f = rc.parse_coeffs(fine_case)
    print(f"fine:      Cd={cd_f:.4f}  Cl={cl_f:.4f}")
    diff = 100 * abs(cd_vf - cd_f) / cd_vf
    print(f"\nCd difference fine vs very-fine: {diff:.1f}%")
    print("verdict:", "FINE mesh validated — run `python3 run_batch.py`"
          if diff < 5 else
          "still mesh-dependent — refine further before running the batch")
except RuntimeError:
    print("(couldn't find the fine-mesh result in cases/mesh_check — "
          "compare manually: fine Cd was 0.1036)")
