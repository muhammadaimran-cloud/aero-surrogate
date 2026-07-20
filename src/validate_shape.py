"""
validate_shape.py — test the surrogate on any shape YOU choose.

Two modes:
  1. Prediction only (instant):
       python3 validate_shape.py 0.30 3.0 0.38 12 18
     (order: nose_frac nose_power tail_frac backlight_deg boattail_deg)

  2. Prediction + real CFD on the same shape (~8 min, fine mesh):
       python3 validate_shape.py 0.30 3.0 0.38 12 18 --cfd
     Builds the STL, runs the full OpenFOAM pipeline, and prints
     surrogate-vs-CFD side by side. Does NOT touch results.csv.

Needs: surrogate.joblib, geometry.py, run_cfd_local.py in this folder.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np

PARAM_NAMES = ["nose_frac", "nose_power", "tail_frac",
               "backlight_deg", "boattail_deg"]


def main():
    args = [a for a in sys.argv[1:] if a != "--cfd"]
    do_cfd = "--cfd" in sys.argv
    if len(args) != 5:
        sys.exit(f"need 5 numbers: {' '.join(PARAM_NAMES)}  [--cfd]")
    params = dict(zip(PARAM_NAMES, map(float, args)))

    # ---- surrogate prediction --------------------------------------------
    b = joblib.load("surrogate.joblib")
    X = b["scaler"].transform(
        np.array([params[k] for k in b["features"]]).reshape(1, -1))
    cd_pred, cd_std = b["models"]["Cd"].predict(X, return_std=True)
    cl_pred = b["models"]["Cl"].predict(X)
    cd_pred, cd_std, cl_pred = float(cd_pred[0]), float(cd_std[0]), float(cl_pred[0])

    print("shape:", ", ".join(f"{k}={params[k]:g}" for k in PARAM_NAMES))
    print(f"surrogate:  Cd = {cd_pred:.4f} ± {cd_std:.4f}   Cl = {cl_pred:.4f}")
    if cd_std > 0.01:
        print("  (note: high uncertainty — this shape is far from the "
              "training data)")
    if not do_cfd:
        print("\nadd --cfd to verify this prediction with a real OpenFOAM run")
        return

    # ---- real CFD on the same shape --------------------------------------
    import geometry
    import run_cfd_local as rc

    # fine production mesh (same as run_batch.py)
    rc.FILES["system/snappyHexMeshDict"] = (
        rc.FILES["system/snappyHexMeshDict"]
        .replace("level (3 4);", "level (4 5);")
        .replace("levels ((1e15 2))", "levels ((1e15 3))")
        .replace("level 4;", "level 5;"))

    rc.check_docker()
    stl = Path("geometries") / "custom_test.stl"
    stl.parent.mkdir(exist_ok=True)
    geometry.build_mesh(params).export(stl)
    aref = geometry.frontal_area(params)

    print("\nrunning CFD (fine mesh, ~8 min)…")
    t0 = time.time()
    case = rc.CASES_DIR / "custom_test"
    rc.write_case(case, stl, aref, 8, rc.ITERS)
    rc.run_case(case, 8)
    cd_cfd, cl_cfd = rc.parse_coeffs(case)
    print(f"CFD:        Cd = {cd_cfd:.4f}            Cl = {cl_cfd:.4f}   "
          f"({(time.time()-t0)/60:.1f} min)")

    err = abs(cd_pred - cd_cfd)
    print(f"\nsurrogate error on Cd: {err:.4f} "
          f"({100*err/cd_cfd:.1f}% of CFD value)")
    print("within the model's own uncertainty band"
          if err <= 2 * cd_std else
          "outside 2x the model's claimed uncertainty — investigate before trusting")


if __name__ == "__main__":
    main()
