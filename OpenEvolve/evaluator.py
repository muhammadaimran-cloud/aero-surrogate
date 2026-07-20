"""
evaluator.py — scores candidate shapes with the CFD-trained GPR surrogate.

The LLM never sees or edits this file. It evolves only the 5 numbers in
initial_program.py; this evaluator:
  1. executes the candidate file in an empty namespace
  2. clamps every variable to the CFD dataset bounds (no extrapolation)
  3. asks the GPR surrogate for Cd + its uncertainty (std)
  4. score = -(Cd + UNCERTAINTY_WEIGHT * std)   [maximize => minimize Cd]

The uncertainty penalty keeps the optimizer inside the region where the
surrogate is trustworthy — it can't win by exploiting model error in
corners of the design space the CFD never sampled.
"""

from pathlib import Path

import joblib
import numpy as np

BOUNDS = {
    "nose_frac":     (0.15, 0.35),
    "nose_power":    (1.5,  4.0),
    "tail_frac":     (0.15, 0.40),
    "backlight_deg": (5.0,  30.0),
    "boattail_deg":  (0.0,  20.0),
}
UNCERTAINTY_WEIGHT = 1.0
FAIL_SCORE = -10.0

_here = Path(__file__).resolve().parent
_candidates = [_here / "surrogate.joblib", _here.parent / "surrogate.joblib"]
_bundle = joblib.load(next(p for p in _candidates if p.exists()))
_cd_model = _bundle["models"]["Cd"]
_cl_model = _bundle["models"]["Cl"]
_scaler = _bundle["scaler"]
_features = _bundle["features"]


def evaluate(program_path):
    # --- run the candidate file, paranoia everywhere -----------------------
    try:
        src = Path(program_path).read_text()
        ns = {}
        exec(src, ns)                                   # noqa: S102
    except Exception as e:
        return {"combined_score": FAIL_SCORE, "error": f"exec failed: {e}"}

    vals = []
    for name in _features:
        v = ns.get(name)
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            return {"combined_score": FAIL_SCORE,
                    "error": f"bad or missing variable: {name}"}
        lo, hi = BOUNDS[name]
        vals.append(float(np.clip(v, lo, hi)))

    # --- surrogate prediction ---------------------------------------------
    X = _scaler.transform(np.array(vals).reshape(1, -1))
    cd, cd_std = _cd_model.predict(X, return_std=True)
    cl = _cl_model.predict(X)
    cd, cd_std, cl = float(cd[0]), float(cd_std[0]), float(cl[0])

    score = -(cd + UNCERTAINTY_WEIGHT * cd_std)
    return {
        "combined_score": score,
        "cd": cd,
        "cd_uncertainty": cd_std,
        "cl": cl,
        **dict(zip(_features, vals)),
    }
