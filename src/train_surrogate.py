"""
train_surrogate.py — ML surrogate for the CFD dataset.  (v2)

v2 protocol fixes (post-review):
  - StandardScaler is fit on the TRAINING split only (no preprocessing
    leakage into the held-out test set)
  - model selection (GPR vs gradient boosting) is done by 5-fold
    cross-validation on the training set; the test set is used exactly
    once, to report the selected model's final performance
  - the shipped model is then refit on all 120 samples (standard practice
    once evaluation is locked in)

Outputs:
  surrogate.joblib        (dict: models, scaler, feature names, metadata)
  surrogate_parity.png

Usage:  python3 train_surrogate.py       (run from repo root; needs results.csv)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["nose_frac", "nose_power", "tail_frac",
            "backlight_deg", "boattail_deg"]
TARGETS = ["Cd", "Cl"]


def make_gpr():
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(len(FEATURES)),
                                           nu=2.5)
              + WhiteKernel(noise_level=1e-4))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=8, random_state=SEED)


def make_gbr():
    return GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                     max_depth=3, subsample=0.8,
                                     random_state=SEED)


def pipe(factory):
    """Scaler + model as one unit, so CV refits the scaler per fold."""
    return make_pipeline(StandardScaler(), factory())


def main():
    df = pd.read_csv("results.csv")
    X = df[FEATURES].values

    winners = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    for row, target in enumerate(TARGETS):
        y = df[target].values
        # split FIRST — the test set never influences any fitted component
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                              random_state=SEED)
        cv = KFold(5, shuffle=True, random_state=SEED)
        scores = {}
        for col, (name, factory) in enumerate(
                [("GPR", make_gpr), ("GradBoost", make_gbr)]):
            cv_r2 = cross_val_score(pipe(factory), Xtr, ytr, cv=cv,
                                    scoring="r2").mean()
            scores[name] = cv_r2

            # fit on the training split and report test metrics (for the
            # parity plot); selection itself uses only cv_r2
            model = pipe(factory).fit(Xtr, ytr)
            pred = model.predict(Xte)
            mae = mean_absolute_error(yte, pred)
            r2 = r2_score(yte, pred)
            print(f"{target:>2} | {name:<9} | 5-fold CV R2={cv_r2:.3f} "
                  f"| test R2={r2:.3f} | test MAE={mae:.4f}")

            ax = axes[row, col]
            ax.scatter(yte, pred, s=28, alpha=0.8, edgecolor="k", lw=0.4)
            lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
            ax.plot(lim, lim, "r--", lw=1)
            ax.set_xlabel(f"CFD {target}")
            ax.set_ylabel(f"predicted {target}")
            ax.set_title(f"{name} — {target}  (R²={r2:.3f}, MAE={mae:.4f})")

        chosen = max(scores, key=scores.get)          # selected by CV only
        winners[target] = chosen
        print(f"{target}: selected {chosen} by cross-validation "
              f"(CV R2 {scores[chosen]:.3f} vs {min(scores.values()):.3f})")

    # permutation importance of the selected Cd model on the held-out set
    y_cd = df["Cd"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y_cd, test_size=0.2,
                                          random_state=SEED)
    cd_factory = make_gpr if winners["Cd"] == "GPR" else make_gbr
    cd_model = pipe(cd_factory).fit(Xtr, ytr)
    imp = permutation_importance(cd_model, Xte, yte, n_repeats=30,
                                 random_state=SEED)
    order = np.argsort(imp.importances_mean)[::-1]
    print(f"\nfeature importance for Cd ({winners['Cd']}):")
    for i in order:
        print(f"   {FEATURES[i]:<15} {imp.importances_mean[i]:.4f}")

    plt.tight_layout()
    plt.savefig("surrogate_parity.png", dpi=130)

    # ship: refit scaler + selected models on ALL data
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    final = {}
    for target in TARGETS:
        factory = make_gpr if winners[target] == "GPR" else make_gbr
        final[target] = factory().fit(Xs, df[target].values)
        print(f"\nshipped {target} model: {winners[target]} "
              f"(selected by CV), retrained on all {len(df)} samples")

    joblib.dump(dict(models=final, scaler=scaler, features=FEATURES,
                     metadata=dict(n_samples=len(df), seed=SEED,
                                   protocol="v2: scaler fit on train split; "
                                            "model selection by 5-fold CV",
                                   source="OpenFOAM kOmegaSST fine mesh, "
                                          "U=30 m/s, Re_L=2e6")),
                "surrogate.joblib")
    print("\nsaved -> surrogate.joblib, surrogate_parity.png")


if __name__ == "__main__":
    main()
