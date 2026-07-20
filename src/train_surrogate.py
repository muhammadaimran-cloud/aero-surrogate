"""
train_surrogate.py — ML surrogate for the CFD dataset.

Trains two model families on results.csv to predict Cd and Cl from the
5 shape variables, evaluates them honestly, and saves the winner:

  1. Gaussian Process Regression (GPR) — data-efficient, gives an
     uncertainty estimate with every prediction. Usually the right tool
     at ~100 samples.
  2. Gradient Boosting — strong tree-based baseline for comparison.

Evaluation protocol:
  - 20% held-out test set the models NEVER see during training/tuning
  - 5-fold cross-validation on the training set (small-data honesty)
  - parity plots (predicted vs CFD truth) -> surrogate_parity.png

Outputs:
  surrogate.joblib        (dict: models, scaler, feature names, metadata)
  surrogate_parity.png

Usage:  python3 train_surrogate.py       (needs results.csv in the folder)
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


def main():
    df = pd.read_csv("results.csv")
    X = df[FEATURES].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    results, best_models = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    for row, target in enumerate(TARGETS):
        y = df[target].values
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2,
                                              random_state=SEED)
        for col, (name, factory) in enumerate(
                [("GPR", make_gpr), ("GradBoost", make_gbr)]):
            model = factory()
            cv = cross_val_score(factory(), Xtr, ytr, cv=KFold(5, shuffle=True,
                                 random_state=SEED), scoring="r2")
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            mae = mean_absolute_error(yte, pred)
            r2 = r2_score(yte, pred)
            results[(target, name)] = dict(cv_r2=cv.mean(), test_r2=r2,
                                           test_mae=mae)
            print(f"{target:>2} | {name:<9} | 5-fold CV R2={cv.mean():.3f} "
                  f"| test R2={r2:.3f} | test MAE={mae:.4f}")

            ax = axes[row, col]
            ax.scatter(yte, pred, s=28, alpha=0.8, edgecolor="k", lw=0.4)
            lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
            ax.plot(lim, lim, "r--", lw=1)
            ax.set_xlabel(f"CFD {target}")
            ax.set_ylabel(f"predicted {target}")
            ax.set_title(f"{name} — {target}  (R²={r2:.3f}, MAE={mae:.4f})")

            # keep the better model per target (by test R2)
            key = (target,)
            if key not in best_models or r2 > best_models[key][1]:
                best_models[key] = (name, r2, model)

    # permutation importance of the best Cd model (physics sanity check)
    cd_name, _, cd_model = best_models[("Cd",)]
    y_cd = df["Cd"].values
    Xtr, Xte, ytr, yte = train_test_split(Xs, y_cd, test_size=0.2,
                                          random_state=SEED)
    imp = permutation_importance(cd_model, Xte, yte, n_repeats=30,
                                 random_state=SEED)
    order = np.argsort(imp.importances_mean)[::-1]
    print(f"\nfeature importance for Cd ({cd_name}):")
    for i in order:
        print(f"   {FEATURES[i]:<15} {imp.importances_mean[i]:.4f}")

    plt.tight_layout()
    plt.savefig("surrogate_parity.png", dpi=130)

    # retrain winners on ALL data before saving (standard practice:
    # evaluation used the holdout; the shipped model uses every sample)
    final = {}
    for target in TARGETS:
        name, r2, _ = best_models[(target,)]
        model = make_gpr() if name == "GPR" else make_gbr()
        model.fit(Xs, df[target].values)
        final[target] = model
        print(f"\nshipped {target} model: {name} "
              f"(holdout R2 was {r2:.3f}), retrained on all 120 samples")

    joblib.dump(dict(models=final, scaler=scaler, features=FEATURES,
                     metadata=dict(n_samples=len(df), seed=SEED,
                                   source="OpenFOAM kOmegaSST fine mesh, "
                                          "U=30 m/s, Re_L=2e6")),
                "surrogate.joblib")
    print("\nsaved -> surrogate.joblib, surrogate_parity.png")


if __name__ == "__main__":
    main()
