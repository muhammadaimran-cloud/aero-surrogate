# AeroSurrogate — CFD-Trained ML Surrogate for Aerodynamic Shape Optimization

**120 automated OpenFOAM simulations → Gaussian Process surrogate (R² = 0.98) → evolutionary optimization → CFD-verified result.**

The optimizer found a shape with **Cd = 0.047** — 64% lower drag than the mid-range seed design and **20% lower than any design in the training data** — and a blind CFD run confirmed the surrogate's prediction within its own stated uncertainty.

<p align="center">
  <img src="figures/optimized.png" width="360" alt="Optimized geometry">
</p>
<p align="center"><em>The optimized body: near-maximum tail length and boat-tailing, moderate 11.8° backlight angle — physics the model learned purely from data.</em></p>

## The workflow

```
parametric geometry (5 variables, fixed frontal area)
        │  Latin Hypercube sampling, N = 120
        ▼
automated CFD pipeline — OpenFOAM in Docker (RANS, k-ω SST, Re = 2×10⁶)
        │  120/120 runs, 14.9 h on one laptop, mesh independence verified
        ▼
Gaussian Process surrogate — Cd R² = 0.981, MAE = 0.005 on held-out designs
        │  ~100,000× cheaper per evaluation than CFD, with uncertainty estimates
        ▼
OpenEvolve optimization — score = −(Cd + σ), uncertainty-penalized
        │  200 iterations, LLM-driven evolutionary search
        ▼
blind CFD validation — predicted Cd 0.0493 ± 0.0057, CFD gave 0.0470 ✓
```

## Key results

| Stage | Result |
|---|---|
| CFD campaign | 120/120 runs completed unattended, 7.4 min average per case |
| Mesh study | coarse mesh over-predicted Cd by 18% (caught before the campaign); production mesh within 0.6% of reference |
| Surrogate (GPR) | Cd: R² = 0.981, MAE = 0.0052 · Cl: R² = 0.997 — beats gradient boosting at this data scale |
| Blind test 1 (hand-picked shape) | predicted 0.0549 ± 0.0057, CFD 0.0589 — within uncertainty |
| Optimization | best design at iteration 192 of 200 |
| Blind test 2 (optimized shape) | predicted 0.0493 ± 0.0057, **CFD 0.0470** — within uncertainty |
| Bottom line | optimized Cd is **20% below the entire 120-run training distribution** |

<p align="center">
  <img src="figures/parity.png" width="520" alt="Parity plots">
</p>
<p align="center"><em>Predicted vs CFD on 24 held-out designs the models never saw.</em></p>

<p align="center">
  <img src="figures/cd_distribution.png" width="480" alt="Cd distribution">
</p>
<p align="center"><em>The optimized design (red) sits below every design in the training data — generalization, not memorization.</em></p>

## Why the details matter

- **Mesh independence first.** A three-level refinement ladder (coarse → fine → very fine) showed the coarse mesh inflated drag by 18%. Skipping this check would have trained the model on mesh error. Production mesh agreed with a much finer reference within 0.6%.
- **Fixed frontal area.** All 120 designs share the same frontal area, so Cd differences isolate *shape* effects — the optimizer can't cheat by shrinking the body.
- **Uncertainty-penalized optimization.** The GPR reports its own confidence; the evaluator scores candidates by −(Cd + σ), so the optimizer can't exploit regions the CFD never sampled.
- **The physics checks out.** Permutation importance ranks boat-tail angle and tail length as dominant for drag — consistent with wake physics of ground vehicles — and the optimizer avoided the steep-backlight regime associated with the Ahmed-body drag crisis without being told about it.

## Limitations

- **Simplified geometry.** The 5-parameter body family captures gross shape effects (nose, tail, taper) but not real-vehicle features like wheels, underbody detail, or surface curvature continuity.
- **Mesh independence was verified on one design.** The convergence ladder used run_000; mesh error could differ elsewhere in the design space, particularly at extreme tail angles.
- **No prism layers.** The mesh resolves pressure drag well but under-resolves wall friction, so absolute Cd values are likely biased low; comparisons *between* designs (the quantity the surrogate learns) are less affected.
- **Two blind CFD validations.** Both landed within the model's uncertainty band, but two points is evidence, not proof, of accuracy everywhere in the design space.
- **Single operating point.** All data is at 30 m/s, zero yaw, steady RANS — no crosswind, transient wake dynamics, or Reynolds sweep.

Each of these is addressable with more compute: mesh checks at design-space corners, prism-layer meshing, more validation points, and multi-condition datasets.

## Repository guide

| File | Role |
|---|---|
| `src/geometry.py` | parametric body → watertight STL (5 shape variables, pinned frontal area) |
| `src/sample_designs.py` | Latin Hypercube DoE → `designs.csv` + STLs |
| `src/run_cfd_local.py` | automated OpenFOAM pipeline (Docker): mesh, solve, extract Cd/Cl |
| `src/mesh_check.py`, `src/mesh_check2.py` | mesh independence ladder |
| `src/run_batch.py` | full 120-run campaign on the validated production mesh |
| `src/train_surrogate.py` | GPR + gradient boosting, holdout validation, parity plots → `surrogate.joblib` |
| `src/validate_shape.py` | predict any shape, optionally verify with a real CFD run (`--cfd`) |
| `OpenEvolve/` | OpenEvolve optimization (3-file pattern) |
| `results.csv` | the 120-run CFD dataset (design variables → Cd, Cl, forces) |
| `results_coarse.csv` | archived coarse-mesh results from the mesh study |
| `report/` | full lab report (PDF) |
| `legacy/run_cfd.py` | original SimScale API runner — retired when the LBM solver required GPU quota unavailable on the academic plan; the pipeline was rebuilt on containerized OpenFOAM |

## Reproducing it

```bash
pip install -r requirements.txt
# run all commands below from the repository root
# Docker Desktop must be running
# for the optimization step: export OPENAI_API_KEY=<your Gemini API key>

python3 src/sample_designs.py            # generate the 120-design DoE
python3 src/mesh_check.py                # (optional) mesh sensitivity, rung 1
python3 src/run_batch.py                 # full CFD campaign (~15 h)
python3 src/train_surrogate.py           # train + validate the surrogate
python3 src/validate_shape.py 0.30 3.0 0.38 12 18 --cfd    # blind-test any shape
python3 -m openevolve.cli OpenEvolve/initial_program.py OpenEvolve/evaluator.py --config OpenEvolve/config.yaml --iterations 200
```

Stack: Python, OpenFOAM v2412 (Docker), scikit-learn, trimesh, SciPy, OpenEvolve (Gemini).

---

*Muhammad Abdullah Imran · Mechanical Engineering, CSULB · part of a portfolio of optimization projects (airfoil, bridge truss, bluff body, heat sink, 3D car body) built on the same OpenEvolve pattern.*
