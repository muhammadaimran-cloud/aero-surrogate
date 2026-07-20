"""
geometry.py — parametric streamlined body generator.

Builds a generic vehicle-LIKE body (not an actual car): rounded nose,
constant mid-section, tapered Kammback tail with a flat base.
Think "Ahmed body's smoother cousin".

Overall size is FIXED (length, max width, max height) so the surrogate
learns SHAPE effects on Cd, not size effects. Only 5 shape variables vary:

    nose_frac      fraction of length used by the nose        (0.15 - 0.35)
    nose_power     nose bluntness exponent, higher = blunter  (1.5  - 4.0)
    tail_frac      fraction of length used by the tail        (0.15 - 0.40)
    backlight_deg  downward slope of the tail roof, degrees   (5    - 30)
    boattail_deg   inward taper of the tail sides, degrees    (0    - 20)

Usage:
    from geometry import build_mesh, BOUNDS
    mesh = build_mesh(params_dict)          # trimesh.Trimesh, watertight
    mesh.export("run_001.stl")

Run directly to produce a demo STL:
    python3 geometry.py
"""

import numpy as np
import trimesh

# ---- fixed dimensions (metres) — Ahmed-body-like scale --------------------
L = 1.00   # overall length
W = 0.39   # max width
H = 0.29   # max height (bottom of body sits at z = 0)

Z_TIP = 0.45 * H   # height of the nose tip
M_SEC = 4.0        # cross-section superellipse exponent (rounded rectangle)
MIN_BASE_H = 0.18 * H   # tail base never collapses below this height
MIN_BASE_W = 0.15 * W   # tail base never collapses below this width

BOUNDS = {
    "nose_frac":     (0.15, 0.35),
    "nose_power":    (1.5,  4.0),
    "tail_frac":     (0.15, 0.40),
    "backlight_deg": (5.0,  30.0),
    "boattail_deg":  (0.0,  20.0),
}


def clamp_params(params):
    """Force every variable inside its bounds (paranoid-evaluator habit)."""
    return {k: float(np.clip(params[k], *BOUNDS[k])) for k in BOUNDS}


def _dims_at(x, p):
    """Return (width, z_bottom, z_top) of the cross-section at station x."""
    Ln = p["nose_frac"] * L
    Lt = p["tail_frac"] * L

    if x <= Ln:                                   # nose: superellipse blend
        s = x / Ln
        scale = (1.0 - (1.0 - s) ** p["nose_power"]) ** (1.0 / p["nose_power"])
        w = W * scale
        z_bot = Z_TIP * (1.0 - scale)             # underside rises to the tip
        z_top = Z_TIP + (H - Z_TIP) * scale       # roof falls to the tip
    elif x >= L - Lt:                             # tail: linear tapers
        d = x - (L - Lt)
        z_top = max(H - np.tan(np.radians(p["backlight_deg"])) * d,
                    MIN_BASE_H)
        w = max(W - 2.0 * np.tan(np.radians(p["boattail_deg"])) * d,
                MIN_BASE_W)
        z_bot = 0.0
    else:                                         # constant mid-section
        w, z_bot, z_top = W, 0.0, H
    return w, z_bot, z_top


def _section_points(x, p, n_perim):
    """Perimeter points of one cross-section (rounded rectangle)."""
    w, z_bot, z_top = _dims_at(x, p)
    h = z_top - z_bot
    t = np.linspace(0.0, 2.0 * np.pi, n_perim, endpoint=False)
    e = 2.0 / M_SEC
    cy = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** e
    cz = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** e
    pts = np.empty((n_perim, 3))
    pts[:, 0] = x
    pts[:, 1] = 0.5 * w * cy
    pts[:, 2] = z_bot + 0.5 * h * (1.0 + cz)
    return pts


def build_mesh(params, n_sections=80, n_perim=48):
    """Build a watertight triangle mesh of the body."""
    p = clamp_params(params)

    # station spacing clustered at nose and tail (cosine spacing)
    u = np.linspace(0.0, 1.0, n_sections)
    xs = L * 0.5 * (1.0 - np.cos(np.pi * u))
    xs = xs[1:]                                   # x=0 handled by tip vertex

    verts = [np.array([[0.0, 0.0, Z_TIP]])]       # vertex 0 = nose tip
    for x in xs:
        verts.append(_section_points(x, p, n_perim))
    base_center = np.array([[L, 0.0,
                             0.5 * sum(_dims_at(L, p)[1:])]])
    verts.append(base_center)                     # last vertex = base centre
    V = np.vstack(verts)

    faces = []
    n = n_perim
    # nose fan: tip -> first ring
    for k in range(n):
        faces.append([0, 1 + k, 1 + (k + 1) % n])
    # side strips between consecutive rings
    n_rings = len(xs)
    for i in range(n_rings - 1):
        a0 = 1 + i * n
        b0 = 1 + (i + 1) * n
        for k in range(n):
            k1 = (k + 1) % n
            faces.append([a0 + k, b0 + k, a0 + k1])
            faces.append([a0 + k1, b0 + k, b0 + k1])
    # tail base fan: last ring -> base centre
    c = len(V) - 1
    a0 = 1 + (n_rings - 1) * n
    for k in range(n):
        faces.append([a0 + k, c, a0 + (k + 1) % n])

    mesh = trimesh.Trimesh(vertices=V, faces=np.array(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        raise RuntimeError("mesh not watertight — bad parameter combo?")
    return mesh


def frontal_area(params):
    """Projected frontal area (m^2) — needed later for Cd extraction."""
    mesh = build_mesh(params)
    ys, zs = mesh.vertices[:, 1], mesh.vertices[:, 2]
    # body is convex in projection; superellipse area factor is exact enough:
    # integrate section outline at the widest station instead of guessing.
    w, z_bot, z_top = _dims_at(0.55 * L, clamp_params(params))
    # superellipse area = w*h * gamma-factor; for m=4 factor ≈ 0.927
    return 0.927 * w * (z_top - z_bot)


if __name__ == "__main__":
    demo = {"nose_frac": 0.25, "nose_power": 2.5, "tail_frac": 0.30,
            "backlight_deg": 15.0, "boattail_deg": 10.0}
    m = build_mesh(demo)
    m.export("demo_body.stl")
    print(f"demo_body.stl written | watertight={m.is_watertight} "
          f"| A_front={frontal_area(demo):.4f} m^2 | volume={m.volume:.4f} m^3")
