"""
run_cfd.py — automated SimScale CFD batch runner.  (v3)

For every row of designs.csv (that isn't already in results.csv):
    upload STL -> import geometry -> set up incompressible LBM external
    aero case -> run -> download force data -> compute Cd/Cl -> append row.

v3: ground is now a NO-SLIP wall (solver requires at least one no-slip /
moving wall; also matches SimScale's own vehicle-aero example).
v2: added mandatory advanced_modelling, transient/statistical/snapshot
result controls, and a mesh refinement region around the body.

Flow setup (identical for EVERY run — consistency makes the dataset trainable):
    U = 30 m/s, air (rho = 1.225), Re_L = 2e6
    domain x[-3,6] y[-2,2] z[-0.05,2.5]  (~1% blockage)
    no-slip ground 5 cm below the body, slip sides/top, velocity inlet,
    pressure outlet; forces averaged over the last 40% of the run

Setup:
    pip3 install simscale-sdk pandas
    export SIMSCALE_API_URL="https://api.simscale.com"
    export SIMSCALE_API_KEY="your-key"

Usage:
    python3 run_cfd.py --test          # ONLY run_000, verbose (do this first!)
    python3 run_cfd.py                 # full batch, 2 sims in flight
    python3 run_cfd.py --parallel 3
Safe to Ctrl-C and rerun: finished runs are skipped via results.csv.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import urllib3
from simscale_sdk import (
    AdvancedModelling, ApiClient, ApiException, AutomaticReferenceLength,
    CoarseResolution, Configuration, ConstantFunction, DecimalVector,
    DimensionalFunctionDimensionless, DimensionalFunctionSpeed,
    DimensionalLength, DimensionalTime, DimensionalVectorAngle,
    DimensionalVectorLength, FixedMagnitudeVBC, FlowDomainBoundaries,
    FluidResultControls, FluidSimulationControl, ForcesMomentsResultControl,
    GeometriesApi, GeometryImportRequest, GeometryImportRequestLocation,
    GeometryImportRequestOptions, GeometryImportsApi, HighResolution,
    IncompressiblePacefish, LocalCartesianBox, NoSlipVBC,
    NoSlipWallEquivalentSandRoughness, PacefishAutomesh,
    PacefishFinenessCoarse, PressureOutletBC, Project, ProjectsApi, Region,
    RotatableCartesianBox, SimulationRun, SimulationRunsApi, SimulationSpec,
    SimulationsApi, SlipVBC, SnapshotResultControl,
    StatisticalAveragingResultControlV2, StorageApi, TopologicalReference,
    TransientResultControl, TurbulenceIntensityTIBC, VelocityInletBC, WallBC,
)

# ---- flow constants (NEVER change these mid-dataset) ----------------------
U_INF = 30.0        # m/s
RHO = 1.225         # kg/m^3
END_TIME_S = 1.2    # physical seconds (~4 flow passes)
AVG_FRACTION = 0.4  # average forces over last 40% of the run
PROJECT_NAME = "ML Surrogate Dataset - Streamlined Bodies"

RESULTS_CSV = "results.csv"
RAW_DIR = Path("cfd_raw")        # raw force CSVs, one per run (never lose data)
PROJECT_ID_FILE = ".simscale_project_id"
API_KEY_HEADER = "X-API-KEY"


# ---------------------------------------------------------------- API setup
def connect():
    key, url = os.getenv("SIMSCALE_API_KEY"), os.getenv("SIMSCALE_API_URL")
    if not key or not url:
        sys.exit("Set SIMSCALE_API_KEY and SIMSCALE_API_URL env vars first.")
    cfg = Configuration()
    cfg.host = url + "/v0"
    cfg.api_key = {API_KEY_HEADER: key}
    client = ApiClient(cfg)
    retry = urllib3.Retry(connect=5, read=5, redirect=0, status=5,
                          backoff_factor=0.3)
    client.rest_client.pool_manager.connection_pool_kw["retries"] = retry
    return client, key


def get_project(project_api):
    """Reuse one project across the whole campaign."""
    if Path(PROJECT_ID_FILE).exists():
        return Path(PROJECT_ID_FILE).read_text().strip()
    project = project_api.create_project(Project(
        name=PROJECT_NAME, description="Automated DoE for ML surrogate",
        measurement_system="SI"))
    Path(PROJECT_ID_FILE).write_text(project.project_id)
    return project.project_id


# ------------------------------------------------------------- per-design
def upload_and_import(client, apis, project_id, stl_path, name):
    storage_api, geo_import_api = apis["storage"], apis["geo_import"]
    storage = storage_api.create_storage()
    with open(stl_path, "rb") as f:
        client.rest_client.PUT(url=storage.url,
                               headers={"Content-Type": "application/octet-stream"},
                               body=f.read())
    req = GeometryImportRequest(
        name=name, location=GeometryImportRequestLocation(storage.storage_id),
        format="STL", input_unit="m",
        options=GeometryImportRequestOptions(
            facet_split=False, sewing=False, improve=True,
            optimize_for_lbm_solver=True))
    imp = geo_import_api.import_geometry(project_id, req)
    t0 = time.time()
    while imp.status not in ("FINISHED", "CANCELED", "FAILED"):
        if time.time() - t0 > 600:
            raise TimeoutError(f"geometry import timeout for {name}")
        time.sleep(8)
        imp = geo_import_api.get_geometry_import(project_id, imp.geometry_import_id)
    if imp.status != "FINISHED":
        raise RuntimeError(f"geometry import {imp.status} for {name}")
    return imp.geometry_id


def body_entities(apis, project_id, geometry_id):
    """All face entities of the imported STL (used for force integration)."""
    maps = apis["geometry"].get_geometry_mappings(project_id, geometry_id,
                                                  _class="face")
    return [m.name for m in maps.embedded]


def make_spec(sim_api, project_id, geometry_id, entities, run_name):
    domain = RotatableCartesianBox(
        name="Flow domain",
        min=DimensionalVectorLength(value=DecimalVector(x=-3.0, y=-2.0, z=-0.05), unit="m"),
        max=DimensionalVectorLength(value=DecimalVector(x=6.0, y=2.0, z=2.5), unit="m"),
        rotation_point=DimensionalVectorLength(value=DecimalVector(x=0, y=0, z=0), unit="m"),
        rotation_angles=DimensionalVectorAngle(value=DecimalVector(x=0, y=0, z=0), unit="°"))
    domain_uuid = sim_api.create_geometry_primitive(project_id, domain).geometry_primitive_id

    # mesh refinement box around the body + near wake
    mesh_region = LocalCartesianBox(
        name="Body mesh region",
        orientation_reference="GEOMETRY",
        min=DimensionalVectorLength(value=DecimalVector(x=-0.5, y=-0.6, z=-0.05), unit="m"),
        max=DimensionalVectorLength(value=DecimalVector(x=2.5, y=0.6, z=0.8), unit="m"))
    mesh_region_uuid = sim_api.create_geometry_primitive(
        project_id, mesh_region).geometry_primitive_id

    model = IncompressiblePacefish(
        bounding_box_uuid=domain_uuid,
        flow_domain_boundaries=FlowDomainBoundaries(
            xmin=VelocityInletBC(
                name="Inlet",
                velocity=FixedMagnitudeVBC(value=DimensionalFunctionSpeed(
                    value=ConstantFunction(value=U_INF), unit="m/s")),
                turbulence_intensity=TurbulenceIntensityTIBC(
                    value=DimensionalFunctionDimensionless(
                        value=ConstantFunction(value=0.01), unit=""))),
            xmax=PressureOutletBC(name="Outlet"),
            ymin=WallBC(name="Side-", velocity=SlipVBC()),
            ymax=WallBC(name="Side+", velocity=SlipVBC()),
            zmin=WallBC(
                name="Ground",
                velocity=NoSlipVBC(
                    no_slip_wall_roughness_type=NoSlipWallEquivalentSandRoughness(
                        surface_roughness=DimensionalLength(value=0, unit="m")))),
            zmax=WallBC(name="Top", velocity=SlipVBC())),
        simulation_control=FluidSimulationControl(
            end_time=DimensionalTime(value=END_TIME_S, unit="s")),
        advanced_modelling=AdvancedModelling(),
        result_control=FluidResultControls(
            forces_moments=[ForcesMomentsResultControl(
                name="body_forces",
                center_of_rotation=DimensionalVectorLength(
                    value=DecimalVector(x=0, y=0, z=0), unit="m"),
                write_control=HighResolution(),
                fraction_from_end=AVG_FRACTION,
                export_statistics=True,
                topological_reference=TopologicalReference(entities=entities))],
            transient_result_control=TransientResultControl(
                write_control=CoarseResolution(),
                export_fluid=True,
                geometry_primitive_uuids=[domain_uuid]),
            statistical_averaging_result_control=StatisticalAveragingResultControlV2(
                sampling_interval=CoarseResolution(),
                export_fluid=True,
                geometry_primitive_uuids=[domain_uuid],
                export_surface=True,
                topological_reference=TopologicalReference(entities=entities)),
            snapshot_result_control=SnapshotResultControl(
                export_fluid=True,
                geometry_primitive_uuids=[domain_uuid])),
        mesh_settings_new=PacefishAutomesh(
            new_fineness=PacefishFinenessCoarse(),
            reference_length_computation=AutomaticReferenceLength(),
            primary_topology=Region(geometry_primitive_uuids=[mesh_region_uuid])))

    spec = SimulationSpec(name=run_name, geometry_id=geometry_id, model=model)
    sim_id = sim_api.create_simulation(project_id, spec).simulation_id
    check = sim_api.check_simulation_setup(project_id, sim_id)
    errors = [e for e in check.entries if e.severity == "ERROR"]
    if errors:
        raise RuntimeError(f"setup check failed for {run_name}: {errors}")
    return sim_id


def fetch_forces(client, api_key, run_api, project_id, sim_id, run_id, rid,
                 verbose=False):
    """Download force results; save raw CSVs; return mean (Fx, Fz)."""
    results = run_api.get_simulation_run_results(project_id, sim_id, run_id,
                                                 page=1, limit=200)
    force_items = []
    for item in results.embedded:
        if verbose:
            print(f"   result: category={item.category!r} name={getattr(item, 'name', '?')!r}")
        if "FORCE" in (item.category or "").upper():
            force_items.append(item)
    if not force_items:
        raise RuntimeError(f"no FORCE results found for {rid} "
                           f"(rerun with --test to list categories)")
    RAW_DIR.mkdir(exist_ok=True)
    frames = []
    for j, item in enumerate(force_items):
        resp = client.rest_client.GET(url=item.download.url,
                                      headers={API_KEY_HEADER: api_key},
                                      _preload_content=False)
        raw = RAW_DIR / f"{rid}_{item.category}_{j}.csv"
        raw.write_bytes(resp.data)
        frames.append(raw)

    import pandas as pd
    for raw in frames:                     # prefer plain time-series data
        try:
            df = pd.read_csv(raw)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        fx = next((cols[c] for c in cols if "force" in c and "x" in c), None)
        fz = next((cols[c] for c in cols if "force" in c and "z" in c), None)
        if fx and fz:
            tail = df.iloc[int(len(df) * (1 - AVG_FRACTION)):]
            return float(tail[fx].mean()), float(tail[fz].mean())
    raise RuntimeError(f"downloaded force CSVs for {rid} but couldn't parse "
                       f"columns — inspect files in {RAW_DIR}/")


# ------------------------------------------------------------------- batch
def load_designs():
    with open("designs.csv") as f:
        return list(csv.DictReader(f))


def done_ids():
    if not Path(RESULTS_CSV).exists():
        return set()
    with open(RESULTS_CSV) as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def append_result(design, cd, cl, fx, fz):
    new = not Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow([*design.keys(), "drag_N", "lift_N", "Cd", "Cl"])
        wr.writerow([*design.values(), f"{fx:.4f}", f"{fz:.4f}",
                     f"{cd:.5f}", f"{cl:.5f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="only run_000, verbose")
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()

    client, api_key = connect()
    apis = {"project": ProjectsApi(client), "storage": StorageApi(client),
            "geo_import": GeometryImportsApi(client),
            "geometry": GeometriesApi(client), "sim": SimulationsApi(client),
            "run": SimulationRunsApi(client)}
    project_id = get_project(apis["project"])
    print(f"project: {project_id}")

    designs = load_designs()
    if args.test:
        designs, args.parallel = designs[:1], 1
    todo = [d for d in designs if d["run_id"] not in done_ids()]
    print(f"{len(todo)} designs to run")

    active = []   # list of dicts: design, sim_id, run_id, started
    while todo or active:
        # top up the in-flight pool
        while todo and len(active) < args.parallel:
            d = todo.pop(0)
            rid = d["run_id"]
            try:
                print(f"[{rid}] uploading geometry…")
                geo_id = upload_and_import(client, apis, project_id,
                                           d["stl_file"], rid)
                ents = body_entities(apis, project_id, geo_id)
                sim_id = make_spec(apis["sim"], project_id, geo_id, ents, rid)
                run = apis["run"].create_simulation_run(
                    project_id, sim_id, SimulationRun(name=rid))
                apis["run"].start_simulation_run(project_id, sim_id, run.run_id)
                active.append({"d": d, "sim": sim_id, "run": run.run_id,
                               "t0": time.time()})
                print(f"[{rid}] running…")
            except (ApiException, RuntimeError, TimeoutError) as e:
                print(f"[{rid}] FAILED at setup: {e} — skipping")
        # poll the pool
        time.sleep(30)
        for slot in active[:]:
            rid = slot["d"]["run_id"]
            run = apis["run"].get_simulation_run(project_id, slot["sim"],
                                                 slot["run"])
            if run.status in ("FINISHED", "CANCELED", "FAILED"):
                active.remove(slot)
                if run.status != "FINISHED":
                    print(f"[{rid}] {run.status} — skipping")
                    continue
                try:
                    fx, fz = fetch_forces(client, api_key, apis["run"],
                                          project_id, slot["sim"],
                                          slot["run"], rid, verbose=args.test)
                    q = 0.5 * RHO * U_INF ** 2
                    a = float(slot["d"]["frontal_area_m2"])
                    cd, cl = fx / (q * a), fz / (q * a)
                    append_result(slot["d"], cd, cl, fx, fz)
                    print(f"[{rid}] done  Cd={cd:.4f}  Cl={cl:.4f}")
                except (ApiException, RuntimeError) as e:
                    print(f"[{rid}] result extraction failed: {e}")
            elif time.time() - slot["t0"] > 3 * 3600:
                print(f"[{rid}] exceeded 3 h — abandoning slot")
                active.remove(slot)

    print("\nbatch complete -> results.csv")


if __name__ == "__main__":
    main()
