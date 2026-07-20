"""
run_cfd_local.py — local OpenFOAM CFD batch runner.  (v2)

v2: mesh quality settings written inline (the #includeEtc file doesn't
exist in the opencfd Docker image).

For every row of designs.csv not already in results.csv:
    build an OpenFOAM case -> mesh the STL with snappyHexMesh ->
    steady RANS (simpleFoam, k-omega SST) -> average Cd/Cl over the last
    30% of iterations -> append to results.csv.

Everything runs inside Docker, so no OpenFOAM install is needed — only
Docker Desktop. The entire OpenFOAM case template is embedded in this file.

Flow setup (identical for EVERY run — consistency makes the dataset trainable):
    U = 30 m/s air, Re_L = 2e6, k-omega SST with wall functions
    domain x[-3,6] y[-2,2] z[-0.05,2.5]  (~1% blockage)
    no-slip ground 5 cm below the body, slip sides/top
    no prism layers (robustness across 120 auto-meshed shapes; fine for a
    comparative pressure-drag-dominated dataset)

Usage:
    python3 run_cfd_local.py --prepare-only   # build case for run_000, no CFD
    python3 run_cfd_local.py --test           # run ONLY run_000 end-to-end
    python3 run_cfd_local.py                  # full batch, sequential
    python3 run_cfd_local.py --np 8           # more CPU cores per case
Safe to Ctrl-C and rerun: finished runs are skipped via results.csv.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---- constants (NEVER change mid-dataset) ---------------------------------
U_INF = 30.0
RHO = 1.225
K_INLET = 0.135          # 1% turbulence intensity
OMEGA_INLET = 10.0
ITERS = 1000             # simpleFoam iterations
AVG_FRACTION = 0.3       # average coefficients over last 30% of iterations
IMAGE = "opencfd/openfoam-default:2412"
BASHRC = "/usr/lib/openfoam/openfoam2412/etc/bashrc"

RESULTS_CSV = "results.csv"
CASES_DIR = Path("cases")

# =========================================================================
#  Embedded OpenFOAM case template.  {AREF} {NP} {ITERS} substituted.
# =========================================================================
HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""

FILES = {}

FILES["system/blockMeshDict"] = HEADER.format(cls="dictionary", obj="blockMeshDict") + """
scale 1;
vertices
(
    (-3 -2 -0.05) (6 -2 -0.05) (6 2 -0.05) (-3 2 -0.05)
    (-3 -2  2.5 ) (6 -2  2.5 ) (6 2  2.5 ) (-3 2  2.5 )
);
blocks ( hex (0 1 2 3 4 5 6 7) (60 27 17) simpleGrading (1 1 1) );
boundary
(
    inlet  {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    ground {{ type wall;  faces ((0 3 2 1)); }}
    sides  {{ type patch; faces ((0 1 5 4) (3 7 6 2)); }}
    top    {{ type patch; faces ((4 5 6 7)); }}
);
""".replace("{{", "{").replace("}}", "}")

FILES["system/surfaceFeatureExtractDict"] = HEADER.format(
    cls="dictionary", obj="surfaceFeatureExtractDict") + """
body.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;
}
"""

FILES["system/snappyHexMeshDict"] = HEADER.format(
    cls="dictionary", obj="snappyHexMeshDict") + """
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    body.stl { type triSurfaceMesh; name body; }
    refinementBox
    {
        type searchableBox;
        min (-0.5 -0.7 -0.05);
        max ( 2.5  0.7  0.9 );
    }
}

castellatedMeshControls
{
    maxLocalCells       2000000;
    maxGlobalCells      8000000;
    minRefinementCells  10;
    nCellsBetweenLevels 3;
    features ( { file "body.eMesh"; level 4; } );
    refinementSurfaces { body { level (3 4); } }
    resolveFeatureAngle 30;
    refinementRegions   { refinementBox { mode inside; levels ((1e15 2)); } }
    locationInMesh      (-2.5 -1.5 1.0);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      30;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers {}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
    relaxed { maxNonOrtho 75; }
}

mergeTolerance 1e-6;
"""

FILES["system/controlDict"] = HEADER.format(cls="dictionary", obj="controlDict") + """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {ITERS};
deltaT          1;
writeControl    timeStep;
writeInterval   {ITERS};
purgeWrite      1;
writeFormat     binary;
writePrecision  7;
timeFormat      general;
runTimeModifiable false;

functions
{
    forceCoeffs1
    {
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (body);
        rho             rhoInf;
        rhoInf          1.225;
        magUInf         30;
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        CofR            (0 0 0);
        pitchAxis       (0 1 0);
        lRef            1.0;
        Aref            {AREF};
    }
}
"""

FILES["system/fvSchemes"] = HEADER.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default cellLimited Gauss linear 1; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default                     none;
    div(phi,U)                  bounded Gauss linearUpwind grad(U);
    div(phi,k)                  bounded Gauss upwind;
    div(phi,omega)              bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""

FILES["system/fvSolution"] = HEADER.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }
}
SIMPLE
{
    consistent      no;
    nNonOrthogonalCorrectors 0;
}
relaxationFactors
{
    fields    { p 0.3; }
    equations { U 0.7; k 0.7; omega 0.7; }
}
"""

FILES["system/decomposeParDict"] = HEADER.format(
    cls="dictionary", obj="decomposeParDict") + """
numberOfSubdomains {NP};
method scotch;
"""

FILES["constant/turbulenceProperties"] = HEADER.format(
    cls="dictionary", obj="turbulenceProperties") + """
simulationType RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""

FILES["constant/transportProperties"] = HEADER.format(
    cls="dictionary", obj="transportProperties") + """
transportModel  Newtonian;
nu              1.5e-05;
"""

FILES["0.orig/U"] = HEADER.format(cls="volVectorField", obj="U") + """
dimensions [0 1 -1 0 0 0 0];
internalField uniform (30 0 0);
boundaryField
{
    inlet  { type fixedValue; value uniform (30 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (30 0 0); }
    ground { type noSlip; }
    sides  { type slip; }
    top    { type slip; }
    body   { type noSlip; }
}
"""

FILES["0.orig/p"] = HEADER.format(cls="volScalarField", obj="p") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    ground { type zeroGradient; }
    sides  { type slip; }
    top    { type slip; }
    body   { type zeroGradient; }
}
"""

FILES["0.orig/k"] = HEADER.format(cls="volScalarField", obj="k") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0.135;
boundaryField
{
    inlet  { type fixedValue; value uniform 0.135; }
    outlet { type inletOutlet; inletValue uniform 0.135; value uniform 0.135; }
    ground { type kqRWallFunction; value uniform 0.135; }
    sides  { type slip; }
    top    { type slip; }
    body   { type kqRWallFunction; value uniform 0.135; }
}
"""

FILES["0.orig/omega"] = HEADER.format(cls="volScalarField", obj="omega") + """
dimensions [0 0 -1 0 0 0 0];
internalField uniform 10;
boundaryField
{
    inlet  { type fixedValue; value uniform 10; }
    outlet { type inletOutlet; inletValue uniform 10; value uniform 10; }
    ground { type omegaWallFunction; value uniform 10; }
    sides  { type slip; }
    top    { type slip; }
    body   { type omegaWallFunction; value uniform 10; }
}
"""

FILES["0.orig/nut"] = HEADER.format(cls="volScalarField", obj="nut") + """
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    ground { type nutkWallFunction; value uniform 0; }
    sides  { type slip; }
    top    { type slip; }
    body   { type nutkWallFunction; value uniform 0; }
}
"""


# =========================================================================
#  Case handling
# =========================================================================
def write_case(case: Path, stl_path: Path, aref: float, np_: int, iters: int):
    if case.exists():
        shutil.rmtree(case)
    for rel, content in FILES.items():
        f = case / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content.replace("{AREF}", f"{aref}")
                            .replace("{NP}", str(np_))
                            .replace("{ITERS}", str(iters)))
    tri = case / "constant/triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    shutil.copy(stl_path, tri / "body.stl")


def docker_run(case: Path, cmd: str, log_name: str, timeout_min=120):
    """Run one OpenFOAM command inside the container, log to file."""
    full = (f"source {BASHRC} && cd /case && {cmd}")
    log = case / f"log.{log_name}"
    with open(log, "w") as lf:
        r = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{case.resolve()}:/case",
             IMAGE, "bash", "-c", full],
            stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_min * 60)
    if r.returncode != 0:
        tail = "".join(open(log).readlines()[-15:])
        raise RuntimeError(f"{log_name} failed (see {log}):\n{tail}")


def run_case(case: Path, np_: int):
    docker_run(case, "surfaceFeatureExtract", "surfaceFeatureExtract", 10)
    docker_run(case, "blockMesh", "blockMesh", 10)
    docker_run(case, "snappyHexMesh -overwrite", "snappyHexMesh", 60)
    docker_run(case, "cp -r 0.orig 0", "copyZero", 5)
    if np_ > 1:
        docker_run(case, "decomposePar", "decomposePar", 15)
        docker_run(case,
                   f"mpirun --allow-run-as-root --oversubscribe -np {np_} "
                   f"simpleFoam -parallel", "simpleFoam", 180)
    else:
        docker_run(case, "simpleFoam", "simpleFoam", 300)


def parse_coeffs(case: Path):
    """Average Cd, Cl over the last AVG_FRACTION of iterations."""
    candidates = list(case.glob("postProcessing/forceCoeffs1/*/coefficient.dat"))
    if not candidates:
        candidates = list(case.glob("postProcessing/forceCoeffs1/*/forceCoeffs.dat"))
    if not candidates:
        raise RuntimeError("no force coefficient output found")
    lines = candidates[0].read_text().splitlines()
    header = [l for l in lines if l.startswith("#")][-1].lstrip("#").split()
    rows = [l.split() for l in lines if not l.startswith("#")]
    i_cd, i_cl = header.index("Cd"), header.index("Cl")
    tail = rows[int(len(rows) * (1 - AVG_FRACTION)):]
    cd = sum(float(r[i_cd]) for r in tail) / len(tail)
    cl = sum(float(r[i_cl]) for r in tail) / len(tail)
    return cd, cl


def cleanup(case: Path):
    """Free disk: drop mesh + fields, keep logs and postProcessing."""
    for sub in ["processor*", "constant/polyMesh", "0", str(ITERS), "0.orig"]:
        for p in case.glob(sub):
            shutil.rmtree(p, ignore_errors=True)


# =========================================================================
#  Batch driver
# =========================================================================
def load_designs():
    with open("designs.csv") as f:
        return list(csv.DictReader(f))


def done_ids():
    if not Path(RESULTS_CSV).exists():
        return set()
    with open(RESULTS_CSV) as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def append_result(design, cd, cl):
    q = 0.5 * RHO * U_INF ** 2
    a = float(design["frontal_area_m2"])
    new = not Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow([*design.keys(), "Cd", "Cl", "drag_N", "lift_N"])
        wr.writerow([*design.values(), f"{cd:.5f}", f"{cl:.5f}",
                     f"{cd * q * a:.4f}", f"{cl * q * a:.4f}"])


def check_docker():
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("Docker isn't running. Start Docker Desktop first.")
    r = subprocess.run(["docker", "image", "inspect", IMAGE],
                       capture_output=True)
    if r.returncode != 0:
        print(f"pulling {IMAGE} (one-time, ~1 GB)…")
        subprocess.run(["docker", "pull", IMAGE], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="only run_000")
    ap.add_argument("--prepare-only", action="store_true",
                    help="write case files for run_000 and stop (no docker)")
    ap.add_argument("--np", type=int, default=6, help="CPU cores per case")
    ap.add_argument("--keep", action="store_true",
                    help="keep full case data (large!)")
    args = ap.parse_args()

    designs = load_designs()
    if args.prepare_only:
        d = designs[0]
        case = CASES_DIR / d["run_id"]
        write_case(case, Path(d["stl_file"]),
                   float(d["frontal_area_m2"]), args.np, ITERS)
        print(f"case written to {case}/ — inspect it, then rerun with --test")
        return

    check_docker()
    if args.test:
        designs = designs[:1]
    todo = [d for d in designs if d["run_id"] not in done_ids()]
    print(f"{len(todo)} designs to run, {args.np} cores each")

    for i, d in enumerate(todo):
        rid = d["run_id"]
        case = CASES_DIR / rid
        t0 = time.time()
        try:
            print(f"[{rid}] ({i+1}/{len(todo)}) meshing + solving…", flush=True)
            write_case(case, Path(d["stl_file"]),
                       float(d["frontal_area_m2"]), args.np, ITERS)
            run_case(case, args.np)
            cd, cl = parse_coeffs(case)
            append_result(d, cd, cl)
            if not args.keep:
                cleanup(case)
            print(f"[{rid}] done in {(time.time()-t0)/60:.1f} min  "
                  f"Cd={cd:.4f}  Cl={cl:.4f}")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"[{rid}] FAILED: {e} — continuing with next design")

    print("\nbatch complete -> results.csv")


if __name__ == "__main__":
    main()
