"""
run_batch.py — full 120-design campaign on the FINE (production) mesh.

Same engine as run_cfd_local.py, but with the fine mesh settings that the
sensitivity study validated (surface level (4 5), features 5, wake box 3).

On first launch it archives any old coarse-mesh results.csv to
results_coarse.csv so coarse and fine data never mix. Fully resumable:
Ctrl-C anytime, rerun later, finished designs are skipped.

Recommended launch (prevents Mac idle-sleep from pausing the batch):

    caffeinate -i python3 run_batch.py

Keep the lid open and the charger plugged in. Expect roughly 7-10 min
per design => 14-20 h total. Run it overnight; progress appends to
results.csv as it goes.
"""

import shutil
import sys
from pathlib import Path

import run_cfd_local as rc

# ---- fine production mesh (validated by mesh_check / mesh_check2) --------
rc.FILES["system/snappyHexMeshDict"] = (
    rc.FILES["system/snappyHexMeshDict"]
    .replace("level (3 4);", "level (4 5);")
    .replace("levels ((1e15 2))", "levels ((1e15 3))")
    .replace("level 4;", "level 5;"))

MARKER = Path(".results_mesh_level")

# never mix coarse-mesh and fine-mesh rows in one dataset
needs_archive = Path(rc.RESULTS_CSV).exists() and (
    not MARKER.exists() or MARKER.read_text().strip() != "fine")
if needs_archive:
    shutil.move(rc.RESULTS_CSV, "results_coarse.csv")
    print("archived old coarse results -> results_coarse.csv")
MARKER.write_text("fine")

sys.argv = ["run_batch.py", "--np", "8"]
rc.main()
