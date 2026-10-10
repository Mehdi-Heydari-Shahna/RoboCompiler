"""Package paths and import order for the preserved original source.

The excavator model is not part of the repository: the model paths below
(CMG_PATH, MAPPING_PATH, MESH_DIR, VISUAL_XML and the files in DATA and in
ORIGINAL/outputs) are package locations that ``model_file`` maps to the model
folder supplied at run time (see Excavator_CMG_PACDM/model_access/excavator_model.py).
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent / 'model_access') not in sys.path:
    sys.path.append(str(ROOT.parent / 'model_access'))
import excavator_model  # noqa: E402

model_file = excavator_model.mirror
ORIGINAL = ROOT / 'original' / 'RoboCompiler_Excavator_Soil_v01'
V26 = ORIGINAL / 'v26'
V21 = V26 / 'source' / 'Excavator_RoboIR_full_body_v21'
RESULTS = ROOT / 'results'
DATA = ROOT / 'data'
CMG_PATH = V21 / 'data' / 'accepted_cmg_v04.json'
MAPPING_PATH = V21 / 'data' / 'accepted_mujoco_mapping.json'
MESH_DIR = V21 / 'assets' / 'meshes'
VISUAL_XML = V21 / 'assets' / 'visual_actual.xml'
PACDM_CORE = V21 / 'pacdm.py'
PACDM_SHA256 = 'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de'
SOURCE_ARCHIVE_SHA256 = '18049463738af22d6f6b142ff4dec45d3fe8ebff163a07d8bf93848d2822dc9d'


def add_original_to_path():
    """Search order of the original run_soil.py + v26/project.py: V21, V26, ORIGINAL."""
    sys.dont_write_bytecode = True  # never write caches into the preserved tree
    for folder in (ORIGINAL, V26, V21):   # each inserted at 0, so V21 ends first
        text = str(folder)
        if text not in sys.path:
            sys.path.insert(0, text)
