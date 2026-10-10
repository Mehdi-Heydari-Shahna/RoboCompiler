"""Paths and common reproducibility helpers for the tracked extension."""
from pathlib import Path
import os
import sys
import json
import tempfile
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source' / 'Excavator_RoboIR_full_body_v21'
RESULTS = ROOT / 'results'
sys.path.insert(0, str(SOURCE))
RESULTS.mkdir(exist_ok=True)
# The excavator model is supplied at run time (see Excavator_CMG_PACDM/model_access/excavator_model.py).
_ACCESS = next(p / 'model_access' for p in ROOT.parents if (p / 'model_access' / 'excavator_model.py').is_file())
if str(_ACCESS) not in sys.path:
    sys.path.append(str(_ACCESS))
import excavator_model  # noqa: E402


def source_base_position():
    """Position of the undercarriage (body_53) frame in the source world: joint fixed_world_base."""
    import numpy as np
    cmg = json.loads(excavator_model.source('data/accepted_cmg_v04.json').read_text())
    joint = next(j for j in cmg['joints'] if j['id'] == 'fixed_world_base')
    return (np.asarray(joint['T_BJ']) @ np.linalg.inv(np.asarray(joint['T_FJ'])))[:3, 3]


def save_json(path, value):
    def convert(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        temporary=Path(f.name)
        f.write(contents)
    os.replace(temporary,path)

def save_npz(path, **arrays):
    """Publish a complete, CRC-checked trajectory without replacing good evidence early."""
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix=path.name+'.',
                                     suffix='.tmp',delete=False) as f:
        temporary=Path(f.name)
        np.savez_compressed(f,**arrays)
    with zipfile.ZipFile(temporary) as archive:
        bad=archive.testzip()
        if bad is not None:raise IOError('Trajectory CRC failure: '+bad)
    os.replace(temporary,path)
