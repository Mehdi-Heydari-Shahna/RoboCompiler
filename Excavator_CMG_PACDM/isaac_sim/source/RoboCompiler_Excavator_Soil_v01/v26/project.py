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
