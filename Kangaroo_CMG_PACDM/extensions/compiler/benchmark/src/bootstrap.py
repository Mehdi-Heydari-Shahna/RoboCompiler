"""Import paths for the unchanged original sources shipped in ``original/``."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'original'
V22 = ORIGINAL / 'original_v22'
sys.dont_write_bytecode = True
for path in (ORIGINAL, V22):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
