"""Import paths for the preserved original sources (no file is modified)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'original'
V21 = ORIGINAL / 'v21'
VALIDATED = ORIGINAL / 'validated_backend'
for path in (V21, VALIDATED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
