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
# The excavator model (records, route and parameters) is supplied at run time;
# see Excavator_CMG_PACDM/model_access/excavator_model.py.
if str(ROOT.parent / 'model_access') not in sys.path:
    sys.path.append(str(ROOT.parent / 'model_access'))
import excavator_model  # noqa: E402
