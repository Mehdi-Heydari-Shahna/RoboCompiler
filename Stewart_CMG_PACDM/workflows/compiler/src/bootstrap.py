"""Locate the unmodified original PACDM and prior independent reference."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'prior_benchmark'
for path in (PRIOR, PRIOR/'original'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
