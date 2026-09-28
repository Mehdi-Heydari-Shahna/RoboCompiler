from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/"original"
if str(ORIGINAL) not in sys.path: sys.path.insert(0,str(ORIGINAL))
