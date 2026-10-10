#!/usr/bin/env python3
"""Single-command Franka benchmark. Run from this directory."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
from src.runner import main
if __name__=='__main__':raise SystemExit(main())
