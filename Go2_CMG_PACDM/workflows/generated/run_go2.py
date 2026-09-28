#!/usr/bin/env python
"""One-command Go2 benefit benchmark; no paths inside source need editing."""
import sys
try:
    from src.runner import main
except ModuleNotFoundError as exc:
    print(f'Missing dependency: {exc.name}. Activate the intended conda environment, then run:')
    print('  python -m pip install -r requirements.txt')
    print('For native checks activate an environment with robotics Pinocchio installed.')
    raise SystemExit(1)
if __name__=='__main__':raise SystemExit(main())
