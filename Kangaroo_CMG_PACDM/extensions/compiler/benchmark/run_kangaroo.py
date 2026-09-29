"""Kangaroo CMG/PACDM framework-benefit benchmark (see the repository README).

Example:  python run_kangaroo.py --profile full --native required --out results_local
"""
import sys

sys.dont_write_bytecode = True

from src.runner import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
