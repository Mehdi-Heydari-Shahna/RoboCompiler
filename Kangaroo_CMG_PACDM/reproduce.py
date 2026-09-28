"""Regenerate the full Kangaroo v22 benchmark and supplementary figures."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args):
    command = [sys.executable, *map(str, args)]
    print('\nRunning:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-results', action='store_true',
                        help='Recheck an existing run and regenerate its figures')
    parser.add_argument('--simulation-only', action='store_true',
                        help='Skip figure preparation and rendering')
    args = parser.parse_args()
    run('preflight.py', *([] if args.simulation_only else ['--plots']))
    run('run_kangaroo_v22.py', *(['--reuse-results'] if args.reuse_results else []))
    if not args.simulation_only:
        run('figures/prepare_plot_data.py')
        run('figures/make_figures.py')
        run('figures/draw_cmg.py')
        run('figures/draw_flow.py')
    print('\nFinished. Validation: results/complete_validation_v22.json')
    if not args.simulation_only:
        print('Supplementary figures: results/supplement_figures/')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode or 1)
