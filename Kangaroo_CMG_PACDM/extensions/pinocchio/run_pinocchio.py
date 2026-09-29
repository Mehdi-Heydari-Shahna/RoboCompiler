"""Regenerate and verify the complete Kangaroo Pinocchio/PACDM release.

No MuJoCo module is imported by any action of this script.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

# Set before scientific imports; inherited by spawned worker processes.
# Bytecode caches are not written, so original_v22/ stays byte-identical.
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def execute_case(name):
    """Run one declared case in this process with a fresh plant."""
    from kangaroo_pin.cases import run
    started = time.perf_counter()
    run(name, progress=True)
    return name, time.perf_counter() - started


def verify_manifest():
    """Standard-library-only integrity check of every file."""
    import hashlib
    manifest = json.loads((ROOT / 'SHA256SUMS.json').read_text())
    bad = [k for k, v in manifest.items()
           if not (ROOT / k).is_file() or hashlib.sha256((ROOT / k).read_bytes()).hexdigest() != v]
    if bad:
        raise RuntimeError('Missing/modified files: ' + ', '.join(bad))
    print(f'Integrity PASS: {len(manifest)} files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-existing', action='store_true',
                        help='Check recorded SHA-256 hashes and the recorded verdict; does not simulate.')
    action.add_argument('--render-only', action='store_true',
                        help='Render the saved nominal states to video; does not simulate.')
    action.add_argument('--case', help='Run one declared case only (invalidates the aggregate verdict '
                                       'until --analyze-existing is run).')
    action.add_argument('--analyze-existing', action='store_true',
                        help='Rerun every independent check, audit and gate on the saved trajectories.')
    parser.add_argument('--workers', type=int, default=1, help='Parallel case processes (1 to 8).')
    parser.add_argument('--no-video', action='store_true', help='Skip video rendering.')
    args = parser.parse_args()
    if args.verify_existing:
        verify_manifest()
        verdict = json.loads((ROOT / 'results/validation.json').read_text())
        print(f"Recorded verdict: {'PASS' if verdict.get('passed') else 'FAIL'} "
              f"{verdict.get('passed_count')}/{verdict.get('check_count')} gates.")
        if not verdict.get('passed'):
            raise SystemExit(1)
        return
    import pinocchio as pin
    if pin.__version__ != '3.8.0':
        raise RuntimeError(f'Verified release requires Pinocchio 3.8.0; found {pin.__version__}. '
                           'Use environment.yml or requirements-linux.txt.')
    from kangaroo_pin.cases import ALL, BY_NAME, EVIDENCE, POSITIVE
    from kangaroo_pin.pin_results import aggregate, hash_manifest, write_json
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    r = ROOT / 'results'
    r.mkdir(exist_ok=True)
    if args.render_only:
        from kangaroo_pin.pin_render import render_video
        from kangaroo_pin.pin_report import build_report
        render_video(ROOT)
        execution = json.loads((r / 'execution.json').read_text()) if (r / 'execution.json').exists() else {}
        execution['video_requested'] = True
        write_json(r / 'execution.json', execution)
        build_report(ROOT)
        hash_manifest(ROOT)
        return
    if args.case:
        if args.case not in BY_NAME:
            parser.error(f'unknown case {args.case}; choose from {sorted(BY_NAME)}')
        write_json(r / 'validation.json', dict(passed=False, status='Single case rerun; aggregate acceptance incomplete.'))
        execute_case(args.case)
        print('Case completed. Run --analyze-existing to update aggregate acceptance.')
        return
    write_json(r / 'validation.json', dict(passed=False, status='Run in progress; aggregate acceptance incomplete.'))
    from kangaroo_pin.pin_checks import (audit_rollout, make_oracle, validate_audit_negative_controls,
                                         validate_mechanics)
    from kangaroo_pin.pin_evidence import audit_case
    from kangaroo_pin.pin_simulation import KangarooPlant
    from kangaroo_pin.reference import rebuild
    timings = {}
    if not args.analyze_existing:
        started = time.perf_counter()
        names = [c['name'] for c in sorted(ALL, key=lambda c: -c['duration'] / c['dt'])]
        if args.workers == 1:
            for name in names:
                timings[name] = execute_case(name)[1]
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for name, elapsed in pool.map(execute_case, names):
                    timings[name] = elapsed
                    print(f'Completed {name} in {elapsed:.0f} s', flush=True)
        # Preliminary record, kept by a later --analyze-existing if the analysis is repeated.
        write_json(r / 'execution.json', dict(
            action='simulations_completed_analysis_pending',
            simulation_completed_utc=datetime.now(timezone.utc).isoformat(),
            simulation_driver=f'run_pinocchio.py, {args.workers} worker process(es)',
            simulation_wall_time_s=time.perf_counter() - started, simulation_case_wall_times_s=timings,
            platform=platform.platform(), processor_count=os.cpu_count()))
    analysis_started = time.perf_counter()
    print('Rebuilding the accepted contact reference without MuJoCo...', flush=True)
    write_json(r / 'reference_rebuild.json', rebuild())
    plant = KangarooPlant()
    print('Independent Pinocchio mechanics checks...', flush=True)
    mechanics = validate_mechanics(plant)
    write_json(r / 'mechanics.json', mechanics)
    oracle = make_oracle(plant)
    audits = {}
    for case in POSITIVE:
        print(f"Native dynamics/closure/contact audit: {case['name']}", flush=True)
        audits[case['name']] = audit_rollout(plant, oracle, r / f"{case['name']}.npz", case['dt'],
                                             {**plant.A.DEFAULTS, **case['config']}['friction'])
    write_json(r / 'trajectory_audits.json', audits)
    nominal = BY_NAME['landing_nominal']
    write_json(r / 'audit_negative_controls.json',
               validate_audit_negative_controls(plant, oracle, r / 'landing_nominal.npz', nominal['dt'],
                                                plant.A.DEFAULTS['friction']))
    evidence = {case['name']: audit_case(ROOT, plant, case) for case in EVIDENCE}
    write_json(r / 'case_evidence.json', evidence)
    result = aggregate(ROOT)
    versions = {name: importlib.metadata.version(name)
                for name in ['numpy', 'scipy', 'matplotlib', 'pillow', 'imageio', 'imageio-ffmpeg', 'vtk']}
    versions['pinocchio'] = pin.__version__
    versions['python'] = platform.python_version()
    previous = json.loads((r / 'execution.json').read_text()) if (r / 'execution.json').exists() else {}
    completed = datetime.now(timezone.utc).isoformat()
    simulations = {}
    for case in ALL:
        path = r / f"{case['name']}.json"
        summary = json.loads(path.read_text())
        simulations[case['name']] = dict(
            elapsed_s=summary.get('elapsed_s'),
            result_file_modified_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
    write_json(r / 'execution.json', dict(
        completed_utc=completed,
        simulation_completed_utc=previous.get('simulation_completed_utc'),
        simulation_driver=previous.get('simulation_driver'),
        simulation_wall_time_s=previous.get('simulation_wall_time_s'),
        simulation_case_wall_times_s=previous.get('simulation_case_wall_times_s'),
        simulations=simulations, analysis_wall_time_s=time.perf_counter() - analysis_started,
        platform=platform.platform(), processor_count=os.cpu_count(), versions=versions,
        action='audit_existing_trajectories' if args.analyze_existing else 'fresh_full_suite',
        video_requested=not args.no_video, mujoco_imported='mujoco' in sys.modules,
        independent_constraint_solver='Pinocchio constraintDynamics (CONTACT_3D) + native universal rows',
        contact_solver='Pinocchio PGSContactSolver / ADMMContactSolver',
        simulation_cases=ALL))
    if not args.no_video:
        from kangaroo_pin.pin_render import render_video
        render_video(ROOT)
    from kangaroo_pin.pin_report import build_report
    build_report(ROOT)
    hash_manifest(ROOT)
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['passed_count']}/{result['check_count']} aggregate gates.")
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
