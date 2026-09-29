"""Regenerate and verify the complete Stewart Pinocchio/PACDM release."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

# Set before scientific imports, also inherited by spawned processes on Windows.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parent


def execute_case(item):
    from stewart.pin_simulation import run_case
    from stewart.pin_results import write_json
    name,dt,payload,ff = item
    cmg = json.loads((ROOT/'data/stewart.cmg.json').read_text())
    # Original model declares payload inertia proportional to its mass.
    body = next(b for b in cmg['bodies'] if b['id']=='payload')
    factor = payload/body['mass_kg']; body['mass_kg'] = payload
    body['inertia_kg_m2'] = [[value*factor for value in row] for row in body['inertia_kg_m2']]
    result = run_case(cmg,ROOT/'results/reference.npz',ROOT/'results',name,dt,feedforward=ff)
    write_json(ROOT/f'results/{name}.json',result)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-existing',action='store_true',help='Check recorded SHA-256 hashes; does not simulate.')
    action.add_argument('--render-only',action='store_true',help='Render saved nominal states; does not simulate.')
    action.add_argument('--case',choices=['coarse','nominal','fine','heavy_payload','no_feedforward'],help='Run one case using existing reference (invalidates aggregate status).')
    action.add_argument('--analyze-existing',action='store_true',help='Rerun independent numerical audits and acceptance on existing trajectories.')
    parser.add_argument('--workers',type=int,default=1,help='Number of independent case processes (1 to 5).')
    parser.add_argument('--no-video',action='store_true',help='Skip video rendering after simulation; final status excludes video.')
    args = parser.parse_args()
    if args.verify_existing:
        # The checksum command needs only Python's standard library: the
        # shipped release manifest, then this run's evidence manifest if present.
        import hashlib
        for manifest_path,label in (('SHA256SUMS.json','release'),('results/SHA256SUMS.json','run evidence')):
            if label=='run evidence' and not (ROOT/manifest_path).is_file():
                print(f'No run evidence manifest ({manifest_path}); run python run_pinocchio.py to create it.'); continue
            manifest = json.loads((ROOT/manifest_path).read_text())
            bad = [k for k,v in manifest.items() if not (ROOT/k).is_file() or hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
            if bad: raise RuntimeError(f'Missing/modified {label} files: '+', '.join(bad))
            print(f'Integrity PASS ({label}): {len(manifest)} files.')
        return
    import pinocchio as pin
    if pin.__version__ != '3.8.0':
        raise RuntimeError(f'Verified release requires Pinocchio 3.8.0; found {pin.__version__}. Use environment.yml.')
    from stewart.pin_results import CASES, write_json, aggregate, hash_manifest
    if not 1 <= args.workers <= 5: parser.error('--workers must be between 1 and 5')
    r = ROOT/'results'; r.mkdir(exist_ok=True)
    if args.render_only:
        from stewart.pin_render import render_video
        render_video(ROOT)
        execution = json.loads((r/'execution.json').read_text()) if (r/'execution.json').exists() else {}
        execution['video_requested'] = True
        write_json(r/'execution.json',execution)
        from stewart.pin_report import build_report
        build_report(ROOT); hash_manifest(ROOT); return
    write_json(r/'validation.json',dict(passed=False,status='Run in progress; aggregate acceptance incomplete.'))
    if args.case:
        execute_case(next(x for x in CASES if x[0]==args.case))
        print('Case completed. Run --analyze-existing to update aggregate acceptance.'); return
    cmg = json.loads((ROOT/'data/stewart.cmg.json').read_text())
    from stewart.pin_checks import validate_mechanics, audit_rollout, validate_audit_negative_controls
    print('Independent Pinocchio mechanics checks...',flush=True)
    mechanics = validate_mechanics(cmg); write_json(r/'mechanics.json',mechanics)
    if not mechanics['passed']: raise RuntimeError('Mechanics acceptance failed; see results/mechanics.json.')
    if not args.analyze_existing:
        from stewart.reference import build_reference
        print('Rebuilding PACDM mission reference...',flush=True)
        write_json(r/'reference.json',build_reference(cmg,r/'reference.npz'))
        if args.workers==1:
            for item in CASES: execute_case(item)
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for name in pool.map(execute_case,CASES): print(f'Completed {name}',flush=True)
    audits = {}
    for name,*_ in CASES:
        print(f'Independent acceleration/closure audit: {name}',flush=True)
        actual = json.loads((r/f'{name}.cmg.json').read_text())
        audits[name] = audit_rollout(actual,r/f'{name}.npz',samples=111)
    write_json(r/'trajectory_audits.json',audits)
    write_json(r/'audit_negative_controls.json',validate_audit_negative_controls(cmg,r/'nominal.npz'))
    from stewart.pin_evidence import audit_case
    evidence = {name:audit_case(ROOT,name,dt,payload,ff) for name,dt,payload,ff in CASES}
    write_json(r/'case_evidence.json',evidence)
    result = aggregate(ROOT)
    versions = {name:importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib','pillow','imageio','imageio-ffmpeg']}
    versions['pinocchio'] = pin.__version__; versions['python'] = platform.python_version()
    previous = json.loads((r/'execution.json').read_text()) if (r/'execution.json').exists() else {}
    completed = datetime.now(timezone.utc).isoformat()
    simulation_completed = (previous.get('simulation_completed_utc') or (previous.get('completed_utc') if previous.get('action')=='fresh_full_suite' else None)) if args.analyze_existing else completed
    write_json(r/'execution.json',dict(completed_utc=completed,simulation_completed_utc=simulation_completed,platform=platform.platform(),versions=versions,
        action='audit_existing_trajectories' if args.analyze_existing else 'fresh_full_suite',video_requested=not args.no_video,
        independent_constraint_solver='Pinocchio constraintDynamics',mujoco_used_for_new_simulation=False,
        reference_interval_s=.02,simulation_cases=CASES))
    if not args.no_video:
        from stewart.pin_render import render_video
        render_video(ROOT)
    from stewart.pin_report import build_report
    build_report(ROOT); hash_manifest(ROOT)
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['passed_count']}/{result['check_count']} aggregate gates.")
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__':
    main()
