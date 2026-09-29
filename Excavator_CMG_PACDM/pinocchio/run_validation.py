#!/usr/bin/env python3
"""Complete Pinocchio-backend validation of the RoboCompiler excavator soil prototype.

MuJoCo is never imported: a fail-closed sentinel is installed before any
preserved source module is loaded, and the validated environment does not
install MuJoCo.  From the package root::

    python run_validation.py               # full run: mechanics, route, 8 cases, cross-engine, video, report
    python run_validation.py --workers 1   # serial case execution
    python run_validation.py --audit       # keep existing case results; recompute every check and report
    python run_validation.py --skip-render # no video (every other gate still runs)

Exit status 0 only if every declared gate passes.
"""
from __future__ import annotations

import os

for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')   # deterministic single-threaded BLAS; no oversubscription

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from excavator_pin import sentinel  # noqa: E402

sentinel.install()
from excavator_pin import paths  # noqa: E402

V26_RESULTS_EXISTED = (paths.V26 / 'results').exists()
MANIFEST = ROOT / 'MANIFEST_SHA256.json'
RENDERED_CASES = ('nominal', 'stress_load')
EXCLUDED_PARTS = {'.venv', 'venv', '__pycache__', '.git', '.pytest_cache'}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(value, limit, relation, unit, description=''):
    value = float(value)
    passed = {'<=': value <= limit, '>=': value >= limit, '==': value == limit}[relation]
    return dict(passed=bool(passed), value=value, limit=float(limit), relation=relation, unit=unit,
                description=description)


def original_tree_status():
    manifest = json.loads((ROOT / 'ORIGINAL_TREE_SHA256.json').read_text())
    tree = paths.ORIGINAL
    mismatched = [rel for rel, digest in manifest['files'].items()
                  if not (tree / rel).is_file() or sha256(tree / rel) != digest]
    present = {str(p.relative_to(tree)).replace('\\', '/') for p in tree.rglob('*') if p.is_file()}
    extra = sorted(present - set(manifest['files']))
    expected_dirs = {str(Path(rel).parents[k]).replace('\\', '/') for rel in manifest['files']
                     for k in range(len(Path(rel).parents) - 1)}
    dirs = {str(p.relative_to(tree)).replace('\\', '/') for p in tree.rglob('*') if p.is_dir()}
    extra_dirs = sorted(dirs - expected_dirs)
    release = json.loads((tree / 'RELEASE_SHA256.json').read_text())
    release_absent = sorted(k for k in release if not (tree / k).is_file())
    release_bad = sorted(k for k in release if (tree / k).is_file() and sha256(tree / k) != release[k])
    return dict(archive_sha256=manifest['archive_sha256'], files=manifest['file_count'],
                mismatched=mismatched, extra_files=extra, extra_directories=extra_dirs,
                release_manifest_listed=len(release),
                release_manifest_absent_from_package=release_absent, release_manifest_mismatched=release_bad)


def remove_import_side_effect():
    """Importing the original v26/project.py creates an empty v26/results directory."""
    created = paths.V26 / 'results'
    if not V26_RESULTS_EXISTED and created.is_dir() and not any(created.iterdir()):
        created.rmdir()


def sentinel_self_test():
    """In a fresh interpreter: MuJoCo attributes and submodules must be refused."""
    code = ('import sys; sys.path.insert(0, %r)\n'
            'from excavator_pin import sentinel; sentinel.install()\n'
            'import mujoco\n'
            'try:\n    mujoco.MjModel\n    print("ATTRIBUTE_ALLOWED")\n'
            'except sentinel.MuJoCoAccessError:\n    print("ATTRIBUTE_REFUSED")\n'
            'try:\n    import mujoco.viewer\n    print("SUBMODULE_ALLOWED")\n'
            'except ImportError:\n    print("SUBMODULE_REFUSED")\n') % str(ROOT)
    out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=120)
    return dict(stdout=out.stdout.split(), returncode=out.returncode,
                passed=out.returncode == 0 and out.stdout.split() == ['ATTRIBUTE_REFUSED', 'SUBMODULE_REFUSED'])


def environment():
    versions = {}
    for name in ('numpy', 'scipy', 'pin', 'vtk', 'matplotlib', 'imageio', 'imageio-ffmpeg', 'pillow',
                 'cmeel-urdfdom', 'cmeel-tinyxml2'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        importlib.metadata.version('mujoco')
        mujoco_installed = True
    except importlib.metadata.PackageNotFoundError:
        mujoco_installed = False
    import numpy
    import pinocchio
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                cpu_count=os.cpu_count(), numpy=numpy.__version__, pinocchio=pinocchio.__version__,
                scipy=versions['scipy'], vtk=versions['vtk'], packages=versions,
                mujoco_installed=mujoco_installed,
                blas_threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                            'MKL_NUM_THREADS')})


def _case_worker(args):
    name, results_dir = args
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    sys.dont_write_bytecode = True
    from excavator_pin.simulation import run_case
    started = time.time()
    with open(Path(results_dir) / f'{name}.log', 'w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        report = run_case(name, results_dir)
    return name, report['status'], time.time() - started


def run_cases(names, results_dir, workers):
    # Most expensive first so that two workers finish together.
    cost = {'quarter_step': 4, 'half_step': 2, 'initial_offset_half_step': 2, 'stress_load_position_only': .4,
            'stiff_soil_regularization': .3}
    ordered = sorted(names, key=lambda n: -cost.get(n, 1))
    if workers <= 1:
        for name in ordered:
            print('case', *_case_worker((name, results_dir)), flush=True)
        return
    from multiprocessing import get_context
    with get_context('spawn').Pool(workers) as pool:
        for name, status, wall in pool.imap_unordered(_case_worker, [(n, str(results_dir)) for n in ordered]):
            print(f'case {name}: {status} ({wall:.0f} s)', flush=True)


def write_manifest():
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if (not path.is_file() or path == MANIFEST or path.suffix == '.pyc'
                or EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)):
            continue
        files[str(path.relative_to(ROOT)).replace('\\', '/')] = sha256(path)
    MANIFEST.write_text(json.dumps(dict(algorithm='sha256', excluded=sorted(EXCLUDED_PARTS | {'*.pyc'}),
                                        file_count=len(files), files=files), indent=2) + '\n', encoding='utf-8')
    return len(files)


def main(argv=None):
    # Provenance of the preserved original package BEFORE any original module is imported.
    before = original_tree_status()
    from excavator_pin.simulation import CASES
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--audit', action='store_true', help='reuse existing case results')
    parser.add_argument('--skip-render', action='store_true')
    parser.add_argument('--results', default=str(paths.RESULTS))
    args = parser.parse_args(argv)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    console = open(results / 'validation_console.log', 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, console)
    started = time.time()
    extra = {}
    print(f'Excavator Pinocchio validation — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} UTC')
    env = environment()
    print(f"Python {env['python']} · Pinocchio {env['pinocchio']} · NumPy {env['numpy']} · MuJoCo installed: "
          f"{env['mujoco_installed']}")

    # 1. provenance of the preserved original package, before the run
    extra['provenance.original_tree_unchanged_before_run'] = check(
        len(before['mismatched']) + len(before['extra_files']) + len(before['extra_directories']), 0, '==',
        'files or directories', 'every original/ file matches its recorded SHA-256 '
        '(ORIGINAL_TREE_SHA256.json); no extra file or directory')
    extra['provenance.original_release_manifest_consistent'] = check(
        len(before['release_manifest_mismatched']), 0, '==', 'files',
        "present files match the original package's own RELEASE_SHA256.json")
    extra['provenance.PACDM_core_unchanged'] = check(int(sha256(paths.PACDM_CORE) == paths.PACDM_SHA256), 1, '==',
                                                     'boolean', 'pacdm.py SHA-256 equals the Panda/excavator core')
    selftest = sentinel_self_test()
    extra['provenance.mujoco_sentinel_self_test'] = check(int(selftest['passed']), 1, '==', 'boolean',
                                                          'fresh interpreter refuses mujoco attributes and submodules')
    print('provenance:', json.dumps({k: v['passed'] for k, v in extra.items()}))

    # 2. mechanics and route witnesses (a crash is recorded as a failed check, never hides the verdict)
    from excavator_pin import mechanics, route, crossengine, gates, report
    import traceback

    def stage(label, function):
        started_stage = time.time()
        try:
            result = function()
            print(f'{label}: done ({time.time() - started_stage:.0f} s)', flush=True)
            return result
        except Exception:
            text = traceback.format_exc()
            print(f'{label}: FAILED\n{text}', flush=True)
            extra[f'stages.{label}_completed'] = check(0, 1, '==', 'boolean', text[-2000:])
            return None

    mech = stage('mechanics', lambda: mechanics.run(results / 'mechanics.json'))
    if mech:
        print(f"mechanics: {mech['passed_checks']}/{mech['total_checks']}", flush=True)
    rte = stage('route', lambda: route.run(results / 'route.json'))
    if rte:
        print(f"route: {rte['passed_checks']}/{rte['total_checks']}", flush=True)

    # 3. closed-loop cases
    if not args.audit:
        stage('cases', lambda: run_cases(list(CASES), results, args.workers))
    else:
        missing = [n for n in CASES if not (results / f'{n}.json').exists() or not (results / f'{n}.npz').exists()]
        if missing:
            raise SystemExit(f'--audit requires existing case results; missing: {missing}')
        print('cases: audit mode, existing results reused', flush=True)

    # 4. cross-engine comparison, videos and aggregation
    cross = stage('crossengine', lambda: crossengine.run(results, results / 'crossengine.json'))
    if cross:
        print(f"crossengine: {cross['passed_checks']}/{cross['total_checks']}", flush=True)
    render_meta = None
    if not args.skip_render:
        def videos():
            from excavator_pin import render
            metas = {}
            for case in RENDERED_CASES:
                meta = render.render(case, results)
                metas[case] = meta
                extra[f'render.{case}_video_decodes_all_frames'] = check(
                    int(meta['frames_decoded'] == meta['frames_written'] and meta['frames_written'] > 0),
                    1, '==', 'boolean', 'written MP4 re-read with the expected frame count')
                extra[f'render.{case}_video_decodes_all_frames'].update(category='operational', kind='consistency')
                extra[f'render.{case}_visual_frames_match_plant_FK'] = check(
                    meta['visual_frame_check']['max_abs_difference'], 1e-9, '<=', 'm or matrix entry',
                    'visual_actual.xml tree evaluated independently vs plant FK body frames')
                print(f"render {case}: {meta['frames_decoded']} frames", flush=True)
            return dict(metas['nominal'], additional={k: v for k, v in metas.items() if k != 'nominal'})
        render_meta = stage('render', videos)
    remove_import_side_effect()
    after = original_tree_status()
    extra['provenance.original_tree_unchanged_after_run'] = check(
        len(after['mismatched']) + len(after['extra_files']) + len(after['extra_directories']), 0, '==',
        'files or directories', 'no original file or directory created or modified by the run (after removing '
        'the empty v26/results directory created by importing the original project.py)')
    extra['provenance.no_mujoco_access_in_runner'] = check(
        sentinel.report()['attempted_access_count'] + len(sentinel.report()['real_mujoco_submodules_loaded']),
        0, '==', 'count', 'mechanics, route, cross-engine, gates and video rendering run in this process')
    for key, value in extra.items():
        value.setdefault('category', key.split('.', 1)[0])
        value.setdefault('kind', 'evidence')
    try:
        validation = gates.aggregate(results, extra)
    except Exception:  # never leave the run without a verdict file
        import traceback
        validation = dict(passed=False, passed_checks=0, total_checks=1, checks={'aggregate.evaluated': check(
            0, 1, '==', 'boolean', traceback.format_exc())}, details={})
    validation['details'].update(original_tree=before, sentinel_self_test=selftest,
                                 v26_results_directory_note=(
                                     'Importing the preserved v26/project.py creates an empty v26/results '
                                     'directory (import side effect). The runner removes it again if it did '
                                     'not exist before and is still empty; no file is involved.'))
    env['finished_utc'] = f'{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}'
    env['total_wall_s'] = time.time() - started
    env['audit_mode'] = bool(args.audit)
    validation['environment'] = env
    (results / 'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False) + '\n',
                                             encoding='utf-8')
    (results / 'execution_environment.json').write_text(json.dumps(env, indent=2) + '\n', encoding='utf-8')
    try:
        report.write(results, env, render_meta)
    except Exception:  # a missing report is a failed validation, recorded in validation.json
        text = traceback.format_exc()
        print(f'report: FAILED\n{text}', flush=True)
        validation['checks']['stages.report_completed'] = dict(
            check(0, 1, '==', 'boolean', text[-2000:]), category='stages', kind='evidence')
        validation['passed'] = False
        validation['total_checks'] += 1
        (results / 'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False) + '\n',
                                                 encoding='utf-8')
    failed = [k for k, c in validation['checks'].items() if not c['passed']]
    print(f"VALIDATION {'PASSED' if validation['passed'] else 'FAILED'}: "
          f"{validation['passed_checks']}/{validation['total_checks']} checks "
          f"(evidence {validation.get('evidence_checks')}, consistency {validation.get('consistency_checks')})",
          flush=True)
    for key in failed:
        print('  FAILED', key, validation['checks'][key]['value'], validation['checks'][key]['relation'],
              validation['checks'][key]['limit'])
    print(f'total wall time {time.time() - started:.0f} s', flush=True)
    sys.stdout = sys.__stdout__
    console.close()
    if results.resolve() == paths.RESULTS.resolve():
        count = write_manifest()
        print(f'manifest: {count} files -> {MANIFEST.name}')
    else:  # a reproduction into another directory must not rewrite the recorded integrity record
        print('manifest: not rewritten (results written outside the package results directory)')
    return 0 if validation['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    finally:
        remove_import_side_effect()
