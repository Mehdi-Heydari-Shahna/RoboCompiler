"""Extract labelled legacy evidence from a rerun of the unchanged v22 MuJoCo package.

This script belongs to the *legacy* MuJoCo environment (the supplied v22
``requirements_validated.txt``: numpy 2.3.5, scipy 1.15.3, mujoco 3.3.7,
pin 3.8.0).  It is never imported by the Pinocchio pipeline.

Procedure used for this release (see PROVENANCE.json)::

    # fresh copy of the supplied v22 package, unchanged
    cd Kangaroo_RoboIR_full_body_v22
    python run_kangaroo_v22.py            # baseline + reference + 8 contact runs + gates
    python extract_legacy_evidence.py --v22-root Kangaroo_RoboIR_full_body_v22 --log mj_legacy.log

The full-resolution v22 trajectories (25 us / 12.5 us histories, up to
117 MB each) are not shipped.  For every contact case this script stores the
v22 5 ms samples (pelvis pose, all 76 coordinates in CMG order, drive forces,
commands, foot forces/positions, COM, push) and the v22 per-step history
decimated to 1 ms, plus the unchanged v22 JSON outputs.  MuJoCo is used here
only to read joint addresses of the v22-generated models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ['landing_nominal', 'landing_refined', 'landing_low_friction', 'landing_higher_drop_push',
         'landing_slow_actuators', 'negative_no_contact', 'negative_no_loops_contact', 'negative_passive']
REPORTS = ['full_validation.json', 'contact_validation.json', 'complete_validation_v22.json',
           'contact_reference.json', 'contact_urdf_export.json', 'reference.json', 'instantaneous.json',
           'motion_summary.json', 'floating_motion_summary.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(v22, name, output):
    import mujoco
    sys_path = str(v22)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from contact_task import HISTORY_COLUMNS
    cmg = json.loads((v22 / 'data/whole_body_cmg.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(v22 / 'models' / f'{name}.xml'))
    address = np.array([model.jnt_qposadr[model.joint(j).id] for j in cmg['coordinate_ids']])
    source = v22 / 'results' / f'{name}.npz'
    summary = json.loads((v22 / 'results' / f'{name}.json').read_text())
    with np.load(source, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    dt = float(summary['timestep_s'])
    stride = max(1, round(1e-3 / dt))
    history = a['history'][::stride]
    if not np.allclose(history[:, 0], np.round(history[:, 0] / 1e-3) * 1e-3, atol=1e-9):
        raise ValueError('History decimation is not on the 1 ms grid')
    q = a['q']
    quat = q[:, 3:7][:, [1, 2, 3, 0]]  # MuJoCo w,x,y,z -> x,y,z,w
    arrays = dict(time=a['time'], base_position=q[:, :3], base_quaternion_xyzw=quat,
                  coordinates=q[:, address], coordinate_ids=np.asarray(cmg['coordinate_ids']),
                  motor_force_N=a['act'], command_N=a['command'], foot_force_N=a['foot_force'],
                  foot_position_m=a['foot_position'], com_m=a['com'], push_N=a['push'],
                  contact_count=a['contact_count'], history_1ms=history,
                  history_columns=np.asarray(HISTORY_COLUMNS), timestep_s=np.asarray(dt),
                  source_npz_sha256=np.asarray(sha(source)), engine=np.asarray(f'MuJoCo {mujoco.__version__}'))
    np.savez_compressed(output / f'{name}.npz', **arrays)
    shutil.copy(v22 / 'results' / f'{name}.json', output / f'{name}.json')
    audit = v22 / 'results' / f'{name}_audit.json'
    if audit.is_file():
        shutil.copy(audit, output / f'{name}_audit.json')
    return dict(source_npz=source.name, source_npz_bytes=source.stat().st_size, source_npz_sha256=sha(source),
                source_json_sha256=sha(v22 / 'results' / f'{name}.json'), timestep_s=dt,
                samples_5ms=int(len(a['time'])), history_rows_1ms=int(len(history)),
                extracted_npz_sha256=sha(output / f'{name}.npz'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v22-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    v22, output = args.v22_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {name: extract(v22, name, output) for name in CASES}
    copied = {}
    for report in REPORTS:
        path = v22 / 'results' / report
        if path.is_file():
            shutil.copy(path, output / report)
            copied[report] = sha(path)
    if args.log:
        shutil.copy(args.log, output / 'mujoco_v22_console.log')
    manifest = json.loads((v22 / 'MANIFEST_SHA256.json').read_text())['files']
    code = {p: sha(v22 / p) for p in manifest if p.endswith('.py') or p.startswith('data/')}
    unchanged = all(sha(v22 / p) == manifest[p] for p in code)
    verdict = json.loads((v22 / 'results/complete_validation_v22.json').read_text())
    versions = {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'mujoco', 'pin']}
    record = dict(
        label='LEGACY EVIDENCE: unchanged v22 MuJoCo pipeline rerun; not produced by the Pinocchio backend.',
        v22_verdict=dict(status=verdict['status'], gates_passed=verdict['gates_passed'],
                         gates_total=verdict['gates_total']),
        command='python run_kangaroo_v22.py (no arguments: baseline, reference, 8 contact runs, gates, URDF export)',
        v22_code_and_data_match_supplied_manifest=unchanged, checked_files=len(code),
        python=platform.python_version(), platform=platform.platform(), versions=versions,
        cases=records, copied_reports=copied,
        stored_content='v22 5 ms samples (pose, 76 coordinates in CMG order, drive force/command, foot force/position, '
                       'COM, push) and the v22 per-step history decimated to 1 ms; full-resolution files are not '
                       'shipped (hashes recorded).')
    (output / 'PROVENANCE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(v22_verdict=record['v22_verdict'], unchanged=unchanged), indent=2))


if __name__ == '__main__':
    main()
