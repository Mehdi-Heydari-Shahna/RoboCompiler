"""Extract compact arm traces from the ORIGINAL MuJoCo soil runs (no MuJoCo import).

The original ``run_soil.py`` saves native ``qpos`` at 25 Hz.  MuJoCo assigns
position addresses to joints in depth-first XML body order (free joint: 7,
ball: 4, hinge/slide: 1).  This script reproduces that address map by parsing
the saved ``scene.xml`` and validates the result against two independent
facts recorded by the original run:

1. at t = 0 the 23 arm coordinates equal the original mission's
   ``initial_tree`` (the original sets ``d.qpos[qi] = mission.initial_tree``);
2. the bucket-lip position reconstructed from the extracted free-base pose and
   the independent source-record forward kinematics equals the recorded
   ``lip_position`` (the lip path crosses no loop cut, so agreement is exact to
   round-off).

When the case directory is a regeneration of a shipped case (the shipped
``soil_demo`` and ``soil_final`` folders contain no ``trajectory.npz``), pass the
shipped case folder with ``--shipped-case``: the SHA-256 of ``scene.xml``,
``settled_state.npz`` and ``last_state.npz`` are compared, every common
``report.json`` field is compared, and all differences are recorded (only
wall-clock timings and the run name are expected to differ).  The SHA-256 of
the regenerated ``trajectory.npz`` is also compared with the entry for the
omitted file in the original ``RELEASE_SHA256.json``.

Usage (Pinocchio environment, from the package root):
    python tools/extract_mujoco_traces.py CASE_DIR OUTPUT.npz [--shipped-case SHIPPED_DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator_pin import sentinel, paths  # noqa: E402

sentinel.install()
paths.add_original_to_path()
_V26_RESULTS_EXISTED = (paths.V26 / 'results').exists()  # the original project.py creates it on import


def qpos_addresses(scene_xml):
    root = ET.parse(scene_xml).getroot()
    addresses, cursor = {}, 0

    def visit(body):
        nonlocal cursor
        for child in body:
            if child.tag == 'freejoint':
                addresses[child.get('name')] = (cursor, 'free')
                cursor += 7
            elif child.tag == 'joint':
                kind = child.get('type', 'hinge')
                size = {'free': 7, 'ball': 4, 'hinge': 1, 'slide': 1}[kind]
                addresses[child.get('name')] = (cursor, kind)
                cursor += size
        for child in body:
            if child.tag == 'body':
                visit(child)

    visit(root.find('worldbody'))
    return addresses, cursor


EXPECTED_RUN_DIFFERENCES = ('elapsed_wall_s', 'args.name', 'arm_controller.measured_control_mean_wall_s',
                            'arm_controller.measured_control_max_wall_s')


def _flatten(value, prefix=''):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten(item, f'{prefix}{key}.'))
    else:
        out[prefix[:-1]] = value
    return out


def compare_reports(regenerated, shipped):
    a, b = _flatten(shipped), _flatten(regenerated)
    common = sorted(set(a) & set(b))
    differing = [k for k in common if a[k] != b[k]]
    return dict(compared_fields=len(common), differing_fields=differing,
                unexpected_differences=[k for k in differing if k not in EXPECTED_RUN_DIFFERENCES],
                fields_only_in_regenerated=sorted(set(b) - set(a)),
                fields_only_in_shipped=sorted(set(a) - set(b)))


def extract(case_dir, output, shipped_case=None):
    from excavator_pin.engine import context
    from digging_path import LIP
    case_dir = Path(case_dir)
    trace_path = case_dir / 'trajectory.npz'
    scene = case_dir / 'scene.xml'
    report = json.loads((case_dir / 'report.json').read_text())
    cmg, mapping, e, a, r, inverse = context()
    addresses, nq = qpos_addresses(scene)
    with np.load(trace_path, allow_pickle=False) as f:
        tr = {k: f[k] for k in f.files}
    if tr['qpos'].shape[1] != nq:
        raise ValueError(f'qpos width {tr["qpos"].shape[1]} differs from parsed {nq}')
    arm_index = np.array([addresses[j][0] for j in e.tree_ids])
    arm = tr['qpos'][:, arm_index]
    base_adr = addresses['floating_undercarriage'][0]
    base = tr['qpos'][:, base_adr:base_adr + 7]
    record = next(j for j in cmg['joints'] if j['id'] == 'fixed_world_base')
    T0 = np.asarray(record['T_BJ']) @ np.linalg.inv(np.asarray(record['T_FJ']))
    lip = np.empty((len(arm), 3))
    for k, (q, b) in enumerate(zip(arm, base)):
        native = np.eye(4)
        native[:3, :3] = Rotation.from_quat(b[[4, 5, 6, 3]]).as_matrix()
        native[:3, 3] = b[:3]
        virtual = native @ np.linalg.inv(T0)
        pose = virtual @ e.source.evaluate(q, np.zeros(23), [0, 0, -9.81])['poses']['body_56']
        lip[k] = pose[:3, :3] @ LIP + pose[:3, 3]
    lip_error = np.linalg.norm(lip - tr['lip_position'], axis=1)
    arrays = dict(time=tr['time'], phase=tr['phase'], tree_ids=np.array(e.tree_ids),
                  arm_q=arm, independent=arm[:, e.active], base_qpos=base,
                  lip_position=tr['lip_position'], lip_reconstructed=lip,
                  soil_bucket_force=tr['soil_bucket_force'], bucket_mass=tr['bucket_mass'],
                  lifted_mass=tr['lifted_mass'], deposited_mass=tr['deposited_mass'],
                  spill_mass=tr['spill_mass'], source_mass=tr['source_mass'], soil_speed=tr['soil_speed'],
                  tracking_error=tr['tracking_error'], closure_error=tr['closure_error'],
                  pressure_max=tr['pressure_max'], ctrl=tr['ctrl'])
    np.savez_compressed(output, **arrays)
    meta = dict(source_case=case_dir.name,
                source_trajectory_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                source_scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                source_report_sha256=hashlib.sha256((case_dir / 'report.json').read_bytes()).hexdigest(),
                extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                samples=int(len(arm)), qpos_width=int(nq), status=report['status'],
                events=report['events'], failures=report['failures'], args=report['args'],
                validation=dict(initial_arm_max_abs_difference=float(np.max(np.abs(arm[0] - np.asarray(
                    _initial_tree(e, r, report['args']))))),
                                lip_reconstruction_max_m=float(lip_error.max()),
                                lip_reconstruction_median_m=float(np.median(lip_error))),
                output_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest())
    if shipped_case is not None:
        shipped_case = Path(shipped_case)
        files = {}
        for name in ('scene.xml', 'settled_state.npz', 'last_state.npz'):
            shipped = hashlib.sha256((shipped_case / name).read_bytes()).hexdigest()
            regenerated = hashlib.sha256((case_dir / name).read_bytes()).hexdigest()
            files[name] = dict(shipped_sha256=shipped, regenerated_sha256=regenerated,
                               identical=shipped == regenerated)
        meta['shipped_case'] = shipped_case.name
        meta['shipped_file_comparison'] = files
        # The original release manifest lists a trajectory that the source package omitted.
        manifest = shipped_case.parents[1] / 'RELEASE_SHA256.json'
        if manifest.exists():
            listed = json.loads(manifest.read_text()).get(f'outputs/{shipped_case.name}/trajectory.npz')
            meta['release_manifest_trajectory'] = dict(
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                listed_sha256=listed, regenerated_sha256=meta['source_trajectory_sha256'],
                identical=listed == meta['source_trajectory_sha256'])
        meta['shipped_report_sha256'] = hashlib.sha256((shipped_case / 'report.json').read_bytes()).hexdigest()
        meta['shipped_report_comparison'] = compare_reports(
            report, json.loads((shipped_case / 'report.json').read_text()))
    Path(str(output).replace('.npz', '.json')).write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def _initial_tree(e, r, args):
    from soil_mission import SoilMission
    mission = SoilMission(e, r, cut_depth=args.get('cut_depth', .11),
                          cut_policy=args.get('cut_policy', 'position-only'))
    return mission.initial_tree


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir')
    parser.add_argument('output')
    parser.add_argument('--shipped-case', default=None)
    a = parser.parse_args()
    try:
        print(json.dumps(extract(a.case_dir, a.output, a.shipped_case), indent=2))
    finally:
        created = paths.V26 / 'results'
        if not _V26_RESULTS_EXISTED and created.is_dir() and not any(created.iterdir()):
            created.rmdir()
