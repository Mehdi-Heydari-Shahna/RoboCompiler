"""Cross-engine comparison: Pinocchio backend runs versus the original MuJoCo runs.

The MuJoCo evidence is read from compact traces in ``data/`` that were
extracted (without importing MuJoCo) from the original runs; see
``tools/extract_mujoco_traces.py`` and the ``data/*.json`` provenance files.
The constraint sites are read from the original runs' ``scene.xml`` (byte-
identical between the shipped and the regenerated runs).

The two engines do NOT simulate the same physical system:

* MuJoCo: floating undercarriage on articulated tracks, 18 soft ``connect``
  equalities for the nine arm loop cuts, rigid spherical grains (172 in
  ``soil_demo``, 105 in ``soil_final``, none in ``face_empty``), contact solver;
* Pinocchio backend: undercarriage fixed to the world, exact native loop
  constraints, declared smooth soil surrogate (no grains, no contact).

Gated cross-engine checks are therefore restricted to what must agree when the
shared parts are the same:

1. **Mechanism model.** MuJoCo's 36 loop-constraint sites are the plant's
   constraint points (joint origin and 0.1 m along the cut axis).  The mass,
   COM and COM inertia of the 24 arm bodies in
   each MuJoCo scene equal the plant's CMG values (the undercarriage body_53
   differs because MuJoCo moves part of its mass into the track bodies; it is
   fixed here and does not enter the dynamics).  For every recorded MuJoCo
   sample of all three runs,
   the arm-loop constraint violation that MuJoCo recorded natively
   (``max |efc_pos|`` of its 18 arm ``connect`` rows) is recomputed from the
   recorded configuration with Pinocchio plant forward kinematics, MuJoCo's own
   constraint sites and the recorded floating-base pose.  The bucket-lip
   position is recomputed the same way.  Both must agree to round-off.
2. **Qualitative task outcome.** Both no-soil runs record the original failure
   "Lift completed with less than 1 kg in bucket"; both soil pairs complete all
   eleven phases and deposit more than the mission's 1 kg threshold.

Disclosure of a post-data change: a 0.035 rad agreement gate on the
phase-aligned commanded coordinates of the no-soil pair was declared before the
MuJoCo traces were examined.  Independent review showed it is ill-posed: the
floating MuJoCo undercarriage settles about 8 cm and pitches about 0.08 rad
in phase 0 and pitches back about 0.07 rad with 7-10 cm of motion in phase 1,
and MuJoCo's own tracking error exceeds 0.035 rad inside phases 0-2.
The gate was withdrawn; the differences are reported below, not claimed as
agreement.  Payload, forces, phase durations and guard timing are reported
only: the surrogate is not calibrated to the grain bed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import paths

TRACKING_GUARD_RAD = .035          # SoilMission.update: tracking_error < .035 (reported reference line)
REQUIRED_PAYLOAD_KG = 1.           # SoilMission.required_payload_kg
KINEMATIC_TOLERANCE_M = 1e-12      # round-off level for recomputed MuJoCo quantities
PAIRS = {
    'no_soil': dict(pinocchio='empty_bed', mujoco='face_empty'),
    'filled_bed': dict(pinocchio='nominal', mujoco='soil_demo'),
    'exposed_face': dict(pinocchio='exposed_face', mujoco='soil_final'),
}
LIFT_FAILURE = 'Lift completed with less than 1 kg in bucket'


def _entries(events, count):
    entries = [0.]
    for event in events[:count - 1]:
        entries.append(float(event['time_s']))
    return entries


def phase_aligned(pin_time, pin_values, pin_events, mj_time, mj_values, mj_phase, mj_events, phases):
    """Differences at MuJoCo samples, each phase measured from its own entry time."""
    pin_entry = _entries(pin_events, phases + 1)
    mj_entry = _entries(mj_events, phases + 1)
    per_phase = []
    for p in range(min(len(pin_entry), len(mj_entry))):
        mask = mj_phase == p
        if not np.any(mask):
            continue
        tau = mj_time[mask] - mj_entry[p]
        target = pin_entry[p] + tau
        # compare only inside the Pinocchio phase window and inside the trace
        end = pin_entry[p + 1] if p + 1 < len(pin_entry) else pin_time[-1]
        keep = (target <= min(end, pin_time[-1]) + 1e-9)
        if not np.any(keep):
            continue
        interpolated = np.column_stack([np.interp(target[keep], pin_time, pin_values[:, j])
                                        for j in range(pin_values.shape[1])])
        diff = np.abs(interpolated - mj_values[mask][keep])
        per_phase.append(dict(phase=p, samples=int(keep.sum()), max_abs=diff.max(axis=0).tolist(),
                              max=float(diff.max())))
    return per_phase


def _load_pin(results_dir, name):
    report = json.loads((Path(results_dir) / f'{name}.json').read_text())
    with np.load(Path(results_dir) / f'{name}.npz', allow_pickle=False) as f:
        trace = {k: f[k] for k in f.files}
    return report, trace


def _load_mj(name):
    meta = json.loads((paths.DATA / f'mujoco_{name}.json').read_text())
    with np.load(paths.DATA / f'mujoco_{name}.npz', allow_pickle=False) as f:
        trace = {k: f[k] for k in f.files}
    return meta, trace


def mujoco_inertial_consistency(case):
    """Arm-body inertials written in the MuJoCo scene versus the CMG used by the plant."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from .engine import load_inputs
    cmg, _ = load_inputs()
    bodies = {b['id']: b for b in cmg['bodies'] if b['kind'] == 'rigid_body'}
    root = ET.parse(paths.ORIGINAL / 'outputs' / case / 'scene.xml').getroot()
    worst, excluded, compared = dict(mass_kg=0., com_m=0., inertia_relative=0.), {}, 0
    for body in root.iter('body'):
        name = body.get('name')
        if name not in bodies:
            continue
        inertial = body.find('inertial')
        q = np.fromstring(inertial.get('quat', '1 0 0 0'), sep=' ')
        rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        inertia = rotation @ np.diag(np.fromstring(inertial.get('diaginertia'), sep=' ')) @ rotation.T
        record = bodies[name]
        reference = np.asarray(record['inertia_com_kg_m2'], dtype=float)
        diff = dict(mass_kg=abs(float(inertial.get('mass')) - float(record['mass_kg'])),
                    com_m=float(np.max(np.abs(np.fromstring(inertial.get('pos'), sep=' ')
                                              - np.asarray(record['com_m'], dtype=float)))),
                    inertia_relative=float(np.max(np.abs(inertia - reference)) / max(1., np.max(np.abs(reference)))))
        if name == 'body_53':  # undercarriage: fixed here; MuJoCo moves mass into the track bodies
            excluded[name] = diff
            continue
        compared += 1
        worst = {k: max(worst[k], diff[k]) for k in worst}
    return dict(case=case, arm_bodies_compared=compared, max_difference=worst, excluded=excluded)


def mujoco_connect_sites(case):
    """MuJoCo loop-constraint sites versus the plant's two constraint points (same body frames)."""
    import xml.etree.ElementTree as ET
    from .engine import load_inputs, context
    cmg, _ = load_inputs()
    _, _, e, _, _, _ = context()
    joints = {j['id']: j for j in cmg['joints']}
    root = ET.parse(paths.ORIGINAL / 'outputs' / case / 'scene.xml').getroot()
    sites = {}
    for body in root.iter('body'):
        for site in body.findall('site'):
            if site.get('name', '').startswith('loop_'):
                sites[site.get('name')] = (body.get('name'), np.fromstring(site.get('pos'), sep=' '))
    worst, compared = 0., 0
    for cut in e.cut_ids:
        joint = joints[cut]
        axis = np.asarray(joint['axis'], dtype=float)
        for k, lever in enumerate((0., .1)):   # the plant's two points (ExcavatorPinModel lever 0.1 m)
            for side, key, body_key in (('base', 'T_BJ', 'base_body'), ('follower', 'T_FJ', 'follower_body')):
                transform = np.asarray(joint[key], dtype=float)
                point = transform[:3, 3] + transform[:3, :3] @ (lever * axis)
                body, site = sites[f'loop_{cut}_{side}_{k}']
                if body != joint[body_key]:
                    return dict(case=case, error=f'site loop_{cut}_{side}_{k} on {body}, expected {joint[body_key]}',
                                max_distance_m=float('inf'), sites_compared=compared)
                worst = max(worst, float(np.linalg.norm(site - point)))
                compared += 1
    return dict(case=case, sites_compared=compared, max_distance_m=worst)


def base_motion_by_phase(case):
    """Floating-base translation and rotation within each phase of a MuJoCo run (from data/)."""
    from scipy.spatial.transform import Rotation
    _, trace = _load_mj(case)
    base, phase = trace['base_qpos'], trace['phase'].astype(int)
    rows = []
    for p in range(int(phase.max()) + 1):
        mask = np.flatnonzero(phase == p)
        if mask.size == 0:
            continue
        start = Rotation.from_quat(base[mask[0]][[4, 5, 6, 3]])
        rotation = max((start.inv() * Rotation.from_quat(base[i][[4, 5, 6, 3]])).magnitude() for i in mask)
        translation = float(np.max(np.linalg.norm(base[mask, :3] - base[mask[0], :3], axis=1)))
        rows.append(dict(phase=p, rotation_within_phase_rad=float(rotation),
                         translation_within_phase_m=translation))
    return rows


def mujoco_kinematic_consistency(case):
    """Recompute MuJoCo's recorded loop gap and lip from its recorded state with the Pinocchio plant."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from .engine import load_inputs, context
    from .plant import ExcavatorPinModel, LIP_LOCAL
    cmg, _ = load_inputs()
    _, _, e, _, _, _ = context()
    model = ExcavatorPinModel(cmg, e.cut_ids)
    perm = np.array([model.index[j] for j in e.tree_ids])
    record = next(j for j in cmg['joints'] if j['id'] == 'fixed_world_base')
    T0 = np.asarray(record['T_BJ']) @ np.linalg.inv(np.asarray(record['T_FJ']))
    scene = paths.ORIGINAL / 'outputs' / case / 'scene.xml'
    root = ET.parse(scene).getroot()
    sites = {}
    for body in root.iter('body'):
        for site in body.findall('site'):
            if site.get('name', '').startswith('loop_'):
                sites[site.get('name')] = (body.get('name'), np.fromstring(site.get('pos'), sep=' '))
    pairs = [(c.get('site1'), c.get('site2')) for c in root.iter('connect')
             if c.get('name', '').startswith('closure_')]
    _, trace = _load_mj(case)
    gap_error, lip_error, gaps = 0., 0., []
    for q_ctrl, recorded_gap, base, recorded_lip in zip(trace['arm_q'], trace['closure_error'],
                                                        trace['base_qpos'], trace['lip_position']):
        q = np.empty(model.model.nq)
        q[perm] = q_ctrl
        poses = model.body_poses(q)
        native = np.eye(4)
        native[:3, :3] = Rotation.from_quat(base[[4, 5, 6, 3]]).as_matrix()
        native[:3, 3] = base[:3]
        virtual = native @ np.linalg.inv(T0)
        gap = 0.
        for s1, s2 in pairs:
            (b1, p1), (b2, p2) = sites[s1], sites[s2]
            w1 = poses[b1][:3, :3] @ p1 + poses[b1][:3, 3]
            w2 = poses[b2][:3, :3] @ p2 + poses[b2][:3, 3]
            gap = max(gap, float(np.max(np.abs(virtual[:3, :3] @ (w1 - w2)))))
        gaps.append(gap)
        gap_error = max(gap_error, abs(gap - float(recorded_gap)))
        bucket = virtual @ poses['body_56']
        lip = bucket[:3, :3] @ LIP_LOCAL + bucket[:3, 3]
        lip_error = max(lip_error, float(np.linalg.norm(lip - recorded_lip)))
    return dict(case=case, samples=len(gaps), arm_connect_pairs=len(pairs),
                scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                recorded_loop_gap_max_m=float(np.max(trace['closure_error'])),
                loop_gap_recomputation_error_m=gap_error, lip_recomputation_error_m=lip_error)


def _base_motion(base_qpos):
    """Translation and tilt of MuJoCo's floating undercarriage relative to t = 0."""
    from scipy.spatial.transform import Rotation
    translation = np.linalg.norm(base_qpos[:, :3] - base_qpos[0, :3], axis=1)
    r0 = Rotation.from_quat(base_qpos[0, [4, 5, 6, 3]])
    angle = np.array([(r0.inv() * Rotation.from_quat(b[[4, 5, 6, 3]])).magnitude() for b in base_qpos])
    return float(translation.max()), float(angle.max())


def compare_pair(results_dir, label, spec):
    pin_report, pin = _load_pin(results_dir, spec['pinocchio'])
    mj_meta, mj = _load_mj(spec['mujoco'])
    mj_time, mj_phase = mj['time'], mj['phase'].astype(int)
    phases = int(mj_phase.max()) + 1
    commanded = phase_aligned(pin['time'], pin['u'][:, :6], pin_report['events'], mj_time,
                              mj['independent'][:, :6], mj_phase, mj_meta['events'], phases)
    lip = phase_aligned(pin['time'], pin['lip'], pin_report['events'], mj_time, mj['lip_position'],
                        mj_phase, mj_meta['events'], phases)
    translation, tilt = _base_motion(mj['base_qpos'])
    pin_events = {e['phase']: e for e in pin_report['events']}
    mj_events = {e['phase']: e for e in mj_meta['events']}
    event_rows = []
    for name in pin_report['phase_names']:
        a, b = pin_events.get(name), mj_events.get(name)
        event_rows.append(dict(phase=name,
                               pinocchio_time_s=None if a is None else a['time_s'],
                               mujoco_time_s=None if b is None else b['time_s'],
                               pinocchio_bucket_mass_kg=None if a is None else a['bucket_mass_kg'],
                               mujoco_bucket_mass_kg=None if b is None else b['bucket_mass_kg'],
                               pinocchio_deposited_kg=None if a is None else a['deposited_mass_kg'],
                               mujoco_deposited_kg=None if b is None else b['deposited_mass_kg']))
    window = mj_time[-1]
    pin_window = pin['time'] <= window + 1e-9
    summary = dict(
        pinocchio_case=spec['pinocchio'], mujoco_case=spec['mujoco'],
        mujoco_source=dict(trajectory_sha256=mj_meta['source_trajectory_sha256'],
                           status=mj_meta['status'], failures=mj_meta['failures'], samples=mj_meta['samples']),
        pinocchio_status=pin_report['status'], pinocchio_failures=pin_report['failures'],
        mujoco_status=mj_meta['status'], mujoco_failures=mj_meta['failures'],
        compared_phases=[row['phase'] for row in commanded],
        commanded_coordinate_difference_rad=dict(max=max(r['max'] for r in commanded), per_phase=commanded),
        lip_position_difference_m=dict(max=max(r['max'] for r in lip), per_phase=lip),
        mujoco_undercarriage_motion=dict(max_translation_m=translation, max_rotation_rad=tilt),
        events=event_rows,
        peak_tracking_error_rad=dict(pinocchio=float(pin['tracking_error'][pin_window].max()),
                                     mujoco=float(mj['tracking_error'].max())),
        peak_bucket_mass_kg=dict(pinocchio=float(pin['payload_mass'].max()), mujoco=float(mj['bucket_mass'].max())),
        final_deposited_kg=dict(pinocchio=float(pin['deposited_mass'][-1]), mujoco=float(mj['deposited_mass'][-1])),
        peak_soil_force_N=dict(pinocchio_surrogate_cutting=float(np.linalg.norm(pin['cutting_force'], axis=1).max()),
                               mujoco_grain_contact_on_bucket=float(np.linalg.norm(mj['soil_bucket_force'],
                                                                                   axis=1).max())),
        mujoco_loop_closure_error_max_m=float(mj['closure_error'].max()),
        mujoco_tracking_error_by_phase_rad=[float(mj['tracking_error'][mj_phase == p].max())
                                            for p in range(phases)],
        mujoco_force_by_phase_N=[dict(median=float(np.median(np.linalg.norm(mj['soil_bucket_force'][mj_phase == p],
                                                                            axis=1))),
                                      max=float(np.linalg.norm(mj['soil_bucket_force'][mj_phase == p], axis=1).max()))
                                 for p in range(phases)],
        mujoco_duration_s=float(mj_time[-1]), mujoco_run_args=mj_meta['args'],
        pinocchio_projection_residual_max_m=float(pin_report['plant_numerics']['projection_residual_m']),
        guard_clock_note=('MuJoCo accumulates d.time in floating point (e.g. 2.0399999999998863); '
                          'the Pinocchio runner uses exact step/rate times, so each MuJoCo phase can '
                          'end one 40 ms guard tick later; comparisons are phase-aligned.'))
    return summary


def run(results_dir=paths.RESULTS, output=None):
    report = {'passed': False, 'checks': {}, 'details': {'failures': [], 'scope': __doc__.strip().split('\n')[0]}}
    checks, details = report['checks'], report['details']

    def check(name, value, limit, relation, unit, category='cross_engine', kind='evidence'):
        value = float(value)
        passed = bool(np.isfinite(value) and {'<=': value <= limit, '>=': value >= limit,
                                              '==': value == limit}[relation])
        checks[name] = dict(passed=passed, value=value if np.isfinite(value) else None, limit=float(limit),
                            relation=relation, unit=unit, category=category, kind=kind)

    try:
        runs = ('face_empty', 'soil_final', 'soil_demo')
        inertials = {case: mujoco_inertial_consistency(case) for case in runs}
        details['mujoco_inertial_consistency'] = inertials
        check('mujoco_arm_inertials_equal_plant_CMG',
              max(max(v['max_difference'].values()) for v in inertials.values()), 1e-12, '<=',
              'kg, m or relative', 'cross_engine_model')
        check('mujoco_arm_bodies_compared', min(v['arm_bodies_compared'] for v in inertials.values()), 24, '==',
              'count', 'cross_engine_model', 'consistency')
        sites = {case: mujoco_connect_sites(case) for case in runs}
        details['mujoco_connect_sites'] = sites
        check('mujoco_connect_sites_equal_plant_constraint_points',
              max(v['max_distance_m'] for v in sites.values()), 1e-12, '<=', 'm', 'cross_engine_model')
        details['mujoco_base_motion_by_phase'] = {case: base_motion_by_phase(case) for case in runs}
        kinematics = {case: mujoco_kinematic_consistency(case) for case in runs}
        details['mujoco_kinematic_consistency'] = kinematics
        check('mujoco_loop_gap_recomputed_by_plant_FK',
              max(k['loop_gap_recomputation_error_m'] for k in kinematics.values()), KINEMATIC_TOLERANCE_M,
              '<=', 'm', 'cross_engine_kinematics')
        check('mujoco_lip_recomputed_by_plant_FK',
              max(k['lip_recomputation_error_m'] for k in kinematics.values()), KINEMATIC_TOLERANCE_M, '<=', 'm',
              'cross_engine_kinematics')
        pairs = {label: compare_pair(results_dir, label, spec) for label, spec in PAIRS.items()}
        details['pairs'] = pairs
        details['withdrawn_gate'] = dict(
            name='no_soil_commanded_coordinates_agree_within_tracking_guard', limit_rad=TRACKING_GUARD_RAD,
            value_rad=pairs['no_soil']['commanded_coordinate_difference_rad']['max'],
            reason='post-data withdrawal after independent review: the floating MuJoCo base settles about 8 cm '
                   'and pitches about 0.08 rad in phase 0 and pitches back about 0.07 rad with 7-10 cm of motion '
                   'in phase 1, and MuJoCo itself exceeds the 0.035 rad tracking guard inside phases 0-2; the '
                   'guard is a phase-end criterion, not an equivalence tolerance between two plants')
        empty = pairs['no_soil']
        check('no_soil_both_engines_record_lift_failure',
              int(LIFT_FAILURE in empty['pinocchio_failures'] and LIFT_FAILURE in empty['mujoco_failures']),
              1, '==', 'boolean', 'cross_engine_outcome', 'consistency')
        for label in ('filled_bed', 'exposed_face'):
            pair = pairs[label]
            check(f'{label}_both_engines_complete_all_phases',
                  int(pair['pinocchio_status'] == 'COMPLETED' and pair['mujoco_status'] == 'COMPLETED'
                      and all(row['pinocchio_time_s'] is not None and row['mujoco_time_s'] is not None
                              for row in pair['events'])), 1, '==', 'boolean', 'cross_engine_outcome', 'consistency')
            check(f'{label}_both_engines_deposit_more_than_mission_threshold',
                  min(pair['final_deposited_kg']['pinocchio'], pair['final_deposited_kg']['mujoco']),
                  REQUIRED_PAYLOAD_KG, '>=', 'kg', 'cross_engine_outcome', 'consistency')
    except Exception:
        import traceback
        details['failures'].append(traceback.format_exc())
        checks['crossengine_comparison_completed'] = dict(passed=False, value=0., limit=1., relation='==',
                                                          unit='boolean')
    report['passed_checks'] = sum(c['passed'] for c in checks.values())
    report['total_checks'] = len(checks)
    report['passed'] = bool(checks and all(c['passed'] for c in checks.values()))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
