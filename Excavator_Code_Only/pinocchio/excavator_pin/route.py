"""Mission-route checks: the unchanged original ``SoilMission`` on the Pinocchio plant.

The mission (tip goals, knots, receiver registration, eleven guarded phases) is
constructed by the preserved original code in the MuJoCo-free environment.  This
module checks, without running the closed loop:

* the tip goals written in the original ``soil_mission.py`` (parsed from the
  source file, not retyped) are reproduced by the *Pinocchio plant* forward
  kinematics at the mission knots, to the original 1e-7 acceptance;
* the receiver registration and the initial closed tree equal the values
  recorded by the original MuJoCo runs (scene metadata and trajectory t = 0);
* along the nominal phase schedule the reference is continuous at every phase
  boundary, its velocity and acceleration are the derivatives of its position,
  every 10 ms reference sample is closed by the native Pinocchio constraints,
  and the plant closure agrees with the original continuation reconstruction;
* planned geometric intent: lip depth at the cut knots, lift clearance,
  receiver footprint at the dump pose and bed clearance outside the digging
  phases; the deepest planned lip point between the knots is reported (the
  joint-space quintic between two knots at the cut depth dips below it).

The nominal schedule assumes every guard passes at the end of its phase
duration; guard timing in closed loop is reported by the case runs.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from . import paths, sentinel
from .engine import context
from .plant import ReducedPlant
from .soil import box

sentinel.install()
paths.add_original_to_path()
from soil_mission import SoilMission  # noqa: E402  (preserved original)
from digging_path import tip_pose  # noqa: E402

RECIPES = {  # RUN_RECIPES.json of the preserved package
    # 'mirrors' is the MuJoCo case whose recipe (cut depth and cut policy) the
    # Pinocchio case reproduces; 'mujoco_cases' share the same cut depth and
    # hence the same mission construction (the cut policy only changes a guard).
    'filled_bed': dict(cut_depth=.11, cut_policy='position-only', mirrors='soil_demo',
                       mujoco_cases=('soil_demo',)),
    'exposed_face': dict(cut_depth=.18, cut_policy='load-aware', mirrors='soil_final',
                         mujoco_cases=('soil_final', 'face_empty', 'face_cut_demo', 'face_no_soil_contact')),
}
MUJOCO_TRACE = {'soil_demo': 'mujoco_soil_demo.npz', 'soil_final': 'mujoco_soil_final.npz',
                'face_empty': 'mujoco_face_empty.npz'}
SAMPLE_PERIOD_S = .01
DIGGING_PHASES = (0, 1, 2, 3)   # settle, lower, draw, curl: the bucket may be below the surface

THRESHOLDS = {
    'plant_FK_tip_goals': (1e-7, '<=', 'm or rad'),
    'plant_FK_vs_original_tip_pose_at_knots': (1e-12, '<=', 'm or rad'),
    # Platform tolerance: the mission is rebuilt by SciPy least squares in this
    # environment and compared with values recorded by the MuJoCo environment.
    'receiver_center_vs_MuJoCo_metadata': (1e-9, '<=', 'm'),
    'receiver_half_size_vs_MuJoCo_metadata': (1e-12, '<=', 'm'),
    'initial_tree_vs_MuJoCo_trajectory_t0': (1e-12, '<=', 'rad or m'),
    'reference_position_jump_at_phase_boundaries': (1e-12, '<=', 'rad'),
    'reference_velocity_jump_at_phase_boundaries': (1e-12, '<=', 'rad/s'),
    'reference_velocity_vs_finite_difference': (1e-7, '<=', 'rad/s'),
    'reference_acceleration_vs_finite_difference': (1e-6, '<=', 'rad/s2'),
    'reference_closure_residual_by_original_backend': (1e-11, '<=', 'm'),
    'reference_plant_vs_original_reconstruct': (1e-10, '<=', 'rad or m'),
    'reference_newton_iterations_max': (3, '<=', 'iterations'),
    'planned_lip_depth_at_cut_knots': (1e-7, '<=', 'm'),
    'planned_lift_clearance_margin': (0., '>=', 'm above 0.8 m guard'),
    'planned_receiver_indicator_at_dump': (1. - 1e-12, '>=', 'fraction'),
    'planned_bed_clearance_outside_digging': (0., '>=', 'm'),
}


def source_goals(cut_depth):
    """Evaluate the ``goals`` list literally written in SoilMission.__init__."""
    tree = ast.parse((paths.ORIGINAL / 'soil_mission.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'goals'):
            expression = ast.Expression(node.value)
            return np.asarray(eval(compile(expression, 'soil_mission.py', 'eval'),  # noqa: S307
                                   {'__builtins__': {}}, {'cut_depth': cut_depth}), dtype=float)
    raise ValueError('goals assignment not found in soil_mission.py')


def schedule(mission):
    starts = np.r_[0., np.cumsum(mission.durations)]
    return starts


def reference_at(mission, phase, t, entered):
    mission.phase, mission.entered = phase, entered
    return mission.arm_reference(t)


def run(output=None):
    report = {'passed': False, 'checks': {}, 'details': {
        'scope': 'Open-loop route checks of the unchanged original mission on the Pinocchio plant; '
                 'nominal phase schedule; no closed-loop, soil or contact claims.', 'failures': []}}
    checks, details = report['checks'], report['details']
    values = {}

    def worst(name, value, mode='max'):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'nonfinite route witness {name}')
        values[name] = (max if mode == 'max' else min)(values.get(name, value), value)

    try:
        cmg, mapping, e, a, r, inverse = context()
        plant = ReducedPlant(cmg, e.cut_ids, e.independent_ids)
        pm = plant.model
        perm = np.array([pm.index[j] for j in e.tree_ids])
        backend = e.native.pin_backend
        details['source_sha256'] = {
            name: hashlib.sha256((paths.ORIGINAL / name).read_bytes()).hexdigest()
            for name in ('soil_mission.py', 'RUN_RECIPES.json')}
        recipes = json.loads((paths.ORIGINAL / 'RUN_RECIPES.json').read_text(encoding='utf-8'))['cases']
        for label, recipe in RECIPES.items():
            mirrored = recipes[recipe['mirrors']]
            if mirrored['cut_depth'] != recipe['cut_depth'] or mirrored['cut_policy'] != recipe['cut_policy']:
                raise ValueError(f'{label} recipe differs from RUN_RECIPES.json {recipe["mirrors"]}')
            for case in recipe['mujoco_cases']:
                if recipes[case]['cut_depth'] != recipe['cut_depth']:
                    raise ValueError(f'{case} does not share the {label} mission construction')
        per_recipe = {}
        for label, recipe in RECIPES.items():
            mission = SoilMission(e, r, cut_depth=recipe['cut_depth'], cut_policy=recipe['cut_policy'])
            info = {}
            goals = source_goals(recipe['cut_depth'])
            knots = [mission.starts[0], mission.ends[1], mission.ends[2], mission.ends[3]]
            for goal, knot in zip(goals, knots):
                q_ctrl = r.reconstruct(knot)
                q = np.empty(plant.n)
                q[perm] = q_ctrl
                plant.initialize(q)
                pts, _ = pm.points(q, ('lip', 'heel'))
                lip, heel = pts['lip'][0], pts['heel'][0]
                plant_tip = np.array([lip[0], lip[2], np.arctan2(lip[2] - heel[2], lip[0] - heel[0])])
                worst('plant_FK_tip_goals', np.max(np.abs(plant_tip - goal)))
                worst('plant_FK_vs_original_tip_pose_at_knots',
                      np.max(np.abs(plant_tip - tip_pose(e, q_ctrl))))
            info['tip_goals_from_source'] = goals.tolist()
            # Receiver registration and initial tree against the original MuJoCo records.
            for case in recipe['mujoco_cases']:
                meta = json.loads((paths.ORIGINAL / 'outputs' / case / 'scene_metadata.json').read_text())
                worst('receiver_center_vs_MuJoCo_metadata',
                      np.max(np.abs(np.asarray(meta['depot_center_m']) - mission.depot_center)))
                worst('receiver_half_size_vs_MuJoCo_metadata',
                      np.max(np.abs(np.asarray(meta['depot_half_size_m']) - mission.depot_half_size)))
                trace = MUJOCO_TRACE.get(case)
                if trace is not None:
                    with np.load(paths.DATA / trace, allow_pickle=False) as f:
                        worst('initial_tree_vs_MuJoCo_trajectory_t0',
                              np.max(np.abs(f['arm_q'][0] - mission.initial_tree)))
            info['receiver_center_m'] = mission.depot_center.tolist()
            info['receiver_half_size_m'] = mission.depot_half_size.tolist()
            # Nominal schedule sampled every 10 ms.
            starts = schedule(mission)
            q = np.empty(plant.n)
            q[perm] = mission.initial_tree
            plant.initialize(q)
            iterations_before = plant.stats['newton_iterations']
            lip_z, mouth_xy, phase_of = [], [], []
            samples = 0
            for phase in range(len(mission.names)):
                t0, t1 = starts[phase], starts[phase + 1]
                if phase > 0:  # continuity with the previous phase end
                    before = reference_at(mission, phase - 1, t0, starts[phase - 1])
                    after = reference_at(mission, phase, t0, t0)
                    worst('reference_position_jump_at_phase_boundaries', np.max(np.abs(before[0] - after[0])))
                    worst('reference_velocity_jump_at_phase_boundaries', np.max(np.abs(before[1] - after[1])))
                count = int(round((t1 - t0) / SAMPLE_PERIOD_S))
                for k in range(count + 1):
                    t = t0 + k * SAMPLE_PERIOD_S
                    desired = reference_at(mission, phase, t, t0)
                    if 0 < k < count:
                        h = 1e-5
                        plus = reference_at(mission, phase, t + h, t0)
                        minus = reference_at(mission, phase, t - h, t0)
                        worst('reference_velocity_vs_finite_difference',
                              np.max(np.abs((plus[0] - minus[0]) / (2 * h) - desired[1])))
                        worst('reference_acceleration_vs_finite_difference',
                              np.max(np.abs((plus[1] - minus[1]) / (2 * h) - desired[2])))
                    q_now, _, _, error = plant.project(desired[0])
                    # Re-measurement with the original BFS backend closure rows (same arithmetic as the
                    # plant's Newton stop criterion; counted as a consistency check).
                    backend.set_configuration(q_now[perm])
                    worst('reference_closure_residual_by_original_backend',
                          np.max(np.abs(backend.closure()[0])))
                    samples += 1
                    if k % 50 == 0:  # every 0.5 s: original continuation reconstruction
                        worst('reference_plant_vs_original_reconstruct',
                              np.max(np.abs(q_now[perm] - r.reconstruct(desired[0]))))
                    pts, rotation = pm.points(q_now, ('lip', 'mouth'))
                    lip_z.append(pts['lip'][0][2])
                    mouth_xy.append(pts['mouth'][0][:2])
                    phase_of.append(phase)
                if phase in (1, 2):  # phase-end samples are the two cut knots
                    worst('planned_lip_depth_at_cut_knots', abs(pts['lip'][0][2] + recipe['cut_depth']))
                if phase == 4:
                    worst('planned_lift_clearance_margin', pts['lip'][0][2] - .8, 'min')
                if phase == 7:
                    center = mission.depot_center[:2]
                    worst('planned_receiver_indicator_at_dump',
                          box(pts['mouth'][0][:2], center - mission.depot_half_size,
                              center + mission.depot_half_size, .05), 'min')
            worst('reference_newton_iterations_max', plant.stats['max_newton_iterations'])
            lip_z, phase_of = np.asarray(lip_z), np.asarray(phase_of)
            outside = ~np.isin(phase_of, DIGGING_PHASES)
            worst('planned_bed_clearance_outside_digging', lip_z[outside].min(), 'min')
            draw = np.isin(phase_of, (1, 2))
            info.update(planned_lip_min_z_during_lower_and_draw_m=float(lip_z[draw].min()),
                        planned_dip_below_cut_depth_m=float(-recipe['cut_depth'] - lip_z[draw].min()))
            info.update(samples=samples, newton_iterations=plant.stats['newton_iterations'] - iterations_before,
                        phase_names=mission.names, phase_durations_s=mission.durations,
                        nominal_phase_starts_s=starts.tolist(),
                        lip_height_min_outside_digging_m=float(lip_z[outside].min()),
                        lip_height_min_m=float(lip_z.min()))
            per_recipe[label] = info
        details['recipes'] = per_recipe
        for name, (limit, relation, unit) in THRESHOLDS.items():
            value = values.get(name, np.nan)
            passed = bool(np.isfinite(value) and (value <= limit if relation == '<=' else value >= limit))
            checks[name] = dict(passed=passed, value=float(value) if np.isfinite(value) else None,
                                limit=float(limit), relation=relation, unit=unit, category='route')
    except Exception:
        import traceback
        details['failures'].append(traceback.format_exc())
        checks['route_validation_completed'] = dict(passed=False, value=0., limit=1., relation='==',
                                                     unit='boolean')
    details['mujoco'] = sentinel.report()
    report['passed_checks'] = sum(c['passed'] for c in checks.values())
    report['total_checks'] = len(checks)
    report['passed'] = bool(checks and all(c['passed'] for c in checks.values()))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
