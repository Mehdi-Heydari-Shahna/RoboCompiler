"""Acceptance gates for the executed Kangaroo Pinocchio cases; no simulation here.

Gate groups (every gate is recorded with value, limit, relation and unit):

* ``source.*``      original v22 files, unchanged PACDM, MuJoCo not imported;
* ``reference.*``   MuJoCo-free regeneration of the original contact reference;
* ``mechanics.*``   instantaneous three-route mechanics checks (``pin_checks``);
* ``evidence.*``    full replay of every saved case (``pin_evidence``);
* ``native_audit.*`` native Pinocchio dynamics/closure/contact audits;
* ``auditor_controls.*`` deliberately corrupted rollouts must be rejected;
* ``<case>.*``      the v22 task acceptance (``data/contact_acceptance.json``,
  unchanged) plus Pinocchio-specific numerical gates;
* ``refinement.*``  the v22 time-refinement gates (1 ms vs 0.5 ms);
* ``negative.*``    the v22 negative controls;
* ``legacy_mujoco.*`` separately labelled cross-engine agreement bands with
  the unchanged v22 MuJoCo pipeline, rerun as legacy evidence.  Bands are one
  quarter of the corresponding v22 task tolerances (40 mm final position,
  8 deg tilt); they were declared before the comparison was computed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .cases import ALL, BY_NAME, NEGATIVE, POSITIVE

PACDM_SHA256 = 'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de'
LEGACY_BANDS = dict(base_position_m=.010, base_orientation_deg=2., final_base_position_m=.010)
PIN_LIMITS = dict(maximum_pacdm_closure=5e-13, maximum_tangent_residual=1e-9,
                  maximum_acceleration_closure=1e-8, maximum_reduced_equation_residual=1e-9,
                  maximum_ncp_primal=1e-8, maximum_ncp_dual=1e-8, maximum_ncp_complementarity=1e-8,
                  maximum_penetration_m=1e-5)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _grid(a, times):
    """Indices of a Pinocchio log at the given times (exact multiples of its step)."""
    dt = float(a['timestep_s'])
    k = np.rint(np.asarray(times) / dt).astype(int)
    if np.any(k < 0) or np.any(k >= len(a['time'])) or not np.allclose(a['time'][k], times, rtol=0., atol=1e-9):
        raise ValueError('Comparison grid is not contained in the saved time grid')
    return k


def _stored(a, steps):
    where = {int(s): i for i, s in enumerate(a['stored_step'])}
    return a['stored_z'][[where[int(s)] for s in steps]]


def compare_runs(fine, coarse, cmg):
    """Two Pinocchio runs on their common 5 ms stored grid (time refinement)."""
    a, b = _load(fine), _load(coarse)
    end = min(a['time'][-1], b['time'][-1])
    times = np.round(np.arange(0., end + 1e-9, .005), 10)
    ka, kb = _grid(a, times), _grid(b, times)
    position = np.linalg.norm(a['base'][ka, :3] - b['base'][kb, :3], axis=1)
    angle = np.degrees((Rotation.from_quat(a['base'][ka, 3:7]).inv()
                        * Rotation.from_quat(b['base'][kb, 3:7])).magnitude())
    za, zb = _stored(a, ka)[:, :len(cmg['coordinate_ids'])], _stored(b, kb)[:, :len(cmg['coordinate_ids'])]
    slide = np.array([j['type'] == 'prismatic' for j in _joints(cmg)])
    return dict(grid='common 5 ms stored states', samples=len(times),
                max_base_position_difference_m=float(position.max()),
                final_base_position_difference_m=float(position[-1]),
                max_base_orientation_difference_deg=float(angle.max()),
                max_prismatic_coordinate_difference_m=float(np.abs(za - zb)[:, slide].max()),
                max_revolute_coordinate_difference_rad=float(np.abs(za - zb)[:, ~slide].max()))


def _joints(cmg):
    records = {j['id']: j for j in cmg['joints']}
    return [records[i] for i in cmg['coordinate_ids']]


def compare_legacy(pin_path, legacy_path, cmg, force_bounds):
    """Pinocchio case vs the rerun v22 MuJoCo case at the legacy 5 ms samples."""
    a, m = _load(pin_path), _load(legacy_path)
    if list(m['coordinate_ids']) != list(cmg['coordinate_ids']):
        raise ValueError('Legacy coordinate order differs from the CMG')
    end = min(a['time'][-1], m['time'][-1])
    use = m['time'] <= end + 1e-12
    times = m['time'][use]
    k = _grid(a, times)
    position = np.linalg.norm(a['base'][k, :3] - m['base_position'][use], axis=1)
    angle = np.degrees((Rotation.from_quat(a['base'][k, 3:7]).inv()
                        * Rotation.from_quat(m['base_quaternion_xyzw'][use])).magnitude())
    ids = list(cmg['coordinate_ids'])
    motor_index = [ids.index(x['joint']) for x in cmg['actuators']]
    motor = np.abs(a['motor'][k] - m['coordinates'][use][:, motor_index])
    z = _stored(a, k)[:, :len(ids)]
    slide = np.array([j['type'] == 'prismatic' for j in _joints(cmg)])
    coordinate = np.abs(z - m['coordinates'][use])
    force_pin = np.clip(a['act'][k], force_bounds[:, 0], force_bounds[:, 1])
    force = force_pin - m['motor_force_N'][use]
    feet = a['lam'][k].reshape(len(k), 2, 4, 3).sum(axis=2) / float(a['timestep_s'])
    stance = times > .5
    normal = feet[stance, :, 2] - m['foot_force_N'][use][stance][:, :, 2]
    return dict(grid='legacy 5 ms samples', samples=int(use.sum()),
                max_base_position_difference_m=float(position.max()),
                rms_base_position_difference_m=float(np.sqrt(np.mean(position ** 2))),
                final_base_position_difference_m=float(position[-1]),
                max_base_orientation_difference_deg=float(angle.max()),
                max_motor_coordinate_difference_m=float(motor.max()),
                max_prismatic_coordinate_difference_m=float(coordinate[:, slide].max()),
                max_revolute_coordinate_difference_rad=float(coordinate[:, ~slide].max()),
                rms_drive_force_difference_N=float(np.sqrt(np.mean(force ** 2))),
                max_drive_force_difference_N=float(np.abs(force).max()),
                rms_stance_foot_normal_force_difference_N=float(np.sqrt(np.mean(normal ** 2))) if normal.size else 0.,
                note='Pinocchio foot force is the step-average impulse / dt; MuJoCo foot force is the '
                     'instantaneous soft-contact force at the sample time.')


def _gate(checks, name, value, limit, unit='', relation='<=', group=None):
    value = value if isinstance(value, (bool, str)) or value is None else float(value)
    if relation == '<=':
        passed = value is not None and np.isfinite(value) and value <= limit
    elif relation == '>=':
        passed = value is not None and np.isfinite(value) and value >= limit
    elif relation == '>':
        passed = value is not None and np.isfinite(value) and value > limit
    else:
        passed = value == limit
    checks[name] = dict(passed=bool(passed), value=value, limit=limit, unit=unit, relation=relation,
                        group=group or name.split('.')[0])


def source_integrity(root):
    """The original v22 package's own manifest and provenance hashes."""
    base = Path(root) / 'original_v22'
    manifest = json.loads((base / 'MANIFEST_SHA256.json').read_text())
    entries = manifest.get('files', manifest)
    present, missing, mismatched = 0, [], []
    for path, digest in entries.items():
        target = base / path
        if not target.is_file():
            missing.append(path)
        elif _sha(target) != digest:
            mismatched.append(path)
        else:
            present += 1
    provenance = json.loads((base / 'data/provenance.json').read_text())
    core = {p: _sha(base / p) == d for p, d in provenance['core_sha256'].items()}
    assets = {p: (base / p).is_file() and _sha(base / p) == d for p, d in provenance['source_files'].items()}
    return dict(manifest_entries=len(entries), matching=present, mismatched=mismatched, missing=missing,
                missing_outside_results_or_videos=[p for p in missing
                                                   if not p.startswith(('results/', 'videos/'))],
                core_bytes_preserved=all(core.values()), core_files=len(core),
                upstream_assets_preserved=all(assets.values()), upstream_files=len(assets))


def aggregate(root):
    root = Path(root)
    r = root / 'results'
    cmg = json.loads((root / 'original_v22/data/whole_body_cmg.json').read_text())
    acceptance = json.loads((root / 'original_v22/data/contact_acceptance.json').read_text())
    bounds = np.array([a['force_bounds_N'] for a in cmg['actuators']])
    mechanics = json.loads((r / 'mechanics.json').read_text())
    audits = json.loads((r / 'trajectory_audits.json').read_text())
    controls = json.loads((r / 'audit_negative_controls.json').read_text())
    evidence = json.loads((r / 'case_evidence.json').read_text())
    rebuild = json.loads((r / 'reference_rebuild.json').read_text())
    cases = {c['name']: json.loads((r / f"{c['name']}.json").read_text()) for c in ALL}
    positive = [c['name'] for c in POSITIVE]
    if set(audits) != set(positive):
        raise ValueError('Native audits must cover exactly the five positive cases')
    checks = {}
    g = lambda *a, **k: _gate(checks, *a, **k)
    # ------------------------------------------------------------ source
    integrity = source_integrity(root)
    g('source.unchanged_PACDM_sha256', _sha(root / 'original_v22/pacdm.py'), PACDM_SHA256, relation='==')
    g('source.supplied_manifest_no_mismatch', len(integrity['mismatched']), 0, 'files', '==')
    g('source.supplied_manifest_missing_only_results_videos', len(integrity['missing_outside_results_or_videos']),
      0, 'files', '==')
    g('source.v22_core_bytes_preserved', integrity['core_bytes_preserved'], True, relation='==')
    g('source.upstream_assets_preserved', integrity['upstream_assets_preserved'], True, relation='==')
    g('source.mujoco_not_imported', 'mujoco' in sys.modules, False, relation='==')
    # ------------------------------------------------------------ reference
    g('reference.rebuilt_without_mujoco_matches_original', rebuild['maximum_array_difference'], 1e-9, 'mixed SI')
    g('reference.rebuild_mujoco_not_imported', 'MuJoCo not imported' in rebuild['method'], True, relation='==')
    built = rebuild['rebuilt_summary']
    for key, limit in acceptance['reference'].items():
        g('reference.v22.' + key, built[key], limit, 'mixed SI')
    g('reference.v22.all_12_physical_actuators_move', min(built['actuator_travel_m']), .001, 'm', '>')
    g('reference.v22.no_reacquisition', built['reacquisitions'], 0, 'events', '==')
    g('reference.v22.planned_contact_normals_positive', built['minimum_planned_corner_normal_N'], 0., 'N', '>')
    # ------------------------------------------------------------ mechanics
    for key, check in mechanics['checks'].items():
        g('mechanics.' + key, check['value'], check['limit'], check.get('units', ''), check['relation'])
    # ------------------------------------------------------------ evidence replay and native audits
    for name, record in evidence.items():
        passed = sum(c['passed'] for c in record['checks'].values())
        g(f'evidence.{name}', int(record['passed']), 1, f'{passed}/{len(record["checks"])} replay checks', '==')
    for name, record in audits.items():
        for key, check in record['checks'].items():
            g(f'native_audit.{name}.{key}', check['value'], check['limit'], check.get('units', ''),
              check['relation'])
    for key, check in controls['checks'].items():
        g('auditor_controls.' + key, int(check['passed']), 1, f"expected {check['expected']}", '==')
    # ------------------------------------------------------------ positive cases
    limits = acceptance['positive_trials']
    for name in positive:
        c = cases[name]
        spec = BY_NAME[name]
        cfg = c['configuration']
        weight = c['mass_kg'] * 9.81
        audit = audits[name]['checks']
        g(f'{name}.declared_scenario', c['timestep_s'] == spec['dt'] and c['duration_s'] == 10.
          and c['completed_steps'] == c['steps'] and c['stop_reason'] is None, True, relation='==')
        # v22 acceptance (unchanged thresholds); loop gap/universal from the native audit
        g(f'{name}.v22.maximum_loop_gap_m', audit['stored_state_native_point_gap']['value'],
          limits['maximum_loop_gap_m'], 'm')
        g(f'{name}.v22.maximum_universal_dot', audit['stored_state_native_universal_residual']['value'],
          limits['maximum_universal_dot'])
        for key, unit in [('maximum_motor_force_N', 'N'), ('maximum_penetration_m', 'm'),
                          ('maximum_tilt_deg', 'deg'), ('final_tilt_deg', 'deg'),
                          ('maximum_final_base_speed_m_s', 'm/s'), ('final_position_error_m', 'm'),
                          ('maximum_energy_ledger_error_J', 'J')]:
            g(f'{name}.v22.{key}', c[key], limits[key], unit)
        g(f'{name}.v22.foot_drift', max(c['maximum_foot_drift_after_landing_m']),
          acceptance['maximum_foot_drift_after_landing_m'], 'm')
        g(f'{name}.v22.final_weight_balance', abs(c['mean_final_ground_normal_N'] - weight),
          acceptance['maximum_final_weight_error_N'], 'N')
        g(f'{name}.v22.full_model', c['body_count'] == 78 and c['motor_count'] == 12, True, relation='==')
        g(f'{name}.v22.slide_margin', c['minimum_slide_margin_m'], acceptance['joint_margin_lower_bound'], 'm', '>=')
        g(f'{name}.v22.hinge_margin', c['minimum_hinge_margin_rad'], acceptance['joint_margin_lower_bound'], 'rad', '>=')
        g(f'{name}.v22.no_saturation', c['saturated_control_updates'], 0, 'updates', '==')
        g(f'{name}.v22.actual_landing', c['touchdown_s'] is not None and c['touchdown_s'] > .05
          and c['maximum_ground_normal_N'] > 1.5 * weight, True, relation='==')
        g(f'{name}.v22.crouch_completed', c['achieved_crouch_m'], acceptance['minimum_achieved_crouch_m'], 'm', '>=')
        g(f'{name}.v22.weight_shift_completed', c['achieved_lateral_excursion_m'],
          acceptance['minimum_achieved_lateral_excursion_m'], 'm', '>=')
        g(f'{name}.v22.friction_cone', c['maximum_contact_friction_ratio'], cfg['friction'] + 1e-8, 'ratio')
        g(f'{name}.v22.unilateral_normal', -c['minimum_contact_normal_N'], 1e-8, 'N')
        # Pinocchio / PACDM numerical gates
        for key, limit in PIN_LIMITS.items():
            g(f'{name}.pinocchio.{key}', c[key], limit)
        g(f'{name}.pinocchio.rank_full', c['minimum_rank_full'], 128, 'rank', '==')
        g(f'{name}.pinocchio.rank_passive', c['minimum_rank_passive'], 128, 'rank', '==')
        g(f'{name}.pinocchio.mapping_conditioning', c['minimum_mapping_rcond'], 1e-5, 'reciprocal condition', '>=')
        g(f'{name}.pinocchio.ideal_loop_and_limit_work_zero',
          abs(c['work_J']['loop']) + abs(c['work_J']['limit']), 0., 'J', '==')
    # ------------------------------------------------------------ refinement (v22 definition)
    coarse, fine = cases['landing_nominal'], cases['landing_refined']
    refinement = compare_runs(r / 'landing_refined.npz', r / 'landing_nominal.npz', cmg)
    g('refinement.final_position', float(np.linalg.norm(np.subtract(coarse['final_base_position_m'],
                                                                   fine['final_base_position_m']))),
      acceptance['refinement_final_position_difference_m'], 'm')
    g('refinement.motor_work', abs(coarse['work_J']['motor'] - fine['work_J']['motor']),
      acceptance['refinement_motor_work_difference_J'], 'J')
    g('refinement.fine_energy_error_improves',
      fine['maximum_energy_ledger_error_J'] < coarse['maximum_energy_ledger_error_J'], True, relation='==')
    # ------------------------------------------------------------ negative controls (v22 definition)
    with np.load(root / 'original_v22/data/contact_reference.npz') as z:
        home = float(z['home_height'])
    drop = BY_NAME['negative_no_contact']
    neg = cases['negative_no_contact']
    g('negative.without_contact_falls', home + neg['configuration']['drop_height_m'] - neg['final_base_position_m'][2],
      acceptance['negative_no_contact_minimum_fall_m'], 'm', '>')
    g('negative.without_contact_zero_ground_force', neg['maximum_ground_normal_N'], 0., 'N', '==')
    g('negative.without_contact_declared', neg['duration_s'] == drop['duration'] and neg['no_contact'], True,
      relation='==')
    loops = cases['negative_no_loops_contact']
    g('negative.without_loops_fails_closure', loops['maximum_loop_gap_m'], acceptance['negative_no_loop_minimum_gap_m'],
      'm', '>')
    g('negative.without_loops_declared', loops['open_tree'] and loops['duration_s'] == .3, True, relation='==')
    passive = cases['negative_passive']
    g('negative.without_drives_does_not_hold_height', home - passive['final_base_position_m'][2], .05, 'm', '>')
    g('negative.without_drives_zero_force', passive['maximum_motor_force_N'], 0., 'N', '==')
    g('negative.without_drives_declared', passive['passive_control'] and passive['duration_s'] == 1., True,
      relation='==')
    # ------------------------------------------------------------ cross-engine (labelled legacy evidence)
    legacy = {}
    for name in positive:
        path = root / 'legacy_evidence' / f'{name}.npz'
        legacy[name] = compare_legacy(r / f'{name}.npz', path, cmg, bounds)
        mj = json.loads((root / 'legacy_evidence' / f'{name}.json').read_text())
        c = cases[name]
        legacy[name].update(
            mujoco_timestep_s=mj['timestep_s'], pinocchio_timestep_s=c['timestep_s'],
            touchdown_s=dict(pinocchio=c['touchdown_s'], mujoco=mj['touchdown_s']),
            maximum_ground_normal_N=dict(pinocchio=c['maximum_ground_normal_N'], mujoco=mj['maximum_ground_normal_N']),
            mean_final_ground_normal_N=dict(pinocchio=c['mean_final_ground_normal_N'],
                                            mujoco=mj['mean_final_ground_normal_N']),
            maximum_motor_force_N=dict(pinocchio=c['maximum_motor_force_N'], mujoco=mj['maximum_motor_force_N']),
            motor_work_J=dict(pinocchio=c['work_J']['motor'], mujoco=mj['work_J']['motor']),
            achieved_crouch_m=dict(pinocchio=c['achieved_crouch_m'], mujoco=mj['achieved_crouch_m']),
            achieved_lateral_excursion_m=dict(pinocchio=c['achieved_lateral_excursion_m'],
                                              mujoco=mj['achieved_lateral_excursion_m']),
            maximum_tilt_deg=dict(pinocchio=c['maximum_tilt_deg'], mujoco=mj['maximum_tilt_deg']),
            maximum_energy_ledger_error_J=dict(pinocchio=c['maximum_energy_ledger_error_J'],
                                               mujoco=mj['maximum_energy_ledger_error_J']),
            maximum_loop_gap_m=dict(pinocchio=audits[name]['checks']['stored_state_native_point_gap']['value'],
                                    mujoco=mj['maximum_loop_gap_m']))
        g(f'legacy_mujoco.{name}.base_position_agreement', legacy[name]['max_base_position_difference_m'],
          LEGACY_BANDS['base_position_m'], 'm')
        g(f'legacy_mujoco.{name}.base_orientation_agreement', legacy[name]['max_base_orientation_difference_deg'],
          LEGACY_BANDS['base_orientation_deg'], 'deg')
        g(f'legacy_mujoco.{name}.final_base_position_agreement', legacy[name]['final_base_position_difference_m'],
          LEGACY_BANDS['final_base_position_m'], 'm')
    groups = {}
    for key, check in checks.items():
        entry = groups.setdefault(check['group'], [0, 0])
        entry[0] += check['passed']
        entry[1] += 1
    result = dict(
        passed=all(c['passed'] for c in checks.values()),
        passed_count=sum(c['passed'] for c in checks.values()), check_count=len(checks),
        groups={k: dict(passed=v[0], total=v[1]) for k, v in groups.items()},
        failed=[k for k, c in checks.items() if not c['passed']],
        checks=checks, cases=cases, refinement=refinement, legacy_mujoco_comparison=legacy,
        legacy_bands=LEGACY_BANDS, source_integrity=integrity, reference_rebuild=rebuild,
        summary='Simulation verification of the v22 Kangaroo CMG (78 bodies, floating pelvis, '
              '24 loop cuts, 12 motor slides) with Pinocchio 3.8.0 dynamics, the unchanged PACDM closure '
              'algorithm and native Pinocchio rigid unilateral Coulomb contact, on the v22 landing/crouch/'
              'shift/push task.',
        limits=[
            'Contact is rigid (Pinocchio NCP) at the eight bottom corners of the two source foot boxes on a flat '
            'floor; MuJoCo v22 uses soft contact on the box geometry. Trajectories agree within the declared bands, '
            'not identically.',
            'The independent native loop oracle shares the CMG-compiled Pinocchio tree with the integrator; it '
            'verifies the PACDM reduction and force mapping, not the authored inertias. The accepted NumPy source '
            'dynamics route shares only the CMG records.',
            'Joint limits are enforced as PACDM branch bounds; no joint-limit impact is modelled. Every accepted '
            'state is inside all 76 coordinate ranges.',
            'Timestep refinement covers 1 ms and 0.5 ms only; no global convergence theorem is implied.',
            'The reconstructed model is not hardware-identified; walking, running, jump take-off and '
            'self-collision are outside this scope.',
            'The MuJoCo comparison uses the unchanged v22 pipeline rerun as labelled legacy evidence in a separate '
            'environment; the Pinocchio pipeline never imports MuJoCo.',
            'The video is CPU-rendered playback of saved Pinocchio/PACDM states; acceptance comes from the gates.'])
    write_json(r / 'validation.json', result)
    return result


def hash_manifest(root):
    root = Path(root)
    hashes = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS.json' and p.suffix != '.pyc'}
    write_json(root / 'SHA256SUMS.json', hashes)
    return len(hashes)
