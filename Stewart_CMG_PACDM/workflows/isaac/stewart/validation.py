"""Deterministic mechanics acceptance tests for the authored six-UPS CMG.

These tests compare independent CMG lowerings to MuJoCo and Pinocchio.  They
do not identify real hardware, certify an entire workspace, or replace the
separate mission simulation.  Analytic inverse kinematics supplies reference
poses and commanded leg lengths; every tested forward solution uses the
unchanged PACDM solver, starting from a perturbed seed.
"""
from __future__ import annotations

import platform
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import scipy

from vendor.pacdm_original import PACDM, PointGraph, skew
from .model import inverse_seed, rotation
from .pin_backend import PinBackend


def _poses(cmg):
    nominal = np.asarray(cmg['geometry']['nominal_pose'], dtype=float)
    # Source platform order is x, y, z, yaw, pitch, roll.
    extent = np.array([.055, .055, .040, .180, .130, .130])
    result = [nominal.copy()]
    for axis in range(6):
        for sign in (-1., 1.):
            value = nominal.copy()
            value[axis] += sign * extent[axis]
            result.append(value)
    rng = np.random.default_rng(20260923)
    corners = rng.choice(64, 12, replace=False)
    for corner in corners:
        signs = np.array([1. if corner & (1 << k) else -1. for k in range(6)])
        result.append(nominal + signs * extent)
    return result


def _point_constraints(graph, q):
    """World point residual/Jacobian independent of PACDM SE(3) cut charts."""
    poses, spatial = graph.poses(q)
    residuals, derivatives = [], []
    for cut in graph.cuts:
        values = []
        for body_key, point_key in (('body1', 'point1_m'), ('body2', 'point2_m')):
            body = cut[body_key]
            pose = poses[body]
            point = pose[:3, 3] + pose[:3, :3] @ np.asarray(cut[point_key])
            jacobian = np.c_[-skew(point), np.eye(3)] @ spatial[body][:, :graph.nt]
            values.append((point, jacobian))
        residuals.append(values[1][0] - values[0][0])
        derivatives.append(values[1][1] - values[0][1])
    return np.concatenate(residuals), np.vstack(derivatives)


def _scaled_error(actual, reference):
    return float(np.linalg.norm(actual - reference, ord=np.inf)
                 / max(1., np.linalg.norm(reference, ord=np.inf)))


def validate_mechanics(cmg, xml_path):
    """Return measured checks; any failed sample/exception makes passed false.

    ``xml_path`` must be the MJCF compiled from the same CMG.  PinBackend
    reads CMG directly and does not consume that XML.  Native engine dynamics
    comparisons are unconstrained CRBA/RNEA at identical source coordinates;
    a separate full KKT comparison checks the PACDM reduced dynamics.
    """
    checks = {}
    report = {
        'passed': False, 'checks': checks,
        'details': {
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'scipy': scipy.__version__, 'mujoco': mujoco.__version__,
                         'pinocchio': pin.__version__},
            'xml_file': Path(xml_path).name,
            'sample_count_requested': 25,
            'random_seed': 20260923,
            'coordinate_order': list(cmg['coordinate_ids']),
            'method': 'Independent CMG lowerings, perturbed-seed PACDM forward closure, '
                      'nonzero-velocity native tree CRBA/RNEA and full-constraint KKT.',
            'scope': 'Finite deterministic model checks; ideal rigid-body parameters, '
                     'no measured hardware or global workspace certification.',
            'sample_failures': [],
        },
    }

    def check(name, value, limit, unit='dimensionless', relation='<='):
        finite = value is not None and bool(np.isfinite(value))
        accepted = finite and {'<=': lambda: value <= limit,
                               '>=': lambda: value >= limit,
                               '==': lambda: value == limit}[relation]()
        checks[name] = {'passed': bool(accepted), 'value': value if finite else None,
                        'limit': limit, 'unit': unit, 'relation': relation}

    try:
        seed = inverse_seed(cmg, cmg['geometry']['nominal_pose'])
        graph = PointGraph(cmg, seed)
        solver = PACDM(graph)
        backend = PinBackend(cmg)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        q_indices, v_indices = [], []
        for name in cmg['coordinate_ids']:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f'Missing source joint in MuJoCo: {name}')
            q_indices.append(int(model.jnt_qposadr[joint]))
            v_indices.append(int(model.jnt_dofadr[joint]))
        q_indices, v_indices = np.asarray(q_indices), np.asarray(v_indices)
        if len(set(q_indices)) != graph.nt or len(set(v_indices)) != graph.nt:
            raise ValueError('MuJoCo coordinate mapping is not one-to-one')
        check('physical_coordinates', graph.nt, 24, 'count', '==')
        check('augmented_coordinates', graph.n, 42, 'count', '==')
        check('augmented_closure_rows', graph.residual(graph.lift(seed))[1].shape[0],
              36, 'count', '==')
        check('point_closures', graph.nc, 6, 'count', '==')
        check('independent_coordinates', len(graph.active), 6, 'count', '==')
        check('mujoco_velocities', model.nv, 24, 'count', '==')
        check('pinocchio_velocities', backend.nv, 24, 'count', '==')
        check('mujoco_actuators', model.nu, 6, 'count', '==')
        check('mujoco_connect_equalities', model.neq, 6, 'count', '==')
        passive_force = max(float(np.max(np.abs(model.dof_damping), initial=0.)),
                            float(np.max(np.abs(model.dof_armature), initial=0.)),
                            float(np.max(np.abs(model.dof_frictionloss), initial=0.)),
                            float(np.max(np.abs(model.jnt_stiffness), initial=0.)))
        check('unmodeled_joint_effects', passive_force, 0.)
        if model.nq != graph.nt or model.nv != graph.nt or model.nu != 6:
            raise ValueError('The mechanics suite requires the authored scalar-tree model')
    except Exception as error:
        report['details']['initialization_error'] = f'{type(error).__name__}: {error}'
        check('initialization_success', 0, 1, 'boolean', '==')
        return report

    maxima = {}
    minima = {}
    completed = 0
    rng = np.random.default_rng(20260923)
    references = _poses(cmg)
    report['details']['reference_poses_xyz_yaw_pitch_roll_SI'] = [p.tolist() for p in references]
    report['details']['seed_perturbation_max_abs_SI'] = .002
    report['details']['minimum_acquisition_homotopy_steps'] = None
    report['details']['maximum_acquisition_homotopy_steps'] = None

    def maximum(name, value):
        value = float(value)
        maxima[name] = max(maxima.get(name, -np.inf), value)

    def minimum(name, value):
        value = float(value)
        minima[name] = min(minima.get(name, np.inf), value)

    B = np.zeros((graph.nt, len(graph.active)))
    B[graph.active] = np.eye(len(graph.active))
    for index, pose in enumerate(references):
        try:
            exact = inverse_seed(cmg, pose)
            initial = exact + rng.uniform(-.002, .002, graph.nt)
            initial[graph.active] = exact[graph.active]
            acquired, acquisition = solver.acquire(exact[graph.active], graph.lift(initial))
            if not acquisition['success']:
                raise RuntimeError(f'PACDM forward acquisition failed: {acquisition}')
            N, mapping = solver.mapping(acquired)
            if N is None or not mapping['success']:
                raise RuntimeError(f'PACDM mapping failed: {mapping}')
            steps = acquisition['accepted_steps']
            key = 'minimum_acquisition_homotopy_steps'
            report['details'][key] = min(report['details'][key] or steps, steps)
            key = 'maximum_acquisition_homotopy_steps'
            report['details'][key] = max(report['details'][key] or steps, steps)
            physical = acquired[:graph.nt]
            Np = N[:graph.nt]
            r, J, _ = graph.residual(acquired)
            point_r, point_J = _point_constraints(graph, acquired)
            minimum('rank_augmented', mapping['rank_full'])
            minimum('rank_passive', mapping['rank_passive'])
            minimum('rank_point', np.linalg.matrix_rank(point_J, tol=1e-9))
            minimum('pacdm_rcond', mapping['rcond'])
            maximum('closure', np.max(np.abs(point_r)))
            maximum('augmented_closure', np.max(np.abs(r)))
            maximum('JN', np.max(np.abs(J @ N)))
            maximum('physical_JN', np.max(np.abs(point_J @ Np)))
            maximum('forward_translation', np.linalg.norm(physical[:3] - pose[:3]))
            relative_R = rotation(pose).T @ rotation(physical[:6])
            maximum('forward_rotation', np.linalg.norm(relative_R - np.eye(3), ord=2))
            maximum('forward_all_coordinates', np.max(np.abs(physical - exact)))
            minimum('joint_limit_margin', np.min(np.r_[physical - graph.lower[:graph.nt],
                                                        graph.upper[:graph.nt] - physical]))
            reference_poses, _ = graph.poses(acquired)
            pin_poses = backend.poses(physical)
            for body, expected in reference_poses.items():
                maximum('pin_fk_position', np.linalg.norm(pin_poses[body][:3, 3] - expected[:3, 3]))
                maximum('pin_fk_rotation', np.linalg.norm(pin_poses[body][:3, :3] - expected[:3, :3], ord=2))

            length_velocity = rng.uniform(-.13, .13, 6)
            velocity = Np @ length_velocity
            acceleration = rng.uniform(-.75, .75, graph.nt)
            data.qpos[q_indices] = physical
            data.qvel[v_indices] = velocity
            data.ctrl[:] = 0.
            mujoco.mj_forward(model, data)
            for body, expected in reference_poses.items():
                if body.startswith('chart_'):
                    continue  # Collapsed platform chart frames have no MJ body.
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                if body_id < 0:
                    raise ValueError(f'Missing physical MuJoCo body: {body}')
                maximum('mj_fk_position', np.linalg.norm(data.xpos[body_id] - expected[:3, 3]))
                maximum('mj_fk_rotation', np.linalg.norm(data.xmat[body_id].reshape(3, 3) - expected[:3, :3], ord=2))
            native_mass = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, native_mass, data.qM)
            mj_mass = native_mass[np.ix_(v_indices, v_indices)].copy()
            mj_bias = data.qfrc_bias[v_indices].copy()
            data.qacc[v_indices] = acceleration
            native_inverse = np.empty(model.nv)
            mujoco.mj_rne(model, data, 1, native_inverse)
            mj_inverse = native_inverse[v_indices].copy()
            pin_mass = backend.mass(physical)
            pin_bias = backend.bias(physical, velocity)
            pin_inverse = backend.inverse(physical, velocity, acceleration)
            maximum('mass', _scaled_error(pin_mass, mj_mass))
            maximum('bias', _scaled_error(pin_bias, mj_bias))
            maximum('inverse', _scaled_error(pin_inverse, mj_inverse))
            maximum('pin_rnea_identity', _scaled_error(pin_inverse, pin_mass @ acceleration + pin_bias))
            maximum('mj_rnea_identity', _scaled_error(mj_inverse, mj_mass @ acceleration + mj_bias))
            minimum('mass_min_eigenvalue', np.min(np.linalg.eigvalsh(pin_mass)))
            reduced_mass = Np.T @ pin_mass @ Np
            minimum('reduced_mass_min_eigenvalue', np.min(np.linalg.eigvalsh(reduced_mass)))
            maximum('reduced_mass', _scaled_error(reduced_mass, Np.T @ mj_mass @ Np))

            force = rng.uniform(-150., 150., 6)
            for i, joint_name in enumerate(cmg['independent_ids']):
                actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_' + joint_name)
                if actuator < 0:
                    raise ValueError(f'Missing leg actuator: motor_{joint_name}')
                data.ctrl[actuator] = force[i]
            mujoco.mj_fwdActuation(model, data)
            actual_force = data.qfrc_actuator[v_indices].copy()
            maximum('actuator_effort_map', np.max(np.abs(actual_force - B @ force)))
            maximum('reduced_actuator_map', np.max(np.abs(Np.T @ B - np.eye(6))))
            maximum('virtual_work', abs(velocity @ actual_force - length_velocity @ force))

            # A separate geometric reference differentiates physical coordinates.
            # It does not call PACDM, use N, or solve its Jacobian system.
            direction = rng.uniform(-1., 1., 6)
            h = 1e-6
            plus = inverse_seed(cmg, pose + h * direction)
            minus = inverse_seed(cmg, pose - h * direction)
            dq_reference = (plus - minus) / (2. * h)
            dl_reference = dq_reference[graph.active]
            maximum('N_finite_difference', np.max(np.abs(Np @ dl_reference - dq_reference)))

            # Independent full physical point-constraint KKT versus reduction.
            # The two acceleration-bias estimates use distinct Jacobians:
            # PACDM SE(3) charts and direct world endpoint point differences.
            augmented_velocity = N @ length_velocity
            hd = 2e-6 / max(1., np.linalg.norm(augmented_velocity))
            Jdot = (graph.residual(acquired + hd * augmented_velocity)[1]
                    - graph.residual(acquired - hd * augmented_velocity)[1]) / (2. * hd)
            rows = np.asarray(mapping['rows'], dtype=int)
            k = np.zeros(graph.n)
            k[graph.passive] = -np.linalg.solve(J[np.ix_(rows, graph.passive)],
                                               (Jdot @ augmented_velocity)[rows])
            point_Jdot = (_point_constraints(graph, acquired + hd * augmented_velocity)[1]
                          - _point_constraints(graph, acquired - hd * augmented_velocity)[1]) / (2. * hd)
            length_acc = np.linalg.solve(reduced_mass,
                                         force - Np.T @ (pin_bias + pin_mass @ k[:graph.nt]))
            reduced_acc = Np @ length_acc + k[:graph.nt]
            kkt_matrix = np.block([[mj_mass, -point_J.T],
                                   [point_J, np.zeros((18, 18))]])
            kkt_rhs = np.r_[B @ force - mj_bias, -point_Jdot @ velocity]
            kkt_acc = np.linalg.solve(kkt_matrix, kkt_rhs)[:graph.nt]
            maximum('constrained_acceleration', _scaled_error(reduced_acc, kkt_acc))
            maximum('acceleration_closure', np.max(np.abs(point_J @ reduced_acc + point_Jdot @ velocity)))
            completed += 1
        except Exception as error:
            report['details']['sample_failures'].append(
                {'sample_index': index, 'pose': pose.tolist(),
                 'error': f'{type(error).__name__}: {error}'})

    report['details']['sample_count_completed'] = completed
    check('completed_samples', completed, 25, 'count', '==')
    check('sample_exceptions', len(report['details']['sample_failures']), 0, 'count', '==')
    for metric, label, limit, unit in [
        ('closure', 'physical_point_closure_max_abs', 1e-8, 'm'),
        ('augmented_closure', 'pacdm_full_SE3_residual_max_abs', 1e-8, 'mixed SI'),
        ('JN', 'pacdm_JN_max_abs', 1e-9, 'mixed SI'),
        ('physical_JN', 'physical_point_JN_max_abs', 1e-8, 'mixed SI'),
        ('forward_translation', 'forward_pose_translation_error', 2e-8, 'm'),
        ('forward_rotation', 'forward_pose_rotation_matrix_error', 2e-7, 'dimensionless'),
        ('forward_all_coordinates', 'forward_physical_coordinate_error', 2e-7, 'mixed SI'),
        ('pin_fk_position', 'pinocchio_CMG_FK_position_error', 2e-11, 'm'),
        ('pin_fk_rotation', 'pinocchio_CMG_FK_rotation_error', 2e-11, 'dimensionless'),
        ('mj_fk_position', 'mujoco_CMG_FK_position_error', 2e-11, 'm'),
        ('mj_fk_rotation', 'mujoco_CMG_FK_rotation_error', 2e-11, 'dimensionless'),
        ('mass', 'native_mass_matrix_scaled_error', 2e-9, 'dimensionless'),
        ('bias', 'native_nonzero_velocity_bias_scaled_error', 2e-9, 'dimensionless'),
        ('inverse', 'native_nonzero_acceleration_RNEA_scaled_error', 2e-9, 'dimensionless'),
        ('pin_rnea_identity', 'pinocchio_RNEA_mass_bias_identity_error', 2e-9, 'dimensionless'),
        ('mj_rnea_identity', 'mujoco_RNEA_mass_bias_identity_error', 2e-9, 'dimensionless'),
        ('reduced_mass', 'reduced_mass_matrix_scaled_error', 2e-9, 'dimensionless'),
        ('actuator_effort_map', 'mujoco_actuator_effort_map_error', 1e-10, 'N or Nm'),
        ('reduced_actuator_map', 'reduced_actuator_identity_error', 1e-12, 'dimensionless'),
        ('virtual_work', 'actuator_virtual_power_error', 1e-9, 'W'),
        ('N_finite_difference', 'PACDM_map_vs_geometric_finite_difference', 2e-6, 'mixed SI'),
        ('constrained_acceleration', 'PACDM_vs_full_KKT_acceleration_scaled_error', 3e-6, 'dimensionless'),
        ('acceleration_closure', 'physical_acceleration_closure_error', 2e-6, 'm/s^2'),
    ]:
        check(label, maxima.get(metric), limit, unit)
    for metric, label, limit, unit, relation in [
        ('rank_augmented', 'minimum_augmented_constraint_rank', 36, 'count', '=='),
        ('rank_passive', 'minimum_passive_constraint_rank', 36, 'count', '=='),
        ('rank_point', 'minimum_physical_point_constraint_rank', 18, 'count', '=='),
        ('pacdm_rcond', 'minimum_selected_passive_reciprocal_condition', 1e-10, 'dimensionless', '>='),
        ('joint_limit_margin', 'minimum_physical_joint_limit_margin', 0., 'mixed SI', '>='),
        ('mass_min_eigenvalue', 'minimum_tree_mass_eigenvalue', 1e-8, 'mixed SI', '>='),
        ('reduced_mass_min_eigenvalue', 'minimum_reduced_mass_eigenvalue', 1e-8, 'kg', '>='),
    ]:
        check(label, minima.get(metric), limit, unit, relation)
    report['passed'] = bool(checks) and all(item['passed'] for item in checks.values())
    return report
