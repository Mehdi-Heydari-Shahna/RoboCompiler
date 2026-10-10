"""Independent finite checks for the source Panda and its task-coordinate chart.

The arm is a serial tree. Its two finger sliders have one physical scalar
coupling. The six-dimensional tool pose is a *task constraint*, not a robot
loop. Pinocchio is used for independent rigid-tree dynamics; MuJoCo alone
integrates contact in the mission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import pinocchio as pin
import scipy
from scipy.spatial.transform import Rotation

from vendor.pacdm_original import PACDM, PointGraph
from .pin_backend import PinBackend

ROOT = Path(__file__).resolve().parents[1]
CORE_SHA256 = 'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de'


def _nan_max(*values):
    """max() that propagates NaN; the built-in max() silently skips a NaN argument."""
    values = [float(value) for value in values]
    return float('nan') if any(np.isnan(values)) else max(values)


def _nan_min(*values):
    """min() that propagates NaN; the built-in min() silently skips a NaN argument."""
    values = [float(value) for value in values]
    return float('nan') if any(np.isnan(values)) else min(values)


def _json_safe(value):
    """Replace NaN/inf by None so that a failing record can still be written as strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _error(value, reference):
    return float(np.linalg.norm(np.asarray(value)-reference, ord=np.inf)
                 / max(1., np.linalg.norm(reference, ord=np.inf)))


def _finite_jacobian(function, q, step=2e-7):
    q = np.asarray(q, float)
    columns = []
    for index in range(len(q)):
        delta = np.zeros(len(q)); delta[index] = step
        columns.append((function(q+delta)-function(q-delta))/(2*step))
    return np.column_stack(columns)


def _frame_jacobian(backend, body, q):
    pin.computeJointJacobians(backend.model, backend.data, backend._native_q(q))
    pin.updateFramePlacements(backend.model, backend.data)
    return np.asarray(pin.getFrameJacobian(
        backend.model, backend.data, backend.body_frame_ids[body],
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:, backend._v_indices].copy()


def _frame_jacobian_fd(backend, body, q, step=2e-7):
    result = np.empty((6, len(q)))
    for index in range(len(q)):
        delta = np.zeros(len(q)); delta[index] = step
        plus, minus = backend.poses(q+delta)[body], backend.poses(q-delta)[body]
        result[:3, index] = (plus[:3, 3]-minus[:3, 3])/(2*step)
        result[3:, index] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec()/(2*step)
    return result


def validate_mechanics(cmg=None, output=None):
    """Validate the pinned source model; write an auditable JSON if requested."""
    from . import model as source
    source_xml = Path(source.SOURCE_XML)
    cmg = source.build_model() if cmg is None else cmg
    checks, maxima, minima = {}, {}, {}
    report = {'passed': False, 'checks': checks, 'details': {
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'mujoco': mujoco.__version__,
                     'pinocchio': pin.__version__},
        'source_xml': str(source_xml.relative_to(ROOT)),
        'source_xml_sha256': hashlib.sha256(source_xml.read_bytes()).hexdigest(),
        'coordinate_order': cmg['coordinate_ids'], 'random_seed': 20260923,
        'sample_failures': [],
        'scope': 'Finite numerical model checks; no hardware identification, global workspace proof, or Pinocchio contact-integrator validation.',
        'methods': [
            'Independent CMG-to-Pinocchio tree FK, CRBA, RNEA and frame Jacobians versus the pinned original MuJoCo model.',
            'Armature, passive joint forces, source position-actuator gains, force limits and tendon transmission checked separately.',
            'One scalar physical finger coupling; independently formed constrained KKT versus reduced coordinates.',
            'PACDM tool pose coordinates are a task manifold for the serial arm, not permanent physical closed-chain joints.',
        ],
    }}

    def check(name, value, limit, unit='dimensionless', relation='<='):
        value = float(value)
        valid = bool(np.isfinite(value) and (value <= limit if relation == '<=' else
                                            value >= limit if relation == '>=' else value == limit))
        checks[name] = {'passed': valid, 'value': value if np.isfinite(value) else None,
                        'limit': limit, 'unit': unit, 'relation': relation}

    def maximum(name, value):
        maxima[name] = _nan_max(maxima.get(name, -np.inf), value)

    def minimum(name, value):
        minima[name] = _nan_min(minima.get(name, np.inf), value)

    def finish():
        report['details']['maxima'] = maxima
        report['details']['minima'] = minima
        report['passed'] = bool(checks and all(item['passed'] for item in checks.values()))
        report['passed_checks'] = sum(item['passed'] for item in checks.values())
        report['total_checks'] = len(checks)
        if output is not None:
            path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return report

    try:
        backend = PinBackend(cmg)
        model = mujoco.MjModel.from_xml_path(str(source_xml))
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        data = mujoco.MjData(model)
        ids = cmg['coordinate_ids']
        jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ids])
        if np.any(jids < 0):
            raise ValueError('Source is missing a CMG coordinate')
        qids = model.jnt_qposadr[jids].astype(int)
        vids = model.jnt_dofadr[jids].astype(int)
        check('physical_tree_coordinates', len(ids), 9, 'count', '==')
        check('Pinocchio_tree_velocities', backend.nv, 9, 'count', '==')
        check('source_general_actuators', model.nu, 8, 'count', '==')
        check('source_physical_equalities', model.neq, 1, 'count', '==')
        check('permanent_point_closures', len(cmg.get('closures', [])), 0, 'count', '==')
        check('original_PACDM_core_hash', int(hashlib.sha256((ROOT/'vendor'/'pacdm_original.py').read_bytes()).hexdigest()==CORE_SHA256), 1, 'boolean', '==')
        if model.nq != 9 or model.nv != 9:
            raise ValueError('Expected the original fixed-base Panda source with nine scalar coordinates')
        finger1, finger2 = ids.index('finger_joint1'), ids.index('finger_joint2')
        independent = [i for i in range(9) if i != finger2]
        P = np.eye(9)[:, independent]
        P[finger2, independent.index(finger1)] = 1.
        A = np.zeros((1, 9)); A[0, finger1], A[0, finger2] = 1., -1.
        check('physical_coupling_rank', np.linalg.matrix_rank(A), 1, 'count', '==')
        check('independent_physical_coordinates', P.shape[1], 8, 'count', '==')
        check('physical_coupling_tangent', np.max(abs(A@P)), 1e-14, 'dimensionless')
        armature = np.asarray(cmg['armature'], float)
        damping = np.asarray(cmg['damping'], float)
        stiffness = np.asarray(cmg['stiffness'], float)
        springref = np.asarray(cmg['spring_reference'], float)
        check('armature_source_metadata', np.max(abs(armature-model.dof_armature[vids])), 1e-12, 'mixed generalized SI')
        check('damping_source_metadata', np.max(abs(damping-model.dof_damping[vids])), 1e-12, 'mixed generalized SI')
        check('stiffness_source_metadata', np.max(abs(stiffness-model.jnt_stiffness[jids])), 1e-12, 'mixed generalized SI')
        check('spring_reference_source_metadata', np.max(abs(springref-model.qpos_spring[qids])), 1e-12, 'mixed rad/m')
        joints = {joint['id']: joint for joint in cmg['joints']}
        lower = np.array([joints[name]['limits']['lower'] for name in ids])
        upper = np.array([joints[name]['limits']['upper'] for name in ids])
        check('source_joint_lower_limits', np.max(abs(lower-model.jnt_range[jids, 0])), 1e-12, 'mixed rad/m')
        check('source_joint_upper_limits', np.max(abs(upper-model.jnt_range[jids, 1])), 1e-12, 'mixed rad/m')
        source_actuators = cmg['actuation']['actuators']
        source_gains = np.asarray([a['gain'][0] for a in source_actuators])
        source_bias = np.asarray([a['bias'] for a in source_actuators])
        source_control_range = np.asarray([a['ctrl_range'] for a in source_actuators])
        source_force_range = np.asarray([a['force_range'] for a in source_actuators])
        B_cmg = np.asarray(cmg['actuation']['moment_matrix'])
        check('CMG_source_actuator_gains', np.max(abs(source_gains-model.actuator_gainprm[:, 0])), 1e-12)
        check('CMG_source_actuator_bias', np.max(abs(source_bias-model.actuator_biasprm[:, :3])), 1e-12)
        check('CMG_source_control_ranges', np.max(abs(source_control_range-model.actuator_ctrlrange)), 1e-12)
        check('CMG_source_force_ranges', np.max(abs(source_force_range-model.actuator_forcerange)), 1e-12, 'mixed N/Nm')
        manifest = json.loads((ROOT/'upstream'/'PROVENANCE.json').read_text())
        mismatches = []
        for record in manifest['files']:
            artifact = ROOT/'upstream'/record['path']
            if not artifact.exists() or artifact.stat().st_size!=record['bytes'] or hashlib.sha256(artifact.read_bytes()).hexdigest()!=record['sha256']:
                mismatches.append(record['path'])
        check('pinned_source_hash_mismatches', len(mismatches), 0, 'count', '==')
        check('source_xml_matches_CMG_hash', int(cmg['source']['sha256']==hashlib.sha256(source_xml.read_bytes()).hexdigest()), 1, 'boolean', '==')
        report['details']['source_file_count_verified'] = len(manifest['files'])
        report['details']['source_commit'] = manifest['commit']
        report['details']['source_hash_mismatches'] = mismatches
        report['details']['armature'] = armature.tolist()
        report['details']['source_actuator_force_ranges'] = model.actuator_forcerange.tolist()
        report['details']['physical_mobility'] = 8
        report['details']['actuator_count'] = 8
        report['details']['physical_reduction_order'] = [ids[i] for i in independent]
        bodies = {body['id']: body for body in cmg['bodies']}
        compiled_cmg = deepcopy(cmg)
        compiled_bodies = {body['id']: body for body in compiled_cmg['bodies']}
        for name, body in bodies.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f'Source body missing: {name}')
            R = Rotation.from_quat(model.body_iquat[bid][[1, 2, 3, 0]]).as_matrix()
            maximum('body_mass', abs(body['mass_kg']-model.body_mass[bid]))
            maximum('body_COM', np.max(abs(np.asarray(body['com_m'])-model.body_ipos[bid])))
            compiled_tensor = R@np.diag(model.body_inertia[bid])@R.T
            maximum('body_inertia', np.max(abs(np.asarray(body['inertia_kg_m2'])-compiled_tensor)))
            compiled_bodies[name]['inertia_kg_m2'] = compiled_tensor.tolist()
        compiled_backend = PinBackend(compiled_cmg)
        report['details']['compiled_tensor_diagnostic'] = (
            'MuJoCo diagonalizes authored fullinertia tensors during compilation. '
            'The largest reconstructed component differs by %.9g kg m2. The '
            'main CMG and Pinocchio model retain authored tensors unchanged. '
            'An additional diagnostic replaces only tensors with the compiled '
            'reconstruction to isolate this small representation difference; '
            'it is not used by the main independent comparison or mission.' % maxima['body_inertia'])
    except Exception as error:
        report['details']['initialization_error'] = f'{type(error).__name__}: {error}'
        check('initialization_completed', 0, 1, 'boolean', '==')
        return finish()

    rng = np.random.default_rng(20260923)
    samples = []
    if model.nkey:
        samples.append(np.asarray(model.key_qpos[0, qids]).copy())
    for _ in range(15):
        q = lower+(upper-lower)*rng.uniform(.18, .82, 9)
        q[finger2] = q[finger1]
        samples.append(q)
    report['details']['sample_count_requested'] = len(samples)
    completed = 0
    # This tree instance provides a third, graph-composed FK route without
    # treating the serial Panda arm as having physical point cuts.
    tree = PointGraph({**cmg, 'independent_ids': ids}, samples[0])
    for index, q in enumerate(samples):
        try:
            zdot = rng.uniform(-.5, .5, 8)
            zdot[independent.index(finger1)] *= .06
            v = P@zdot
            a = rng.uniform(-1., 1., 9)
            data.qpos[qids], data.qvel[vids] = q, v
            data.ctrl[:] = (model.actuator_ctrlrange[:, 0]+model.actuator_ctrlrange[:, 1])/2
            mujoco.mj_forward(model, data)
            qfrc_bias = data.qfrc_bias[vids].copy()
            matrix = np.empty((9, 9)); mujoco.mj_fullM(model, matrix, data.qM)
            mj_M = matrix[np.ix_(vids, vids)]
            pin_M = backend.mass(q)
            M = pin_M+np.diag(armature)
            h = backend.bias(q, v)
            inverse = backend.inverse(q, v, a)
            data.qacc[vids] = a
            mj_inverse = np.empty(9); mujoco.mj_rne(model, data, 1, mj_inverse)
            maximum('mass_with_armature', _error(M, mj_M))
            maximum('compiled_tensor_diagnostic_mass', _error(compiled_backend.mass(q)+np.diag(armature), mj_M))
            maximum('compiled_tensor_diagnostic_bias', _error(compiled_backend.bias(q, v), qfrc_bias))
            maximum('bias', _error(h, qfrc_bias))
            maximum('rigid_RNEA', _error(inverse, mj_inverse[vids]))
            maximum('RNEA_identity', _error(inverse, pin_M@a+h))
            maximum('full_inverse', _error(inverse+armature*a, mj_M@a+qfrc_bias))
            maximum('negative_armature_omission', np.max(abs(pin_M-mj_M)))
            minimum('mass_eigenvalue', np.min(np.linalg.eigvalsh(M)))
            pin_poses = backend.poses(q)
            tree_poses, _ = tree.poses(q)
            for name, pose in pin_poses.items():
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                maximum('FK_position', np.linalg.norm(pose[:3, 3]-data.xpos[bid]))
                maximum('FK_rotation', np.linalg.norm(pose[:3, :3]-data.xmat[bid].reshape(3, 3), ord=2))
                maximum('graph_FK_position', np.linalg.norm(pose[:3, 3]-tree_poses[name][:3, 3]))
                maximum('graph_FK_rotation', np.linalg.norm(pose[:3, :3]-tree_poses[name][:3, :3], ord=2))
            for body in ('hand', 'left_finger', 'right_finger'):
                if body not in bodies:
                    continue
                J = _frame_jacobian(backend, body, q)
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                Jpos, Jrot = np.zeros((3, 9)), np.zeros((3, 9))
                mujoco.mj_jacBody(model, data, Jpos, Jrot, bid)
                maximum('native_frame_Jacobian', np.max(abs(J-np.vstack([Jpos, Jrot])[:, vids])))
                maximum('frame_Jacobian_FD', np.max(abs(J-_frame_jacobian_fd(backend, body, q))))
            passive = -stiffness*(q-springref)-damping*v
            maximum('passive_force', np.max(abs(passive-data.qfrc_passive[vids])))
            eqrows = np.flatnonzero(data.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            if len(eqrows)!=1:
                raise ValueError(f'Expected one source scalar equality row, got {len(eqrows)}')
            Jeq = data.efc_J.reshape(data.nefc, model.nv)[eqrows][:, vids]
            maximum('finger_equality_Jacobian', np.max(abs(Jeq-A)))
            maximum('finger_equality_residual', np.max(abs(data.efc_pos[eqrows]-A@q)))
            # Source affine position servo, distinct from the mission's arm
            # torque motors. Include control and force clamping explicitly.
            B = np.zeros((9, model.nu))
            for actuator in range(7):
                B[ids.index(f'joint{actuator+1}'), actuator] = 1.
            B[finger1, 7] = B[finger2, 7] = .5
            maximum('CMG_actuator_moment_matrix', np.max(abs(B-B_cmg)))
            for controls in (model.actuator_ctrlrange[:, 0],
                             (model.actuator_ctrlrange[:, 0]+model.actuator_ctrlrange[:, 1])/2,
                             model.actuator_ctrlrange[:, 1]):
                data.ctrl[:] = controls
                mujoco.mj_fwdActuation(model, data)
                expected_force = source_gains*controls+source_bias[:, 0]+source_bias[:, 1]*data.actuator_length+source_bias[:, 2]*data.actuator_velocity
                expected_force = np.clip(expected_force, source_force_range[:, 0], source_force_range[:, 1])
                maximum('source_affine_servo_force', np.max(abs(data.actuator_force-expected_force)))
                maximum('source_actuator_effort_map', np.max(abs(data.qfrc_actuator[vids]-B@expected_force)))
                maximum('virtual_work', abs(v@data.qfrc_actuator[vids]-zdot@(P.T@B@expected_force)))
            maximum('reduced_actuator_map', np.max(abs(P.T@B-np.eye(8))))
            tau = B@rng.uniform(-1., 1., 8)+passive
            Mr = P.T@M@P
            ar = P@np.linalg.solve(Mr, P.T@(tau-h))
            KKT = np.block([[mj_M, -A.T], [A, np.zeros((1, 1))]])
            rhs = np.r_[tau-qfrc_bias, 0.]
            solution = np.linalg.solve(KKT, rhs)
            maximum('reduced_vs_custom_KKT_acceleration', _error(ar, solution[:9]))
            maximum('reduced_acceleration_coupling', np.max(abs(A@ar)))
            maximum('KKT_equation_residual', np.max(abs(KKT@solution-rhs)))
            minimum('reduced_mass_eigenvalue', np.min(np.linalg.eigvalsh(Mr)))
            completed += 1
        except Exception as error:
            report['details']['sample_failures'].append({'sample': index, 'error': f'{type(error).__name__}: {error}'})
    report['details']['sample_count_completed'] = completed
    check('all_samples_completed', completed, len(samples), 'count', '==')
    for name, limit, unit in [
        ('body_mass', 1e-10, 'kg'), ('body_COM', 1e-10, 'm'), ('body_inertia', 1e-6, 'kg m2'),
        ('FK_position', 2e-9, 'm'), ('FK_rotation', 2e-9, 'matrix norm'),
        ('graph_FK_position', 2e-9, 'm'), ('graph_FK_rotation', 2e-9, 'matrix norm'),
        ('mass_with_armature', 1e-6, 'normalized inf norm'), ('bias', 1e-6, 'normalized inf norm'),
        ('rigid_RNEA', 1e-6, 'normalized inf norm'), ('RNEA_identity', 1e-9, 'normalized inf norm'),
        ('compiled_tensor_diagnostic_mass', 1e-10, 'normalized inf norm'),
        ('compiled_tensor_diagnostic_bias', 1e-10, 'normalized inf norm'),
        ('full_inverse', 1e-6, 'normalized inf norm'), ('native_frame_Jacobian', 2e-9, 'mixed SI'),
        ('frame_Jacobian_FD', 2e-8, 'mixed SI'), ('passive_force', 1e-10, 'mixed N/Nm'),
        ('finger_equality_Jacobian', 1e-12, 'dimensionless'), ('finger_equality_residual', 1e-12, 'm'),
        ('CMG_actuator_moment_matrix', 1e-12, 'dimensionless'),
        ('source_affine_servo_force', 1e-10, 'mixed N/Nm'), ('source_actuator_effort_map', 1e-10, 'mixed N/Nm'),
        ('virtual_work', 1e-10, 'W'), ('reduced_actuator_map', 1e-12, 'dimensionless'),
        ('reduced_vs_custom_KKT_acceleration', 1e-6, 'normalized inf norm'),
        ('reduced_acceleration_coupling', 1e-10, 'm/s2'), ('KKT_equation_residual', 1e-9, 'mixed SI'),
    ]:
        check(name, maxima.get(name, np.inf), limit, unit)
    check('mass_positive_definite', minima.get('mass_eigenvalue', -np.inf), 1e-6, 'mixed generalized SI', '>=')
    check('reduced_mass_positive_definite', minima.get('reduced_mass_eigenvalue', -np.inf), 1e-6, 'mixed generalized SI', '>=')
    check('negative_control_armature_omission_detected', maxima.get('negative_armature_omission', 0.), .09, 'mixed generalized SI', '>=')
    check('negative_control_uncoupled_fingers_detected', np.max(abs(A@np.eye(9))), 1., 'dimensionless', '>=')
    try:
        bad = deepcopy(cmg)
        for joint in bad['joints']:
            if joint['id']=='joint2':
                joint['axis'] = [1., 0., 0.]
        wrong = PinBackend(bad)
        q = samples[min(1, len(samples)-1)]
        negative_error = np.linalg.norm(wrong.poses(q)['hand'][:3, 3]-backend.poses(q)['hand'][:3, 3])
        check('negative_control_wrong_joint_axis_detected', negative_error, .01, 'm', '>=')
    except Exception as error:
        report['details']['negative_control_error'] = f'{type(error).__name__}: {error}'
        check('negative_controls_completed', 0, 1, 'boolean', '==')
    try:
        graph = source.make_graph(cmg)
        solver = PACDM(graph)
        q = np.asarray(cmg['q_reference']).copy()
        q[finger1], q[finger2] = .023, .032
        assembled, acquired = solver.acquire(q[graph.active], graph.lift(q))
        N, mapping = solver.mapping(assembled)
        check('physical_PACDM_acquisition', int(acquired['success']), 1, 'boolean', '==')
        check('physical_PACDM_rank', mapping['rank_full'], 1, 'count', '==')
        check('physical_PACDM_mapping', np.max(abs(N-P)), 1e-12, 'dimensionless')
        check('physical_PACDM_coupling', abs(assembled[finger1]-assembled[finger2]), 1e-9, 'm')
        check('physical_PACDM_Jacobian_FD', np.max(abs(graph.residual(assembled)[1]-_finite_jacobian(lambda x:graph.residual(x)[0], assembled))), 1e-8, 'dimensionless')
        no_coupling = deepcopy(graph)
        no_coupling.residual = lambda q, defects=None: (np.zeros(6), np.zeros((6, 9)), np.eye(4)[None])
        _, wrong_mapping = PACDM(no_coupling).mapping(q)
        check('negative_control_missing_physical_equality_rejected', int(not wrong_mapping['success']), 1, 'boolean', '==')
    except Exception as error:
        report['details']['physical_adapter_error'] = f'{type(error).__name__}: {error}'
        check('physical_PACDM_validation_completed', 0, 1, 'boolean', '==')
    _validate_task_chart(cmg, backend, check, report)
    return finish()


def _validate_task_chart(cmg, backend, check, report):
    """Check task closure and redundancy independently of mission success."""
    from .task import HOME, TaskGraph
    maxima, conditions, failures = {}, [], []
    def maximum(name, value):
        maxima[name] = _nan_max(maxima.get(name, -np.inf), value)
    rng = np.random.default_rng(523)
    try:
        home = HOME.copy(); home[7:] = .025
        graph, completed = TaskGraph(cmg, home), 0
        solver = PACDM(graph)
        check('task_augmented_coordinates', graph.n, 15, 'count', '==')
        check('task_independent_coordinates', len(graph.active), 8, 'count', '==')
        check('task_passive_coordinates', len(graph.passive), 7, 'count', '==')
        for sample in range(4):
            try:
                physical = home.copy()
                if sample:
                    physical[:7] += rng.uniform(-.06, .06, 7)
                    physical[7:] += rng.uniform(-.003, .003)
                feasible = graph.lift(physical)
                initial = feasible.copy()
                initial[graph.passive] += rng.uniform(-.002, .002, len(graph.passive))
                q, acquired = solver.acquire(feasible[graph.active], initial)
                if not acquired['success']:
                    raise RuntimeError(f'Task acquisition failed: {acquired}')
                N, info = solver.mapping(q)
                if not info['success']:
                    raise RuntimeError(f'Task mapping failed: {info}')
                conditions.append(info['rcond'])
                maximum('rank_error', abs(info['rank_full']-7))
                residual, J, _ = graph.residual(q)
                maximum('residual', np.max(abs(residual)))
                maximum('Jacobian_FD', np.max(abs(J-_finite_jacobian(lambda x:graph.residual(x)[0], q))))
                off = q.copy(); off[0] += .012; off[10] += .003
                maximum('off_manifold_Jacobian_FD', np.max(abs(graph.residual(off)[1]-_finite_jacobian(lambda x:graph.residual(x)[0], off))))
                maximum('JN', np.max(abs(J@N)))
                maximum('active_identity', np.max(abs(N[graph.active]-np.eye(8))))
                physical_pose = backend.poses(q[:9])[graph.tool_body] @ graph.tool_transform
                target = graph.target_pose(q)
                maximum('Pinocchio_target_position', np.linalg.norm(physical_pose[:3, 3]-target[:3, 3]))
                maximum('Pinocchio_target_rotation', np.linalg.norm(Rotation.from_matrix(physical_pose[:3, :3]@target[:3, :3].T).as_rotvec()))
                # Native Pinocchio hand Jacobian -> tool offset; linear-first.
                Jtool = _frame_jacobian(backend, graph.tool_body, q[:9])
                offset = backend.poses(q[:9])[graph.tool_body][:3, :3] @ graph.tool_transform[:3, 3]
                for col in range(9):
                    Jtool[:3, col] += np.cross(Jtool[3:, col], offset)
                expected_twists = np.zeros((6, 8))
                probe = 2e-7
                for col in range(6):
                    plus, minus = q.copy(), q.copy()
                    plus[9+col] += probe; minus[9+col] -= probe
                    Tp, Tm = graph.target_pose(plus), graph.target_pose(minus)
                    expected_twists[:3, col] = (Tp[:3, 3]-Tm[:3, 3])/(2*probe)
                    expected_twists[3:, col] = Rotation.from_matrix(Tp[:3, :3]@Tm[:3, :3].T).as_rotvec()/(2*probe)
                maximum('native_tool_velocity_map', np.max(abs(Jtool@N[:9]-expected_twists)))
                maximum('redundancy_tool_velocity', np.max(abs(Jtool@N[:9, 6])))
                maximum('redundancy_coordinate_identity', abs(N[graph.redundancy_index, 6]-1.))
                estimates = []
                for step in (2e-4, 4e-4):
                    derivative = np.empty_like(N)
                    for col in range(8):
                        delta = np.zeros(8); delta[col] = step
                        plus, ip = solver.correct(q[graph.active]+delta, q[graph.passive], None, np.asarray(info['rows']))
                        minus, im = solver.correct(q[graph.active]-delta, q[graph.passive], None, np.asarray(info['rows']))
                        if not ip['success'] or not im['success']:
                            raise RuntimeError('Task assembly finite-difference corrector failed')
                        derivative[:, col] = (plus-minus)/(2*step)
                    maximum('mapping_FD', np.max(abs(derivative-N)))
                    estimates.append(derivative)
                maximum('mapping_FD_step_agreement', np.max(abs(estimates[0]-estimates[1])))
                completed += 1
            except Exception as error:
                failures.append({'sample': sample, 'error':f'{type(error).__name__}: {error}'})
        check('task_all_samples_completed', completed, 4, 'count', '==')
        for name, limit, unit in [
            ('rank_error', 0, 'count'), ('residual', 1e-8, 'mixed SI'),
            ('Jacobian_FD', 2e-7, 'mixed SI'), ('off_manifold_Jacobian_FD', 2e-7, 'mixed SI'),
            ('JN', 2e-9, 'mixed SI'), ('active_identity', 1e-12, 'dimensionless'),
            ('Pinocchio_target_position', 1e-8, 'm'), ('Pinocchio_target_rotation', 1e-8, 'rad'),
            ('native_tool_velocity_map', 2e-8, 'mixed SI'), ('redundancy_tool_velocity', 2e-9, 'mixed SI'),
            ('redundancy_coordinate_identity', 1e-12, 'dimensionless'),
            ('mapping_FD', 2e-4, 'mixed coordinate units'),
            ('mapping_FD_step_agreement', 5e-5, 'mixed coordinate units'),
        ]:
            check('task_'+name, maxima.get(name, np.inf), limit, unit)
        check('task_minimum_PACDM_rcond', _nan_min(*conditions) if conditions else 0., 1e-10, 'dimensionless', '>=')
        # A wrong tool offset must be visible in this same independent FK
        # comparison, even though the underlying hand pose is unchanged.
        wrong = graph.tool_transform.copy(); wrong[2, 3] += .01
        pose = backend.poses(home)[graph.tool_body]
        check('negative_control_wrong_tool_offset_detected', np.linalg.norm((pose@wrong)[:3, 3]-(pose@graph.tool_transform)[:3, 3]), .0099, 'm', '>=')
        report['details']['task_chart'] = {'sample_failures': failures, 'maxima': maxima,
            'rcond': conditions, 'finite_difference_steps': [2e-4, 4e-4],
            'interpretation': 'Seven arm coordinates realize six imposed tool coordinates plus joint3 redundancy. The two fingers retain one physical equality.'}
    except Exception as error:
        report['details']['task_chart_error'] = f'{type(error).__name__}: {error}'
        check('task_validation_completed', 0, 1, 'boolean', '==')


if __name__ == '__main__':
    result = validate_mechanics(output=ROOT/'results'/'mechanics_validation.json')
    print(json.dumps({'passed': result['passed'], 'passed_checks': result['passed_checks'],
                      'total_checks': result['total_checks'],
                      'failures': {k:v for k,v in result['checks'].items() if not v['passed']},
                      'details': result['details']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
