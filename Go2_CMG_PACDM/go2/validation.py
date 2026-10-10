"""Independent native free-quaternion / scalar-CMG mechanics audits.

The six base chart variables are unactuated. Native velocity transforms
require the convective E-dot term. Support closures are ideal no-slip
contact modes, not permanent robot joints or contact-force feasibility.
"""
from pathlib import Path
import hashlib
import json
import platform

import mujoco
import numpy as np
import pinocchio as pin
import scipy
from scipy.spatial.transform import Rotation

from vendor.pacdm_original import PACDM, PointGraph
from .pin_backend import PinBackend
from . import model as source

ROOT = Path(__file__).resolve().parents[1]
CORE_SHA256 = 'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de'


def _error(value, reference):
    return float(np.linalg.norm(np.asarray(value)-reference, ord=np.inf) / max(1., np.linalg.norm(reference, ord=np.inf)))


def _fd(function, q, step=2e-7):
    columns = []
    for i in range(len(q)):
        d = np.zeros(len(q)); d[i] = step
        columns.append((np.asarray(function(q+d))-function(q-d))/(2*step))
    return np.stack(columns, axis=-1)


class _Audit:
    def __init__(self, output):
        self.output = output
        self.checks, self.maxima, self.minima = {}, {}, {}
        self.details = {'sample_failures': []}

    def check(self, name, value, limit, unit='dimensionless', relation='<='):
        value = float(value)
        ok = bool(np.isfinite(value) and (value <= limit if relation == '<=' else value >= limit if relation == '>=' else value == limit))
        self.checks[name] = dict(passed=ok, value=value if np.isfinite(value) else None, limit=limit, unit=unit, relation=relation)

    def maximum(self, name, value):
        self.maxima[name] = max(self.maxima.get(name, -np.inf), float(value))

    def minimum(self, name, value):
        self.minima[name] = min(self.minima.get(name, np.inf), float(value))

    def finish(self):
        self.details.update(maxima=self.maxima, minima=self.minima)
        report = dict(passed=bool(self.checks and all(c['passed'] for c in self.checks.values())), passed_checks=sum(c['passed'] for c in self.checks.values()), total_checks=len(self.checks), checks=self.checks, details=self.details)
        if self.output is not None:
            path = Path(self.output); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return report


def validate_mechanics(cmg=None, output=None):
    """Compare authored-CMG/Pinocchio dynamics with native MuJoCo."""
    cmg = source.build_model() if cmg is None else cmg
    audit = _Audit(output)
    check, maximum, minimum = audit.check, audit.maximum, audit.minimum
    audit.details.update(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, mujoco=mujoco.__version__, pinocchio=pin.__version__), coordinate_order=cmg['coordinate_ids'], random_seed=20260923, source_xml=str(source.SOURCE_XML.relative_to(ROOT)), scope='Finite numerical rigid-body checks; no hardware identification or Pinocchio contact-integrator validation.', methods=['Authored XML inertias compiled independently through CMG into Pinocchio.', 'M_chart=E.T M_native E; h_chart=E.T(h_native+M_native E_dot qdot); source armature separate.', 'Native quaternion differentiation verifies the free-base velocity map.', 'Twelve source torque motors act on leg hinges; six base chart stages are unactuated.'])
    try:
        backend = PinBackend(cmg)
        native = mujoco.MjModel.from_xml_path(str(source.SOURCE_XML)); data = mujoco.MjData(native)
        ids = cmg['coordinate_ids']; home = np.asarray(cmg['q_reference'])
        joints = {j['id']: j for j in cmg['joints']}
        jids = np.array([mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ids[6:]])
        for name, value, expected in [('native_configuration_dimension', native.nq, 19), ('native_velocity_dimension', native.nv, 18), ('CMG_scalar_coordinates', len(ids), 18), ('Pinocchio_velocity_dimension', backend.nv, 18), ('source_leg_motors', native.nu, 12), ('source_free_joints', np.count_nonzero(native.jnt_type==mujoco.mjtJoint.mjJNT_FREE), 1), ('source_permanent_equalities', native.neq, 0), ('CMG_permanent_point_closures', len(cmg.get('closures', [])), 0)]:
            check(name, value, expected, 'count', '==')
        check('original_PACDM_core_hash', int(hashlib.sha256((ROOT/'vendor'/'pacdm_original.py').read_bytes()).hexdigest()==CORE_SHA256), 1, 'boolean', '==')
        if native.nq != 19 or native.nv != 18 or np.any(jids < 0):
            raise ValueError('Source free-joint dimensions or names changed')
        check('source_hinge_qpos_order', np.max(abs(native.jnt_qposadr[jids]-np.arange(7, 19))), 0, 'count', '==')
        check('source_hinge_velocity_order', np.max(abs(native.jnt_dofadr[jids]-np.arange(6, 18))), 0, 'count', '==')
        armature, damping, friction = [np.asarray(cmg[name]) for name in ('armature', 'damping', 'frictionloss')]
        for name, value, expected in [('source_armature', armature, native.dof_armature), ('source_damping', damping, native.dof_damping), ('source_frictionloss', friction, native.dof_frictionloss)]:
            check(name, np.max(abs(value-expected)), 1e-12, 'generalized SI')
        check('base_chart_armature', np.max(abs(armature[:6])), 0., 'generalized inertia', '==')
        check('base_chart_damping', np.max(abs(damping[:6])), 0., 'generalized damping', '==')
        lower = np.array([joints[name]['limits']['lower'] for name in ids[6:]])
        upper = np.array([joints[name]['limits']['upper'] for name in ids[6:]])
        check('source_hinge_lower_limits', np.max(abs(lower-native.jnt_range[jids, 0])), 1e-12, 'rad')
        check('source_hinge_upper_limits', np.max(abs(upper-native.jnt_range[jids, 1])), 1e-12, 'rad')
        B = np.asarray(cmg['actuation']['moment_matrix'])
        ctrlrange = np.asarray([a['ctrl_range'] for a in cmg['actuation']['actuators']])
        for name, value in [('source_motor_control_ranges', np.max(abs(ctrlrange-native.actuator_ctrlrange))), ('zero_base_actuator_map', np.max(abs(B[:6]))), ('identity_leg_actuator_map', np.max(abs(B[6:]-np.eye(12)))), ('native_actuators_target_leg_hinges', np.max(abs(native.actuator_trnid[:, 0]-jids))), ('native_torque_motor_gain', np.max(abs(native.actuator_gainprm[:, 0]-1.))), ('native_torque_motor_zero_bias', np.max(abs(native.actuator_biasprm)))]:
            check(name, value, 1e-12)
        check('source_home_configuration', np.max(abs(source.native_qpos_from_chart(home)-native.key_qpos[0])), 1e-12, 'mixed SI')
        manifest = json.loads((ROOT/'upstream'/'PROVENANCE.json').read_text())
        mismatch = []
        for record in manifest['files']:
            path = ROOT/'upstream'/record['path']
            if not path.exists() or path.stat().st_size != record['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                mismatch.append(record['path'])
        check('pinned_source_hash_mismatches', len(mismatch), 0, 'count', '==')
        check('source_XML_CMG_hash', int(cmg['source']['sha256']==hashlib.sha256(source.SOURCE_XML.read_bytes()).hexdigest()), 1, 'boolean', '==')
        audit.details.update(source_hash_mismatches=mismatch, source_file_count_verified=len(manifest['files']), source_commit=manifest['commit'], physical_mobility=18, actuator_count=12, armature=armature.tolist(), source_motor_control_ranges=ctrlrange.tolist())
        for body in cmg['bodies']:
            if body['mass_kg'] == 0.:
                continue
            bid = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, body['id'])
            rotation = Rotation.from_quat(native.body_iquat[bid][[1, 2, 3, 0]]).as_matrix()
            maximum('source_body_mass', abs(body['mass_kg']-native.body_mass[bid]))
            maximum('source_body_COM', np.max(abs(np.asarray(body['com_m'])-native.body_ipos[bid])))
            maximum('source_body_inertia', np.max(abs(np.asarray(body['inertia_kg_m2'])-rotation@np.diag(native.body_inertia[bid])@rotation.T)))
        check('total_source_mass', abs(cmg['total_mass_kg']-sum(native.body_mass)), 1e-10, 'kg')
        tree = PointGraph(cmg, home)
        from .contact import FootKinematics
        feet = FootKinematics(cmg, home)
    except Exception as error:
        audit.details['initialization_error'] = f'{type(error).__name__}: {error}'
        check('initialization_completed', 0, 1, 'boolean', '==')
        return audit.finish()
    rng = np.random.default_rng(20260923)
    samples = [home.copy()]
    for _ in range(11):
        q = home.copy(); q[:3] = rng.uniform([-.3, -.3, .3], [.3, .3, .8]); q[3:6] = rng.uniform([-.9, -.5, -.5], [.9, .5, .5]); q[6:] = lower+(upper-lower)*rng.uniform(.2, .8, 12)
        samples.append(q)
    completed = 0
    for index, q in enumerate(samples):
        try:
            v, a = rng.uniform(-.8, .8, 18), rng.uniform(-1.2, 1.2, 18)
            E = source.velocity_map(q); convective = source.convective_velocity(q, v); eps = 1e-5
            v_fd = np.empty(18)
            mujoco.mj_differentiatePos(native, v_fd, 2*eps, source.native_qpos_from_chart(q-eps*v), source.native_qpos_from_chart(q+eps*v))
            maximum('velocity_map_native_quaternion_FD', np.max(abs(E@v-v_fd)))
            maximum('velocity_map_dot_FD', np.max(abs((source.velocity_map(q+eps*v)-source.velocity_map(q-eps*v))/(2*eps)-source.velocity_map_dot(q, v))))
            maximum('chart_configuration_roundtrip', np.max(abs(source.chart_from_native(source.native_qpos_from_chart(q))-q)))
            maximum('chart_velocity_roundtrip', np.max(abs(source.chart_velocity_from_native(q, E@v)-v)))
            minimum('velocity_map_singular_value', np.linalg.svd(E, compute_uv=False)[-1])
            data.qpos[:] = source.native_qpos_from_chart(q); data.qvel[:] = E@v; data.ctrl[:] = 0.
            mujoco.mj_forward(native, data)
            Mnative = np.empty((18, 18)); mujoco.mj_fullM(native, Mnative, data.qM)
            hnative = data.qfrc_bias.copy(); M = backend.mass(q)+np.diag(armature); h = backend.bias(q, v)
            expected_M = E.T@Mnative@E; expected_h = E.T@(hnative+Mnative@convective)
            maximum('mass_velocity_congruence', _error(M, expected_M)); maximum('bias_with_convective_term', _error(h, expected_h))
            maximum('negative_missing_convective_term', np.max(abs(h-E.T@hnative))); maximum('negative_missing_armature', np.max(abs(backend.mass(q)-expected_M)))
            minimum('mass_eigenvalue', np.linalg.eigvalsh(M)[0])
            data.qacc[:] = E@a+convective
            effort = np.empty(18); mujoco.mj_rne(native, data, 1, effort)
            inverse = backend.inverse(q, v, a)
            maximum('rigid_inverse_dynamics', _error(inverse, E.T@effort))
            maximum('full_inverse_dynamics', _error(inverse+armature*a, E.T@(Mnative@(E@a+convective)+hnative)))
            maximum('RNEA_identity', _error(inverse, backend.mass(q)@a+h)); maximum('passive_damping_force', np.max(abs(E.T@data.qfrc_passive+damping*v)))
            for controls in (ctrlrange[:, 0]*1.2, rng.uniform(ctrlrange[:, 0], ctrlrange[:, 1]), ctrlrange[:, 1]*1.2):
                data.ctrl[:] = controls; mujoco.mj_fwdActuation(native, data)
                tau = np.clip(controls, ctrlrange[:, 0], ctrlrange[:, 1])
                maximum('source_torque_motor_clipping', np.max(abs(data.actuator_force-tau)))
                maximum('source_actuator_effort_map', np.max(abs(E.T@data.qfrc_actuator-B@tau)))
                maximum('zero_native_base_motor_effort', np.max(abs(data.qfrc_actuator[:6])))
                maximum('motor_virtual_work', abs(v@(B@tau)-(E@v)@data.qfrc_actuator))
            pin_poses = backend.poses(q); graph_poses, _ = tree.poses(q)
            for name, pose in pin_poses.items():
                maximum('graph_Pinocchio_FK', np.max(abs(pose-graph_poses[name])))
                bid = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0: continue
                maximum('source_FK_position', np.linalg.norm(pose[:3, 3]-data.xpos[bid]))
                maximum('source_FK_rotation', np.linalg.norm(pose[:3, :3]-data.xmat[bid].reshape(3, 3), ord=2))
            points, point_J = feet.points_and_jacobians(q)
            for i, foot in enumerate(cmg['feet']):
                bid = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, foot['body']); gid = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_GEOM, foot['id'])
                maximum('source_foot_center', np.max(abs(points[i]-data.geom_xpos[gid])))
                maximum('source_foot_radius', abs(foot['radius_m']-native.geom_size[gid, 0]))
                Jp, Jr = np.zeros((3, 18)), np.zeros((3, 18)); mujoco.mj_jac(native, data, Jp, Jr, points[i], bid)
                maximum('toe_Jacobian_native', np.max(abs(point_J[i]-Jp@E)))
            maximum('toe_Jacobian_FD', np.max(abs(point_J-_fd(feet.points, q))))
            energies = backend.energy(q, v); mujoco.mj_energyPos(native, data); mujoco.mj_energyVel(native, data)
            maximum('potential_energy', abs(energies['potential_J']-data.energy[0]))
            maximum('kinetic_energy', abs(energies['kinetic_J']+.5*np.dot(armature*v, v)-data.energy[1]))
            maximum('kinetic_mass_identity', abs(energies['kinetic_J']+.5*np.dot(armature*v, v)-.5*v@M@v))
            if index < 3:
                maximum('potential_gradient_gravity', np.max(abs(_fd(lambda x: backend.energy(x, np.zeros(18))['potential_J'], q)-backend.bias(q, np.zeros(18)))))
            completed += 1
        except Exception as error:
            audit.details['sample_failures'].append(dict(sample=index, error=f'{type(error).__name__}: {error}'))
    audit.details.update(sample_count_requested=len(samples), sample_count_completed=completed)
    check('all_model_samples_completed', completed, len(samples), 'count', '==')
    limits = [('source_body_mass', 1e-10, 'kg'), ('source_body_COM', 1e-12, 'm'), ('source_body_inertia', 1e-10, 'kg m2'), ('velocity_map_native_quaternion_FD', 2e-8, 'mixed SI'), ('velocity_map_dot_FD', 2e-8, '1/s'), ('chart_configuration_roundtrip', 2e-10, 'mixed rad/m'), ('chart_velocity_roundtrip', 1e-12, 'mixed SI'), ('mass_velocity_congruence', 1e-9, 'normalized inf norm'), ('bias_with_convective_term', 1e-9, 'normalized inf norm'), ('rigid_inverse_dynamics', 1e-9, 'normalized inf norm'), ('full_inverse_dynamics', 1e-9, 'normalized inf norm'), ('RNEA_identity', 1e-10, 'normalized inf norm'), ('passive_damping_force', 1e-10, 'mixed SI'), ('source_torque_motor_clipping', 1e-10, 'Nm'), ('source_actuator_effort_map', 1e-10, 'mixed N/Nm'), ('zero_native_base_motor_effort', 1e-12, 'mixed N/Nm'), ('motor_virtual_work', 1e-10, 'W'), ('graph_Pinocchio_FK', 2e-9, 'matrix entries'), ('source_FK_position', 2e-9, 'm'), ('source_FK_rotation', 2e-9, 'matrix norm'), ('source_foot_center', 2e-9, 'm'), ('source_foot_radius', 1e-12, 'm'), ('toe_Jacobian_native', 2e-9, 'mixed SI'), ('toe_Jacobian_FD', 2e-8, 'mixed SI'), ('potential_energy', 2e-9, 'J'), ('kinetic_energy', 2e-9, 'J'), ('kinetic_mass_identity', 2e-9, 'J'), ('potential_gradient_gravity', 2e-6, 'mixed N/Nm')]
    for name, limit, unit in limits: check(name, audit.maxima.get(name, np.inf), limit, unit)
    check('mass_positive_definite', audit.minima.get('mass_eigenvalue', -np.inf), 1e-5, 'generalized inertia', '>=')
    check('sample_chart_nonsingularity', audit.minima.get('velocity_map_singular_value', -np.inf), .4, 'dimensionless', '>=')
    check('negative_control_omitted_convective_term_detected', audit.maxima.get('negative_missing_convective_term', 0.), 1e-3, 'mixed N/Nm', '>=')
    check('negative_control_omitted_armature_detected', audit.maxima.get('negative_missing_armature', 0.), .0099, 'generalized inertia', '>=')
    return audit.finish()


def validate_contacts(cmg=None, output=None):
    """Verify full PACDM residuals, tangent map and ideal-support KKT.

    Algebraic test multipliers need not satisfy unilateral/friction bounds;
    physical contact is evaluated in the separate simulated mission.
    """
    from .contact import ContactGraph, FootKinematics, StanceGraph
    cmg = source.build_model() if cmg is None else cmg
    audit = _Audit(output); check, maximum, minimum = audit.check, audit.maximum, audit.minimum
    audit.details.update(random_seed=20260924, support_modes=[], scope='Ideal fixed-point support and moving-toe task geometry; native unilateral contacts are validated separately.', task_chart='30 coordinates: 18 physical and 12 massless targets. 24 residual rows, rank12.', support_chart='18 physical coordinates: rank3m and mobility18-3m for m fixed toes.', orientation='Zero rotational subgroup rows; toe orientation and sphere rotation are unconstrained.')
    rng = np.random.default_rng(20260924)
    try:
        q = np.asarray(cmg['q_reference']).copy(); q[2] = .31
        backend = PinBackend(cmg); kin = FootKinematics(cmg, q); graph = ContactGraph(cmg, q); initial = kin.points(q)
        for name, value, expected in [('physical_coordinates', graph.nt, 18), ('task_chart_coordinates', graph.n, 30), ('task_independent_coordinates', len(graph.active), 18), ('task_dependent_leg_coordinates', len(graph.passive), 12)]: check(name, value, expected, 'count', '==')
        completed = 0
        for index in range(9):
            base = q[:6].copy(); base[:3] += [.012*np.sin(index*.4), .008*np.sin(index*.7), .008*np.cos(index*.5)]; base[3:6] += [.08*np.sin(index*.3), .035*np.sin(index*.5), .03*np.cos(index*.4)]
            targets = initial.copy(); targets[index%4, 2] += .02*np.sin(index*.5)**2
            solved, N, info = graph.solve_feet(base, targets, q if index==0 else None)
            lifted = np.r_[solved, targets.ravel()]; residual, J, defects = graph.residual(lifted)
            maximum('task_full_residual', np.max(abs(residual))); maximum('task_toe_error', np.max(abs(kin.points(solved)-targets)))
            maximum('task_rotational_rows', np.max(abs(residual.reshape(4, 6)[:, :3]))); maximum('task_rotational_Jacobian_rows', np.max(abs(J.reshape(4, 6, 30)[:, :3])))
            full_N = np.zeros((30, 18)); full_N[:18] = N; full_N[18:, 6:] = np.eye(12)
            maximum('task_tangent_residual', np.max(abs(J@full_N))); maximum('task_residual_Jacobian_FD', np.max(abs(J-_fd(lambda x: graph.residual(x)[0], lifted))))
            maximum('task_constraint_rank_error', abs(np.linalg.matrix_rank(J, tol=1e-9)-12)); minimum('task_mapping_rcond', info['rcond'])
            maximum('task_active_identity', np.max(abs(full_N[graph.active]-np.eye(18))))
            if index < 3:
                direction = rng.uniform(-.05, .05, 18); step = 2e-3
                plus, pinfo = graph.solver.correct(lifted[graph.active]+step*direction, lifted[graph.passive]+step*(full_N@direction)[graph.passive], None, np.asarray(info['rows']), maxiter=30)
                minus, minfo = graph.solver.correct(lifted[graph.active]-step*direction, lifted[graph.passive]-step*(full_N@direction)[graph.passive], None, np.asarray(info['rows']), maxiter=30)
                if not pinfo['success'] or not minfo['success']: raise ValueError('Nonlinear FD chart correction failed')
                maximum('task_tangent_nonlinear_FD', np.max(abs((plus-minus)/(2*step)-full_N@direction)))
            completed += 1
        check('all_task_samples_completed', completed, 9, 'count', '==')
        for stance in ([0, 1, 2, 3], [0, 1, 2], [1, 2, 3], [0, 3], [1, 2]):
            support = StanceGraph(cmg, solved, stance); r, J, _ = support.residual(solved)
            N, info = PACDM(support).mapping(solved); nc = len(stance); name = '_'.join(map(str, stance))
            check('support_'+name+'_rank', np.linalg.matrix_rank(J, tol=1e-9), 3*nc, 'count', '==')
            check('support_'+name+'_mobility', N.shape[1] if N is not None else 0, 18-3*nc, 'count', '==')
            check('support_'+name+'_PACDM_mapping', int(info['success']), 1, 'boolean', '==')
            if N is None or not info['success']: raise ValueError('Support mapping failed')
            minimum('support_mapping_rcond', info['rcond']); maximum('support_tangent_residual', np.max(abs(J@N)))
            rows = np.asarray(info['rows']); A = J[rows]; v = N@rng.uniform(-.06, .06, N.shape[1]); eps = 1e-5
            Jdot = (support.residual(solved+eps*v)[1]-support.residual(solved-eps*v)[1])/(2*eps)
            curvature = np.zeros(18); curvature[support.passive] = np.linalg.solve(A[:, support.passive], -(Jdot@v)[rows])
            maximum('support_acceleration_constraint', np.max(abs(J@curvature+Jdot@v)))
            M = backend.mass(solved)+np.diag(cmg['armature']); h = backend.bias(solved, v); tau = np.r_[np.zeros(6), rng.uniform(-4., 4., 12)]
            reduced_M = N.T@M@N; a = N@np.linalg.solve(reduced_M, N.T@(tau-h-M@curvature))+curvature
            KKT = np.block([[M, -A.T], [A, np.zeros((len(rows), len(rows)))]])
            rhs = np.r_[tau-h, -(Jdot@v)[rows]]; solution = np.linalg.solve(KKT, rhs)
            maximum('support_reduced_vs_KKT', _error(a, solution[:18])); maximum('support_full_KKT_residual', np.max(abs(KKT@solution-rhs)))
            maximum('support_projected_dynamics', np.max(abs(N.T@(M@a+h-tau)))); maximum('support_full_acceleration_closure', np.max(abs(J@a+Jdot@v)))
            minimum('support_reduced_mass_eigenvalue', np.linalg.eigvalsh(reduced_M)[0])
            maximum('support_reaction_virtual_work', abs(v@(A.T@rng.uniform(-10., 10., len(rows)))))
            audit.details['support_modes'].append(dict(stance=[cmg['feet'][i]['id'] for i in stance], rank=int(np.linalg.matrix_rank(J, tol=1e-9)), mobility=int(N.shape[1]), residual_rows=len(r), rcond=float(info['rcond'])))
    except Exception as error:
        audit.details['sample_failures'].append(dict(error=f'{type(error).__name__}: {error}')); check('contact_audit_completed', 0, 1, 'boolean', '==')
    limits = [('task_full_residual', 1e-8, 'm'), ('task_toe_error', 1e-8, 'm'), ('task_rotational_rows', 0., 'rad'), ('task_rotational_Jacobian_rows', 0., 'mixed SI'), ('task_tangent_residual', 1e-9, 'mixed SI'), ('task_residual_Jacobian_FD', 2e-8, 'mixed SI'), ('task_constraint_rank_error', 0., 'count'), ('task_active_identity', 1e-12, 'dimensionless'), ('task_tangent_nonlinear_FD', 3e-5, 'mixed SI'), ('support_tangent_residual', 1e-9, 'mixed SI'), ('support_acceleration_constraint', 1e-9, 'm/s2'), ('support_reduced_vs_KKT', 1e-8, 'normalized inf norm'), ('support_full_KKT_residual', 1e-8, 'mixed SI'), ('support_projected_dynamics', 1e-8, 'mixed N/Nm'), ('support_full_acceleration_closure', 1e-8, 'm/s2'), ('support_reaction_virtual_work', 1e-9, 'W')]
    for name, limit, unit in limits: check(name, audit.maxima.get(name, np.inf), limit, unit)
    check('task_mapping_condition', audit.minima.get('task_mapping_rcond', -np.inf), 1e-6, 'dimensionless', '>=')
    check('support_mapping_condition', audit.minima.get('support_mapping_rcond', -np.inf), 1e-6, 'dimensionless', '>=')
    check('support_reduced_mass_positive_definite', audit.minima.get('support_reduced_mass_eigenvalue', -np.inf), 1e-5, 'generalized inertia', '>=')
    return audit.finish()


def validate(output_dir=None):
    path = None if output_dir is None else Path(output_dir)
    mechanics = validate_mechanics(output=None if path is None else path/'mechanics_validation.json')
    contacts = validate_contacts(output=None if path is None else path/'contact_validation.json')
    return dict(passed=mechanics['passed'] and contacts['passed'], mechanics=mechanics, contacts=contacts, passed_checks=mechanics['passed_checks']+contacts['passed_checks'], total_checks=mechanics['total_checks']+contacts['total_checks'])


if __name__ == '__main__':
    result = validate(ROOT/'results')
    print(json.dumps({name: result[name] for name in ('passed', 'passed_checks', 'total_checks')}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
