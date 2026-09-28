"""Independent native Pinocchio checks of the unchanged PACDM reduction.

Both routes share one CMG-compiled Pinocchio rigid-body tree and its declared
inertias. They do not share the closure Jacobian/reduction: the reference
route uses six native CONTACT_3D constraints and Pinocchio constraintDynamics.
Native point Jacobians and classical accelerations independently check PACDM's
configuration, tangent, curvature, and force maps. No MuJoCo runtime is used.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pinocchio as pin

from vendor.pacdm_original import PointGraph, PACDM, skew
from .model import inverse_seed
from .pin_backend import PinBackend, _se3


class PinConstraintOracle:
    """Six unregularized point constraints assembled directly from CMG.

    Native CONTACT_3D preserves free relative rotation at each spherical cut.
    Gains Kp=Kd=0 deliberately disable stabilization: comparisons are at
    feasible q and tangent v, with the exact acceleration constraint.
    External wrench order is world-aligned [Fx,Fy,Fz,Mx,My,Mz] at the
    platform-frame origin. All public joint vectors use CMG ordering.
    """

    def __init__(self, cmg):
        self.backend = PinBackend(cmg)
        self.cmg = cmg
        self.model = self.backend.model
        self.active = np.array([cmg['coordinate_ids'].index(j)
                                for j in cmg['independent_ids']], dtype=int)
        self.passive = np.setdiff1d(np.arange(self.model.nv), self.active)
        self.attachments = []
        self.constraints = []
        for cut in cmg['closures']:
            ends = []
            for side in (1, 2):
                body = cut[f'body{side}']
                placement = self.backend._placements[body].copy()
                placement[:3, 3] += placement[:3, :3] @ np.asarray(cut[f'point{side}_m'])
                ends.append((self.backend._supports[body], _se3(placement)))
            self.attachments.append(ends)
            c = pin.RigidConstraintModel(
                pin.ContactType.CONTACT_3D, self.model,
                ends[0][0], ends[0][1], ends[1][0], ends[1][1],
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            c.name = cut['id']
            if np.any(c.corrector.Kp) or np.any(c.corrector.Kd):
                raise RuntimeError("Native constraints must default to zero stabilization")
            self.constraints.append(c)
        self.constraint_data = [c.createData() for c in self.constraints]
        self.data = self.model.createData()
        self.geometry_data = self.model.createData()
        pin.initConstraintDynamics(self.model, self.data, self.constraints)

    def geometry(self, q, velocity=None):
        """Return point difference, exact point Jacobian and Jdot(q,v) v.

        Jdot*v uses native classical frame accelerations with qdd=0, not a
        numerical derivative and not PACDM path or orientation charts.
        """
        b, m, d = self.backend, self.model, self.geometry_data
        velocity = np.zeros(m.nv) if velocity is None else np.asarray(velocity)
        qn = b._native_q(q)
        vn = b._native_v(velocity, 'velocity')
        pin.computeJointJacobians(m, d, qn)
        pin.forwardKinematics(m, d, qn, vn, np.zeros(m.nv))
        pin.updateFramePlacements(m, d)
        differences, jacobians, bias_accelerations = [], [], []
        for ends in self.attachments:
            positions, js, accelerations = [], [], []
            for joint_id, placement in ends:
                positions.append((d.oMi[joint_id] * placement).translation.copy())
                js.append(np.asarray(pin.getFrameJacobian(
                    m, d, joint_id, placement,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:3, b._v_indices].copy())
                accelerations.append(np.asarray(pin.getFrameClassicalAcceleration(
                    m, d, joint_id, placement,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear).copy())
            differences.append(positions[0] - positions[1])
            jacobians.append(js[0] - js[1])
            bias_accelerations.append(accelerations[0] - accelerations[1])
        platform_j = np.asarray(pin.getFrameJacobian(
            m, d, b.body_frame_ids['platform'],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:, b._v_indices].copy()
        return dict(point_difference=np.concatenate(differences),
                    jacobian=np.vstack(jacobians),
                    acceleration_bias=np.concatenate(bias_accelerations),
                    platform_jacobian=platform_j,
                    poses={name: np.asarray(d.oMf[frame].homogeneous).copy()
                           for name, frame in b.body_frame_ids.items()})

    def acceleration(self, q, velocity, force, wrench=None):
        b = self.backend
        tau = np.zeros(self.model.nv)
        tau[self.active] = np.asarray(force)
        if wrench is not None and np.any(wrench):
            tau += self.geometry(q, velocity)['platform_jacobian'].T @ np.asarray(wrench)
        # mu=0: exact saddle-point solve, without compliant contacts. No
        # stabilization or previous contact-force guess enters the result.
        settings = pin.ProximalSettings(1e-13, 1e-13, 0., 50)
        result = pin.constraintDynamics(
            self.model, self.data, b._native_q(q),
            b._native_v(velocity, 'velocity'), b._native_v(tau, 'effort'),
            self.constraints, self.constraint_data, settings)
        result = np.asarray(result)[b._v_indices].copy()
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('Native Pinocchio constraintDynamics returned nonfinite acceleration')
        return result


def pacdm_differentials(graph, solver, q_augmented, active_velocity):
    """The same unchanged PACDM map/curvature used in the rollout."""
    N, info = solver.mapping(q_augmented)
    if not info['success']:
        raise RuntimeError(f'PACDM mapping failed: {info}')
    velocity = N @ np.asarray(active_velocity)
    h = 1e-5 / max(1., np.linalg.norm(velocity))
    J = graph.residual(q_augmented)[1]
    derivative = (graph.residual(q_augmented+h*velocity)[1]
                  - graph.residual(q_augmented-h*velocity)[1]) / (2*h)
    rows = np.asarray(info['rows'])
    curvature = np.zeros(graph.n)
    curvature[graph.passive] = -np.linalg.solve(
        J[np.ix_(rows, graph.passive)], (derivative @ velocity)[rows])
    return N[:graph.nt], velocity[:graph.nt], curvature[:graph.nt], info


def _le(value, limit, units='', description=''):
    return dict(passed=bool(np.isfinite(value) and value <= limit),
                value=float(value), limit=float(limit), relation='<=',
                units=units, description=description)


def _ge(value, limit, units='', description=''):
    return dict(passed=bool(np.isfinite(value) and value >= limit),
                value=float(value), limit=float(limit), relation='>=',
                units=units, description=description)


def validate_mechanics(cmg):
    """32 deterministic feasible poses, moving force cases, and acquisition.

    Limits are fixed, not fitted to observed results. Mixed-coordinate
    residuals use infinity norms of their stated SI components.
    """
    rng = np.random.default_rng(20260924)
    oracle = PinConstraintOracle(cmg)
    backend = oracle.backend
    nominal = inverse_seed(cmg, cmg['geometry']['nominal_pose'])
    graph = PointGraph(cmg, nominal)
    solver = PACDM(graph)
    maxima = {key: 0. for key in [
        'closure', 'fk', 'tangent', 'mapping', 'curvature', 'native_acceleration',
        'native_acceleration_relative', 'native_acceleration_closure',
        'mass_rnea', 'gravity_energy', 'kinetic_energy', 'platform_jacobian',
        'virtual_power', 'wrench_reduction', 'force_duality',
        'cmg_kinetic_energy', 'cmg_potential_energy']}
    minima = {'mass_eigenvalue': np.inf, 'reduced_mass_eigenvalue': np.inf}
    samples = []
    for index in range(32):
        pose = np.asarray(cmg['geometry']['nominal_pose']) + rng.uniform(
            [-.045, -.045, -.035, -.14, -.11, -.11],
            [.045, .045, .045, .14, .11, .11])
        if index == 0:
            pose = np.asarray(cmg['geometry']['nominal_pose'])
        q = inverse_seed(cmg, pose)
        state = graph.lift(q)
        active_velocity = rng.uniform(-.055, .055, len(graph.active))
        P, velocity, curvature, info = pacdm_differentials(
            graph, solver, state, active_velocity)
        geom = oracle.geometry(q, velocity)
        J, gamma = geom['jacobian'], geom['acceleration_bias']
        native_mapping = np.zeros_like(P)
        native_mapping[graph.active] = np.eye(len(graph.active))
        native_mapping[oracle.passive] = -np.linalg.solve(
            J[:, oracle.passive], J[:, graph.active])
        native_curvature = np.zeros(graph.nt)
        native_curvature[oracle.passive] = -np.linalg.solve(J[:, oracle.passive], gamma)
        graph_poses, graph_twists = graph.poses(state)
        position = graph_poses['platform'][:3, 3]
        spatial = graph_twists['platform'][:, :graph.nt]
        graph_platform_j = np.vstack((spatial[3:] - skew(position) @ spatial[:3], spatial[:3]))
        M = backend.mass(q)
        b = backend.bias(q, velocity)
        Mr = P.T @ M @ P
        gravity = backend.bias(q, np.zeros(graph.nt))
        force = P.T @ gravity + rng.uniform(-35., 35., len(graph.active))
        wrench = rng.uniform([-14.,-14.,-10.,-3.,-3.,-3.], [14.,14.,10.,3.,3.,3.])
        tau_external = geom['platform_jacobian'].T @ wrench
        reduced_acceleration = np.linalg.solve(Mr, force + P.T @ (tau_external - M @ curvature - b))
        acceleration = P @ reduced_acceleration + curvature
        native_acceleration = oracle.acceleration(q, velocity, force, wrench)
        basis = np.eye(graph.nt)
        differential_mass = np.column_stack([
            backend.inverse(q, np.zeros(graph.nt), basis[:, k]) - gravity
            for k in range(graph.nt)])
        potential_derivative = np.zeros(graph.nt)
        h = 1e-6
        for k in range(graph.nt):
            potential_derivative[k] = (
                backend.energy(q+h*basis[:, k], np.zeros(graph.nt))['potential_J']
                - backend.energy(q-h*basis[:, k], np.zeros(graph.nt))['potential_J']) / (2*h)
        energy = backend.energy(q, velocity)
        # Independently sum the declared CMG body inertias in world axes.
        # This catches wrong fixed-body inertia placement and COM handling,
        # which CRBA-versus-RNEA alone cannot detect because both use model I.
        cmg_kinetic = 0.
        cmg_potential = 0.
        for body in cmg['bodies']:
            body_id = body['id']
            T = geom['poses'][body_id]
            R = T[:3, :3]
            offset = R @ np.asarray(body['com_m'])
            frame_j = np.asarray(pin.getFrameJacobian(
                oracle.model, oracle.geometry_data, backend.body_frame_ids[body_id],
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:, backend._v_indices]
            twist = frame_j @ velocity
            com_velocity = twist[:3] + np.cross(twist[3:], offset)
            mass_body = body['mass_kg']
            I_world = R @ np.asarray(body['inertia_kg_m2']) @ R.T
            cmg_kinetic += .5*mass_body*(com_velocity@com_velocity) + .5*twist[3:]@I_world@twist[3:]
            cmg_potential -= mass_body*np.asarray(cmg['gravity_m_s2'])@(T[:3,3]+offset)
        physical_power = float(wrench @ (geom['platform_jacobian'] @ velocity))
        pacdm_wrench = P.T @ graph_platform_j.T @ wrench
        torque_trial = rng.normal(size=graph.nt)
        values = dict(
            closure=np.max(np.abs(geom['point_difference'])),
            fk=max(np.max(abs(graph_poses[name]-T)) for name, T in geom['poses'].items()),
            tangent=np.max(np.abs(J @ P)),
            mapping=np.max(np.abs(P-native_mapping)),
            curvature=np.max(np.abs(curvature-native_curvature)),
            native_acceleration=np.max(np.abs(acceleration-native_acceleration)),
            native_acceleration_relative=np.max(np.abs(acceleration-native_acceleration))/max(1.,np.max(abs(native_acceleration))),
            native_acceleration_closure=np.max(abs(J@native_acceleration+gamma)),
            mass_rnea=np.max(np.abs(M-differential_mass)),
            gravity_energy=np.max(np.abs(gravity-potential_derivative)),
            kinetic_energy=abs(energy['kinetic_J']-.5*velocity@M@velocity),
            cmg_kinetic_energy=abs(energy['kinetic_J']-cmg_kinetic),
            cmg_potential_energy=abs(energy['potential_J']-cmg_potential),
            platform_jacobian=np.max(abs(graph_platform_j-geom['platform_jacobian'])),
            virtual_power=abs(float(pacdm_wrench@active_velocity)-physical_power),
            wrench_reduction=np.max(abs(pacdm_wrench-native_mapping.T@tau_external)),
            force_duality=abs(float((P.T@torque_trial)@active_velocity)-float(torque_trial@(native_mapping@active_velocity))))
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError(f'Nonfinite mechanics residual in case {index}')
        for name, value in values.items():
            maxima[name] = max(maxima[name], float(value))
        minima['mass_eigenvalue'] = min(minima['mass_eigenvalue'], float(np.linalg.eigvalsh(M)[0]))
        minima['reduced_mass_eigenvalue'] = min(minima['reduced_mass_eigenvalue'], float(np.linalg.eigvalsh(Mr)[0]))
        samples.append(dict(index=index, pose=pose.tolist(), active_velocity=active_velocity.tolist(),
                            force_N=force.tolist(), wrench_world=wrench.tolist(),
                            native_acceleration_error_SI=values['native_acceleration'],
                            physical_constraint_rank=int(np.linalg.matrix_rank(J)),
                            selected_block_condition=float(1/info['rcond'])))
    acquisition = []
    for index in range(8):
        target_pose = np.asarray(cmg['geometry']['nominal_pose']) + rng.uniform(
            [-.035,-.035,-.03,-.10,-.09,-.09], [.035,.035,.035,.10,.09,.09])
        truth = inverse_seed(cmg, target_pose)
        seed = truth.copy()
        # Perturb passive physical coordinates; target actuator lengths are
        # held exactly by the PACDM fixed-row defect homotopy.
        seed[:6] += rng.uniform(-.005,.005,6)
        for i in range(6):
            seed[6+3*i:8+3*i] += rng.uniform(-.008,.008,2)
        augmented_seed = graph.lift(seed)
        initial_gap = float(np.max(abs(oracle.geometry(seed)['point_difference'])))
        solved, info = solver.acquire(truth[graph.active], augmented_seed)
        final_gap = float(np.max(abs(oracle.geometry(solved[:graph.nt])['point_difference'])))
        final_error = float(np.max(abs(solved[:graph.nt]-truth)))
        acquisition.append(dict(index=index, success=bool(info['success']),
                                initial_point_gap_m=initial_gap, final_point_gap_m=final_gap,
                                physical_coordinate_error=final_error,
                                accepted_steps=info['accepted_steps'], rejected_steps=info['rejected_steps']))
    limits = dict(closure=2e-12, fk=2e-12, tangent=1e-10, mapping=1e-9,
                  curvature=2e-7, native_acceleration=2e-6,
                  native_acceleration_relative=1e-7,
                  native_acceleration_closure=2e-9, mass_rnea=2e-10,
                  gravity_energy=2e-6, kinetic_energy=2e-10,
                  platform_jacobian=2e-12, virtual_power=2e-10,
                  wrench_reduction=2e-9, force_duality=2e-10,
                  cmg_kinetic_energy=2e-10, cmg_potential_energy=2e-10)
    checks = {name: _le(maxima[name], limit) for name, limit in limits.items()}
    checks['positive_tree_mass'] = _ge(minima['mass_eigenvalue'], 1e-7, 'mixed SI')
    checks['positive_reduced_mass'] = _ge(minima['reduced_mass_eigenvalue'], 1e-5, 'kg')
    checks['physical_constraint_rank'] = _ge(min(s['physical_constraint_rank'] for s in samples),18,'rank')
    checks['perturbed_seed_success_count'] = _ge(sum(a['success'] for a in acquisition),8,'cases')
    checks['perturbed_seed_final_closure'] = _le(max(a['final_point_gap_m'] for a in acquisition),1e-8,'m')
    checks['perturbed_seed_coordinate_error'] = _le(max(a['physical_coordinate_error'] for a in acquisition),2e-7,'m or rad')
    return dict(passed=all(check['passed'] for check in checks.values()), checks=checks,
                details=dict(pinocchio_version=pin.__version__, random_seed=20260924,
                             moving_force_cases=32, perturbed_seed_cases=8,
                             native_constraint_type='CONTACT_3D', native_constraint_count=6,
                             native_constraint_rows=18, native_stabilization_gains=0.,
                             native_regularization_mu=0.,
                             independence='Native point constraints and constraintDynamics independently check PACDM reduction; both use the same CMG-compiled Pinocchio rigid-body model.',
                             samples=samples, acquisition=acquisition))


def audit_rollout(cmg, npz_path, samples=111):
    """Check logged dynamic states against native constrained dynamics.

    The acceleration must be logged at the same pre-step q, velocity, force,
    and wrench. This is a statewise independent audit, not a second rollout.
    Samples include the maximum absolute sample for each external wrench
    component, in addition to uniform temporal coverage.
    """
    required = ('q','velocity','acceleration','force','wrench','time')
    with np.load(Path(npz_path), allow_pickle=False) as saved:
        absent = [key for key in required if key not in saved]
        if absent:
            raise ValueError(f'Rollout lacks audit fields: {absent}')
        data = {key: saved[key] for key in required}
        if 'coordinate_ids' not in saved or not np.array_equal(
                saved['coordinate_ids'], np.asarray(cmg['coordinate_ids'])):
            raise ValueError('Saved coordinate_ids missing or different from the CMG')
    count = len(data['time'])
    if count == 0 or samples < 1:
        raise ValueError('Rollout and requested audit sample count must be nonempty')
    shapes = {'q': (count, len(cmg['coordinate_ids'])),
              'velocity': (count, len(cmg['coordinate_ids'])),
              'acceleration': (count, len(cmg['coordinate_ids'])),
              'force': (count, len(cmg['independent_ids'])),
              'wrench': (count, 6), 'time': (count,)}
    for key, expected in shapes.items():
        if data[key].shape != expected or not np.all(np.isfinite(data[key])):
            raise ValueError(f'Invalid shape or nonfinite saved {key}: expected {expected}')
    if count > 1 and np.any(np.diff(data['time']) <= 0.):
        raise ValueError('Saved timestamps must increase strictly')
    indices = set(np.linspace(0,count-1,min(samples,count),dtype=int).tolist())
    indices.update(np.argmax(np.abs(data['wrench']),axis=0).tolist())
    indices = sorted(indices)
    oracle = PinConstraintOracle(cmg)
    values = {name:0. for name in ('acceleration','relative_acceleration','position_closure',
                                   'velocity_closure','acceleration_closure','native_acceleration_closure')}
    records=[]
    for i in indices:
        q, v, a, force, wrench = (data[key][i] for key in required[:-1])
        geom = oracle.geometry(q,v)
        reference = oracle.acceleration(q,v,force,wrench)
        difference = float(np.max(abs(a-reference)))
        sample_values=dict(acceleration=difference,
                           relative_acceleration=difference/max(1.,float(np.max(abs(reference)))),
                           position_closure=float(np.max(abs(geom['point_difference']))),
                           velocity_closure=float(np.max(abs(geom['jacobian']@v))),
                           acceleration_closure=float(np.max(abs(geom['jacobian']@a+geom['acceleration_bias']))),
                           native_acceleration_closure=float(np.max(abs(geom['jacobian']@reference+geom['acceleration_bias']))))
        if not all(np.isfinite(value) for value in sample_values.values()):
            raise FloatingPointError(f'Nonfinite native audit residual at sample {i}')
        for key,value in sample_values.items():
            values[key]=max(values[key],value)
        records.append(dict(index=i,time_s=float(data['time'][i]),acceleration_error_SI=difference,
                            external_wrench_nonzero=bool(np.any(wrench))))
    limits=dict(acceleration=1e-5,relative_acceleration=2e-6,position_closure=1e-8,
                velocity_closure=2e-8,acceleration_closure=2e-6,native_acceleration_closure=2e-6)
    checks={name:_le(value,limits[name]) for name,value in values.items()}
    return dict(passed=all(c['passed'] for c in checks.values()),checks=checks,
                details=dict(samples=len(indices),source=Path(npz_path).name,
                             dynamics='Pinocchio constraintDynamics, CONTACT_3D, Kp=Kd=0, mu=0',
                             sampled_states=records))


def validate_audit_negative_controls(cmg, npz_path):
    """Prove the audit rejects altered acceleration and malformed state logs.

    These tiny fixtures derive three existing state samples and are deleted
    after the check. The source run and its files remain untouched.
    """
    from tempfile import TemporaryDirectory
    required = ('q','velocity','acceleration','force','wrench','time')
    with np.load(Path(npz_path), allow_pickle=False) as source:
        ids = np.unique(np.linspace(0,len(source['time'])-1,3,dtype=int))
        if len(ids) < 3:
            raise ValueError('Negative controls require at least three logged states')
        fixture = {key:source[key][ids].copy() for key in required}
        fixture['coordinate_ids'] = source['coordinate_ids'].copy()
    outcomes = {}
    with TemporaryDirectory(prefix='pacdm_audit_controls_') as work:
        path = Path(work)/'control.npz'
        np.savez_compressed(path,**fixture)
        baseline = audit_rollout(cmg,path,samples=3)
        outcomes['unaltered_fixture'] = dict(passed=bool(baseline['passed']),
            expected='accept', observed='accept' if baseline['passed'] else 'reject')
        altered = {k:v.copy() for k,v in fixture.items()}
        altered['acceleration'][:,0] += .01
        np.savez_compressed(path,**altered)
        check = audit_rollout(cmg,path,samples=3)
        outcomes['acceleration_plus_0_01_m_s2'] = dict(
            passed=bool(not check['passed'] and not check['checks']['acceleration']['passed']),
            expected='reject', observed='accept' if check['passed'] else 'reject',
            measured_acceleration_discrepancy=check['checks']['acceleration']['value'],
            tolerance=check['checks']['acceleration']['limit'])
        for name in ('nonfinite_acceleration','reordered_coordinates','nonincreasing_time','wrong_velocity_shape'):
            altered = {k:v.copy() for k,v in fixture.items()}
            if name == 'nonfinite_acceleration':
                altered['acceleration'][1,0] = np.nan
            elif name == 'reordered_coordinates':
                altered['coordinate_ids'] = altered['coordinate_ids'][::-1]
            elif name == 'nonincreasing_time':
                altered['time'][1] = altered['time'][0]
            elif name == 'wrong_velocity_shape':
                altered['velocity'] = altered['velocity'][:,:-1]
            np.savez_compressed(path,**altered)
            try:
                audit_rollout(cmg,path,samples=3)
            except ValueError as error:
                outcomes[name] = dict(passed=True,expected='reject',observed='reject',reason=str(error))
            else:
                outcomes[name] = dict(passed=False,expected='reject',observed='accept')
    return dict(passed=all(item['passed'] for item in outcomes.values()), checks=outcomes,
                source=Path(npz_path).name,
                entrypoint='stewart.pin_checks.validate_audit_negative_controls(cmg,npz_path)',
                fixture_samples=len(ids), source_sample_indices=ids.tolist())
