"""Guarded ideal constrained motion over a larger numerical test neighborhood.

Pinocchio supplies rigid-body and kinematic quantities. RoboCompiler assembles
the physical loops, solves ideal acceleration constraints, and integrates seven
independent coordinates with SciPy DOP853. This is not Pinocchio time stepping,
hardware validation, hydraulic simulation, or an automatic workspace proof.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pinocchio as pin
from scipy.integrate import solve_ivp

from constrained_dynamics import finite_vector, reduce_kinematics, solve_kkt
from motion_experiments import effort_at


POSITION_TOLERANCE_M = 2e-12
RANK_TOLERANCE = 1e-9
CONTINUATION_INCREMENT_RAD = .02
RETAINED_SINGULAR_VALUE_FLOOR = 1e-6
RETAINED_RATIO_FLOOR = 1e-3
DISCARDED_RATIO_CEILING = 1e-9
GUARD_KEYS = ('independent_rad', 'passive_revolute_rad', 'passive_prismatic_m')


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or np.iscomplexobj(value)):
        raise ValueError(f'{name} must be a positive finite scalar')
    result = float(value)
    if not np.isfinite(result) or result <= 0.:
        raise ValueError(f'{name} must be a positive finite scalar')
    return result


def _skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class ExtendedReference:
    """Continue the accepted nominal branch, retaining its internal pin freedom.

    guards contains independent_rad, passive_revolute_rad, passive_prismatic_m.
    Every guard is an absolute displacement from the ORIGINAL nominal model,
    including for cases whose initial configuration is offset. These are
    numerical guards only. All passive solves use all 54 physical closure rows.
    This object shares mutable native model data with engine; do not run it
    concurrently with another calculation using the same engine.
    """

    def __init__(self, engine, mapper, guards):
        self.engine, self.mapper = engine, mapper
        self.tree_ids = list(engine.tree_ids)
        self.independent_ids = list(engine.independent_ids)
        self.active = np.asarray(engine.active, dtype=int)
        self.passive = np.asarray(engine.passive, dtype=int)
        if (len(self.tree_ids) != 23 or len(self.active) != 7
                or len(self.passive) != 16 or list(mapper.tree_ids) != self.tree_ids
                or np.asarray(mapper.E).shape != (23, 8)
                or not np.all(np.isfinite(mapper.E))):
            raise ValueError('Reference requires the accepted 23-tree/7-independent/eight-port model')
        if set(guards) != set(GUARD_KEYS):
            raise ValueError(f'Numerical guards must specify exactly {GUARD_KEYS}')
        self.guards = {key: _positive(guards[key], key) for key in GUARD_KEYS}
        self.nominal = np.asarray([engine.native.pin_backend.reference[jid]
                                   for jid in self.tree_ids], dtype=float)
        self.nominal_independent = self.nominal[self.active].copy()
        joints = {record['id']: record for record in engine.native.pin_backend.cmg['joints']}
        self._revolute = np.asarray([joints[jid]['type'] == 'revolute' for jid in self.tree_ids])
        self._passive_revolute = self._revolute[self.passive]
        self._passive_guards = np.where(self._passive_revolute,
                                       self.guards['passive_revolute_rad'],
                                       self.guards['passive_prismatic_m'])
        self.last_reconstruction = None
        self._endpoints = []
        backend = self.engine.native.pin_backend
        for jid in backend.cut_ids:
            joint = joints[jid]
            axis = np.asarray(joint['axis'], dtype=float)
            for lever in (0., backend.axis_lever_m):
                pair = []
                for body_key, transform_key in (('base_body', 'T_BJ'), ('follower_body', 'T_FJ')):
                    transform = np.asarray(joint[transform_key], dtype=float)
                    local = transform[:3, 3] + transform[:3, :3] @ (lever*axis)
                    pair.append((joint[body_key], local))
                self._endpoints.append(pair)
        self.nominal = self.reconstruct(self.nominal_independent)
        self.engine.native._audit_options()

    def _check_guards(self, q):
        if np.max(np.abs(q[self.active]-self.nominal_independent)) > self.guards['independent_rad']:
            raise ValueError('Independent coordinate exceeded its numerical guard from original nominal')
        if np.any(np.abs(q[self.passive]-self.nominal[self.passive]) > self._passive_guards):
            raise ValueError('Passive coordinate exceeded its numerical guard from original nominal')

    def reconstruct(self, independent_position, q_seed=None):
        """Reconstruct by bounded continuation; only passive coordinates change.

        Without a seed, the path starts at original nominal, making offset-case
        initialization reproducible. A cached, already closed seed can accelerate
        RHS evaluation. Each continuation increment is <=.02 rad in infinity
        norm even when an adaptive solver evaluates times out of order.
        """
        u = finite_vector(independent_position, 7, 'Independent position')
        if np.max(np.abs(u-self.nominal_independent)) > self.guards['independent_rad']:
            raise ValueError('Independent position is outside the original-nominal numerical guard')
        q = self.nominal.copy() if q_seed is None else finite_vector(q_seed, 23, 'Closed reconstruction seed')
        self._check_guards(q)
        backend = self.engine.native.pin_backend
        evaluations = iterations = 0
        maximum_residual = maximum_increment = 0.
        max_correction_rad = max_correction_m = 0.
        minimum_full = minimum_passive = float('inf')
        minimum_full_ratio = minimum_passive_ratio = float('inf')
        maximum_discarded_ratio = 0.

        def evaluate(value):
            nonlocal evaluations
            backend.set_configuration(value)
            residual, jacobian = backend.closure()
            evaluations += 1
            if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(jacobian))):
                raise ValueError('Nonfinite physical position closure')
            return residual, jacobian

        residual, jacobian = evaluate(q)
        if np.max(np.abs(residual)) > 1e-10:
            raise ValueError('Reconstruction seed must already satisfy physical closure')
        start = q[self.active].copy()
        count = max(1, int(np.ceil(np.max(np.abs(u-start))/CONTINUATION_INCREMENT_RAD)))
        for index in range(1, count+1):
            target = u.copy() if index == count else start+(u-start)*(index/count)
            maximum_increment = max(maximum_increment, float(np.max(np.abs(target-q[self.active]))))
            q[self.active] = target
            residual, jacobian = evaluate(q)
            for iteration in range(31):
                passive_jacobian = jacobian[:, self.passive]
                passive_singular = np.linalg.svd(passive_jacobian, compute_uv=False)
                if int(np.count_nonzero(passive_singular > RANK_TOLERANCE)) != 16:
                    raise ValueError('Passive reconstruction lost rank 16')
                error = float(np.max(np.abs(residual)))
                if error <= POSITION_TOLERANCE_M:
                    full_singular = np.linalg.svd(jacobian, compute_uv=False)
                    if int(np.count_nonzero(full_singular > RANK_TOLERANCE)) != 16:
                        raise ValueError('Assembled physical closure does not retain rank 16')
                    # Explicit nondimensionalization: every closure row /1 m;
                    # revolute coordinate columns *1 rad; prismatic columns
                    # *1 m. These unit-valued SI scales leave the numeric J
                    # unchanged and are evidence conventions, not unit-free
                    # assertions about mechanical conditioning.
                    full_ratio = float(full_singular[15]/full_singular[0])
                    passive_ratio = float(passive_singular[-1]/passive_singular[0])
                    discarded_ratio = float(full_singular[16]/full_singular[0])
                    if (min(full_singular[15], passive_singular[-1]) < RETAINED_SINGULAR_VALUE_FLOOR
                            or min(full_ratio, passive_ratio) < RETAINED_RATIO_FLOOR
                            or discarded_ratio > DISCARDED_RATIO_CEILING):
                        raise ValueError('Assembled ideal branch does not satisfy declared singular-value separation')
                    minimum_full = min(minimum_full, float(full_singular[15]))
                    minimum_passive = min(minimum_passive, float(passive_singular[-1]))
                    minimum_full_ratio = min(minimum_full_ratio, full_ratio)
                    minimum_passive_ratio = min(minimum_passive_ratio, passive_ratio)
                    maximum_discarded_ratio = max(maximum_discarded_ratio, discarded_ratio)
                    maximum_residual = max(maximum_residual, error)
                    break
                if iteration == 30:
                    raise ValueError('Passive Newton reconstruction did not converge')
                correction = np.linalg.lstsq(passive_jacobian, -residual, rcond=1e-12)[0]
                if not np.all(np.isfinite(correction)):
                    raise ValueError('Passive Newton correction is nonfinite')
                max_correction_rad = max(max_correction_rad, float(np.max(np.abs(correction[self._passive_revolute]), initial=0.)))
                max_correction_m = max(max_correction_m, float(np.max(np.abs(correction[~self._passive_revolute]), initial=0.)))
                previous_norm = float(np.linalg.norm(residual))
                for reduction in range(15):
                    trial = q.copy()
                    trial[self.passive] += 2.**(-reduction)*correction
                    if np.any(np.abs(trial[self.passive]-self.nominal[self.passive]) > self._passive_guards):
                        continue
                    trial_residual, trial_jacobian = evaluate(trial)
                    if (np.max(np.abs(trial_residual)) <= POSITION_TOLERANCE_M
                            or np.linalg.norm(trial_residual) < previous_norm):
                        q, residual, jacobian = trial, trial_residual, trial_jacobian
                        iterations += 1
                        break
                else:
                    raise ValueError('Passive Newton correction cannot reduce physical closure inside guards')
        if not np.array_equal(q[self.active], u):
            raise ValueError('Passive reconstruction modified an independent coordinate')
        self._check_guards(q)
        self.last_reconstruction = {
            'position_closure_m': float(np.max(np.abs(residual))),
            'maximum_continuation_closure_m': maximum_residual,
            'constraint_rank': 16, 'passive_rank': 16,
            'continuation_steps': count, 'maximum_independent_increment_rad': maximum_increment,
            'newton_iterations': iterations, 'position_evaluations': evaluations,
            'smallest_retained_constraint_singular_value': minimum_full,
            'smallest_passive_singular_value': minimum_passive,
            'minimum_retained_constraint_singular_ratio': minimum_full_ratio,
            'minimum_passive_singular_ratio': minimum_passive_ratio,
            'maximum_discarded_constraint_singular_ratio': maximum_discarded_ratio,
            'singular_value_scales': {'closure_row_m': 1., 'revolute_column_rad': 1., 'prismatic_column_m': 1.},
            'max_passive_newton_correction_rad': max_correction_rad,
            'max_passive_newton_correction_m': max_correction_m,
            'max_independent_displacement_rad': float(np.max(np.abs(u-self.nominal_independent))),
            'max_passive_revolute_displacement_rad': float(np.max(np.abs(q[self.passive[self._passive_revolute]]-self.nominal[self.passive[self._passive_revolute]]), initial=0.)),
            'max_passive_prismatic_displacement_m': float(np.max(np.abs(q[self.passive[~self._passive_revolute]]-self.nominal[self.passive[~self._passive_revolute]]), initial=0.)),
            'seed_semantics': 'original nominal' if q_seed is None else 'previously closed cached seed',
            'guard_reference': 'original CMG nominal, never shifted to case initial state'}
        return q.copy()

    def _pin_closure(self, q, v):
        """Pinocchio-only physical point closure and exact Jacobian derivative."""
        backend = self.engine.native.pin_backend
        backend.set_configuration(q)
        pin.computeJointJacobiansTimeVariation(backend.model, backend.data, q, v)
        pin.updateFramePlacements(backend.model, backend.data)
        poses, jacobians = backend.body_poses(), backend.body_jacobians()
        derivatives = {body: np.asarray(pin.getFrameJacobianTimeVariation(
            backend.model, backend.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).copy()
            for body, fid in backend.body_frame_ids.items()}
        residual, jacobian, jacobian_dot = [], [], []
        for endpoints in self._endpoints:
            values = []
            for body, local in endpoints:
                transform, j, dj = poses[body], jacobians[body], derivatives[body]
                displacement = transform[:3, :3] @ local
                displacement_dot = np.cross(j[3:] @ v, displacement)
                values.append((transform[:3, 3]+displacement,
                               j[:3]-_skew(displacement) @ j[3:],
                               dj[:3]-_skew(displacement_dot) @ j[3:]-_skew(displacement) @ dj[3:]))
            residual.append(values[0][0]-values[1][0])
            jacobian.append(values[0][1]-values[1][1])
            jacobian_dot.append(values[0][2]-values[1][2])
        return np.concatenate(residual), np.vstack(jacobian), np.vstack(jacobian_dot)

    def component(self, q, independent_velocity, gravity):
        """Native Pinocchio components and all-row ideal differential map."""
        q = finite_vector(q, 23, 'Tree configuration')
        udot = finite_vector(independent_velocity, 7, 'Independent velocity')
        gravity = finite_vector(gravity, 3, 'Gravity')
        self._check_guards(q)
        backend = self.engine.native.pin_backend
        backend.set_configuration(q)
        residual, jacobian = backend.closure()
        tangent = np.zeros((23, 7))
        tangent[self.active] = np.eye(7)
        tangent[self.passive] = np.linalg.lstsq(jacobian[:, self.passive], -jacobian[:, self.active], rcond=1e-12)[0]
        velocity = tangent @ udot
        residual, jacobian, jacobian_dot = self._pin_closure(q, velocity)
        kin = reduce_kinematics(residual, jacobian, jacobian_dot, velocity, self.active,
                                closure_tolerance=1e-10, velocity_tolerance=1e-9)
        native = self.engine.native
        model, data = native.pin_model, native.pin_data
        model.gravity = pin.Motion.Zero()
        model.gravity.linear = gravity
        cq, cv = q[native._compact_source_indices], velocity[native._compact_source_indices]
        upper = np.triu(np.asarray(pin.crba(model, data, cq)))
        compact_mass = upper+np.triu(upper, 1).T
        mass = compact_mass[np.ix_(native._source_compact_indices, native._source_compact_indices)]
        bias = np.asarray(pin.nonLinearEffects(model, data, cq, cv))[native._source_compact_indices].copy()
        kinetic = float(pin.computeKineticEnergy(model, data, cq, cv))
        potential = float(pin.computePotentialEnergy(model, data, cq))
        ground = model.inertias[0]
        ground_offset = -float(ground.mass)*float(np.dot(ground.lever, gravity))
        component = {**kin, 'mass_matrix': mass, 'bias_forces': bias,
                     'kinetic_energy': kinetic, 'potential_energy': potential,
                     'potential_energy_ground_offset': ground_offset}
        if any(not np.all(np.isfinite(value)) for value in component.values()):
            raise ValueError('Native Pinocchio component evaluation produced nonfinite values')
        return component

    def initial_state(self, case):
        """Start every case by deterministic continuation from original nominal."""
        offset = finite_vector(case['initial_independent_offset_rad'], 7, 'Initial independent offset')
        velocity = finite_vector(case['initial_independent_velocity_rad_s'], 7, 'Initial independent velocity')
        gravity = finite_vector(case['gravity_m_s2'], 3, 'Gravity')
        effort_at(case, 0.)  # No silent missing-p0 or invalid-load default.
        q0 = self.reconstruct(self.nominal_independent+offset)
        reconstruction = deepcopy(self.last_reconstruction)
        component = self.component(q0, velocity, gravity)
        return {'q0': q0.tolist(), 'v0': component['velocity'].tolist(),
                'u0': q0[self.active].tolist(), 'udot0': velocity.tolist(),
                'continuation': reconstruction,
                'original_nominal_qtree': self.nominal.tolist(),
                'original_nominal_independent': self.nominal_independent.tolist()}

    def integrate(self, case, sample_times, rtol=1e-10, atol=1e-12, max_step=.01):
        """Integrate seven angles, seven velocities, and mechanical applied work.

        All outputs are JSON-ready. Numerical guards, physical rank, position,
        velocity and acceleration closure are checked at EVERY RHS evaluation,
        including rejected/adaptive out-of-order stages, and at output samples.
        Independent states and energy/work are never projected or corrected.
        """
        rtol, atol, max_step = (_positive(rtol, 'rtol'), _positive(atol, 'atol'), _positive(max_step, 'max_step'))
        duration = _positive(case['duration_s'], 'duration_s')
        times = np.asarray(sample_times, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0.) or times[0] != 0.
                or abs(times[-1]-duration) > 1e-12*max(1., duration)
                or np.any(times < 0.) or np.any(times > duration)):
            raise ValueError('Observation times must strictly increase from zero through duration')
        initial = self.initial_state(case)
        self.engine.native._audit_options()
        gravity = finite_vector(case['gravity_m_s2'], 3, 'Gravity')
        seed = np.asarray(initial['q0'])
        y0 = np.r_[initial['u0'], initial['udot0'], 0.]
        maxima = {key: 0. for key in (
            'position_closure_m', 'velocity_closure_m_s', 'acceleration_closure_m_s2',
            'reaction_power_W', 'force_balance_norm', 'force_balance_scaled',
            'max_independent_displacement_rad', 'max_passive_revolute_displacement_rad',
            'max_passive_prismatic_displacement_m', 'maximum_continuation_increment_rad')}
        rank_values = set()
        effort_min, effort_max = np.full(8, np.inf), np.full(8, -np.inf)
        minimum_full = minimum_passive = float('inf')
        minimum_full_ratio = minimum_passive_ratio = float('inf')
        maximum_discarded_ratio = 0.
        evaluations = reconstruction_evaluations = continuation_steps = 0

        def evaluate(t, y):
            nonlocal seed, evaluations, minimum_full, minimum_passive, reconstruction_evaluations, continuation_steps
            nonlocal minimum_full_ratio, minimum_passive_ratio, maximum_discarded_ratio
            y = finite_vector(y, 15, 'Adaptive independent state and work')
            q = self.reconstruct(y[:7], seed)
            seed = q.copy()
            info = self.last_reconstruction
            reconstruction_evaluations += info['position_evaluations']
            continuation_steps += info['continuation_steps']
            minimum_full = min(minimum_full, info['smallest_retained_constraint_singular_value'])
            minimum_passive = min(minimum_passive, info['smallest_passive_singular_value'])
            minimum_full_ratio = min(minimum_full_ratio, info['minimum_retained_constraint_singular_ratio'])
            minimum_passive_ratio = min(minimum_passive_ratio, info['minimum_passive_singular_ratio'])
            maximum_discarded_ratio = max(maximum_discarded_ratio, info['maximum_discarded_constraint_singular_ratio'])
            for key in ('max_independent_displacement_rad', 'max_passive_revolute_displacement_rad', 'max_passive_prismatic_displacement_m'):
                maxima[key] = max(maxima[key], info[key])
            maxima['maximum_continuation_increment_rad'] = max(maxima['maximum_continuation_increment_rad'], info['maximum_independent_increment_rad'])
            component = self.component(q, y[7:14], gravity)
            efforts = finite_vector(effort_at(case, float(t)), 8, 'Eight explicitly specified effort ports')
            tree_effort = self.mapper.E @ efforts
            solved = solve_kkt(component, tree_effort)
            velocity, acceleration = component['velocity'], solved['acceleration']
            position_error = float(np.max(np.abs(component['residual'])))
            velocity_error = float(np.max(np.abs(component['jacobian'] @ velocity)))
            acceleration_error = float(np.max(np.abs(component['jacobian'] @ acceleration+component['jdot_velocity'])))
            if position_error > 1e-10 or velocity_error > 1e-9 or acceleration_error > 1e-7:
                raise ValueError('Ideal stage violates all-row physical closure')
            reaction = solved['generalized_reaction']
            balance = component['mass_matrix'] @ acceleration+component['bias_forces']-tree_effort-reaction
            norm = float(np.linalg.norm(balance))
            scale = max(1., float(np.linalg.norm(component['mass_matrix'] @ acceleration)),
                        float(np.linalg.norm(component['bias_forces'])), float(np.linalg.norm(tree_effort)), float(np.linalg.norm(reaction)))
            values = {'position_closure_m': position_error, 'velocity_closure_m_s': velocity_error,
                      'acceleration_closure_m_s2': acceleration_error, 'reaction_power_W': abs(float(reaction @ velocity)),
                      'force_balance_norm': norm, 'force_balance_scaled': norm/scale}
            for key, value in values.items():
                maxima[key] = max(maxima[key], value)
            rank_values.add(int(component['constraint_rank']))
            effort_min[:] = np.minimum(effort_min, efforts)
            effort_max[:] = np.maximum(effort_max, efforts)
            evaluations += 1
            power = float(tree_effort @ velocity)
            return q, component, efforts, solved, power

        def rhs(t, y):
            _, _, _, solved, power = evaluate(t, y)
            return np.r_[y[7:14], solved['acceleration'][self.active], power]

        result = solve_ivp(rhs, (0., duration), y0, method='DOP853', t_eval=times,
                           rtol=rtol, atol=atol, max_step=max_step, dense_output=False)
        if (not result.success or result.status != 0 or result.y.shape != (15, len(times))
                or result.t[-1] != times[-1] or not np.all(np.isfinite(result.y))):
            raise ValueError('Extended ideal reference failed to cover the complete experiment')
        rhs_evaluations = evaluations
        fields = ('qtree', 'vtree', 'efforts', 'accels', 'independent_accels', 'kinetic',
                  'potential', 'power', 'closure_position', 'closure_velocity', 'closure_acceleration', 'rank')
        output = {field: [] for field in fields}
        for index, time in enumerate(result.t):
            q, component, efforts, solved, power = evaluate(time, result.y[:, index])
            values = {'qtree': q.tolist(), 'vtree': component['velocity'].tolist(), 'efforts': efforts.tolist(),
                      'accels': solved['acceleration'].tolist(), 'independent_accels': solved['acceleration'][self.active].tolist(),
                      'kinetic': component['kinetic_energy'], 'potential': component['potential_energy']+component['potential_energy_ground_offset'],
                      'power': power, 'closure_position': float(np.max(np.abs(component['residual']))),
                      'closure_velocity': float(np.max(np.abs(component['jacobian'] @ component['velocity']))),
                      'closure_acceleration': float(np.max(np.abs(component['jacobian'] @ solved['acceleration']+component['jdot_velocity']))),
                      'rank': int(component['constraint_rank'])}
            for field in fields:
                output[field].append(values[field])
        energy = np.asarray(output['kinetic'])+np.asarray(output['potential'])
        energy_change = energy-energy[0]
        energy_error = energy_change-result.y[-1]
        if rank_values != {16}:
            raise ValueError('Ideal reference changed its physical constraint rank')
        return {
            'case': str(case['id']),
            'method': 'SciPy DOP853 integrating native Pinocchio components with RoboCompiler ideal KKT constraints',
            'scope': 'Expanded numerical neighborhood of the source CMG branch; hardware, hydraulics and global workspace are outside this scope',
            'tree_ids': self.tree_ids, 'independent_ids': self.independent_ids,
            'port_ids': list(self.mapper.port_ids), 'port_units': list(self.mapper.port_units),
            'times': result.t.tolist(), 'u': result.y[:7].T.tolist(), 'udot': result.y[7:14].T.tolist(),
            'integrated_work': result.y[-1].tolist(), 'total_energy': energy.tolist(),
            'energy_change': energy_change.tolist(), 'energy_balance_error': energy_error.tolist(),
            **output, 'initial_state': initial,
            'guards_from_original_nominal': dict(self.guards),
            'guards_are_hardware_limits': False, 'independent_states_projected': False,
            'native_backend_time_integrator_used': False,
            'rtol': rtol, 'atol': atol, 'max_step': max_step,
            'rhs_evaluations': rhs_evaluations, 'scipy_nfev': int(result.nfev),
            'sample_verification_evaluations': evaluations-rhs_evaluations,
            'position_reconstruction_evaluations': reconstruction_evaluations,
            'continuation_steps_total': continuation_steps,
            'maximum_stage_and_sample_residuals': maxima, 'constraint_ranks': sorted(rank_values),
            'smallest_retained_constraint_singular_value': minimum_full,
            'smallest_passive_singular_value': minimum_passive,
            'minimum_retained_constraint_singular_ratio': minimum_full_ratio,
            'minimum_passive_singular_ratio': minimum_passive_ratio,
            'maximum_discarded_constraint_singular_ratio': maximum_discarded_ratio,
            'singular_value_scales': {'closure_row_m': 1., 'revolute_column_rad': 1., 'prismatic_column_m': 1.},
            'effort_minimum_over_evaluations': effort_min.tolist(), 'effort_maximum_over_evaluations': effort_max.tolist(),
            'solver_success': bool(result.success), 'solver_status': int(result.status),
            'solver_message': str(result.message)}
