"""Instantaneous constrained inverse dynamics for the accepted excavator source.

Six preferred independent accelerations are requested. The seventh independent
coordinate q22 is an unactuated pin: its acceleration is solved from dynamics.
The source's unconnected p0 effort is a mandatory explicit experiment input.
This module does not infer hydraulics, calibrated channel units, or motor limits.
"""
from __future__ import annotations

import numpy as np

from constrained_dynamics import finite_vector

INDEPENDENT_IDS = ['q23', 'q7', 'q4', 'q0', 'q1', 'q21', 'q22']


def _finite_matrix(value, shape, label):
    if np.iscomplexobj(value):
        raise ValueError(f'{label} must be real')
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f'{label} must have finite shape {shape}')
    return a.copy()


def _equilibrated_solve(matrix, rhs, row_scale, rank_tolerance=1e-10):
    """Square exact solve after mass-informed row and column equilibration.

    Column scales are numerical preconditioners, not actuator effort weights or
    a physical condition number. No least-squares motion approximation occurs.
    """
    a = row_scale[:, None] * matrix
    column_norm = np.linalg.norm(a, axis=0)
    if np.any(column_norm <= np.finfo(float).tiny):
        raise ValueError('Inverse-dynamics augmented system has a zero column')
    column_scale = 1. / column_norm
    normalized = a * column_scale[None, :]
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.count_nonzero(singular > rank_tolerance * singular[0]))
    if rank != matrix.shape[1]:
        raise ValueError('Six channels and free pin acceleration do not define a unique inverse solution')
    x = column_scale * np.linalg.solve(normalized, row_scale * rhs)
    weighted_rhs = row_scale * rhs
    residual = np.linalg.norm(row_scale * (matrix @ x - rhs), ord=np.inf)
    denominator = max(1., np.linalg.norm(weighted_rhs, ord=np.inf),
                      np.linalg.norm(normalized, ord=np.inf) *
                      np.linalg.norm(x / column_scale, ord=np.inf))
    if not np.all(np.isfinite(x)) or residual / denominator > 1e-10:
        raise ValueError('Inverse-dynamics augmented solve fails its scaled equilibrium equation')
    return x, {'augmented_rank': rank,
               'augmented_scaled_singular_values': singular,
               'augmented_scaled_condition': float(singular[0] / singular[-1]),
               'row_scale': row_scale, 'column_scale': column_scale,
               'augmented_normalized_residual': float(residual / denominator)}


class SourceInverseDynamics:
    """Exact mechanical inverse using the source's six known numeric channels.

    ``solve(component, acceleration6, p0_force=...)`` returns the required
    channel numbers, eight physical port efforts, and the dynamically required
    q22 acceleration. ``solve_full_request`` additionally checks feasibility of
    a supplied seventh acceleration and rejects incompatible requests.
    """

    def __init__(self, mapper, independent_ids=INDEPENDENT_IDS):
        if list(independent_ids) != INDEPENDENT_IDS:
            raise ValueError('Independent order must preserve six preferred coordinates followed by q22')
        self.mapper = mapper
        self.independent_ids = list(independent_ids)
        self.tree_ids = list(mapper.tree_ids)
        self.active = [self.tree_ids.index(j) for j in self.independent_ids]
        self.p0_index = mapper.port_ids.index('p0')
        if 'q22' in mapper.port_ids:
            raise ValueError('The source internal pin must not receive an invented actuator')

    def _components(self, component):
        nt = len(self.tree_ids)
        m = _finite_matrix(component['mass_matrix'], (nt, nt), 'Tree mass')
        h = finite_vector(component['bias_forces'], nt, 'Tree bias')
        n = _finite_matrix(component['tangent_map'], (nt, 7), 'Tangent map')
        b = finite_vector(component['curvature'], nt, 'Acceleration curvature')
        v = finite_vector(component['velocity'], nt, 'Tree velocity')
        raw_j = np.asarray(component['jacobian'])
        if raw_j.ndim != 2:
            raise ValueError('Physical closure Jacobian must be a matrix')
        j = _finite_matrix(raw_j, (raw_j.shape[0], nt), 'Physical closure Jacobian')
        r = finite_vector(component['residual'], j.shape[0], 'Physical closure residual')
        gamma = finite_vector(component['jdot_velocity'], j.shape[0], 'Closure curvature')
        if np.max(np.abs(r)) > 1e-8 or np.max(np.abs(j @ v)) > 1e-8:
            raise ValueError('Inverse dynamics requires physically closed position and tangent velocity')
        rank = int(np.linalg.matrix_rank(j, tol=1e-9))
        passive = [i for i in range(nt) if i not in self.active]
        if rank != nt-7 or np.linalg.matrix_rank(j[:, passive], tol=1e-9) != nt-7:
            raise ValueError('Physical closure or independent partition has an unsupported rank')
        if int(component['constraint_rank']) != rank:
            raise ValueError('Supplied constraint rank disagrees with the physical Jacobian')
        if (np.max(abs(n[self.active]-np.eye(7))) > 1e-9 or
                np.max(abs(b[self.active])) > 1e-9 or
                np.max(abs(j @ n)) > 1e-8 or
                np.max(abs(j @ b + gamma)) > 1e-8 or
                np.max(abs(v-n @ v[self.active])) > 1e-8):
            raise ValueError('Tangent and curvature data fail physical coordinate compatibility')
        md = np.diag(m)
        if np.any(md <= 0):
            raise ValueError('Tree mass must have positive diagonal inertia')
        ms = 1. / np.sqrt(md)
        normalized_mass = ms[:, None] * m * ms[None, :]
        if np.max(abs(normalized_mass-normalized_mass.T)) > 1e-10:
            raise ValueError('Tree mass is not symmetric')
        try:
            np.linalg.cholesky((normalized_mass+normalized_mass.T)/2.)
        except np.linalg.LinAlgError as error:
            raise ValueError('Tree mass is not positive definite') from error
        mr, hr = n.T @ m @ n, n.T @ (h+m @ b)
        if np.any(np.diag(mr) <= 0):
            raise ValueError('Reduced mass must retain the pin inertia')
        row_scale = 1. / np.sqrt(np.diag(mr))
        try:
            np.linalg.cholesky(row_scale[:, None]*((mr+mr.T)/2.)*row_scale[None, :])
        except np.linalg.LinAlgError as error:
            raise ValueError('Reduced inertia is not positive definite') from error
        e = _finite_matrix(self.mapper.E, (nt, 8), 'Port effort map')
        d = _finite_matrix(self.mapper.D, (8, 6), 'Six-channel map')
        if np.any(d[self.p0_index] != 0.):
            raise ValueError('Unknown p0 must not be silently driven by a known source channel')
        channel_map, p0_map = n.T @ e @ d, n.T @ e[:, self.p0_index]
        weighted = row_scale[:, None] * channel_map
        norms = np.linalg.norm(weighted, axis=0)
        normalized = weighted / np.where(norms > np.finfo(float).tiny, norms, 1.)[None, :]
        singular = np.linalg.svd(normalized, compute_uv=False)
        channel_rank = int(np.count_nonzero(singular > 1e-10*max(singular[0], 1.)))
        if channel_rank != 6:
            raise ValueError('Six known source channels do not have rank six in reduced mechanics')
        return dict(m=m, h=h, n=n, b=b, v=v, j=j, gamma=gamma, mr=mr, hr=hr,
                    row_scale=row_scale, E=e, D=d, channel_map=channel_map,
                    p0_map=p0_map, channel_rank=channel_rank,
                    channel_scaled_singular_values=singular, closure_rank=rank)

    def solve(self, component, requested_acceleration6, *, p0_force):
        desired = finite_vector(requested_acceleration6, 6, 'Six requested independent accelerations')
        p0 = finite_vector([p0_force], 1, 'Explicit p0 experiment force')[0]
        c = self._components(component)
        m, h, n, b, v = (c[k] for k in ['m', 'h', 'n', 'b', 'v'])
        mr, hr, B, B0 = (c[k] for k in ['mr', 'hr', 'channel_map', 'p0_map'])
        augmented = np.column_stack((B, -mr[:, 6]))
        rhs = mr[:, :6] @ desired + hr - B0*p0
        solution, diagnostic = _equilibrated_solve(augmented, rhs, c['row_scale'])
        channels, pin_acceleration = solution[:6], solution[6]
        effort = self.mapper.source_efforts(channels, unresolved_efforts={'p0': p0})
        acceleration = np.r_[desired, pin_acceleration]
        tree_acceleration = n @ acceleration + b
        tree_effort = c['E'] @ effort
        reaction = m @ tree_acceleration + h - tree_effort
        closure_acceleration = c['j'] @ tree_acceleration + c['gamma']
        projected_reaction = n.T @ reaction
        reduced_equilibrium = mr @ acceleration + hr - B @ channels - B0*p0
        scale = c['row_scale']
        equilibrium_size = max(1., np.linalg.norm(scale*(mr @ acceleration), ord=np.inf),
                               np.linalg.norm(scale*hr, ord=np.inf),
                               np.linalg.norm(scale*(n.T @ tree_effort), ord=np.inf))
        equilibrium_error = float(np.linalg.norm(scale*reduced_equilibrium, ord=np.inf)/equilibrium_size)
        reaction_error = float(np.linalg.norm(scale*projected_reaction, ord=np.inf)/equilibrium_size)
        closure_size = max(1., np.linalg.norm(c['j'], ord=np.inf)*np.linalg.norm(tree_acceleration, ord=np.inf),
                           np.linalg.norm(c['gamma'], ord=np.inf))
        if equilibrium_error > 1e-9 or reaction_error > 1e-9 or np.max(abs(closure_acceleration))/closure_size > 1e-9:
            raise ValueError('Inverse result fails full physical equilibrium or acceleration closure')
        port_velocity = c['E'].T @ v
        channel_rate = c['D'].T @ port_velocity
        physical_power = float(effort @ port_velocity)
        tree_power = float(tree_effort @ v)
        channel_power = float(channels @ channel_rate + p0*port_velocity[self.p0_index])
        reaction_power = float(reaction @ v)
        return {
            'mode': 'source_channels_with_explicit_p0',
            'requested_independent_acceleration': desired,
            'independent_acceleration': acceleration,
            'pin_acceleration': float(pin_acceleration),
            'acceleration': tree_acceleration,
            'channels': channels,
            'physical_efforts': effort,
            'p0_force': float(p0),
            'generalized_reaction': reaction,
            'tree_effort': tree_effort,
            'reduced_mass': mr,
            'reduced_bias': hr,
            'channel_to_reduced_effort_map': B,
            'p0_to_reduced_effort_map': B0,
            'port_velocity': port_velocity,
            'channel_conjugate_rate': channel_rate,
            'closure_acceleration': closure_acceleration,
            'reduced_equilibrium_residual': reduced_equilibrium,
            'tangent_reaction': projected_reaction,
            'power': {'physical_port_W': physical_power, 'tree_W': tree_power,
                      'source_channel_plus_p0_W': channel_power,
                      'generalized_reaction_W': reaction_power},
            'diagnostics': {**diagnostic, 'channel_rank': c['channel_rank'],
                            'channel_scaled_singular_values': c['channel_scaled_singular_values'],
                            'physical_constraint_rank': c['closure_rank'],
                            'reduced_equilibrium_normalized_residual': equilibrium_error,
                            'tangent_reaction_normalized_residual': reaction_error,
                            'closure_acceleration_max_abs': float(np.max(abs(closure_acceleration))),
                            'physical_tree_power_error_W': abs(physical_power-tree_power),
                            'source_channel_power_error_W': abs(physical_power-channel_power)},
            'source_p0_effort_resolved': False,
            'hydraulics_calibrated': False,
            'actuator_limits_applied': False,
            'semantics': 'Instantaneous mechanical effort solution; q22 acceleration computed dynamically; channel numbers use source gains only.',
        }

    def solve_full_request(self, component, requested_acceleration7, *, p0_force,
                           acceleration_absolute_tolerance=1e-8,
                           acceleration_relative_tolerance=1e-8):
        """Accept seven requested accelerations only if passive dynamics permit.

        Comparing q22 acceleration separately avoids hiding an incompatible
        request behind its small inertia in a global force-residual norm.
        """
        desired = finite_vector(requested_acceleration7, 7, 'Seven requested accelerations')
        tolerances = finite_vector([acceleration_absolute_tolerance, acceleration_relative_tolerance],
                                  2, 'Feasibility tolerances')
        if np.any(tolerances < 0) or not np.any(tolerances > 0):
            raise ValueError('Feasibility tolerances must be nonnegative and not both zero')
        result = self.solve(component, desired[:6], p0_force=p0_force)
        error = abs(result['pin_acceleration']-desired[6])
        threshold = tolerances[0]+tolerances[1]*max(abs(desired[6]), abs(result['pin_acceleration']))
        if error > threshold:
            raise ValueError('Requested q22 acceleration is dynamically infeasible for six source channels and explicit p0')
        result['full_request_feasible'] = True
        result['full_requested_independent_acceleration'] = desired
        result['pin_request_acceleration_error'] = float(error)
        result['pin_request_acceleration_tolerance'] = float(threshold)
        return result
