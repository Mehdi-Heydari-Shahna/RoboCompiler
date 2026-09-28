"""Mechanics witnesses for the excavator Pinocchio backend.

The plant (``plant.py``) is compared with references that already exist in
the preserved source, and with finite differences.  Two kinds of reference
are distinguished (``category`` in every check):

* ``witness_same_convention`` - the original BFS ``PinocchioBackend``, its
  compact ``NativeDynamics`` copy and the original continuation/component
  code.  The plant compiler composes CMG placements the same way (only the
  traversal order differs), so these comparisons verify the depth-first
  re-implementation and the inertia assembly but cannot expose a conceptual
  error that both would share;
* ``witness_independent`` - structurally independent implementations: the
  NumPy-only ``SourceBiasDynamics`` body sums (no Pinocchio), the unchanged
  PACDM SE(3) closure graph and differential map, the original controller
  inverse model built on ``FloatingSource`` (round trip), and RNEA vs CRBA;
* ``finite_difference``, ``certificate`` (residuals of the native solution)
  and ``structure``.

Negative controls inject one modelling fault at a time; a control with a
continuous metric passes when the fault moves the corresponding acceptance
metric at least 100x beyond its acceptance limit, at every sample where it is
injected (minimum over samples); the loop-removal control is a rank count.
Algebraic identities that cannot fail (virtual-work identities, the sign of the
regularized soil force, the original hydraulic fluid-power identity) are
recorded in ``details['identities']`` and are not counted as checks.
No MuJoCo module is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pinocchio as pin
import scipy

from . import paths, sentinel
from .engine import (context, FixedBaseArmController, PORTS)
from .plant import ReducedPlant, ExcavatorPinModel, scaled_body_cmg
from .soil import SoilParameters, SoilSurrogate, BucketGeometry

sentinel.install()
paths.add_original_to_path()
from constrained_dynamics import solve_kkt, solve_reduced  # noqa: E402
from excavator_pacdm import compile_pacdm  # noqa: E402
from floating_dynamics import FloatingSource  # noqa: E402
from native_dynamics import NativeDynamics  # noqa: E402
from hydraulic_actuators import HydraulicBank  # noqa: E402
from digging_export import SECTION, BUCKET_X  # noqa: E402
from digging_path import LIP, HEEL  # noqa: E402

RANDOM_SEED = 20260926
GRAVITY = np.array([0., 0., -9.81])

W = 'witness_independent'
S = 'witness_same_convention'
F = 'finite_difference'
C = 'certificate'
THRESHOLDS = {
    # name: (limit, relation, unit, category)
    'plant_vs_original_backend_FK_position': (1e-12, '<=', 'm', S),
    'plant_vs_original_backend_FK_rotation': (1e-12, '<=', 'matrix entry', S),
    'plant_vs_source_sums_FK_position': (1e-11, '<=', 'm', W),
    'plant_closure_residual_vs_original_rows': (1e-12, '<=', 'm', S),
    'native_constraint_jacobian_vs_finite_difference': (5e-7, '<=', 'm per unit coordinate', F),
    'native_constraint_jacobian_vs_original': (1e-11, '<=', 'm per unit coordinate', S),
    'dependent_coordinates_plant_vs_PACDM_polished': (1e-10, '<=', 'mixed rad/m', W),
    'dependent_coordinates_plant_vs_original_reconstruct': (1e-10, '<=', 'mixed rad/m', S),
    'tangent_map_plant_vs_PACDM': (1e-9, '<=', 'mixed SI', W),
    'tangent_map_plant_vs_original_component': (1e-9, '<=', 'mixed SI', S),
    'tangent_map_plant_vs_projection_finite_difference': (2e-6, '<=', 'mixed SI', F),
    'constraint_drift_native_vs_original_analytic': (1e-9, '<=', 'm/s2', S),
    'constraint_drift_native_vs_finite_difference': (2e-6, '<=', 'm/s2', F),
    'mass_plant_vs_original_compact': (1e-12, '<=', 'normalized', S),
    'mass_plant_vs_source_sums': (1e-10, '<=', 'normalized', W),
    'bias_plant_vs_source_sums': (1e-10, '<=', 'normalized', W),
    'kinetic_energy_plant_vs_source_sums': (1e-9, '<=', 'relative', W),
    'potential_energy_change_plant_vs_source_sums': (1e-9, '<=', 'J per 1e5 J', W),
    'RNEA_vs_CRBA_plus_bias': (1e-12, '<=', 'normalized', W),
    # Acceleration comparisons use the force-equivalent (backward-error) metric
    # |M (a1 - a2)|_inf / max(1, |tau|_inf, |h|_inf): the tree mass matrix spans
    # heavy boom links and light cylinder rods, so raw acceleration differences
    # in rod coordinates are conditioning-amplified (reported in details).
    # Two load families: realistic port efforts from the original inverse model
    # ('ports') and dense random generalized forces ('random').
    'native_constraint_dynamics_vs_reduced': (1e-10, '<=', 'relative force-equivalent', C),
    'native_constraint_dynamics_vs_original_KKT': (1e-10, '<=', 'relative force-equivalent', S),
    'native_constraint_dynamics_vs_original_reduced': (1e-10, '<=', 'relative force-equivalent', S),
    'native_constraint_acceleration_residual': (1e-10, '<=', 'm/s2 per max(1,|a|)', C),
    'native_tangent_dynamics_residual': (1e-11, '<=', 'normalized', C),
    'constraint_reaction_power': (1e-7, '<=', 'W per 1e5 W', C),
    'inverse_forward_round_trip_commanded': (1e-8, '<=', 'normalized', W),
    'inverse_forward_round_trip_pin_q22': (1e-7, '<=', 'rad/s2', W),
}
# Negative controls: each value is the SMALLEST detection margin over the
# evaluated samples (a fault must be detected everywhere it is injected), in the
# units of the acceptance metric that must reject it; limits are >= 100x the
# corresponding acceptance limit.
NEGATIVE = {
    'negative_wrong_joint_axis_closure_gap': (1e-3, '>=', 'm'),                     # closure accepted <= 1e-12 m
    'negative_missing_loop_q9_mobility_increase': (1, '>=', 'dof'),                 # rank must stay 16
    'negative_omitted_constraint_drift_force_equivalent': (1e-8, '>=', 'relative force-equivalent'),  # <= 1e-10
    'negative_swapped_port_round_trip': (1e-2, '>=', 'normalized'),                 # round trip <= 1e-8
    'negative_bucket_mass_round_trip': (1e-4, '>=', 'normalized'),                  # round trip <= 1e-8
}


IDENTITIES = ('port_virtual_work', 'soil_wrench_virtual_work', 'soil_force_dissipative',
              'hydraulic_fluid_identity')


def _normalized(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.max(np.abs(actual - expected)) / max(1., float(np.max(np.abs(expected)))))


def _rotation_error(a, b):
    return float(np.max(np.abs(a[:3, :3] - b[:3, :3])))


class Witness:
    def __init__(self):
        self.maxima, self.minima, self.notes = {}, {}, {}

    def max(self, name, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite witness {name}')
        self.maxima[name] = max(self.maxima.get(name, -np.inf), value)

    def min(self, name, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite witness {name}')
        self.minima[name] = min(self.minima.get(name, np.inf), value)


def continuation(plant, q_seed, u_target, increment=.02):
    """Reach a target independent state from a closed seed in bounded steps."""
    plant.initialize(q_seed)
    start = q_seed[plant.active].copy()
    count = max(1, int(np.ceil(np.max(np.abs(u_target - start)) / increment)))
    for k in range(1, count + 1):
        q, jacobian, tangent, error = plant.project(start + (u_target - start) * k / count)
    return q, jacobian, tangent, error


def run(output=None):
    report = {'passed': False, 'checks': {}, 'details': {
        'scope': 'Finite model checks of the Pinocchio plant against preserved original implementations; '
                 'no contact, soil or hardware validation.',
        'versions': dict(python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__, pinocchio=pin.__version__),
        'random_seed': RANDOM_SEED, 'failures': []}}
    checks, details = report['checks'], report['details']

    def check(name, value, limit, relation, unit, category='structure'):
        value = float(value)
        passed = bool(np.isfinite(value) and {'<=': value <= limit, '>=': value >= limit,
                                              '==': value == limit}[relation])
        checks[name] = dict(passed=passed, value=value if np.isfinite(value) else None,
                            limit=float(limit), relation=relation, unit=unit, category=category)

    w = Witness()
    try:
        cmg, mapping, e, a, r, inverse = context()
        plant = ReducedPlant(cmg, e.cut_ids, e.independent_ids)
        pm = plant.model
        perm = np.array([pm.index[j] for j in e.tree_ids])
        ports = perm[[e.tree_ids.index(k) for k in PORTS]]
        E = np.zeros((plant.n, 8))
        E[ports, np.arange(8)] = 1.
        graph, pacdm = compile_pacdm(cmg, mapping, e.tree_ids)
        backend = e.native.pin_backend
        native = e.native
        floating = FloatingSource(cmg, e.source)
        geometry = BucketGeometry(SECTION, BUCKET_X, LIP, HEEL)
        pm.add_bucket_point('centroid', geometry.centroid_local)
        soil = SoilSurrogate(SoilParameters(), geometry)
        rng = np.random.default_rng(RANDOM_SEED)
        nominal_ctrl = r.nominal.copy()
        # The original controller object supplies the original PACDM polish step.
        controller = FixedBaseArmController(cmg, mapping, e, a, r, inverse, nominal_ctrl)

        def ctrl(x):
            return np.asarray(x)[perm]

        def plant_order(x):
            y = np.empty(plant.n)
            y[perm] = x
            return y

        def controller_inverse(qc, velocity, requested):
            """Original fixed-base controller inverse dynamics (FloatingSource rows 6:)."""
            fl = floating.evaluate(qc, np.zeros(3), np.eye(3), np.r_[np.zeros(6), velocity], GRAVITY)
            component = r.component(qc, velocity[e.active], GRAVITY)
            comp_ctrl = dict(component)
            comp_ctrl['mass_matrix'] = fl['mass_matrix'][6:, 6:]
            comp_ctrl['bias_forces'] = fl['bias_forces'][6:]
            return inverse.solve(comp_ctrl, requested, p0_force=0.)

        # --- structure --------------------------------------------------------
        details['pacdm_core_sha256'] = hashlib.sha256(paths.PACDM_CORE.read_bytes()).hexdigest()  # gated by runner
        check('plant_tree_coordinates', plant.n, 23, '==', 'count')
        check('plant_native_constraint_rows', 3 * len(pm.constraints), 54, '==', 'rows')
        check('plant_independent_coordinates', plant.active.size, 7, '==', 'count')
        check('plant_depth_first_compact_subtrees', int(NativeDynamics._has_compact_subtrees(pm.model)), 1,
              '==', 'boolean')
        total = sum(float(b['mass_kg']) for b in cmg['bodies'])
        plant_mass = sum(float(pm.model.inertias[i].mass) for i in range(pm.model.njoints))
        check('plant_total_mass_matches_CMG', abs(plant_mass - total), 1e-9, '<=', 'kg')
        check('plant_joint_numbering_differs_from_original_BFS', int(pm.tree_ids != list(e.tree_ids)), 1,
              '==', 'boolean')
        from .plant import LIP_LOCAL, HEEL_LOCAL
        check('plant_bucket_points_equal_source_constants',
              float(max(np.max(np.abs(LIP_LOCAL - LIP)), np.max(np.abs(HEEL_LOCAL - HEEL)))), 0., '==', 'm')
        details['plant_tree_order'] = pm.tree_ids
        details['controller_tree_order'] = list(e.tree_ids)
        details['cut_joint_ids'] = list(e.cut_ids)
        details['total_mass_kg'] = total

        # --- closed sample states -----------------------------------------------
        q_nominal = plant_order(nominal_ctrl)
        samples = [q_nominal.copy()]
        for _ in range(24):
            u = q_nominal[plant.active] + rng.uniform(-.35, .35, 7)
            q, _, _, _ = continuation(plant, q_nominal, u)
            samples.append(q.copy())
        details['closed_sample_count'] = len(samples)
        details['sample_independent_range'] = dict(
            minimum=np.min([s[plant.active] for s in samples], axis=0).tolist(),
            maximum=np.max([s[plant.active] for s in samples], axis=0).tolist(),
            ids=list(plant.independent_ids))
        drift_effect = []
        for index, q in enumerate(samples):
            plant.initialize(q)
            J = pm.closure_jacobian(q)
            N = plant.tangent(J)
            rank = int(np.linalg.matrix_rank(J, tol=1e-9))
            w.min('native_constraint_rank_min', rank)
            w.max('native_constraint_rank_max', rank)
            qc = ctrl(q)
            # FK against original BFS compiler and source sums.
            poses = pm.body_poses(q)
            backend.set_configuration(qc)
            original = backend.body_poses()
            source = e.source.evaluate(qc, np.zeros(23), GRAVITY)
            for body, pose in poses.items():
                w.max('plant_vs_original_backend_FK_position', np.linalg.norm(pose[:3, 3] - original[body][:3, 3]))
                w.max('plant_vs_original_backend_FK_rotation', _rotation_error(pose, original[body]))
                if body in source['poses']:
                    w.max('plant_vs_source_sums_FK_position',
                          np.linalg.norm(pose[:3, 3] - source['poses'][body][:3, 3]))
            # Closure rows: same geometric definition, also off the manifold.
            for offset in (0., 1.):
                qq = q + offset * rng.uniform(-.01, .01, plant.n)
                ours = pm.closure_residual(qq)
                backend.set_configuration(ctrl(qq))
                theirs, their_jacobian = backend.closure()
                w.max('plant_closure_residual_vs_original_rows', np.max(np.abs(ours - theirs)))
                ours_j = pm.closure_jacobian(qq)
                w.max('native_constraint_jacobian_vs_original', np.max(np.abs(ours_j[:, perm] - their_jacobian)))
                if index < 6:
                    fd = np.empty_like(ours_j)
                    for col in range(plant.n):
                        step = np.zeros(plant.n)
                        step[col] = 1e-7
                        fd[:, col] = (pm.closure_residual(qq + step) - pm.closure_residual(qq - step)) / 2e-7
                    w.max('native_constraint_jacobian_vs_finite_difference', np.max(np.abs(fd - ours_j)))
            # Dependent coordinates: PACDM (+ original controller polish) and original reconstruct.
            u = q[plant.active]
            seed = graph.lift(nominal_ctrl)
            assembled, info = pacdm.acquire(qc[e.active], seed)
            if not info['success']:
                raise RuntimeError(f'PACDM acquisition failed at sample {index}: {info}')
            w.max('dependent_coordinates_plant_vs_PACDM_raw_acquire', np.max(np.abs(assembled[:23] - qc)))
            polished = controller._polish(assembled, np.asarray(info['mapping']['rows'], dtype=int))
            w.max('dependent_coordinates_plant_vs_PACDM_polished', np.max(np.abs(polished[:23] - qc)))
            rebuilt = r.reconstruct(qc[e.active])
            w.max('dependent_coordinates_plant_vs_original_reconstruct', np.max(np.abs(rebuilt - qc)))
            # Tangent maps.
            Npacdm, minfo = pacdm.mapping(polished)
            if Npacdm is None:
                raise RuntimeError(f'PACDM mapping failed at sample {index}: {minfo}')
            w.max('tangent_map_plant_vs_PACDM', np.max(np.abs(N[perm] - Npacdm[:23])))
            udot = rng.uniform(-.5, .5, 7)
            component = r.component(qc, udot, GRAVITY)
            w.max('tangent_map_plant_vs_original_component', np.max(np.abs(N[perm] - component['tangent_map'])))
            if index < 6:
                fdN = np.empty_like(N)
                for col in range(7):
                    step = np.zeros(7)
                    step[col] = 1e-6
                    plant.initialize(q)
                    qp, _, _, _ = plant.project(u + step)
                    plant.initialize(q)
                    qm, _, _, _ = plant.project(u - step)
                    fdN[:, col] = (qp - qm) / 2e-6
                plant.initialize(q)
                w.max('tangent_map_plant_vs_projection_finite_difference', np.max(np.abs(fdN - N)))
            v = N @ udot
            gamma = pm.gamma(q, v)
            w.max('constraint_drift_native_vs_original_analytic',
                  np.max(np.abs(gamma - component['jdot_velocity'])))
            eps = 1e-6
            fd_gamma = (pm.closure_jacobian(q + eps * v) - pm.closure_jacobian(q - eps * v)) / (2 * eps) @ v
            w.max('constraint_drift_native_vs_finite_difference', np.max(np.abs(fd_gamma - gamma)))
            # Mass, bias, energy.
            M, h = pm.mass_bias(q, v)
            native.pin_model.gravity.linear = GRAVITY
            compact = np.array(pin.crba(native.pin_model, native.pin_data, qc[native._compact_source_indices]))
            compact = np.triu(compact) + np.triu(compact, 1).T
            compact = compact[np.ix_(native._source_compact_indices, native._source_compact_indices)]
            w.max('mass_plant_vs_original_compact', _normalized(M[np.ix_(perm, perm)], compact))
            sums = e.source.evaluate(qc, ctrl(v), GRAVITY)
            w.max('mass_plant_vs_source_sums', _normalized(M[np.ix_(perm, perm)], sums['mass_matrix']))
            w.max('bias_plant_vs_source_sums', _normalized(h[perm], sums['bias_forces']))
            kinetic, potential = pm.energy(q, v)
            w.max('kinetic_energy_plant_vs_source_sums',
                  abs(kinetic - sums['kinetic_energy']) / max(1., abs(sums['kinetic_energy'])))
            if index == 0:
                pe_offset = potential - sums['potential_energy']
                details['potential_energy_reference_offset_J'] = float(pe_offset)
            w.max('potential_energy_change_plant_vs_source_sums',
                  abs(potential - sums['potential_energy'] - pe_offset) / 1e5)
            acc_test = rng.uniform(-1., 1., plant.n)
            rnea = np.array(pin.rnea(pm.model, pm.data, q, v, acc_test))
            w.max('RNEA_vs_CRBA_plus_bias', _normalized(rnea, M @ acc_test + h))
            # Original controller inverse dynamics at this state (realistic port efforts).
            requested = rng.uniform(-.8, .8, 6)
            solution = controller_inverse(qc, ctrl(v), requested)
            effort = solution['physical_efforts']
            # Constrained forward dynamics for two load families:
            #   'ports'  - realistic hydraulic port efforts from the original inverse model;
            #   'random' - dense random generalized forces (linear-algebra stress test).
            loads = {'ports': E @ effort, 'random': rng.uniform(-2e4, 2e4, plant.n)}
            for family, tau in loads.items():
                native_a = plant.forward(q, v, tau)
                reduced_a = plant.reduced(q, v, tau, J, N, gamma)
                force_scale = max(1., float(np.max(np.abs(tau))), float(np.max(np.abs(h))))
                comp = dict(component)
                comp['mass_matrix'] = M[np.ix_(perm, perm)]
                comp['bias_forces'] = h[perm]
                kkt_a = plant_order(solve_kkt(comp, ctrl(tau))['acceleration'])
                red_a = plant_order(solve_reduced(comp, ctrl(tau))['acceleration'])
                for label, other in (('reduced', reduced_a), ('original_KKT', kkt_a),
                                     ('original_reduced', red_a)):
                    w.max(f'native_constraint_dynamics_vs_{label}',
                          np.max(np.abs(M @ (native_a - other))) / force_scale)
                    w.max(f'raw_acceleration_difference_vs_{label}_{family}_SI',
                          np.max(np.abs(native_a - other)))
                scale = max(1., float(np.max(np.abs(native_a))))
                w.max('native_constraint_acceleration_residual', np.max(np.abs(J @ native_a + gamma)) / scale)
                w.max(f'native_constraint_acceleration_residual_{family}_absolute_m_s2',
                      np.max(np.abs(J @ native_a + gamma)))
                tangent_residual = N.T @ (M @ native_a + h - tau)
                w.max('native_tangent_dynamics_residual',
                      float(np.max(np.abs(tangent_residual))) / max(1., float(np.max(np.abs(N.T @ tau))),
                                                                   float(np.max(np.abs(N.T @ h)))))
                reaction = M @ native_a + h - tau
                w.max('constraint_reaction_power', abs(float(reaction @ v)) / 1e5)
                w.max(f'peak_acceleration_{family}_SI', np.max(np.abs(native_a)))
                if family == 'ports':
                    # Negative control: a solver that omits J'(q)qdot, measured with the
                    # same force-equivalent metric as the native-vs-reduced gate.
                    omitted = plant.reduced(q, v, tau, J, N, np.zeros_like(gamma))
                    w.min('negative_omitted_constraint_drift_force_equivalent',
                          np.max(np.abs(M @ (omitted - reduced_a))) / force_scale)
                    drift_effect.append(float(np.max(np.abs(omitted[plant.active] - reduced_a[plant.active]))
                                              / max(1., float(np.max(np.abs(reduced_a[plant.active]))))))
                    a_rt = native_a
                    w.max('inverse_forward_round_trip_commanded', _normalized(a_rt[plant.active][:6], requested))
                    w.max('inverse_forward_round_trip_pin_q22',
                          abs(a_rt[plant.active][6] - solution['pin_acceleration']))
            port_velocity = v[ports]
            w.max('port_virtual_work', abs(effort @ port_velocity - (E @ effort) @ v) / 1e5)
            w.max('port_virtual_work', abs(effort @ port_velocity - (N.T @ (E @ effort)) @ udot) / 1e5)
            # Negative control: swapped cylinder port columns (p3 <-> p5).
            swapped = E.copy()
            swapped[:, [3, 5]] = swapped[:, [5, 3]]
            a_bad = plant.forward(q, v, swapped @ effort)
            w.min('negative_swapped_port_round_trip', _normalized(a_bad[plant.active][:6], requested))
            # Soil surrogate wrench mapping (generalized force and power identity).
            pts, rotation = pm.points(q, ('lip', 'mouth', 'centroid'))
            lip, j_lip = pts['lip']
            centroid, j_c = pts['centroid']
            lip_velocity = j_lip @ v
            lip_test = np.array([6.8, 0., -.12])  # inside the declared bed: exercise the force law
            sample = soil.evaluate(lip_test, lip_velocity, pts['mouth'][0],
                                   rotation @ geometry.mouth_normal_local, 50.)
            tau_soil = j_lip.T @ sample['cutting_force_N'] + j_c.T @ sample['weight_force_N']
            direct = sample['cutting_force_N'] @ lip_velocity + sample['weight_force_N'] @ (j_c @ v)
            w.max('soil_wrench_virtual_work', abs(tau_soil @ v - direct) / 1e3)
            w.max('soil_force_dissipative', float(sample['cutting_force_N'] @ lip_velocity))
            # Negative control: bucket (body_56) mass and inertia +25 % in the plant only.
            if index == 0:
                heavy = ReducedPlant(scaled_body_cmg(cmg, 'body_56', 1.25), e.cut_ids, e.independent_ids)
            heavy.initialize(q)
            a_heavy = heavy.forward(q, v, E @ effort)
            w.min('negative_bucket_mass_round_trip', _normalized(a_heavy[heavy.active][:6], requested))
        check('native_constraint_rank_all_samples', int(w.minima['native_constraint_rank_min'] == 16
                                                        and w.maxima['native_constraint_rank_max'] == 16),
              1, '==', 'boolean')
        details['omitted_drift_relative_independent_acceleration_change'] = dict(
            minimum=min(drift_effect), maximum=max(drift_effect), samples=len(drift_effect))

        # --- hydraulic model identities (original bank) -------------------------
        bank = HydraulicBank(nominal_ctrl[[e.tree_ids.index(k) for k in PORTS]], np.zeros(8))
        for _ in range(24):
            bank.pressure = rng.uniform(.3e6, 24e6, (6, 2))
            bank.rotary = rng.uniform(-1e3, 1e3, 2)
            qp = nominal_ctrl[[e.tree_ids.index(k) for k in PORTS]] + rng.uniform(-.2, .2, 8)
            vp = rng.uniform(-.2, .2, 8)
            item = bank.evaluate(qp, vp, rng.uniform(-2e5, 2e5, 8), 1e-3)
            w.max('hydraulic_fluid_identity', abs(item['fluid_identity_W']))

        # --- topology negative controls ------------------------------------------
        bad = deepcopy(cmg)
        for joint in bad['joints']:
            if joint['id'] == 'q7':
                joint['axis'] = [1., 0., 0.]
        wrong = ExcavatorPinModel(bad, e.cut_ids)
        for q in samples:
            wq = np.empty(wrong.model.nq)
            for j, value in zip(pm.tree_ids, q):
                wq[wrong.index[j]] = value
            w.min('negative_wrong_joint_axis_closure_gap', np.max(np.abs(wrong.closure_residual(wq))))
        contributions = {}
        for q in samples:
            J = pm.closure_jacobian(q)
            full = np.linalg.matrix_rank(J, tol=1e-9)
            for i, cut in enumerate(e.cut_ids):
                keep = np.r_[0:6 * i, 6 * (i + 1):6 * len(e.cut_ids)]
                gain = int(full - np.linalg.matrix_rank(J[keep], tol=1e-9))
                contributions.setdefault(cut, set()).add(gain)
                if cut == 'q9':
                    w.min('negative_missing_loop_q9_mobility_increase', gain)
        details['per_cut_rank_contribution'] = {k: sorted(v) for k, v in contributions.items()}
        single = ExcavatorPinModel(cmg, e.cut_ids, lever=0.)
        single_rank = sorted({int(np.linalg.matrix_rank(single.closure_jacobian(q), tol=1e-9)) for q in samples})
        details['single_point_cut_rank'] = single_rank
        details['closure_topology_notes'] = [
            'q15 and q18 (twin side links) are mutually redundant: removing either one leaves rank 16.',
            'q19 and q20 each contribute rank 1; all other cuts contribute rank 2 (planar loops).',
            'Replacing each revolute cut by a single-point (spherical) closure also gives rank 16 at every '
            'sampled state, i.e. the second point adds no independent row in this geometry (consistent '
            'with planar loops); no negative control is claimed for it.',
            'Removing the stick-cylinder rod pin q9 raises the mobility by 2 (negative control).']

        for name, (limit, relation, unit, category) in THRESHOLDS.items():
            check(name, w.maxima.get(name, np.inf), limit, relation, unit, category)
        for name, (limit, relation, unit) in NEGATIVE.items():
            check(name + '_detected', w.minima.get(name, -np.inf), limit, relation, unit, 'negative_control')
        details['identities'] = {k: w.maxima[k] for k in IDENTITIES if k in w.maxima}
        details['identities_note'] = ('Algebraic identities recorded for completeness; they cannot fail except '
                                      'through round-off and are not counted as checks.')
    except Exception:  # a failed check is evidence, not a crash
        import traceback
        details['failures'].append(traceback.format_exc())
        check('mechanics_validation_completed', 0, 1, '==', 'boolean')
    details['maxima'] = w.maxima
    details['minima'] = w.minima
    details['mujoco'] = sentinel.report()
    report['passed_checks'] = sum(c['passed'] for c in checks.values())
    report['total_checks'] = len(checks)
    report['passed'] = bool(checks and all(c['passed'] for c in checks.values()))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False, default=_plain) + '\n',
                                encoding='utf-8')
    return report


def _plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)
