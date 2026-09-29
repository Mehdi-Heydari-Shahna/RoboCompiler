"""Independent checks of the PACDM-reduced Kangaroo simulation.

Three separately implemented descriptions of the same CMG are compared:

1. the mission integrator's route: unchanged PACDM (accepted CutGraph
   kinematics, SE(3) cut residuals, tangent map ``N`` and curvature) plus the
   Pinocchio tree mass/bias of ``FloatingPinBackend``;
2. the native Pinocchio loop oracle (``native_oracle.NativeLoopOracle``):
   ``RigidConstraintModel(CONTACT_3D)`` for every cut point, native
   ``constraintDynamics`` and native frame kinematics for the universal
   angular rows;
3. the accepted NumPy source dynamics (``FloatingSource``), the accepted
   analytic closure rows (``UniversalMechanism``) and the accepted KKT
   solver (``constraint_solvers.solve_kkt``) from the original v22 package.

Routes 1 and 2 share the CMG-compiled Pinocchio tree and its inertias, so
their agreement verifies the loop reduction and force mapping, not the
authored inertias.  Route 3 shares only the CMG records.

Tolerances are written in this file and are not changed by any run.  They
were set during development, before the final runs; some limits were
temporarily widened during development and then restored after the
accepted closure polish was added to the integrator.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from .native_oracle import NativeLoopOracle
from .pin_backend import FloatingPinBackend
from .pin_simulation import KangarooPlant, PIN_SETTINGS

LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED


def _le(value, limit, units='', description=''):
    value = float(value)
    return dict(passed=bool(np.isfinite(value) and value <= limit), value=value,
                limit=float(limit), relation='<=', units=units, description=description)


def _ge(value, limit, units='', description=''):
    value = float(value)
    return dict(passed=bool(np.isfinite(value) and value >= limit), value=value,
                limit=float(limit), relation='>=', units=units, description=description)


def _eq(value, target, units='', description=''):
    return dict(passed=bool(value == target), value=value if isinstance(value, (int, float)) else str(value),
                limit=target, relation='==', units=units, description=description)


def world_transform(R):
    """Map world-convention floating velocity [pdot, omega_w, qd] to public local."""
    T = np.eye(82)
    T[:3, :3] = R.T
    T[3:6, 3:6] = R.T
    return T


def make_oracle(plant):
    return NativeLoopOracle(plant.backend, plant.cmg, plant.armature)


# ---------------------------------------------------------------------------
# Instantaneous mechanics suite
# ---------------------------------------------------------------------------
def validate_mechanics(plant: KangarooPlant | None = None, seed=20260926, states=24,
                       perturbed=8):
    plant = KangarooPlant() if plant is None else plant
    A, B, graph, solver = plant.A, plant.backend, plant.graph, plant.solver
    oracle = make_oracle(plant)
    source = A.FloatingSource(plant.cmg)
    accepted_pin = A.PinFloating(plant.cmg)
    rng = np.random.default_rng(seed)
    with np.load(plant.reference_path, allow_pickle=False) as z:
        ref_q = z['qaug'].copy()
    maxima = {}
    minima = {}

    def upd(key, value):
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError(f'Nonfinite mechanics value {key}')
        maxima[key] = max(maxima.get(key, 0.), value)

    def low(key, value):
        minima[key] = min(minima.get(key, np.inf), float(value))

    picks = np.linspace(0, len(ref_q) - 1, states).astype(int)
    samples = []
    for index, pick in enumerate(picks):
        seed_state = ref_q[pick].copy()
        # Leave the accepted reference manifold: new motor coordinates, PACDM assembly.
        u = seed_state[plant.active] + (0. if index == 0 else 1.) * rng.uniform(-.0015, .0015, len(plant.active))
        z, info = solver.acquire(u, seed_state)
        if not info['success']:
            raise RuntimeError(f'Mechanics sample {index}: PACDM acquisition failed {info}')
        # Accepted v22 practice for instantaneous checks: all-row Newton polish
        # (< 5e-13) after PACDM acceptance, so the 1e-9 corrector tolerance cannot
        # create spurious singular values in the accepted rank tests.
        z = A.polish(graph, z)
        N, mi = solver.mapping(z)
        upd('pacdm_all_row_residual', mi['residual_inf'])
        low('pacdm_rank_full', mi['rank_full'])
        low('pacdm_rank_passive', mi['rank_passive'])
        low('pacdm_rcond', mi['rcond'])
        R = Rotation.from_rotvec(rng.uniform(-.35, .35, 3)).as_matrix()
        p = rng.uniform(-.3, .3, 3) + [0., 0., .85]
        base = np.r_[p, pin.Quaternion(R).coeffs()]
        xi = np.r_[rng.uniform(-.4, .4, 6), rng.uniform(-.03, .03, len(plant.active))]
        N, info, J, Jdot, zd, k = plant.differential(z, xi[6:])
        q = plant.configuration(base, z)
        G, nu = plant.lift_velocity(N, xi)
        kf = np.r_[np.zeros(6), k[:plant.nt]]
        # ---- tree kinematics: accepted graph FK and NumPy source vs Pinocchio
        gposes, _ = graph.poses(z)
        bposes = B.poses(q)
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = p
        upd('fk_graph_vs_pinocchio', max(np.max(np.abs(T0 @ gposes[b] - bposes[b])) for b in bposes))
        T = world_transform(R)
        v_world = np.linalg.solve(T, nu)
        gravity = [0., 0., -9.81]
        src = source.evaluate(z[:plant.nt], p, R, v_world, gravity)
        M = B.mass(q)
        h = B.bias(q, nu)
        td = np.zeros(82)
        td[:3] = -R.T @ np.cross(v_world[3:6], v_world[:3])
        upd('mass_vs_numpy_source', np.max(np.abs(T.T @ M @ T - src['mass_matrix'])))
        upd('bias_vs_numpy_source', np.max(np.abs(T.T @ (h + M @ td) - src['bias_forces'])))
        energy = B.energy(q, nu)
        upd('kinetic_vs_numpy_source', abs(energy['kinetic_J'] - src['kinetic_energy']))
        upd('potential_vs_numpy_source', abs(energy['potential_J'] - src['potential_energy']))
        upd('poses_vs_numpy_source', max(np.max(np.abs(bposes[b] - src['poses'][b])) for b in bposes))
        frames = B.frame_jacobians(q, list(B.body_frame_ids))
        upd('jacobian_vs_numpy_source', max(np.max(np.abs(frames[b][1] @ T - src['jacobians'][b]))
                                            for b in B.body_frame_ids))
        acc_pin = accepted_pin.evaluate(z[:plant.nt], p, R, v_world, gravity)
        upd('mass_vs_accepted_pinfloating', np.max(np.abs(T.T @ M @ T - acc_pin['mass_matrix'])))
        upd('bias_vs_accepted_pinfloating', np.max(np.abs(T.T @ (h + M @ td) - acc_pin['bias_forces'])))
        # ---- tree dynamics identities
        g = B.gravity(q)
        Mcols = np.column_stack([B.inverse(q, np.zeros(82), e) - g for e in np.eye(82)])
        upd('crba_vs_rnea', np.max(np.abs(M - Mcols)))
        upd('kinetic_energy_identity', abs(energy['kinetic_J'] - .5 * nu @ M @ nu))
        grad = np.zeros(82)
        step = 1e-6
        for col in range(82):
            dv = np.zeros(82)
            dv[col] = step
            up = B.energy(B.integrate(q, dv, 1.), np.zeros(82))['potential_J']
            dn = B.energy(B.integrate(q, -dv, 1.), np.zeros(82))['potential_J']
            grad[col] = (up - dn) / (2 * step)
        upd('gravity_vs_potential_gradient', np.max(np.abs(grad - g)))
        body_ke = body_pe = 0.
        for body in plant.cmg['bodies']:
            Tb, Jb = frames[body['id']]
            twist = Jb @ nu
            off = Tb[:3, :3] @ np.asarray(body['com_m'])
            vc = twist[:3] + np.cross(twist[3:], off)
            Iw = Tb[:3, :3] @ np.asarray(body['inertia_com_kg_m2']) @ Tb[:3, :3].T
            body_ke += .5 * body['mass_kg'] * vc @ vc + .5 * twist[3:] @ Iw @ twist[3:]
            body_pe -= body['mass_kg'] * np.array(gravity) @ (Tb[:3, 3] + off)
        upd('cmg_body_kinetic_sum', abs(body_ke - energy['kinetic_J']))
        upd('cmg_body_potential_sum', abs(body_pe - energy['potential_J']))
        # ---- native loop geometry and PACDM tangent/curvature
        geo = oracle.geometry(q, nu)
        Jn = geo['jacobian']
        upd('native_point_gap', np.max(np.abs(geo['point_residual'])))
        upd('native_universal_residual', np.max(np.abs(geo['angular_residual'])))
        low('native_constraint_rank', np.linalg.matrix_rank(Jn, tol=1e-9))
        upd('native_tangent_JG', np.max(np.abs(Jn @ G)))
        upd('native_velocity_closure', np.max(np.abs(Jn @ nu)))
        upd('native_curvature_closure', np.max(np.abs(Jn @ kf + geo['bias'])))
        # ---- constrained forward dynamics: three routes
        F = rng.uniform(-400., 400., len(plant.active))
        corner = plant.corner_names[int(rng.integers(len(plant.corner_names)))]
        cf = rng.uniform([-60, -60, 0], [60, 60, 250])
        push = rng.uniform(-80., 80., 3)
        cj = B.frame_jacobians(q, [corner, 'torso_com'])
        tau = (plant.S @ F - plant.damping * nu + cj[corner][1][:3].T @ cf
               + cj['torso_com'][1][:3].T @ push)
        Ma = M + np.diag(plant.armature)
        Mr = G.T @ Ma @ G
        br = G.T @ (h + Ma @ kf)
        a_pacdm = G @ np.linalg.solve(Mr, G.T @ tau - br) + kf
        a_native, stats = oracle.acceleration(q, nu, tau)
        a_kkt, _ = oracle.acceleration_kkt(q, nu, tau)
        scale = max(1., float(np.max(np.abs(a_native))))
        upd('pacdm_vs_native_constraint_dynamics', np.max(np.abs(a_pacdm - a_native)) / scale)
        upd('native_vs_dense_native_kkt', np.max(np.abs(a_kkt - a_native)) / scale)
        upd('native_acceleration_closure', np.max(np.abs(Jn @ a_native + geo['bias'])))
        upd('pacdm_acceleration_closure', np.max(np.abs(Jn @ a_pacdm + geo['bias'])))
        low('reduced_mass_min_eigenvalue', np.linalg.eigvalsh(Mr).min())
        low('schur_condition_inverse', 1. / stats['schur_condition'])
        # Accepted NumPy route: world convention, accepted closure rows and KKT.
        # add_drive_terms adds armature to M and the viscous term D v to the
        # bias, so the applied load passed to solve_kkt excludes damping.
        ud_world = np.r_[v_world[:6], xi[6:]]
        src_state = source.state(z[:plant.nt], p, R, ud_world, gravity)
        upd('accepted_tangent_velocity', np.max(np.abs(src_state['velocity'] - v_world)))
        comp = A.add_drive_terms(src_state, plant.cmg)
        applied = (plant.S @ F + cj[corner][1][:3].T @ cf + cj['torso_com'][1][:3].T @ push)
        kkt = A.solve_kkt(comp, T.T @ applied)
        # a_local = T a_world + Tdot v_world with Tdot v_world = td.
        a_world = np.linalg.solve(T, a_native - td)
        upd('native_vs_accepted_numpy_kkt', np.max(np.abs(kkt['acceleration'] - a_world))
            / max(1., float(np.max(np.abs(a_world)))))
        # ---- virtual power of the ideal loop reaction
        reaction = Ma @ a_native + h - tau
        upd('loop_reaction_reduced', np.max(np.abs(G.T @ reaction)) / max(1., float(np.max(np.abs(tau)))))
        upd('loop_reaction_power', abs(reaction @ nu))
        # ---- contact point Jacobian vs finite motion along the PACDM manifold
        eps = 1e-6
        zp, _ = solver.correct(z[plant.active] + eps * xi[6:], z[plant.passive] + eps * N[plant.passive] @ xi[6:],
                               None, np.asarray(info['rows']), maxiter=20)
        zm, _ = solver.correct(z[plant.active] - eps * xi[6:], z[plant.passive] - eps * N[plant.passive] @ xi[6:],
                               None, np.asarray(info['rows']), maxiter=20)
        bp = pin.SE3ToXYZQUAT(pin.XYZQUATToSE3(base) * pin.exp6(pin.Motion(eps * xi[:6])))
        bm = pin.SE3ToXYZQUAT(pin.XYZQUATToSE3(base) * pin.exp6(pin.Motion(-eps * xi[:6])))
        xp = B.frame_placement(plant.configuration(bp, zp), corner)[:3, 3]
        xm = B.frame_placement(plant.configuration(bm, zm), corner)[:3, 3]
        upd('corner_velocity_vs_manifold_difference', np.max(np.abs((xp - xm) / (2 * eps) - cj[corner][1][:3] @ nu)))
        # ---- free-flyer conventions
        upd('free_flyer_integrate_vs_exp6', np.max(np.abs(
            B.integrate(q, np.r_[xi[:6], np.zeros(plant.nt)], 1e-3)[:7]
            - pin.SE3ToXYZQUAT(pin.XYZQUATToSE3(base) * pin.exp6(pin.Motion(1e-3 * xi[:6]))))))
        lin, ang = B.centroidal_momentum(q, nu)
        com = B.center_of_mass(q)
        body_lin = np.zeros(3)
        body_ang = np.zeros(3)
        for body in plant.cmg['bodies']:
            Tb, Jb = frames[body['id']]
            twist = Jb @ nu
            off = Tb[:3, :3] @ np.asarray(body['com_m'])
            vc = twist[:3] + np.cross(twist[3:], off)
            Iw = Tb[:3, :3] @ np.asarray(body['inertia_com_kg_m2']) @ Tb[:3, :3].T
            body_lin += body['mass_kg'] * vc
            body_ang += np.cross(Tb[:3, 3] + off - com, body['mass_kg'] * vc) + Iw @ twist[3:]
        upd('centroidal_momentum_vs_body_sum', max(np.max(np.abs(lin - body_lin)), np.max(np.abs(ang - body_ang))))
        samples.append(dict(index=index, reference_sample=int(pick), motor_coordinates=u.tolist(),
                            base=base.tolist(), reduced_velocity=xi.tolist(), motor_force_N=F.tolist(),
                            corner=corner, corner_force_N=cf.tolist(), push_N=push.tolist(),
                            native_iterations=int(stats['point_iterations'])))
    # ---- perturbed-seed acquisitions
    acquisitions = []
    for index in range(perturbed):
        pick = int(rng.integers(len(ref_q)))
        truth = ref_q[pick].copy()
        seed_state = truth.copy()
        phys = plant.passive[plant.passive < plant.nt]
        seed_state[phys] += rng.uniform(-.004, .004, len(phys))
        seed_state = graph.lift(seed_state[:plant.nt])
        seed_state[plant.active] = truth[plant.active]
        solved, info = solver.acquire(truth[plant.active], seed_state)
        q_s = plant.configuration(np.r_[0, 0, .85, 0, 0, 0, 1], solved)
        geo = oracle.geometry(q_s)
        acquisitions.append(dict(reference_sample=pick, success=bool(info['success']),
                                 physical_coordinate_error=float(np.max(np.abs(solved[:plant.nt] - truth[:plant.nt]))),
                                 native_point_gap=float(np.max(np.abs(geo['point_residual']))),
                                 native_universal_residual=float(np.max(np.abs(geo['angular_residual']))),
                                 accepted_steps=info['accepted_steps'], rejected_steps=info['rejected_steps']))
    # ---- contact solver cross-check (PGS vs ADMM, both native) at a stance state
    contact = contact_solver_crosscheck(plant)
    # ---- mechanics negative controls
    negative = mechanics_negative_controls(plant, oracle)
    limits = dict(pacdm_all_row_residual=1e-8, fk_graph_vs_pinocchio=1e-12,
                  mass_vs_numpy_source=1e-11, bias_vs_numpy_source=1e-9,
                  kinetic_vs_numpy_source=1e-11, potential_vs_numpy_source=1e-11,
                  poses_vs_numpy_source=1e-12, jacobian_vs_numpy_source=1e-11,
                  mass_vs_accepted_pinfloating=1e-11, bias_vs_accepted_pinfloating=1e-9,
                  crba_vs_rnea=1e-10, kinetic_energy_identity=1e-10,
                  gravity_vs_potential_gradient=5e-6, cmg_body_kinetic_sum=1e-10,
                  cmg_body_potential_sum=1e-10, native_point_gap=1e-8,
                  native_universal_residual=1e-8, native_tangent_JG=1e-9,
                  native_velocity_closure=1e-9, native_curvature_closure=1e-6,
                  pacdm_vs_native_constraint_dynamics=1e-8, native_vs_dense_native_kkt=1e-6,
                  native_acceleration_closure=1e-6, pacdm_acceleration_closure=1e-6,
                  native_vs_accepted_numpy_kkt=1e-6, accepted_tangent_velocity=1e-9,
                  loop_reaction_reduced=1e-9,
                  loop_reaction_power=1e-8, corner_velocity_vs_manifold_difference=1e-6,
                  free_flyer_integrate_vs_exp6=1e-14, centroidal_momentum_vs_body_sum=1e-10)
    checks = {name: _le(maxima[name], limit) for name, limit in limits.items()}
    checks['pacdm_rank_full'] = _ge(minima['pacdm_rank_full'], 128, 'rank')
    checks['pacdm_rank_passive'] = _ge(minima['pacdm_rank_passive'], 128, 'rank')
    checks['pacdm_rcond'] = _ge(minima['pacdm_rcond'], 1e-6, 'reciprocal condition')
    checks['native_constraint_rank'] = _ge(minima['native_constraint_rank'], 64, 'rank')
    checks['positive_reduced_mass'] = _ge(minima['reduced_mass_min_eigenvalue'], 1e-4, 'mixed SI')
    checks['schur_conditioning'] = _ge(minima['schur_condition_inverse'], 1e-6, 'reciprocal condition')
    checks['perturbed_seed_success_count'] = _ge(sum(a['success'] for a in acquisitions), perturbed, 'cases')
    checks['perturbed_seed_coordinate_error'] = _le(max(a['physical_coordinate_error'] for a in acquisitions), 1e-7, 'm or rad')
    checks['perturbed_seed_native_gap'] = _le(max(a['native_point_gap'] for a in acquisitions), 1e-8, 'm')
    for key, value in contact['checks'].items():
        checks['contact_solver.' + key] = value
    for key, value in negative['checks'].items():
        checks['negative_control.' + key] = value
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks,
                details=dict(pinocchio_version=pin.__version__, random_seed=seed, states=len(samples),
                             perturbed_seed_cases=perturbed, maxima=maxima,
                             minima={k: float(v) for k, v in minima.items()},
                             native_constraint_count=len(oracle.constraints),
                             native_point_rows=oracle.point_rows, native_angular_rows=oracle.angular_rows,
                             native_proximal_settings=dict(accuracy=oracle.settings[0], mu=oracle.settings[2],
                                                           max_iterations=oracle.settings[3]),
                             independence='Route 1: unchanged PACDM + Pinocchio tree. Route 2: native Pinocchio CONTACT_3D constraints, constraintDynamics and native universal rows. Route 3: accepted NumPy source dynamics, closures and KKT. Routes 1 and 2 share the Pinocchio tree inertias.',
                             samples=samples, acquisitions=acquisitions, contact_solver=contact['details'],
                             negative_controls=negative['details']))


def contact_problem(plant, base, z, xi, dt=1e-3, force=None, mu=.8):
    """Rigid-contact Delassus problem exactly as assembled by the integrator."""
    B = plant.backend
    N, info, J, Jdot, zd, k = plant.differential(z, xi[6:])
    q = plant.configuration(base, z)
    G, nu = plant.lift_velocity(N, xi)
    kf = np.r_[np.zeros(6), k[:plant.nt]]
    M = B.mass(q) + np.diag(plant.armature)
    h = B.bias(q, nu)
    frames = B.frame_jacobians(q, plant.corner_names)
    Mr = G.T @ M @ G
    br = G.T @ (h + M @ kf)
    Dr = G.T @ (plant.damping[:, None] * G)
    force = np.zeros(len(plant.active)) if force is None else force
    Aimp = Mr + dt * Dr
    xi_free = np.linalg.solve(Aimp, Mr @ xi + dt * (G.T @ (plant.S @ force) - br))
    Jc = np.vstack([frames[n][1][:3] for n in plant.corner_names]) @ G
    gaps = np.array([frames[n][0][2, 3] for n in plant.corner_names])
    Ainv_JcT = np.linalg.solve(Aimp, Jc.T)
    W = Jc @ Ainv_JcT
    W = .5 * (W + W.T)
    g = Jc @ xi_free
    g[2::3] += gaps / dt
    return W, g, gaps, Jc, Ainv_JcT, xi_free


def ncp_residuals(lam, sigma, mu):
    """Native cone checks for a stacked (t1, t2, n) impulse/velocity pair."""
    cone = pin.CoulombFrictionCone(mu)
    primal = dual = comp = 0.
    for i in range(len(lam) // 3):
        l = lam[3 * i:3 * i + 3]
        s = sigma[3 * i:3 * i + 3].copy()
        primal = max(primal, float(np.linalg.norm(l - cone.project(l))))
        corrected = s + np.array([0., 0., mu * np.linalg.norm(s[:2])])
        dual = max(dual, float(np.linalg.norm(corrected - cone.dual().project(corrected))))
        comp = max(comp, abs(float(l @ corrected)))
    return primal, dual, comp


def contact_solver_crosscheck(plant, mu=.8):
    """Solve stance contact problems with native PGS and native ADMM."""
    with np.load(plant.reference_path, allow_pickle=False) as z:
        ref = {k: z[k] for k in ['qaug', 'base', 'rotvec']}
    rng = np.random.default_rng(11)
    maxima = dict(pgs_ncp_primal=0., pgs_ncp_dual=0., pgs_ncp_complementarity=0.,
                  admm_ncp_primal=0., admm_ncp_dual=0., admm_ncp_complementarity=0.,
                  post_velocity_pgs_vs_admm=0., foot_impulse_pgs_vs_admm=0.)
    details = []
    for pick in [40, 120, 200, 300, 380]:
        R = Rotation.from_rotvec(ref['rotvec'][pick]).as_matrix()
        base = np.r_[ref['base'][pick], pin.Quaternion(R).coeffs()]
        xi = np.r_[rng.uniform(-.05, .05, 6), rng.uniform(-.005, .005, len(plant.active))]
        W, g, gaps, Jc, Ainv_JcT, xi_free = contact_problem(plant, base, ref['qaug'][pick], xi, mu=mu)
        cones = pin.StdVec_CoulombFrictionCone()
        for _ in plant.corner_names:
            cones.append(pin.CoulombFrictionCone(mu))
        pgs = pin.PGSContactSolver(len(g))
        pgs.setAbsolutePrecision(PIN_SETTINGS['pgs_absolute_precision'])
        pgs.setRelativePrecision(PIN_SETTINGS['pgs_relative_precision'])
        pgs.setMaxIterations(20000)
        x_pgs = np.zeros(len(g))
        pgs.solve(W, g, cones, x_pgs)
        admm = pin.ADMMContactSolver(len(g), 1e-8)
        admm.setAbsolutePrecision(1e-12)
        admm.setRelativePrecision(1e-14)
        admm.setMaxIterations(20000)
        admm.solve(pin.DelassusOperatorDense(W), g, cones, np.zeros(len(g)), np.zeros(len(g)))
        x_admm = np.asarray(admm.getPrimalSolution()).copy()
        for label, x in [('pgs', x_pgs), ('admm', x_admm)]:
            primal, dual, comp = ncp_residuals(x, W @ x + g, mu)
            maxima[f'{label}_ncp_primal'] = max(maxima[f'{label}_ncp_primal'], primal)
            maxima[f'{label}_ncp_dual'] = max(maxima[f'{label}_ncp_dual'], dual)
            maxima[f'{label}_ncp_complementarity'] = max(maxima[f'{label}_ncp_complementarity'], comp)
        maxima['post_velocity_pgs_vs_admm'] = max(maxima['post_velocity_pgs_vs_admm'],
                                                  float(np.max(np.abs(Ainv_JcT @ (x_pgs - x_admm)))))
        foot = lambda x: np.r_[x[:12].reshape(4, 3).sum(axis=0), x[12:].reshape(4, 3).sum(axis=0)]
        maxima['foot_impulse_pgs_vs_admm'] = max(maxima['foot_impulse_pgs_vs_admm'],
                                                 float(np.max(np.abs(foot(x_pgs) - foot(x_admm)))))
        details.append(dict(reference_sample=pick, pgs_iterations=pgs.getIterationCount(),
                            admm_iterations=admm.getIterationCount(),
                            normal_impulse_Ns=float(x_pgs[2::3].sum())))
    limits = dict(pgs_ncp_primal=1e-12, pgs_ncp_dual=1e-8, pgs_ncp_complementarity=1e-10,
                  admm_ncp_primal=1e-10, admm_ncp_dual=1e-8, admm_ncp_complementarity=1e-10,
                  post_velocity_pgs_vs_admm=1e-7, foot_impulse_pgs_vs_admm=1e-6)
    return dict(checks={k: _le(maxima[k], v) for k, v in limits.items()},
                details=dict(maxima=maxima, problems=details,
                             note='Each foot has four coplanar corners, so individual corner impulses are not unique; post-contact velocities and per-foot impulses are.'))


def mechanics_negative_controls(plant, oracle):
    """The checks above must detect deliberately wrong models."""
    with np.load(plant.reference_path, allow_pickle=False) as z:
        state = z['qaug'][200].copy()
    B = plant.backend
    rng = np.random.default_rng(5)
    base = np.r_[0., 0., .8, 0., 0., 0., 1.]
    xi = np.r_[rng.uniform(-.2, .2, 6), rng.uniform(-.02, .02, len(plant.active))]
    N, info, J, Jdot, zd, k = plant.differential(state, xi[6:])
    q = plant.configuration(base, state)
    G, nu = plant.lift_velocity(N, xi)
    kf = np.r_[np.zeros(6), k[:plant.nt]]
    M = B.mass(q) + np.diag(plant.armature)
    h = B.bias(q, nu)
    tau = plant.S @ rng.uniform(-300, 300, len(plant.active)) - plant.damping * nu
    a_native, _ = oracle.acceleration(q, nu, tau)
    Mr = G.T @ M @ G
    good = G @ np.linalg.solve(Mr, G.T @ tau - G.T @ (h + M @ kf)) + kf
    # 1: omitted curvature
    no_curv = G @ np.linalg.solve(Mr, G.T @ tau - G.T @ h)
    # 2: omitted armature
    M0 = B.mass(q)
    no_arm = G @ np.linalg.solve(G.T @ M0 @ G, G.T @ tau - G.T @ (h + M0 @ kf)) + kf
    # 3: universal angular rows removed from the native oracle (point rows only)
    B.model.armature[:] = oracle.armature_native
    try:
        a_points, _ = oracle._native_dynamics(B.native_q(q), B.native_v(nu), B.native_v(tau, 'tau'))
    finally:
        B.model.armature[:] = 0.
    a_points = B.from_native_v(a_points)
    # 4: wrong joint axis in one hip joint
    altered = deepcopy(plant.cmg)
    for joint in altered['joints']:
        if joint['id'] == 'leg_left_2_joint':
            joint['axis'] = list(np.roll(np.asarray(joint['axis']), 1))
    wrong = FloatingPinBackend(altered)
    fk_error = max(np.max(np.abs(wrong.poses(q)[b] - B.poses(q)[b])) for b in B.body_frame_ids)
    scale = max(1., float(np.max(np.abs(a_native))))
    values = dict(omitted_curvature=np.max(np.abs(no_curv - a_native)) / scale,
                  omitted_armature=np.max(np.abs(no_arm - a_native)) / scale,
                  omitted_universal_rows=np.max(np.abs(a_points - a_native)) / scale,
                  wrong_joint_axis_fk_m=fk_error,
                  correct_model=np.max(np.abs(good - a_native)) / scale)
    # "Detected" = the error exceeds 100 x the acceptance limit of the check that guards it
    # (pacdm_vs_native_constraint_dynamics <= 1e-8; fk_graph_vs_pinocchio <= 1e-12 m).
    guard = 'detection requires >= 100 x the guarding acceptance limit'
    checks = {'omitted_curvature_detected': _ge(values['omitted_curvature'], 100 * 1e-8, 'relative', guard),
              'omitted_armature_detected': _ge(values['omitted_armature'], 100 * 1e-8, 'relative', guard),
              'omitted_universal_rows_detected': _ge(values['omitted_universal_rows'], 100 * 1e-8, 'relative', guard),
              'wrong_joint_axis_detected': _ge(values['wrong_joint_axis_fk_m'], 100 * 1e-12, 'm', guard),
              'unaltered_model_accepted': _le(values['correct_model'], 1e-8, 'relative')}
    return dict(checks=checks, details={k: float(v) for k, v in values.items()})


# ---------------------------------------------------------------------------
# Trajectory audits against native Pinocchio dynamics
# ---------------------------------------------------------------------------
def select_audit_steps(arrays, uniform=111):
    """Uniform stored states plus every stored contact event."""
    stored = arrays['stored_step']
    reasons = arrays['stored_reason']
    sample = stored[np.char.find(reasons.astype(str), 'sample') >= 0]
    picks = set(sample[np.unique(np.linspace(0, len(sample) - 1, min(uniform, len(sample))).astype(int))].tolist())
    events = stored[np.char.find(reasons.astype(str), 'sample') < 0]
    picks.update(events.tolist())
    # The largest-impulse, largest-force and push-peak stored states.
    ground = arrays['lam'][stored, 2::3].sum(axis=1)
    picks.add(int(stored[np.argmax(ground)]))
    picks.add(int(stored[np.argmax(np.abs(arrays['act'][stored]).max(axis=1))]))
    picks.add(int(stored[np.argmax(np.linalg.norm(arrays['push'][stored], axis=1))]))
    last = len(arrays['time']) - 1
    picks = sorted(p for p in picks if p < last)
    return picks


def audit_rollout(plant, oracle, npz_path, dt, mu, uniform=111):
    """Native dynamics, closure, contact and momentum audit of one saved rollout."""
    with np.load(Path(npz_path), allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    B = plant.backend
    if list(a['coordinate_ids']) != plant.ids or list(a['corner_names']) != plant.corner_names:
        raise ValueError('Saved coordinate or corner order differs from the plant')
    n = len(a['time'])
    required = dict(base=(n, 7), xi=(n, 6 + plant.na), act=(n, plant.na), lam=(n, 24), push=(n, 3))
    for key, shape in required.items():
        if a[key].shape != shape or not np.all(np.isfinite(a[key])):
            raise ValueError(f'Invalid saved {key}: expected {shape}')
    if n > 1 and np.any(np.diff(a['time']) <= 0.):
        raise ValueError('Saved timestamps must increase strictly')
    stored = {int(s): i for i, s in enumerate(a['stored_step'])}
    picks = select_audit_steps(a, uniform)
    maxima = {key: 0. for key in [
        'dynamics_relative', 'dynamics_absolute', 'native_point_gap', 'native_universal_residual',
        'native_velocity_closure', 'native_post_velocity_closure', 'native_acceleration_closure',
        'tangent_JG', 'ncp_primal', 'ncp_dual', 'ncp_complementarity', 'loop_reaction_power',
        'post_step_gap_prediction']}
    records = []
    bounds = plant.force_bounds
    for k in picks:
        z_k = a['stored_z'][stored[k]]
        if not np.array_equal(z_k[plant.active], a['motor'][k]):
            raise ValueError(f'Stored state {k} does not match its motor coordinates')
        base = a['base'][k]
        xi = a['xi'][k]
        xi_next = a['xi'][k + 1]
        N, info, J, Jdot, zd, kz = plant.differential(z_k, xi[6:])
        q = plant.configuration(base, z_k)
        G, nu = plant.lift_velocity(N, xi)
        nu_next = G @ xi_next
        kf = np.r_[np.zeros(6), kz[:plant.nt]]
        force = np.clip(a['act'][k], bounds[:, 0], bounds[:, 1])
        frames = B.frame_jacobians(q, plant.corner_names + ['torso_com'])
        lam = a['lam'][k]
        tau = (plant.S @ force - plant.damping * nu_next + frames['torso_com'][1][:3].T @ a['push'][k]
               + sum(frames[c][1][:3].T @ lam[3 * i:3 * i + 3] for i, c in enumerate(plant.corner_names)) / dt)
        acc_int = G @ ((xi_next - xi) / dt) + kf
        acc_nat, _ = oracle.acceleration(q, nu, tau)
        scale = max(1., float(np.max(np.abs(acc_nat))))
        diff = float(np.max(np.abs(acc_int - acc_nat)))
        geo = oracle.geometry(q, nu)
        Jn = geo['jacobian']
        maxima['dynamics_absolute'] = max(maxima['dynamics_absolute'], diff)
        maxima['dynamics_relative'] = max(maxima['dynamics_relative'], diff / scale)
        maxima['native_point_gap'] = max(maxima['native_point_gap'], float(np.max(np.abs(geo['point_residual']))))
        maxima['native_universal_residual'] = max(maxima['native_universal_residual'],
                                                  float(np.max(np.abs(geo['angular_residual']))))
        maxima['native_velocity_closure'] = max(maxima['native_velocity_closure'], float(np.max(np.abs(Jn @ nu))))
        maxima['native_post_velocity_closure'] = max(maxima['native_post_velocity_closure'],
                                                     float(np.max(np.abs(Jn @ nu_next))))
        maxima['native_acceleration_closure'] = max(maxima['native_acceleration_closure'],
                                                    float(np.max(np.abs(Jn @ acc_int + geo['bias']))))
        maxima['tangent_JG'] = max(maxima['tangent_JG'], float(np.max(np.abs(Jn @ G))))
        M = B.mass(q) + np.diag(plant.armature)
        h = B.bias(q, nu)
        reaction = M @ acc_nat + h - tau
        maxima['loop_reaction_power'] = max(maxima['loop_reaction_power'], abs(float(reaction @ nu)))
        # Contact: native corner Jacobians, post-step velocity, cone checks.
        Jc = np.vstack([frames[c][1][:3] for c in plant.corner_names])
        gaps = np.array([frames[c][0][2, 3] for c in plant.corner_names])
        sigma = Jc @ nu_next
        sigma[2::3] += gaps / dt
        primal, dual, comp = ncp_residuals(lam, sigma, mu)
        maxima['ncp_primal'] = max(maxima['ncp_primal'], primal)
        maxima['ncp_dual'] = max(maxima['ncp_dual'], dual)
        maxima['ncp_complementarity'] = max(maxima['ncp_complementarity'], comp)
        maxima['post_step_gap_prediction'] = max(maxima['post_step_gap_prediction'],
                                                 float(np.max(np.abs((gaps + dt * (Jc @ nu_next)[2::3]) - a['gap'][k + 1]))))
        records.append(dict(step=int(k), time_s=float(a['time'][k]), dynamics_relative=diff / scale,
                            ground_impulse_Ns=float(lam[2::3].sum()), reason=str(a['stored_reason'][stored[k]])))
    # Every stored state: native closure (positions only, independent of PACDM).
    closure_gap = closure_universal = 0.
    for i, k in enumerate(a['stored_step']):
        geo = oracle.geometry(plant.configuration(a['base'][k], a['stored_z'][i]))
        closure_gap = max(closure_gap, float(np.max(np.abs(geo['point_residual']))))
        closure_universal = max(closure_universal, float(np.max(np.abs(geo['angular_residual']))))
    # Every step: discrete linear/angular momentum balance with external impulses.
    m = plant.mass_total
    gravity = np.array([0., 0., -9.81])
    lin = a['linear_momentum']
    ground = a['lam'].reshape(n, -1, 3).sum(axis=1)
    impulse = ground[:-1] + dt * (m * gravity + a['push'][:-1])
    lin_err = np.cumsum((lin[1:] - lin[:-1]) - impulse, axis=0)
    ang = a['angular_momentum']
    ang_impulse = (a['contact_moment'][:-1]
                   + dt * (np.cross(a['com'][:-1], m * gravity) + np.cross(a['torso_com'][:-1], a['push'][:-1])))
    ang_err = np.cumsum((ang[1:] - ang[:-1]) - ang_impulse, axis=0)
    limits = dict(dynamics_relative=1e-7, dynamics_absolute=1e-4, native_point_gap=1e-8,
                  native_universal_residual=1e-8, native_velocity_closure=1e-8,
                  native_post_velocity_closure=1e-8, native_acceleration_closure=1e-5,
                  tangent_JG=1e-8, ncp_primal=1e-9, ncp_dual=1e-6, ncp_complementarity=1e-8,
                  loop_reaction_power=1e-6, post_step_gap_prediction=1e-5)
    checks = {k: _le(v, limits[k]) for k, v in maxima.items()}
    checks['stored_state_native_point_gap'] = _le(closure_gap, 1e-8, 'm')
    checks['stored_state_native_universal_residual'] = _le(closure_universal, 1e-8)
    checks['linear_impulse_balance'] = _le(float(np.max(np.abs(lin_err))) if len(lin_err) else 0., .05, 'N s')
    checks['angular_impulse_balance'] = _le(float(np.max(np.abs(ang_err))) if len(ang_err) else 0., .05, 'N m s')
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks,
                details=dict(source=Path(npz_path).name, audited_steps=len(picks),
                             stored_states_checked=int(len(a['stored_step'])),
                             dynamics='native constraintDynamics (CONTACT_3D, Kp=Kd=0) + native universal rows via Schur complement',
                             records=records))


def validate_audit_negative_controls(plant, oracle, npz_path, dt, mu):
    """The auditor must reject deliberately corrupted rollouts."""
    from tempfile import TemporaryDirectory
    with np.load(Path(npz_path), allow_pickle=False) as z:
        source = {k: z[k] for k in z.files}
    # Small fixture in stance: the stored state with the largest ground impulse
    # and its two stored neighbours (so every mutation acts on a loaded contact).
    stored = source['stored_step']
    ground = source['lam'][stored, 2::3].sum(axis=1)
    peak = int(np.argmax(ground))
    lo = max(0, min(peak - 1, len(stored) - 3))
    keep_steps = sorted(set(stored[lo:lo + 3].tolist()))
    last = keep_steps[-1] + 2
    if last > len(source['time']) or ground[peak] <= 0.:
        raise ValueError('Audit-control fixture needs a loaded stored state before the final step')
    fixture = {}
    for key, value in source.items():
        if value.ndim and len(value) == len(source['time']):
            fixture[key] = value[:last].copy()
        else:
            fixture[key] = value.copy()
    mask = np.isin(source['stored_step'], keep_steps)
    fixture['stored_step'] = source['stored_step'][mask]
    fixture['stored_z'] = source['stored_z'][mask]
    fixture['stored_reason'] = source['stored_reason'][mask]
    outcomes = {}
    with TemporaryDirectory(prefix='kangaroo_audit_controls_') as work:
        path = Path(work) / 'control.npz'

        def run(data):
            np.savez_compressed(path, **data)
            return audit_rollout(plant, oracle, path, dt, mu, uniform=3)

        baseline = run(fixture)
        outcomes['unaltered_fixture'] = dict(passed=bool(baseline['passed']), expected='accept',
                                             observed='accept' if baseline['passed'] else 'reject')
        for label, mutate, key in [
                ('velocity_update_plus_1e-3', lambda d: d['xi'].__setitem__((slice(1, None), 8), d['xi'][1:, 8] + 1e-3), 'dynamics_relative'),
                ('contact_impulse_scaled_1.01', lambda d: d['lam'].__setitem__(slice(None), d['lam'] * 1.01), 'dynamics_relative'),
                ('actuator_force_plus_5N', lambda d: d['act'].__setitem__((slice(None), 3), d['act'][:, 3] + 5.), 'dynamics_relative')]:
            altered = {k: v.copy() for k, v in fixture.items()}
            mutate(altered)
            check = run(altered)
            outcomes[label] = dict(passed=bool(not check['passed'] and not check['checks'][key]['passed']),
                                   expected='reject', observed='accept' if check['passed'] else 'reject',
                                   measured=check['checks'][key]['value'], tolerance=check['checks'][key]['limit'])
        for label in ('nonfinite_velocity', 'reordered_coordinates', 'nonincreasing_time', 'wrong_impulse_shape'):
            altered = {k: v.copy() for k, v in fixture.items()}
            if label == 'nonfinite_velocity':
                altered['xi'][1, 0] = np.nan
            elif label == 'reordered_coordinates':
                altered['coordinate_ids'] = altered['coordinate_ids'][::-1]
            elif label == 'nonincreasing_time':
                altered['time'][1] = altered['time'][0]
            elif label == 'wrong_impulse_shape':
                altered['lam'] = altered['lam'][:, :-3]
            try:
                run(altered)
            except ValueError as error:
                outcomes[label] = dict(passed=True, expected='reject', observed='reject', reason=str(error))
            else:
                outcomes[label] = dict(passed=False, expected='reject', observed='accept')
    return dict(passed=all(v['passed'] for v in outcomes.values()), checks=outcomes,
                source=Path(npz_path).name, fixture_steps=last, fixture_stored_steps=keep_steps,
                fixture_ground_impulse_Ns=float(ground[peak]))
