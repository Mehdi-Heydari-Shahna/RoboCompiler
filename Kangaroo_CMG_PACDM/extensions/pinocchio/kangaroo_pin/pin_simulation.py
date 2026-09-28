"""Floating-base Kangaroo contact task on Pinocchio dynamics and unchanged PACDM.

State: free pelvis pose, the 140 PACDM graph coordinates (76 physical, 64
massless cut-chart coordinates), the reduced velocity ``xi`` = [pelvis twist
in the pelvis frame (6), twelve motor-slide speeds (12)], twelve force
activation states and the 1 kHz force command.  No MuJoCo module is imported.

Every step (semi-implicit Euler with PACDM projection):

1. PACDM maps the twelve motor coordinates to all graph coordinates (``N``)
   and supplies the curvature ``k`` solving ``J k = -Jdot zdot`` (centred
   directional difference of the analytic cut Jacobian, as in the Stewart
   release).
2. Pinocchio supplies the tree mass matrix (plus source armature), the bias
   ``h(q, v)`` and native frame Jacobians of the eight foot-box corners and of
   the torso COM.  With ``G = diag(I6, N_phys)``: ``Mr = G^T M G``,
   ``br = G^T (h + M [0; k])`` and the source viscous damping
   ``Dr = G^T D G``, which is treated implicitly.
3. Rigid unilateral Coulomb contact at the eight corners is solved by
   Pinocchio's native ``PGSContactSolver`` (de Saxce-corrected nonlinear
   complementarity problem) on the reduced Delassus operator
   ``W = Jc G (Mr + dt Dr)^-1 G^T Jc^T``.  If PGS stalls, Pinocchio's native
   ``ADMMContactSolver`` is used on the same problem; the accepted solution
   must satisfy the native cone checks.
4. Motor lengths advance with the new speeds, PACDM re-assembles every
   passive coordinate, and the pelvis moves on SE(3) with Pinocchio's
   exponential map.  No state is ever set from the reference.
5. Every PACDM-accepted state is refined by the accepted v22
   ``reconstructed_model.polish`` (all 144 cut rows below 5e-13, ranks and
   joint ranges rechecked).  The 24 cuts carry 16 redundant physical rows;
   at the 1e-9 PACDM corrector tolerance the tangent map deviates from the
   exact tangent space by ~1e-8, which lets the large internal loop forces
   of stance leak into the reduced equations (~1e-5 m/s^2).  The polish is
   the supplied package's own remedy for this redundancy.

The drive model, 1 kHz command schedule, feedforward gating, disturbance and
initial release reuse the accepted v22 definitions (``contact_task.DEFAULTS``
and ``external_push``) so the task matches the supplied MuJoCo benchmark.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline

from .legacy import load, SOURCE
from .pin_backend import FloatingPinBackend

ROOT = Path(__file__).resolve().parents[1]
FOOT_NAMES = ['left_ankle_roll', 'right_ankle_roll']
ACT_RANGE_N = 5000.  # accepted command clip and MuJoCo actrange, all ports
SAMPLE_PERIOD_S = .005
PIN_SETTINGS = dict(
    integrator='semi-implicit Euler on (pelvis twist, motor speeds); implicit source viscous damping; '
               'PACDM passive assembly every step; pelvis on SE(3) via pin.exp6',
    contact_model='rigid unilateral Coulomb contact at the 8 source foot-box bottom corners, plane z = 0',
    contact_solver='Pinocchio PGSContactSolver, fallback Pinocchio ADMMContactSolver',
    pgs_absolute_precision=1e-11, pgs_relative_precision=1e-13, pgs_max_iterations=3000,
    admm_proximal=1e-10, admm_absolute_precision=1e-12, admm_relative_precision=1e-14,
    admm_max_iterations=50000, ncp_acceptance=1e-8,
    gap_term='sigma_n = v_n(+) + gap / dt (no penetration after the step, full correction of drift)',
    contact_detection_gap_m=1e-6,
    jdot_probe='1e-5 / max(1, |zdot|), centred directional difference of the analytic PACDM Jacobian',
    passive_predictor='N dl + 0.5 dt^2 k, then unchanged PACDM correct (8 iterations), acquire fallback',
    corrector_max_iterations=8,
    closure_polish='accepted v22 reconstructed_model.polish after every PACDM acceptance '
                   '(all 144 rows < 5e-13; PACDM ranks and joint ranges rechecked)')


CODE_FILES = ('kangaroo_pin/legacy.py', 'kangaroo_pin/pin_backend.py', 'kangaroo_pin/pin_simulation.py',
              'original_v22/pacdm.py', 'original_v22/reconstructed_model.py',
              'original_v22/whole_body_dynamics.py', 'original_v22/contact_task.py',
              'original_v22/contact_reference.py', 'original_v22/mechanism_backend.py',
              'original_v22/source_dynamics.py', 'original_v22/source_bias.py',
              'original_v22/constraint_solvers.py', 'original_v22/import_full_model.py',
              'original_v22/data/whole_body_cmg.json', 'original_v22/data/contact_reference.npz')


def code_sha256():
    """SHA-256 of every file that determines a simulated trajectory."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}


def make_graph(accepted, cmg):
    """Unchanged accepted CutGraph with an exact residual cache.

    The cache returns the identical arrays for a byte-identical augmented
    configuration (no defect).  It removes repeated evaluations inside one
    step; it never alters an evaluated value.
    """
    class CachedCutGraph(accepted.CutGraph):
        def __init__(self, c, seed):
            super().__init__(c, seed)
            self._key = None
            self._value = None
            self.evaluations = 0
            self.cache_hits = 0

        def residual(self, q, defects=None):
            if defects is not None:
                self.evaluations += 1
                return super().residual(q, defects)
            q = np.asarray(q, dtype=float)
            key = q.tobytes()
            if key == self._key:
                self.cache_hits += 1
                return self._value
            self.evaluations += 1
            value = super().residual(q)
            for item in value:
                item.setflags(write=False)
            self._key, self._value = key, value
            return value

    return CachedCutGraph(cmg, np.array(cmg['initial_seed']))


def corner_frames(accepted):
    """The eight bottom corners of the two source foot collision boxes."""
    corners = accepted.foot_corners({name: np.eye(4) for name in FOOT_NAMES})
    frames = {}
    for i, (body, local, _) in enumerate(corners):
        T = np.eye(4)
        T[:3, 3] = local
        frames[f'corner_{i}'] = (body, T)
    return frames


def ncp_metrics(lam, sigma, mu):
    """Native cone residuals of a stacked (t1, t2, n) impulse/velocity pair."""
    cone = pin.CoulombFrictionCone(float(mu))
    dual_cone = cone.dual()
    primal = dual = comp = 0.
    for i in range(len(lam) // 3):
        l = lam[3 * i:3 * i + 3]
        s = sigma[3 * i:3 * i + 3]
        corrected = s + np.array([0., 0., mu * np.hypot(s[0], s[1])])
        primal = max(primal, float(np.linalg.norm(l - cone.project(l))))
        dual = max(dual, float(np.linalg.norm(corrected - dual_cone.project(corrected))))
        comp = max(comp, abs(float(l @ corrected)))
    return primal, dual, comp


class ContactSolver:
    """Native Pinocchio NCP solve with PGS first and ADMM as fallback."""

    def __init__(self, size, mu):
        self.mu = float(mu)
        self.cones = pin.StdVec_CoulombFrictionCone()
        for _ in range(size // 3):
            self.cones.append(pin.CoulombFrictionCone(self.mu))
        s = PIN_SETTINGS
        self.pgs = pin.PGSContactSolver(size)
        self.pgs.setAbsolutePrecision(s['pgs_absolute_precision'])
        self.pgs.setRelativePrecision(s['pgs_relative_precision'])
        self.pgs.setMaxIterations(s['pgs_max_iterations'])
        self.admm = pin.ADMMContactSolver(size, s['admm_proximal'])
        self.admm.setAbsolutePrecision(s['admm_absolute_precision'])
        self.admm.setRelativePrecision(s['admm_relative_precision'])
        self.admm.setMaxIterations(s['admm_max_iterations'])

    def solve(self, W, g, warm):
        x = warm.copy()
        self.pgs.solve(W, g, self.cones, x)
        iterations = self.pgs.getIterationCount()
        metric = ncp_metrics(x, W @ x + g, self.mu)
        used = 0
        if iterations >= PIN_SETTINGS['pgs_max_iterations'] or max(metric) > PIN_SETTINGS['ncp_acceptance']:
            self.admm.solve(pin.DelassusOperatorDense(W), g, self.cones, np.zeros(len(g)), x.copy())
            y = np.asarray(self.admm.getPrimalSolution()).copy()
            other = ncp_metrics(y, W @ y + g, self.mu)
            if max(other) < max(metric):
                x, metric, used = y, other, 1
                iterations = self.admm.getIterationCount()
        if not np.all(np.isfinite(x)) or max(metric) > PIN_SETTINGS['ncp_acceptance']:
            raise FloatingPointError(f'Contact NCP not solved to acceptance: {metric}')
        return x, used, iterations, metric


class KangarooPlant:
    """Model pieces shared by all cases (CMG, backend, PACDM graph, drives)."""

    def __init__(self, cmg=None):
        self.A = load()
        self.cmg = (json.loads((SOURCE / 'data/whole_body_cmg.json').read_text())
                    if cmg is None else cmg)
        c = self.cmg
        self.frames = corner_frames(self.A)
        torso = next(b for b in c['bodies'] if b['id'] == 'torso')
        T = np.eye(4)
        T[:3, 3] = torso['com_m']
        extra = dict(self.frames)
        extra['torso_com'] = ('torso', T)
        self.corner_names = list(self.frames)
        self.backend = FloatingPinBackend(c, extra_frames=extra)
        self.graph = make_graph(self.A, c)
        self.solver = self.A.PACDM(self.graph)
        self.ids = list(c['coordinate_ids'])
        self.nt = len(self.ids)
        self.active = np.asarray(self.graph.active, dtype=int)
        self.passive = np.asarray(self.graph.passive, dtype=int)
        if list(np.asarray(self.ids)[self.active]) != [a['joint'] for a in c['actuators']]:
            raise ValueError('PACDM independent coordinates must be the actuated motor slides')
        self.na = len(self.active)
        self.nv = 6 + self.nt
        self.S = np.zeros((self.nv, self.na))
        self.S[6 + self.active, np.arange(self.na)] = 1.
        self.armature = np.r_[np.zeros(6), [c['armature'][j] for j in self.ids]]
        self.damping = np.r_[np.zeros(6), [c['joint_dissipation'][j]['damping'] for j in self.ids]]
        joints = {j['id']: j for j in c['joints']}
        self.lower = np.array([joints[j]['limits']['lower'] for j in self.ids])
        self.upper = np.array([joints[j]['limits']['upper'] for j in self.ids])
        self.slide = np.array([joints[j]['type'] == 'prismatic' for j in self.ids])
        self.force_bounds = np.array([a['force_bounds_N'] for a in c['actuators']])
        self.mass_total = self.backend.total_mass
        self.reference_path = SOURCE / 'data/contact_reference.npz'

    def differential(self, z, ld):
        """PACDM tangent map, info, Jacobians and curvature at augmented z, motor speed ld."""
        N, info = self.solver.mapping(z)
        if not info['success']:
            raise RuntimeError(f'PACDM tangent/rank failed: {info}')
        _, J, _ = self.graph.residual(z)
        zd = N @ ld
        h = 1e-5 / max(1., float(np.linalg.norm(zd)))
        Jdot = (self.graph.residual(z + h * zd)[1] - self.graph.residual(z - h * zd)[1]) / (2 * h)
        rows = np.asarray(info['rows'])
        k = np.zeros(self.graph.n)
        k[self.passive] = -np.linalg.solve(J[np.ix_(rows, self.passive)], (Jdot @ zd)[rows])
        return N, info, J, Jdot, zd, k

    def configuration(self, base, z):
        return np.r_[base, z[:self.nt]]

    def lift_velocity(self, N, xi):
        G = np.zeros((self.nv, 6 + self.na))
        G[:6, :6] = np.eye(6)
        G[6:, 6:] = N[:self.nt]
        return G, G @ xi


def load_reference(path=None):
    path = SOURCE / 'data/contact_reference.npz' if path is None else Path(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class Controller:
    """Accepted v22 drive command: feedforward + motor PD, 1 kHz, clip, slew."""

    def __init__(self, reference, cfg, na):
        self.us = CubicSpline(reference['t'], reference['u'], axis=0)
        self.ff = CubicSpline(reference['t'], reference['force'], axis=0)
        self.cfg = cfg
        self.command = np.zeros(na)
        self.slew = cfg['command_slew_N_s'] * cfg['control_period_s']

    def update(self, t, l, ld, contact_seen, passive):
        cfg = self.cfg
        raw = ((self.ff(t) if contact_seen else np.zeros_like(l))
               + cfg['kp_N_m'] * (self.us(t) - l) + cfg['kd_N_s_m'] * (self.us(t, 1) - ld))
        if passive:
            raw = np.zeros_like(raw)
        saturated = bool(np.any(np.abs(raw) > ACT_RANGE_N))
        bounded = np.clip(raw, -ACT_RANGE_N, ACT_RANGE_N)
        self.command = self.command + np.clip(bounded - self.command, -self.slew, self.slew)
        return self.command.copy(), saturated


def run_case(name='landing_nominal', dt=.001, config=None, duration=None, no_contact=False,
             passive=False, output=None, reference_path=None, progress=True, plant=None,
             stop_on_bound=False):
    """Integrate one case with PACDM closures; write ``<output>/<name>.npz/json``."""
    plant = KangarooPlant() if plant is None else plant
    A = plant.A
    cfg = {**A.DEFAULTS, **(config or {})}
    duration = A.DURATION if duration is None else float(duration)
    output = ROOT / 'results' if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    steps = round(duration / dt)
    if abs(steps * dt - duration) > 1e-12:
        raise ValueError('duration must be an integer number of steps')
    control_stride = round(cfg['control_period_s'] / dt)
    sample_stride = round(SAMPLE_PERIOD_S / dt)
    if (control_stride < 1 or abs(control_stride * dt - cfg['control_period_s']) > 1e-12
            or abs(sample_stride * dt - SAMPLE_PERIOD_S) > 1e-12):
        raise ValueError('control and sample periods must be integer numbers of steps')
    reference = load_reference(reference_path)
    controller = Controller(reference, cfg, plant.na)
    backend, graph, solver = plant.backend, plant.graph, plant.solver
    contact = ContactSolver(3 * len(plant.corner_names), cfg['friction'])
    evaluations0, hits0 = graph.evaluations, graph.cache_hits

    z0 = np.array(reference['qaug'][0], dtype=float)
    z = A.polish(graph, z0)
    polish_shift = float(np.max(np.abs(z - z0)))
    base = np.r_[reference['base'][0] + [0., 0., cfg['drop_height_m']], 0., 0., 0., 1.]
    xi = np.zeros(6 + plant.na)
    act = np.zeros(plant.na)
    command = np.zeros(plant.na)
    Lam = np.zeros(3 * len(plant.corner_names))
    filt = 1. - np.exp(-dt / cfg['actuator_time_constant_s'])
    contact_seen = False
    fallback = 0
    corrector_iterations = 0
    stop_reason = None
    log = {k: [] for k in [
        'time', 'base', 'motor', 'xi', 'act', 'command', 'contact_seen', 'lam', 'gap', 'push',
        'kinetic', 'potential', 'armature_kinetic', 'work_step', 'pacdm_closure', 'tangent',
        'acceleration_closure', 'rcond', 'rank', 'reduced_equation', 'contact_solver',
        'contact_iterations', 'ncp', 'tilt', 'com', 'linear_momentum', 'angular_momentum',
        'foot_position', 'torso_com', 'saturated', 'contact_moment', 'polish_shift']}
    samples = {k: [] for k in ['step', 'z', 'reason']}
    xi_next = xi.copy()
    loaded_before = np.zeros(len(plant.corner_names), dtype=bool)
    for k in range(steps + 1):
        t = k * dt
        l = z[plant.active].copy()
        ld = xi[6:].copy()
        saturated = False
        if k % control_stride == 0:
            command, saturated = controller.update(t, l, ld, contact_seen, passive)
        force = np.clip(act, plant.force_bounds[:, 0], plant.force_bounds[:, 1])
        # PACDM differential map and Pinocchio tree terms at the pre-step state.
        N, info, J, Jdot, zd, kz = plant.differential(z, ld)
        q = plant.configuration(base, z)
        G, nu = plant.lift_velocity(N, xi)
        kfull = np.r_[np.zeros(6), kz[:plant.nt]]
        M = backend.mass(q) + np.diag(plant.armature)
        h = backend.bias(q, nu)
        names = plant.corner_names + ['torso_com'] + FOOT_NAMES
        frames = backend.frame_jacobians(q, names)
        push = np.asarray(A.external_push(t, cfg), dtype=float)
        Jpush = frames['torso_com'][1][:3]
        Mr = G.T @ M @ G
        br = G.T @ (h + M @ kfull)
        Dr = G.T @ (plant.damping[:, None] * G)
        tau_r = G.T @ (plant.S @ force + Jpush.T @ push)
        Aimp = Mr + dt * Dr
        xi_free = np.linalg.solve(Aimp, Mr @ xi + dt * (tau_r - br))
        gaps = np.array([frames[n][0][2, 3] for n in plant.corner_names])
        Jc = np.vstack([frames[n][1][:3] for n in plant.corner_names]) @ G
        if no_contact:
            Lam = np.zeros_like(Lam)
            xi_next = xi_free
            used, iterations, metric = -1, 0, (0., 0., 0.)
        else:
            Ainv_JcT = np.linalg.solve(Aimp, Jc.T)
            W = Jc @ Ainv_JcT
            W = .5 * (W + W.T)
            g = Jc @ xi_free
            g[2::3] += gaps / dt
            Lam, used, iterations, metric = contact.solve(W, g, Lam)
            xi_next = xi_free + Ainv_JcT @ Lam
        # Energy terms with the step's midpoint velocity (exact for constant M).
        energy = backend.energy(q, nu)
        ke_arm = .5 * float(np.sum(plant.armature * nu * nu))
        xm = .5 * (xi + xi_next)
        work = np.array([dt * float(force @ xm[6:]), -dt * float(xm @ (Dr @ xi_next)), 0.,
                         float(Lam @ (Jc @ xm)), 0., dt * float(push @ (Jpush @ (G @ xm)))])
        reduced_eq = float(np.max(np.abs(Mr @ (xi_next - xi) / dt + br + Dr @ xi_next
                                          - tau_r - Jc.T @ Lam / dt)))
        zdd = N @ ((xi_next[6:] - xi[6:]) / dt) + kz
        acc_closure = float(np.max(np.abs(J @ zdd + Jdot @ zd)))
        lin, ang_c = backend.centroidal_momentum(q, nu)
        com = backend.center_of_mass(q)
        R = pin.Quaternion(base[3:7]).toRotationMatrix()
        log['time'].append(t)
        log['base'].append(base.copy())
        log['motor'].append(l)
        log['xi'].append(xi.copy())
        log['act'].append(act.copy())
        log['command'].append(command.copy())
        log['contact_seen'].append(contact_seen)
        log['lam'].append(Lam.copy())
        log['gap'].append(gaps)
        log['push'].append(push)
        log['kinetic'].append(energy['kinetic_J'])
        log['potential'].append(energy['potential_J'])
        log['armature_kinetic'].append(ke_arm)
        log['work_step'].append(work)
        log['pacdm_closure'].append(info['residual_inf'])
        log['tangent'].append(info['tangent_residual'])
        log['acceleration_closure'].append(acc_closure)
        log['rcond'].append(info['rcond'])
        log['rank'].append([info['rank_full'], info['rank_passive']])
        log['reduced_equation'].append(reduced_eq)
        log['contact_solver'].append(used)
        log['contact_iterations'].append(iterations)
        log['ncp'].append(metric)
        log['tilt'].append(float(np.arccos(np.clip(R[2, 2], -1., 1.))))
        log['com'].append(com)
        log['linear_momentum'].append(lin)
        log['angular_momentum'].append(ang_c + np.cross(com, lin))  # about the world origin
        log['foot_position'].append(np.array([frames[f][0][:3, 3] for f in FOOT_NAMES]))
        log['torso_com'].append(frames['torso_com'][0][:3, 3].copy())
        log['saturated'].append(saturated)
        corners = np.array([frames[n][0][:3, 3] for n in plant.corner_names])
        log['contact_moment'].append(np.cross(corners, Lam.reshape(-1, 3)).sum(axis=0))
        log['polish_shift'].append(polish_shift)
        loaded = Lam[2::3] > 1e-12
        reasons = []
        if k % sample_stride == 0:
            reasons.append('sample')
        if used == 1:
            reasons.append('admm')
        if np.any(loaded != loaded_before):
            reasons.append('active_set_change')
        loaded_before = loaded
        if reasons:
            samples['step'].append(k)
            samples['z'].append(z.copy())
            samples['reason'].append('+'.join(reasons))
        if progress and k % max(1, round(1. / dt)) == 0:
            normal = Lam[2::3].sum() / dt
            print(f'{name}: t={t:5.2f}s z={base[2]:.4f} tilt={np.degrees(log["tilt"][-1]):.3f}deg '
                  f'|F|max={np.max(np.abs(force)):7.1f}N ground={normal:7.1f}N', flush=True)
        if k == steps:
            break
        # Advance: contact flag for the next command, pelvis on SE(3), motors, PACDM.
        contact_seen = bool(np.any(Lam[2::3] > 0.) or np.any(gaps <= PIN_SETTINGS['contact_detection_gap_m']))
        base_next = pin.SE3ToXYZQUAT(pin.XYZQUATToSE3(base) * pin.exp6(pin.Motion(xi_next[:6] * dt)))
        l_next = l + dt * xi_next[6:]
        predicted = z[plant.passive] + N[plant.passive] @ (l_next - l) + .5 * dt * dt * kz[plant.passive]
        rows = np.asarray(info['rows'])
        candidate, correction = solver.correct(l_next, predicted, None, rows,
                                               maxiter=PIN_SETTINGS['corrector_max_iterations'])
        corrector_iterations += int(correction.get('iterations', 0))
        if not correction['success']:
            fallback += 1
            candidate, correction = solver.acquire(l_next, z)
        if not correction['success']:
            if stop_on_bound:
                stop_reason = f'PACDM assembly stopped at {t + dt:.4f} s: {correction.get("message")}'
                break
            raise RuntimeError(f'{name}: PACDM assembly failed at {t}: {correction}')
        try:
            polished = A.polish(graph, candidate)
        except ValueError as error:
            if stop_on_bound:
                stop_reason = f'PACDM assembly stopped at {t + dt:.4f} s: polish failed ({error})'
                break
            raise RuntimeError(f'{name}: closure polish failed at {t}: {error}') from error
        polish_shift = float(np.max(np.abs(polished - candidate)))
        z = polished
        base = base_next
        xi = xi_next
        act = np.clip(act + (np.clip(command, plant.force_bounds[:, 0], plant.force_bounds[:, 1]) - act) * filt,
                      -ACT_RANGE_N, ACT_RANGE_N)
    arrays = {key: np.asarray(value) for key, value in log.items()}
    arrays['stored_step'] = np.asarray(samples['step'], dtype=int)
    arrays['stored_z'] = np.asarray(samples['z'])
    arrays['stored_reason'] = np.asarray(samples['reason'])
    arrays['final_z'] = z.copy()
    arrays['final_xi_next'] = xi_next.copy()
    energy = arrays['kinetic'] + arrays['potential'] + arrays['armature_kinetic']
    work_total = np.vstack([np.zeros(6), np.cumsum(arrays['work_step'][:-1], axis=0)])
    arrays['energy'] = energy
    arrays['work'] = work_total
    arrays['ledger'] = energy - energy[0] - work_total.sum(axis=1)
    arrays['coordinate_ids'] = np.asarray(plant.ids)
    arrays['actuator_ids'] = np.asarray([a['id'] for a in plant.cmg['actuators']])
    arrays['corner_names'] = np.asarray(plant.corner_names)
    arrays['timestep_s'] = np.asarray(dt)
    arrays['dynamics_backend'] = np.asarray(
        f'Pinocchio {pin.__version__} + unchanged PACDM + native PGS/ADMM contact')
    summary = dict(name=name, timestep_s=dt, duration_s=duration, steps=steps,
                   completed_steps=len(arrays['time']) - 1, configuration=cfg,
                   pinocchio_settings=PIN_SETTINGS, no_contact=bool(no_contact),
                   passive_control=bool(passive), open_tree=False, stop_reason=stop_reason,
                   fallback_count=fallback, corrector_iterations=corrector_iterations,
                   graph_evaluations=graph.evaluations - evaluations0,
                   graph_cache_hits=graph.cache_hits - hits0,
                   pinocchio_version=pin.__version__, numpy_version=np.__version__,
                   code_sha256=code_sha256(), elapsed_s=time.perf_counter() - started)
    summary.update(case_metrics(plant, arrays, reference, cfg, dt, duration))
    np.savez_compressed(output / f'{name}.npz', **arrays)
    (output / f'{name}.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary, arrays


def case_metrics(plant, a, reference, cfg, dt, duration):
    """Task metrics defined as in the accepted v22 contact benchmark."""
    time_ = a['time']
    force = np.clip(a['act'], plant.force_bounds[:, 0], plant.force_bounds[:, 1])
    lam = a['lam']
    normal_corner = lam[:, 2::3] / dt
    ground = lam.reshape(len(time_), -1, 3).sum(axis=1) / dt
    margins = np.minimum(a['motor'] - plant.lower[plant.active], plant.upper[plant.active] - a['motor'])
    # Every stored assembled state (5 ms and events): all 76 physical coordinates.
    physical = a['stored_z'][:, :plant.nt]
    stored_margin = np.minimum(physical - plant.lower, plant.upper - physical)
    samples = np.arange(0, len(time_), round(SAMPLE_PERIOD_S / dt))
    ts = time_[samples]
    base = a['base'][samples]
    ss = (ts > 1) & (ts < 6.5)
    tail = time_ > duration - .5
    planned = CubicSpline(reference['t'], reference['base'], axis=0)(time_[-1])
    R = np.array([pin.Quaternion(b[3:7]).toRotationMatrix() for b in a['base']])
    world_linear = np.einsum('nij,nj->ni', R, a['xi'][:, :3])
    tail_speed = np.linalg.norm(world_linear[tail], axis=1) if np.any(tail) else np.zeros(1)
    feet = a['foot_position']
    after = time_ > .8
    if np.any(after):
        first = int(np.argmax(after))
        drift = np.max(np.linalg.norm(feet[after, :, :2] - feet[first, :, :2], axis=2), axis=0)
    else:
        drift = np.zeros(2)
    touch = np.nonzero(ground[:, 2] > 1.)[0]
    # Port power with the step midpoint speed, exactly the motor work channel.
    speeds = a['xi'][:, 6:]
    nxt = np.vstack([speeds[1:], a['final_xi_next'][None, 6:]])
    port_power = force * .5 * (speeds + nxt)
    port_work = dt * port_power[:-1]
    friction_ratio = 0.
    loaded = normal_corner > 1e-6
    if np.any(loaded):
        tang = np.hypot(lam[:, 0::3], lam[:, 1::3]) / dt
        friction_ratio = float(np.max(tang[loaded] / normal_corner[loaded]))
    rate = float(np.max(np.abs(np.diff(force, axis=0))) / dt) if len(time_) > 1 else 0.
    work_final = a['work'][-1]
    us = CubicSpline(reference['t'], reference['u'], axis=0)
    return dict(
        mass_kg=plant.mass_total, body_count=len(plant.cmg['bodies']), motor_count=plant.na,
        touchdown_s=float(time_[touch[0]]) if len(touch) else None,
        maximum_pacdm_closure=float(a['pacdm_closure'].max()),
        maximum_polish_shift=float(a['polish_shift'].max()),
        maximum_tangent_residual=float(a['tangent'].max()),
        maximum_acceleration_closure=float(a['acceleration_closure'].max()),
        maximum_reduced_equation_residual=float(a['reduced_equation'].max()),
        minimum_rank_full=int(a['rank'][:, 0].min()), minimum_rank_passive=int(a['rank'][:, 1].min()),
        minimum_mapping_rcond=float(a['rcond'].min()),
        minimum_slide_margin_m=float(min(margins.min(), stored_margin[:, plant.slide].min())),
        minimum_hinge_margin_rad=float(stored_margin[:, ~plant.slide].min()),
        maximum_motor_force_N=float(np.max(np.abs(force))),
        maximum_motor_error_m=float(np.max(np.abs(a['motor'] - us(time_)))),
        maximum_penetration_m=float(max(0., -float(a['gap'].min()))),
        maximum_tilt_deg=float(np.degrees(a['tilt'].max())), final_tilt_deg=float(np.degrees(a['tilt'][-1])),
        maximum_foot_drift_after_landing_m=[float(x) for x in drift],
        maximum_ground_normal_N=float(ground[:, 2].max()),
        mean_final_ground_normal_N=float(ground[tail, 2].mean()) if np.any(tail) else 0.,
        achieved_crouch_m=float(np.ptp(base[ss, 2])) if np.any(ss) else 0.,
        achieved_lateral_excursion_m=float(np.ptp(base[ss, 1])) if np.any(ss) else 0.,
        maximum_final_base_speed_m_s=float(tail_speed.max()),
        final_base_position_m=[float(x) for x in a['base'][-1, :3]],
        final_position_error_m=float(np.linalg.norm(a['base'][-1, :3] - planned)),
        final_motor_work_J=port_work.sum(axis=0).tolist(),
        positive_motor_work_J=np.maximum(port_work, 0.).sum(axis=0).tolist(),
        negative_motor_work_J=np.minimum(port_work, 0.).sum(axis=0).tolist(),
        work_J=dict(zip(['motor', 'passive', 'loop', 'contact', 'limit', 'disturbance'],
                        [float(x) for x in work_final])),
        energy_change_J=float(a['energy'][-1] - a['energy'][0]),
        maximum_energy_ledger_error_J=float(np.max(np.abs(a['ledger']))),
        final_energy_ledger_error_J=float(a['ledger'][-1]),
        maximum_contact_friction_ratio=friction_ratio,
        minimum_contact_normal_N=float(normal_corner.min()),
        maximum_ncp_primal=float(a['ncp'][:, 0].max()), maximum_ncp_dual=float(a['ncp'][:, 1].max()),
        maximum_ncp_complementarity=float(a['ncp'][:, 2].max()),
        admm_fallback_steps=int(np.sum(a['contact_solver'] == 1)),
        maximum_contact_iterations=int(a['contact_iterations'].max()),
        peak_force_rate_N_s=rate, saturated_control_updates=int(np.sum(a['saturated'])),
        final_base_z_m=float(a['base'][-1, 2]))


def run_open_tree(name='negative_no_loops_contact', dt=.001, config=None, duration=.3,
                  output=None, reference_path=None, progress=True, plant=None):
    """Negative control: the same drives and rigid contact on the open tree.

    All 24 loop cuts are removed (no PACDM projection, no native loop
    constraint).  The 82-coordinate tree is integrated with the same
    semi-implicit scheme, implicit viscous damping and native PGS/ADMM contact.
    Native Pinocchio loop gaps are measured at every step; they must open.
    """
    from .native_oracle import NativeLoopOracle
    plant = KangarooPlant() if plant is None else plant
    A = plant.A
    cfg = {**A.DEFAULTS, **(config or {})}
    output = ROOT / 'results' if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    steps = round(duration / dt)
    control_stride = round(cfg['control_period_s'] / dt)
    reference = load_reference(reference_path)
    controller = Controller(reference, cfg, plant.na)
    backend = plant.backend
    oracle = NativeLoopOracle(backend, plant.cmg, plant.armature)
    contact = ContactSolver(3 * len(plant.corner_names), cfg['friction'])
    z0 = A.polish(plant.graph, np.array(reference['qaug'][0], dtype=float))
    q = plant.configuration(np.r_[reference['base'][0] + [0., 0., cfg['drop_height_m']], 0., 0., 0., 1.], z0)
    v = np.zeros(plant.nv)
    act = np.zeros(plant.na)
    command = np.zeros(plant.na)
    Lam = np.zeros(3 * len(plant.corner_names))
    filt = 1. - np.exp(-dt / cfg['actuator_time_constant_s'])
    contact_seen = False
    log = {k: [] for k in ['time', 'base', 'point_gap', 'universal_residual', 'act', 'lam', 'force']}
    for k in range(steps + 1):
        t = k * dt
        l = q[7 + plant.active].copy()
        ld = v[6 + plant.active].copy()
        if k % control_stride == 0:
            command, _ = controller.update(t, l, ld, contact_seen, False)
        force = np.clip(act, plant.force_bounds[:, 0], plant.force_bounds[:, 1])
        M = backend.mass(q) + np.diag(plant.armature)
        h = backend.bias(q, v)
        frames = backend.frame_jacobians(q, plant.corner_names + ['torso_com'])
        push = np.asarray(A.external_push(t, cfg), dtype=float)
        tau = plant.S @ force + frames['torso_com'][1][:3].T @ push
        Aimp = M + dt * np.diag(plant.damping)
        v_free = np.linalg.solve(Aimp, M @ v + dt * (tau - h))
        gaps = np.array([frames[n][0][2, 3] for n in plant.corner_names])
        Jc = np.vstack([frames[n][1][:3] for n in plant.corner_names])
        Ainv_JcT = np.linalg.solve(Aimp, Jc.T)
        W = Jc @ Ainv_JcT
        W = .5 * (W + W.T)
        g = Jc @ v_free
        g[2::3] += gaps / dt
        Lam, _, _, _ = contact.solve(W, g, Lam)
        v_next = v_free + Ainv_JcT @ Lam
        geo = oracle.geometry(q)
        point = np.max(np.linalg.norm(geo['point_residual'].reshape(-1, 3), axis=1))
        log['time'].append(t)
        log['base'].append(q[:7].copy())
        log['point_gap'].append(point)
        log['universal_residual'].append(float(np.max(np.abs(geo['angular_residual']))))
        log['act'].append(act.copy())
        log['lam'].append(Lam.copy())
        log['force'].append(force.copy())
        if progress and k % max(1, round(.1 / dt)) == 0:
            print(f'{name}: t={t:.2f}s z={q[2]:.4f} loop gap={point * 1e3:.3f} mm', flush=True)
        if k == steps:
            break
        contact_seen = bool(np.any(Lam[2::3] > 0.) or np.any(gaps <= PIN_SETTINGS['contact_detection_gap_m']))
        q = backend.integrate(q, v_next, dt)
        v = v_next
        act = np.clip(act + (np.clip(command, plant.force_bounds[:, 0], plant.force_bounds[:, 1]) - act) * filt,
                      -ACT_RANGE_N, ACT_RANGE_N)
    arrays = {key: np.asarray(value) for key, value in log.items()}
    summary = dict(name=name, timestep_s=dt, duration_s=duration, steps=steps, configuration=cfg,
                   open_tree=True, no_contact=False, passive_control=False,
                   maximum_loop_gap_m=float(arrays['point_gap'].max()),
                   maximum_universal_residual=float(arrays['universal_residual'].max()),
                   final_base_position_m=[float(x) for x in arrays['base'][-1, :3]],
                   maximum_motor_force_N=float(np.max(np.abs(arrays['force']))),
                   maximum_ground_normal_N=float(np.max(arrays['lam'][:, 2::3].sum(axis=1)) / dt),
                   pinocchio_version=pin.__version__, numpy_version=np.__version__,
                   code_sha256=code_sha256(), elapsed_s=time.perf_counter() - started)
    np.savez_compressed(output / f'{name}.npz', **arrays)
    (output / f'{name}.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary, arrays
