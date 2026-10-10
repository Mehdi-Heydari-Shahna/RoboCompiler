"""Development harness: run the Kangaroo native loop on NVIDIA ovphysx (CPU PhysX).

This mirrors kangaroo_isaac/native.py step for step, but replaces the Isaac
Sim World/tensor calls with the standalone ovphysx runtime (same Omni PhysX USD
parsing family, newer PhysX SDK line than Isaac Sim 6.1). It loads a USD scene
authored by the package (or exported by Isaac) and uses the package's own
Controller, ContactGate, Recorder and gate evaluator unchanged.

Checkpoint/resume exists only because the development container can restart.
A resumed run restores the articulation state (float32 joint and root state,
exactly as read back) and all controller/recorder Python state, but PhysX
contact caches and friction anchors are rebuilt on the first resumed step. The
number of resumes is recorded in result.json.

It is a development/diagnostic tool, not Isaac Sim evidence.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='package root that contains kangaroo_isaac/')
    p.add_argument('--usd', default=None, help='existing USD; omit with --build')
    p.add_argument('--build', action='store_true', help='author the USD with the package builder (shim) from the case config')
    p.add_argument('--case', default='nominal')
    p.add_argument('--solver-profile', default=None)
    p.add_argument('--duration', type=float, default=None, help='diagnostic shortening only')
    p.add_argument('--out', required=True)
    p.add_argument('--print-every', type=float, default=0.5)
    p.add_argument('--config-json', default=None, help='JSON dict of Config overrides (diagnostics)')
    p.add_argument('--trace-window', type=float, nargs=2, default=None, help='record every physics step in [a,b)')
    p.add_argument('--checkpoint-every', type=float, default=0.25, help='simulated seconds between checkpoints; 0 disables')
    args = p.parse_args(argv)
    sys.path.insert(0, str(Path(args.src).resolve()))
    from scipy.spatial.transform import Rotation
    from kangaroo_isaac.model import Model, FOOT_NAMES, finite, poses_from_xyzw
    from kangaroo_isaac.control import Reference, Controller, case_config, external_push
    from kangaroo_isaac.observations import Recorder
    from kangaroo_isaac.contact_gate import ContactGate, ControllerTrace
    from kangaroo_isaac.gates import assess
    from kangaroo_isaac.scene_paths import BODIES
    from kangaroo_isaac.io_utils import name_map, write_json
    from kangaroo_isaac.runtime_bridge import fill_disturbance_buffers
    import ovphysx, ovstage
    from ovphysx import PhysX
    from ovphysx.types import TensorType as TT

    kw = {} if args.solver_profile is None else {'solver_profile': args.solver_profile}
    cfg = case_config(args.case, **kw)
    if args.config_json:
        cfg = replace(cfg, **json.loads(args.config_json)).validate()
    if args.duration:
        cfg = replace(cfg, duration_s=args.duration).validate()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        print('already complete:', out); return 0
    ckpt_path = out / 'checkpoint.pkl'
    ckpt = None
    if ckpt_path.exists():
        with ckpt_path.open('rb') as f:
            ckpt = pickle.load(f)
        if ckpt['configuration'] != asdict(cfg):
            raise SystemExit('checkpoint configuration differs; refusing to resume')
    if args.build:
        import build_usd
        args.usd = str(out / 'kangaroo_scene_built.usda')
        _, usd_audit = build_usd.build(args.src, cfg, args.usd)
        write_json(out / 'usd_audit.json', usd_audit)
    if not args.usd:
        raise SystemExit('--usd or --build is required')
    model = Model(); ref = Reference(); controller = Controller(model, ref, cfg)
    gate = ContactGate(cfg); trace = ControllerTrace(cfg)
    base = ref.data['base'][0].copy(); base[2] += cfg.drop_height_m
    P0 = model.fk(ref.data['q'][0], base, Rotation.from_rotvec(ref.data['rotvec'][0]).as_matrix())

    ovstage.population.register_usd_schemas([str(ovphysx.codeless_schema_root())])
    PhysX.set_cpu_mode(True)
    physx = PhysX()
    stage = ovstage.Stage('kangaroo')
    ovstage.population.open_usd(stage, str(Path(args.usd).resolve()), ordinal=1,
                                domains=ovstage.PopulationDomain.ALL)
    stage.advance_write_floor(ordinal=1).wait()
    physx.attach_ovstage(stage, read_ordinal=1)
    physx.wait_all()
    root = BODIES + '/' + model.c['root_body']
    import warnings
    warnings.simplefilter('ignore', DeprecationWarning)
    def bind(tt):
        b = physx.create_tensor_binding(prim_paths=[root], tensor_type=tt)
        if b.count != 1:
            raise RuntimeError('binding count')
        return b
    b_q = bind(TT.ARTICULATION_DOF_POSITION); b_qd = bind(TT.ARTICULATION_DOF_VELOCITY)
    b_eff = bind(TT.ARTICULATION_DOF_ACTUATION_FORCE)
    b_pose = bind(TT.ARTICULATION_LINK_POSE); b_vel = bind(TT.ARTICULATION_LINK_VELOCITY)
    b_wrench = bind(TT.ARTICULATION_LINK_WRENCH)
    b_root = bind(TT.ARTICULATION_ROOT_POSE); b_rootv = bind(TT.ARTICULATION_ROOT_VELOCITY)
    qm = name_map(model.ids, list(b_q.dof_names)); bm = name_map(model.names, list(b_q.body_names))
    contacts = physx.create_contact_binding(
        sensor_patterns=[BODIES + '/' + n for n in FOOT_NAMES], max_contact_data_count=64)
    order = [list(contacts.sensor_paths).index(BODIES + '/' + n) for n in FOOT_NAMES]
    detail = None
    if args.trace_window:
        detail = physx.create_contact_binding(sensor_patterns=[BODIES + '/' + n for n in FOOT_NAMES],
                                              filter_patterns=['/World/Ground', '/World/Ground'], filters_per_sensor=1,
                                              max_contact_data_count=64)
        dorder = [list(detail.sensor_paths).index(BODIES + '/' + n) for n in FOOT_NAMES]
        C = 64
        cf = np.zeros((C, 1), np.float32); cp = np.zeros((C, 3), np.float32); cn = np.zeros((C, 3), np.float32)
        cs = np.zeros((C, 1), np.float32); cc = np.zeros((2, 1), np.int32); cst = np.zeros((2, 1), np.int32)
        ff_ = np.zeros((C, 3), np.float32); fpnt = np.zeros((C, 3), np.float32); fc = np.zeros((2, 1), np.int32); fst = np.zeros((2, 1), np.int32)
    physx.warmup()
    physx.wait_all()
    if ckpt is None:
        # Initialisation only, as native.py: q0, zero velocity, root pose, zero root velocity.
        a = np.zeros((1, model.n), np.float32); a[0, qm] = ref.data['q'][0]
        b_q.write(a)
        b_qd.write(np.zeros((1, model.n), np.float32))
        rootpose = np.r_[P0[model.root, :3, 3], Rotation.from_matrix(P0[model.root, :3, :3]).as_quat()]
        b_root.write(rootpose[None].astype(np.float32))
        b_rootv.write(np.zeros((1, 6), np.float32))
    else:
        phys = ckpt['physx_state']
        b_q.write(phys['q']); b_qd.write(phys['qd']); b_root.write(phys['root_pose']); b_rootv.write(phys['root_velocity'])
    physx.update_articulations_kinematic()
    qbuf = np.zeros((1, model.n), np.float32); qdbuf = np.zeros_like(qbuf)
    posebuf = np.zeros((1, model.nb, 7), np.float32); velbuf = np.zeros((1, model.nb, 6), np.float32)
    fbuf = np.zeros((2, 3), np.float32)

    def state():
        physx.update_articulations_kinematic()
        b_pose.read(posebuf); b_vel.read(velbuf); b_q.read(qbuf); b_qd.read(qdbuf)
        xyzw = posebuf[0, bm].astype(float)
        return (poses_from_xyzw(xyzw), velbuf[0, bm].astype(float), qbuf[0, qm].astype(float),
                qdbuf[0, qm].astype(float), xyzw)

    P, V, q, qd, xyzw = state()
    nsteps = round(cfg.duration_s / cfg.dt_s); sample_stride = round(cfg.sample_period_s / cfg.dt_s)
    ckpt_stride = round(args.checkpoint_every / cfg.dt_s) if args.checkpoint_every > 0 else 0
    effort = np.zeros((1, model.n), np.float32)
    wrench = np.zeros((1, model.nb, 9), np.float32)
    bodyforce = np.zeros((1, model.nb, 3)); bodyposition = np.zeros((1, model.nb, 3))
    step_log = {k: [] for k in ('t','fn','foot_z','foot_vz','base_z','base_vz','motor_force','motor_q','motor_qd','sole_h',
                                'foot_xy','n_contacts','normal_sum','min_sep','n_anchors','friction_xy','friction_ratio')}
    recorder = Recorder(model, ref, cfg)
    if ckpt is None:
        init = {'max_initial_pose_error': float(abs(P - P0).max()),
                'max_initial_q_error': float(abs(q - ref.data['q'][0]).max()),
                'max_initial_speed': float(abs(V).max())}
        print('init', init, flush=True)
        footforce = np.zeros((2, 3))
        recorder.observe(0, P, V, q, qd, np.zeros(12), controller.command, footforce, np.zeros(3), xyzw, save=True)
        k_start = 0; wall_before = 0.; resumes = []
    else:
        init = ckpt['init']; footforce = ckpt['footforce']; k_start = ckpt['k']
        for name, value in ckpt['controller'].items():
            setattr(controller, name, value)
        trace = ckpt['trace']
        rec_state = ckpt['recorder']
        for name, value in rec_state.items():
            setattr(recorder, name, value)
        step_log = ckpt['step_log']; wall_before = ckpt['wall_s']; resumes = ckpt['resumes'] + [k_start]
        restored = {'max_abs_q_error_after_restore': float(abs(q - ckpt['q_measured']).max()),
                    'max_abs_pose_error_after_restore': float(abs(P - ckpt['P_measured']).max())}
        print('resumed at step', k_start, 't=', k_start * cfg.dt_s, restored, flush=True)
    t_start = time.perf_counter()
    next_print = (int(k_start * cfg.dt_s / args.print_every) + 1) * args.print_every

    def save_checkpoint(k_next):
        rec = {k: v for k, v in recorder.__dict__.items() if k not in ('m', 'ref', 'cfg')}
        a_q = np.zeros((1, model.n), np.float32); a_qd = np.zeros_like(a_q)
        a_root = np.zeros((1, 7), np.float32); a_rootv = np.zeros((1, 6), np.float32)
        b_q.read(a_q); b_qd.read(a_qd); b_root.read(a_root); b_rootv.read(a_rootv)
        data = {'k': k_next, 'configuration': asdict(cfg), 'init': init, 'footforce': footforce,
                'controller': {n: getattr(controller, n) for n in ('command', 'force', 'last_raw', 'saturation_updates', 'update_count')},
                'trace': trace, 'recorder': rec, 'step_log': step_log,
                'wall_s': wall_before + time.perf_counter() - t_start, 'resumes': resumes,
                'physx_state': {'q': a_q, 'qd': a_qd, 'root_pose': a_root, 'root_velocity': a_rootv},
                'q_measured': q.copy(), 'P_measured': P.copy()}
        tmp = ckpt_path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, ckpt_path)

    for k in range(k_start, nsteps):
        t = k * cfg.dt_s
        if k % controller.control_stride == 0:
            contact = gate.evaluate(model.foot_points(P), footforce)
            controller.update_command(k, q[model.active], qd[model.active], contact.selected)
            trace.record(k, contact, q[model.active], qd[model.active], controller)
        force = controller.force.astype(np.float32).astype(float)
        effort.fill(0.); effort[0, qm[model.active]] = force
        b_eff.write(effort)
        push = external_push(t, cfg)
        fill_disturbance_buffers(model, P, bm, push, bodyforce, bodyposition)
        wrench[0, :, 0:3] = bodyforce[0]; wrench[0, :, 3:6] = 0.; wrench[0, :, 6:9] = bodyposition[0]
        b_wrench.write(wrench)
        preV = V; preqd = qd
        physx.step_sync(cfg.dt_s)
        P, V, q, qd, xyzw = state()
        contacts.read_net_forces(fbuf)
        footforce = fbuf[order].astype(float)
        if args.trace_window and args.trace_window[0] <= (k+1)*cfg.dt_s < args.trace_window[1]:
            step_log['t'].append((k+1)*cfg.dt_s); step_log['fn'].append(footforce[:, 2].copy())
            step_log['foot_z'].append(P[model.feet, 2, 3].copy()); step_log['foot_vz'].append(V[model.feet, 2].copy())
            step_log['base_z'].append(P[model.root, 2, 3]); step_log['base_vz'].append(V[model.root, 2])
            step_log['motor_force'].append(force.copy()); step_log['motor_q'].append(q[model.active].copy()); step_log['motor_qd'].append(qd[model.active].copy())
            step_log['sole_h'].append(model.foot_points(P)[:, :, 2].min(axis=1) - 0.0)
            step_log['foot_xy'].append(P[model.feet, :2, 3].copy())
            detail.read_contact_data(cf, cp, cn, cs, cc, cst); detail.read_friction_data(ff_, fpnt, fc, fst)
            nc = []; ns = []; ms = []; na = []; fxy = []; fr = []
            for i in dorder:
                c0, n0 = int(cst[i, 0]), int(cc[i, 0]); a0, m0 = int(fst[i, 0]), int(fc[i, 0])
                nsum = float(cf[c0:c0+n0, 0].sum()); fsum = ff_[a0:a0+m0].sum(axis=0)
                nc.append(n0); ns.append(nsum); ms.append(float(cs[c0:c0+n0, 0].min()) if n0 else 1.)
                na.append(m0); fxy.append(fsum[:2].astype(float)); fr.append(float(np.linalg.norm(fsum)) / nsum if nsum > 1e-9 else 0.)
            step_log['n_contacts'].append(nc); step_log['normal_sum'].append(ns); step_log['min_sep'].append(ms)
            step_log['n_anchors'].append(na); step_log['friction_xy'].append(fxy); step_log['friction_ratio'].append(fr)
        recorder.integrate_step(force, preqd, qd, push, preV, V)
        controller.advance_filter()
        recorder.observe(k + 1, P, V, q, qd, force, controller.command, footforce, push, xyzw,
                         save=((k + 1) % sample_stride == 0 or k + 1 == nsteps))
        if not (cfg.no_contact or cfg.no_loops or cfg.passive):
            if P[model.root, 2, 3] < .1 or P[model.root, 2, 2] < np.cos(np.radians(45.)):
                raise RuntimeError('Safety stop: the native robot fell')
            if recorder.metrics['maximum_loop_gap_m'] > .05:
                raise RuntimeError('Safety stop: native closure gap exceeded 50 mm')
        if (k + 1) * cfg.dt_s >= next_print - 1e-12:
            next_print += args.print_every
            el = wall_before + time.perf_counter() - t_start
            print(f'{args.case}: t={(k+1)*cfg.dt_s:.3f}s wall={el:.0f}s drift={np.round(recorder.foot_drift*1000,2)}mm '
                  f'loop={recorder.metrics["maximum_loop_gap_m"]:.2e} tilt={recorder.metrics["maximum_tilt_deg"]:.2f}', flush=True)
        if ckpt_stride and (k + 1) % ckpt_stride == 0 and k + 1 < nsteps:
            save_checkpoint(k + 1)
    elapsed = wall_before + time.perf_counter() - t_start
    metrics = recorder.summarize()
    metrics.update(wall_simulation_s=elapsed, measured_real_time_factor=cfg.duration_s / elapsed,
                   saturation_updates=controller.saturation_updates, control_updates=controller.update_count)
    metrics['feedforward_contact_gate'] = trace.summary()
    validation = assess(args.case, cfg, metrics, {'status': 'PASS'}, True)
    result = {'engine': 'ovphysx (standalone PhysX; development harness, NOT Isaac Sim)',
              'ovphysx_version': getattr(ovphysx, '__version__', '?'), 'case': args.case,
              'usd': str(args.usd), 'configuration': asdict(cfg), 'init': init,
              'resumed_at_steps': resumes,
              'resume_note': 'contact caches/friction anchors rebuilt at each resume' if resumes else 'uninterrupted',
              'metrics': metrics, 'validation': validation}
    np.savez_compressed(out / 'native_trace.npz', **recorder.arrays())
    np.savez_compressed(out / 'controller_trace.npz', **trace.arrays())
    if args.trace_window:
        np.savez_compressed(out / 'step_trace.npz', **{k: np.asarray(v) for k, v in step_log.items()})
    write_json(out / 'result.json', result)
    if ckpt_path.exists():
        ckpt_path.unlink()
    fails = [g for g in validation['gates'] if g.get('status') != 'PASS']
    print('FUNCTIONAL:', validation['functional_status'], 'fails:', [(g['name'], g.get('value')) for g in fails], flush=True)
    physx.destroy(); stage.destroy()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
