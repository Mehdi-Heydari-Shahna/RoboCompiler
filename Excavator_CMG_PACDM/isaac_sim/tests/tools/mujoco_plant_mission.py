"""Research tool: run the soil mission with MuJoCo as the plant, through bridge.py.

The plant
is MuJoCo (mj_step) instead of PhysX, but the controller path is the same
ControllerBridge.tick()/finalize() used by run_isaac.py over the socket. It is
not evidence of PhysX behaviour. About 30-40 s wall time per simulated second.

Options:
  --pin-spin-rate R       drive the q22 pin spin toward R rad/s (PhysX-like drift)
  --no-rolling            zero MuJoCo rolling and torsional friction (PhysX has none)
  --coulomb-rolling MU    runner-style rolling resistance from net contact force
  --coulomb-torsional MU  torsional coefficient (default: same as rolling)
  --grain-angular-damping C   rejected viscous alternative, for comparison
  --perturb-joint J --perturb-amount X   initial-state perturbation (sensitivity)
  --record-efforts FILE   save the effort sequence (.npy)

Example:
  python tests/tools/mujoco_plant_mission.py --package . --no-rolling \
      --coulomb-rolling 0.008 --coulomb-torsional 0.003 --pin-spin-rate 0.12 --out k4.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument('--package', required=True, type=Path)
ap.add_argument('--duration', type=float, default=65.)
ap.add_argument('--pin-torque', type=float, default=0.)
ap.add_argument('--pin-spin-rate', type=float, default=0., help='drive q22 spin toward this rate (rad/s), emulating PhysX drift')
ap.add_argument('--no-rolling', action='store_true')
ap.add_argument('--grain-angular-damping', type=float, default=0.)
ap.add_argument('--out', required=True, type=Path)
ap.add_argument('--log-every', type=float, default=1.0)
ap.add_argument('--record-efforts', type=Path)
ap.add_argument('--perturb-grain-m', type=float, default=0., help='add this x offset to the first grain (chaos control)')
ap.add_argument('--perturb-joint', default=None)
ap.add_argument('--coulomb-rolling', type=float, default=0., help='runner-style rolling resistance: mu_r (m) * |net contact force|')
ap.add_argument('--coulomb-omega-reg', type=float, default=0.5)
ap.add_argument('--coulomb-torsional', type=float, default=None, help='defaults to --coulomb-rolling (isotropic)')
ap.add_argument('--perturb-amount', type=float, default=0.)
args = ap.parse_args()
if args.coulomb_torsional is None:
    args.coulomb_torsional = args.coulomb_rolling

pkg = args.package.resolve()
sys.path.insert(0, str(pkg))
from bridge import ControllerBridge, kinematic_observation  # noqa: E402

manifest = json.loads((pkg/'generated/soil_final/manifest.json').read_text())
base = pkg/'generated/soil_final'
src = (base/manifest['source_root']).resolve()
xml = (base/manifest['source_scene']).resolve()
m = mujoco.MjModel.from_xml_path(str(xml))
d = mujoco.MjData(m)
d.qpos[:] = np.asarray(manifest['initial_qpos'])
d.qvel[:] = 0.
grains = [i for i in range(1, m.nbody) if m.body(i).name.startswith('grain_')]
changes = {}
if args.no_rolling:
    changes['rolling_friction_zeroed_geoms'] = int(np.count_nonzero(m.geom_friction[:, 2]))
    changes['torsional_friction_zeroed_geoms'] = int(np.count_nonzero(m.geom_friction[:, 1]))
    m.geom_friction[:, 1:] = 0.
if args.grain_angular_damping > 0:
    for b in grains:
        j = m.body_jntadr[b]; v = m.jnt_dofadr[j]
        inertia = m.body_inertia[b]
        m.dof_damping[v+3:v+6] = args.grain_angular_damping*inertia
    changes['grain_angular_damping_1_per_s'] = args.grain_angular_damping
if args.perturb_joint:
    d.qpos[m.jnt_qposadr[m.joint(args.perturb_joint).id]] += args.perturb_amount
if args.perturb_grain_m:
    j = m.body_jntadr[grains[0]]; d.qpos[m.jnt_qposadr[j]] += args.perturb_grain_m
mujoco.mj_forward(m, d)
pin = m.body('body_59').id
grains_arr = np.asarray(grains, dtype=int)
rolling_torque = np.zeros((len(grains), 3))
grain_inertia = m.body_inertia[grains_arr].max(axis=1)
def coulomb_rolling_torque(m, d):
    # Emulates the Isaac runner: net contact force per grain from the last step,
    # world angular velocity, regularized Coulomb rolling-resistance torque.
    mujoco.mj_rnePostConstraint(m, d)
    force = d.cfrc_ext[grains_arr, 3:]
    normal = np.linalg.norm(force, axis=1)
    omega = np.empty((len(grains), 3))
    for k, b in enumerate(grains_arr):
        v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(b), v, 0); omega[k] = v[:3]
    axis = force/np.maximum(normal, 1e-300)[:, None]
    w_n = np.einsum('ij,ij->i', omega, axis)[:, None]*axis
    w_t = omega-w_n
    cap = 0.5*grain_inertia/m.opt.timestep
    g_r = np.minimum(args.coulomb_rolling*normal/np.maximum(np.linalg.norm(w_t, axis=1), args.coulomb_omega_reg), cap)
    g_s = np.minimum(args.coulomb_torsional*normal/np.maximum(np.linalg.norm(w_n, axis=1), args.coulomb_omega_reg), cap)
    return -g_r[:, None]*w_t-g_s[:, None]*w_n
pin_joint = m.joint('q22').id
bridge = ControllerBridge(src, xml, manifest['initial_qpos'], dt=m.opt.timestep, case='soil_final', mode='mission')
dt = m.opt.timestep
steps = int(round(args.duration/dt))
log = []
wall = time.time()
q22_adr = m.jnt_qposadr[pin_joint]
nominal_q22 = None
status = 'RUNNING'; error = None
recorded = []
last_reply = None
try:
    for step in range(steps):
        t = step*dt
        obs = kinematic_observation(m, d)
        reply = bridge.tick(t, obs['positions'], obs['quaternions_wxyz'],
                            obs['linear_velocities'], obs['angular_velocities'])
        last_reply = reply
        if args.record_efforts is not None:
            recorded.append(np.asarray(reply['efforts'], float))
        d.ctrl[:] = reply['efforts']
        d.xfrc_applied[:] = 0.
        if args.coulomb_rolling:
            d.xfrc_applied[grains_arr, 3:] = rolling_torque
        if args.pin_torque or args.pin_spin_rate:
            axis_world = d.xmat[pin].reshape(3, 3) @ m.jnt_axis[pin_joint]
            tau = args.pin_torque
            if args.pin_spin_rate:
                axial = 0.005033475242787526
                tau += axial*(args.pin_spin_rate - d.qvel[m.jnt_dofadr[pin_joint]])/0.5
            d.xfrc_applied[pin, 3:] = tau*axis_world
        mujoco.mj_step(m, d)
        if args.coulomb_rolling:
            rolling_torque = coulomb_rolling_torque(m, d)
        if not np.all(np.isfinite(d.qpos)):
            raise RuntimeError('MuJoCo state became nonfinite')
        met = reply.get('metrics', {})
        if (step+1) % int(round(args.log_every/dt)) == 0 or met.get('done'):
            rec = dict(t=(step+1)*dt, phase=met.get('phase'), phase_name=met.get('phase_name'),
                       bucket=met.get('bucket_mass_kg'), lifted=met.get('lifted_mass_kg'),
                       deposited=met.get('deposited_mass_kg'), spill=met.get('spill_mass_kg'),
                       soil_speed=met.get('soil_speed_m_s'), tracking=met.get('tracking_error_rad'),
                       q22=float(d.qpos[q22_adr]), pin_info=reply.get('passive_pin_gauge'),
                       wall=time.time()-wall)
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if met.get('done'):
            break
    obs = kinematic_observation(m, d)
    final = bridge.finalize((step+1)*dt, obs['positions'], obs['quaternions_wxyz'],
                            obs['linear_velocities'], obs['angular_velocities'])
    status = 'FINISHED'
except Exception as exc:  # record, do not hide
    import traceback
    error = '%s: %s' % (type(exc).__name__, exc)
    traceback.print_exc()
    final = bridge.summary()
    status = 'FAILED'
res = dict(status=status, error=error, changes=changes, pin_torque=args.pin_torque, pin_spin_rate=args.pin_spin_rate,
           steps=step+1, sim_time=(step+1)*dt, wall_s=time.time()-wall,
           summary=final, log=log)
args.out.write_text(json.dumps(res, indent=1, default=float))
if args.record_efforts is not None:
    np.save(args.record_efforts, np.asarray(recorded))
lm = (final or {}).get('last_metrics', {})
print('RESULT', status, error, 'done', lm.get('done'), 'failures', lm.get('failures'),
      'deposited', lm.get('deposited_mass_kg'), 'events', [e['phase'] for e in lm.get('events', [])])
