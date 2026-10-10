"""Pinocchio rigid dynamics, exact physical finger reduction, explicit RK4.

No MuJoCo imports or state stepping. PACDM provides q/v/a references only;
all recorded q and v are integrated plant states. External load is a wrench
at the tool, not an attached object or a frictional contact model.
"""
from pathlib import Path
import json
import numpy as np
import pinocchio as pin
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from panda.pin_backend import PinBackend
from panda.task import TOOL_TRANSFORM, REFERENCE_ROTATION

CASES = {
    'nominal': dict(dt=.0005, load_kg=.15),
    'half_step': dict(dt=.00025, load_kg=.15),
    'quarter_step': dict(dt=.000125, load_kg=.15),
    'heavy_load': dict(dt=.0005, load_kg=.30),
    'initial_offset': dict(dt=.0005, load_kg=.15, initial_offset=True),
    'no_feedforward': dict(dt=.0005, load_kg=.15, feedforward=False),
}


def smooth(s):
    s = np.clip(s, 0., 1.)
    return s**3 * (10.-15.*s+6.*s*s)


def applied_wrench(t, load_kg):
    """World force/torque at the tool; smooth compact pulses, SI units."""
    w = np.zeros(6)
    loading = smooth((t-4.5)/.5) * (1.-smooth((t-19.)/.5))
    w[2] = -9.81 * load_kg * loading
    if 13. < t < 13.16:
        w[:3] += np.array([2., -1.1, 0.]) * np.sin(np.pi*(t-13.)/.16)**2
    if 14.5 < t < 14.62:
        w[3:] += np.array([.008, .012, 0.]) * np.sin(np.pi*(t-14.5)/.12)**2
    return w


class Plant:
    def __init__(self, cmg, reference, load_kg=.15, feedforward=True):
        self.backend = PinBackend(cmg)
        self.cmg = cmg
        self.feedforward = feedforward
        self.load_kg = load_kg
        self.reference = BPoly.from_derivatives(reference['time'], np.stack(
            [reference['q'], reference['v'], reference['a']], axis=1))
        self.target = BPoly.from_derivatives(reference['time'], np.stack(
            [reference['active'], reference['active_v'], reference['active_a']], axis=1))
        self.S = np.zeros((9, 8))
        self.S[:8, :8] = np.eye(8)
        self.S[8, 7] = 1.
        self.B = np.asarray(cmg['actuation']['moment_matrix'])
        self.armature = np.asarray(cmg['armature'])
        self.damping = np.asarray(cmg['damping'])
        actuators = cmg['actuation']['actuators']
        self.kp = np.array([a['gain'][0] for a in actuators[:7]])
        self.kd = np.array([-a['bias'][2] for a in actuators[:7]])
        self.control_ranges = np.array([a['ctrl_range'] for a in actuators[:7]])
        self.force_ranges = np.array([a['force_range'] for a in actuators])
        b = self.backend
        hand_frame = b.body_frame_ids['hand']
        frame = b.model.frames[hand_frame]
        placement = frame.placement * pin.SE3(TOOL_TRANSFORM[:3, :3], TOOL_TRANSFORM[:3, 3])
        self.tool_id = b.model.addFrame(pin.Frame('validation_tool', frame.parentJoint,
            hand_frame, placement, pin.FrameType.OP_FRAME), False)
        b.data = b.model.createData()
        self.saturation_evaluations = 0
        self.control_clip_evaluations = 0
        self.max_residual = 0.

    def geometry(self, q, jacobian=False):
        b = self.backend
        nq = b._native_q(q)
        if jacobian:
            j = np.asarray(pin.computeFrameJacobian(b.model, b.data, nq, self.tool_id,
                pin.LOCAL_WORLD_ALIGNED))[:, b._v_indices].copy()
        pin.forwardKinematics(b.model, b.data, nq)
        pin.updateFramePlacements(b.model, b.data)
        pose = b.data.oMf[self.tool_id].homogeneous.copy()
        return (pose, j) if jacobian else pose

    def evaluate(self, t, state, accumulate=True):
        q, v = self.S @ state[:8], self.S @ state[8:16]
        qr = self.reference(t)
        vr = self.reference(t, nu=1)
        ar = self.reference(t, nu=2)
        b = self.backend
        mass = b.mass(q) + np.diag(self.armature)
        bias = b.bias(q, v)
        ff = b.inverse(q, v, ar)+self.armature*ar+self.damping*v
        if not self.feedforward:
            # Isolate arm compensation. The finger controller remains identical
            # across cases, including gravity compensation at its end stops.
            ff[:7] = 0.
        command = qr[:7] + (self.kd*vr[:7]+ff[:7])/self.kp
        clipped_command = np.clip(command, self.control_ranges[:, 0], self.control_ranges[:, 1])
        # Algebraically the source affine arm position actuators, including
        # source setpoint and effort limits. Finger control is documented below.
        effort = np.empty(8)
        effort[:7] = self.kp*(clipped_command-q[:7])-self.kd*v[:7]
        # Reduced tendon effort, B[7:9,7]=0.5. Feedforward is S^T ff.
        # 1000 N/m + 30 Ns/m benchmark servo with desired velocity and FF.
        effort[7] = ff[7:].sum()+1000.*(qr[7]-q[7])+30.*(vr[7]-v[7])
        unclipped = effort.copy()
        effort = np.clip(effort, self.force_ranges[:, 0], self.force_ranges[:, 1])
        tau = self.B @ effort
        wrench = applied_wrench(t, self.load_kg)
        if np.any(wrench):
            pose, jac = self.geometry(q, True)
            external = jac.T @ wrench
        else:
            pose = None
            external = np.zeros(9)
        rhs = tau+external-bias-self.damping*v
        reduced_acc = np.linalg.solve(self.S.T @ mass @ self.S, self.S.T @ rhs)
        acc = self.S @ reduced_acc
        residual = mass @ acc - rhs
        reaction = (residual[7]-residual[8])/2.
        residual[7] -= reaction
        residual[8] += reaction
        residual_norm = float(np.max(np.abs(residual)))
        if accumulate:
            self.saturation_evaluations += int(np.any(abs(unclipped-effort)>1e-10))
            self.control_clip_evaluations += int(np.any(abs(command-clipped_command)>1e-12))
            self.max_residual = max(self.max_residual, residual_norm)
        # Integrate actuator + external work minus passive damping work with
        # the very same RK4 quadrature, independent of the energy evaluator.
        power = float(v @ (tau+external-self.damping*v))
        derivative = np.r_[state[8:16], reduced_acc, power]
        return derivative, dict(q=q, v=v, a=acc, q_ref=qr, torque=tau,
            actuator_effort=effort, actuator_effort_demand=unclipped, ctrl=clipped_command, wrench=wrench,
            reaction=reaction, residual=residual_norm, pose=pose)


def run_case(root, name='nominal', dt=.0005, load_kg=.15,
             initial_offset=False, feedforward=True):
    root = Path(root)
    cmg = json.loads((root/'data/panda_cmg.json').read_text())
    with np.load(root/'data/reference.npz', allow_pickle=False) as f:
        reference = {key: f[key] for key in f.files}
    plant = Plant(cmg, reference, load_kg, feedforward)
    state = np.r_[reference['q'][0, :8], reference['v'][0, :8], 0.]
    if initial_offset:
        state[:7] += np.array([1., -.6, .4, .7, -.5, .3, -.4])*.001
    duration = float(reference['time'][-1])
    steps = round(duration/dt)
    stride = round(.01/dt)
    if abs(steps*dt-duration)>1e-12 or abs(stride*dt-.01)>1e-12:
        raise ValueError('dt must divide duration and 10 ms recording spacing')
    e0 = plant.backend.energy(plant.S@state[:8], plant.S@state[8:16])['total_J']
    e0 += .5*float((plant.S@state[8:16])**2 @ plant.armature)
    records = {k: [] for k in ['time','q','v','a','q_ref','torque','actuator_effort','ctrl',
        'wrench','tool_pos','tool_R','target_pos','target_R','pose_error','angle_error',
        'coupling_error','dynamics_residual','constraint_reaction','energy','work','energy_balance']}
    extrema = dict(position=0., rotation=0., joint=0., coupling=0., residual=0., energy_balance=0.,
                   arm_margin=np.inf, finger_margin=np.inf)
    joints = {j['id']: j for j in cmg['joints']}
    lower = np.array([joints[j]['limits']['lower'] for j in cmg['coordinate_ids']])
    upper = np.array([joints[j]['limits']['upper'] for j in cmg['coordinate_ids']])
    peak_effort = np.zeros(8)
    peak_demand = np.zeros(8)  # actuator effort before the source force-range clip
    sq_error = 0.
    for k in range(steps+1):
        t = k*dt
        k1, obs = plant.evaluate(t, state)
        q, v = obs['q'], obs['v']
        pose = plant.geometry(q) if obs['pose'] is None else obs['pose']
        target = plant.target(t)
        rotation = REFERENCE_ROTATION @ Rotation.from_euler('XYZ', target[3:6]).as_matrix()
        pe = float(np.linalg.norm(pose[:3, 3]-target[:3]))
        ae = float(Rotation.from_matrix(rotation.T @ pose[:3, :3]).magnitude())
        energy = plant.backend.energy(q, v)['total_J']+.5*float(v*v @ plant.armature)
        balance = energy-e0-state[16]
        extrema['position'] = max(extrema['position'], pe)
        extrema['rotation'] = max(extrema['rotation'], ae)
        extrema['joint'] = max(extrema['joint'], float(np.max(abs(q[:7]-obs['q_ref'][:7]))))
        extrema['coupling'] = max(extrema['coupling'], abs(q[7]-q[8]))
        extrema['residual'] = max(extrema['residual'], obs['residual'])
        extrema['energy_balance'] = max(extrema['energy_balance'], abs(balance))
        margin = np.minimum(q-lower, upper-q)
        extrema['arm_margin'] = min(extrema['arm_margin'], float(margin[:7].min()))
        extrema['finger_margin'] = min(extrema['finger_margin'], float(margin[7:].min()))
        peak_effort = np.maximum(peak_effort, abs(obs['actuator_effort']))
        peak_demand = np.maximum(peak_demand, abs(obs['actuator_effort_demand']))
        sq_error += pe*pe
        if k % stride == 0:
            values = [t,q,v,obs['a'],obs['q_ref'],obs['torque'],obs['actuator_effort'],obs['ctrl'],
                obs['wrench'],pose[:3,3],pose[:3,:3],target[:3],rotation,pe,ae,q[7]-q[8],
                obs['residual'],obs['reaction'],energy,state[16],balance]
            for key, value in zip(records, values):
                records[key].append(np.array(value, copy=True))
        if k < steps:
            k2, _ = plant.evaluate(t+dt/2, state+dt*k1/2)
            k3, _ = plant.evaluate(t+dt/2, state+dt*k2/2)
            k4, _ = plant.evaluate(t+dt, state+dt*k3)
            state += dt*(k1+2*k2+2*k3+k4)/6
            if not np.all(np.isfinite(state)):
                raise FloatingPointError(f'{name}: nonfinite integrated state at {t}')
    output = root/'results'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/f'{name}.npz', **{k: np.asarray(v) for k,v in records.items()})
    result = dict(case=name, dt_s=dt, steps=steps, duration_s=duration,
        load_kg_equivalent=load_kg, arm_feedforward_enabled=feedforward,
        finger_feedforward_enabled=True,
        initial_offset_enabled=initial_offset, integrator='RK4 on reduced positions/velocities and work',
        state_engine='Pinocchio 3.8.0 CRBA and nonlinear effects, exact scalar finger reduction',
        max_tool_position_error_m=extrema['position'], max_tool_orientation_error_deg=float(np.rad2deg(extrema['rotation'])),
        max_arm_joint_error_deg=float(np.rad2deg(extrema['joint'])),
        rms_tool_position_error_m=float(np.sqrt(sq_error/(steps+1))),
        max_coupling_error_m=extrema['coupling'], max_dynamics_residual=max(extrema['residual'],plant.max_residual),
        max_energy_work_balance_error_J=extrema['energy_balance'], minimum_arm_limit_margin_rad=extrema['arm_margin'],
        minimum_finger_limit_margin_m=extrema['finger_margin'], peak_actuator_effort=peak_effort.tolist(), peak_actuator_effort_demand=peak_demand.tolist(),
        saturation_rhs_evaluations=plant.saturation_evaluations, setpoint_clip_rhs_evaluations=plant.control_clip_evaluations,
        final_tool_error_m=pe, final_tool_orientation_error_deg=float(np.rad2deg(ae)),
        stored_samples=len(records['time']), stored_spacing_s=.01,
        scope='Tool wrench benchmark; no payload contact, collision, friction, grasp or socket simulation')
    (output/f'{name}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
