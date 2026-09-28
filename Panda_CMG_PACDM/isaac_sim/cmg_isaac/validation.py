"""Predeclared acceptance gates. No missing data is treated as PASS."""
import math
import numpy as np

# Wrench time-step refinement (suite level, report.py).
# PhysX stores articulation joint angles in float32 and advances them once per
# step (PGS; once per sub-step with TGS), so each step rounds the joint
# increment by up to half a float32 spacing (ulp). At a steady speed that rounding acts like a velocity
# bias of up to ulp/(2*dt), which the source PD servo converts into a position
# offset of about (Kd/Kp)*ulp/(2*dt): 0.1 s * 3.0e-8 rad / 62.5e-6 s = 4.8e-5 rad
# for the 0.0625 ms case at |q_native| < 1 rad, i.e. up to ~29 um at the tool.
# The floor therefore GROWS as dt shrinks, so the original double-precision RK4
# limits (10 um and a contraction test with a 2 um floor) cannot be met by any
# float32 engine at 0.125/0.0625 ms. Declared PhysX limits: 30 um for both the
# per-pair tool-position difference and the contraction floor. Rotation
# (0.005 deg) and joint (1e-4 rad) limits are unchanged. The report still
# records whether the original 10 um value was met.
WRENCH_REFINEMENT_TOOL_POSITION_M = 3e-5
WRENCH_REFINEMENT_FLOAT32_FLOOR_M = 3e-5


def check(value,limit,relation='<='):
    try:
        x=float(value)
        ok=math.isfinite(x) and (x<=limit if relation=='<=' else x>=limit if relation=='>=' else x==limit)
    except (TypeError,ValueError):
        x=None;ok=False
    return dict(passed=bool(ok),value=x if x is not None and math.isfinite(x) else None,
                limit=limit,relation=relation)


def validate_case(summary,mechanics,smoke=False):
    s=summary;checks={}
    def gate(name,key,limit,relation='<='):
        checks[name]=check(s.get(key),limit,relation)
    checks['native_mechanics']=check(int(mechanics.get('passed',False)),1,'==')
    gate('complete_duration','duration_s',.25 if smoke else 22.,'>=')
    gate('native_errors','physics_error_count',0,'==')
    gate('arm_limit_margin','min_arm_joint_margin_rad',.05,'>=')
    gate('finger_limit_margin','min_finger_joint_margin_m',-.0001,'>=')
    gate('source_torque_limits','max_arm_torque_ratio',1.000001)
    gate('source_all_actuator_limits','max_actuator_effort_ratio',1.000001)
    gate('coupled_fingers','max_finger_coupling_error_m',.0005)
    if s['mode']=='contact':
        gate('native_contact_instrumentation','native_contact_points',1,'>=')
    base=dict(checks)
    if smoke:
        return dict(status='ENGINE_SMOKE_PASS' if all(v['passed'] for v in checks.values()) else 'ENGINE_SMOKE_FAIL',
                    passed=False,smoke_passed=all(v['passed'] for v in checks.values()),
                    data_valid=all(v['passed'] for v in checks.values()),checks=checks,
                    scope='0.25 s smoke test only. This does not validate the 22 s manipulation task.')
    gate('tool_translation','max_tool_position_error_m',.002)
    gate('tool_rotation','max_tool_orientation_error_deg',.5)
    gate('no_arm_saturation','arm_saturation_steps',0,'==')
    gate('no_control_clipping','arm_control_clip_steps',0,'==')
    if s['mode']=='contact':
        gate('no_unexpected_contact','unexpected_contact_steps',0,'==')
        gate('lift','max_lift_m',.23,'>=')
        gate('airborne_transfer','min_transfer_height_m',.28,'>=')
        gate('barrier_clearance','minimum_payload_barrier_clearance_m',.05,'>=')
        gate('grasp_slip','max_grasp_slip_m',.005)
        gate('left_pad_load','min_transfer_left_normal_N',.1,'>=')
        gate('right_pad_load','min_transfer_right_normal_N',.1,'>=')
        gate('bilateral_pad_contacts','bilateral_contact_all_transfer',1,'==')
        gate('socket_xy','placement_xy_error_m',.003)
        gate('socket_z','placement_z_error_m',.001)
        gate('socket_rotation','placement_angle_error_deg',2.)
        # Instantaneous PhysX velocity, as in v1. summary.json also reports the
        # logged-motion speeds (settled_speed_m_s, settled_angular_speed_rad_s)
        # because a resting body's solver velocity can contain a residual.
        gate('settled_translation','final_speed_m_s',.001)
        gate('settled_rotation','final_angular_speed_rad_s',.02)
        gate('contact_force_scale','initial_support_relative_error',.30)
    else:
        gate('arm_joint_tracking','max_arm_joint_error_deg',.5)
        gate('final_tool_recovery','final_tool_error_m',1e-4)
        # PhysX first-order step + trapezoidal work, NOT source RK4 tolerance.
        gate('energy_balance','max_energy_balance_error_J',.020)
    if s['case']=='no_grasp':
        negative=dict(base)
        negative['open_hand_no_lift']=check(s.get('max_lift_m'),.03)
        negative['open_hand_misses_socket']=check(s.get('placement_xy_error_m'),.15,'>=')
        ok=all(v['passed'] for v in negative.values())
        return dict(status='NEGATIVE_CONTROL_PASS' if ok else 'NEGATIVE_CONTROL_FAIL',passed=ok,
                    task_passed=all(v['passed'] for v in checks.values()),data_valid=all(v['passed'] for v in base.values()),
                    checks=negative,task_checks=checks,
                    scope='Expected failure of manipulation with fingers held open; not a successful grasp.')
    if s['case']=='no_feedforward':
        ok=all(v['passed'] for v in base.values())
        return dict(status='ABLATION_RECORDED' if ok else 'ABLATION_INVALID',passed=None,
                    task_passed=all(v['passed'] for v in checks.values()),data_valid=ok,checks=base,task_checks=checks,
                    scope='Ablation data only. Feedforward benefit is tested against nominal in the full suite.')
    ok=all(v['passed'] for v in checks.values())
    return dict(status='TASK_PASS' if ok else 'TASK_FAIL',passed=ok,data_valid=all(v['passed'] for v in base.values()),checks=checks,
                scope='Finite single-case native evidence. Full suite additionally requires ablation and refinement checks.')
