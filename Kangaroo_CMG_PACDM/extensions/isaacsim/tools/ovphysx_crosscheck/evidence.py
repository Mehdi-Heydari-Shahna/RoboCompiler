"""Collect development evidence into JSON (ovphysx cross-check; NOT Isaac Sim)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

RUNS = Path(os.environ.get('KANGAROO_OVPHYSX_RUNS', 'runs'))
ISAAC = Path(os.environ.get('KANGAROO_ISAAC_RESULTS', 'isaac_results'))


def chatter_stats(ctrl_npz, windows=((1.0, 2.5), (8.0, 10.0))):
    c = np.load(ctrl_npz); t = c['time_s']; fn = c['foot_normal_N']; h = c['sole_min_height_m']
    out = {}
    for a, b in windows:
        w = (t >= a) & (t < b)
        if not w.any():
            continue
        tot = fn[w].sum(axis=1)
        out[f'{a}-{b}s'] = {
            'zero_normal_force_tick_fraction': [round(float((fn[w, i] == 0).mean()), 3) for i in (0, 1)],
            'both_feet_zero_fraction': round(float(((fn[w, 0] == 0) & (fn[w, 1] == 0)).mean()), 3),
            'total_normal_N_mean_std_max': [round(float(tot.mean()), 1), round(float(tot.std()), 1), round(float(tot.max()), 1)],
            'lowest_sole_corner_um_min_max': [round(float(h[w].min() * 1e6), 2), round(float(h[w].max() * 1e6), 2)]}
    return out


def gate_summary(result):
    m = result['metrics']; v = result['validation']
    return {'functional_status': v['functional_status'],
            'failed_gates': [(g['name'], g.get('value')) for g in v['gates'] if g.get('status') != 'PASS'],
            'foot_drift_mm': np.round(np.array(m['maximum_foot_drift_after_landing_m']) * 1000, 3).tolist(),
            'final_base_position_m': m['final_base_position_m'],
            'final_position_error_m': m['final_position_error_m'],
            'maximum_final_base_speed_m_s': m['maximum_final_base_speed_m_s'],
            'maximum_penetration_m': m['maximum_penetration_m'],
            'maximum_tilt_deg': m['maximum_tilt_deg'], 'final_tilt_deg': m['final_tilt_deg'],
            'achieved_crouch_m': m['achieved_crouch_m'], 'achieved_lateral_excursion_m': m['achieved_lateral_excursion_m'],
            'final_weight_error_N': m['final_weight_error_N'], 'maximum_loop_gap_m': m['maximum_loop_gap_m'],
            'maximum_motor_force_N': m['maximum_motor_force_N'],
            'total_motor_work_J': float(sum(m['final_motor_work_J'])),
            'legacy_force_gate_transitions': m.get('feedforward_contact_gate', {}).get('legacy_force_gate_transitions'),
            'wall_s': m.get('wall_simulation_s')}


def convergence(a, b):
    pa = np.array(a['metrics']['final_base_position_m']); pb = np.array(b['metrics']['final_base_position_m'])
    wa = sum(a['metrics']['final_motor_work_J']); wb = sum(b['metrics']['final_motor_work_J'])
    pd = float(np.linalg.norm(pa - pb)); wd = float(abs(wa - wb))
    return {'final_position_difference_m': pd, 'position_threshold_m': .005,
            'motor_work_difference_J': wd, 'work_threshold_J': 1.,
            'status': 'KINEMATIC_AND_WORK_GATES_PASSED' if pd <= .005 and wd <= 1. else 'FAILED'}


if __name__ == '__main__':
    out = {}
    for name in sys.argv[1:]:
        r = json.loads((RUNS / name / 'result.json').read_text())
        out[name] = gate_summary(r)
        if (RUNS / name / 'controller_trace.npz').exists():
            out[name]['chatter'] = chatter_stats(RUNS / name / 'controller_trace.npz')
    print(json.dumps(out, indent=1, default=float))
