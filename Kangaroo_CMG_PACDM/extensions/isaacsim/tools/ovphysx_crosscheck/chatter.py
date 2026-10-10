"""Summarize contact chatter and foot creep from a run directory (development only)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np


def summarize(run, windows=((1.0, 2.5), (8.0, 10.0))):
    run = Path(run)
    c = np.load(run / 'controller_trace.npz'); n = np.load(run / 'native_trace.npz')
    t = c['time_s']; fn = c['foot_normal_N']; h = c['sole_min_height_m']
    out = {}
    for a, b in windows:
        w = (t >= a) & (t < b)
        if not w.any():
            continue
        tot = fn[w].sum(axis=1)
        out[f'{a}-{b}s'] = {
            'zero_force_fraction': [round(float((fn[w, i] == 0).mean()), 3) for i in (0, 1)],
            'both_zero_fraction': round(float(((fn[w, 0] == 0) & (fn[w, 1] == 0)).mean()), 3),
            'total_normal_mean_std_max': [round(float(tot.mean()), 1), round(float(tot.std()), 1), round(float(tot.max()), 1)],
            'sole_height_um_range': [round(float(h[w].min() * 1e6), 2), round(float(h[w].max() * 1e6), 2)]}
    tn = n['time']; fp = n['foot_position']
    k0 = int(np.flatnonzero(tn >= 1.0)[0]) if (tn >= 1.0).any() else None
    if k0 is not None:
        d = np.linalg.norm(fp[k0:] - fp[k0], axis=2)
        dur = tn[-1] - tn[k0]
        out['sampled_drift_mm_final'] = np.round(d[-1] * 1000, 3).tolist()
        out['sampled_drift_mm_max'] = np.round(d.max(axis=0) * 1000, 3).tolist()
        out['creep_rate_mm_per_s'] = np.round(d[-1] * 1000 / max(dur, 1e-9), 3).tolist()
    r = json.loads((run / 'result.json').read_text())
    m = r['metrics']
    out['gate_drift_mm'] = np.round(np.array(m['maximum_foot_drift_after_landing_m']) * 1000, 3).tolist()
    out['max_final_speed'] = m.get('maximum_final_base_speed_m_s')
    out['final_pos_err'] = m.get('final_position_error_m')
    out['wall_s'] = round(m.get('wall_simulation_s', 0), 1)
    out['fails'] = [(g['name'], g.get('value')) for g in r['validation']['gates'] if g.get('status') != 'PASS']
    return out


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p); print(json.dumps(summarize(p), indent=1))
