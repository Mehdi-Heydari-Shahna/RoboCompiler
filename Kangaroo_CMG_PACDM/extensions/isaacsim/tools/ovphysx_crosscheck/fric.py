"""Friction diagnostics from an ovrun step trace (development only)."""
from __future__ import annotations
import json, sys
import numpy as np


def friction_stats(path, window=None):
    d = np.load(path); t = d['t']
    w = np.ones_like(t, bool) if window is None else (t >= window[0]) & (t < window[1])
    fr = d['friction_ratio'][w]; fxy = d['friction_xy'][w]; fn = d['fn'][w]
    xy = d['foot_xy'][w]; dt = float(np.median(np.diff(t[w])))
    out = {'window_s': [float(t[w][0]), float(t[w][-1])], 'steps': int(w.sum()), 'dt_s': dt}
    for i in (0, 1):
        f = fxy[:, i, :]; mag = np.linalg.norm(f, axis=1)
        # step-to-step sign reversal of the dominant friction component
        comp = f[:, np.argmax(np.abs(f).mean(axis=0))]
        flips = np.mean(np.sign(comp[1:]) * np.sign(comp[:-1]) < 0)
        v = np.diff(xy[:, i, :], axis=0) / dt
        out[f'foot{i}'] = {
            'normal_mean_std_N': [round(float(fn[:, i].mean()), 1), round(float(fn[:, i].std()), 1)],
            'friction_mag_mean_std_N': [round(float(mag.mean()), 1), round(float(mag.std()), 1)],
            'friction_ratio_median': round(float(np.median(fr[:, i])), 3),
            'friction_ratio_ge_0p79_fraction': round(float((fr[:, i] >= .79).mean()), 3),
            'friction_sign_flip_fraction': round(float(flips), 3),
            'tangential_speed_rms_mm_s': round(float(np.sqrt((v**2).sum(axis=1).mean()) * 1e3), 3),
            'net_slide_um': round(float(np.linalg.norm(xy[-1, i] - xy[0, i]) * 1e6), 2)}
    return out


if __name__ == '__main__':
    win = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else None
    print(json.dumps(friction_stats(sys.argv[1], win), indent=1))
