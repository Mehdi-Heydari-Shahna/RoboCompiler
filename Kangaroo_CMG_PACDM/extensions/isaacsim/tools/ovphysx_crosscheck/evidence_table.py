"""Summarize ovphysx development runs into JSON + Markdown (NOT Isaac Sim evidence)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import gate_summary, convergence

RUNS = Path(os.environ.get('KANGAROO_OVPHYSX_RUNS', 'runs'))


def load(name):
    return json.loads((RUNS / name / 'result.json').read_text())


def table(groups):
    """groups: {label: {'nominal': run, 'refined': run, 'solver_check': run}}"""
    out = {}; md = []
    md.append('| configuration | case | status | foot drift L/R mm | final base x/y mm | Σ motor work J | final speed m/s | max tilt ° | loop gap m | wall s | resumes |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for label, cases in groups.items():
        out[label] = {'cases': {}, 'convergence': {}}
        loaded = {}
        for case, run in cases.items():
            if not (RUNS / run / 'result.json').exists():
                out[label]['cases'][case] = {'run': run, 'status': 'NOT_COMPLETED'}
                md.append(f'| {label} | {case} | NOT COMPLETED | | | | | | | | |'); continue
            r = load(run); loaded[case] = r; g = gate_summary(r)
            g.update(run=run, resumed_at_steps=r.get('resumed_at_steps', []),
                     configuration={k: r['configuration'].get(k, 'rigid (field absent)' if k == 'contact_model' else None) for k in ('dt_s', 'solver_type', 'solver_position_iterations',
                                                                       'solver_velocity_iterations', 'contact_model')})
            out[label]['cases'][case] = g
            b = g['final_base_position_m']
            md.append(f"| {label} | {case} | {g['functional_status']} | {g['foot_drift_mm'][0]:.2f} / {g['foot_drift_mm'][1]:.2f} | "
                      f"{b[0]*1e3:.2f} / {b[1]*1e3:.2f} | {g['total_motor_work_J']:.3f} | {g['maximum_final_base_speed_m_s']:.4f} | "
                      f"{g['maximum_tilt_deg']:.2f} | {g['maximum_loop_gap_m']:.2e} | {g['wall_s']:.0f} | {len(g['resumed_at_steps'])} |")
        for fine in ('refined', 'solver_check'):
            if 'nominal' in loaded and fine in loaded:
                out[label]['convergence'][fine] = convergence(loaded['nominal'], loaded[fine])
    md.append('')
    md.append('| configuration | comparison | Δ final base position mm (≤ 5) | Δ Σ motor work J (≤ 1) | status |')
    md.append('|---|---|---|---|---|')
    for label, v in out.items():
        for fine, c in v['convergence'].items():
            md.append(f"| {label} | nominal vs {fine} | {c['final_position_difference_m']*1e3:.2f} | {c['motor_work_difference_J']:.3f} | {c['status']} |")
    return out, '\n'.join(md)


if __name__ == '__main__':
    groups = json.loads(sys.argv[1])
    data, md = table(groups)
    print(md)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(data, indent=1, default=float))
