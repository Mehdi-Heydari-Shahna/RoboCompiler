"""Offline HTML report generated only from executed result files (no canned numbers)."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

LABELS = {'landing_nominal': 'Nominal · 1 ms', 'landing_refined': 'Refined · 0.5 ms',
          'landing_low_friction': 'Low friction · μ 0.4', 'landing_higher_drop_push': 'Drop 8 cm · push 80 N',
          'landing_slow_actuators': 'Slow drives · τ 4 ms'}
COLORS = {'landing_nominal': '#2176d2', 'landing_refined': '#d49a24', 'landing_low_friction': '#22a89a',
          'landing_higher_drop_push': '#b8433a', 'landing_slow_actuators': '#9861ba'}
NEGATIVE = {'negative_no_contact': 'Contact removed', 'negative_no_loops_contact': 'All 24 loop cuts removed',
            'negative_passive': 'Drive forces removed'}


def _read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _n(value, digits=4):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return f'{value:.{digits}g}'
    return html.escape(str(value))


def _gates(checks):
    rows = []
    for name, check in checks.items():
        good = check.get('passed', False)
        rows.append('<tr><td class="gate-name">' + html.escape(name) + '</td><td>' + _n(check.get('value'), 7)
                    + '</td><td>' + html.escape(str(check.get('relation', '<='))) + ' ' + _n(check.get('limit'), 7)
                    + '</td><td>' + html.escape(str(check.get('unit', check.get('units', '')))) + '</td><td class="'
                    + ('pass' if good else 'fail') + '">' + ('PASS' if good else 'FAIL') + '</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Check</th><th>Observed</th><th>Acceptance</th><th>Unit</th>'
            '<th>Result</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def _figures(root, result):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = root / 'results'
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
             'axes.titleweight': 'bold', 'axes.labelcolor': '#25364b', 'xtick.color': '#4d5d72',
             'ytick.color': '#4d5d72'}
    weight = result['cases']['landing_nominal']['mass_kg'] * 9.81
    paths = []
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
        for name, label in LABELS.items():
            path = r / f'{name}.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as a:
                t, dt = a['time'], float(a['timestep_s'])
                stride = max(1, round(.002 / dt))
                kw = dict(color=COLORS[name], label=label, linewidth=1.2, alpha=.9)
                axes[0, 0].plot(t[::stride], a['base'][::stride, 2], **kw)
                normal = a['lam'][:, 2::3].sum(axis=1) / dt / weight
                axes[0, 1].plot(t[::stride], normal[::stride], **kw)
                axes[1, 0].plot(t[::stride], np.max(np.abs(a['act'][::stride]), axis=1), **kw)
                axes[1, 1].plot(t[::stride], a['ledger'][::stride], **kw)
        legacy = root / 'legacy_evidence/landing_nominal.npz'
        if legacy.exists():
            with np.load(legacy, allow_pickle=False) as m:
                kw = dict(color='#111111', linestyle=(0, (4, 3)), linewidth=1.1, label='MuJoCo v22 nominal (legacy)')
                axes[0, 0].plot(m['time'], m['base_position'][:, 2], **kw)
                cols = list(m['history_columns'])
                h = m['history_1ms']
                axes[0, 1].plot(h[:, 0], h[:, cols.index('ground_Fz_N')] / weight, **kw)
                axes[1, 0].plot(m['time'], np.max(np.abs(m['motor_force_N']), axis=1), **kw)
        for ax, title, unit in [(axes[0, 0], 'Pelvis height', 'z (m)'),
                                (axes[0, 1], 'Total ground normal force / weight', 'ratio'),
                                (axes[1, 0], 'Largest absolute drive force', 'N'),
                                (axes[1, 1], 'Discrete energy-ledger error', 'J')]:
            ax.set_title(title, loc='left', pad=10)
            ax.set_xlabel('Simulation time (s)')
            ax.set_ylabel(unit)
            ax.set_xlim(0, 10)
            ax.grid(alpha=.16)
            ax.axvspan(6.7, 6.95, color='#d95d39', alpha=.10, linewidth=0)
        peak = max([line.get_ydata().max() for line in axes[0, 1].get_lines()] + [5.])
        axes[0, 1].set_ylim(0, 1.05 * peak)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(.5, .955), fontsize=9)
        fig.suptitle('Kangaroo · executed Pinocchio / PACDM contact simulation', x=.07, ha='left', fontsize=17,
                     weight='bold', color='#15273c')
        fig.text(.07, .02, 'Shaded band: sideways push. Pinocchio ground force is the rigid-contact impulse per step '
                           'divided by the step; MuJoCo force is the instantaneous soft-contact force (1 ms samples).',
                 fontsize=9, color='#566678')
        fig.subplots_adjust(top=.86, bottom=.1, left=.07, right=.975, hspace=.4, wspace=.24)
        path = r / 'performance.png'
        fig.savefig(path, dpi=150, facecolor='white')
        plt.close(fig)
        paths.append(path)
        # Landing detail: where rigid (Pinocchio) and soft (MuJoCo) contact differ most.
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        for name, style in [('landing_nominal', '-'), ('landing_refined', '-')]:
            path = r / f'{name}.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as a:
                dt = float(a['timestep_s'])
                use = a['time'] <= .6
                kw = dict(color=COLORS[name], label='Pinocchio ' + LABELS[name], linewidth=1.3)
                axes[0].plot(a['time'][use], a['base'][use, 2], **kw)
                axes[1].plot(a['time'][use], a['lam'][use, 2::3].sum(axis=1) / dt, **kw)
        for name, color in [('landing_nominal', '#111111'), ('landing_refined', '#777777')]:
            path = root / 'legacy_evidence' / f'{name}.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as m:
                kw = dict(color=color, linestyle=(0, (4, 3)), linewidth=1.2,
                          label=f"MuJoCo v22 {'nominal 25 us' if name == 'landing_nominal' else 'refined 12.5 us'}")
                use = m['time'] <= .6
                axes[0].plot(m['time'][use], m['base_position'][use, 2], **kw)
                cols = list(m['history_columns'])
                h = m['history_1ms']
                use = h[:, 0] <= .6
                axes[1].plot(h[use, 0], h[use, cols.index('ground_Fz_N')], **kw)
        axes[0].set_title('Pelvis height during landing', loc='left')
        axes[0].set_ylabel('z (m)')
        axes[1].set_title('Total ground normal force during landing', loc='left')
        axes[1].set_ylabel('N')
        for ax in axes:
            ax.set_xlabel('Simulation time (s)')
            ax.set_xlim(0, .6)
            ax.grid(alpha=.16)
        axes[1].legend(frameon=False, fontsize=8)
        fig.text(.06, .01, 'Pinocchio: rigid NCP impulse per step / step (1 ms and 0.5 ms). MuJoCo: soft contact, force '
                           'sampled every 1 ms (5 ms for pelvis height).', fontsize=9, color='#566678')
        fig.subplots_adjust(top=.9, bottom=.2, left=.06, right=.98, wspace=.22)
        path = r / 'landing_detail.png'
        fig.savefig(path, dpi=150, facecolor='white')
        plt.close(fig)
        paths.append(path)
        comparison = result.get('legacy_mujoco_comparison', {})
        if comparison:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
            names = [n for n in LABELS if n in comparison]
            x = np.arange(len(names))
            bands = result['legacy_bands']
            for ax, key, band, unit, scale, title in [
                    (axes[0], 'max_base_position_difference_m', bands['base_position_m'], 'mm', 1e3,
                     'Max pelvis position difference'),
                    (axes[1], 'max_base_orientation_difference_deg', bands['base_orientation_deg'], 'deg', 1.,
                     'Max pelvis orientation difference'),
                    (axes[2], 'max_motor_coordinate_difference_m', None, 'mm', 1e3,
                     'Max motor-slide coordinate difference')]:
                values = [comparison[n][key] * scale for n in names]
                ax.bar(x, values, color=[COLORS[n] for n in names])
                if band is not None:
                    ax.axhline(band * scale, color='#b13c27', linestyle=':', linewidth=1.2, label='declared band')
                    ax.legend(frameon=False, fontsize=8)
                ax.set_xticks(x, [LABELS[n].split(' · ')[0] for n in names], rotation=20, ha='right', fontsize=8)
                ax.set_title(title, loc='left', fontsize=10)
                ax.set_ylabel(unit)
                ax.grid(axis='y', alpha=.16)
                ax.set_axisbelow(True)
            fig.suptitle('Pinocchio (rigid contact) vs rerun MuJoCo v22 (soft contact), 5 ms samples', x=.06,
                         ha='left', fontsize=13, weight='bold', color='#15273c')
            fig.subplots_adjust(top=.82, bottom=.24, left=.06, right=.98, wspace=.28)
            path = r / 'cross_engine.png'
            fig.savefig(path, dpi=150, facecolor='white')
            plt.close(fig)
            paths.append(path)
    return paths


def _video(root, execution):
    r = root / 'results'
    meta = _read(r / 'render_metadata.json', {})
    if not meta:
        return ('<p class="notice">No video matching the saved trajectory is present. Run '
                '<code>python run_pinocchio.py --render-only</code>.</p>')
    video, poster = r / meta['video'], r / meta['poster']
    current = (video.is_file() and _hash(video) == meta.get('video_sha256')
               and (r / meta['source_trajectory']).is_file()
               and _hash(r / meta['source_trajectory']) == meta.get('source_trajectory_sha256'))
    if not current:
        return ('<p class="notice">The video does not match the current saved trajectory. Run '
                '<code>python run_pinocchio.py --render-only</code>.</p>')
    return (f'<video controls preload="metadata" poster="results/{html.escape(poster.name)}">'
            f'<source src="results/{html.escape(video.name)}" type="video/mp4"></video>'
            f'<p class="caption">{_n(meta["duration_s"])} s · {meta["fps"]} fps · {meta["resolution"][0]} × '
            f'{meta["resolution"][1]} · {meta["frames"]} frames, each an exact saved 5 ms state of '
            f'{html.escape(meta["source_trajectory"])} (variable replay speed, shown on screen). Meshes: original '
            f'upstream STL files placed by Pinocchio forward kinematics; largest difference between rendered and '
            f'logged foot positions {_n(meta["max_rendered_foot_fk_vs_logged_m"], 2)} m. '
            f'<a href="results/{html.escape(video.name)}">Open MP4</a> · '
            '<a href="results/render_metadata.json">Render provenance</a></p>')


def build_report(root):
    root = Path(root)
    r = root / 'results'
    result = _read(r / 'validation.json', {})
    execution = _read(r / 'execution.json', {})
    passed = bool(result.get('passed'))
    cases = result.get('cases', {})
    nominal = cases.get('landing_nominal', {})
    figures = _figures(root, result) if cases else []
    comparison = result.get('legacy_mujoco_comparison', {})
    rows = []
    for name, label in LABELS.items():
        c = cases.get(name)
        if not c:
            continue
        cfg = c['configuration']
        values = [c['timestep_s'] * 1e3, cfg['friction'], cfg['drop_height_m'] * 100, cfg['push_force_N'],
                  cfg['actuator_time_constant_s'] * 1e3, c['touchdown_s'], c['maximum_ground_normal_N'],
                  c['maximum_motor_force_N'], c['achieved_crouch_m'] * 100, c['achieved_lateral_excursion_m'] * 100,
                  c['maximum_tilt_deg'], c['final_position_error_m'] * 1e3, c['maximum_energy_ledger_error_J'],
                  c['admm_fallback_steps'], c['maximum_contact_iterations'], c['fallback_count']]
        rows.append('<tr><td>' + html.escape(label) + '</td>' + ''.join(f'<td>{_n(v)}</td>' for v in values) + '</tr>')
    case_table = ('<div class="scroll"><table><thead><tr><th>Case</th><th>dt (ms)</th><th>μ</th><th>Drop (cm)</th>'
                  '<th>Push (N)</th><th>τ drive (ms)</th><th>Touchdown (s)</th><th>Peak ground normal (N)</th>'
                  '<th>Peak drive force (N)</th><th>Crouch (cm)</th><th>Lateral (cm)</th><th>Peak tilt (°)</th>'
                  '<th>Final position error (mm)</th><th>Max ledger error (J)</th><th>ADMM steps</th>'
                  '<th>Max NCP iterations</th><th>PACDM re-acquisitions</th></tr></thead><tbody>' + ''.join(rows)
                  + '</tbody></table></div>')
    rows = []
    for name, label in LABELS.items():
        c = comparison.get(name)
        if not c:
            continue
        values = [c['max_base_position_difference_m'] * 1e3, c['rms_base_position_difference_m'] * 1e3,
                  c['final_base_position_difference_m'] * 1e3, c['max_base_orientation_difference_deg'],
                  c['max_motor_coordinate_difference_m'] * 1e3, c['rms_drive_force_difference_N'],
                  c['rms_stance_foot_normal_force_difference_N'], c['touchdown_s']['pinocchio'],
                  c['touchdown_s']['mujoco'], c['maximum_ground_normal_N']['pinocchio'],
                  c['maximum_ground_normal_N']['mujoco'], c['maximum_loop_gap_m']['pinocchio'],
                  c['maximum_loop_gap_m']['mujoco']]
        rows.append('<tr><td>' + html.escape(label) + '</td>' + ''.join(f'<td>{_n(v)}</td>' for v in values) + '</tr>')
    legacy_table = ('<div class="scroll"><table><thead><tr><th>Case</th><th>Max pelvis Δ (mm)</th><th>RMS pelvis Δ (mm)'
                    '</th><th>Final pelvis Δ (mm)</th><th>Max orientation Δ (°)</th><th>Max motor Δ (mm)</th>'
                    '<th>RMS drive force Δ (N)</th><th>RMS stance foot normal Δ (N)</th><th>Touchdown Pin (s)</th>'
                    '<th>Touchdown MJ (s)</th><th>Peak normal Pin (N)</th><th>Peak normal MJ (N)</th>'
                    '<th>Loop gap Pin (m)</th><th>Loop gap MJ (m)</th></tr></thead><tbody>' + ''.join(rows)
                    + '</tbody></table></div>')
    neg_rows = []
    for name, label in NEGATIVE.items():
        c = cases.get(name, {})
        if not c:
            continue
        if name == 'negative_no_loops_contact':
            observed = f"maximum native loop gap {_n(c['maximum_loop_gap_m'])} m"
        elif name == 'negative_no_contact':
            observed = (f"pelvis falls to {_n(c['final_base_position_m'][2])} m in {_n(c['duration_s'])} s; "
                        f"ground force {_n(c['maximum_ground_normal_N'])} N")
        else:
            observed = (f"pelvis at {_n(c['final_base_position_m'][2])} m when the run ended "
                        f"({html.escape(str(c.get('stop_reason') or 'completed'))}); drive force "
                        f"{_n(c['maximum_motor_force_N'])} N")
        neg_rows.append(f'<tr><td>{html.escape(label)}</td><td>{observed}</td></tr>')
    neg_table = ('<div class="scroll"><table><thead><tr><th>Control</th><th>Observed</th></tr></thead><tbody>'
                 + ''.join(neg_rows) + '</tbody></table></div>')
    ref = result.get('refinement', {})
    fine, coarse = cases.get('landing_refined', {}), cases.get('landing_nominal', {})
    refinement = ''
    if ref and fine and coarse:
        refinement = (f"<p>1 ms vs 0.5 ms (common 5 ms grid): maximum pelvis difference "
                      f"{_n(ref['max_base_position_difference_m'] * 1e3)} mm, final "
                      f"{_n(ref['final_base_position_difference_m'] * 1e3)} mm, orientation "
                      f"{_n(ref['max_base_orientation_difference_deg'])}°; motor work "
                      f"{_n(coarse['work_J']['motor'])} J vs {_n(fine['work_J']['motor'])} J; maximum energy-ledger "
                      f"error {_n(coarse['maximum_energy_ledger_error_J'])} J vs "
                      f"{_n(fine['maximum_energy_ledger_error_J'])} J.</p>")
    groups = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v["passed"]} / {v["total"]}</td></tr>'
                     for k, v in result.get('groups', {}).items())
    nested = ''
    by_group = {}
    for key, check in result.get('checks', {}).items():
        by_group.setdefault(check.get('group', key.split('.')[0]), {})[key] = check
    for group, checks in by_group.items():
        good = sum(c['passed'] for c in checks.values())
        nested += (f'<details><summary>{html.escape(group)} · {good}/{len(checks)}</summary>' + _gates(checks)
                   + '</details>')
    evidence = _read(r / 'case_evidence.json', {})
    for name, audit in evidence.items():
        erows = []
        for key, check in audit.get('checks', {}).items():
            if 'max_absolute_error' in check:
                observed = _n(check['max_absolute_error'], 6)
                requirement = f"atol {_n(check.get('absolute_tolerance'))}; rtol {_n(check.get('relative_tolerance'))}"
            else:
                observed = 'consistent' if check.get('passed') else 'inconsistent'
                requirement = 'exact / structural'
            erows.append(f'<tr><td class="gate-name">{html.escape(key)}</td><td>{observed}</td><td>{requirement}'
                         f'</td><td class="{"pass" if check.get("passed") else "fail"}">'
                         f'{"PASS" if check.get("passed") else "FAIL"}</td></tr>')
        nested += (f'<details><summary>Saved-evidence replay · {html.escape(name)} · '
                   f'{sum(c["passed"] for c in audit.get("checks", {}).values())}/{len(audit.get("checks", {}))}'
                   '</summary><div class="scroll"><table><thead><tr><th>Check</th><th>Max abs error / status</th>'
                   '<th>Requirement</th><th>Result</th></tr></thead><tbody>' + ''.join(erows)
                   + '</tbody></table></div></details>')
    versions = ', '.join(f'{html.escape(k)} {html.escape(str(v))}' for k, v in execution.get('versions', {}).items())
    limits = ''.join(f'<li>{html.escape(s)}</li>' for s in result.get('limits', []))
    status = 'PASS' if passed else 'INCOMPLETE / FAIL'
    worst = max((c['max_base_position_difference_m'] for c in comparison.values()), default=None)
    mj_gap = max((c['maximum_loop_gap_m']['mujoco'] for c in comparison.values()), default=None)
    def fig(name):
        return ''.join(f'<a href="results/{p.name}"><img src="results/{p.name}" alt="{html.escape(p.stem)}"></a>'
                       for p in figures if p.stem == name)
    integrity = result.get('source_integrity', {})
    rebuild = result.get('reference_rebuild', {})
    if not rebuild:
        rebuild_text = 'not available'
    elif rebuild.get('file_bit_identical'):
        rebuild_text = 'the rebuilt file is bit-identical to the original contact_reference.npz'
    else:
        rebuild_text = ('largest array difference to the original file '
                        + _n(rebuild.get('maximum_array_difference')))
    content = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kangaroo · Pinocchio / PACDM validation</title>
<style>
:root{{--ink:#172a41;--muted:#58697d;--blue:#1d6eb9;--line:#dbe4ed;--bg:#f3f6fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,Segoe UI,sans-serif}}
header{{background:#142c46;color:#fff;padding:46px max(25px,calc((100% - 1200px)/2)) 38px}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#8fc7ee;font-weight:700}}
h1{{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:12px 0 16px}}header p{{max-width:940px;color:#ccdaea;margin:0}}
main{{max-width:1280px;margin:0 auto;padding:26px}}section{{background:white;border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:22px}}
h2{{font-size:23px;margin:0 0 14px}}p{{margin:10px 0}}a{{color:var(--blue)}}
.status{{display:inline-block;border-radius:20px;padding:5px 13px;font-weight:750;background:{'#d9f6e9' if passed else '#ffe6de'};color:{'#126247' if passed else '#9c3522'};margin-right:8px}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:23px 0 0}}.tile{{background:#f3f7fb;padding:16px;border-radius:8px}}.tile strong{{display:block;font-size:24px;color:#1a5c91}}.tile span{{font-size:13px;color:var(--muted)}}
video{{width:100%;display:block;background:#0e1b2c;border-radius:8px}}img{{max-width:100%;height:auto;display:block;margin:12px 0}}.caption,.small{{font-size:13px;color:var(--muted)}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{background:#eef4f9;color:#314f6a;text-align:left}}th,td{{padding:9px 11px;border-bottom:1px solid #e4ebf2;vertical-align:top}}td{{font-variant-numeric:tabular-nums}}.gate-name{{overflow-wrap:anywhere}}.pass{{color:#137452;font-weight:700}}.fail{{color:#b13c27;font-weight:700}}
details{{border:1px solid var(--line);border-radius:7px;margin:10px 0;padding:0 13px}}summary{{cursor:pointer;font-weight:650;padding:12px 0}}code{{font-size:13px;background:#edf2f7;padding:2px 5px;border-radius:4px}}.notice{{background:#fff3d9;padding:18px;border-radius:7px}}
ul{{padding-left:21px}}li{{margin:8px 0}}.links{{display:flex;gap:18px;flex-wrap:wrap}}footer{{font-size:12px;color:var(--muted);padding:0 6px 20px}}
@media(max-width:760px){{main{{padding:15px}}section{{padding:18px}}.tiles{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<header><div class="eyebrow">Simulation verification · v22 CMG · no MuJoCo in this pipeline</div>
<h1>Kangaroo full body<br>Pinocchio + unchanged PACDM</h1>
<p>Floating-base landing, crouch, weight shift, turn, rise and push recovery of the 78-body Kangaroo reconstruction:
Pinocchio 3.8.0 tree dynamics, the unchanged PACDM closure algorithm for all 24 loop cuts, and Pinocchio's native
rigid-contact NCP solvers at the foot corners. Independent native audits, full evidence replay and a labelled
comparison with the rerun MuJoCo v22 pipeline.</p></header>
<main><section><h2><span class="status">{status}</span> {_n(result.get('passed_count'))} / {_n(result.get('check_count'))} aggregate gates</h2>
<p>{html.escape(result.get('summary', result.get('status', 'Aggregate validation has not completed.')))}</p>
<div class="tiles"><div class="tile"><strong>{_n(nominal.get('maximum_motor_force_N'))} N</strong><span>Nominal peak drive force (limit 5000 N)</span></div>
<div class="tile"><strong>{_n(nominal.get('maximum_tilt_deg'), 3)}°</strong><span>Nominal peak pelvis tilt (limit 8°)</span></div>
<div class="tile"><strong>{_n(nominal.get('maximum_pacdm_closure'), 2)}</strong><span>Nominal max all-row loop closure after polish (limit 5e-13)</span></div>
<div class="tile"><strong>{_n(worst * 1e3 if worst is not None else None, 3)} mm</strong><span>Largest pelvis difference vs MuJoCo v22 over 5 cases (band 10 mm)</span></div></div>
<table><thead><tr><th>Gate group</th><th>Passed</th></tr></thead><tbody>{groups}</tbody></table></section>
<section><h2>Recorded nominal run</h2>{_video(root, execution)}</section>
<section><h2>Five executed task cases</h2>{case_table}
<p class="small">All cases run the complete 10 s v22 task: release 5 cm above the floor, landing, 12 cm crouch, lateral
weight shift with pelvis yaw, rise and a sideways push at the torso centre of mass (50 N peak, 6.7–6.95 s). Drive
commands follow the unchanged v22 law (1 kHz, feedforward after contact, PD on motor length, ±5000 N clip,
1.5 MN/s slew, first-order force response). Acceptance thresholds are the unchanged v22 values. ADMM steps are
steps where PGS did not reach its tolerance; the NCP iteration count is the largest used by the accepted solver in a
step (the ADMM cap is 50 000). Every accepted contact solution satisfies the native cone residual limit 1e-8, and all
ADMM steps are included in the native audits. Pinocchio's peak ground force is an impulse per step, so it grows as
the step shrinks.</p>{fig('performance')}</section>
<section><h2>Cross-engine comparison (labelled legacy evidence)</h2>
<p>The unchanged v22 MuJoCo pipeline was rerun separately in its validated environment (MuJoCo 3.3.7, 25 µs / 12.5 µs
steps, soft contact). Its 5 ms samples are compared with the Pinocchio runs at identical times. Declared agreement
bands (¼ of the v22 task tolerances): pelvis position 10 mm at every sample and at the end, orientation 2°.</p>{legacy_table}
{fig('cross_engine')}{fig('landing_detail')}
<p class="small">Differences are expected: MuJoCo resolves soft, penetrating box contact and soft loop constraints
(maximum v22 loop gap over these runs {_n(mj_gap)} m); Pinocchio uses rigid corner contact and PACDM closure. See
<a href="legacy_evidence/PROVENANCE.json">legacy provenance</a>.</p></section>
<section><h2>Time refinement and negative controls</h2>{refinement}{neg_table}</section>
<section><h2>What is verified</h2>
<p>The v22 CMG (78 bodies, 77 tree joints, 76 coordinates, 24 loop cuts: 16 point and 8 universal) is compiled
depth-first into a Pinocchio free-flyer model. The unchanged PACDM (<code>original_v22/pacdm.py</code>, SHA-256
{html.escape(result.get('checks', {}).get('source.unchanged_PACDM_sha256', {}).get('value', '')[:16])}…) assembles all passive
coordinates at every step and supplies the tangent map and curvature; every accepted state is then refined by the
accepted v22 <code>polish</code> (all 144 cut rows below 5e-13), because 16 of the 80 physical closure rows are
redundant. Motor slides and the pelvis twist are the only dynamic states; there is no imposed pose after release.</p>
<p>Independent checks: (1) a native Pinocchio loop oracle built from <code>RigidConstraintModel(CONTACT_3D)</code> for all
24 cut points plus native universal rows, solved with <code>constraintDynamics</code>; (2) a dense KKT solve; (3) the
accepted v22 NumPy source dynamics and KKT solver; (4) full replay of every saved trajectory (control law, actuator
filter, disturbance, integration rule, energy ledger, recomputed metrics); (5) auditor negative controls; (6) the
MuJoCo-free regeneration of the original contact reference: {rebuild_text}.</p>
<p class="small">Source package: {integrity.get('matching', '—')} of {integrity.get('manifest_entries', '—')} manifest
entries match; {len(integrity.get('missing', []))} listed files (results/ and videos/) were not in the archive.</p></section>
<section><h2>Acceptance evidence</h2>{nested}
<p class="links"><a href="results/validation.json">All numerical results</a><a href="results/mechanics.json">Mechanics</a>
<a href="results/trajectory_audits.json">Native audits</a><a href="results/case_evidence.json">Evidence replay</a>
<a href="results/reference_rebuild.json">Reference rebuild</a></p></section>
<section><h2>Limits</h2><ul>{limits}</ul></section>
<section><h2>Reproduce</h2><p><code>python run_pinocchio.py --verify-existing</code> checks every
file hash; <code>--analyze-existing</code> reruns all checks on the saved trajectories; no argument reruns everything.</p>
<p class="small">Analysis completed UTC: {html.escape(str(execution.get('completed_utc', '—')))}<br>Simulation completed UTC:
{html.escape(str(execution.get('simulation_completed_utc', '—')))}<br>Action: {html.escape(str(execution.get('action', '—')))}<br>
{html.escape(str(execution.get('platform', '')))}<br>{versions}</p></section>
<footer>Results apply to the checked model and task.</footer></main></body></html>'''
    output = root / 'report.html'
    output.write_text(content, encoding='utf-8')
    return output
