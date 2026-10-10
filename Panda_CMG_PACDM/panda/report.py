"""Scientific figures and a self-contained report from the recorded states."""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .render import phases, _position_error

INK = '#193348'
MUTED = '#657d90'
TEAL = '#008e99'
AMBER = '#df971e'
COLORS = ['#007e9a', '#c78723', '#9855ae', '#3f9163', '#bd546b', '#6575b7', '#7d7662']


def _num(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        if value == 0:
            return '0'
        return f'{value:.3e}' if abs(value) < .001 or abs(value) >= 1e5 else f'{value:.5g}'
    return str(value)


def _embed(path):
    return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def _windows(log):
    active = np.linalg.norm(log.get('wrench', np.zeros((len(log['time']), 6))), axis=1) > 1e-8
    edge = np.diff(np.r_[False, active, False].astype(int))
    return [(float(log['time'][a]), float(log['time'][min(b, len(active)-1)]))
            for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def _time_axis(ax, time, windows, timeline):
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('Time [s]')
    for start, _, _ in timeline[1:]:
        ax.axvline(start, color='#c7d3dd', lw=.7, ls=':', zorder=0)
    for a, b in windows:
        ax.axvspan(a, b, color=AMBER, alpha=.18, lw=0, zorder=0)
    ax.grid(alpha=.2, lw=.6)
    ax.spines[['top', 'right']].set_visible(False)


def make_plots(root, log):
    root = Path(root)
    t = log['time']; p = log['object_pos']; windows = _windows(log); timeline = phases(log)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                        'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlepad': 13,
                        'axes.labelcolor': MUTED, 'axes.edgecolor': '#c4d1da',
                        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig = plt.figure(figsize=(14.3, 12), layout='constrained')
    grid = fig.add_gridspec(3, 2, hspace=.08, wspace=.1)
    ax = fig.add_subplot(grid[0, 0], projection='3d')
    ax.plot(*p.T*1000, color=TEAL, lw=1.8, label='Free payload')
    ax.plot(*log['tool_pos'].T*1000, color='#b4c4cd', lw=.8, label='Measured tool')
    ax.scatter(*p[0]*1000, color=INK, s=26, label='Start')
    ax.scatter(*p[-1]*1000, marker='x', color=AMBER, s=55, label='Final payload')
    ax.set(xlabel='x [mm]', ylabel='y [mm]', zlabel='z [mm]', title='Lift, turn and transfer through contact')
    ax.view_init(25, -63)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(t, _position_error(log)*1000, color=TEAL, lw=1.2, label='Position error')
    handles = ax.lines[:1]
    if 'angle_error' in log:
        angle = np.asarray(log['angle_error'])
        angle = angle if angle.ndim == 1 else np.linalg.norm(angle, axis=1)
        twin = ax.twinx()
        twin.plot(t, np.rad2deg(angle), color=AMBER, lw=1., label='Orientation error')
        twin.set_ylabel('Orientation error [deg]', color=AMBER)
        twin.tick_params(axis='y', colors=AMBER); twin.spines[['top', 'left']].set_visible(False)
        handles += twin.lines
    ax.set(title='Tool tracking of the task-manifold reference', ylabel='Position error [mm]')
    ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper left')
    _time_axis(ax, t, windows, timeline)

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(t, log['normal_force'], color=TEAL, lw=1.2, label='Pad normal-force sum')
    for key, label, color in [('left_normal_force', 'Left finger', '#6575b7'),
                               ('right_normal_force', 'Right finger', '#9855ae')]:
        if key in log:
            ax.plot(t, log[key], color=color, lw=.9, ls='--', label=label)
    handles = ax.lines[:]
    if 'pad_contacts' in log:
        twin = ax.twinx()
        twin.step(t, log['pad_contacts'], color=AMBER, lw=.7, alpha=.65, where='post', label='Contact points')
        twin.set_ylabel('Contact count', color=AMBER); twin.tick_params(axis='y', colors=AMBER)
        twin.spines[['top', 'left']].set_visible(False)
        handles += twin.lines
    ax.set(title='Finger–payload contact during the mission', ylabel='Normal force [N]')
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc='lower left')
    _time_axis(ax, t, windows, timeline)

    ax = fig.add_subplot(grid[1, 1])
    if 'torque' in log:
        values = np.asarray(log['torque'])[:, :7]
        for i in range(values.shape[1]):
            ax.plot(t, values[:, i], color=COLORS[i], lw=.9, label=f'Joint {i+1}')
        ax.set(title='Seven actuated arm joints', ylabel='Applied motor torque [N m]')
    else:
        for i in range(min(7, log['q'].shape[1])):
            ax.plot(t, np.rad2deg(log['q'][:, i]), color=COLORS[i], lw=1., label=f'Joint {i+1}')
        ax.set(title='Seven arm joint coordinates', ylabel='Joint angle [deg]')
    ax.legend(ncol=4, frameon=False, fontsize=8, loc='lower left')
    _time_axis(ax, t, windows, timeline)

    ax = fig.add_subplot(grid[2, 0])
    ax.plot(t, p[:, 2]*1000, color=TEAL, lw=1.3, label='Payload centre')
    ax.plot(t, log['tool_pos'][:, 2]*1000, color='#6c8697', lw=1.1, label='Measured tool')
    ax.plot(t, log['target_pos'][:, 2]*1000, color=AMBER, lw=.9, ls='--', label='Tool reference')
    ax.set(title='Lift and placement at the second station', ylabel='World height [mm]')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _time_axis(ax, t, windows, timeline)

    ax = fig.add_subplot(grid[2, 1])
    q = np.asarray(log['q'])
    if q.shape[1] >= 9:
        ax.plot(t, q[:, 7]*1000, color=TEAL, lw=1.2, label='Left finger')
        ax.plot(t, q[:, 8]*1000, color='#4b6c91', lw=.9, ls='--', label='Right finger')
        ax.set_ylabel('Finger joint displacement [mm]')
    handles = ax.lines[:]
    if 'gear_error' in log:
        error = np.asarray(log['gear_error'])
        error = error if error.ndim == 1 else np.max(np.abs(error), axis=1)
        twin = ax.twinx()
        twin.plot(t, np.abs(error)*1e6, color=AMBER, lw=.9, label='Coupling residual')
        twin.set_ylabel('Native coupling residual [µm]', color=AMBER)
        twin.tick_params(axis='y', colors=AMBER); twin.spines[['top', 'left']].set_visible(False)
        handles += twin.lines
    ax.set_title('Coupled gripper coordinates and soft equality')
    ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper left')
    _time_axis(ax, t, windows, timeline)
    fig.suptitle('CMG / PACDM  ·  FRANKA PANDA DEXTEROUS TRANSFER', fontsize=19, weight='bold')
    out = root/'results/performance.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


def _check_rows(checks):
    items = checks.items() if isinstance(checks, dict) else enumerate(checks)
    rows = []
    for name, record in items:
        if isinstance(record, dict):
            passed = record.get('passed', record.get('pass'))
            label = str(record.get('name', name)).replace('_', ' ')
            value = _num(record.get('value', record.get('measured', '')))
            unit = record.get('unit', '')
            limit = record.get('limit', record.get('tolerance'))
            annotation = '' if limit is None else f" ({record.get('relation', '≤')} {_num(limit)})"
            text = f'{value} {unit}{annotation}'
        else:
            passed = record if isinstance(record, bool) else None
            label, text = str(name).replace('_', ' '), _num(record)
        state = 'PASS' if passed is True else ('FAIL' if passed is False else 'RECORDED')
        color = 'ok' if passed is True else ('fail' if passed is False else '')
        rows.append(f'<tr><th>{html.escape(label)}</th><td><span class="tag {color}">{state}</span> {html.escape(text)}</td></tr>')
    return ''.join(rows)


def _case_table(validation):
    cases = validation.get('cases', {})
    case_rows = []
    items = cases.items() if isinstance(cases, dict) else enumerate(cases)
    for name, data in items:
        if not isinstance(data, dict):
            continue
        label = str(data.get('case', name)).replace('_', ' ')
        vals = [data.get('max_lift_m'), data.get('placement_xy_error_m'), data.get('max_grasp_slip_m')]
        cells = ''.join('<td>'+('—' if x is None else f'{x*1000:.3f} mm')+'</td>' for x in vals)
        case_rows.append('<tr><th>'+html.escape(label)+'</th>'+cells+'</tr>')
    return ('<table><thead><tr><th>Case</th><td>Maximum lift</td><td>Final XY error</td><td>Transfer slip</td></tr></thead><tbody>'+''.join(case_rows)+'</tbody></table>') if case_rows else '<p class="small">The exact validation record below contains the available test cases and outcomes.</p>'


def _framework_evidence(root, log):
    source = root/'data/reference.npz'
    if not source.exists():
        return ''
    with np.load(source, allow_pickle=False) as saved:
        ref = {key: saved[key] for key in saved.files}
    if not all(key in ref for key in ('time', 'q', 'rcond', 'closure_residual')):
        return ''
    t = ref['time']
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.25), layout='constrained')
    axes[0].plot(t, np.rad2deg(ref['q'][:, 2]), color=TEAL, lw=1.7, label='PACDM reference')
    axes[0].plot(log['time'], np.rad2deg(log['q'][:, 2]), color=AMBER, lw=1., ls='--', label='Actual arm')
    axes[0].set(title='Explicit elbow redundancy', ylabel='Joint 3 [deg]')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    for key, label, color in [('closure_residual', 'Closure', TEAL),
                               ('tangent_residual', 'J N', AMBER),
                               ('acceleration_constraint_residual', 'J a + dJ/dt v', COLORS[2])]:
        if key in ref:
            axes[1].semilogy(t, np.maximum(ref[key], 1e-18), color=color, lw=1., label=label)
    axes[1].set(title='Ideal task-constraint diagnostics', ylabel='Maximum component [declared SI chart]')
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    axes[2].plot(t, ref['rcond'], color=TEAL, lw=1.4)
    axes[2].set(title='Selected passive solve conditioning', ylabel='Reciprocal condition estimate')
    for ax in axes:
        _time_axis(ax, t, [], phases(log))
    output = root/'results/framework_performance.png'
    fig.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '<section class="section"><h2>Direct PACDM evidence</h2>' \
        '<p>The reference is assembled with the original PACDM core. The augmented graph has nine physical coordinates and six virtual tool-target coordinates. Its eight independent coordinates are the six target coordinates, arm joint 3 and one finger coordinate; the seven dependent coordinates satisfy the tool pose and finger coupling.</p>' \
        f'<img class="performance" src="{_embed(output)}" alt="PACDM elbow reference and measured arm motion, ideal constraint diagnostics, and passive mapping conditioning.">' \
        '<p class="small">The residual plot uses the SI coordinates declared by the graph, including angular and translational components; it is a numerical diagnostic, not a length error. The reference satisfies ideal task constraints. Actual servo-controlled motion has the separately measured tracking errors shown above.</p></section>'


def build_report(root: Path):
    root = Path(root); results = root/'results'
    with np.load(results/'nominal.npz', allow_pickle=False) as saved:
        log = {key: saved[key] for key in saved.files}
    validation = json.loads((results/'validation.json').read_text()) if (results/'validation.json').exists() else {}
    metrics = json.loads((results/'nominal.json').read_text()) if (results/'nominal.json').exists() else {}
    plot = make_plots(root, log)
    max_lift = metrics.get('max_lift_m', float(np.max(log['object_pos'][:, 2])-log['object_pos'][0, 2]))*1000
    placement = metrics.get('placement_xy_error_m')
    # A goal-relative metric must come from the simulator's declared task goal;
    # do not substitute distance to an assumed or visually estimated dock.
    cards = [('Maximum lift', f'{max_lift:.3f}', 'mm'),
             ('Final XY error', '—' if placement is None else f'{placement*1000:.3f}', 'mm'),
             ('Peak pad normal', f'{np.max(log["normal_force"]):.3f}', 'N'),
             ('Tool RMS error', f'{np.sqrt(np.mean(_position_error(log)**2))*1000:.3f}', 'mm')]
    card_html = ''.join(f'<div class="metric"><span>{name}</span><strong>{value}<small>{unit}</small></strong></div>' for name, value, unit in cards)
    status = validation.get('passed')
    status_text = 'Saved validation: PASS' if status is True else 'Saved validation: FAIL' if status is False else 'Validation not yet recorded'
    passed_count = validation.get('passed_count', validation.get('passed_checks'))
    check_count = validation.get('check_count', validation.get('total_checks'))
    if passed_count is not None and check_count is not None:
        status_text += f" · {passed_count}/{check_count} checks"
    status_html = '<span class="tag '+('ok' if status is True else 'fail' if status is False else '')+'">'+status_text+'</span>'
    poster = results/'poster.png'
    poster_html = f'<img class="poster" src="{_embed(poster)}" alt="Recorded MuJoCo Panda arm carrying the free payload, with a simultaneous gripper close-up.">' if poster.exists() else ''
    timeline = phases(log)
    phase_html = ''.join(f'<article><span>{start:g}–{end:g} s</span><b>{html.escape(label)}</b></article>' for start, end, label in timeline)
    windows = _windows(log)
    event_text = 'Applied payload disturbance windows: '+', '.join(f'{a:.2f}–{b:.2f} s' for a, b in windows)+'. These intervals are shaded amber in the plots.' if windows else 'No payload disturbance is present in this saved log.'
    template = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CMG / PACDM — Franka Panda dexterous transfer</title><style>
    :root{--ink:#193348;--muted:#657d90;--teal:#008e99}*{box-sizing:border-box}body{margin:0;background:#f1f5f8;color:var(--ink);font:16px/1.62 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1170px;margin:auto;padding:48px 30px 64px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.15em;color:var(--teal);text-transform:uppercase}h1{font-size:clamp(31px,5vw,53px);line-height:1.07;letter-spacing:-.035em;margin:13px 0 19px}h2{font-size:24px;letter-spacing:-.02em;margin:0 0 17px}p{margin:0 0 17px}.lead{max-width:920px;font-size:18px;color:var(--muted)}.tag{display:inline-block;font-size:11px;font-weight:800;border-radius:6px;background:#e4ecf2;padding:4px 9px}.ok{background:#d9f1e6;color:#216444}.fail{background:#f9dfde;color:#a32a29}.poster{display:block;width:100%;margin:26px 0 0;border-radius:13px;box-shadow:0 12px 38px #14283c20}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:26px 0}.metric{background:white;border:1px solid #dfe8ef;border-radius:12px;padding:22px}.metric span{display:block;font-size:11px;color:var(--muted);font-weight:750;text-transform:uppercase}.metric strong{display:block;font-size:32px;margin-top:7px;letter-spacing:-.02em}.metric small{font-size:12px;color:var(--muted);margin-left:6px}.section{background:white;border:1px solid #dfe8ef;border-radius:14px;padding:30px;margin:22px 0}.phases{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}.phases article{border-top:3px solid #58bfc5;padding-top:10px}.phases span{font-size:12px;font-weight:750;color:var(--teal)}.phases b{display:block;font-size:15px}.small{font-size:13px;color:var(--muted)}.performance{display:block;width:100%}.scope{display:grid;grid-template-columns:1fr 1fr;gap:24px}.scope article{border-left:3px solid #d3e7ec;padding-left:18px}.scope b{display:block;font-size:16px;margin-bottom:6px}.scope p{font-size:14px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 11px;border-bottom:1px solid #e4ecf2;text-align:left;vertical-align:top}th{font-weight:500;color:var(--muted);overflow-wrap:anywhere}td{font-variant-numeric:tabular-nums}thead th,thead td{font-size:11px;font-weight:750;text-transform:uppercase}details{border-top:1px solid #e2eaf0;padding-top:14px;margin-top:21px}summary{cursor:pointer;font-weight:650;color:var(--teal)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:17px;border-radius:8px;font-size:12px}.footer{font-size:12px;color:var(--muted)}a{color:var(--teal)}@media(max-width:760px){main{padding:28px 16px}.metrics,.phases{grid-template-columns:repeat(2,1fr)}.section{padding:20px}.metric{padding:16px}.scope{grid-template-columns:1fr}table{font-size:11px}th,td{padding:8px 5px}}@media print{body{background:white}main{padding:0}.section{break-inside:avoid}details{display:none}}
    </style></head><body><main><div class="eyebrow">CMG / PACDM · Franka Panda benchmark</div>
    <h1>CMG-based PACDM for the Franka Panda</h1>
    <p class="lead">A fixed-base, seven-joint arm grasps a free cartridge, lifts it over a barrier, reorients it, pauses for tilted inspection and releases it at a second station. The trajectory combines a six-dimensional tool-pose task, an elbow redundancy coordinate, coupled fingers and changing physical contact.</p>
    {{STATUS}}{{POSTER}}<div class="metrics">{{CARDS}}</div>
    <p class="small">Metrics are computed from the saved nominal simulation. Pad force is the sum of finger–payload contact normal forces. The payload remains a free rigid body during grasping, transport and release.</p>
    <section class="section"><h2>The {{DURATION}}-second mission</h2><div class="phases">{{PHASES}}</div><p class="small" style="margin-top:22px">{{EVENTS}}</p><p class="small">The arm uses the source position servos with Pinocchio inverse-dynamics feedforward encoded as position-command offsets. MuJoCo integrates the resulting arm motion and native contact.</p></section>
    <section class="section"><h2>Measured motion, torque and contact</h2><p class="small">All traces come from the saved simulation. Dotted lines separate task phases. The native finger-coupling residual measures MuJoCo's compliant equality and is distinct from the ideal PACDM assembly tolerance.</p><img class="performance" src="{{PLOT}}" alt="Measured payload trajectory, tool tracking errors, finger contact forces, arm torques, lift height and finger coupling residual."></section>
    <section class="section"><h2>Mechanism, task and validation</h2><div class="scope"><article><b>A redundant serial arm</b><p>The Panda arm is a serial kinematic tree. Its seven revolute joints allow a six-dimensional tool-pose task with one remaining arm coordinate. Joint 3 is scheduled independently during the inspection segment.</p></article><article><b>Task constraints in the same framework</b><p>CMG describes the bodies, joint frames, inertias and finger coupling. PACDM assembles the tool-pose task using an explicit redundancy coordinate. The tool relation is a task constraint while the physical Panda graph remains a tree.</p></article><article><b>Real contact in the numerical model</b><p>Native MuJoCo contact provides support, finger forces, slip and release. The payload remains a free body as contact constraints change during grasp, transport and release.</p></article><article><b>Independent rigid-body mechanics</b><p>Pinocchio evaluates the declared arm-and-gripper tree. Numerical checks compare rigid-body mechanics against MuJoCo, with model armature and passive terms handled explicitly. The recorded comparisons quantify agreement in mass, bias and constrained dynamics.</p></article></div></section>
    {{FRAMEWORK}}
    <section class="section"><h2>Cases and acceptance checks</h2>{{CASES}}<p class="small" style="margin-top:18px">Pass labels refer to the saved automated checks and declared numerical thresholds. The open-hand negative control is accepted only when its test detects the intended task failure.</p>{{ABLATION}}<details><summary>Individual acceptance checks</summary><table><tbody>{{ROWS}}</tbody></table></details><details><summary>Exact machine-readable validation record</summary><pre>{{JSON}}</pre></details></section>
    <section class="section"><h2>Reproducibility and scope</h2><p>The repository identifies the pinned MuJoCo Menagerie model, license and file hashes. The upstream files are preserved alongside the declared benchmark modifications. The PACDM core is unchanged; the Panda graph and task adapters are separate modules.</p><p>The payload, support fixtures, controller, contact settings and disturbances are declared benchmark choices. The prescribed route places the cartridge within the socket clearance; the recorded tests quantify mechanics, contact and placement in simulation.</p><p class="small">The video replays recorded native coordinates. Its inset shows the same physical state from a second camera; a cyan trail follows the recorded payload origin. The embedded plots and complete validation record can be viewed offline.</p></section>
    <p class="footer">Report compiled from the simulation and validation records by panda/report.py.</p></main></body></html>'''
    replacements = {'STATUS': status_html, 'POSTER': poster_html, 'CARDS': card_html,
                    'DURATION': f'{timeline[-1][1]:g}', 'PHASES': phase_html,
                    'EVENTS': html.escape(event_text), 'PLOT': _embed(plot),
                    'CASES': _case_table(validation), 'FRAMEWORK': _framework_evidence(root, log),
                    'ABLATION': '<p class="small">'+html.escape(str(validation['ablation_scope']))+'</p>' if validation.get('ablation_scope') else '',
                    'ROWS': _check_rows(validation.get('checks', {})),
                    'JSON': html.escape(json.dumps(validation, indent=2))}
    document = re.sub(r'\{\{(STATUS|POSTER|CARDS|DURATION|PHASES|EVENTS|PLOT|CASES|FRAMEWORK|ABLATION|ROWS|JSON)\}\}',
                      lambda m: replacements[m.group(1)], template)
    output = results/'report.html'
    output.write_text(document, encoding='utf-8')
    return output
