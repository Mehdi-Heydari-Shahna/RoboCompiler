"""Offline report generated from executed numerical results, never canned scores."""
import hashlib
import html
import json
from pathlib import Path


LABELS = {'coarse': 'Coarse · 4 ms', 'nominal': 'Nominal · 2 ms',
          'fine': 'Fine · 1 ms', 'heavy_payload': 'Heavy payload · 14 kg',
          'no_feedforward': 'Feedforward disabled'}
COLORS = {'coarse': '#22a89a', 'nominal': '#2176d2', 'fine': '#d49a24',
          'heavy_payload': '#9861ba', 'no_feedforward': '#6a7180'}


def _read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value, digits=4):
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
        rows.append('<tr><td class="gate-name">' + html.escape(name) + '</td><td>'
                    + _number(check.get('value'), 7) + '</td><td>'
                    + html.escape(check.get('relation', '<=')) + ' '
                    + _number(check.get('limit'), 7) + '</td><td>'
                    + html.escape(check.get('unit', check.get('units', ''))) + '</td><td class="'
                    + ('pass' if good else 'fail') + '">' + ('PASS' if good else 'FAIL') + '</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Check</th><th>Observed</th>'
            '<th>Acceptance</th><th>Unit</th><th>Result</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


def _performance(root, result):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#25364b',
                         'xtick.color': '#4d5d72', 'ytick.color': '#4d5d72'}):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
        for name in LABELS:
            path = root/'results'/f'{name}.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                t = data['time']; style = '--' if name == 'no_feedforward' else '-'
                args = dict(color=COLORS[name], label=LABELS[name], linewidth=1.35,
                            linestyle=style, alpha=.9)
                axes[0, 0].plot(t, data['pose_error_m']*1000, **args)
                axes[0, 1].plot(t, np.rad2deg(data['angle_error_rad']), **args)
                axes[1, 0].plot(t, np.max(np.abs(data['force']), axis=1), **args)
        for ax, title, unit in [(axes[0, 0], 'Platform position error', 'Error (mm)'),
                                (axes[0, 1], 'Platform orientation error', 'Geodesic angle (deg)'),
                                (axes[1, 0], 'Largest absolute strut force', 'Force (N)')]:
            ax.set_title(title, loc='left', pad=11)
            ax.set_xlabel('Simulation time (s)'); ax.set_ylabel(unit)
            ax.set_xlim(0, 22); ax.grid(alpha=.16)
            for start, end in [(5.5, 5.85), (11.2, 11.55), (14.2, 14.6)]:
                ax.axvspan(start, end, color='#d95d39', alpha=.09, linewidth=0)
        ref = result.get('refinement', {})
        x = np.arange(2); width = .34
        pos = [ref.get(k, {}).get('max_position_difference_m', np.nan)/.0005
               for k in ['4ms_vs_2ms', '2ms_vs_1ms']]
        ang = [ref.get(k, {}).get('max_orientation_difference_deg', np.nan)/.03
               for k in ['4ms_vs_2ms', '2ms_vs_1ms']]
        ax = axes[1, 1]
        ax.bar(x-width/2, pos, width, color='#2176d2', label='Position / 0.5 mm')
        ax.bar(x+width/2, ang, width, color='#9861ba', label='Orientation / 0.03°')
        ax.set_xticks(x, ['4 ms vs 2 ms', '2 ms vs 1 ms'])
        ax.axhline(1., color='#566678', linestyle=':', linewidth=1)
        ax.set_title('Complete-mission timestep refinement', loc='left', pad=11)
        ax.set_ylabel('Difference / fine-pair acceptance threshold')
        ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
                   bbox_to_anchor=(.5, .955), fontsize=10)
        fig.suptitle('Stewart platform · executed Pinocchio / PACDM simulation',
                     x=.075, ha='left', fontsize=17, weight='bold', color='#15273c')
        fig.text(.075, .025, 'Shaded bands: applied disturbances. The ablation is exempt from tracking gates. '
                 'Refinement reuses one 20 ms reference; the threshold applies to 2 ms vs 1 ms.',
                 fontsize=9, color='#566678')
        fig.subplots_adjust(top=.85, bottom=.11, left=.075, right=.975, hspace=.39, wspace=.28)
        path = root/'results/performance.png'
        fig.savefig(path, dpi=170, facecolor='white'); plt.close(fig)
    return path


def _video(root, execution):
    r = root/'results'
    if execution.get('video_requested') is False:
        return '<p class="notice">Video was skipped for this execution. Run <code>python run_pinocchio.py --render-only</code> to render the saved nominal trajectory.</p>'
    meta = _read(r/'render_metadata.json', {})
    checks = [('nominal.npz', 'source_result_sha256'),
              ('Stewart_Pinocchio.mp4', 'video_sha256')]
    current = bool(meta) and all((r/name).is_file() and _hash(r/name) == meta.get(key)
                                 for name, key in checks)
    cmg_path = root/meta.get('source_cmg_file', 'data/stewart.cmg.json')
    current = current and cmg_path.is_file() and _hash(cmg_path) == meta.get('source_cmg_sha256')
    if not current:
        return '<p class="notice">A video matching the current saved trajectory is not available. Numerical results remain inspectable below. Run <code>python run_pinocchio.py --render-only</code> to create it.</p>'
    poster = ' poster="results/poster.png"' if (r/'poster.png').exists() else ''
    return ('<video controls preload="metadata"' + poster + '><source src="results/Stewart_Pinocchio.mp4" '
            'type="video/mp4">Open the MP4 using the link below.</video><p class="caption">'
            + _number(meta.get('video_duration_s')) + ' s · ' + _number(meta.get('fps'))
            + ' fps · ' + _number(meta.get('width'), 5) + ' × ' + _number(meta.get('height'), 5)
            + '. CPU rendering of recorded nominal states; body transforms come from Pinocchio forward kinematics. '
            'Video and source hashes match. <a href="results/Stewart_Pinocchio.mp4">Open MP4</a> · '
            '<a href="results/render_metadata.json">Render provenance</a></p>')


def build_report(root):
    """Write report.html and results/performance.png using the current result files."""
    root = Path(root); result = _read(root/'results/validation.json', {})
    execution = _read(root/'results/execution.json', {})
    passed = result.get('passed', False)
    status = 'PASS' if passed else 'INCOMPLETE / FAIL'
    cases = result.get('cases', {})
    nominal = cases.get('nominal', {})
    rms_mm = nominal.get('rms_position_error_m')
    peak_mm = nominal.get('max_position_error_m')
    rms_mm = rms_mm*1000 if rms_mm is not None else None
    peak_mm = peak_mm*1000 if peak_mm is not None else None
    figure = _performance(root, result) if cases else None
    rows = []
    for name, label in LABELS.items():
        c = cases.get(name)
        if not c:
            continue
        values = [c['timestep_s']*1000, c['payload_mass_kg'], c['rms_position_error_m']*1000,
                  c['max_position_error_m']*1000, c['max_orientation_error_deg'],
                  c['max_actuator_force_N'], c['docking_max_position_error_m']*1000,
                  c['max_closure_error_m']]
        rows.append('<tr><td>' + html.escape(label) + '</td>'
                    + ''.join('<td>' + _number(v) + '</td>' for v in values) + '</tr>')
    case_table = ('<div class="scroll"><table><thead><tr><th>Case</th><th>dt (ms)</th><th>Payload (kg)</th>'
                  '<th>RMS position (mm)</th><th>Peak position (mm)</th><th>Peak orientation (°)</th>'
                  '<th>Peak force (N)</th><th>Final 1 s position (mm)</th><th>Closure (m)</th>'
                  '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
    nested = '<details><summary>Independent mechanics gates</summary>' + _gates(result.get('mechanics', {}).get('checks', {})) + '</details>'
    for name, audit in result.get('trajectory_audits', {}).items():
        nested += ('<details><summary>Native constraint audit · ' + html.escape(LABELS.get(name, name))
                   + ' · ' + _number(audit.get('details', {}).get('samples')) + ' sampled states</summary>'
                   + _gates(audit.get('checks', {})) + '</details>')
    negative = result.get('auditor_negative_controls', {})
    if negative:
        negative_rows = ''.join('<tr><td>' + html.escape(name) + '</td><td>'
                                + html.escape(check.get('expected', '')) + '</td><td>'
                                + html.escape(check.get('observed', '')) + '</td><td class="'
                                + ('pass' if check.get('passed') else 'fail') + '">'
                                + ('PASS' if check.get('passed') else 'FAIL') + '</td></tr>'
                                for name, check in negative.get('checks', {}).items())
        nested += ('<details><summary>Auditor negative controls</summary><p class="small">'
                   'An unaltered fixture must be accepted; intentionally corrupted data must be rejected.</p>'
                   '<div class="scroll"><table><thead><tr><th>Fixture</th><th>Expected</th><th>Observed</th>'
                   '<th>Result</th></tr></thead><tbody>' + negative_rows + '</tbody></table></div></details>')
    evidence = result.get('case_evidence', _read(root/'results/case_evidence.json', {}))
    for name, audit in evidence.items():
        evidence_rows = []
        for key, check in audit.get('checks', {}).items():
            if 'max_absolute_error' in check:
                observed = _number(check['max_absolute_error'], 6)
                requirement = ('atol ' + _number(check.get('absolute_tolerance'))
                               + '; rtol ' + _number(check.get('relative_tolerance')))
            else:
                observed = 'consistent' if check.get('passed') else 'inconsistent'
                requirement = 'exact / structural check'
            evidence_rows.append('<tr><td class="gate-name">' + html.escape(key) + '</td><td>'
                                 + observed + '</td><td>' + requirement + '</td><td class="'
                                 + ('pass' if check.get('passed') else 'fail') + '">'
                                 + ('PASS' if check.get('passed') else 'FAIL') + '</td></tr>')
        nested += ('<details><summary>Saved evidence consistency · ' + html.escape(LABELS.get(name, name))
                   + '</summary><p class="small">Recorded states, scenario/model, control law, disturbances '
                   'and recomputed summary metrics must agree. Numerical comparisons combine absolute and relative tolerance.</p>'
                   '<div class="scroll"><table><thead><tr><th>Check</th><th>Maximum absolute error / status</th>'
                   '<th>Requirement</th><th>Result</th></tr></thead><tbody>' + ''.join(evidence_rows)
                   + '</tbody></table></div></details>')
    versions = ', '.join(html.escape(k) + ' ' + html.escape(str(v))
                         for k, v in execution.get('versions', {}).items())
    limits = ''.join('<li>' + html.escape(s) + '</li>' for s in result.get('limits', []))
    comparison = result.get('legacy_mujoco_comparison', {})
    legacy = ('Compared with the supplied historical MuJoCo nominal recording: maximum position difference '
              + _number(comparison.get('max_position_difference_m', 0)*1000) + ' mm; orientation difference '
              + _number(comparison.get('max_orientation_difference_deg')) + '°. MuJoCo was not rerun for this release.') if comparison else 'Historical comparison has not completed.'
    completed = html.escape(execution.get('completed_utc', 'No completed execution recorded.'))
    simulated = html.escape(execution.get('simulation_completed_utc', execution.get('completed_utc', 'unavailable')))
    content = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stewart · Pinocchio / PACDM validation</title>
<style>
:root{{--ink:#172a41;--muted:#58697d;--blue:#1d6eb9;--line:#dbe4ed;--bg:#f3f6fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,Segoe UI,sans-serif}}
header{{background:#142c46;color:#fff;padding:46px max(25px,calc((100% - 1200px)/2)) 38px}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#8fc7ee;font-weight:700}}
h1{{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:12px 0 16px;letter-spacing:-.025em}}header p{{max-width:920px;color:#ccdaea;margin:0}}
main{{max-width:1250px;margin:0 auto;padding:26px}}section{{background:white;border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:22px}}
h2{{font-size:23px;margin:0 0 14px;letter-spacing:-.02em}}h3{{font-size:17px;margin:20px 0 8px}}p{{margin:10px 0}}a{{color:var(--blue);text-underline-offset:3px}}
.status{{display:inline-block;border-radius:20px;padding:5px 13px;font-weight:750;background:{'#d9f6e9' if passed else '#ffe6de'};color:{'#126247' if passed else '#9c3522'};margin-right:8px}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:23px 0 0}}.tile{{background:#f3f7fb;padding:16px;border-radius:8px}}.tile strong{{display:block;font-size:25px;font-weight:750;color:#1a5c91}}.tile span{{font-size:13px;color:var(--muted)}}
video{{width:100%;display:block;background:#0e1b2c;border-radius:8px}}img{{max-width:100%;height:auto;display:block}}.caption,.small{{font-size:13px;color:var(--muted)}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{background:#eef4f9;color:#314f6a;text-align:left}}th,td{{padding:10px 12px;border-bottom:1px solid #e4ebf2;vertical-align:top}}td{{font-variant-numeric:tabular-nums}}td:first-child{{min-width:140px}}.gate-name{{overflow-wrap:anywhere}}.pass{{color:#137452;font-weight:700}}.fail{{color:#b13c27;font-weight:700}}
details{{border:1px solid var(--line);border-radius:7px;margin:10px 0;padding:0 13px}}summary{{cursor:pointer;font-weight:650;padding:13px 0}}details table{{margin-bottom:14px}}code{{font-size:13px;background:#edf2f7;padding:2px 5px;border-radius:4px}}.notice{{background:#fff3d9;padding:18px;border-radius:7px}}
ul{{padding-left:21px}}li{{margin:8px 0}}.links{{display:flex;gap:18px;flex-wrap:wrap}}footer{{font-size:12px;color:var(--muted);padding:0 6px 20px;overflow-wrap:anywhere}}
@media(max-width:760px){{main{{padding:15px}}section{{padding:18px}}.tiles{{grid-template-columns:repeat(2,1fr)}}header{{padding:30px 22px}}}}
</style></head><body>
<header><div class="eyebrow">Executed simulation verification · declared rigid model</div><h1>Stewart platform<br>Pinocchio + PACDM</h1>
<p>Force-driven six-axis motion, independent native constraint audits, and a video built from the saved simulation states. The supplied PACDM source is retained byte-for-byte.</p></header>
<main><section><h2><span class="status">{status}</span> {_number(result.get('passed_count'))} / {_number(result.get('check_count'))} aggregate gates</h2>
<p>{html.escape(result.get('claim', result.get('status', 'Aggregate validation has not completed.')))}</p>
<div class="tiles"><div class="tile"><strong>{_number(rms_mm)} mm</strong><span>Nominal RMS position error</span></div>
<div class="tile"><strong>{_number(peak_mm)} mm</strong><span>Nominal peak position error</span></div>
<div class="tile"><strong>{_number(nominal.get('max_orientation_error_deg'))}°</strong><span>Nominal peak orientation error</span></div>
<div class="tile"><strong>{_number(nominal.get('max_actuator_force_N'))} N</strong><span>Nominal peak actuator force</span></div></div></section>
<section><h2>Recorded full mission</h2>{_video(root, execution)}
<p class="small">The video illustrates the nominal 22 s run. It is CPU-rendered playback; acceptance is based on the recorded numerical results and independent checks below.</p></section>
<section><h2>Five executed cases</h2>{case_table}<p class="small">All cases cover the same 22 s mission and disturbances. Nominal payload is 8 kg. The 14 kg case retains nominal 8 kg feedforward to test unmodeled load. The feedforward ablation retains feedback and is exempt from the main tracking tolerances. Docking values are maximum errors over the final 1 s.</p>
{('<a href="results/performance.png"><img src="results/performance.png" alt="Executed tracking error, orientation error, actuator forces and timestep refinement across five cases"></a>' if figure else '')}</section>
<section><h2>What is being verified</h2>
<p>The CMG defines an idealized six-UPS Stewart mechanism: 24 physical tree coordinates, 18 point-closure equations, and six independent prismatic lengths. The unchanged PACDM implementation assembles passive coordinates and supplies their tangent mapping and curvature.</p>
<p>Pinocchio supplies tree mass, nonlinear bias and inverse dynamics. The mission integrates the six actuator lengths under computed forces with semi-implicit Euler and PACDM assembly at every step. There is no imposed platform pose after initialization.</p>
<p>A separately constructed set of six native Pinocchio <code>CONTACT_3D</code> constraints checks accelerations using <code>constraintDynamics</code> at sampled states. This is an independent closure implementation sharing the same Pinocchio tree mass model; it is not a second full mission integrator.</p>
<div class="scroll"><table><thead><tr><th>Time</th><th>Mission</th><th>Applied disturbance</th></tr></thead><tbody>
<tr><td>0–2 s</td><td>Initial hold</td><td>None</td></tr><tr><td>2–9 s</td><td>Helical inspection sweep</td><td>80 N x-force pulse at 5.5–5.85 s</td></tr>
<tr><td>9–17 s</td><td>Figure-eight motion</td><td>120 N y-force pulse at 11.2–11.55 s; 12 N·m x-moment pulse at 14.2–14.6 s</td></tr>
<tr><td>17–20 s</td><td>Precision docking</td><td>None</td></tr><tr><td>20–22 s</td><td>Dock hold</td><td>None</td></tr></tbody></table></div>
<p class="small">Disturbances are half-sine pulses, expressed in world axes at the platform origin. All timestep cases share the same reference sampled at 20 ms.</p></section>
<section><h2>Acceptance evidence</h2><p>Aggregate gates include source integrity, full mission coverage, mechanism closure, rank and conditioning, dynamics residuals, tracking, force and joint limits, timestep refinement, feedforward ablation, and historical trajectory agreement.</p>
<details><summary>All {_number(result.get('check_count'))} aggregate gates</summary>{_gates(result.get('checks', {}))}</details>{nested}
<p class="links"><a href="results/validation.json">All numerical results</a><a href="results/mechanics.json">Mechanics samples</a><a href="results/trajectory_audits.json">Trajectory audits</a><a href="METHODS.md">Detailed method</a></p></section>
<section><h2>Historical comparison and limits</h2><p>{legacy}</p><p>This is the idealized mechanism in the supplied package. It is not a CAD-specific Stewart model or a hardware-identified plant.</p><ul>{limits}</ul></section>
<section><h2>Reproduce and inspect</h2><p>Open <a href="README.md">README.md</a> for Miniforge setup and commands. The delivered report, MP4, plots, JSON and NPZ files work offline; rerunning uses the supplied environment.</p>
<p class="links"><a href="results/execution.json">Execution record</a><a href="data/stewart.cmg.json">CMG model</a><a href="vendor/pacdm_original.py">Unchanged PACDM source</a><a href="SHA256SUMS.json">File checksums</a></p>
<p class="small">Most recent analysis completed UTC: {completed}<br>Full simulation completed UTC: {simulated}<br>Action: {html.escape(execution.get('action', 'unavailable'))}<br>{html.escape(execution.get('platform', ''))}<br>{versions}</p></section>
<footer>Reproducibility status: Linux runtime executed; Windows / Miniforge launcher supplied but not executed here. Confidence applies to the checked model and tested mission, not to unmeasured hardware behavior.</footer></main></body></html>'''
    output = root/'report.html'
    output.write_text(content, encoding='utf-8')
    return output
