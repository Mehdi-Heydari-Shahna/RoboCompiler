"""Build an evidence-led HTML report and a static scientific summary figure.

The report reads final numerical artifacts. It never substitutes a video or a
successful plot for a failed acceptance gate. Run from the project root with
``python -m go2.pin_report`` after ``pin_acceptance.aggregate``.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np


LABELS = {
    'nominal': 'Nominal', 'fine': 'Half step', 'finer': 'Quarter step',
    'low_friction': 'Lower friction', 'payload': '+1.5 kg payload',
    'strong_push': '25% stronger pushes', 'no_actuation': 'No actuation',
}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _esc(value):
    return html.escape(str(value), quote=True)


def _num(value, digits=3):
    if value is None:
        return '—'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not np.isfinite(number):
        return _esc(value)
    return f'{number:.{digits}g}'


def _badge(passed, label=None):
    return f'<span class="badge {"pass" if passed else "fail"}">{_esc(label or ("PASS" if passed else "FAIL"))}</span>'


def _rows(checks):
    if isinstance(checks, dict):
        return [(name, value) for name, value in checks.items()]
    return [(value.get('name', str(i)), value) for i, value in enumerate(checks)]


def _checks_table(checks):
    rows = []
    for name, check in _rows(checks):
        if not isinstance(check, dict):
            continue
        measured, limit = check.get('value'), check.get('limit')
        measured = _num(measured) if np.isscalar(measured) or measured is None else _esc(measured)
        limit = _num(limit) if np.isscalar(limit) or limit is None else _esc(limit)
        relation = _esc(check.get('relation', ''))
        rows.append(f'<tr><td>{_esc(name)}</td><td>{measured}</td><td>{relation} {limit}</td><td>{_badge(check.get("passed", False))}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>Check</th><th>Measured</th><th>Acceptance</th><th>Result</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def _figure(out, cases, validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    series = {}
    for name in ('nominal', 'fine', 'finer'):
        with np.load(out / f'{name}.npz', allow_pickle=False) as archive:
            series[name] = {key: archive[key].copy() for key in ('time', 'q', 'q_ref', 'body_error')}
    actual, ref, time = series['nominal']['q'], series['nominal']['q_ref'], series['nominal']['time']
    colors = {'nominal': '#1567ad', 'fine': '#df8837', 'finer': '#17877d'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .16,
                         'figure.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff'}):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(ref[:, 0], ref[:, 1], color='#303f50', ls='--', lw=2.2, label='PACDM reference')
        ax.plot(actual[:, 0], actual[:, 1], color=colors['nominal'], lw=1.6, label='Simulated base')
        ax.scatter(actual[0, 0], actual[0, 1], s=42, c='#17877d', marker='o', zorder=3, label='Start')
        ax.scatter(actual[-1, 0], actual[-1, 1], s=60, c='#db7030', marker='x', zorder=3, label='Finish')
        ax.set(title='A   Base path · nominal', xlabel='World x (m)', ylabel='World y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='upper left', fontsize=9, ncol=2)

        ax = axes[0, 1]
        ax.plot(time, ref[:, 2], color='#303f50', ls='--', lw=2, label='PACDM reference')
        ax.plot(time, actual[:, 2], color=colors['nominal'], lw=1.4, label='Simulated base')
        ax.set(title='B   Base height · nominal', xlabel='Time (s)', ylabel='World z (m)', xlim=(time[0], time[-1]))
        ax.legend(loc='best', fontsize=9)

        ax = axes[1, 0]
        for name in ('nominal', 'fine', 'finer'):
            d = series[name]
            ax.plot(d['time'], 1000*d['body_error'], lw=1.3, color=colors[name],
                    label=f"dt = {1000*cases[name]['dt_s']:g} ms")
        peak_limit = validation.get('checks', {}).get('nominal.peak_body_error_m', {}).get('limit')
        if peak_limit is not None:
            peak = max(float(np.max(series[name]['body_error'])) for name in series)
            if peak_limit <= 1.25*peak:
                ax.axhline(1000*peak_limit, color='#a85050', lw=1, ls=':', label='Peak-error limit')
            else:
                ax.text(.02, .94, f'Declared peak limit: {1000*peak_limit:g} mm (above plotted range)',
                        transform=ax.transAxes, va='top', color='#8c4149', fontsize=9)
                ax.set_ylim(-.5, 1000*peak*1.3)
        ax.set(title='C   Base position error', xlabel='Time (s)', ylabel='3D position error (mm)', xlim=(time[0], time[-1]))
        ax.legend(loc='upper left', bbox_to_anchor=(.015, .87), fontsize=9, ncol=3, borderaxespad=0)

        ax = axes[1, 1]
        for coarse, fine, color in [('nominal', 'fine', '#815fa0'), ('fine', 'finer', '#17877d')]:
            a, b = series[coarse], series[fine]
            if a['time'].shape != b['time'].shape or not np.allclose(a['time'], b['time'], rtol=0, atol=1e-12):
                raise ValueError('Refinement traces have different timestamps; refusing an invalid plot.')
            difference = np.linalg.norm(a['q'][:, :3]-b['q'][:, :3], axis=1)
            label = f"{1000*cases[coarse]['dt_s']:g} vs {1000*cases[fine]['dt_s']:g} ms"
            # Zero at the common initial condition is omitted from logarithmic axes.
            shown = np.where(difference > 0, difference*1000, np.nan)
            ax.semilogy(a['time'], shown, lw=1.3, color=color, label=label)
        ax.set(title='D   Timestep sensitivity', xlabel='Time (s)', ylabel='Base difference (mm, log scale)', xlim=(time[0], time[-1]))
        ax.legend(loc='best', fontsize=9)
        fig.suptitle('Go2 CMG / PACDM · Pinocchio simulation evidence', fontsize=17, weight='bold')
        fig.savefig(out / 'summary.png', dpi=170, bbox_inches='tight')
        plt.close(fig)


def build_report(root):
    """Write ``results_pinocchio/report.html`` and ``summary.png``; return paths."""
    root = Path(root).resolve()
    out = root / 'results_pinocchio'
    validation = _read(out / 'validation.json')
    cases = validation['cases']
    checks = validation['checks']
    failures = {key: value for key, value in checks.items() if not value.get('passed', False)}
    _figure(out, cases, validation)
    chart_data = base64.b64encode((out / 'summary.png').read_bytes()).decode('ascii')

    positive = [name for name in cases if name != 'no_actuation']
    clearances = {name: _read(out / f'{name}_clearance.json') for name in positive}
    dense_clearance = all(value.get('all_integration_states_checked', False) for value in clearances.values())
    nominal = cases['nominal']
    nominal_clearance = clearances['nominal']
    status = bool(validation['passed']) and not failures
    status_word = 'PASS' if status else 'FAIL'
    banner = ('All declared aggregate acceptance gates passed.' if status
              else 'Acceptance failed. The evidence below does not establish the declared simulation outcome.')
    failure_block = '' if not failures else '<section class="failure-panel"><h2>Failed acceptance gates</h2>' + _checks_table(failures) + '</section>'
    if not dense_clearance:
        failure_block += '<section class="failure-panel"><h2>Incomplete geometric sampling</h2><p>At least one clearance audit does not cover every recorded integration state. Do not interpret its clearance as a dense-step check.</p></section>'

    case_rows = []
    for name in positive:
        case, clearance = cases[name], clearances[name]
        local = [v for k, v in checks.items() if k.startswith(name + '.')]
        passed = bool(local) and all(v['passed'] for v in local)
        case_rows.append('<tr>' + ''.join([
            f'<td><strong>{_esc(LABELS.get(name, name))}</strong><small>μ = {_num(case["friction"])}; payload {_num(case["point_payload_kg"])} kg; push ×{_num(case["push_scale"])}</small></td>',
            f'<td>{_num(1000*case["dt_s"])} ms</td>',
            f'<td>{_num(case["simulated_s"])} s</td>',
            f'<td>{_num(1000*case["final_position_error_m"])} mm</td>',
            f'<td>{_num(1000*case["rms_body_error_m"])} mm</td>',
            f'<td>{_num(1000*case["peak_body_error_m"])} mm</td>',
            f'<td>{_num(1000*clearance["minimum_clearance_m"])} mm</td>',
            f'<td>{_badge(passed)}</td>',
        ]) + '</tr>')
    limits = {key: checks.get('nominal.'+key, {}).get('limit') for key in ('final_position_error_m', 'rms_body_error_m', 'peak_body_error_m')}
    limit_cells = ''.join(f'<td>≤ {_num(1000*limits[key])} mm</td>' if limits[key] is not None else '<td>—</td>' for key in limits)
    case_rows.append('<tr class="limits"><td>Declared tolerance</td><td>—</td><td>26 s</td>' + limit_cells + '<td>No forbidden intersection*</td><td>All gates</td></tr>')

    refinement_rows = []
    for name, data in validation.get('refinement', {}).items():
        if not isinstance(data, dict):
            continue
        refinement_rows.append('<tr>' + ''.join([
            f'<td>{_esc(name.replace("_", " → "))}</td>',
            f'<td>{_num(1000*data["max_base_difference_m"])} mm</td>',
            f'<td>{_num(1000*data["rms_base_difference_m"])} mm</td>',
            f'<td>{_num(1000*data["final_base_difference_m"])} mm</td>',
            f'<td>{_num(data["max_chart_orientation_difference_rad"])} rad</td>',
            f'<td>{_num(data["max_joint_difference_rad"])} rad</td>',
        ]) + '</tr>')
    ratio = validation.get('refinement', {}).get('rms_refinement_ratio')
    ratio_limit = checks.get('refinement.rms_difference_contracts', {}).get('limit')
    refinement_limits = []
    for metric, multiplier, unit in [
        ('max_base_difference_m', 1000, 'mm'),
        (None, 1, ''),
        ('final_base_difference_m', 1000, 'mm'),
        ('max_chart_orientation_difference_rad', 1, 'rad'),
        ('max_joint_difference_rad', 1, 'rad'),
    ]:
        limit = checks.get('refinement.nominal_fine.' + metric, {}).get('limit') if metric else None
        refinement_limits.append(f'<td>≤ {_num(multiplier*limit)} {unit}</td>' if limit is not None else '<td>Ratio below</td>')
    refinement_rows.append('<tr class="limits"><td>Declared tolerance</td>' + ''.join(refinement_limits) + '</tr>')

    audit_rows = []
    audit_descriptions = {
        'mechanics_validation': ('CMG → Pinocchio mechanics', 'Compiled rigid-body model, source consistency, kinematics, Jacobians and dynamics.'),
        'contact_validation': ('PACDM task and ideal-support algebra', 'Foot assembly, differential maps, rank/mobility and constrained dynamics; distinct from compliant plant contact.'),
        'task_validation': ('Full reference audit', 'Prescribed foot closures, velocity/acceleration constraints and support schedule.'),
        'contact_law_audit': ('Independent plant contact audit', 'Sphere-surface Jacobians, local force law, friction, release and momentum balance.'),
    }
    for stem, (title, description) in audit_descriptions.items():
        data = _read(out / f'{stem}.json')
        audit_checks = _rows(data.get('checks', []))
        passed_count = sum(bool(value.get('passed', False)) for _, value in audit_checks)
        count = len(audit_checks)
        audit_rows.append(f'<tr><td><a href="{stem}.json">{_esc(title)}</a><small>{_esc(description)}</small></td><td>{passed_count} / {count}</td><td>{_badge(data.get("passed", False))}</td></tr>')
    reference_path = out / 'reference_regeneration.json'
    if reference_path.exists():
        regenerated = _read(reference_path)
        differences = regenerated.get('regenerated_against_supplied_reference', {})
        max_difference = max((float(abs(value)) for value in differences.values()), default=float('nan'))
        reference_note = f'<p>The reference was regenerated with the original PACDM core: {_esc(regenerated.get("samples", "—"))} samples; maximum assembly residual {_num(regenerated.get("max_residual_inf"))}; maximum difference from the original reference across recorded arrays {_num(max_difference)}. <a href="reference_regeneration.json">Reference regeneration evidence</a>.</p>'
    else:
        reference_note = '<p class="warning">Reference regeneration evidence was not available when this report was built.</p>'

    negative = cases.get('no_actuation', {})
    negative_checks = {key: value for key, value in checks.items() if key.startswith('negative_control.')}
    negative_pass = bool(negative_checks) and all(value['passed'] for value in negative_checks.values())
    negative_description = ('The posture/fall guard stopped the unactuated plant' if not negative.get('completed', True)
                            else 'The unactuated plant completed the horizon; the expected fall was not observed')
    negative_text = f'{negative_description} at {_num(negative.get("simulated_s"))} s. Zero commanded motor torque is checked separately. The expected negative-control failure is not counted as a successful locomotion case.'

    all_gate_table = _checks_table(checks)
    min_clearance = min(float(c['minimum_clearance_m']) for c in clearances.values())
    geometric_samples = sum(int(c.get('checked_samples', 0)) for c in clearances.values())
    css = '''
    :root{color-scheme:light;--ink:#183047;--muted:#586b7b;--line:#dce5eb;--blue:#146ca5;--green:#12634d;--red:#a62932}
    *{box-sizing:border-box}body{margin:0;background:#edf2f6;color:var(--ink);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1120px;margin:36px auto;padding:0 28px 48px}header{background:#112d46;border-radius:18px;padding:38px;color:white}header p{color:#d9e5ef;max-width:830px}h1{font-size:36px;line-height:1.14;margin:10px 0 18px;letter-spacing:-.7px}h2{font-size:22px;line-height:1.3;margin:0 0 16px}h3{font-size:17px;margin:18px 0 8px}p{margin:10px 0 16px}.eyebrow{letter-spacing:2px;text-transform:uppercase;font-size:12px;font-weight:750;color:#b7d5e8}section{background:white;border:1px solid var(--line);border-radius:14px;padding:26px;margin-top:22px}a{color:var(--blue);text-underline-offset:3px}.status{display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:#f2f8f5;border-left:5px solid var(--green)}.status.failed,.failure-panel{background:#fff3f2;border-left:5px solid var(--red)}.badge{display:inline-block;white-space:nowrap;font-size:12px;font-weight:800;line-height:1.4;padding:4px 9px;border-radius:30px}.pass{color:#105c46;background:#d9f0e5}.fail{color:#9c1d2d;background:#ffe0e1}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:22px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}.card strong{display:block;font-size:26px;letter-spacing:-.7px}.card span,small,.note{font-size:13px;color:var(--muted)}small{display:block;line-height:1.4;margin-top:4px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:11px;letter-spacing:.6px;text-transform:uppercase;background:#f1f5f8}td:not(:first-child){font-variant-numeric:tabular-nums}.limits{background:#f5f8fa;font-weight:650}.limits td{font-size:12px}img.chart{display:block;width:100%;height:auto}video{display:block;width:100%;max-height:690px;background:#102638;border-radius:10px}.warning{color:var(--red);font-weight:650}.columns{display:grid;grid-template-columns:1fr 1fr;gap:28px}ul{padding-left:21px}li{margin:6px 0}details{margin-top:18px}summary{cursor:pointer;font-weight:700;color:var(--blue)}code{font-size:12px;background:#eef3f6;padding:2px 5px;border-radius:3px}footer{font-size:13px;color:var(--muted);padding:20px 4px} @media(max-width:750px){main{padding:0 14px;margin-top:14px}header{padding:25px}h1{font-size:28px}.cards{grid-template-columns:1fr 1fr}.columns{grid-template-columns:1fr}section{padding:19px}.card strong{font-size:22px}} @media print{body{background:white}main{max-width:none;margin:0;padding:0}header{background:white;color:var(--ink);border:1px solid var(--line)}header p{color:var(--muted)}video{max-height:350px}section,.cards{break-inside:avoid}details{display:none}}
    '''
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Go2 CMG / PACDM — Pinocchio validation</title><style>{css}</style></head>
<body><main>
<header><div class="eyebrow">Reproducible simulation evidence</div><h1>Go2 CMG / PACDM<br>Pinocchio validation</h1><p>A motor-driven 26-second gait with obstacle clearance, disturbance recovery, parameter variations and full-cycle timestep refinement. The video replays states generated by the Pinocchio plant.</p></header>
<section class="status {'' if status else 'failed'}">{_badge(status, status_word)}<strong>{_esc(banner)}</strong><span>{validation['passed_count']} / {validation['check_count']} aggregate gates</span></section>
{failure_block}
<div class="cards"><div class="card"><strong>{_num(1000*nominal['final_position_error_m'])} mm</strong><span>Nominal final base error</span></div><div class="card"><strong>{_num(1000*nominal['rms_body_error_m'])} mm</strong><span>Nominal RMS base error</span></div><div class="card"><strong>{_num(1000*min_clearance)} mm</strong><span>Minimum forbidden-pair clearance across positive cases</span></div><div class="card"><strong>{len(positive)} cases</strong><span>Positive cases; negative control reported separately</span></div></div>
<section><h2>What was validated</h2><div class="columns"><div><h3>Algorithm and controller</h3><p>The unchanged PACDM core assembles the base and Cartesian foot tasks and their differential maps. The original whole-body torque controller is retained. Only the 12 leg motors are actuated; the six base coordinates are unactuated.</p><p>Go2 has no permanent structural kinematic loops. Its PACDM tests concern foot-task assembly and changing ideal support constraints, not validation of a permanent-loop mechanism.</p></div><div><h3>Independent simulation plant</h3><p>Pinocchio 3.8.0 provides the CMG-compiled rigid-body mechanics, mass and bias terms, frame Jacobians and configuration integration. A separate custom compliant contact solver supplies unilateral spherical-foot contact, a circular Coulomb friction cap and tangential elastic/plastic memory.</p><p>The plant is driven by motor torques. It computes contact reactions from its evolving state. The controller's predicted contact forces are diagnostic QP outputs and are not injected as plant reactions.</p></div></div><p class="note">This is finite numerical evidence under the stated models and acceptance limits. It is not hardware validation, contact calibration, a general stability proof, or a test of identical MuJoCo contact behavior. MuJoCo is used for source-model cross-checks where documented; the Pinocchio rollouts check that it is not imported into the simulation process.</p></section>
<section><h2>Simulation video</h2><video controls preload="metadata" poster="poster.png"><source src="Go2_Pinocchio.mp4" type="video/mp4">Open <a href="Go2_Pinocchio.mp4">Go2_Pinocchio.mp4</a> to view the simulation.</video><p class="note"><a href="Go2_Pinocchio.mp4">Open video</a> · <a href="poster.png">Poster</a>. The images are rendered from the nominal saved simulated states; visual appearance alone is not an acceptance test.</p></section>
<section><h2>Tracking and timestep evidence</h2><img class="chart" src="data:image/png;base64,{chart_data}" alt="Four plots: actual and reference base path, actual and reference base height, base position error for three timesteps, and logarithmic pairwise base differences."><p class="note">Nominal, half-step and quarter-step runs keep the same 4 ms control period. Logged comparison states share 10 ms timestamps. Differences in panel D are pairwise simulation differences, not errors against exact dynamics.</p></section>
<section><h2>Measured case outcomes</h2><div class="table-wrap"><table><thead><tr><th>Case</th><th>Plant step</th><th>Horizon</th><th>Final base error</th><th>RMS base error</th><th>Peak base error</th><th>Min. clearance*</th><th>All case gates</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table></div><p class="note">Base errors are 3D Euclidean position errors. *Clearance concerns eligible forbidden collision pairs; expected sphere-foot support contacts are excluded. Full gates additionally cover yaw, joint errors and limits, torque limits, posture, speed, measured support modes, contact-law residuals, momentum balance, finite states and prescribed push impulses.</p><h3>Geometric audit scope</h3><p>{'Every recorded integration state is checked' if dense_clearance else 'Available saved states are checked'} using Coal signed distances on the source collision primitives ({geometric_samples:,} checked states across positive cases). Nominal minimum clearance is {_num(1000*nominal_clearance['minimum_clearance_m'])} mm for <code>{_esc(nominal_clearance['minimum_pair'])}</code> at {_num(nominal_clearance['minimum_time_s'])} s. The geometry audit does not resolve forbidden collisions.</p><p class="note">Even checking dense q at every integration step is a discrete-time audit, not a proof of continuous-time collision freedom. Visual meshes are not substituted for the source collision primitives.</p></section>
<section><h2>Timestep refinement over the complete cycle</h2><div class="table-wrap"><table><thead><tr><th>Comparison</th><th>Max. base difference</th><th>RMS base difference</th><th>Final base difference</th><th>Max. chart-angle difference</th><th>Max. joint difference</th></tr></thead><tbody>{''.join(refinement_rows)}</tbody></table></div><p>RMS refinement ratio: <strong>{_num(ratio)}</strong>; declared limit: ≤ {_num(ratio_limit)}. The ratio compares the 0.5 → 0.25 ms RMS base difference with the 1 → 0.5 ms difference.</p><p class="note">These are three full-cycle finite-step runs. A contracting pairwise difference supports timestep consistency for this benchmark; it does not identify an exact solution or prove asymptotic convergence order.</p></section>
<section><h2>Mechanics, PACDM and contact evidence</h2><div class="table-wrap"><table><thead><tr><th>Nested audit</th><th>Passed / total checks</th><th>Result</th></tr></thead><tbody>{''.join(audit_rows)}</tbody></table></div><p class="note">Nested counts are shown separately. Each audit contributes its own passed flag to the aggregate; the nested checks are not added to the {validation['check_count']} aggregate-gate count.</p>{reference_note}<h3>Expected negative control</h3><p>{_badge(negative_pass, 'EXPECTED FAILURE OBSERVED' if negative_pass else 'NEGATIVE CONTROL FAILED')} {_esc(negative_text)}</p><p class="note">Failure of this control supports that the displayed trajectory depends on actuation. It does not independently establish physical fidelity.</p></section>
<section><h2>Evidence files and declared acceptance limits</h2><p>The machine-readable <a href="validation.json">validation.json</a> is the source for the acceptance status. Inspect <a href="nominal.json">nominal metrics</a>, <a href="nominal_clearance.json">nominal clearance</a> and the case NPZ files for measured trajectories. <a href="summary.png">Download the scientific figure</a>.</p><details><summary>Show every aggregate gate, measured value and tolerance</summary>{all_gate_table}</details></section>
<footer>Software reference: <a href="https://github.com/stack-of-tasks/pinocchio/tree/v3.8.0">official Pinocchio v3.8.0 source</a>. Numbers in this report come from the accompanying computed artifacts. The chart and styling are embedded; keep the adjacent video and evidence files with this report for local playback and links.</footer>
</main></body></html>'''
    report_path = out / 'report.html'
    report_path.write_text(page, encoding='utf-8')
    return {'report': str(report_path), 'figure': str(out / 'summary.png')}


if __name__ == '__main__':
    print(json.dumps(build_report(Path(__file__).resolve().parents[1]), indent=2))
