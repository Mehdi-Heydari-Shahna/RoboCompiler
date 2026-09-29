"""Create the tracked-excavator report directly from recorded results.

The HTML embeds its scientific figures and can be opened offline. Missing and
failed checks are exposed; rendering a report never changes a validation result.
"""
from project import ROOT, RESULTS
import argparse
import base64
from datetime import datetime, timezone
import html
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#333333']
FIGURES = RESULTS / 'figures'
MECHANICAL_KEYS = ['arm_mechanical', 'drive_mechanical', 'equality', 'contact', 'passive']
RELEASE_CASES = {'coarse', 'nominal', 'fine', 'low_traction', 'heavy_payload', 'no_drive', 'no_bucket_contact'}
CONTROL_REVISION = 'speed_scheduled_lateral_yaw_damping_multirate_v2'
AUDIT_REVISION = 'complete_native_global_wrench_v2'


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_task_audit(name):
    """Only a hash-matched independent audit may supply accepted task metrics."""
    path=RESULTS/f'{name}_task_audit.json'
    audit=read_json(path)
    if audit is None:return {'audit':None,'fresh':False,'status':'Missing'}
    inputs={'input_model_sha256':RESULTS/'assets'/f'{name}.xml',
            'input_trace_sha256':RESULTS/f'{name}.npz','input_report_sha256':RESULTS/f'{name}.json'}
    matches={key:file.exists() and audit.get(key)==sha256(file) for key,file in inputs.items()}
    fresh=all(matches.values())
    return {'audit':audit,'fresh':fresh,'status':'Fresh' if fresh else 'Stale / mismatched inputs','input_matches':matches}


def audited_task(item):
    return item['audit'].get('task',{}) if item.get('fresh') and item.get('audit') else {}


def load_dynamics_audit(name):
    """Bind momentum figures to the current oracle, inputs and original metadata."""
    audit=read_json(RESULTS/f'{name}_audit.json')
    if audit is None:return {'audit':None,'fresh':False,'status':'Missing'}
    hashes=audit.get('input_sha256',{})
    fresh=(bool(hashes) and audit.get('revision')==AUDIT_REVISION and
           audit.get('physical_trace_unchanged') is True and audit.get('audit_times_unchanged') is True and
           all((ROOT/path).is_file() and sha256(ROOT/path)==expected for path,expected in hashes.items()))
    return {'audit':audit,'fresh':fresh,'status':'Fresh' if fresh else 'Stale / mismatched inputs'}


def momentum_summary(item):
    """All maxima are over independently audited snapshots, not every time step."""
    if not item.get('fresh') or not item.get('audit'):return [None]*4
    snapshots=item['audit'].get('snapshots',[])
    if not snapshots:return [None]*4
    couples=[];raw=[];finite_difference=[];analytic=[]
    for snapshot in snapshots:
        ledger=snapshot.get('momentum_accounting',{})
        wrench=ledger.get('finite_gap_equality_global_wrench')
        if wrench is None or len(wrench)!=6:return [None]*4
        couples.append(float(np.linalg.norm(wrench[3:])))
        raw.append(ledger.get('raw_ideal_angular_balance_relative'))
        finite_difference.append(snapshot.get('metrics',{}).get('angular_momentum_rate_relative'))
        analytic.append(snapshot.get('metrics',{}).get('analytic_angular_momentum_rate_relative'))
    return [max(values) if all(value is not None for value in values) else None
            for values in [couples,raw,finite_difference,analytic]]


def esc(value):
    return html.escape(str(value))


def number(value, digits=3):
    if value is None:
        return 'Unavailable'
    if isinstance(value, (bool, np.bool_)):
        return 'Yes' if value else 'No'
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return 'Nonfinite'
        if value == 0:
            return '0'
        if abs(value) < 0.001 or abs(value) >= 1e6:
            return f'{value:.2e}'
        return f'{value:,.{digits}f}'
    return str(value)


def badge(value, true='PASS', false='FAIL'):
    label = true if value is True else false if value is False else 'UNASSESSED'
    cls = 'pass' if value is True else 'fail' if value is False else 'pending'
    return f'<span class="badge {cls}">{label}</span>'


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def figure_html(path, caption):
    data = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def save_figure(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=165, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return FIGURES / filename


def configure_plots():
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.prop_cycle': matplotlib.cycler(color=COLORS), 'grid.alpha': .22,
        'axes.grid': True, 'figure.constrained_layout.use': True})


def make_figures(name, report, trace):
    configure_plots()
    t = trace['time']
    output = []
    fig, axs = plt.subplots(2, 2, figsize=(11.3, 7.5))
    a = axs[0, 0]
    a.plot(trace['target'][:, 0], trace['target'][:, 1], '--', color='#888888', label='Commanded route')
    a.plot(trace['pose'][:, 0], trace['pose'][:, 1], label='Native base motion')
    a.scatter(*trace['pose'][0, :2], marker='o', color=COLORS[2], label='Start', zorder=5)
    a.scatter(*trace['pose'][-1, :2], marker='x', color=COLORS[1], label='Final', zorder=5)
    a.set(xlabel='Base x displacement (m)', ylabel='Base y displacement (m)', title='Base route in the source world frame')
    a.axis('equal'); a.legend()
    a = axs[0, 1]
    a.plot(t, trace['pose'][:, 0], label='Actual x'); a.plot(t, trace['target'][:, 0], '--', label='Commanded x')
    a.plot(t, trace['pose'][:, 1], label='Actual y')
    a.set(xlabel='Simulation time (s)', ylabel='Displacement (m)', title='Translation'); a.legend()
    a = axs[1, 0]
    a.plot(t, np.rad2deg(trace['pose'][:, 2]), label='Actual heading')
    a.plot(t, np.rad2deg(trace['target'][:, 2]), '--', label='Commanded heading')
    a.set(xlabel='Simulation time (s)', ylabel='Heading (degrees)', title='Turning'); a.legend()
    a = axs[1, 1]
    for key, label in [('captured_mass', 'Inside bucket'), ('carried_remaining_mass', 'Ever-carried union inside (preliminary)'), ('delivered_mass', 'Ever-carried union at destination (preliminary)')]:
        a.plot(t, trace[key], label=label)
    a.set(xlabel='Simulation time (s)', ylabel='Material mass (kg)', title='Live bucket and destination occupancy'); a.legend(fontsize=8)
    output.append((save_figure(fig, name + '_mission.png'), 'Recorded route, heading and preliminary occupancy traces. The ever-carried union does not establish qualified transport or settled delivery; those outcomes come only from the fresh independent particle audit below.'))

    fig, axs = plt.subplots(3, 2, figsize=(11.3, 10.7))
    pressure_key = 'drive_pressure_applied' if 'drive_pressure_applied' in trace else 'drive_pressure'
    for i, side in enumerate(['Left', 'Right']):
        axs[0, 0].plot(t, trace['drive_effort'][:, i] / 1000, label=side)
        axs[0, 1].plot(t, trace['drive_speed'][:, i], label=side + ' actual')
        axs[0, 1].plot(t, trace['drive_desired_speed'][:, i], '--', color=COLORS[i], alpha=.6, label=side + ' command')
        for j, chamber in enumerate(['A', 'B']):
            axs[1, 0].plot(t, trace[pressure_key][:, i, j] / 1e6, color=COLORS[i], linestyle='-' if j == 0 else '--', label=f'{side} chamber {chamber}')
    axs[0, 0].set(ylabel='Sprocket effort (kN m)', title='Hydraulic travel-drive output')
    axs[0, 1].set(ylabel='Sprocket speed (rad/s)', title='Drive speed tracking')
    axs[1, 0].set(ylabel='Pressure (MPa)', title='Applied travel-drive chamber pressures' if pressure_key=='drive_pressure_applied' else 'Travel-drive chamber states (legacy trace)')
    for i in range(min(6, trace['arm_effort'].shape[1])):
        axs[1, 1].plot(t, trace['arm_effort'][:, i] / 1000, label=f'p{i}')
    axs[1, 1].set(ylabel='Cylinder force (kN)', title='All six hydraulic cylinder efforts')
    for i, port_name in enumerate(['q21', 'q23'], start=6):
        if i < trace['arm_effort'].shape[1]:
            axs[2, 0].plot(t, trace['arm_effort'][:, i] / 1000, label=port_name+' actual')
    axs[2, 0].set(ylabel='Rotary effort (kN m)', title='Both original rotary effort ports')
    if 'arm_pressure' in trace:
        for i in range(trace['arm_pressure'].shape[1]):
            axs[2, 1].plot(t, (trace['arm_pressure'][:, i, 0]-trace['arm_pressure'][:, i, 1])/1e6, label=f'p{i}')
    axs[2, 1].set(ylabel='Chamber A minus B pressure (MPa)', title='Six cylinder pressure differences')
    for a in axs.flat:
        a.set_xlabel('Simulation time (s)'); a.legend(ncol=2)
    pressure_scope = 'Drive pressures are the applied pressures used to calculate effort.' if pressure_key=='drive_pressure_applied' else 'Legacy trace: drive pressures are pre-step stored states; applied-pressure history is unavailable.'
    output.append((save_figure(fig, name + '_actuation.png'), 'All ten actual actuator efforts, travel speeds and hydraulic pressure histories. '+pressure_scope+' Cylinder p0 has zero requested force; actual friction and pressure forces may remain nonzero. Cylinder pressure differences do not directly equal force divided by one area because the chamber areas differ.'))

    powers = {key: i for i, key in enumerate(report.get('power_order', []))}
    if powers:
        work = trace['work']; energy = trace['energy'].sum(axis=1)
        midpoint = trace.get('mechanical_midpoint_work')
        midpoint_available = midpoint is not None
        mechanical_work = midpoint if midpoint_available else np.column_stack([work[:, powers[k]] for k in MECHANICAL_KEYS])
        fig, axs = plt.subplots(3, 2, figsize=(11.3, 10.7))
        for key, label in [('arm_mechanical', 'Arm mechanical'), ('drive_mechanical', 'Travel mechanical'), ('arm_supply', 'Arm hydraulic supply'), ('drive_supply', 'Travel hydraulic supply')]:
            values = mechanical_work[:, MECHANICAL_KEYS.index(key)] if key in MECHANICAL_KEYS else work[:, powers[key]]
            axs[0, 0].plot(t, values / 1000, label=label)
        axs[0, 0].set(ylabel='Accumulated work (kJ)', title='Mechanical and hydraulic work' if midpoint_available else 'Legacy rectangular work quadrature'); axs[0, 0].legend()
        rhs = mechanical_work.sum(axis=1)
        rectangle_rhs = sum(work[:, powers[k]] for k in MECHANICAL_KEYS)
        axs[0, 1].plot(t, (energy-energy[0]) / 1000, label='Native mechanical energy change')
        axs[0, 1].plot(t, rhs / 1000, '--', label='Native force × midpoint velocity work' if midpoint_available else 'Legacy rectangular power integral')
        axs[0, 1].set(ylabel='Energy / work (kJ)', title='Mechanical balance'); axs[0, 1].legend()
        axs[1, 0].plot(t, (energy-energy[0]-rhs) / 1000, color=COLORS[1], label='Midpoint work defect' if midpoint_available else 'Legacy rectangular work defect')
        if midpoint_available:
            axs[1, 0].plot(t, (energy-energy[0]-rectangle_rhs) / 1000, '--', color=COLORS[0], label='Rectangular work defect')
        axs[1, 0].legend()
        axs[1, 0].set(ylabel='Energy defect (kJ)', title='Mechanical defect, shown without correction')
        for key, label in [('drive_throttle', 'Drive throttle loss'), ('drive_leakage', 'Drive leakage loss'), ('drive_relief', 'Drive relief loss'), ('drive_friction', 'Drive friction loss'), ('drive_numerical_storage_loss', 'Drive storage discretization loss')]:
            if key in powers:
                axs[1, 1].plot(t, work[:, powers[key]] / 1000, label=label)
        axs[1, 1].set(ylabel='Accumulated energy (kJ)', title='Travel-drive loss accounting'); axs[1, 1].legend()
        w={key:work[:,idx]-work[0,idx] for key,idx in powers.items()}
        arm_net=w['arm_supply']-w['arm_throttle']-w['arm_leakage']-w['arm_relief']-w['arm_friction']+w['arm_compressibility_geometry']+w['arm_rotary']
        drive_net=w['drive_supply']-w['drive_throttle']-w['drive_leakage']-w['drive_relief']-w['drive_friction']-w['drive_numerical_storage_loss']
        arm_delta=trace['arm_fluid_energy']-trace['arm_fluid_energy'][0]
        drive_delta=trace['drive_fluid_energy']-trace['drive_fluid_energy'][0]
        arm_defect=arm_delta-arm_net+w['arm_mechanical']
        drive_defect=drive_delta-drive_net+w['drive_mechanical']
        total_defect=energy-energy[0]+arm_delta+drive_delta-arm_net-drive_net-sum(w[k] for k in ['equality','contact','passive'])
        axs[2,0].plot(t,total_defect/1000,label='Full coupled defect (rectangular environment)')
        axs[2,0].plot(t,arm_defect/1000,label='Arm-fluid held-speed defect')
        axs[2,0].plot(t,drive_defect/1000,label='Drive-fluid held-speed defect')
        axs[2,0].set(ylabel='Raw balance defect (kJ)',title='Full hydraulic + mechanical accounting');axs[2,0].legend(fontsize=8)
        axs[2,1].plot(t,arm_delta/1000,label='Arm-fluid stored-energy change')
        axs[2,1].plot(t,(arm_net-w['arm_mechanical'])/1000,'--',label='Arm-fluid net input after held shaft work')
        axs[2,1].set(ylabel='Energy (kJ)',title='Arm-fluid finite-step storage discrepancy');axs[2,1].legend(fontsize=8)
        for a in axs.flat:
            a.set_xlabel('Simulation time (s)')
        quadrature_scope = 'Mechanical work uses native force times average pre/post-step velocity; the rectangular ledger remains separate for hydraulic bookkeeping. Both numerical defects are shown.' if midpoint_available else 'Legacy trace: midpoint mechanical work is unavailable, so these plots explicitly use rectangular quadrature.'
        output.append((save_figure(fig, name + '_energy.png'), quadrature_scope+' The full hydraulic and mechanical defect is distinct from the small midpoint mechanical defect; arm-fluid storage error remains exposed. Hydraulic supply work is not fuel consumption.'))

    fig, axs = plt.subplots(2, 2, figsize=(11.3, 7.5))
    axs[0, 0].semilogy(t, np.maximum(trace['loop_gap'], 1e-15), label='Arm cut-joint closure')
    axs[0, 0].semilogy(t, np.maximum(trace['track_gap'], 1e-15), label='Track closing joint')
    axs[0, 0].axhline(1e-4, color=COLORS[0], linestyle=':', label='Arm acceptance limit')
    axs[0, 0].axhline(1e-3, color=COLORS[1], linestyle=':', label='Track acceptance limit')
    axs[0, 0].set(ylabel='Maximum point residual (m)', title='Native closure constraints'); axs[0, 0].legend()
    err = trace['independent'][:, :6] - trace['desired']
    for i in range(err.shape[1]):
        axs[0, 1].plot(t, err[:, i], label=['q23', 'q7', 'q4', 'q0', 'q1', 'q21'][i])
    axs[0, 1].set(ylabel='Tracking error (rad)', title='Preferred arm coordinates'); axs[0, 1].legend(ncol=3)
    names = report.get('body_names', [])
    if 'body_contact_wrenches' in trace and names:
        forces = trace['body_contact_wrenches'][:, :, :3]
        shoe = [i for i, n in enumerate(names) if n.startswith('track_') and '_shoe_' in n]
        if shoe:
            support = forces[:, shoe].sum(axis=1)
            for i, label in enumerate(['World x', 'World y', 'World z']):
                axs[1, 0].plot(t, support[:, i] / 1000, label=label)
        if 'body_56' in names:
            axs[1, 1].plot(t, np.linalg.norm(forces[:, names.index('body_56')], axis=1) / 1000, label='Bucket net external contact force')
        axs[1, 0].legend(); axs[1, 1].legend()
    axs[1, 0].set(ylabel='Net force (kN)', title='External contact force on all track shoes')
    axs[1, 1].set(ylabel='Force magnitude (kN)', title='Bucket contact load')
    for a in axs.flat:
        a.set_xlabel('Simulation time (s)')
    output.append((save_figure(fig, name + '_forces_closure.png'), 'Closure, control and explicit contact-load histories. Net shoe contact includes track mechanism contacts as well as ground; internal shoe-pair forces cancel in the sum. These are not unique closed-loop pin reactions.'))
    return output


def make_figures_only(case_names=None):
    """Regenerate selected case figures without assembling the historical suite report.

    This reads existing case JSON/NPZ files and uses the unchanged plot routines.
    It neither runs a simulation nor creates or changes an acceptance result.
    """
    if case_names is None:
        preferred = ['nominal', 'coarse', 'fine', 'low_traction', 'heavy_payload', 'no_drive', 'no_bucket_contact']
        case_names = [name for name in preferred
                      if (RESULTS/f'{name}.json').is_file() and (RESULTS/f'{name}.npz').is_file()]
        if not case_names:
            case_names = sorted(path.stem for path in RESULTS.glob('*.json')
                                if (RESULTS/f'{path.stem}.npz').is_file())
    if not case_names:
        raise FileNotFoundError('No selected case has existing result JSON and trajectory NPZ files.')
    output = []
    for name in case_names:
        report_path, trace_path = RESULTS/f'{name}.json', RESULTS/f'{name}.npz'
        if not report_path.is_file() or not trace_path.is_file():
            raise FileNotFoundError(f'Case {name} requires both {report_path} and {trace_path}.')
        report = read_json(report_path)
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = {key: archive[key] for key in archive.files}
        for path, _caption in make_figures(name, report, trace):
            output.append(path)
            print(path, flush=True)
    return output


def flatten_checks(value, prefix=''):
    """Accept common validation schemas without silently discarding failures."""
    rows = []
    if isinstance(value, list):
        for i, item in enumerate(value):
            rows.extend(flatten_checks(item, prefix + f'[{i}]'))
    elif isinstance(value, dict):
        if isinstance(value.get('passed'), bool) and any(k in value for k in ['name', 'check', 'value', 'limit']):
            rows.append({'name': value.get('name', value.get('check', prefix)), **value})
        else:
            for key, item in value.items():
                if key == 'checks' and isinstance(item, dict):
                    for name, result in item.items():
                        if isinstance(result, bool):
                            rows.append({'name': prefix + '/' + name, 'passed': result,
                                         'value': value.get('metrics', {}).get(name), 'limit': value.get('limits', {}).get(name)})
                        else:
                            rows.extend(flatten_checks(result, prefix + '/' + name))
                elif isinstance(item, (dict, list)):
                    rows.extend(flatten_checks(item, prefix + '/' + key))
    return rows


def make_report(case_names=None, validation_file='validation.json'):
    if case_names is None:
        preferred = ['nominal', 'coarse', 'fine', 'low_traction', 'heavy_payload', 'no_drive', 'no_bucket_contact']
        case_names = [x for x in preferred if (RESULTS / f'{x}.json').exists()]
        if not case_names:
            case_names = [p.stem for p in RESULTS.glob('*.json') if (RESULTS / (p.stem+'.npz')).exists()]
    reports = {name: read_json(RESULTS / f'{name}.json') for name in case_names}
    reports = {name: item for name, item in reports.items() if item is not None}
    if not reports:
        raise FileNotFoundError('No completed simulation result and trace are available.')
    main_name = 'nominal' if 'nominal' in reports else next(iter(reports))
    main = reports[main_name]
    task_audits={name:load_task_audit(name) for name in reports}
    dynamics_audits={name:load_dynamics_audit(name) for name in reports}
    main_task_audit=task_audits[main_name]
    main_task=audited_task(main_task_audit)
    energy_audits={name:(item['audit'].get('energy_decomposition',{}) if item.get('fresh') else {}) for name,item in task_audits.items()}
    main_energy=energy_audits[main_name]
    with np.load(RESULTS / f'{main_name}.npz', allow_pickle=False) as a:
        traces = {k: a[k] for k in a.files}
    figs = make_figures(main_name, main, traces)
    vp = Path(validation_file)
    if not vp.is_absolute(): vp = RESULTS / vp
    validation = read_json(vp)
    checks = flatten_checks(validation) if validation is not None else []
    failed = [x for x in checks if x.get('passed') is False]
    root_pass = validation.get('passed') if isinstance(validation, dict) else None
    task_acceptance=main_task.get('passed') if main_task.get('available') else None
    complete_case_set=set(reports)==RELEASE_CASES
    fresh_case_audits=complete_case_set and all(item.get('fresh') for item in task_audits.values())
    fresh_dynamics=complete_case_set and all(item.get('fresh') and item.get('audit',{}).get('passed') is True for item in dynamics_audits.values())
    current_controls=complete_case_set and all(r.get('scene',{}).get('travel_controller',{}).get('revision')==CONTROL_REVISION for r in reports.values())
    servo=read_json(RESULTS/'travel_servo_validation.json',{})
    servo_hashes=servo.get('source_sha256',{})
    servo_fresh=bool(servo_hashes) and all((ROOT/name).is_file() and sha256(ROOT/name)==servo_hashes.get(name) for name in ['track_drive.py','track_servo.py','travel_servo_validation.py'])
    solver_component=read_json(ROOT/'outputs/native_solver_accuracy/report.json',{})
    solver_hashes=solver_component.get('input_sha256',{})
    solver_component_fresh=bool(solver_hashes) and all((ROOT/name).is_file() and sha256(ROOT/name)==expected for name,expected in solver_hashes.items())
    guarded_cases=complete_case_set and all(r.get('numerical_solver',{}).get('policy')=='residual_controlled_native_warmstart_retry' and r.get('numerical_solver',{}).get('maximum_final_residual',float('inf'))<=1e-8 for r in reports.values())
    current_receivers=complete_case_set and all(r.get('scene',{}).get('task_times_s',{}).get('end')==90. and r.get('scene',{}).get('receiving_bay',{}).get('native_contact') is True for r in reports.values())
    overall = False if failed or root_pass is False or task_acceptance is False else True if root_pass is True and checks and task_acceptance is True and fresh_case_audits and fresh_dynamics and current_controls and servo_fresh and servo.get('passed') is True and solver_component_fresh and solver_component.get('passed') is True and guarded_cases and current_receivers else None
    total = len(checks); passed = sum(x.get('passed') is True for x in checks)
    tracks = main.get('scene', {}).get('tracks', {})
    arms = main.get('arm_controller', {})
    peaks = main.get('peak', {})
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    status_text = 'All recorded acceptance checks passed' if overall is True else 'Recorded acceptance failures remain' if overall is False else 'Validation is not yet complete'
    cards = [
        ('Audited delivered material', number(main_task.get('delivered_mass_kg'), 1)+' kg' if main_task.get('available') else 'Unavailable', 'Qualified transported particles outside the bucket and settled at the destination'),
        ('Audited loaded displacement', number(main_task.get('loaded_base_displacement_m'), 2)+' m' if main_task.get('transport_phase_available') else 'Unavailable', 'Measured over the registered departure-to-arrival interval'),
        ('Final position error', number(main.get('final_position_error_m'), 3)+' m', 'Distance to the final commanded base position'),
        ('Mechanical defect', number(100*main['mechanical_balance_relative'], 2)+'%' if 'mechanical_balance_relative' in main else 'Unavailable', 'Raw defect '+number(main.get('mechanical_balance_defect_J'),1)+' J; normalized by absolute mechanical activity')]
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoboIR — Articulated-track excavator v26</title><style>
    :root{{--ink:#142735;--muted:#526371;--line:#dce3e8;--blue:#0072b2;--paper:#f3f6f8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink)}}header{{background:#142735;color:white;padding:46px max(5vw,24px) 34px}}header p{{max-width:850px;color:#c6d7e3}}.eyebrow{{letter-spacing:.17em;font-size:12px;font-weight:750;color:#81cfe5}}h1{{font-size:clamp(30px,4vw,49px);line-height:1.15;max-width:1000px;margin:14px 0}}h2{{margin:0 0 14px;font-size:27px}}h3{{font-size:19px;margin-top:26px}}nav{{display:flex;gap:20px;flex-wrap:wrap;margin-top:24px}}nav a{{color:#c9e9f6;text-decoration:none;font-size:14px}}main{{max-width:1230px;margin:28px auto;padding:0 24px}}section{{background:white;border:1px solid var(--line);border-radius:12px;padding:28px;margin:22px 0}}p{{max-width:1050px}}.muted,figcaption{{color:var(--muted)}}.status{{background:#fff;border-left:5px solid var(--blue)}}.status.fail-state{{border-color:#b33036}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0}}.card{{background:white;padding:21px;border:1px solid var(--line);border-radius:10px}}.card-label{{color:var(--muted);font-size:13px}}.metric{{display:block;font-size:30px;font-weight:730;margin:7px 0}}.card small{{color:var(--muted);font-size:12px;display:block;line-height:1.5}}.badge{{display:inline-block;border-radius:4px;padding:3px 9px;font-size:11px;font-weight:800;letter-spacing:.05em;white-space:nowrap}}.pass{{background:#e1f1e8;color:#14683f}}.fail{{background:#fbe3e4;color:#a4222b}}.pending{{background:#fff0ce;color:#795100}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{background:#edf3f7;text-align:left;color:#304b5f;font-size:12px}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}}td:first-child{{font-weight:550}}.table-scroll{{overflow-x:auto}}figure{{margin:26px 0}}figure img{{width:100%;display:block}}figcaption{{font-size:13px;padding-top:9px}}a{{color:#00689b}}code{{font:12px ui-monospace,monospace;background:#edf3f7;border-radius:3px;padding:2px 4px}}ul{{padding-left:22px}}li{{margin:8px 0}}.notice{{background:#fff6e4;border-left:4px solid #e3ae36;padding:14px 18px}}details{{margin:16px 0}}summary{{cursor:pointer;font-weight:650}}footer{{padding:18px 0 32px;color:var(--muted);font-size:12px}}@media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:19px}}}}@media print{{header{{background:white;color:var(--ink)}}nav{{display:none}}section{{break-inside:avoid;border:0;padding:0}}body{{background:white}}.cards{{grid-template-columns:repeat(4,1fr)}}}}
    </style></head><body><header><div class="eyebrow">ROBOIR / PACDM · RELEASE v26</div><h1>Full-body excavator with articulated tracks</h1><p>Hydraulic travel drives, explicit track–sprocket engagement, native material contact, and independently checked rigid-body dynamics.</p><nav><a href="#outcome">Outcome</a><a href="#model">Model</a><a href="#mission">Mission</a><a href="#actuation">Actuation</a><a href="#energy">Energy</a><a href="#validation">Validation</a><a href="#reproduce">Reproduction</a></nav></header><main>
    <section id="outcome" class="status {'fail-state' if overall is False else ''}"><h2>{status_text} {badge(overall)}</h2><p>Report case: <strong>{esc(main_name)}</strong>. Recorded simulation: {number(main.get('duration_s'),1)} s, timestep {number(main.get('dt_s'),6)} s. {passed}/{total} leaf checks passed in the validation file.</p>'''
    if overall is not True:
        content += '<p class="notice">Validation is incomplete. Missing checks and recorded failures must be resolved or retained as explicit limitations before the mobile task is reported as validated.</p>'
    content += '</section><div class="cards">' + ''.join(f'<div class="card"><span class="card-label">{esc(k)}</span><strong class="metric">{esc(v)}</strong><small>{esc(d)}</small></div>' for k,v,d in cards) + '</div>'
    content += '<section id="model"><h2>What is physically modeled</h2><p>The source excavator arm and its closure equations are retained. The original rigid undercarriage is replaced in the compiled scene by an articulated track assembly. Its mass, center of mass and inertia are partitioned at the assembly reference; this is a new multibody undercarriage, not an unchanged per-body inertia model.</p>'
    model_rows = [
        ('Source excavator mass', number(main.get('source_mass_kg'))+' kg'),
        ('Articulated shoes', number(2*tracks['shoes_per_side']) if 'shoes_per_side' in tracks else 'Unavailable'),
        ('Track mechanism', '2 toothed sprockets, 2 idlers, 8 road rollers, 4 carrier rollers, 2 explicit motor rotors'),
        ('Drive mechanism', 'Two pressure- and flow-limited hydraulic drives; torque enters at the sprockets'),
        ('Arm actuation', 'Six hydraulic cylinders and two bounded rotary effort ports; q22 remains passive'),
        ('Native rigid bodies, including material', number(main.get('bodies'))),
        ('Native velocity coordinates / effort ports', number(main.get('coordinates'))+' / '+number(main.get('actuators'))),
        ('Track mass reconstruction error', number(tracks.get('mass_reconstruction_error_kg'))+' kg'),
        ('Track COM reconstruction error', number(tracks.get('com_reconstruction_error_m'))+' m'),
        ('Track inertia reconstruction error', number(tracks.get('inertia_reconstruction_error_kg_m2'))+' kg m²'),
        ('State projection during native integration', number(main.get('native_state_projection'))),
        ('Base propulsion wrench', number(main.get('base_propulsion_wrench')))]
    content += table(['Property', 'Recorded model / scope'], [[esc(k),esc(v)] for k,v in model_rows])
    content += '<p class="notice"><strong>Engineering assumptions:</strong> track dimensions, component inertias, drivetrain and hydraulic parameters are synthetic, not identified specifications of the source machine. The material is coarse, noncohesive rigid aggregate. This release assesses numerical consistency and task execution, not field-calibrated terramechanics, soil cutting resistance, durability, or certified real-time operation.</p></section>'
    content += '<section id="mission"><h2>Collect, transport, turn and unload</h2><p>The mission approaches the pile, performs the source digging and lifting motion, reverses while loaded, turns, unloads into a registered destination, and returns. All robot and material motion comes from native forward integration. Qualified transport and delivery are assessed independently from saved particle identities; the preliminary ever-carried union is not used as the accepted outcome.</p>' + figure_html(*figs[0])
    bay=main.get('scene',{}).get('receiving_bay',{})
    receiver=main.get('scene',{}).get('receiver_plan',{})
    if bay:
        center=main.get('scene',{}).get('depot_center_m',[])
        receiverrows=[('Registered receiver center x / y (m)',' / '.join(number(v,5) for v in center[:2])),
                      ('Interior half dimensions (m)',bay.get('inner_half_size_m')),
                      ('Wall height / thickness (m)',number(bay.get('wall_height_m'))+' / '+number(bay.get('wall_thickness_m'))),
                      ('Wall friction coefficients',bay.get('wall_friction')),
                      ('Floor',bay.get('floor')),('Native receiver contact',bay.get('native_contact')),
                      ('Reference joint-guard margin (rad)',receiver.get('minimum_independent_guard_margin_rad'))]
        content+=table(['Registered receiving bay','Recorded model value'],[[esc(k),esc(number(v))] for k,v in receiverrows])
        content+='<p>The final 90-second mission positions the closed bucket above a receiver registered from feasible geometry, opens and recloses while its mouth reference stays fixed, then retracts and returns. The four fixed walls retain aggregate through ordinary native contact. Reference-path geometry is checked separately from native tracking and particle delivery. The recorded third receiver-center coordinate is the bucket-mouth reference height; the bay floor is the existing ground plane.</p>'
    rows=[]
    for name, r in reports.items():
        task=audited_task(task_audits[name])
        transported=(str(task.get('transported_count'))+' / '+number(task.get('transported_mass_kg'),1)) if task.get('transport_phase_available') else 'Unavailable'
        delivered=(str(task.get('delivered_count'))+' / '+number(task.get('delivered_mass_kg'),1)) if task.get('available') else 'Unavailable'
        loaded=number(task.get('loaded_base_displacement_m'),2) if task.get('transport_phase_available') else 'Unavailable'
        task_status=badge(task.get('passed') if task.get('available') else None)
        if name in ('no_drive','no_bucket_contact'):
            case_checks=[check for check in checks if check.get('case')==name]
            expected_outcome=(all(check.get('passed') is True for check in case_checks)
                              if case_checks and task_audits[name].get('fresh') else None)
            task_status='Expected negative outcome<br>'+badge(expected_outcome)
        rows.append([esc(name),esc(number(r.get('dt_s'),6)),esc(number(r.get('duration_s'),1)),esc(transported),esc(delivered),esc(loaded),esc(number(r.get('final_position_error_m'),3)),esc(task_audits[name]['status']),task_status])
    content += table(['Case','dt (s)','Duration (s)','Transported count / kg','Delivered count / kg','Loaded travel (m)','Final error (m)','Audit inputs','Task gates'],rows)
    content+='<p>The short <code>no_drive</code> and <code>no_bucket_contact</code> cases pass when their registered negative outcomes occur: disconnected drives do not complete the approach, and disabled bucket contact does not lift material. Their full transport and delivery fields are intentionally unavailable.</p>'
    content+='<p>Accepted task metrics require matching SHA-256 hashes for the scene, native trace and simulator report. A transported particle must be inside the lifted bucket at both registered departure and arrival and meet the carried-displacement test. Delivery additionally requires the same qualified particle to be outside the bucket, inside the destination and settled at the final state. These checks cover saved samples; retention between samples is not evaluated.</p>'
    if main_task:
        taskrows=[('Full mission covered',main_task.get('full_mission_covered')),('Departure / arrival / required final time (s)',' / '.join(number(main_task.get(k),1) for k in ['departure_time_s','arrival_time_s','required_final_time_s'])),('Lifted departure particles / mass (kg)',str(main_task.get('departure_count'))+' / '+number(main_task.get('departure_mass_kg'),1)),('Qualified transported particle IDs',main_task.get('transported_particle_ids')),('Qualified delivered particle IDs',main_task.get('delivered_particle_ids')),('Maximum saved-state interval (s)',main_task.get('trace_max_sample_interval_s')),('Delivery settled-speed threshold (m/s)',main_task.get('final_settled_linear_speed_upper_m_s'))]
        content+=table(['Primary independent particle audit','Recorded value'],[[esc(k),esc(number(v))] for k,v in taskrows])
    else:
        content+='<p class="notice">The primary particle audit is '+esc(main_task_audit['status'].lower())+'. Qualified task outcomes are unavailable until the audit is regenerated against the current inputs.</p>'
    content+='<details><summary>Preliminary simulator metrics — excluded from accepted task outcomes</summary>'+table(['Case','Ever-carried union (kg)','Union at destination, final (kg)','Snapshot dynamics audit'],[[esc(name),esc(number(r.get('carried_mass_kg'),1)),esc(number(r.get('delivered_mass_kg'),1)),badge(r.get('audit_passed'))] for name,r in reports.items()])+'<p>These occupancy counters can include particles that did not qualify for loaded transport, or have not settled. A snapshot dynamics pass concerns sampled mechanics identities and does not establish task success.</p></details></section>'
    content += '<section id="actuation"><h2>Actuator effort and full-body force accounting</h2>'+figure_html(*figs[1])+figure_html(*figs[-1])
    control=main.get('scene',{}).get('travel_controller',{})
    controlrows=[('Recorded travel-controller revision',control.get('revision','Unavailable')),
                 ('Outer base-pose/reference period (s)',control.get('outer_reference_period_s')),
                 ('Inner sprocket-speed PI period (s)',control.get('inner_effort_period_s')),
                 ('Shaft torque-request bound (N m)',control.get('command_torque_bound_Nm')),
                 ('Yaw-rate request bound (rad/s)',control.get('yaw_rate_bound_rad_s'))]
    content+=table(['Control configuration','Recorded value'],[[esc(k),esc(number(v))] for k,v in controlrows])
    content+='<p>The production arm and outer base loops use a fixed 100 Hz simulated rate; shaft-speed PI uses 1 kHz. Both rates are unchanged across physics-step refinement. Hydraulic states and native mechanics advance at each physics step. Lateral steering scales with signed reference translation speed and vanishes during pivots; measured yaw-rate feedback damps turning. These are simulated sample rates, not real-time certifications.</p>'
    content += '<p>Body contact wrenches are assembled from native contact forces and their moment arms. Internal tree interaction wrenches are representation-dependent; reaction loads at individual pins of the closed linkage are not uniquely identified.</p></section>'
    content += '<section id="energy"><h2>Energy with the numerical defect exposed</h2>'
    content+='<p><strong>The mechanical midpoint defect is not the full hydraulic-system error.</strong> Chamber storage, finite-step pressure/volume coupling and shaft-work conventions can leave a substantially larger combined defect. The following figures and tables expose both.</p>'
    if len(figs)>3: content+=figure_html(*figs[2])
    midpoint_available='mechanical_midpoint_work' in traces
    mech_abs=main.get('mechanical_midpoint_absolute_work_J') if midpoint_available else {k:main.get('absolute_work_J',{}).get(k,0.) for k in MECHANICAL_KEYS}
    mech_denominator=max(1.,sum(mech_abs.values())) if isinstance(mech_abs,dict) else None
    thermal_abs=main.get('absolute_work_J',{})
    thermal_denominator=max(1.,sum(v for k,v in thermal_abs.items() if k not in ['arm_mechanical','drive_mechanical'])) if thermal_abs else None
    energy_rows=[('Mechanical energy change',main.get('mechanical_energy_change_J')),('Fluid stored-energy change',main.get('fluid_energy_change_J')),('Mechanical midpoint balance defect' if midpoint_available else 'Mechanical rectangular balance defect (legacy)',main.get('mechanical_balance_defect_J')),('Mechanical rectangular balance defect',main.get('mechanical_rectangle_defect_J')),('Total hydraulic + mechanical balance defect',main.get('total_balance_defect_J')),('Drive partition / held-speed coupling defect',main.get('drive_partition_coupling_defect_J')),('Mechanical absolute-activity denominator',mech_denominator),('Total gross-activity denominator',thermal_denominator),('Arm hydraulic supply work',main.get('work_J',{}).get('arm_supply')),('Travel hydraulic supply work',main.get('work_J',{}).get('drive_supply'))]
    content += table(['Energy diagnostic','Recorded value (J)'],[[esc(k),esc(number(v))] for k,v in energy_rows])
    if main_energy.get('available'):
        physical=main_energy.get('energy_change_J',{});arm_fluid=main_energy.get('arm_fluid',{});drive_fluid=main_energy.get('drive_fluid',{});total_energy=main_energy.get('total',{})
        content+='<h3>Independent coupled-energy decomposition</h3><p>Arithmetic consistency '+badge(main_energy.get('accounting_consistency_passed'))+'. An arithmetic pass verifies ledger reconstruction; it is not an energy-accuracy acceptance gate.</p>'
        erows=[('Combined mechanical + fluid energy change',physical.get('combined')),('Arm-fluid stored-energy change',physical.get('arm_fluid')),('Arm-fluid held-speed balance defect',arm_fluid.get('balance_defect_J')),('Drive-fluid held-speed balance defect',drive_fluid.get('balance_defect_J')),('Drive numerical storage dissipation',drive_fluid.get('numerical_storage_dissipation_J')),('Full coupled defect, rectangular environmental work',total_energy.get('rectangle_environment_balance_defect_J')),('Full coupled defect, midpoint environmental work',total_energy.get('midpoint_environment_balance_defect_J'))]
        content+=table(['Independent energy term','Raw value (J)'],[[esc(k),esc(number(v))] for k,v in erows])
        denominator_labels={'absolute_combined_mechanical_and_fluid_energy_change_J':'Absolute net combined stored-energy change','sum_absolute_mechanical_and_fluid_energy_changes_J':'Sum of absolute stored-energy changes','held_speed_actuator_mechanical_activity_J':'Absolute actuator mechanical activity','gross_hydraulic_supply_J':'Gross hydraulic supply','gross_nonmechanical_ledger_activity_J':'Gross nonmechanical ledger activity'}
        fractions=total_energy.get('fractions_by_denominator',{});denominators=main_energy.get('denominator_definitions',{})
        content+=table(['Denominator for the full coupled defect','Denominator (J)','Absolute defect / denominator'],[[esc(label),esc(number(denominators.get(key))),esc(number(100*fractions[key],3)+'%') if key in fractions else 'Unavailable'] for key,label in denominator_labels.items()])
        content+='<p>Arm-fluid defect / absolute arm-fluid stored-energy change: <strong>'+esc(number(100*arm_fluid['relative_to_absolute_stored_energy_change'],3)+'%' if arm_fluid.get('relative_to_absolute_stored_energy_change') is not None else 'Unavailable')+'</strong>. This quantity must not be substituted with the much smaller fraction relative to gross supply work.</p>'
        content+='<details><summary>Separate shaft-work coupling terms</summary>'+table(['Coupling diagnostic','Value (J)'],[[esc(key),esc(number(value))] for key,value in main_energy.get('port_work_coupling',{}).items() if key.endswith('_J')])+'</details>'
    else:
        content+='<p class="notice">A fresh independent coupled-energy decomposition is unavailable for this case. The raw simulator ledger is shown above; the mechanical midpoint result alone does not validate complete hydraulic energy accounting.</p>'
    refinement=[]
    for name,r in reports.items():
        ea=energy_audits[name];total_e=ea.get('total',{});frac=total_e.get('fractions_by_denominator',{}).get('absolute_combined_mechanical_and_fluid_energy_change_J')
        refinement.append([esc(name),esc(number(r.get('dt_s'),6)),esc(number(r.get('duration_s'),1)),esc(number(r.get('mechanical_balance_defect_J'))),esc(number(ea.get('arm_fluid',{}).get('balance_defect_J'))),esc(number(total_e.get('rectangle_environment_balance_defect_J'))),esc(number(100*frac,3)+'%') if frac is not None else 'Unavailable'])
    content+='<h3>Cases and timestep evidence</h3>'+table(['Case','dt (s)','Duration (s)','Mechanical defect (J)','Arm-fluid defect (J)','Full coupled defect (J)','Full defect / net storage'],refinement)
    content+='<p class="muted">Only matching task, parameters and simulation horizons support a timestep-refinement comparison. This table does not turn unlike cases or partial probes into convergence evidence. Refinement acceptance remains in the registered validation checks.</p>'
    if midpoint_available:
        content+='<h3>Mechanical midpoint work</h3>'+table(['Native mechanical channel','Signed work (J)','Absolute activity (J)'],[[esc(k),esc(number(main.get('mechanical_midpoint_work_J',{}).get(k))),esc(number(main.get('mechanical_midpoint_absolute_work_J',{}).get(k)))] for k in MECHANICAL_KEYS])
    else:
        content+='<p class="notice">This legacy trace does not contain midpoint mechanical work. Its mechanical plots and defect use rectangular quadrature. It cannot establish the revised quadrature result.</p>'
    content+='<h3>Held-speed hydraulic and rectangular power ledger</h3>'+table(['Power channel','Signed accumulated work (J)','Absolute activity (J)'],[[esc(k),esc(number(v)),esc(number(thermal_abs.get(k)))] for k,v in main.get('work_J',{}).items()])
    content += '<p class="notice"><strong>Normalization boundary:</strong> a small defect divided by gross absolute activity can coexist with a material error relative to net useful work, especially when hydraulic throttle losses are large. Raw defects, supply work, mechanical work and both denominators are therefore retained. A favorable normalized percentage alone does not establish accurate useful-energy prediction.</p>'
    content += '<p>Mechanical balance includes actuator, passive, equality and contact work. The midpoint work quadrature changes the accounting calculation, not the simulated state. Hydraulic losses, storage discretization and the drive partition coupling defect are shown separately. Pinocchio comparisons verify instantaneous rigid-body quantities; those comparisons do not replace an integrated energy balance or timestep refinement.</p></section>'
    content += '<section id="validation"><h2>Acceptance checks and independent validation</h2><p>PACDM validates the original arm closure and tangent. MuJoCo integrates the compiled free-body contact model. Pinocchio checks the compiled rigid-body tree, including the new tracks, using the native contact and constraint forces as explicit inputs.</p>'
    diag=[('Native arm closure peak (m)',peaks.get('arm_gap')),('Native track closure peak (m)',peaks.get('track_gap')),('Native generalized-force balance relative peak',peaks.get('force_balance')),('PACDM closure maximum',arms.get('pacdm_closure_max')),('PACDM tangent maximum',arms.get('pacdm_tangent_max')),('Native solver warnings',main.get('warning_count')),('Mean measured control evaluation (s)',arms.get('measured_control_mean_wall_s')),('Real-time deadline certified',arms.get('real_time_deadline_certified'))]
    diag.extend([('All seven cases present',complete_case_set),('All case task-audit input hashes match',fresh_case_audits),('All complete-ledger dynamics audits pass with matching input hashes',fresh_dynamics),('All cases use the current controller revision',current_controls),('All cases use the accepted force-residual guard',guarded_cases),('All cases use the final registered receiver scene',current_receivers)])
    content+=table(['Diagnostic','Recorded value'],[[esc(k),esc(number(v))] for k,v in diag])
    content+='<h3>Momentum ledger and softened equality constraints</h3><p>A finite closure gap can produce a small net global couple from native equality forces. This is a numerical soft-constraint artifact, not a physical external load. The raw ideal-closure residual assumes exact internal-force cancellation; the complete ledger explicitly includes the measured equality, other constraint, actuator and passive global wrenches. A complete-ledger pass does not establish exact ideal-linkage momentum conservation.</p>'
    content+=table(['Case','Maximum equality couple norm (N m)','Raw ideal angular residual','Complete angular residual (finite difference)','Complete angular residual (analytic)','Audit inputs'],
                   [[esc(name)]+[esc(number(value)) for value in momentum_summary(item)]+[esc(item['status'])] for name,item in dynamics_audits.items()])
    content+='<p class="muted">Values are maxima over the original audited timestamps; the couple uses the Euclidean norm of its moment vector about the world origin. Independent Pinocchio analytic rates and finite differences are compared with the same complete native wrench, with relative acceptance limit <code>2e-5</code>. Original reports and audits are preserved under <code>outputs/momentum_final_diagnostic/original_audits/</code>; replay leaves the physical model, recorded trajectory and original audit schedule unchanged.</p>'
    content+='<h3>Native solver convergence refinement</h3><p>Each forward solve is checked against a relative generalized force-balance target of <code>1e-8</code>. If needed, at most two additional official native forward solves restart only the numerical acceleration initial guess. Unresolved residuals abort the simulation. Physical positions, velocities, activation, controls, applied forces and time remain unchanged during retries. The registered mission force threshold remains <code>1e-6</code>.</p>'
    solverrows=[]
    for name,r in reports.items():
        s=r.get('numerical_solver',{})
        solverrows.append([esc(name),esc(number(s.get('maximum_initial_residual'))),esc(number(s.get('maximum_final_residual'))),esc(number(s.get('retried_steps'))),esc(number(s.get('total_native_restarts'))),esc(number(s.get('total_cold_restarts')))])
    content+=table(['Case','Maximum initial residual','Maximum accepted residual','Retried steps','Extra native solves','Cold restarts'],solverrows)
    content+='<p>Separate convergence component '+badge(solver_component.get('passed') if solver_component_fresh else None)+': '+esc(number(solver_component.get('checks')))+' tested initial-guess/state cases, maximum accepted residual '+esc(number(solver_component.get('maximum_final_residual')))+'. Component input hashes '+('match' if solver_component_fresh else 'are missing or stale')+'. This test does not establish full-mission success.</p>'
    content+='<p>The native-step parity tests compare the official stepping pipeline with the split forward/integration sequence under matching inputs. Production steps may refine the forward solve before integration. Initial and accepted force residuals quantify the effect of this refinement.</p>'
    content+='<h3>Hydraulic speed-servo component assessment</h3><p>Shared-production-code assessment '+badge(servo.get('passed') if servo_fresh else None)+'. Source hashes '+('match' if servo_fresh else 'are missing or stale')+'. The test uses independent known-inertia flywheels; it does not substitute for full articulated-track validation. Its rejected 100 Hz cases retain the observed low-inertia oscillation.</p>'
    if servo.get('cases'):
        content+=table(['Physics dt (s)','Servo period (s)','Flywheel inertia (kg m²)','Final RMS speed error (rad/s)','Shaft-work coupling (J)','Case status'],[[esc(number(c.get('dt_s'),6)),esc(number(c.get('servo_period_s'),6)),esc(number(c.get('inertia_kg_m2'),1)),esc(number(c.get('final_rms_error_rad_s'),6)),esc(number(c.get('partition_coupling_J'))),badge(c.get('passed') if servo_fresh else None)] for c in servo['cases']])
    content+='<p>This report reads the available case and component records. Each result retains its recorded status, and provenance checks identify the source and trajectory versions used in the assessment.</p>'
    if failed:
        content+='<h3>Recorded failures</h3>'+table(['Check','Value','Limit','Status'],[[esc(x.get('name','')),esc(number(x.get('value'))),esc(number(x.get('limit'))),badge(False)] for x in failed])
    if checks:
        content+='<details><summary>All '+str(total)+' recorded leaf checks</summary>'+table(['Check','Case / scope','Value','Limit','Status'],[[esc(x.get('name','')),esc(x.get('case',x.get('scope',''))),esc(number(x.get('value'))),esc(number(x.get('limit'))),badge(x.get('passed'))] for x in checks])+'</details>'
    else:
        content+='<p class="notice">The final validation file is unavailable or contains no recognizable leaf checks. Validation is incomplete.</p>'
    content+='<h3>Evidence boundaries</h3><ul><li>Independent dynamics agreement is agreement for this compiled model, not proof that assumed track or soil properties represent the actual excavator.</li><li>One successful material-transfer cycle does not establish robustness across unspecified terrains, payloads or operating speeds.</li><li>Negative controls and perturbations must be judged against their own declared outcomes; failure to complete the task may be the intended result of a negative control.</li><li>Numerical simulation time and measured wall time are separate. Hard real-time execution is outside this scope.</li></ul></section>'
    content += '<section id="reproduce"><h2>Inspect and reproduce</h2>'
    content += '<p>Clone the repository and keep its directory layout. The repository includes model/configuration assets, simulation and controller code, audit routines, plotting code, and replay renderers. Native trajectories, reports, figures, and videos are generated when the corresponding commands are run.</p>'
    content += '<p>Install <code>Excavator_CMG_PACDM/requirements-tracked.txt</code> in a Python 3.12 environment. From <code>Excavator_CMG_PACDM/v26</code>, run <code>python run_tracked_excavator.py --case nominal</code> to generate the nominal tracked trajectory and its sampled dynamics audit; <code>tracked_task_audit.py</code>, <code>tracked_report.py</code> and <code>tracked_video.py</code> consume the saved trajectory.</p>'
    content += '<p>Source integrity is checked against the <code>source/Excavator_RoboIR_full_body_v21/SOURCE_CODE_SHA256.json</code> manifest. The historical release manifest is retained separately for provenance. This optional HTML view retains the original acceptance logic and shows missing historical suite evidence as unassessed.</p>'
    content += '<ul><li><a href="results/'+esc(main_name)+'.json">Primary case metrics (JSON)</a></li><li><a href="results/'+esc(main_name)+'.npz">Native trajectory (NPZ)</a></li><li><a href="results/'+esc(main_name)+'_audit.json">Independent dynamics snapshots (JSON)</a></li><li><a href="results/'+esc(main_name)+'_task_audit.json">Particle/body audit (JSON)</a></li><li><a href="results/assets/'+esc(main_name)+'.xml">Compiled MuJoCo scene (XML)</a></li><li><a href="../../README.md">Repository instructions</a></li></ul>'
    content += '<p>From <code>v26/</code>, regenerate the selected figures with <code>python tracked_report.py --figures-only --cases '+esc(' '.join(reports))+'</code>. Rebuild this optional HTML view with <code>python tracked_report.py --cases '+esc(' '.join(reports))+'</code>, or render the selected saved trajectory with <code>python tracked_video.py --case '+esc(main_name)+'</code>. Plotting and video replay do not change the simulated trajectory or acceptance results.</p></section>'
    content+=f'<footer>Generated {generated} directly from recorded files. Report case: {esc(main_name)}. No missing or failed result has been converted into a pass.</footer></main></body></html>'
    path=ROOT/'Excavator_Tracked_v26_Report.html'
    path.write_text(content,encoding='utf-8')
    inputs={}
    for file in [vp,RESULTS/'travel_servo_validation.json',ROOT/'outputs/native_solver_accuracy/report.json',ROOT/'tracked_report.py']:
        if file.is_file():inputs[file.relative_to(ROOT).as_posix()]=sha256(file)
    for name in reports:
        for file in [RESULTS/f'{name}.json',RESULTS/f'{name}.npz',RESULTS/f'{name}_audit.json',RESULTS/f'{name}_task_audit.json',RESULTS/'assets'/f'{name}.xml']:
            if file.is_file():inputs[file.relative_to(ROOT).as_posix()]=sha256(file)
    manifest=dict(report=path.name,report_sha256=sha256(path),cases=list(reports),overall_passed=overall,input_sha256=inputs)
    (RESULTS/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(path,flush=True)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='*')
    p.add_argument('--validation',default='validation.json')
    p.add_argument('--figures-only', action='store_true',
                   help='Regenerate figures from selected existing case JSON/NPZ files without the historical suite HTML')
    args=p.parse_args()
    if args.figures_only:
        make_figures_only(args.cases)
    else:
        make_report(args.cases,args.validation)
