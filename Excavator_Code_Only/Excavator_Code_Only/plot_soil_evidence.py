#!/usr/bin/env python3
"""Generate four soil-evidence figures from the two completed native runs.

Usage: python plot_soil_evidence.py --outputs outputs --output figures
Place this script beside run_soil.py. First run the soil_demo and soil_final
recipes, then plot their generated trajectory.npz, report.json and scene.xml.
Default input and output paths resolve relative to this script. Explicit
relative CLI paths resolve relative to the current working directory.

Dependencies: Python 3, NumPy and Matplotlib (requirements-plots.txt).
Uses exact saved samples; imports neither MuJoCo nor Pinocchio and performs
no simulation, smoothing or interpolation. Array units, final relative-energy
normalization, and sampled versus physics-step peaks are recorded in the
generated summary_metrics.json. No result files are required in the source
package: the original simulation drivers generate the inputs locally.
"""
from pathlib import Path
import argparse
import csv
import json
import string
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
CASES = [('soil_demo', 'Filled bed'), ('soil_final', 'Exposed face')]
COLORS = ['#2166ac', '#d97720', '#258653', '#9a4eaa', '#b44747', '#607d8b']
INK = '#203040'
PLOT_FIELDS = ('time', 'lip_position', 'bucket_mass', 'lifted_mass',
               'deposited_mass', 'ctrl', 'pressure_max', 'closure_error',
               'force_residual', 'soil_bucket_force', 'energy', 'mechanical_work')


def load_cases(folder):
    cases = []
    for key, title in CASES:
        case_folder = folder / key
        required = [case_folder / name for name in ('trajectory.npz', 'report.json', 'scene.xml')]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError('Run the matching soil simulation before plotting. Missing: '
                                    + ', '.join(missing))
        with np.load(case_folder / 'trajectory.npz', allow_pickle=False) as f:
            missing_fields = sorted(set(PLOT_FIELDS) - set(f.files))
            if missing_fields:
                raise ValueError(f'{key}: missing trajectory fields {missing_fields}')
            d = {k: f[k].copy() for k in PLOT_FIELDS}
        r = json.loads((folder / key / 'report.json').read_text())
        if r.get('status') != 'COMPLETED':
            raise ValueError(f"{key}: this completed-task figure set requires a COMPLETED run; "
                             f"report status is {r.get('status')!r}. Inspect report.json.")
        scene = ET.parse(case_folder / 'scene.xml').getroot()
        actuator_element = scene.find('actuator')
        if actuator_element is None:
            raise ValueError(f'{key}: scene.xml has no direct actuator section')
        m = {'actuators': [{'index': i, 'type': actuator.tag, **actuator.attrib}
                          for i, actuator in enumerate(actuator_element)]}
        t = d['time']
        if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
            raise ValueError(f'{key}: timestamps must strictly increase')
        if not np.allclose(np.diff(t), .04, atol=1e-9, rtol=0):
            raise ValueError(f'{key}: figure sampling labels require 0.04 s saved intervals')
        for name, values in d.items():
            if values.shape[0] != len(t) or not np.isfinite(values).all():
                raise ValueError(f'{key}: {name} has invalid sample count or nonfinite values')
        expected = ['effort_p0','effort_p1','effort_p2','effort_p3','effort_p4',
                    'effort_p5','effort_q21','effort_q23','track_drive_right','track_drive_left']
        if [a['name'] for a in m['actuators']] != expected:
            raise ValueError(f'{key}: actuator ordering differs from this plotting schema')
        for actuator in m['actuators']:
            gear = np.fromstring(actuator.get('gear', '1'), sep=' ')
            if actuator['type'] != 'motor' or not np.array_equal(gear, [1.]):
                raise ValueError(f'{key}: effort plots require unit-gear native motors')
        for name, columns in [('lip_position',3),('ctrl',10),('soil_bucket_force',3),
                              ('energy',2),('mechanical_work',4)]:
            if d[name].shape != (len(t), columns):
                raise ValueError(f'{key}: {name} must have {columns} columns')
        if len(r.get('events', [])) != 11:
            raise ValueError(f'{key}: expected eleven completed soil-phase events')
        cases.append((key, title, d, r, m))
    return cases


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 11, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.6, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#87929c', 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'grid.color': '#d9e0e6', 'grid.linewidth': .55,
        'axes.grid': True, 'axes.axisbelow': True,
        'lines.linewidth': 1.35, 'pdf.fonttype': 42,
        'savefig.facecolor': 'white',
    })


def panel_labels(axs):
    for letter, ax in zip(string.ascii_lowercase, np.asarray(axs).flat):
        ax.text(-.13, 1.035, f'({letter})', transform=ax.transAxes,
                weight='bold', fontsize=10, va='bottom')


def time_axes(ax, d):
    ax.set_xlim(0, float(d['time'][-1]))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel('Simulation time (s)')


def save(fig, out, name, footnote):
    fig.text(.08, .012, footnote, fontsize=7.2, color='#596774', va='bottom')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=.14)
    fig.savefig(out / f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=.14)
    plt.close(fig)


def plot_mission(cases, out):
    fig, axs = plt.subplots(2, 2, figsize=(10.7, 7.25))
    fig.subplots_adjust(left=.10, right=.985, top=.90, bottom=.105, hspace=.40, wspace=.23)
    fig.suptitle('Soil mission: native bucket motion and material transfer', fontsize=14, weight='bold', y=.986)
    for c, (_, title, d, r, _) in enumerate(cases):
        ax = axs[0,c]; xyz = d['lip_position']; t = d['time']
        ax.set_title(title, pad=13, weight='bold')
        ax.plot(xyz[:,0], xyz[:,2], color=COLORS[0], lw=1.5, label='Actual lip path')
        ax.scatter(xyz[0,0], xyz[0,2], s=30, color=COLORS[2], zorder=4, label='Start')
        ax.scatter(xyz[-1,0], xyz[-1,2], s=38, marker='s', facecolors='white', edgecolors=COLORS[1], zorder=4, label='End')
        ax.axhline(0, color='#75818b', lw=.9, ls=':', label='Surface height')
        ax.set_xlabel('World lip x (m)'); ax.set_ylabel('World lip z (m)')
        ax.set_ylim(-.46, 1.49); ax.set_xlim(2.8, 7.8)
        ax.legend(loc='lower left', ncol=2)
        ax = axs[1,c]
        ax.plot(t,d['bucket_mass'],color=COLORS[0],label='Bucket occupancy')
        ax.plot(t,d['lifted_mass'],color=COLORS[5],ls='--',label='Ever lifted')
        ax.plot(t,d['deposited_mass'],color=COLORS[2],label='Settled delivery')
        # Events mark the end of a phase, not the time it first appears in the trace.
        for idx, short in [(2,'Draw ends'),(6,'Slew ends')]:
            e = r['events'][idx]['time_s']
            ax.axvline(e, color='#a5adb5', lw=.65, ls=':')
            ax.text(e+.35, 133, short, fontsize=7, color='#77818c', rotation=90, va='top')
        time_axes(ax,d); ax.set_ylabel('Material mass (kg)'); ax.set_ylim(-4,137)
        ax.legend(loc='lower left', bbox_to_anchor=(0,1.015), ncol=3,
                  borderaxespad=0, columnspacing=.85, handlelength=1.8, fontsize=7.4)
        ax.text(.98,.045,f"Delivered: {r['deposited_mass_kg']:.3f} kg",ha='right',va='bottom',transform=ax.transAxes,weight='bold',fontsize=8.5)
    panel_labels(axs)
    save(fig,out,'soil_mission','Exact 25 Hz saved native samples. Upper panels show x–z projections of the three-dimensional bucket-lip motion.')


def plot_actuation(cases, out):
    fig, axs = plt.subplots(4,2,figsize=(10.7,10.8))
    fig.subplots_adjust(left=.10,right=.985,top=.925,bottom=.073,hspace=.51,wspace=.23)
    fig.suptitle('Actuation: physical efforts applied to the native plant',fontsize=14,weight='bold',y=.982)
    cyl_labels = ['p0: tilt 1','p1: tilt 2','p2: bucket','p3: boom 1','p4: boom 2','p5: stick']
    for c,(_,title,d,r,m) in enumerate(cases):
        t=d['time']; u=d['ctrl']
        axs[0,c].set_title(title,pad=13,weight='bold')
        for j in range(6):
            axs[0,c].plot(t,u[:,j]/1e3,color=COLORS[j],ls='--' if j==4 else '-',label=cyl_labels[j])
        axs[0,c].set_ylabel('Cylinder force (kN)'); axs[0,c].set_ylim(-360,590)
        axs[0,c].legend(loc='upper right',ncol=3,columnspacing=.6,handlelength=1.7,fontsize=7)
        for j,label,color in [(6,'q21: tiltrotator',COLORS[3]),(7,'q23: slew',COLORS[0])]:
            axs[1,c].plot(t,u[:,j]/1e3,color=color,label=label)
        axs[1,c].set_ylabel('Upper rotary torque (kN m)'); axs[1,c].set_ylim(-68,57)
        axs[1,c].legend(loc='lower right')
        for j,label,color in [(8,'Right sprocket',COLORS[1]),(9,'Left sprocket',COLORS[2])]:
            axs[2,c].plot(t,u[:,j]/1e3,color=color,label=label)
        axs[2,c].set_ylabel('Travel torque (kN m)'); axs[2,c].set_ylim(-7,6)
        axs[2,c].legend(loc='lower right')
        axs[3,c].plot(t,d['pressure_max']/1e6,color=COLORS[0],label='Maximum arm chamber pressure')
        axs[3,c].axhline(25,color='#8b6670',ls='--',lw=.9,label='Model supply pressure')
        axs[3,c].set_ylim(-.8,28); axs[3,c].set_ylabel('Pressure (MPa)')
        axs[3,c].legend(loc='lower right',fontsize=7.2)
        for ax in axs[:,c]:time_axes(ax,d)
    panel_labels(axs)
    save(fig,out,'soil_actuation','Exact 25 Hz saved applied efforts with unit motor gears. p3/p4 overlap. Pressure is the maximum of 12 arm-cylinder chambers.')


def plot_forces_closure(cases,out):
    fig,axs=plt.subplots(3,2,figsize=(10.7,8.7))
    fig.subplots_adjust(left=.10,right=.985,top=.915,bottom=.085,hspace=.48,wspace=.23)
    fig.suptitle('Native plant: closure, force balance and bucket contact',fontsize=14,weight='bold',y=.984)
    for c,(_,title,d,r,_) in enumerate(cases):
        t=d['time'];axs[0,c].set_title(title,pad=13,weight='bold')
        axs[0,c].plot(t,d['closure_error']*1e6,color=COLORS[0])
        axs[0,c].set_ylabel('Arm point-closure gap (µm)');axs[0,c].set_ylim(-.15,5.65)
        axs[0,c].text(.985,.97,f"Reported physics-step maximum: {r['peak']['closure_m']*1e6:.3f} µm",transform=axs[0,c].transAxes,ha='right',va='top',fontsize=7.6)
        axs[1,c].semilogy(t,d['force_residual'],color=COLORS[0],lw=.8)
        axs[1,c].set_ylabel('Force-balance residual (relative)');axs[1,c].set_ylim(4e-17,3e-13)
        axs[1,c].text(.985,.965,f"Reported physics-step maximum: {r['peak']['force_residual']:.3e}",transform=axs[1,c].transAxes,ha='right',va='top',fontsize=7.6,
                       bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=2))
        axs[2,c].plot(t,np.linalg.norm(d['soil_bucket_force'],axis=1)/1e3,color=COLORS[1])
        axs[2,c].set_ylabel('Bucket–grain net force (kN)');axs[2,c].set_ylim(-2,72)
        axs[2,c].text(.985,.965,f"Saved-sample peak: {r['peak']['soil_bucket_force_N']/1e3:.3f} kN",transform=axs[2,c].transAxes,ha='right',va='top',fontsize=7.6)
        for ax in axs[:,c]:time_axes(ax,d)
    panel_labels(axs)
    save(fig,out,'soil_forces_closure','Curves: exact 25 Hz samples. Closure/force-balance annotations: report maxima from 1 ms physics steps; these need not occur at saved sample times.')


def plot_energy(cases,out):
    fig,axs=plt.subplots(2,2,figsize=(10.7,6.9))
    fig.subplots_adjust(left=.10,right=.985,top=.895,bottom=.135,hspace=.46,wspace=.23)
    fig.suptitle('Mechanical energy and cumulative signed work',fontsize=14,weight='bold',y=.982)
    for c,(_,title,d,r,_) in enumerate(cases):
        t=d['time']; E=d['energy'].sum(axis=1);delta=E-E[0];W=d['mechanical_work'].sum(axis=1);defect=delta-W
        axs[0,c].set_title(title,pad=13,weight='bold')
        axs[0,c].plot(t,delta/1e3,color=COLORS[0],lw=2,label='Mechanical energy change')
        axs[0,c].plot(t,W/1e3,color=COLORS[1],ls='--',lw=1.1,label='Total signed work')
        axs[0,c].set_ylabel('Energy / work (kJ)');axs[0,c].set_ylim(-24,52)
        axs[0,c].legend(loc='upper left',fontsize=8)
        axs[1,c].axhline(0,color='#7b8590',lw=.75)
        axs[1,c].plot(t,defect,color=COLORS[3],lw=1)
        axs[1,c].set_ylabel('Signed work–energy defect (J)');axs[1,c].set_ylim(-8,15.5)
        axs[1,c].text(.985,.96,f"Terminal defect: {defect[-1]:+.3f} J\nReported terminal relative defect: {r['mechanical_energy_relative_defect']:.3e}",transform=axs[1,c].transAxes,ha='right',va='top',fontsize=7.8,
                       bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
        for ax in axs[:,c]:time_axes(ax,d)
    panel_labels(axs)
    save(fig,out,'soil_energy','Defect = Δ(kinetic + potential energy) − signed work of actuators, equality constraints, contacts and passive forces.\nReported terminal relative defect uses the full-rate accumulated absolute work activity over these four categories.')


def summary(cases,out):
    rows=[]
    for key,title,d,r,m in cases:
        E=d['energy'].sum(axis=1);defect=E-E[0]-d['mechanical_work'].sum(axis=1)
        rows.append(dict(case=key,label=title,saved_samples=len(d['time']),sample_period_s=float(np.median(np.diff(d['time']))),
            simulated_duration_s=float(d['time'][-1]),lifted_material_kg=float(d['lifted_mass'][-1]),
            settled_delivery_kg=float(d['deposited_mass'][-1]),
            maximum_bucket_occupancy_kg=float(d['bucket_mass'].max()),
            sampled_arm_gap_m=float(d['closure_error'].max()),reported_physics_step_arm_gap_m=r['peak']['closure_m'],
            sampled_force_balance_relative=float(d['force_residual'].max()),reported_physics_step_force_balance_relative=r['peak']['force_residual'],
            sampled_bucket_grain_net_force_N=float(np.linalg.norm(d['soil_bucket_force'],axis=1).max()),
            sampled_arm_chamber_pressure_Pa=float(d['pressure_max'].max()),
            mechanical_energy_change_J=float(E[-1]-E[0]),cumulative_signed_mechanical_work_J=float(d['mechanical_work'][-1].sum()),
            terminal_work_energy_defect_J=float(defect[-1]),sampled_minimum_signed_defect_J=float(defect.min()),
            sampled_maximum_signed_defect_J=float(defect.max()),reported_terminal_relative_defect=r['mechanical_energy_relative_defect'],
            pacdm_continuation_steps_including_initialization=r['arm_controller']['continuation_steps'],
            pacdm_reacquisition_fallbacks=r['arm_controller']['fallback_count'],
            maximum_polished_pacdm_augmented_residual=r['arm_controller']['pacdm_closure_max'],
            maximum_pacdm_tangent_defect=r['arm_controller']['pacdm_tangent_max'],
            maximum_pacdm_point_tangent_difference=r['arm_controller']['pacdm_point_tangent_difference_max'],
            maximum_relative_inverse_equilibrium=r['arm_controller']['inverse_equilibrium_relative_max']))
        if not np.isclose(defect[-1],r['mechanical_energy_defect_J'],atol=1e-9,rtol=0):
            raise ValueError(f'{key}: report and trajectory work-energy endpoints disagree')
    document={'records':rows,'definitions':{
        'sampled':'Exact saved samples, nominally 0.04 s apart; no interpolation, smoothing or physics recomputation.',
        'reported_physics_step_maxima':'Stored report.json peaks; may exceed the saved-sample maxima.',
        'energy_columns':['potential_J','kinetic_J'],
        'mechanical_work_columns':['actuators_J','native_equality_J','native_contact_J','native_passive_J'],
        'relative_defect':'abs(terminal defect) / max(1, sum over physics steps and four work categories of dt*abs(category power)); only terminal value is available.',
        'pacdm':'Aggregate reported maxima only; no PACDM time series is present in the trajectory.',
        'controller_initialization':'One arm evaluation occurs before the loop and another at t=0; both are included in continuation_steps.',
        'phase_events':'Events mark completion of the named phase; trace phase is saved before transition.'}}
    (out/'summary_metrics.json').write_text(json.dumps(document,indent=2)+'\n')
    with (out/'summary_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs',type=Path,default=ROOT/'outputs',
                   help='Folder containing generated soil_demo and soil_final case folders')
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    a=p.parse_args()
    setup();cases=load_cases(a.outputs)
    a.output.mkdir(parents=True,exist_ok=True)
    plot_mission(cases,a.output);plot_actuation(cases,a.output)
    plot_forces_closure(cases,a.output);plot_energy(cases,a.output)
    summary(cases,a.output)
    print(f'Created four PDF/PNG figure pairs and summary_metrics.json/CSV in {a.output.resolve()}')


if __name__=='__main__':main()
