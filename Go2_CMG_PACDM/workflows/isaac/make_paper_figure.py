"""Regenerate the six-panel paper figure from recorded native PhysX traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / 'records' / '20260924T143514Z_e63dcb00'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_figure_data(records):
    """Require complete, aligned positive cases with a shared run/reference."""
    records = Path(records)
    arrays, identities = {}, []
    for name in ('nominal', 'payload', 'fine'):
        case = json.loads((records/name/'case.json').read_text(encoding='utf-8'))
        if case.get('completed') is not True or case.get('simulated_s') != 26.0:
            raise ValueError(f'{name}: complete 26-second execution required')
        if case.get('case', case.get('name')) != name or 'PhysX' not in case.get('engine', ''):
            raise ValueError(f'{name}: wrong case identity or physics engine')
        identities.append(case.get('run_id'))
        with np.load(records/name/'trajectory.npz', allow_pickle=False) as source:
            arrays[name] = {key: source[key].copy() for key in
                            ('time', 'q', 'q_ref', 'feet', 'cmg_feet', 'measured_support_force')}
        data = arrays[name]
        expected = {'time': (2601,), 'q': (2601, 18), 'q_ref': (2601, 18),
                    'feet': (2601, 4, 3), 'cmg_feet': (2601, 4, 3),
                    'measured_support_force': (2601, 4)}
        for key, shape in expected.items():
            if data[key].shape != shape or not np.all(np.isfinite(data[key])):
                raise ValueError(f'{name}/{key}: invalid shape or nonfinite values')
        if not np.allclose(data['time'], np.arange(2601)*.01, rtol=0, atol=1e-10):
            raise ValueError(f'{name}: expected 10-ms samples over 0--26 seconds')
    if not identities[0] or len(set(identities)) != 1:
        raise ValueError('Figure cases must share one recorded run identity')
    for name in ('payload', 'fine'):
        if not np.array_equal(arrays[name]['time'], arrays['nominal']['time']):
            raise ValueError('Figure cases must have identical recorded timestamps')
        if not np.array_equal(arrays[name]['q_ref'], arrays['nominal']['q_ref']):
            raise ValueError('Figure cases must use an identical PACDM reference')
    return arrays, identities[0]


def make_figure(records, output):
    records, out = Path(records).resolve(), Path(output).resolve()
    if out == records or records in out.parents:
        raise ValueError('Figure output must be outside the recorded evidence directory')
    data, run_id = load_figure_data(records)
    n, p, f = (data[name] for name in ('nominal', 'payload', 'fine'))
    out.mkdir(parents=True, exist_ok=True)
    t=n['time']; q=n['q']; ref=n['q_ref']
    err=np.linalg.norm(q[:,:3]-ref[:,:3],axis=1)*1000
    errp=np.linalg.norm(p['q'][:,:3]-p['q_ref'][:,:3],axis=1)*1000
    fk=np.linalg.norm(n['feet']-n['cmg_feet'],axis=2).max(axis=1)*1e6
    delta=np.linalg.norm(q[:,:3]-f['q'][:,:3],axis=1)*1000
    moving=t>=2
    teal='#007F7C'; navy='#214A73'; orange='#D47B22'; purple='#76579C'; black='#272727'
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif','font.size':9,'axes.titlesize':9.4,'axes.labelsize':9,'xtick.labelsize':8.4,'ytick.labelsize':8.4,'legend.fontsize':8.2,'axes.linewidth':.65,'lines.linewidth':1.05,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.axisbelow':True})
    fig, axes=plt.subplots(3,2,figsize=(7.4,6.85))
    fig.subplots_adjust(left=.095,right=.985,top=.963,bottom=.075,wspace=.27,hspace=.46)
    for ax in axes.flat:
     ax.spines[['top','right']].set_visible(False)
     ax.grid(True,color='#d9dfe3',lw=.45,alpha=.75)
     ax.tick_params(direction='out',length=3,width=.6,pad=3)
     ax.yaxis.set_major_locator(MaxNLocator(5))
    def title(ax,label,text): ax.set_title(f'({label}) {text}',loc='left',fontweight='bold',pad=7)
    def time_axis(ax):
     ax.set_xlabel('Time (s)');ax.set_xlim(0,26);ax.set_xticks([0,5,10,15,20,25])
     for a,b in [(1.2,1.35),(24.3,24.5)]:ax.axvspan(a,b,color='#717c82',alpha=.19,lw=0,zorder=0)
    ax=axes[0,0];title(ax,'a','Horizontal base path')
    ax.plot(ref[:,0],ref[:,1],color=black,lw=1.6,ls=(0,(4,2.5)),label='PACDM reference',zorder=3)
    ax.plot(q[:,0],q[:,1],color=teal,lw=.85,label='Isaac Sim',zorder=4)
    ax.set(xlabel='World x (m)',ylabel='World y (m)',xlim=(-.04,2.25),ylim=(-.025,.335))
    ax.legend(loc='upper left',frameon=False,handlelength=2.1)
    ax=axes[0,1];title(ax,'b','Base height')
    ax.plot(t,ref[:,2],color=black,lw=1.5,ls=(0,(4,2.5)),label='PACDM reference',zorder=3)
    ax.plot(t,q[:,2],color=teal,lw=.9,label='Isaac Sim',zorder=4)
    ax.set_ylabel('Height (m)');ax.set_ylim(.25,.343);time_axis(ax)
    ax.legend(loc='lower left',frameon=False,handlelength=2.1)
    ax=axes[1,0];title(ax,'c','Base-position error')
    ax.plot(t,err,color=teal,label=f'Nominal: {np.sqrt(np.mean(err[moving]**2)):.3f} mm')
    ax.plot(t,errp,color=orange,label=f'1.5-kg payload: {np.sqrt(np.mean(errp[moving]**2)):.3f} mm',lw=.9)
    ax.set_ylabel('Euclidean error (mm)');ax.set_ylim(0,35.5);time_axis(ax)
    ax.legend(loc='upper left',frameon=False,handlelength=1.8,fontsize=7.7,title=r'RMS over $t\geq 2$ s',title_fontsize=7.7,alignment='left')
    ax=axes[1,1];title(ax,'d','Measured normal foot support')
    sel=(t>=6)&(t<=7.6)
    for j,(label,color) in enumerate(zip(['FL','FR','RL','RR'],[teal,navy,orange,purple])):
     ax.plot(t[sel],n['measured_support_force'][sel,j],color=color,lw=.9,label=label)
    ax.set(xlabel='Time (s)',ylabel='Normal support (N)',xlim=(6,7.6))
    ax.set_ylim(0,max(220,float(n['measured_support_force'][sel].max())*1.19))
    ax.set_xticks([6,6.4,6.8,7.2,7.6]);ax.legend(ncol=4,loc='upper right',frameon=False,columnspacing=.65,handlelength=1.3,fontsize=8)
    ax=axes[2,0];title(ax,'e','PhysX / CMG foot consistency')
    ax.plot(t,fk,color=navy,lw=.8)
    ax.set_ylabel('Maximum foot discrepancy ($\\mu$m)');ax.set_ylim(0,.95);time_axis(ax)
    ax.text(.025,.94,f'Maximum {fk.max():.3f} $\\mu$m',transform=ax.transAxes,va='top',fontsize=8.3)
    ax=axes[2,1];title(ax,'f','Timestep refinement')
    ax.plot(t,delta,color=purple,lw=.85)
    ax.set_ylabel('Base-path difference (mm)');ax.set_ylim(0,9.2);time_axis(ax)
    ax.text(.025,.95,f'1 ms vs 0.5 ms\\nMaximum {delta.max():.3f} mm; RMS {np.sqrt(np.mean(delta**2)):.3f} mm'.replace('\\n','\n'),transform=ax.transAxes,va='top',fontsize=8.2,linespacing=1.4)
    fig.savefig(out/'Go2_Isaac_Execution.pdf',metadata={'Title':'Unitree Go2: independent Isaac Sim / PhysX execution','Creator':'Matplotlib; plotted from recorded full-run trajectories','Subject':'26-second CMG-PACDM execution, physical contact, kinematic consistency, and timestep refinement'})
    fig.savefig(out/'Go2_Isaac_Execution.png',dpi=240)
    plt.close(fig)
    metrics = {'samples': len(t), 'course_rms_interval_s': [2.0, 26.0],
               'refinement_interval_s': [0.0, 26.0],
               'rms_nominal_mm': float(np.sqrt(np.mean(err[moving]**2))),
               'rms_payload_mm': float(np.sqrt(np.mean(errp[moving]**2))),
               'max_fk_um': float(fk.max()),
               'refinement_max_mm': float(delta.max()),
               'refinement_rms_mm': float(np.sqrt(np.mean(delta**2))),
               'support_window_s': [6.0, 7.6],
               'support_max_N': float(n['measured_support_force'][sel].max()),
               'support_definition': 'Sum of absolute normal contact impulse components / physics step',
               'run_id': run_id,
               'trajectory_sha256': {name: file_hash(records/name/'trajectory.npz')
                                     for name in ('nominal', 'payload', 'fine')}}
    (out/'Go2_Isaac_Execution_metrics.json').write_text(
        json.dumps(metrics, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=DEFAULT_RECORDS,
                        help='Full-run directory containing nominal, payload and fine cases')
    parser.add_argument('--output', type=Path, default=ROOT/'figures',
                        help='Figure output directory, outside the recorded run')
    args = parser.parse_args(argv)
    try:
        metrics = make_figure(args.records, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Figure generation failed: {error}\n')
    print(json.dumps(metrics, indent=2))
    print(f'Figure: {args.output.resolve()/"Go2_Isaac_Execution.pdf"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
