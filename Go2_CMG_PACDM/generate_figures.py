#!/usr/bin/env python3
"""Recompute Go2 evidence and generate four publication figures from local run data.

Run in this package: python generate_figures.py
Dependencies: numpy, scipy, matplotlib. No contact simulator is needed to plot.
Reference residuals, ranks, and saved scalar metrics are checked; input hashes
are recorded alongside the derived diagnostics.
The saved simulation trajectories are post-processed; no rollout is synthesized.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys

import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from go2.contact import ContactGraph, FootKinematics
from vendor.pacdm_original import rank

IN=ROOT; OUT=ROOT/'figures'; DER=ROOT/'derived'
CASES=['nominal','fine','low_friction','payload','strong_push']
LABELS=['Nominal','0.5-ms\nstep',r'$\mu=0.55$','1.5-kg\npayload','25% larger\npushes']
LEGS=['FL','FR','RL','RR']
LIMITS=np.tile([23.7,23.7,45.43],4)
INK='#203B50'; BLUE='#176A91'; ORANGE='#D48626'; GREEN='#228678'; RED='#AA485A'
COLORS=[BLUE,ORANGE,GREEN,'#8863A2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.titlesize':9,'axes.titleweight':'bold','axes.labelsize':8,
 'xtick.labelsize':7.2,'ytick.labelsize':7.2,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#ADB9C1',
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
 'grid.color':'#E5EAED','grid.linewidth':.5,'axes.linewidth':.65,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def load_npz(name,folder='results'):
    with np.load(IN/folder/(name+'.npz'),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}
def load_json(name,folder='results'):
    return json.loads((IN/folder/(name+'.json')).read_text())
def verify_inputs():
    required=['data/go2_cmg.json','data/reference.npz',
              'results/reference.json','results/task_validation.json']
    required += ['results/'+name+'.'+ext
                 for name in CASES+['PD_ablation'] for ext in ['npz','json']]
    missing=[name for name in required if not (IN/name).is_file()]
    if missing:
        raise FileNotFoundError('Run python run_go2.py --workers 3 first. Missing: '
                                + ', '.join(missing))
    hashes={name:hashlib.sha256((IN/name).read_bytes()).hexdigest() for name in required}
    hashes.update({'source/'+name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                   for name in ['go2/contact.py','vendor/pacdm_original.py']})
    (DER/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return len(hashes)
def save(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.045)
    plt.close(fig)
def axis(ax,t=(0,26),push=False):
    ax.set_xlim(*t); ax.set_xlabel('Time [s]'); ax.grid(axis='y',zorder=0)
    if t==(0,26): ax.set_xticks([0,5,10,15,20,25])
    if push:
        for a,b in [(1.20,1.35),(24.30,24.50)]:
            ax.axvspan(a,b,color=ORANGE,alpha=.22,lw=0,zorder=0)
def boxnote(ax,text,xy=(.03,.93)):
    ax.text(*xy,text,transform=ax.transAxes,ha='left',va='top',fontsize=7,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=2))
def compare(got,expected,name,atol=1e-12,rtol=1e-9):
    if not np.isclose(got,expected,atol=atol,rtol=rtol):
        raise ValueError(f'{name}: computed {got} does not match saved {expected}')

def reference_diagnostics():
    ref=load_npz('reference','data'); cmg=load_json('go2_cmg','data')
    q,v,a,N=(ref[x] for x in ('q','v','a','N')); n=len(q)
    kin=FootKinematics(cmg,q[0]); graph=ContactGraph(cmg,q[0])
    desired_map=np.c_[np.zeros((12,6)),np.eye(12)]
    ids=np.unique(np.linspace(0,n-1,min(151,n),dtype=int)); idset=set(ids.tolist())
    rpos=[];rtan=[];rvel=[];rc=[];rf=[];rd=[];rm=[];racc=[]
    for k in range(n):
        points,J=kin.points_and_jacobians(q[k]);J=J.reshape(12,18)
        rpos.append(float(np.max(abs(points-ref['feet'][k]))))
        rtan.append(float(np.max(abs(J@N[k]-desired_map))))
        rvel.append(float(np.max(abs(J@v[k]-ref['active_v'][k,6:]))))
        aug=np.r_[q[k],ref['active'][k,6:]]
        rebuilt,info=graph.solver.mapping(aug)
        if not info['success']: raise ValueError('Invalid PACDM chart at '+str(k))
        rc.append(info['rcond']);rf.append(info['rank_full']);rd.append(info['rank_passive'])
        rm.append(float(np.max(abs(rebuilt[:18]-N[k]))))
        if k in idset:
            eps=5e-6
            Jp=kin.points_and_jacobians(q[k]+eps*v[k])[1].reshape(12,18)
            Jm=kin.points_and_jacobians(q[k]-eps*v[k])[1].reshape(12,18)
            err=J@a[k]+((Jp-Jm)/(2*eps))@v[k]-ref['active_a'][k,6:]
            racc.append(float(np.max(abs(err))))
    d=dict(time=ref['time'],foot_closure_inf_m=np.array(rpos),CE_max_abs=np.array(rtan),
        foot_velocity_inf_m_s=np.array(rvel),rcond_1=np.array(rc),rank_full=np.array(rf),
        rank_dependent=np.array(rd),map_reconstruction_max_abs=np.array(rm),
        acceleration_time=ref['time'][ids],acceleration_indices=ids,
        acceleration_inf_m_s2=np.array(racc))
    recorded=load_json('reference')
    compare(max(rpos),recorded['max_residual_inf'],'reference closure',atol=1e-15)
    compare(max(rtan),recorded['max_tangent_residual'],'tangent residual',atol=1e-17)
    compare(min(rc),recorded['min_rcond'],'selected block conditioning')
    checks={x['name']:x for x in load_json('task_validation')['checks']}
    compare(max(racc),checks['directional-Jdot acceleration constraints']['value'],
            'acceleration constraint',atol=5e-13,rtol=1e-4)
    assert set(rf)=={12} and set(rd)=={12}
    np.savez_compressed(DER/'reference_diagnostics.npz',**d)
    return ref,d

def metrics(logs):
    rows=[]
    for name,log in logs.items():
        e=np.linalg.norm(log['q'][:,:3]-log['q_ref'][:,:3],axis=1)
        use=log['time']>=2
        j=load_json(name)
        rms=float(np.sqrt(np.mean(e[use]**2)));final=float(e[-1])
        compare(rms,j['rms_body_error_m'],name+' RMS')
        compare(final,j['final_position_error_m'],name+' terminal error')
        row=dict(case=name,rms_position_error_mm=1000*rms,
            final_position_error_mm=1000*final,peak_position_error_mm=1000*float(max(e)),
            peak_torque_usage_all_steps_percent=100*j['peak_torque_limit_fraction'],
            peak_torque_usage_logged_percent=100*float(np.max(abs(log['torque'])/LIMITS)),
            source_step_s=j['dt_s'],completed=j['completed'])
        rows.append(row)
    with (DER/'case_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    nom=logs['nominal'];fine=logs['fine'];pd=logs['PD_ablation']
    assert np.array_equal(nom['time'],fine['time'])
    assert np.array_equal(nom['q_ref'],pd['q_ref'])
    delta=np.linalg.norm(nom['q'][:,:3]-fine['q'][:,:3],axis=1)
    np.savetxt(DER/'refinement_trace.csv',np.c_[nom['time'],1000*delta],
               delimiter=',',header='time_s,base_path_difference_mm',comments='')
    nominal=rows[0]
    pdrow=next(r for r in rows if r['case']=='PD_ablation')
    extras=dict(refinement_max_difference_mm=1000*float(max(delta)),
        refinement_rms_difference_mm=1000*float(np.sqrt(np.mean(delta**2))),
        refinement_final_difference_mm=1000*float(delta[-1]),
        pd_rms_reduction_percent=100*(1-nominal['rms_position_error_mm']/pdrow['rms_position_error_mm']),
        nominal_final_yaw_error_deg=float(np.rad2deg(abs(nom['q'][-1,3]-nom['q_ref'][-1,3]))),
        nominal_final_speed_mm_s=float(np.linalg.norm(nom['v'][-1,:3])*1000))
    return rows,extras,delta

def fig_reference(d):
    fig,axs=plt.subplots(2,2,figsize=(7.3,4.7),layout='constrained')
    t=d['time']; ax=axs[0,0]
    ax.plot(t,d['foot_closure_inf_m']*1e9,color=BLUE,lw=.85)
    ax.set(title='(a) Foot-task position closure',ylabel='Maximum component [nm]',ylim=(0,1.1))
    boxnote(ax,f'Maximum: {max(d["foot_closure_inf_m"])*1e9:.3f} nm');axis(ax)
    ax=axs[0,1];ax.plot(t,d['CE_max_abs']*1e16,color=GREEN,lw=.85)
    ax.set(title='(b) Tangent-map consistency',ylabel=r'Maximum $|CE|$ entry [$10^{-16}$]',ylim=(0,5.3))
    boxnote(ax,f'Maximum: {max(d["CE_max_abs"]):.2e}');axis(ax)
    ax=axs[1,0];ax.plot(d['acceleration_time'],d['acceleration_inf_m_s2']*1e9,
        color=BLUE,lw=.8,marker='.',ms=2.5)
    ax.set(title='(c) Acceleration consistency',ylabel=r'Residual [$10^{-9}$ m/s$^2$]',ylim=(0,4.7))
    boxnote(ax,f'{len(d["acceleration_time"])} independent derivative checks');axis(ax)
    ax=axs[1,1];ax.plot(t,d['rcond_1'],color=GREEN,lw=1)
    ax.set(title='(d) Dependent-coordinate conditioning',ylabel=r'Reciprocal condition number (1-norm)')
    ax.set_ylim(.17,float(max(d['rcond_1']))+.06)
    boxnote(ax,f'Minimum: {min(d["rcond_1"]):.4f}; full / dependent rank: 12');axis(ax)
    save(fig,'Go2_Reference_Validation')

def fig_motion(log):
    t=log['time'];q=log['q'];ref=log['q_ref'];use=(t>=6)&(t<=7.6)
    fig,axs=plt.subplots(3,2,figsize=(7.3,7.2),layout='constrained')
    ax=axs[0,0]
    for x,col in [(.48,ORANGE),(.86,ORANGE),(1.65,GREEN)]:
        ax.axvspan(x-.025,x+.025,color=col,alpha=.13,lw=0)
    ax.plot(ref[:,0],ref[:,1],color=ORANGE,ls='--',lw=1.3,label='Reference',zorder=4)
    ax.plot(q[:,0],q[:,1],color=BLUE,lw=1,label='MuJoCo',zorder=3)
    ax.scatter([q[0,0]],[q[0,1]],s=13,color=BLUE,zorder=5)
    ax.scatter([ref[-1,0]],[ref[-1,1]],s=25,marker='x',color=INK,zorder=5)
    ax.set(title='(a) Horizontal base trajectory',xlabel='World x [m]',ylabel='World y [m]',
           xlim=(-.10,2.30),ylim=(-.04,.44))
    ax.grid();ax.legend(frameon=False,loc='upper left',ncol=2)
    ax.text(.48,.32,'Rail 1',rotation=90,ha='center',va='center',fontsize=6.5,color='#9A6D2A')
    ax.text(.86,.32,'Rail 2',rotation=90,ha='center',va='center',fontsize=6.5,color='#9A6D2A')
    ax.text(1.65,.33,'Gate',rotation=90,ha='center',va='center',fontsize=6.5,color=GREEN)
    ax=axs[0,1]
    ax.plot(t,q[:,2]*1000,color=BLUE,lw=1,label='MuJoCo')
    ax.plot(t,ref[:,2]*1000,color=ORANGE,ls='--',lw=1,label='Reference')
    ax.set(title='(b) Crouching and recovery of height',ylabel='Base height [mm]',ylim=(247,340))
    axis(ax,push=True);ax.legend(frameon=False,loc='lower left',ncol=2)
    ax=axs[1,0]
    ax.plot(t,np.rad2deg(q[:,3]),color=BLUE,lw=1,label='MuJoCo')
    ax.plot(t,np.rad2deg(ref[:,3]),color=ORANGE,ls='--',lw=1,label='Reference')
    ax.set(title='(c) Heading through the turn',ylabel='Yaw [deg]');axis(ax,push=True)
    ax.legend(frameon=False,loc='upper left',ncol=2)
    ax=axs[1,1]
    error=np.linalg.norm(q[:,:3]-ref[:,:3],axis=1)*1000
    ax.plot(t,error,color=BLUE,lw=1)
    ax.set(title='(d) Executed base-position error',ylabel='Euclidean error [mm]',ylim=(0,41))
    boxnote(ax,f'RMS: {np.sqrt(np.mean(error[t>=2]**2)):.2f} mm; terminal: {error[-1]:.2f} mm');axis(ax,push=True)
    ax=axs[2,0]
    for j,(name,col) in enumerate(zip(['Hip','Thigh','Calf'],COLORS[:3])):
        ax.plot(t[use],np.rad2deg(q[use,6+j]),color=col,lw=1,label=name)
        ax.plot(t[use],np.rad2deg(ref[use,6+j]),color=col,ls='--',lw=.85)
    ax.set(title='(e) Front-left joint trajectories',ylabel='Joint angle [deg]')
    axis(ax,(6,7.6));ax.legend(frameon=False,ncol=3,loc='center right')
    ax=axs[2,1]
    ax.plot(t[use],log['feet'][use,0,2]*1000,color=BLUE,lw=1,label='MuJoCo')
    ax.plot(t[use],log['foot_ref'][use,0,2]*1000,color=ORANGE,ls='--',lw=1,label='Reference')
    ax.set(title='(f) Front-left foot-center trajectory',ylabel='World height [mm]',ylim=(0,145))
    axis(ax,(6,7.6));ax.legend(frameon=False,loc='upper right',ncol=2)
    save(fig,'Go2_Motion_Tracking')

def spans(t,mask):
    ids=np.diff(np.r_[False,mask,False].astype(int));dt=float(np.median(np.diff(t)))
    return [(float(t[a]),float(t[b-1]+dt-t[a])) for a,b in
            zip(np.flatnonzero(ids==1),np.flatnonzero(ids==-1))]

def fig_contact(log):
    t=log['time'];use=(t>=6)&(t<=7.6)
    fig,axs=plt.subplots(2,2,figsize=(7.3,4.9),layout='constrained')
    ax=axs[0,0]
    for i in range(4):
        ax.broken_barh(spans(t,log['stance'][:,i].astype(bool)),(i-.23,.19),
                       facecolors='#BFCAD2',linewidth=0)
        ax.broken_barh(spans(t,log['support_force'][:,i]>2),(i+.035,.19),
                       facecolors=COLORS[i],linewidth=0)
    ax.set(title='(a) Planned stance and realized support',yticks=range(4),yticklabels=LEGS,ylim=(3.65,-.65))
    axis(ax,(6,7.6));ax.grid(False)
    ax.legend(handles=[Patch(color='#BFCAD2',label='Planned stance'),
                Patch(color=BLUE,label=r'Support: $F_z>2$ N')],
              loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.01),fontsize=6.6)
    ax=axs[0,1]
    for i,leg in enumerate(LEGS):
        ax.plot(t[use],log['support_force'][use,i],color=COLORS[i],lw=.9,label=leg)
    ax.set(title='(b) Native world-vertical foot reactions',ylabel='Vertical force [N]',ylim=(-3,142))
    axis(ax,(6,7.6));ax.legend(frameon=False,ncol=4,loc='upper center',fontsize=6.8)
    ax=axs[1,0]
    ratios=(abs(log['torque'])/LIMITS).reshape(-1,4,3).max(axis=2)*100
    for i,leg in enumerate(LEGS):ax.plot(t,ratios[:,i],color=COLORS[i],lw=.65,label=leg)
    ax.axhline(100,color=RED,lw=.9,ls='--')
    ax.set(title='(c) Motor torque relative to source limits',ylabel='Largest utilization per leg [%]',ylim=(0,138))
    boxnote(ax,f'All-step nominal peak: {100*load_json("nominal")["peak_torque_limit_fraction"]:.2f}%',xy=(.02,.96))
    axis(ax,push=True);ax.legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,.87),fontsize=6.8)
    ax=axs[1,1]
    ax.plot(t,log['push'][:,0],color=BLUE,lw=1,label='World x')
    ax.plot(t,log['push'][:,1],color=ORANGE,lw=1,label='World y')
    ax.set(title='(d) Applied disturbance forces',ylabel='Base force [N]',ylim=(-34,45))
    axis(ax,push=True);ax.legend(frameon=False,ncol=2,loc='upper center')
    boxnote(ax,r'$+y$: 4.8 N s; $-x$: 5.0 N s',xy=(.16,.10))
    save(fig,'Go2_Contact_Actuation')

def fig_robustness(logs,rows,delta):
    fig,axs=plt.subplots(2,2,figsize=(7.3,4.8),layout='constrained')
    lookup={x['case']:x for x in rows}
    for ax,field,title,ylim in [
        (axs[0,0],'rms_position_error_mm','(a) RMS base-position error',(0,17)),
        (axs[0,1],'final_position_error_mm','(b) Terminal base-position error',(0,10.5))]:
        vals=[lookup[x][field] for x in CASES]
        ax.bar(range(5),vals,width=.61,color=[BLUE,'#6395B2',GREEN,ORANGE,'#8D6AA3'],zorder=3)
        for k,val in enumerate(vals):ax.text(k,val+.25,f'{val:.2f}',ha='center',va='bottom',fontsize=7)
        ax.set(title=title,ylabel='Error [mm]',xticks=range(5),xticklabels=LABELS,ylim=ylim)
        ax.tick_params(axis='x',labelsize=6.8,length=0);ax.grid(axis='y',zorder=0)
    ax=axs[1,0]
    for name,col,label in [('PD_ablation',ORANGE,'Bounded joint PD'),('nominal',BLUE,'Whole-body dynamics')]:
        log=logs[name];use=log['time']>=2
        e=np.linalg.norm(log['q'][:,:3]-log['q_ref'][:,:3],axis=1)*1000
        assert np.all(e[use]>0)
        ax.semilogy(log['time'][use],e[use],color=col,lw=1,label=label)
    ax.set(title='(c) Torque execution with the same reference',ylabel='Position error [mm, log scale]',ylim=(.4,5000))
    axis(ax,(2,26));ax.legend(frameon=False,loc='lower right',fontsize=6.8)
    ax=axs[1,1]
    ax.plot(logs['nominal']['time'],delta*1000,color=BLUE,lw=1)
    ax.set(title='(d) Physics-step refinement',ylabel='Base-path difference [mm]',ylim=(0,2.3))
    boxnote(ax,f'1 vs 0.5 ms: maximum {1000*max(delta):.3f} mm\nTerminal difference: {1000*delta[-1]:.3f} mm')
    axis(ax,push=True)
    save(fig,'Go2_Robustness_Ablation')

def main():
    global IN,OUT,DER
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=ROOT,
                        help='Directory containing generated data/ and results/.')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figures')
    parser.add_argument('--diagnostics-dir',type=Path,default=ROOT/'derived')
    args=parser.parse_args()
    IN=args.input_root.resolve();OUT=args.output_dir.resolve();DER=args.diagnostics_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True);DER.mkdir(parents=True,exist_ok=True)
    count=verify_inputs();print(f'Recorded {count} input/source hashes.',flush=True)
    ref,d=reference_diagnostics();print('Recomputed 2,601 PACDM samples and 151 acceleration checks.',flush=True)
    logs={name:load_npz(name) for name in CASES+['PD_ablation']}
    rows,extras,delta=metrics(logs)
    fig_reference(d);fig_motion(logs['nominal']);fig_contact(logs['nominal'])
    fig_robustness(logs,rows,delta)
    summary=dict(source='Go2 CMG/PACDM local run data',
       hashed_input_files=count,plot_environment=dict(python=platform.python_version(),
          numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
       reference=dict(samples=len(ref['time']),acceleration_samples=len(d['acceleration_time']),
          max_foot_closure_component_m=float(max(d['foot_closure_inf_m'])),
          max_tangent_entry=float(max(d['CE_max_abs'])),
          max_foot_velocity_component_m_s=float(max(d['foot_velocity_inf_m_s'])),
          max_foot_acceleration_component_m_s2=float(max(d['acceleration_inf_m_s2'])),
          min_rcond_1=float(min(d['rcond_1'])),full_ranks=sorted(set(d['rank_full'].tolist())),
          dependent_ranks=sorted(set(d['rank_dependent'].tolist())),
          max_rebuilt_map_difference=float(max(d['map_reconstruction_max_abs']))),
       cases=rows,comparisons=extras,
       plot_definitions=dict(rms='Euclidean base position error over logged times >=2 s',
          terminal='Euclidean base position error at 26 s',
          measured_support='Per-foot native world-vertical reaction greater than 2 N',
          torque_histories='10-ms logs; summary peak uses the per-physics-step JSON record',
          tangent='Maximum absolute entry; no clipping or logarithmic floor applied',
          ablation='Same PACDM reference; bounded joint PD replaces whole-body dynamics torque',
          refinement='Euclidean base-position difference at identical logged times'))
    (DER/'figure_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Wrote four PDF/PNG figures and derived evidence.',flush=True)

if __name__=='__main__':main()
