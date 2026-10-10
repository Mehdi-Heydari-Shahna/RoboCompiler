#!/usr/bin/env python3
"""Reproduce publication figures from the regenerated Stewart numerical evidence.

Uses every saved sample; no new dynamics simulation or trajectory smoothing.
Run from any directory: python make_stewart_figures.py
Optional: --results PATH --output PATH
"""
from pathlib import Path
import argparse
import json
import hashlib
import platform
import io
import numpy as np
from scipy.spatial.transform import Rotation
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE=Path(__file__).resolve().parent
P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--results',type=Path,default=BASE.parent/'results')
P.add_argument('--output',type=Path,default=BASE/'figures')
args=P.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
names=['nominal','fine','heavy_payload','no_feedforward','pacdm','reference']
data={}
for name in names:
    with np.load(args.results/f'{name}.npz',allow_pickle=False) as z:
        data[name]={k:z[k].copy() for k in z.files}
    assert np.all(np.diff(data[name]['time'])>0)
    assert data[name]['time'][0]==0 and data[name]['time'][-1]==22
    for k,a in data[name].items():
        if np.issubdtype(a.dtype,np.number):assert np.all(np.isfinite(a)),(name,k)

nom=data['nominal']; direct=data['pacdm']; ref=data['reference']; fine=data['fine']
assert np.array_equal(nom['time'],direct['time'])
active=np.array([list(nom['coordinate_ids']).index(f'leg_{i}_length') for i in range(6)])
def rot(q):return Rotation.from_euler('ZYX',q[:,3:6])
def comparison(a,b):
    # Resample only the 1-ms trace onto the shared 2-ms times for refinement.
    qb=np.column_stack([np.interp(a['time'],b['time'],b['q'][:,k]) for k in range(6)])
    return (np.linalg.norm(a['q'][:,:3]-qb[:,:3],axis=1),
            np.rad2deg((rot(a['q']).inv()*rot(qb)).magnitude()))
pos_direct,ang_direct=comparison(nom,direct)
pos_fine,ang_fine=comparison(nom,fine)
metrics={'provenance':{'method':'Recomputed from saved trajectories; plotting does not integrate dynamics.',
          'plotting_python':platform.python_version(),'numpy':np.__version__,
          'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
          'sha256':{n+'.npz':hashlib.sha256((args.results/(n+'.npz')).read_bytes()).hexdigest() for n in names}},
          'cases':{}}
for name in names[:-1]:
    z=data[name];t=z['time'];dock=t>=21
    ep=np.linalg.norm(z['q'][:,:3]-z['target_pose'][:,:3],axis=1)
    ea=np.rad2deg((rot(z['target_pose']).inv()*rot(z['q'])).magnitude())
    assert np.allclose(ep,z['pose_error_m'],rtol=1e-10,atol=1e-14)
    assert np.allclose(ea,np.rad2deg(z['angle_error_rad']),rtol=1e-8,atol=1e-10)
    m={'samples':len(t),'dt_s':float(np.median(np.diff(t))),
       'rms_position_error_mm':float(np.sqrt(np.mean(ep**2))*1000),
       'peak_position_error_mm':float(ep.max()*1000),
       'peak_orientation_error_deg':float(ea.max()),
       'docking_max_position_error_mm':float(ep[dock].max()*1000),
       'docking_max_orientation_error_deg':float(ea[dock].max()),
       'max_force_N':float(np.max(np.abs(z['force']))),
       'max_closure_gap_m':float(z['closure_error_m'].max()),
       'leg_length_min_m':float(z['q'][:,active].min()),
       'leg_length_max_m':float(z['q'][:,active].max())}
    saved=json.loads((args.results/(name+'.json')).read_text())
    assert np.isclose(m['rms_position_error_mm']/1000,saved['rms_position_error_m'],rtol=1e-11)
    assert np.isclose(m['max_force_N'],saved['max_actuator_force_N'],rtol=1e-11)
    metrics['cases'][name]=m
metrics['comparisons']={
    'feedforward_RMS_reduction_percent':100*(1-metrics['cases']['nominal']['rms_position_error_mm']/metrics['cases']['no_feedforward']['rms_position_error_mm']),
    'refinement_RMS_change_percent':100*(metrics['cases']['fine']['rms_position_error_mm']/metrics['cases']['nominal']['rms_position_error_mm']-1),
    'direct_vs_mujoco_max_position_um':float(pos_direct.max()*1e6),
    'direct_vs_mujoco_max_orientation_deg':float(ang_direct.max()),
    'refinement_max_position_um':float(pos_fine.max()*1e6),
    'refinement_max_orientation_deg':float(ang_fine.max()),
    'reference_max_augmented_closure_entry':float(ref['closure_residual'].max()),
    'reference_max_tangent_entry':float(ref['tangent_residual'].max()),
    'reference_max_acceleration_entry':float(ref['acceleration_residual'].max()),
    'reference_min_reduced_mass_eigenvalue_kg':float(ref['reduced_mass_min_eigenvalue'].min()),
    'reference_max_passive_block_condition':float(ref['selected_block_condition'].max()),
    'direct_max_reduced_equation_residual_N':float(direct['reduced_equation_residual'].max())}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#7F8A96',
    'axes.labelcolor':'#263544','text.color':'#263544','axes.titleweight':'bold',
    'grid.color':'#D9E0E5','grid.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','path.simplify':False})
navy='#1F567E';teal='#00867D';orange='#D16B20';purple='#8062A0';gray='#4B5563'
colors=[navy,orange,teal,purple,'#B4405F','#7B8123']
pulses=[(5.5,5.85),(11.2,11.55),(14.2,14.6)]
def setup(ax,title,ylabel,shade=True):
    ax.set_title(title,loc='left',pad=8)
    ax.set_ylabel(ylabel);ax.set_xlim(0,22);ax.set_xticks([0,2,5,9,13,17,20,22])
    ax.grid(True,alpha=.8);ax.set_axisbelow(True)
    if shade:
        for a,b in pulses:ax.axvspan(a,b,color='#EDD4BF',alpha=.5,lw=0,zorder=0)
    for t in [2,9,17,20]:ax.axvline(t,color='#B5C1CC',ls=':',lw=.7,zorder=0)
def save(fig,name):
    # Finish each export in memory before atomically replacing its output.
    for fmt in ['pdf','png']:
        stream=io.BytesIO()
        fig.savefig(stream,format=fmt,dpi=300)
        payload=stream.getvalue()
        if fmt=='pdf':assert payload.rstrip().endswith(b'%%EOF')
        target=args.output/(name+'.'+fmt)
        temporary=target.with_suffix('.'+fmt+'.tmp')
        temporary.write_bytes(payload)
        temporary.replace(target)
    plt.close(fig)
def phase_strip(fig):
    ax=fig.add_axes([.10,.867,.87,.046])
    spans=[(0,2,'Hold'),(2,9,'Helical sweep'),(9,17,'Figure-eight'),(17,20,'Dock'),(20,22,'Hold')]
    for j,(a,b,label) in enumerate(spans):
        ax.axvspan(a,b,color=['#EAF0F4','#E7F2F0','#EBEFF8','#F4EDE2','#EAF0F4'][j])
        ax.text((a+b)/2,.5,label,ha='center',va='center',fontsize=8)
    ax.set_xlim(0,22);ax.set_ylim(0,1);ax.axis('off')

# 1: Six task coordinates. Pose convention matches rotation(...), ZYX yaw/pitch/roll.
fig,axs=plt.subplots(3,2,figsize=(7.3,7.0))
fig.subplots_adjust(left=.10,right=.97,bottom=.075,top=.825,hspace=.43,wspace=.30)
fig.suptitle('Six-axis platform trajectory',x=.10,ha='left',y=.986,fontsize=14,fontweight='bold')
handles=[Line2D([0],[0],color=gray,lw=1.6,ls='--',label='PACDM reference'),
         Line2D([0],[0],color=navy,lw=1.15,label='Nominal MuJoCo'),
         Patch(facecolor='#EDD4BF',alpha=.5,label='Applied-wrench interval')]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.09,.956),ncol=3,frameon=False)
phase_strip(fig)
for ax,k,title in zip(axs.ravel(),[0,3,1,4,2,5],['(a) Translation x','(b) Yaw','(c) Translation y','(d) Pitch','(e) Translation z','(f) Roll']):
    scale=1000 if k<3 else 180/np.pi
    setup(ax,title,'Position (mm)' if k<3 else 'Angle (deg)')
    ax.plot(nom['time'],nom['q'][:,k]*scale,color=navy,lw=1.1)
    ax.plot(nom['time'],nom['target_pose'][:,k]*scale,color=gray,lw=1.05,ls=(0,(4,3)))
for ax in axs[-1,:]:ax.set_xlabel('Time (s)')
save(fig,'Stewart_Trajectory')

# 2: Native actuator channels and external forcing (all 11,001 samples).
fig,axs=plt.subplots(2,2,figsize=(7.3,5.8))
fig.subplots_adjust(left=.10,right=.97,bottom=.10,top=.85,hspace=.44,wspace=.32)
fig.suptitle('Leg actuation and prescribed disturbances',x=.10,ha='left',y=.984,fontsize=14,fontweight='bold')
fig.legend(handles=[Line2D([0],[0],color=c,lw=1.5,label=f'Leg {i}') for i,c in enumerate(colors)],
           loc='upper left',bbox_to_anchor=(.085,.946),ncol=6,frameon=False)
a,b,c,d=axs.ravel()
setup(a,'(a) Actual leg lengths','Length (m)')
for i,col in enumerate(colors):a.plot(nom['time'],nom['q'][:,active[i]],color=col,lw=.95)
for lim in [.50,.91]:a.axhline(lim,color=gray,ls='--',lw=.85)
a.set_ylim(.48,.93)
setup(b,'(b) Applied leg forces','Force (N)')
for i,col in enumerate(colors):b.plot(nom['time'],nom['force'][:,i],color=col,lw=.95)
b.text(.98,.03,'Peak |f| = 134.01 N; limit = 900 N',transform=b.transAxes,ha='right',va='bottom',fontsize=7.5)
setup(c,'(c) Platform force pulses','Force (N)')
c.plot(nom['time'],nom['wrench'][:,0],color=navy,lw=1.2,label='$F_x$')
c.plot(nom['time'],nom['wrench'][:,1],color=orange,lw=1.2,label='$F_y$')
c.legend(frameon=False,loc='upper right')
setup(d,'(d) Platform moment pulse','Moment (N m)')
d.plot(nom['time'],nom['wrench'][:,3],color=purple,lw=1.2,label='$M_x$')
d.legend(frameon=False,loc='upper right')
for ax in axs[-1,:]:ax.set_xlabel('Time (s)')
save(fig,'Stewart_Actuation')

# 3: Direct numerical evidence for closure, tangent reduction, and dynamics.
fig,axs=plt.subplots(3,2,figsize=(7.3,7.5))
fig.subplots_adjust(left=.12,right=.97,bottom=.075,top=.88,hspace=.46,wspace=.37)
fig.suptitle('PACDM reduction and independent dynamics agreement',x=.12,ha='left',y=.984,fontsize=13.5,fontweight='bold')
fig.text(.12,.945,'Reference diagnostics: 20 ms knots. Rollout comparison: 2 ms steps.',fontsize=8.5,color=gray)
a,b,c,d,e,f=axs.ravel()
setup(a,'(a) Reference closure','Max. entry (mixed SI)',shade=False)
a.semilogy(ref['time'],ref['closure_residual'],color=teal,lw=.8)
a.axhline(1e-8,color=gray,lw=.9,ls='--',label='Acceptance threshold')
a.legend(loc='lower right',frameon=False,fontsize=7)
setup(b,'(b) Tangent consistency',r'Max. entry of $CE$ ($10^{-15}$)',shade=False)
b.plot(ref['time'],ref['tangent_residual']*1e15,color=teal,lw=.8)
setup(c,'(c) Reduced inertia','Smallest eigenvalue (kg)',shade=False)
c.plot(ref['time'],ref['reduced_mass_min_eigenvalue'],color=teal,lw=1.1)
c.set_ylim(bottom=0)
setup(d,'(d) Point-closure separation','Largest endpoint gap (m)')
# Zero residual values are omitted only on the logarithmic axis, not altered.
for z,col,ls,label in [(nom,navy,'-','MuJoCo'),(direct,teal,'--','Direct PACDM')]:
    gap=np.where(z['closure_error_m']>0,z['closure_error_m'],np.nan)
    d.semilogy(z['time'],gap,color=col,ls=ls,lw=.7,label=label)
d.legend(loc='upper right',frameon=False,fontsize=7)
setup(e,'(e) Position agreement',r'Position difference ($\mu$m)')
e.plot(nom['time'],pos_direct*1e6,color=purple,lw=1)
setup(f,'(f) Orientation agreement',r'Rotation difference ($10^{-5}$ deg)')
f.plot(nom['time'],ang_direct*1e5,color=purple,lw=1)
for ax in axs[-1,:]:ax.set_xlabel('Time (s)')
save(fig,'Stewart_PACDM_Validation')

# 4: Feedforward benefit, unchanged-controller payload increase, and refinement.
fig,axs=plt.subplots(2,2,figsize=(7.3,6.1))
fig.subplots_adjust(left=.10,right=.97,bottom=.12,top=.84,hspace=.51,wspace=.32)
fig.suptitle('Tracking benefit, payload robustness and time-step refinement',x=.10,ha='left',y=.986,fontsize=12.8,fontweight='bold')
cases=[('nominal',navy,'Nominal, 8 kg'),('heavy_payload',teal,'Payload, 14 kg'),('no_feedforward',orange,'Feedforward disabled')]
fig.legend(handles=[Line2D([0],[0],color=col,lw=1.5,label=label) for _,col,label in cases],
           loc='upper left',bbox_to_anchor=(.085,.942),ncol=3,frameon=False)
a,b,c,d=axs.ravel()
setup(a,'(a) Platform position error','Position error (mm)')
setup(b,'(b) Platform orientation error','Rotation error (deg)')
for name,col,label in cases:
    z=data[name]
    a.plot(z['time'],z['pose_error_m']*1e3,color=col,lw=1)
    b.plot(z['time'],np.rad2deg(z['angle_error_rad']),color=col,lw=1)
c.set_title('(c) Full-mission position RMS',loc='left',pad=8)
vals=[metrics['cases'][n]['rms_position_error_mm'] for n,_,_ in cases]
c.bar(np.arange(3),vals,color=[x[1] for x in cases],width=.60,zorder=3)
c.set_xticks([0,1,2],['Nominal\n8 kg','Payload\n14 kg','Feedforward\ndisabled'])
c.set_ylabel('RMS error (mm)');c.set_ylim(0,2.9);c.grid(True,axis='y');c.set_axisbelow(True)
for i,v in enumerate(vals):c.text(i,v+.07,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
c.text(.03,.98,'69.8% reduction with feedforward',transform=c.transAxes,va='top',fontsize=8,color=navy)
setup(d,'(d) MuJoCo 2 ms vs. 1 ms',r'Position difference ($\mu$m)')
d.plot(nom['time'],pos_fine*1e6,color=purple,lw=1.1)
d.text(.98,.96,'RMS tracking change: 0.146%',transform=d.transAxes,ha='right',va='top',fontsize=7.7)
for ax in [a,b,d]:ax.set_xlabel('Time (s)')
save(fig,'Stewart_Benefits_Refinement')

(args.output.parent/'recomputed_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({'figures':4,'source_samples':{n:len(data[n]['time']) for n in names},
                  'comparisons':metrics['comparisons']},indent=2))
