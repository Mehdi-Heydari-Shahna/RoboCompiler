"""Create release figures, readable tables and complete CSV force/energy exports."""
import csv,html,json,base64,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_model import ROOT,PORTS,INDEPENDENT,save_json,load_trace
from hydraulic_actuators import PARAMETERS
from digging_export import SOIL

def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)

def export_csv(name,a,r):
 folder=ROOT/'results/csv';folder.mkdir(exist_ok=True)
 keys=r['power_order']
 header=['time_s']+[f'{p}_position_'+('m' if i<6 else 'rad') for i,p in enumerate(PORTS)]+[f'{p}_speed_'+('m_s' if i<6 else 'rad_s') for i,p in enumerate(PORTS)]+[f'{p}_effort_'+('N' if i<6 else 'Nm') for i,p in enumerate(PORTS)]+[f'{p}_power_W' for p in PORTS]+[f'{p}_pressure_{ch}_Pa' for p in PORTS[:6] for ch in ['A','B']]+[f'{p}_flow_{ch}_m3_s' for p in PORTS[:6] for ch in ['A','B']]+['mechanical_energy_J','fluid_energy_J','material_inside_kg']+[f'{p}_work_J' for p in keys]+[f'bucket_{s}' for s in ['Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm']]
 with (folder/f'{name}_actuators_energy_contact.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(header)
  for k,t in enumerate(a['time']):w.writerow(np.r_[t,a['port_position'][k],a['port_velocity'][k],a['effort'][k],a['port_power'][k],a['pressure'][k].ravel(),a['flow'][k].ravel(),a['energy'][k],a['fluid_energy'][k],a['captured_mass'][k],a['work'][k],a['wrenches'][k,0]])
 with (folder/f'{name}_body_forces_energy.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['time_s','source_body','kinetic_J','potential_J','internal_Mx_Nm','internal_My_Nm','internal_Mz_Nm','internal_Fx_N','internal_Fy_N','internal_Fz_N','contact_Fx_N','contact_Fy_N','contact_Fz_N','contact_Mx_Nm','contact_My_Nm','contact_Mz_Nm'])
  for k,t in enumerate(a['time']):
   for j,b in enumerate(r['body_wrench_order']):w.writerow([t,b,*a['body_energy'][k,j],*a['body_internal_wrench'][k,j],*a['body_contact_wrench'][k,j]])

def make_report(reports,validation):
 r=reports['nominal'];a=load_trace('nominal');t=a['time'];out=ROOT/'results/figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.2})
 fig,ax=plt.subplots(3,2,figsize=(13,11),layout='constrained')
 for i,p in enumerate(PORTS[:6]):ax[0,0].plot(t,a['effort'][:,i]/1000,label=p)
 ax[0,0].set(ylabel='Cylinder effort [kN]',title='Six physical cylinder ports');ax[0,0].legend(ncol=3)
 for i,p in enumerate(PORTS[6:]):ax[0,1].plot(t,a['effort'][:,i+6]/1000,label=p)
 ax[0,1].set(ylabel='Torque [kN m]',title='Two rotary drive ports');ax[0,1].legend()
 for i,p in enumerate(PORTS[:6]):ax[1,0].plot(t,a['pressure'][:,i,:].max(axis=1)/1e6,label=p)
 ax[1,0].axhline(PARAMETERS['supply_pressure_Pa']/1e6,color='black',ls='--',lw=1,label='Supply limit');ax[1,0].set(ylabel='Pressure [MPa]',title='Maximum of chambers A/B');ax[1,0].legend(ncol=3)
 ax[1,1].plot(t,np.linalg.norm(a['wrenches'][:,0,:3],axis=1)/1000,color='#aa581a');ax[1,1].set(ylabel='Contact force magnitude [kN]',title='Bucket contact load')
 ax[2,0].plot(t,a['captured_mass'],color='#aa581a',label='Inside bucket');ax[2,0].plot(t,a['lifted_mass'],ls='--',label='All grains above 0.9 m');ax[2,0].set(ylabel='Material mass [kg]',title='Native particle motion');ax[2,0].legend()
 for i,p in enumerate(INDEPENDENT[:6]):ax[2,1].plot(t,(a['independent'][:,:6]-a['desired'])[:,i],label=p)
 ax[2,1].set(ylabel='Tracking error [rad]',title='Six commanded axes; q22 remains passive');ax[2,1].legend(ncol=3)
 for aa in ax.ravel():aa.set_xlabel('Simulation time [s]')
 fig.savefig(out/'actuation_contact.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained');keys=r['power_order'];work={k:a['work'][:,i] for i,k in enumerate(keys)}
 delta=a['energy']-a['energy'][0];rhs=work['mechanical']+work['equality']+work['contact']+work.get('cohesion',0)
 ax[0,0].plot(t,delta/1000,label='Stored mechanical energy change');ax[0,0].plot(t,rhs/1000,ls='--',label='Integrated mechanical + constraint work');ax[0,0].set(ylabel='Energy [kJ]',title='Robot + material energy ledger');ax[0,0].legend(fontsize=8)
 ax[0,1].plot(t,delta-rhs,color='#b64232');ax[0,1].set(ylabel='Balance defect [J]',title='Finite-step mechanical balance defect')
 for key in ['supply','throttle','friction','leakage','mechanical']:ax[1,0].plot(t,work[key]/1000,label=key)
 ax[1,0].set(ylabel='Cumulative work [kJ]',title='Supply, losses and mechanical output');ax[1,0].legend(ncol=2)
 times=[reports[k]['dt_s']*1000 for k in ['coarse','nominal','fine']];errors=[abs(reports[k]['mechanical_balance_defect_J']) for k in ['coarse','nominal','fine']]
 ax[1,1].plot(times,errors,'o-');ax[1,1].set(xlabel='Timestep [ms]',ylabel='Absolute final defect [J]',title='Timestep sensitivity (not particle-path convergence)')
 for aa in ax.ravel()[:3]:aa.set_xlabel('Simulation time [s]')
 fig.savefig(out/'energy_validation.png',dpi=150);plt.close(fig)
 export_csv('nominal',a,r)
 body=[];cmg=json.loads((ROOT/'data/accepted_cmg_v04.json').read_text());bm={b['id']:b for b in cmg['bodies']}
 for i,b in enumerate(r['body_wrench_order']):body.append([b,bm[b]['name'],f"{bm[b]['mass_kg']:.4f}",f"{np.linalg.norm(a['body_internal_wrench'][:,i,3:],axis=1).max()/1000:.3f}",f"{np.linalg.norm(a['body_internal_wrench'][:,i,:3],axis=1).max()/1000:.3f}",f"{a['body_energy'][:,i,0].max():.3f}"])
 with (ROOT/'results/csv/body_summary.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['body','source_name','mass_kg','peak_tree_force_kN','peak_tree_moment_kNm','peak_kinetic_J']);w.writerows(body)
 rows=[]
 for name,v in reports.items():rows.append([name,f"{v['dt_s']*1000:g}",f"{v['peak_loop_gap_m']*1e6:.2f}",f"{v['carried_particle_mass_kg']:.2f}",f"{v['deposited_carried_mass_kg']:.2f}",f"{100*v['mechanical_balance_relative']:.4f}%"])
 txt=f'''# Excavator RoboIR v21 — full-body contact benchmark

**Validation: {validation['passed_count']}/{validation['check_count']} gates {'PASS' if validation['passed'] else 'FAIL'}.**

The accepted excavator has been extended to a floating undercarriage, physical actuators, force and energy accounting, and a native contact-driven scoop–lift–slew–dump task. The release preserves all 25 supplied rigid bodies, their centers of mass and inertia tensors ({r['source_mass_kg']:.6f} kg in total). The same PACDM implementation used in the accepted Kangaroo v20 is reused with an excavator graph adapter.

This is a completed **simulation verification benchmark**. The hydraulic parameters and coarse aggregate are declared study assumptions. No measured hydraulic identification, field soil calibration, excavation productivity validation or hardware experiment is claimed.

## Measured result

At the nominal 0.5 ms timestep, {r['carried_particle_mass_kg']:.2f} kg of identified particles was carried during the lift/slew interval and {r['deposited_carried_mass_kg']:.2f} kg of those particles reached the receiving region. Maximum loop gap was {r['peak_loop_gap_m']*1e6:.2f} μm. Peak chamber pressure was {r['peak_pressure_Pa']/1e6:.2f} MPa against the assumed {r['supply_pressure_Pa']/1e6:.1f} MPa supply limit. No native solver warning occurred.

'''+table(['Case','dt [ms]','Peak loop gap [μm]','Carried [kg]','Deposited [kg]','Mechanical balance defect / activity'],rows)+f'''
Every dynamic case uses native MuJoCo integration. Robot and particle states are initialized once, then are never projected, teleported or attached to the bucket. The inverse-dynamics controller reconstructs only a separate closed reference used to compute efforts. In the contact-disabled control, the same bucket geometry is visualized but generates zero contact wrench.

## Model and method

The starting point is the accepted v14 CMG snapshot, with its retained static source-wiring audit and hashes; MATLAB/Simscape was not executed in this release. The source graph has 32 moving coordinates and 9 cycles. PACDM applies 54 SE(3) residual rows of rank 25, leaving seven independent internal coordinates. Its acquisition, 281-point branch continuation, full residuals, tangent map and finite-difference Jacobian are checked. The independent order is q23, q7, q4, q0, q1, q21, q22. Only the first six are commanded; q22 is a passive internal mode. The native spanning-tree plant has 23 joint coordinates plus a six-DOF free base. Eighteen two-point connect constraints reproduce the source loop geometry (rank 16 in tree coordinates).

Independent source body-sum mechanics, Pinocchio and MuJoCo agree for mass, bias, gravity, kinetic and potential energy over 25 bounded states, three gravity vectors and three requested accelerations per state. Floating-base comparisons use 13 states, including rotated and moving bases. Native contact snapshots additionally check independently summed energy and explicit contact-wrench mapping to generalized forces.

The provided undercarriage is one rigid track assembly. Its two ground patches support a free base; there are no independently driven belt links. CAD is preserved for visualization. Collision shapes are massless proxies: a hollow bucket with six floor/back panels and two side panels, two track patches, and convex proxies for the remaining bodies. Robot self-collision is disabled in this benchmark. Particle–particle, particle–bucket, ground and other-body external contacts are enabled. This is a 3-D rigid aggregate contact model, not a continuum plastic soil model.

## Actuators

Six double-acting cylinders act at p0…p5, and two bounded rotary drives act at q21 and q23. Chamber pressures evolve with compressibility, valve-flow limits, cross-chamber leakage and relief/makeup accounting. Cylinder force is A_A p_A − A_B p_B minus smooth Coulomb/viscous friction. Rotary torque has a first-order response and torque limits. The ideal supply/tank model does not include an engine, pump displacement dynamics, thermal dynamics or a calibrated efficiency map.

The assumed supply is 25 MPa, bulk modulus 0.8 GPa, maximum per-valve flow 0.008 m³/s, pressure response 25 ms and dead chamber volume 0.004 m³. All values and units are in `hydraulic_actuators.PARAMETERS`. The large return-flow preload is part of this synthetic pressure controller; the resulting supply and throttle losses are reported without claiming hydraulic efficiency realism.

'''+table(['Port','A area [m²]','B area [m²]','Source command relation'],[[p,PARAMETERS['area_A_m2'][i],PARAMETERS['area_B_m2'][i],['unresolved source channel; desired force 0','−F5','−F4','−F2/2','−F2/2','−F3'][i]] for i,p in enumerate(PORTS[:6])])+'''
The source rotary effort relations are tau(q21) = 10 F6 and tau(q23) = 25 F1. Source channel dimensions are inferred from the connected physical ports; the original converter signals did not explicitly declare those SI effort units. p0 is retained as a physical cylinder with zero desired effort, but its source input is unresolved. Its actual modeled force can be nonzero during pressure transients and friction. It is not silently assigned a source command. The source boom feedback has a previously audited approximately 3.18% power mismatch, and the tilt feedback reads p0 motion while force is applied at p1. This release uses physical prismatic joint rates for power conjugacy. Source joint limits were disabled; branch and chamber-volume guards in this benchmark are numerical domains, not factory limits.

## Energy and forces

Stored mechanical energy sums kinetic and gravitational potential energy of every robot body and every grain. Stored hydraulic compression energy is Σ V p²/(2β). Mechanical power uses each physical effort times its own joint velocity. The ledger includes signed rotary work, valve throttling, leakage, actuator friction, relief/makeup, the variable-volume compression term, and native equality/contact work. Contact and constraint work are signed; they are not assumed to be pure dissipation. Integration defects are reported explicitly, with timestep refinement.

'''+table(['Nominal energy term','Value [kJ]'],[[k,f'{v/1000:.6f}'] for k,v in r['work_J'].items()])+f'''
Final mechanical energy change: {r['mechanical_energy_change_J']/1000:.6f} kJ. Final stored fluid energy change: {r['fluid_energy_change_J']/1000:.6f} kJ. Mechanical balance defect: {r['mechanical_balance_defect_J']:.6f} J ({100*r['mechanical_balance_relative']:.5f}% of absolute mechanical/constraint work activity). Combined fluid/mechanical balance defect: {r['total_balance_defect_J']:.6f} J ({100*r['total_balance_relative']:.5f}% of absolute supply/loss/constraint work activity). The latter denominator is larger; both absolute defects and the mechanical metric must be considered.

Per-body CSV files contain kinetic/potential energy, native tree interaction wrenches, and external contact wrenches. `body_internal_wrench` is MuJoCo `cfrc_int`: moment then force, world-oriented, about the COM-based subtree frame. These are representation-dependent tree interaction loads, not uniquely identified forces at every redundant cut pin and not structural stress predictions. `body_contact_wrench` is force then moment, in world axes, about each body's own COM. Bucket and track contact wrenches use the same latter convention. Raw equality multipliers are available separately and must not be interpreted as unique pin reactions.

## Material experiment

There are 50 free spheres, radius 0.09 m and density 1800 kg/m³, initially arranged as a stable close-packed mound. Nominal soil–soil sliding friction is 0.65 with rolling resistance 0.008 m. Bucket contacts use priority-selected sliding friction 0.25, torsional 0.001 m and rolling 0.002 m. Ground friction is 1.0. Parameters are synthetic. The particles are deliberately coarse for a compact reproducible benchmark; grain size and contact-law sensitivity remain limits on physical generalization.

“Inside” means the particle center lies inside the CAD-derived cavity half-spaces with a 0.02 m inset; it is a center-based occupancy measure. “Carried” records particle identities inside the bucket with world height above 0.9 m during t = 7…10 s. “Deposited” requires those same identities to finish at world y > 1 m and z < 0.8 m. Motion is never prescribed to any grain. Exact grain trajectories are not expected to converge under timestep changes; the study checks energy errors, bounded dynamics and successful material transfer at each tested timestep.

## All supplied bodies

'''+table(['Body','Source name','Mass [kg]','Peak tree force [kN]','Peak tree moment [kN m]','Peak kinetic energy [J]'],body)+'''
## Contents and reproduction

- `run_excavator_v21.py`: complete validation, all seven contact experiments, report and optional video.
- `results/validation.json`: every numerical gate, threshold and observed value.
- `results/mechanics_validation.json`: PACDM and three-backend mechanics results.
- `results/*.npz`: 100 Hz state, pressure, flow, force and energy traces; native physics uses the stated smaller timestep.
- `results/csv/`: readable nominal traces and all 25 body loads/energies.
- `results/assets/*.xml`: native scenes with relative mesh paths.
- `Excavator_v21_digging.mp4`: rendered recorded states, 0.5× playback; one excavator, plus a close-up of that same bucket.
- `data/` and `assets/`: accepted source snapshot, audits and original CAD.
- `PROVENANCE.json` and `SHA256SUMS.json`: source lineage and release integrity.

The unchanged PACDM core demonstrates reuse from Kangaroo to excavator; task-specific graph adaptation, actuation parameters and collision geometry are explicit. The evidence supports generality across these mechanisms and agreement of rigid-body mechanics across the source implementation, Pinocchio and MuJoCo. Contact digging was executed in MuJoCo only. An Isaac Sim run has not been performed.

## Video explanation

The yellow excavator lowers its bucket, draws through the granular material, curls and lifts it, slews to the side, and opens the bucket to release the carried grains. The inset shows the same bucket at closer range. The displayed load, contact force, pressure and supply work come from the recorded simulation.

## Technical references

MuJoCo's contact model uses soft constraints; see [Computation and contact](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#contact). Collision proxies and contact parameter combination follow [Collision detection and modeling](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#collision-detection). Native wrench arrays follow the [API data definitions](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APItypes.html) and [COM-based spatial-vector convention](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#com-based-spatial-vectors).
'''
 (ROOT/'Excavator_v21_Report.md').write_text(txt,encoding='utf-8')
 # A self-contained, offline readable HTML copy with exact figures.
 def inline(line):
  line=html.escape(line);line=re.sub(r'`([^`]+)`',r'<code>\1</code>',line);line=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',line)
  return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',line)
 chunks=[];lines=txt.splitlines();i=0
 while i<len(lines):
  line=lines[i]
  if line.startswith('| '):
   rows=[]
   while i<len(lines) and lines[i].startswith('| '):rows.append([x.strip() for x in lines[i].strip('|').split('|')]);i+=1
   chunks.append('<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in rr)+'</tr>' for rr in rows[2:])+'</tbody></table>');continue
  if line.startswith('#'):
   level=len(line)-len(line.lstrip('#'));chunks.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
  elif line:chunks.append('<p>'+inline(line)+'</p>')
  i+=1
 pics=''.join('<figure><img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'"></figure>' for name in ['actuation_contact.png','energy_validation.png'])
 body_html=''.join(chunks).replace('<h2>Model and method</h2>',pics+'<h2>Model and method</h2>')
 doc='<!doctype html><html lang="en"><meta charset="utf-8"><title>Excavator v21 verification report</title><style>body{max-width:1100px;margin:45px auto;padding:0 24px;font:16px/1.55 system-ui;color:#182e42}h1,h2{line-height:1.2}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #d8e0e6;text-align:left}th{background:#eaf1f6}img{width:100%}figure{margin:30px 0}p{overflow-wrap:anywhere}code{background:#eef2f5;padding:2px 4px;font-size:.88em}</style>'+body_html+'</html>'
 (ROOT/'Excavator_v21_Report.html').write_text(doc,encoding='utf-8');print('Report and CSV exports complete',flush=True)
