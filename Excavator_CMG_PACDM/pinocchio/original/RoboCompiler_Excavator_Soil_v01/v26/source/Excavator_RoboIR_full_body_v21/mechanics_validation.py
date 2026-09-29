"""PACDM and independent three-backend rigid-body/force/energy checks."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from benchmark_model import *
from digging_path import make_path,reference_at
from digging_export import export_scene
from excavator_pacdm import compile_pacdm
from constrained_dynamics import solve_kkt,solve_reduced
from floating_dynamics import FloatingSource,PinFloating,MJFloating,directed_tree
from hydraulic_actuators import HydraulicBank,PARAMETERS

def validate_mechanics():
 c,m,e,a,r,inverse=context();path=make_path(e,r);checks=[];metrics={}
 def gate(name,value,limit):checks.append(dict(name=name,passed=bool(np.isfinite(value) and value<=limit),value=float(value),limit=float(limit)))
 def keep(name,value):metrics[name]=max(metrics.get(name,0.),float(value))
 graph,pac=compile_pacdm(c,m,e.tree_ids);seed=graph.lift(path['closed_tree'][0]);seed[graph.passive]+=.01*np.sin(np.arange(len(graph.passive)));acquired,info=pac.acquire(path['independent'][0],seed)
 gate('PACDM perturbed assembly reaches physical lambda=1',0 if info['success'] and info['lambda']==1 else 1,0);save_json(ROOT/'results/pacdm_acquisition.json',info)
 times=np.linspace(0,14,281);previous=acquired.copy();N,mi=pac.mapping(previous)
 if N is None:raise ValueError('Initial PACDM mapping failed')
 branch=[]
 for t in times:
  u=reference_at(t,path)[0];pred=previous[graph.passive]+N[graph.passive]@(u-previous[graph.active]);q,ci=pac.correct(u,pred,None,np.array(mi['rows']),maxiter=12)
  if not ci['success']:raise ValueError(f'PACDM correction failed at {t}: {ci}')
  for _ in range(4):
   rr,jj,_=graph.residual(q)
   if np.max(abs(rr))<5e-13:break
   rows=np.array(mi['rows']);q[graph.passive]-=np.linalg.solve(jj[np.ix_(rows,graph.passive)],rr[rows])
  N,mi=pac.mapping(q)
  if N is None:raise ValueError(f'PACDM map failed at {t}: {mi}')
  keep('pacdm_closure',mi['residual_inf']);keep('pacdm_tangent',mi['tangent_residual']);cmp=r.component(q[:23],np.zeros(7),GRAVITY);keep('pacdm_vs_point_map',np.max(abs(N[:23]-cmp['tangent_map'])));branch.append(q.copy());previous=q
 branch=np.array(branch);np.savez_compressed(ROOT/'results/pacdm_branch.npz',time=times,source_coordinates=branch,coordinate_ids=graph.ids)
 for key in ['pacdm_closure','pacdm_tangent','pacdm_vs_point_map']:gate(key,metrics[key],1e-8)
 q=branch[70];J=graph.residual(q)[1];fd=np.zeros_like(J);eps=1e-6
 for k in range(graph.n):
  z=np.zeros(graph.n);z[k]=eps;fd[:,k]=(graph.residual(q+z)[0]-graph.residual(q-z)[0])/(2*eps)
 gate('PACDM Jacobian versus finite differences',np.max(abs(J-fd)),2e-8);bad=q.copy();bad[graph.passive[0]]+=.01;gate('PACDM rejects open configuration',0 if pac.mapping(bad)[0] is None else 1,0)
 gravities=[GRAVITY,np.array([1.,-2.,-8.5]),np.zeros(3)];indices=np.linspace(0,len(branch)-1,25).astype(int)
 for case,i in enumerate(indices):
  q=branch[i,:23];ud=.08*np.sin(np.arange(7)+.3*case)
  for gravity in gravities:
   state=e.state(q,ud,gravity);source=state['components']['source']
   for name in ['pin','mujoco']:
    z=state['components'][name]
    for key in ['mass_matrix','bias_forces','gravity_compensation','kinetic_energy']:
     keep('fixed_'+name+'_'+key,np.max(abs(np.asarray(z[key])-np.asarray(source[key])))/max(1.,np.max(abs(source[key]))))
    pu=z['potential_energy']+z.get('potential_energy_ground_offset',0.);keep('fixed_'+name+'_potential_energy',abs(pu-source['potential_energy'])/max(1.,abs(source['potential_energy'])))
   for rate in [0.,.1,-.1]:
    inv=inverse.solve(source,rate*np.cos(np.arange(6)+case),p0_force=0.);kr=solve_kkt(source,a.E@inv['physical_efforts']);rr=solve_reduced(source,a.E@inv['physical_efforts']);keep('inverse_kkt',np.max(abs(kr['acceleration']-inv['acceleration'])));keep('reduced_kkt',np.max(abs(rr['acceleration']-kr['acceleration'])))
    v=source['velocity'];f=inv['physical_efforts'];N=source['tangent_map'];keep('port_power',abs(f@(a.E.T@v)-(N.T@a.E@f)@ud))
 for key,val in list(metrics.items()):
  if key.startswith('fixed_'):gate(key,val,1e-8)
 for key in ['inverse_kkt','reduced_kkt']:gate(key,metrics[key],1e-6)
 gate('Physical port power identity',metrics['port_power'],1e-7)
 xml=ROOT/'results/assets/floating_oracle.xml';export_scene(xml,soil=False);full=FloatingSource(c,e.source);pb=PinFloating(directed_tree(c,m,e.tree_ids));mb=MJFloating(c,e.tree_ids,xml)
 for i in indices[::2]:
  q=branch[i,:23];v=.12*np.sin(np.arange(29)+.3*i);R=Rotation.from_rotvec([.07*np.sin(i),-.09*np.cos(i),.2*np.sin(i/3)]).as_matrix();pos=np.array([.2,-.1,1.])
  for gravity in gravities:
   z=full.evaluate(q,pos,R,v,gravity)
   for name,back in [('pin',pb),('mujoco',mb)]:
    n=back.evaluate(q,pos,R,v,gravity)
    for key in ['mass_matrix','bias_forces','potential_energy','kinetic_energy']:keep('floating_'+name+'_'+key,np.max(abs(np.asarray(n[key])-np.asarray(z[key])))/max(1.,np.max(abs(z[key]))))
    keep('floating_'+name+'_poses',max(np.max(abs(n['poses'][b]-z['poses'][b])) for b in mb.bodies))
 for key,val in list(metrics.items()):
  if key.startswith('floating_'):gate(key,val,1e-8)
 compiled=mb.model
 compiled.opt.gravity[:]=GRAVITY
 falling=mujoco.MjData(compiled);falling.qpos[2]+=10.;mujoco.mj_forward(compiled,falling)
 gate('Airborne full body has no hidden world anchor',np.max(abs(falling.qacc[:6]-[0,0,-9.81,0,0,0])),1e-7)
 gate('Uniform free fall causes no internal acceleration',np.max(abs(falling.qacc[6:])),1e-6)
 for b in c['bodies']:
  if b['kind']!='rigid_body':continue
  bid=compiled.body(b['id']).id;Q=np.zeros(9);mujoco.mju_quat2Mat(Q,compiled.body_iquat[bid]);Q=Q.reshape(3,3);keep('mass_preservation',abs(compiled.body_mass[bid]-b['mass_kg']));keep('inertia_preservation',np.max(abs(Q@np.diag(compiled.body_inertia[bid])@Q.T-b['inertia_com_kg_m2'])))
 gate('25 body masses preserved',metrics['mass_preservation'],1e-10);gate('Full COM inertia tensors preserved',metrics['inertia_preservation'],1e-8);gate('Floating undercarriage adds six coordinates',abs(compiled.nv-29),0);gate('Eight ports; no q22 actuator',0 if compiled.nu==8 and 'q22' not in PORTS else 1,0)
 nominal=r.nominal[[e.tree_ids.index(x) for x in PORTS]];bank=HydraulicBank(nominal,np.zeros(8),capacity_scale=.4);max_err=0.;limit_seen=False
 for i in range(1000):
  command=np.r_[np.full(6,1e7 if i<500 else -1e7),[1e7,-1e7]];item=bank.evaluate(nominal,np.zeros(8),command,.0005);max_err=max(max_err,abs(item['fluid_identity_W']));limit_seen|=item['pressure_limit_active'];bank.advance(item,.0005)
 probe=HydraulicBank(nominal,np.array([10000.,-5000.,30000.,60000.,60000.,-40000.,300.,4000.]))
 velocity=np.array([.03,-.02,.05,-.04,.015,.025,.1,-.08]);position=nominal+np.array([.02,-.04,.03,-.02,.01,.06,0,0]);item=probe.evaluate(position,velocity,np.zeros(8),.0005)
 pressure0=probe.pressure.copy();eps=1e-7
 probe.pressure=pressure0+eps*item['pressure_derivative'];plus=probe.stored_energy(position+eps*velocity)
 probe.pressure=pressure0-eps*item['pressure_derivative'];minus=probe.stored_energy(position-eps*velocity);probe.pressure=pressure0
 pressure_work=float((probe.A*pressure0[:,0]-probe.B*pressure0[:,1])@velocity[:6]);pw=item['power']
 predicted=pw['supply']-pw['throttle']-pw['leakage']-pw['relief']-pressure_work+pw['compressibility_geometry']
 gate('Moving-chamber stored energy directional derivative',abs((plus-minus)/(2*eps)-predicted),.01)
 gate('Over-command pressure bounds',max(0.,bank.pressure.max()-bank.ps,-bank.pressure.min()),1e-6);gate('Over-command activates limits',0 if limit_seen else 1,0);gate('Fluid power identity under saturation',max_err,1e-6);gate('Rotary torque bounds',max(0.,np.max(abs(bank.rotary)-bank.rotary_limit)),1e-6)
 report=dict(passed=all(x['passed'] for x in checks),checks=checks,metrics=metrics,static_states=25,gravity_vectors=3,acceleration_requests_per_state=3,floating_states=13,floating_coordinates=29,pacdm_states=len(branch),pacdm_full_rank=25,pacdm_physical_coordinates=32,hydraulic_parameters=PARAMETERS)
 save_json(ROOT/'results/mechanics_validation.json',report);print('Mechanics',sum(x['passed'] for x in checks),'/',len(checks),flush=True);return report
if __name__=='__main__':validate_mechanics()
