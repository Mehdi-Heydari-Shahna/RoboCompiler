"""Native contact plant: no robot/grain state overwrite after initialization."""
import time
import numpy as np
import mujoco
from benchmark_model import *
from digging_path import make_path,reference_at
from digging_export import export_scene,bucket_inside
from hydraulic_actuators import HydraulicBank
from soil_cohesion import CohesiveBed
from digging_export import SOIL
POWER_KEYS=['supply','throttle','leakage','relief','friction','compressibility_geometry','rotary','mechanical','equality','contact','cohesion']
class DiggingController:
 def __init__(self,e,r,inverse,path):self.e=e;self.r=r;self.inverse=inverse;self.path=path;self.seed=path['closed_tree'][0].copy();self.shadow_max=0.;self.command_peak=np.zeros(8)
 def evaluate(self,t,q,v):
  u=q[self.e.active];ud=v[self.e.active];closed=self.r.reconstruct(u,self.seed);self.seed=closed.copy();cmp=self.r.component(closed,ud,GRAVITY);des,dv,da=reference_at(t,self.path)
  acc=da[:6]+144*(des[:6]-u[:6])+24*(dv[:6]-ud[:6]);force=self.inverse.solve(cmp,acc,p0_force=0.)['physical_efforts']
  self.shadow_max=max(self.shadow_max,float(np.max(abs(q-closed))));self.command_peak=np.maximum(self.command_peak,abs(force));return force
def contact_wrenches(m,d,bids):
 out=np.zeros((len(bids),6));counts=np.zeros(len(bids),int);raw=np.zeros(6)
 for k in range(d.ncon):
  c=d.contact[k];b1=m.geom_bodyid[c.geom1];b2=m.geom_bodyid[c.geom2]
  if not any(b in (b1,b2) for b in bids):continue
  mujoco.mj_contactForce(m,d,k,raw);frame=c.frame.reshape(3,3)
  for i,bid in enumerate(bids):
   if bid not in (b1,b2):continue
   sign=1 if bid==b2 else -1;f=sign*(frame.T@raw[:3]);torque=sign*(frame.T@raw[3:]);out[i,:3]+=f;out[i,3:]+=torque+np.cross(c.pos-d.xipos[bid],f);counts[i]+=1
 return out,counts

def run_case(name,*,dt=.0005,duration=14.,soil=True,bucket_contact=True,friction=.65,density_scale=1.,capacity_scale=1.,free_base=True,save=True,solver="Newton",iterations=100,cohesion=False):
 started=time.perf_counter();outdir=ROOT/'results';outdir.mkdir(exist_ok=True);cmg,mapping,e,a,r,inverse=context();path=make_path(e,r);xml=outdir/'assets'/f'{name}.xml'
 meta=export_scene(xml,dt=dt,soil=soil,bucket_contact=bucket_contact,friction=friction,density_scale=density_scale,free_base=free_base,solver=solver,iterations=iterations)
 m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);jids=[m.joint(k).id for k in e.tree_ids];qi=m.jnt_qposadr[jids];vi=m.jnt_dofadr[jids]
 pi=np.array([e.tree_ids.index(k) for k in PORTS]);active=np.array(e.active);d.qpos[qi]=path['closed_tree'][0];d.qvel[:]=0
 ctrl=DiggingController(e,r,inverse,path);command=ctrl.evaluate(0,d.qpos[qi],d.qvel[vi]);bank=HydraulicBank(r.nominal[pi],command,capacity_scale)
 bid=m.body('body_56').id;base_id=m.body('body_53').id;bodyids=[m.body(b['id']).id for b in cmg['bodies'] if b['kind']=='rigid_body']
 grains=np.array([m.body(f'grain_{i:03d}').id for i in range(meta['particle_count'])],int);gm=m.body_mass[grains];initial_grains=np.array(meta['particle_initial_positions_m']).reshape(-1,3)
 gqi=np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in grains]);gvi=np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in grains]);bed=CohesiveBed(initial_grains,SOIL['radius_m'],enabled=cohesion)
 keys=['time','qpos','qvel','independent','desired','effort','command','pressure','flow','energy','fluid_energy','work','power','wrenches','contact_count','captured_mass','lifted_mass','base_pose','loop_gap','eq_force','body_internal_wrench','cohesive_energy','fracture_energy','grain_external_force','body_energy','body_contact_wrench','port_position','port_velocity','port_power','fluid_energy_by_cylinder']
 records={k:[] for k in keys};work=np.zeros(len(POWER_KEYS));absolute_work=work.copy();previous_power=None;initial_energy=None
 peak_loop=peak_hyd_identity=peak_power_identity=peak_equilibrium=0.;peak_error=np.zeros(6);peak_force=np.zeros(8);peak_pressure=peak_speed=0.;min_pressure=1e99;max_contacts=0
 valve_steps=pressure_steps=0;sample_every=max(1,round(.01/dt));control_every=max(1,round(.01/dt));warning_before=np.array([x.number for x in d.warning]);nsteps=round(duration/dt)
 carried_ever=np.zeros(len(grains),bool);captured_ever=carried_ever.copy();settled_grains=None
 for step in range(nsteps+1):
  t=step*dt;q=d.qpos[qi].copy();v=d.qvel[vi].copy()
  if step%control_every==0:command=ctrl.evaluate(t,q,v)
  item=bank.evaluate(q[pi],v[pi],command,dt);d.ctrl[:]=item['effort']
  gp=np.array([d.qpos[j:j+3] for j in gqi]).reshape(-1,3);gv=np.array([d.qvel[j:j+3] for j in gvi]).reshape(-1,3);bond_force,bond_energy,bond_power=bed.evaluate(gp,gv)
  d.xfrc_applied[:]=0;d.xfrc_applied[grains,:3]=bond_force;mujoco.mj_forward(m,d)
  jv=np.zeros(d.nefc);mujoco.mj_mulJacVec(m,d,jv,d.qvel);eq=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  ep=float(d.efc_force[eq]@jv[eq]);cp=float(d.efc_force[~eq]@jv[~eq]);power=np.array([item['power'].get(k,ep if k=='equality' else bond_power if k=='cohesion' else cp) for k in POWER_KEYS])
  if previous_power is not None:work+=dt*(power+previous_power)/2;absolute_work+=dt*(abs(power)+abs(previous_power))/2
  previous_power=power.copy();energy=float(d.energy.sum());fluid_energy=bank.stored_energy(q[pi])
  if initial_energy is None:initial_energy=(energy,fluid_energy,bond_energy)
  peak_power_identity=max(peak_power_identity,abs(float(d.qfrc_actuator@d.qvel)-item['power']['mechanical']),abs(float(d.qfrc_constraint@d.qvel)-(ep+cp)))
  peak_hyd_identity=max(peak_hyd_identity,abs(item['fluid_identity_W']));loop=float(np.max(abs(d.efc_pos[eq]),initial=0));peak_loop=max(peak_loop,loop)
  des,_,_=reference_at(t,path);err=abs(q[active[:6]]-des[:6]);peak_error=np.maximum(peak_error,err);peak_force=np.maximum(peak_force,abs(item['effort']))
  peak_pressure=max(peak_pressure,float(bank.pressure.max()));min_pressure=min(min_pressure,float(bank.pressure.min()));peak_speed=max(peak_speed,float(np.max(abs(d.qvel[vi]))));max_contacts=max(max_contacts,d.ncon)
  valve_steps+=int(item['valve_saturated']);pressure_steps+=int(item['pressure_limit_active'])
  if step%sample_every==0 or step==nsteps:
   all_wr,all_counts=contact_wrenches(m,d,bodyids);wr=all_wr[[bodyids.index(bid),bodyids.index(base_id)]];counts=all_counts[[bodyids.index(bid),bodyids.index(base_id)]];mujoco.mj_rnePostConstraint(m,d)
   body_energy=[];jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
   for b in bodyids:
    mujoco.mj_jacBodyCom(m,d,jp,jr,b);vv=jp@d.qvel;ww=d.ximat[b].reshape(3,3).T@(jr@d.qvel)
    body_energy.append([.5*m.body_mass[b]*(vv@vv)+.5*np.dot(m.body_inertia[b],ww**2),-m.body_mass[b]*GRAVITY@d.xipos[b]])
   ma=np.zeros(m.nv);mujoco.mj_mulM(m,d,ma,d.qacc);res=ma-d.qfrc_smooth-d.qfrc_constraint
   peak_equilibrium=max(peak_equilibrium,float(np.max(abs(res)))/max(1.,float(np.max(abs(ma))),float(np.max(abs(d.qfrc_bias)))))
   if len(grains):
    local=(d.xpos[grains]-d.xpos[bid])@d.xmat[bid].reshape(3,3);inside=bucket_inside(local);captured=float(gm[inside].sum());lift=d.xpos[grains,2]>.9;lifted=float(gm[lift].sum());captured_ever|=inside
    if 7<=t<=10:carried_ever|=inside&lift
    if t>=1 and settled_grains is None:settled_grains=d.xpos[grains].copy()
   else:captured=lifted=0.
   vals=dict(time=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),independent=q[active],desired=des[:6],effort=item['effort'],command=command.copy(),pressure=bank.pressure.copy(),flow=item['flow'],energy=energy,fluid_energy=fluid_energy,work=work.copy(),power=power,wrenches=wr,contact_count=counts,captured_mass=captured,lifted_mass=lifted,base_pose=np.r_[d.xpos[base_id],d.xquat[base_id]],loop_gap=loop,eq_force=d.efc_force[eq].copy(),body_internal_wrench=d.cfrc_int[bodyids].copy(),cohesive_energy=bond_energy,fracture_energy=bed.fracture_J,grain_external_force=bond_force.copy(),body_energy=body_energy,body_contact_wrench=all_wr,port_position=q[pi],port_velocity=v[pi],port_power=item['effort']*v[pi],fluid_energy_by_cylinder=np.sum(bank.volumes(q[pi])*bank.pressure**2,axis=1)/(2*bank.p['bulk_modulus_Pa']))
   for k in keys:records[k].append(vals[k])
  if step%max(1,round(2/dt))==0:print(f'{name}: t={t:4.1f}s loop={loop:.2e}m err={np.max(err):.3g}rad contacts={d.ncon} load={captured:.1f}kg it={int(d.solver_niter[0])}',flush=True)
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise ValueError('Nonfinite native state')
  if np.any(np.array([x.number for x in d.warning])!=warning_before):raise ValueError('Native solver warning')
  if step==nsteps:break
  mujoco.mj_step(m,d);bank.advance(item,dt)
 result={k:np.asarray(v) for k,v in records.items()};wi=dict(zip(POWER_KEYS,map(float,work)));aw=dict(zip(POWER_KEYS,map(float,absolute_work)))
 mechanical_defect=energy-initial_energy[0]-wi['mechanical']-wi['equality']-wi['contact']-wi['cohesion']
 rhs=wi['supply']-wi['throttle']-wi['leakage']-wi['relief']-wi['friction']+wi['compressibility_geometry']+wi['rotary']+wi['equality']+wi['contact'];total_defect=energy+fluid_energy+bond_energy-sum(initial_energy)-rhs+bed.fracture_J
 if len(grains):
  final=d.xpos[grains].copy();moved=np.linalg.norm(final-(settled_grains if settled_grains is not None else initial_grains),axis=1)>.30;dumped=(final[:,1]>1.)&(final[:,2]<.80)&carried_ever
  displaced=float(gm[moved].sum());dumped_mass=float(gm[dumped].sum())
 else:displaced=dumped_mass=0.
 report=dict(name=name,cohesion=cohesion,bond_count=len(bed.pairs),broken_bonds=int(np.sum(~bed.active)) if cohesion else 0,fracture_energy_J=bed.fracture_J,cohesive_energy_change_J=bond_energy-initial_energy[2],completed=True,dt_s=dt,duration_s=duration,elapsed_s=time.perf_counter()-started,scene=meta,source_bodies=25,source_mass_kg=sum(b['mass_kg'] for b in cmg['bodies']),physical_ports=PORTS,
  peak_effort_SI=peak_force,peak_command_SI=ctrl.command_peak,peak_loop_gap_m=peak_loop,peak_tracking_error_rad=peak_error,final_tracking_error_rad=err,shadow_position_max_SI=ctrl.shadow_max,peak_tree_speed_SI=peak_speed,peak_pressure_Pa=peak_pressure,min_pressure_Pa=min_pressure,supply_pressure_Pa=bank.ps,
  pressure_limit_fraction=pressure_steps/(nsteps+1),valve_limit_fraction=valve_steps/(nsteps+1),peak_power_identity_error_W=peak_power_identity,peak_fluid_identity_error_W=peak_hyd_identity,peak_native_equilibrium_relative=peak_equilibrium,
  work_J=wi,absolute_work_J=aw,mechanical_energy_change_J=energy-initial_energy[0],fluid_energy_change_J=fluid_energy-initial_energy[1],mechanical_balance_defect_J=mechanical_defect,mechanical_balance_relative=abs(mechanical_defect)/max(1.,aw['mechanical']+aw['equality']+aw['contact']+aw['cohesion']),total_balance_defect_J=total_defect,total_balance_relative=abs(total_defect)/max(1.,sum(aw.values())-aw['mechanical']-aw['cohesion']),
  peak_captured_mass_kg=float(result['captured_mass'].max()),peak_lifted_mass_kg=float(result['lifted_mass'].max()),carried_particle_mass_kg=float(gm[carried_ever].sum()),final_displaced_mass_kg=displaced,deposited_carried_mass_kg=dumped_mass,max_contacts=max_contacts,
  native_state_projection=False,q22_actuated=False,p0_command_N=0.,p0_source_resolved=False,warning_count=0,power_order=POWER_KEYS,body_wrench_order=[b['id'] for b in cmg['bodies'] if b['kind']=='rigid_body'],body_wrench_semantics='MuJoCo cfrc_int, rotation first, world-oriented COM-based subtree frame. Representation-dependent tree interaction wrenches, not unique cut-pin loads.')
 if save:np.savez_compressed(outdir/f'{name}.npz',**result);save_json(outdir/f'{name}.json',report);save_json(outdir/'digging_path.json',path)
 return report,result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--name',default='development');p.add_argument('--dt',type=float,default=.001);p.add_argument('--duration',type=float,default=14.);p.add_argument('--empty',action='store_true');p.add_argument('--no-bucket-contact',action='store_true');a=p.parse_args()
 report,_=run_case(a.name,dt=a.dt,duration=a.duration,soil=not a.empty,bucket_contact=not a.no_bucket_contact);print({k:v for k,v in report.items() if k!='scene'})
