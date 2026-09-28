"""Independent energy sums and contact-wrench/generalized-force agreement."""
import numpy as np
import mujoco
from benchmark_model import *
from floating_dynamics import FloatingSource
from digging_export import SOIL

def audit_contact_run(name):
 a=load_trace(name);m=mujoco.MjModel.from_xml_path(str(ROOT/'results/assets'/f'{name}.xml'));d=mujoco.MjData(m)
 c,mp,e,mapper,r,inv=context();source=FloatingSource(c,e.source)
 js=[m.joint(x).id for x in e.tree_ids];qi=m.jnt_qposadr[js];vi=m.jnt_dofadr[js]
 j=next(x for x in c['joints'] if x['id']=='fixed_world_base');T0=np.asarray(j['T_BJ'])@np.linalg.inv(np.asarray(j['T_FJ']));base=m.body('body_53').id
 particles=[bid for bid in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid) or '').startswith('grain_')]
 max_energy=max_contact=peak_pair_force=max_body_energy=0.;records=[]
 recorded=json.loads((ROOT/'results'/f'{name}.json').read_text());body_lookup={b['id']:b for b in c['bodies']}
 for k in np.linspace(0,len(a['time'])-1,61).astype(int):
  d.qpos[:]=a['qpos'][k];d.qvel[:]=a['qvel'][k];d.ctrl[:]=a['effort'][k];d.xfrc_applied[:]=0
  if 'grain_external_force' in a:d.xfrc_applied[particles,:3]=a['grain_external_force'][k]
  mujoco.mj_forward(m,d)
  RB=d.xmat[base].reshape(3,3);R=RB@T0[:3,:3].T;offset=R@T0[:3,3];p=d.xpos[base]-offset;omega=RB@d.qvel[3:6]
  v=np.r_[d.qvel[:3]-np.cross(omega,offset),omega,d.qvel[vi]];z=source.evaluate(d.qpos[qi],p,R,v,GRAVITY)
  energy=z['kinetic_energy']+z['potential_energy'];mass_soil=0.
  if 'body_energy' in a:
   for j,bid in enumerate(recorded['body_wrench_order']):
    b=body_lookup[bid];T=z['poses'][bid];vc=z['body_com_jacobians'][bid]@v;omega=z['body_velocities'][bid][3:];Iw=T[:3,:3]@b['inertia_com_kg_m2']@T[:3,:3].T
    ek=.5*b['mass_kg']*(vc@vc)+.5*omega@Iw@omega;eu=-b['mass_kg']*GRAVITY@(T[:3,3]+T[:3,:3]@b['com_m'])
    max_body_energy=max(max_body_energy,float(np.max(abs(a['body_energy'][k,j]-[ek,eu]))))
  for b in particles:
   jid=m.body_jntadr[b];qj=m.jnt_qposadr[jid];vj=m.jnt_dofadr[jid];vel=d.qvel[vj:vj+6];mass=m.body_mass[b]
   energy+=.5*mass*(vel[:3]@vel[:3])+.5*np.dot(m.body_inertia[b],vel[3:]**2)-mass*GRAVITY@d.qpos[qj:qj+3];mass_soil+=mass
  error=abs(energy-a['energy'][k]);max_energy=max(max_energy,error)
  explicit=np.zeros(m.nv);raw=np.zeros(6)
  for n in range(d.ncon):
   contact=d.contact[n];mujoco.mj_contactForce(m,d,n,raw);frame=contact.frame.reshape(3,3);f=frame.T@raw[:3];torque=frame.T@raw[3:]
   b1=m.geom_bodyid[contact.geom1];b2=m.geom_bodyid[contact.geom2]
   if b2:mujoco.mj_applyFT(m,d,f,torque,contact.pos,b2,explicit)
   if b1:mujoco.mj_applyFT(m,d,-f,-torque,contact.pos,b1,explicit)
   peak_pair_force=max(peak_pair_force,float(np.linalg.norm(f)))
  ef=d.efc_force.copy();ef[d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)]=0
  generalized=np.zeros(m.nv);mujoco.mj_mulJacTVec(m,d,generalized,ef)
  rel=np.max(abs(explicit-generalized))/max(1.,np.max(abs(generalized)));max_contact=max(max_contact,float(rel))
  records.append(dict(time=float(a['time'][k]),source_energy_J=energy,native_energy_J=float(a['energy'][k]),energy_error_J=error,contact_force_relative_error=float(rel)))
 report=dict(name=name,snapshots=len(records),max_independent_energy_error_J=max_energy,max_individual_body_energy_error_J=max_body_energy,max_contact_wrench_mapping_relative=max_contact,peak_contact_pair_force_N=peak_pair_force,soil_mass_kg=mass_soil,records=records,passed=bool(max_energy<1e-6 and max_body_energy<1e-6 and max_contact<1e-8))
 save_json(ROOT/'results'/f'{name}_contact_audit.json',report);return report
if __name__=='__main__':
 import sys
 print(audit_contact_run(sys.argv[1]))
