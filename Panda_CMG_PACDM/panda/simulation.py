"""Actual-contact Panda manipulation; PACDM references and Pinocchio feedforward."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
import mujoco
from .model import load_model
from .pin_backend import PinBackend

PHASES=[(0,3,'Approach'),(3,5,'Grasp'),(5,8,'Lift'),(8,12,'Clear & turn'),(12,16,'Tilt & disturb'),(16,19,'Insert'),(19,20,'Release'),(20,22,'Retract')]
PICK=np.array([.45,-.20]);DOCK=np.array([.45,.20]);PLINTH=.035

def build_scene(root,name='nominal',dt=.001,mass=.15,width=.044,friction=.8,offset=0.):
 root=Path(root);tree=ET.parse(root/'upstream/franka_emika_panda/panda.xml');e=tree.getroot();e.set('model','CMG PACDM Panda keyed cartridge transfer')
 e.find('compiler').set('meshdir','../upstream/franka_emika_panda/assets')
 opt=e.find('option');opt.attrib.update(timestep=str(dt),integrator='implicitfast',cone='elliptic',impratio='10',iterations='100',tolerance='1e-10')
 for elem in e.findall('keyframe'):e.remove(elem)
 vis=ET.SubElement(e,'visual');ET.SubElement(vis,'global',offwidth='1280',offheight='800');ET.SubElement(vis,'quality',shadowsize='4096');ET.SubElement(vis,'headlight',ambient='.3 .3 .3',diffuse='.75 .75 .75')
 for k in range(1,6):
  pad=e.find(f".//default[@class='fingertip_pad_collision_{k}']/geom")
  pad.attrib.update(friction=f'{friction} .005 .0001',condim='4',priority='1',solref='.006 1',solimp='.95 .99 .001')
 # Benchmark hand servo:10x source position stiffness/gain, original force cap.
 finger_act=e.find("actuator/general[@name='actuator8']")
 finger_act.set('gainprm','0.1568627450980392 0 0');finger_act.set('biasprm','0 -1000 -30')
 hand=e.find(".//body[@name='hand']");ET.SubElement(hand,'site',name='tool',pos='0 0 .1029',size='.003',rgba='0 .8 .9 .4',group='5')
 for b in e.findall('.//body'):
  for k,g in enumerate(b.findall('geom')):
   if not g.get('name'):g.set('name',f"{b.get('name')}_geom{k}")
 w=e.find('worldbody')
 ET.SubElement(w,'light',pos='1 -1 2',dir='-.3 .3 -1',diffuse='.8 .8 .8')
 ET.SubElement(w,'light',pos='-.5 1 1.5',dir='.3 -.2 -1',diffuse='.5 .6 .7')
 ET.SubElement(w,'geom',name='floor',type='plane',size='2 2 .02',rgba='.11 .15 .20 1',friction='.8 .005 .001')
 for label,xy in [('pick',PICK),('dock',DOCK)]:
  ET.SubElement(w,'geom',name=label+'_plinth',type='box',pos=f'{xy[0]} {xy[1]} {PLINTH/2}',size=f'.085 .075 {PLINTH/2}',rgba='.20 .28 .36 1')
  for dx in [-.075,.075]:
   for dy in [-.065,.065]:ET.SubElement(w,'geom',type='box',pos=f'{xy[0]+dx} {xy[1]+dy} {PLINTH+.001}',size='.004 .004 .001',rgba='.15 .85 .9 1',contype='0',conaffinity='0')
 # Keyed rectangular socket for a90-degree-rotated cartridge;3mm nominal clearance.
 sx,sy=.025,.018;wall=.005;h=.018
 for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
  p=np.r_[DOCK,PLINTH+h/2];size=np.array([sx+wall,sy+wall,h/2])
  p[axis]+=sign*([sx,sy][axis]+wall/2);size[axis]=wall/2
  ET.SubElement(w,'geom',name=f'socket_{axis}_{sign}',type='box',pos=' '.join(map(str,p)),size=' '.join(map(str,size)),rgba='.1 .58 .65 1',friction='.5 .005 .001',solref='.008 1')
 ET.SubElement(w,'geom',name='barrier',type='box',pos='.49 0 .10',size='.145 .018 .10',rgba='.28 .35 .43 1',friction='.6 .005 .001')
 ET.SubElement(w,'geom',name='barrier_accent',type='box',pos='.49 0 .201',size='.145 .018 .001',rgba='1 .62 .12 1',contype='0',conaffinity='0')
 obj=ET.SubElement(w,'body',name='cartridge',pos=f'{PICK[0]} {PICK[1]+offset} .0702')
 ET.SubElement(obj,'freejoint',name='object_free')
 ET.SubElement(obj,'inertial',mass=str(mass),pos='.003 0 0',diaginertia=' '.join(map(str,[mass*(width**2+.070**2)/12,mass*(.030**2+.070**2)/12,mass*(.030**2+width**2)/12])))
 ET.SubElement(obj,'geom',name='cartridge_collision',type='box',size=f'.015 {width/2} .035',rgba='.96 .58 .12 1',friction='.8 .005 .001',condim='4',solref='.006 1')
 for z in [-.026,.026]:ET.SubElement(obj,'geom',type='box',pos=f'0 0 {z}',size=f'.0152 {width/2+.0002} .002',rgba='.16 .22 .28 1',contype='0',conaffinity='0',mass='0')
 ET.SubElement(obj,'geom',type='box',pos='.0153 0 .002',size='.0002 .010 .012',rgba='.95 .97 1 1',contype='0',conaffinity='0',mass='0')
 path=root/'results'/f'{name}.xml';path.parent.mkdir(exist_ok=True);ET.indent(tree);tree.write(path,encoding='unicode');return path

def run_case(root,name='nominal',dt=.001,mass=.15,width=.044,friction=.8,offset=0.,grasp=True,feedforward=True):
 root=Path(root);cmg=load_model(root/'data/panda_cmg.json');backend=PinBackend(cmg)
 with np.load(root/'data/reference.npz',allow_pickle=False) as z:ref={k:z[k] for k in z.files}
 interp=BPoly.from_derivatives(ref['time'],np.stack([ref['q'],ref['v'],ref['a']],axis=1))
 actinterp=BPoly.from_derivatives(ref['time'],np.stack([ref['active'],ref['active_v'],ref['active_a']],axis=1))
 path=build_scene(root,name,dt,mass,width,friction,offset);m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
 qadr=np.array([m.jnt_qposadr[m.joint(j).id] for j in cmg['coordinate_ids']]);vadr=np.array([m.jnt_dofadr[m.joint(j).id] for j in cmg['coordinate_ids']])
 d.qpos[qadr]=ref['q'][0];mujoco.mj_forward(m,d)
 obj=m.body('cartridge').id;tool=m.site('tool').id;og=m.geom('cartridge_collision').id
 fingers={m.body('left_finger').id,m.body('right_finger').id}
 padids={i for i in range(m.ngeom) if m.geom_bodyid[i] in fingers and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX}
 robotbodies=set(range(1,obj));scene_obstacles={m.geom('barrier').id,m.geom('pick_plinth').id,m.geom('dock_plinth').id}|{m.geom(f'socket_{a}_{s}').id for a in [0,1] for s in [-1,1]}
 kp=m.actuator_gainprm[:7,0].copy();kd=-m.actuator_biasprm[:7,2].copy();damping=np.asarray(cmg['damping']);armature=np.asarray(cmg['armature'])
 limits=m.actuator_forcerange[:7].copy();ctrl_limits=m.actuator_ctrlrange[:7].copy()
 # Arm servo force before MuJoCo's forcerange clamp: fixed gain, affine bias, no activation dynamics.
 if not (np.all(m.actuator_gaintype[:7]==mujoco.mjtGain.mjGAIN_FIXED) and np.all(m.actuator_biastype[:7]==mujoco.mjtBias.mjBIAS_AFFINE) and np.all(m.actuator_dyntype[:7]==mujoco.mjtDyn.mjDYN_NONE)):raise ValueError('Arm actuators must be source position servos')
 servo_gain=m.actuator_gainprm[:7,0].copy();servo_bias=m.actuator_biasprm[:7,:3].copy();peakdemand=np.zeros(7)
 log={k:[] for k in ['time','qpos','qvel','q','v','q_ref','tool_pos','tool_R','target_pos','target_R','pose_error','angle_error','object_pos','object_R','normal_force','pad_contacts','torque','ctrl','wrench','gear_error','barrier_clearance','unexpected_contacts','bilateral_contact','left_normal_force','right_normal_force']}
 startrel=None;maxslip=0.;minair=1.;minclear=1.;peaknormal=0.;maxp=0.;maxr=0.;saturations=0;ctrlclips=0;unexpected=0;unexpected_instances=0;minmargin=1.;gear=0.;minleft=np.inf;minright=np.inf;minnormal=np.inf;mincontacts=99999;bilateral=True
 duration=float(ref['time'][-1]);steps=round(duration/dt)
 corner=np.array([[x,y,z] for x in [-.015,.015] for y in [-width/2,width/2] for z in [-.035,.035]])
 for k in range(steps+1):
  t=k*dt;qref=interp(t);vref=interp(t,nu=1);aref=interp(t,nu=2);active=actinterp(t)
  target_pos=active[:3];target_R=Rotation.from_euler('x',np.pi).as_matrix()@Rotation.from_euler('XYZ',active[3:6]).as_matrix()
  mujoco.mj_forward(m,d)
  q=d.qpos[qadr].copy();v=d.qvel[vadr].copy()
  ff=backend.inverse(q,v,aref)+armature*aref+damping*v if feedforward else np.zeros(9)
  cmd=qref[:7]+(kd*vref[:7]+ff[:7])/kp
  ctrlclips+=int(np.any((cmd<ctrl_limits[:,0])|(cmd>ctrl_limits[:,1])))
  d.ctrl[:7]=np.clip(cmd,ctrl_limits[:,0],ctrl_limits[:,1])
  d.ctrl[7]=np.clip(qref[7]/.04*255,0,255) if grasp else 255.
  pulse=np.zeros(6)
  if 13.0<=t<13.16:pulse[:3]=[2.0,-1.1,0.]
  if 14.5<=t<14.62:pulse[3:]=[.008,.012,0.]
  d.xfrc_applied[obj]=pulse
  mujoco.mj_forward(m,d)
  normal=0.;left_force=0.;right_force=0.;count=0;left=False;right=False;bad=0
  for i in range(d.ncon):
   c=d.contact[i];b1=m.geom_bodyid[c.geom1];b2=m.geom_bodyid[c.geom2]
   if og in [c.geom1,c.geom2] and (c.geom1 in padids or c.geom2 in padids):
    f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);normal+=f[0];count+=1
    fb=b2 if c.geom1==og else b1;left|=fb==m.body('left_finger').id;right|=fb==m.body('right_finger').id
    if fb==m.body('left_finger').id:left_force+=f[0]
    else:right_force+=f[0]
   if (c.geom1 in scene_obstacles and b2 in robotbodies) or (c.geom2 in scene_obstacles and b1 in robotbodies):bad+=1
   if (c.geom1==m.geom('barrier').id and c.geom2==og) or (c.geom2==m.geom('barrier').id and c.geom1==og):bad+=1
   if b1 in robotbodies and b2 in robotbodies:bad+=1
   if (c.geom1==m.geom('floor').id and b2 in robotbodies-{m.body('link0').id}) or (c.geom2==m.geom('floor').id and b1 in robotbodies-{m.body('link0').id}):bad+=1
   if (c.geom1==og and b2 in robotbodies-fingers) or (c.geom2==og and b1 in robotbodies-fingers):bad+=1
  p=d.site_xpos[tool].copy();Q=d.site_xmat[tool].reshape(3,3).copy();objp=d.xpos[obj].copy();objR=d.xmat[obj].reshape(3,3).copy()
  pe=float(np.linalg.norm(p-target_pos));ae=float(Rotation.from_matrix(target_R.T@Q).magnitude())
  clear=float(np.min((objp+(objR@corner.T).T)[:,2])-.20)
  if startrel is None and t>=7.99:startrel=Q.T@(objp-p)
  if 8<=t<=16:
   minleft=min(minleft,left_force);minright=min(minright,right_force);minnormal=min(minnormal,normal);mincontacts=min(mincontacts,count);bilateral=bilateral and left and right
   maxslip=max(maxslip,float(np.linalg.norm(Q.T@(objp-p)-startrel)));minair=min(minair,objp[2]);peaknormal=max(peaknormal,normal)
  if 8<=t<=12:minclear=min(minclear,clear)
  maxp=max(maxp,pe);maxr=max(maxr,ae);unexpected+=int(bad>0);unexpected_instances+=bad
  demand=servo_gain*d.ctrl[:7]+servo_bias[:,0]+servo_bias[:,1]*d.actuator_length[:7]+servo_bias[:,2]*d.actuator_velocity[:7];peakdemand=np.maximum(peakdemand,np.abs(demand))
  saturations+=int(np.any(np.abs(d.actuator_force[:7])>=limits[:,1]-.001));minmargin=min(minmargin,float(np.min(np.r_[q[:7]-m.jnt_range[:7,0],m.jnt_range[:7,1]-q[:7]])))
  ge=abs(q[7]-q[8]);gear=max(gear,ge)
  if k%max(1,round(.01/dt))==0:
   vals=[t,d.qpos.copy(),d.qvel.copy(),q,v,qref,p,Q,target_pos,target_R,pe,ae,objp,objR,normal,count,d.actuator_force[:7].copy(),d.ctrl.copy(),pulse,ge,clear,bad,int(left and right),left_force,right_force]
   for key,val in zip(log,vals):log[key].append(val)
  if k<steps:mujoco.mj_step(m,d)
 arr={k:np.asarray(v) for k,v in log.items()};np.savez_compressed(root/'results'/f'{name}.npz',**arr)
 mask=(arr['time']>=8)&(arr['time']<=16);final=arr['object_pos'][-1];oa=m.jnt_dofadr[m.joint('object_free').id]
 targetobject=Rotation.from_euler('z',np.pi/2).as_matrix();angle=Rotation.from_matrix(targetobject.T@arr['object_R'][-1]).magnitude()
 summary=dict(case=name,dt_s=dt,payload_mass_kg=mass,payload_width_m=width,pad_friction=friction,initial_offset_m=offset,grasp_enabled=grasp,feedforward_enabled=feedforward,
  max_lift_m=float(np.max(arr['object_pos'][:,2])-.0702),min_transfer_height_m=float(minair),max_grasp_slip_m=float(maxslip),minimum_payload_barrier_clearance_m=float(minclear),peak_transfer_normal_N=float(peaknormal),min_transfer_normal_N=float(minnormal),min_transfer_pad_contacts=int(mincontacts),min_transfer_left_normal_N=float(minleft),min_transfer_right_normal_N=float(minright),bilateral_contact_all_transfer=bool(bilateral),
  final_object_position_m=final.tolist(),placement_xy_error_m=float(np.linalg.norm(final[:2]-DOCK)),placement_z_error_m=float(abs(final[2]-.070)),placement_angle_error_deg=float(np.rad2deg(angle)),final_speed_m_s=float(np.linalg.norm(d.qvel[oa:oa+3])),final_angular_speed_rad_s=float(np.linalg.norm(d.qvel[oa+3:oa+6])),
  max_tool_position_error_m=float(maxp),max_tool_orientation_error_deg=float(np.rad2deg(maxr)),rms_tool_position_error_m=float(np.sqrt(np.mean(arr['pose_error']**2))),arm_saturation_steps=int(saturations),peak_arm_torque_demand_Nm=peakdemand.tolist(),arm_control_clip_steps=int(ctrlclips),min_arm_joint_margin_rad=float(minmargin),unexpected_contact_steps=int(unexpected),unexpected_contact_instances=int(unexpected_instances),max_finger_coupling_error_m=float(gear),maximum_arm_torque_Nm=np.max(np.abs(arr['torque']),axis=0).tolist(),warnings={str(i):int(x.number) for i,x in enumerate(d.warning) if x.number})
 (root/'results'/f'{name}.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');return summary
