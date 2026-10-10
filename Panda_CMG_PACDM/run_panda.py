"""Reproduce the CMG/PACDM Panda manipulation benchmark and its evidence."""
from pathlib import Path
import argparse,hashlib,json,os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco,pinocchio as pin
from scipy.spatial.transform import Rotation
from panda.model import build_model,save_model
from panda.task import build_reference
from panda.task_validation import validate_task
from panda.validation import validate_mechanics
from panda.simulation import run_case
ROOT=Path(__file__).resolve().parent
RUN_MANIFEST=Path('results')/'SHA256SUMS.json'
CASES={'nominal':{},'fine':dict(dt=.0005),'heavy_low_friction':dict(mass=.30,friction=.5),'offset_pick':dict(offset=.004),'tight_socket':dict(width=.048,offset=-.003),'no_grasp':dict(grasp=False),'no_feedforward':dict(feedforward=False)}
POSITIVE=['nominal','fine','heavy_low_friction','offset_pick','tight_socket']

def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read_json(path):return json.loads(Path(path).read_text())
def aggregate(root=ROOT):
 r=root/'results';mechanics=read_json(r/'mechanics.json');task=read_json(r/'task_validation.json');cases={name:read_json(r/f'{name}.json') for name in CASES}
 checks={'mechanics.'+k:v for k,v in mechanics['checks'].items()};checks.update({'task.'+k:v for k,v in task['checks'].items()})
 def gate(name,v,limit,unit='',relation='<='):
  ok=np.isfinite(v) and (v<=limit if relation=='<=' else v>=limit if relation=='>=' else v==limit)
  checks[name]=dict(passed=bool(ok),value=float(v),limit=float(limit),unit=unit,relation=relation)
 for name,c in cases.items():
  gate(name+'.numerical_warnings',sum(c['warnings'].values()),0,'count','==')
  gate(name+'.arm_joint_margin',c['min_arm_joint_margin_rad'],.05,'rad','>=')
  # Servo torque demanded before MuJoCo's forcerange clamp; the applied (clamped) torque is within bounds by construction.
  gate(name+'.source_torque_bounds',max(np.asarray(c['peak_arm_torque_demand_Nm'])/np.array([87,87,87,87,12,12,12])),1.000001,'ratio')
  gate(name+'.finger_coupling',c['max_finger_coupling_error_m'],.0005,'m')
  if name in POSITIVE:
   gate(name+'.no_unexpected_contact',c['unexpected_contact_steps'],0,'steps','==')
   gate(name+'.no_arm_saturation',c['arm_saturation_steps'],0,'steps','==')
   gate(name+'.no_setpoint_clipping',c['arm_control_clip_steps'],0,'steps','==')
   gate(name+'.lift',c['max_lift_m'],.23,'m','>=')
   gate(name+'.airborne_transfer',c['min_transfer_height_m'],.28,'m','>=')
   gate(name+'.barrier_clearance',c['minimum_payload_barrier_clearance_m'],.05,'m','>=')
   gate(name+'.grasp_slip',c['max_grasp_slip_m'],.005,'m')
   gate(name+'.left_finger_load',c['min_transfer_left_normal_N'],.1,'N','>=')
   gate(name+'.right_finger_load',c['min_transfer_right_normal_N'],.1,'N','>=')
   gate(name+'.bilateral_contacts',c['bilateral_contact_all_transfer'],1,'boolean','==')
   gate(name+'.tool_translation',c['max_tool_position_error_m'],.002,'m')
   gate(name+'.tool_rotation',c['max_tool_orientation_error_deg'],.5,'deg')
   gate(name+'.socket_xy',c['placement_xy_error_m'],.003,'m')
   gate(name+'.socket_z',c['placement_z_error_m'],.001,'m')
   gate(name+'.socket_rotation',c['placement_angle_error_deg'],2.,'deg')
   gate(name+'.settled_translation',c['final_speed_m_s'],.001,'m/s')
   gate(name+'.settled_rotation',c['final_angular_speed_rad_s'],.02,'rad/s')
 gate('negative_control.open_hand_no_lift',cases['no_grasp']['max_lift_m'],.03,'m')
 gate('negative_control.open_hand_misses_socket',cases['no_grasp']['placement_xy_error_m'],.15,'m','>=')
 gate('feedforward_ablation.tool_RMS_ratio',cases['nominal']['rms_tool_position_error_m']/cases['no_feedforward']['rms_tool_position_error_m'],.5,'ratio')
 with np.load(r/'nominal.npz') as a,np.load(r/'fine.npz') as b:
  if not np.allclose(a['time'],b['time'],atol=1e-12):raise RuntimeError('Refinement requires common sample times')
  pos=float(np.max(np.linalg.norm(a['object_pos']-b['object_pos'],axis=1)))
  angle=float(np.rad2deg(np.max((Rotation.from_matrix(a['object_R']).inv()*Rotation.from_matrix(b['object_R'])).magnitude())))
 gate('refinement.object_position',pos,.002,'m');gate('refinement.object_rotation',angle,.5,'deg')
 m=mujoco.MjModel.from_xml_path(str(r/'nominal.xml'))
 gate('scene.only_physical_finger_equality',m.neq,1,'count','==');gate('scene.seven_arm_and_one_hand_actuators',m.nu,8,'count','==');gate('scene.free_object',m.jnt_type[m.joint('object_free').id],int(mujoco.mjtJoint.mjJNT_FREE),'enum','==')
 result=dict(passed=all(c['passed'] for c in checks.values()),passed_count=sum(c['passed'] for c in checks.values()),check_count=len(checks),checks=checks,cases=cases,reference=read_json(root/'data/reference.json'),mechanics_details=mechanics['details'],task_details=task['details'],
  task='22s real-contact keyed-cartridge pick, lift, barrier clearance,90deg yaw, tilted inspection with elbow redundancy sweep and force/torque pulses, socket insertion, release and retreat.',
  framework='Original PACDM core unchanged; authored CMG source import; imposed SE3 tool task + physical finger coupling; 15-coordinate task graph with rank7 and8 independent coordinates; native Pinocchio tree RNEA feedforward.',
  scope=['The Panda is a serial arm; its tool-target residual represents an imposed virtual task.',
  'MuJoCo integrates native contact. Pinocchio checks rigid-body mechanics and supplies model-based feedforward.',
  'All seven original bounded arm position actuators receive feedforward through their setpoints; the payload remains a free rigid body.',
  'The benchmark hand servo uses10x source position stiffness/gain,30Ns/m damping and the original100N tendon-effort cap; pad contact is condim4 with0.005m torsional friction length.',
  'The fixed socket has 3 mm per-side nominal clearance; the 48 mm cartridge narrows one clearance to 1 mm.',
  'The route and elbow schedule are prescribed, and collision outcomes are measured in simulation.',
  'All authored inertias are preserved. MuJoCo diagonalization introduces small documented tensor differences; the CMG importer does not read compiled MuJoCo dynamics arrays.',
  'The reported evidence consists of finite simulation runs and numerical checks under the pinned environment.'],
  ablation_scope='The feedforward ablation removes Pinocchio inverse dynamics and source damping compensation while preserving the PACDM reference and desired-velocity term.')
 write_json(r/'validation.json',result);return result

def manifest(root=ROOT):
 project_dirs={'panda','vendor','upstream','tools','data','results'}
 project_files={'run_panda.py','run_panda.bat',
                'requirements-linux.txt','environment.yml','.gitignore'}
 excluded={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}
 files=(p for p in root.rglob('*') if p.is_file()
        and (p.relative_to(root).parts[0] in project_dirs
             or (len(p.relative_to(root).parts)==1 and p.name in project_files))
        and not excluded.intersection(p.relative_to(root).parts)
        and p.suffix!='.pyc' and p.name!='SHA256SUMS.json')
 # Run evidence manifest: written to results/, never to a tracked file.
 write_json(root/RUN_MANIFEST,{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(files)})
def verify(root=ROOT):
 if not (root/RUN_MANIFEST).is_file():raise SystemExit(f'No run evidence manifest ({RUN_MANIFEST.as_posix()}); run python run_panda.py first.')
 hashes=read_json(root/RUN_MANIFEST);bad=[n for n,h in hashes.items() if not (root/n).is_file() or hashlib.sha256((root/n).read_bytes()).hexdigest()!=h]
 if bad:raise RuntimeError('Missing/modified files: '+', '.join(bad))
 print(f'Integrity verified: {len(hashes)} files.')
def _same_values(a,b,atol=1e-12):
 """Structural equality that ignores floating-point round-off."""
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_same_values(a[k],b[k],atol) for k in a)
 if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(_same_values(x,y,atol) for x,y in zip(a,b))
 if isinstance(a,(int,float)) and not isinstance(a,bool):return isinstance(b,(int,float)) and not isinstance(b,bool) and bool(np.isclose(a,b,rtol=0,atol=atol))
 return a==b
def _same_arrays(generated,shipped,rtol=1e-6,atol=1e-7):
 with np.load(generated,allow_pickle=False) as a,np.load(shipped,allow_pickle=False) as b:
  if sorted(a.files)!=sorted(b.files):return 'array names differ'
  for k in a.files:
   x,y=a[k],b[k]
   if x.shape!=y.shape:return f'{k}: shape {x.shape} != {y.shape}'
   if x.dtype.kind in 'fc' or y.dtype.kind in 'fc':
    if not np.allclose(x,y,rtol=rtol,atol=atol):return f'{k}: max |difference| {float(np.max(np.abs(x-y))):.3g}'
   elif not np.array_equal(x,y):return f'{k}: values differ'
 return ''
def check_shipped_data(root=ROOT):
 """The cases read data/panda_cmg.json and data/reference.npz, which this run does not rewrite.
 They must match the model and PACDM reference that the run has just rebuilt into results/."""
 r=root/'results';problems=[]
 if not _same_values(read_json(r/'panda_cmg.json'),read_json(root/'data/panda_cmg.json')):problems.append('panda_cmg.json')
 why=_same_arrays(r/'reference.npz',root/'data/reference.npz')
 if why:problems.append('reference.npz ('+why+')')
 if problems:raise RuntimeError('The rebuilt '+', '.join(problems)+' in results/ differ from the shipped copies in data/. '
  'If the model or task change is intended, copy results/panda_cmg.json, results/reference.npz and results/reference.json into data/ and rerun.')
def replay(root=ROOT):
 import time,mujoco.viewer
 z=np.load(root/'results/nominal.npz');m=mujoco.MjModel.from_xml_path(str(root/'results/nominal.xml'));d=mujoco.MjData(m)
 with mujoco.viewer.launch_passive(m,d) as viewer:
  start=time.perf_counter()
  while viewer.is_running():
   t=(time.perf_counter()-start)%z['time'][-1];i=min(np.searchsorted(z['time'],t),len(z['time'])-1);d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);viewer.sync();time.sleep(.01)

def _job(item):return run_case(ROOT,name=item[0],**item[1])
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render',action='store_true');parser.add_argument('--workers',type=int,default=3);parser.add_argument('--audit-existing',action='store_true');parser.add_argument('--replay',action='store_true');args=parser.parse_args()
 if args.audit_existing:verify();return
 if args.replay:replay();return
 if mujoco.__version__!='3.3.7' or pin.__version__!='3.8.0':raise RuntimeError('MuJoCo 3.3.7 and Pinocchio 3.8.0 are required.')
 r=ROOT/'results';r.mkdir(exist_ok=True);write_json(r/'validation.json',dict(passed=False,status='Run in progress; acceptance incomplete.'))
 # Run outputs go to results/ only; tracked files in data/ are checked, never rewritten.
 cmg=build_model();save_model(cmg,r/'panda_cmg.json')
 print('Checking independent source mechanics...',flush=True);mechanics=validate_mechanics(cmg,output=r/'mechanics.json')
 if not mechanics['passed']:raise RuntimeError('Mechanics failed; see results/mechanics.json')
 print('Assembling 22 s redundant tool task with PACDM...',flush=True);reference=build_reference(cmg)
 np.savez_compressed(r/'reference.npz',**{k:v for k,v in reference.items() if k!='info'});write_json(r/'reference.json',reference['info'])
 check_shipped_data()
 task=validate_task(cmg,reference,output=r/'task_validation.json')
 if not task['passed']:raise RuntimeError('Task reference failed; see results/task_validation.json')
 print('Running seven actual-contact cases...',flush=True)
 if args.workers>1:
  from concurrent.futures import ProcessPoolExecutor
  with ProcessPoolExecutor(max_workers=min(7,args.workers)) as pool:
   for value in pool.map(_job,CASES.items()):print(value['case']+': '+format(value['placement_xy_error_m']*1000,'.3f')+' mm socket error',flush=True)
 else:
  for item in CASES.items():_job(item)
 result=aggregate()
 if args.render:
  from panda.render import render_video
  render_video(ROOT)
 from panda.report import build_report
 build_report(ROOT);manifest()
 print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['passed_count']}/{result['check_count']} checks.",flush=True)
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
