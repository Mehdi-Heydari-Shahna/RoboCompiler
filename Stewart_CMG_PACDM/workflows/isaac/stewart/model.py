"""Explicit six-UPS CMG and matching MuJoCo lowering; SI units throughout.

This is an original idealized benchmark, not an identified commercial machine.
Zero-inertia platform chart frames are computational frames, collapsed into
one MuJoCo multi-joint body. No CAD dimensions or inertias are silently inferred.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def transform(R=None, p=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if p is not None: T[:3, 3] = p
    return T


def rotation(pose):
    return Rotation.from_euler('ZYX', np.asarray(pose)[3:6]).as_matrix()


def make_cmg(payload_mass=8.0):
    rb, rp, height = .55, .35, .60
    ba = np.deg2rad([-10, 10, 110, 130, 230, 250])
    pa = np.deg2rad([-50, 50, 70, 170, 190, 290])
    base = np.c_[rb*np.cos(ba), rb*np.sin(ba), np.zeros(6)]
    top = np.c_[rp*np.cos(pa), rp*np.sin(pa), np.zeros(6)]
    bodies, joints, ids, closures, bases = [], [], [], [], []
    def body(name, mass=0., com=(0.,0.,0.), inertia=(0.,0.,0.)):
        I=np.asarray(inertia, float)
        if I.ndim==1: I=np.diag(I)
        bodies.append(dict(id=name, mass_kg=float(mass), com_m=list(com), inertia_kg_m2=I.tolist()))
    def joint(name, kind, parent, child, axis, T=None, limits=(-.5,.5)):
        joints.append(dict(id=name,type=kind,base_body=parent,follower_body=child,
                           axis=list(axis),T_BJ=(np.eye(4) if T is None else T).tolist(),
                           T_FJ=np.eye(4).tolist(),limits=dict(lower=limits[0],upper=limits[1])))
        if kind!='fixed': ids.append(name)
    body('world')
    parent='world'
    for name,kind,axis,limits in [('x','prismatic',[1,0,0],(-.16,.16)),
                                 ('y','prismatic',[0,1,0],(-.16,.16)),
                                 ('z','prismatic',[0,0,1],(.40,.85)),
                                 ('yaw','revolute',[0,0,1],(-.50,.50)),
                                 ('pitch','revolute',[0,1,0],(-.40,.40)),
                                 ('roll','revolute',[1,0,0],(-.40,.40))]:
        child='platform' if name=='roll' else 'chart_'+name
        if child=='platform': body(child,12.,(0,0,0),(.32,.32,.62))
        else: body(child)
        joint('pose_'+name,kind,parent,child,axis,limits=limits)
        parent=child
    body('payload',payload_mass,(0,0,.11),np.array([.018,.018,.013])*payload_mass)
    joint('payload_mount','fixed','platform','payload',[0,0,1],transform(p=[0,0,.055]))
    for i in range(6):
        d=np.array([0,0,height])+top[i]-base[i]; u=d/np.linalg.norm(d)
        z=np.array([0.,0.,1.]); v=np.cross(z,u); angle=np.arccos(z@u)
        R0=Rotation.from_rotvec(v/np.linalg.norm(v)*angle).as_matrix()
        bases.append(R0.tolist())
        yoke,barrel,rod=[f'leg_{i}_{s}' for s in ['yoke','barrel','rod']]
        body(yoke,.20,(0,0,0),(.0006,.0006,.0006))
        body(barrel,1.,(0,0,.16),(.010,.010,.0010))
        body(rod,.45,(0,0,-.15),(.004,.004,.00035))
        joint(f'leg_{i}_u_x','revolute','world',yoke,[1,0,0],transform(R0,base[i]))
        joint(f'leg_{i}_u_y','revolute',yoke,barrel,[0,1,0])
        joint(f'leg_{i}_length','prismatic',barrel,rod,[0,0,1],limits=(.50,.91))
        closures.append(dict(id=f'cut_{i}',body1=rod,body2='platform',point1_m=[0,0,0],point2_m=top[i].tolist()))
    return dict(schema='cmg.stewart.point-closures/1.0',id='stewart_6ups',root_body='world',
                units='SI',gravity_m_s2=[0,0,-9.81],coordinate_ids=ids,
                independent_ids=[f'leg_{i}_length' for i in range(6)],bodies=bodies,joints=joints,closures=closures,
                geometry=dict(base_anchors_m=base.tolist(),platform_anchors_m=top.tolist(),base_rotations=bases,
                              platform_radius_m=rp,nominal_pose=[0,0,height,0,0,0]),
                actuation=dict(type='prismatic_force',force_limit_N=900.,length_kp_N_m=24000.,length_kd_N_s_m=650.),
                scope='Original idealized rigid six-UPS benchmark; inertias are declared engineering parameters, not identified hardware. Fixed payload, no flexibility, collisions or actuator electrical/hydraulic dynamics.')


def inverse_seed(cmg, pose):
    """Geometric IK used to specify commanded lengths, not the forward solver."""
    p=np.asarray(pose); R=rotation(p); geom=cmg['geometry']; q=list(p)
    for b,a,R0 in zip(geom['base_anchors_m'],geom['platform_anchors_m'],geom['base_rotations']):
        d=p[:3]+R@a-b; length=np.linalg.norm(d); u=np.asarray(R0).T@(d/length)
        q.extend([np.arctan2(-u[1],u[2]),np.arcsin(np.clip(u[0],-1,1)),length])
    return np.asarray(q)


def _fmt(x): return ' '.join(f'{float(v):.14g}' for v in np.asarray(x).ravel())


def compile_mujoco(cmg, path, timestep=.002):
    root=ET.Element('mujoco',model='CMG PACDM Stewart six UPS')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false',fusestatic='false')
    ET.SubElement(root,'option',timestep=str(timestep),gravity=_fmt(cmg['gravity_m_s2']),
                  integrator='implicitfast',solver='Newton',iterations='100',tolerance='1e-12',jacobian='dense')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1440',offheight='900')
    ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
    ET.SubElement(visual,'map',znear='.01',zfar='20')
    asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',name='floor_grid',type='2d',builtin='checker',width='512',height='512',rgb1='.10 .13 .18',rgb2='.13 .17 .22')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_grid',texrepeat='6 6',reflectance='.12')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',contype='0',conaffinity='0',condim='3',rgba='.2 .25 .3 1')
    ET.SubElement(default,'joint',damping='0',armature='0',limited='true')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='1 -2 3',dir='-0.2 0.5 -1',diffuse='.8 .9 1',castshadow='true')
    ET.SubElement(world,'light',pos='-2 1 2',dir='.5 -.2 -1',diffuse='.5 .7 1',castshadow='false')
    ET.SubElement(world,'geom',type='plane',size='3 3 .05',pos='0 0 -.115',material='floor_mat')
    ET.SubElement(world,'geom',name='base_plate',type='cylinder',size='.64 .045',pos='0 0 -.065',rgba='.10 .14 .19 1')
    for a in np.linspace(0,2*np.pi,37)[:-1]:
        ET.SubElement(world,'geom',type='sphere',size='.006',pos=_fmt([.607*np.cos(a),.607*np.sin(a),-.017]),rgba='.05 .85 .95 1')
    bodies={b['id']:b for b in cmg['bodies']}; js={j['id']:j for j in cmg['joints']}
    def inertial(el,name):
        b=bodies[name]; I=np.asarray(b['inertia_kg_m2'])
        ET.SubElement(el,'inertial',mass=str(b['mass_kg']),pos=_fmt(b['com_m']),fullinertia=_fmt([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
    def mjjoint(el,name):
        j=js[name]; ET.SubElement(el,'joint',name=name,type='hinge' if j['type']=='revolute' else 'slide',
                                  axis=_fmt(j['axis']),range=_fmt([j['limits']['lower'],j['limits']['upper']]))
    platform=ET.SubElement(world,'body',name='platform')
    for k in cmg['coordinate_ids'][:6]: mjjoint(platform,k)
    inertial(platform,'platform')
    ET.SubElement(platform,'geom',type='cylinder',size='.39 .025',rgba='.15 .26 .35 1')
    ET.SubElement(platform,'geom',type='cylinder',size='.355 .003',pos='0 0 .028',rgba='.35 .55 .64 1')
    for i,a in enumerate(cmg['geometry']['platform_anchors_m']):
        ET.SubElement(platform,'geom',type='sphere',size='.025',pos=_fmt(a),rgba='.85 .55 .13 1')
        ET.SubElement(platform,'site',name=f'top_{i}',pos=_fmt(a),size='.007',rgba='1 .6 .1 1')
    payload=ET.SubElement(platform,'body',name='payload',pos='0 0 .055')
    inertial(payload,'payload')
    ET.SubElement(payload,'geom',type='box',size='.115 .105 .10',pos='0 0 .10',rgba='.85 .9 .94 1')
    ET.SubElement(payload,'geom',type='box',size='.117 .025 .103',pos='0 0 .10',rgba='.06 .13 .19 1')
    ET.SubElement(payload,'geom',type='cylinder',size='.047 .065',pos='0 0 .25',rgba='.14 .17 .2 1')
    ET.SubElement(payload,'geom',type='sphere',size='.018',pos='0 0 .328',rgba='.04 1 .83 1')
    ET.SubElement(payload,'site',name='probe',pos='0 0 .328',size='.009',rgba='.04 1 .83 1')
    for i in range(6):
        R0=cmg['geometry']['base_rotations'][i]; quat=Rotation.from_matrix(R0).as_quat()[[3,0,1,2]]
        yoke=ET.SubElement(world,'body',name=f'leg_{i}_yoke',pos=_fmt(cmg['geometry']['base_anchors_m'][i]),quat=_fmt(quat))
        mjjoint(yoke,f'leg_{i}_u_x'); inertial(yoke,f'leg_{i}_yoke')
        ET.SubElement(yoke,'geom',type='sphere',size='.037',rgba='.27 .35 .42 1')
        barrel=ET.SubElement(yoke,'body',name=f'leg_{i}_barrel')
        mjjoint(barrel,f'leg_{i}_u_y'); inertial(barrel,f'leg_{i}_barrel')
        ET.SubElement(barrel,'geom',type='cylinder',size='.025 .245',pos='0 0 .245',rgba='.07 .35 .46 1')
        ET.SubElement(barrel,'geom',type='cylinder',size='.029 .020',pos='0 0 .485',rgba='.88 .59 .15 1')
        rod=ET.SubElement(barrel,'body',name=f'leg_{i}_rod')
        mjjoint(rod,f'leg_{i}_length'); inertial(rod,f'leg_{i}_rod')
        ET.SubElement(rod,'geom',type='cylinder',size='.013 .225',pos='0 0 -.225',rgba='.75 .8 .84 1')
        ET.SubElement(rod,'geom',type='sphere',size='.024',rgba='.85 .55 .13 1')
        ET.SubElement(rod,'site',name=f'tip_{i}',pos='0 0 0',size='.007')
    equality=ET.SubElement(root,'equality')
    for i in range(6):
        # Explicit sites preserve both authored local anchors, independent of qpos0.
        ET.SubElement(equality,'connect',name=f'closure_{i}',site1=f'tip_{i}',site2=f'top_{i}',
                      solref='.004 1',solimp='.9999 .9999 .001')
    actuator=ET.SubElement(root,'actuator'); limit=cmg['actuation']['force_limit_N']
    for name in cmg['independent_ids']:
        ET.SubElement(actuator,'motor',name='motor_'+name,joint=name,gear='1',ctrllimited='true',ctrlrange=_fmt([-limit,limit]))
    key=ET.SubElement(root,'keyframe')
    ET.SubElement(key,'key',name='home',qpos=_fmt(inverse_seed(cmg,cmg['geometry']['nominal_pose'])))
    ET.indent(root); path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(path,encoding='unicode',xml_declaration=True)
    return path


def save_cmg(cmg,path):
    Path(path).write_text(json.dumps(cmg,indent=2)+'\n')
