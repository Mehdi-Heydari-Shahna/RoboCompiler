"""Import the pinned HuCeBot leg model from XML into a CMG.

Each connect constraint's second attachment is computed from source tree
kinematics at zero joint coordinates, following the MJCF convention.
Quaternions and joint axes are normalized. Geometry, mass, and rank
tolerances remain explicit source-model quantities."""
from pathlib import Path
import json,hashlib,copy
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from source_dynamics import SourceDynamics

ROOT=Path(__file__).resolve().parent
UP=ROOT/'upstream/hucebot/kangaroo_mujoco'
COMMIT='c020b68f3690930f5228d9f301b59ad9ab405e9a'

def numbers(el,key,default):return np.fromstring(el.get(key,default),sep=' ')

def rotation(el):
    if any(k in el.attrib for k in ['euler','axisangle','xyaxes','zaxis']):
        raise ValueError('Importer supports quaternion orientation only')
    quat=numbers(el,'quat','1 0 0 0');quat=quat/np.linalg.norm(quat)
    return Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()

def transform(el):
    T=np.eye(4);T[:3,:3]=rotation(el);T[:3,3]=numbers(el,'pos','0 0 0');return T

def build():
    cmg={'schema':'roboir.cmg.mechanism/0.16','root_body':'base_link',
         'coordinate_ids':[],'independent_ids':[f'leg_left_{x}_joint' for x in ['1','2','3','femour','4','5']],
         'bodies':[{'id':'base_link','kind':'reference_frame','mass_kg':0.,'com_m':[0,0,0],
                    'inertia_com_kg_m2':np.zeros((3,3)).tolist()}],
         'joints':[],'actuators':[],'closures':[],
         'source':{'repository':'https://github.com/hucebot/mujoco_kangaroo_sim2sim',
                   'commit':COMMIT,'model':'first-generation left leg; fixed base',
                   'geometry_modifications':[]},'joint_dissipation':{},'armature':{}}
    def visit(el,parent):
        name=el.attrib['name'];ie=el.find('inertial')
        if ie is None:raise ValueError('Missing explicit source inertia '+name)
        R=rotation(ie)
        if 'diaginertia' in ie.attrib:I=R@np.diag(numbers(ie,'diaginertia',''))@R.T
        else:
            xx,yy,zz,xy,xz,yz=numbers(ie,'fullinertia','')
            I=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
        cmg['bodies'].append({'id':name,'kind':'rigid_body','mass_kg':float(ie.attrib['mass']),
                             'com_m':numbers(ie,'pos','0 0 0').tolist(),'inertia_com_kg_m2':I.tolist()})
        js=el.findall('joint')
        if len(js)!=1:raise ValueError('Expected one source scalar joint per leg body')
        j=js[0];jid=j.attrib['name'];axis=numbers(j,'axis','0 0 1');axis/=np.linalg.norm(axis)
        if np.linalg.norm(numbers(j,'pos','0 0 0'))>0:raise ValueError('Nonzero joint origin needs explicit adapter')
        kind={'hinge':'revolute','slide':'prismatic'}[j.get('type','hinge')]
        low,high=numbers(j,'range','');cmg['coordinate_ids'].append(jid)
        cmg['joints'].append({'id':jid,'base_body':parent,'follower_body':name,'type':kind,
                              'coordinate_unit':'m' if kind=='prismatic' else 'rad','axis':axis.tolist(),
                              'T_BJ':transform(el).tolist(),'T_FJ':np.eye(4).tolist(),
                              'limits':{'lower':float(low),'upper':float(high)}})
        cmg['armature'][jid]=float(j.get('armature',0))
        cmg['joint_dissipation'][jid]={'damping':float(j.get('damping',3)),
                                      'frictionloss':float(j.get('frictionloss',.1))}
        for child in el.findall('body'):visit(child,name)
    for el in ET.parse(UP/'parts/kangaroo.left_leg.xml').getroot().findall('body'):visit(el,'base_link')
    source=SourceDynamics(cmg,cmg['coordinate_ids']);poses=source.evaluate(np.zeros(36),np.zeros(36),[0,0,0])['poses']
    for c in ET.parse(UP/'constraints/kangaroo.left_leg.equality_constraints.xml').getroot().find('equality'):
        if c.tag!='connect':raise ValueError('Unsupported source equality')
        a=c.attrib['body1'];b=c.attrib['body2'];p=numbers(c,'anchor','0 0 0')
        world=poses[a][:3,:3]@p+poses[a][:3,3]
        pb=poses[b][:3,:3].T@(world-poses[b][:3,3])
        cmg['closures'].append({'id':c.attrib['name'],'type':'point_coincidence','body1':a,'body2':b,
                                'point1_m':p.tolist(),'point2_m':pb.tolist()})
    for a in ET.parse(UP/'controllers/kangaroo.left_leg.motor_controllers.xml').getroot().find('actuator'):
        low,high=numbers(a,'ctrlrange','')
        cmg['actuators'].append({'id':a.attrib['name'],'joint':a.attrib['joint'],'unit':'N',
                                 'gear':1.,'force_bounds_N':[float(low),float(high)]})
    cmg['source']['files']={p.relative_to(UP).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(UP.rglob('*.xml'))}
    return cmg

def native_xml(destination,visual=False,timestep=.0005,joint_limits=False):
    """Independent native model assembled directly from original source XML.

    Fixed root, one full leg, six source force motors. No contacts or
    dry friction in this local benchmark. Limits are configurable so the
    open-tree component checks use the source drive and inertial records.
    Original viscous damping and motor
    armatures retained; closure solver stiffness explicitly specified.
    """
    dest=Path(destination);root=ET.Element('mujoco',model='Kangaroo_full_left_actuators')
    ET.SubElement(root,'compiler',angle='radian',meshdir=str((UP/'assets').resolve()),fusestatic='false')
    opt=ET.SubElement(root,'option',timestep=str(timestep),integrator='implicitfast',solver='Newton',
                      iterations='100',tolerance='1e-12',jacobian='dense',gravity='0 0 -9.81')
    ET.SubElement(opt,'flag',contact='disable',limit='enable' if joint_limits else 'disable',frictionloss='disable',energy='enable')
    defaults=copy.deepcopy(ET.parse(UP/'common.xml').getroot().find('default'));root.append(defaults)
    ET.SubElement(root,'visual');ET.SubElement(root.find('visual'),'global',offwidth='1280',offheight='960')
    if visual:
        asset=ET.SubElement(root,'asset')
        ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.12 .16 .23',rgb2='.035 .05 .08',width='512',height='3072')
        for name in ['kangaroo.visual_assets.xml','kangaroo.collision_assets.xml']:
            for child in ET.parse(UP/'assets'/name).getroot():asset.append(copy.deepcopy(child))
    world=ET.SubElement(root,'worldbody')
    if visual:
        ET.SubElement(world,'light',pos='1 -2 3',dir='-0.3 0.5 -1',diffuse='.8 .8 .8')
        ET.SubElement(world,'light',pos='-1 2 2',dir='.3 -.5 -1',diffuse='.8 .8 .8',castshadow='false')
        ET.SubElement(world,'geom',type='plane',size='2 2 .1',pos='0 0 -1.08',rgba='.88 .9 .93 1',contype='0',conaffinity='0')
    base=ET.SubElement(world,'body',name='base_link')
    if visual:ET.SubElement(base,'geom',type='mesh',mesh='base_link',rgba='.2 .2 .25 1',contype='0',conaffinity='0')
    for el in ET.parse(UP/'parts/kangaroo.left_leg.xml').getroot().findall('body'):
        el=copy.deepcopy(el)
        if not visual:
            for parent in el.iter():
                for geom in list(parent.findall('geom')):parent.remove(geom)
        base.append(el)
    equality=copy.deepcopy(ET.parse(UP/'constraints/kangaroo.left_leg.equality_constraints.xml').getroot().find('equality'))
    for el in equality:el.set('solref','.002 1');el.set('solimp','.9999 .9999 .001')
    root.append(equality)
    root.append(copy.deepcopy(ET.parse(UP/'controllers/kangaroo.left_leg.motor_controllers.xml').getroot().find('actuator')))
    ET.indent(root);ET.ElementTree(root).write(dest,encoding='utf-8',xml_declaration=True)
    return dest

if __name__=='__main__':
    (ROOT/'data/full_cmg.json').write_text(json.dumps(build(),indent=2)+'\n')
    print('Imported 36 tree coordinates, 14 point connects, six physical linear force ports.')
