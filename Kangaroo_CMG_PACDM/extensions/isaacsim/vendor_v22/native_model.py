"""Compile explicit CMG to native MuJoCo; universal = connect + scalar tendon.

The tendon is a mathematical angular constraint, not a physical cable or an
actuator. Its endpoints are perpendicular on the universal-joint manifold.
No mass or inertia is added by the constraint representation.
"""
import copy,os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from import_full_model import UP
from mechanism_backend import fmt


def export(cmg,path,floating=True,drive=False,visual=False,contact=False,timestep=.0001,solref=.001):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    root=ET.Element('mujoco',model='Kangaroo_published_cut_reconstruction')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false',fusestatic='false',meshdir=os.path.relpath(UP/'assets',path.parent),alignfree='false')
    opt=ET.SubElement(root,'option',timestep=str(timestep),gravity='0 0 -9.81',integrator='implicitfast',solver='Newton',jacobian='dense',iterations='100',tolerance='1e-12')
    ET.SubElement(opt,'flag',contact='enable' if contact else 'disable',limit='enable',frictionloss='disable',energy='enable')
    de=ET.SubElement(root,'default');ET.SubElement(de,'geom',contype='0',conaffinity='0');ET.SubElement(de,'site',size='.002',group='5')
    bodies={b['id']:b for b in cmg['bodies']};elements={};visuals={}
    if visual or contact:
        for side in ['left','right']:
            text=(UP/f'parts/kangaroo.{side}_leg.xml').read_text()
            for k in ['4','5']:text=text.replace(f'<!--body name="{side}_{k}_ankle_ball"',f'<body name="{side}_{k}_ankle_ball"')
            text=text.replace('</body-->','</body>')
            for b in ET.fromstring(text).iter('body'):visuals[b.get('name')]=list(b.findall('geom'))
        for b in ET.parse(UP/'kangaroo.xml').getroot().find('worldbody').iter('body'):visuals[b.get('name')]=list(b.findall('geom'))
        asset=ET.SubElement(root,'asset')
        for file in ['kangaroo.visual_assets.xml','kangaroo.collision_assets.xml']:
            for e in ET.parse(UP/'assets'/file).getroot():asset.append(copy.deepcopy(e))
        if visual:
            ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.13 .18 .25',rgb2='.025 .04 .07',width='512',height='3072')
            vi=ET.SubElement(root,'visual');ET.SubElement(vi,'global',offwidth='1600',offheight='1000')
    world=ET.SubElement(root,'worldbody')
    if visual:
        ET.SubElement(world,'light',pos='1 -2 3',dir='-.3 .5 -1',diffuse='.8 .8 .8')
        ET.SubElement(world,'light',pos='-1 2 2',dir='.3 -.5 -1',diffuse='.7 .7 .7',castshadow='false')
    if visual or contact:
        ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .1',pos='0 0 -1.03' if not floating else '0 0 0',rgba='.84 .87 .91 1',contype='1' if contact else '0',conaffinity='1' if contact else '0')
    def pose(t):
        t=np.array(t);return dict(pos=fmt(t[:3,3]),quat=fmt(Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]))
    def inertial(el,name):
        b=bodies[name];I=np.array(b['inertia_com_kg_m2']);eig,axes=np.linalg.eigh(I)
        if np.linalg.det(axes)<0:axes[:,0]*=-1
        ET.SubElement(el,'inertial',mass=str(b['mass_kg']),pos=fmt(b['com_m']),diaginertia=fmt(eig),quat=fmt(Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]))
        for geom in visuals.get(name,[]):
            isvisual=geom.get('class')=='visual';isfoot=geom.get('class')=='box_collision' and name.endswith('ankle_roll')
            if (visual and isvisual) or (contact and isfoot):
                gg=copy.deepcopy(geom);gg.attrib.pop('class',None)
                if isvisual:gg.set('type','mesh');gg.set('contype','0');gg.set('conaffinity','0');gg.set('group','2')
                else:gg.set('type','box');gg.set('contype','1');gg.set('conaffinity','1');gg.set('friction','.8 .005 .0001');gg.set('condim','3');gg.set('rgba','.2 .2 .2 0')
                el.append(gg)
    base=ET.SubElement(world,'body',name=cmg['root_body']);elements[cmg['root_body']]=base;inertial(base,cmg['root_body'])
    if floating:ET.SubElement(base,'freejoint',name='floating_base')
    def visit(parent):
        for j in sorted((j for j in cmg['joints'] if j['base_body']==parent),key=lambda j:j['id']):
            name=j['follower_body'];el=ET.SubElement(elements[parent],'body',name=name,**pose(j['T_BJ']));elements[name]=el;inertial(el,name)
            if j['type']!='fixed':
                ET.SubElement(el,'joint',name=j['id'],type='hinge' if j['type']=='revolute' else 'slide',axis=fmt(j['axis']),limited='true',range=fmt([j['limits']['lower'],j['limits']['upper']]),armature=str(cmg['armature'][j['id']] if drive else 0),damping=str(cmg['joint_dissipation'][j['id']]['damping'] if drive else 0),frictionloss='0')
            visit(name)
    visit(cmg['root_body']);eq=ET.SubElement(root,'equality');tend=ET.SubElement(root,'tendon');L=.1
    for c in cmg['closures']:
        for number,suffix in [(1,'a'),(2,'b')]:ET.SubElement(elements[c[f'body{number}']],'site',name=c['id']+'_'+suffix,pos=fmt(c[f'point{number}_m']))
        pars=dict(solref=f'{solref} 1',solimp='.9999 .9999 .001')
        ET.SubElement(eq,'connect',name=c['id'],site1=c['id']+'_a',site2=c['id']+'_b',**pars)
        if c['type']=='universal':
            for number,suffix,col in [(1,'ua',0),(2,'ub',1)]:
                pos=np.array(c[f'point{number}_m'])+L*np.array(c[f'frame{number}_R'])[:,col]
                ET.SubElement(elements[c[f'body{number}']],'site',name=c['id']+'_'+suffix,pos=fmt(pos))
            t=ET.SubElement(tend,'spatial',name=c['id']+'_orthogonality',group='5',width='.001',rgba='0 0 0 0')
            for suffix in ['ua','ub']:ET.SubElement(t,'site',site=c['id']+'_'+suffix)
            # MuJoCo subtracts the reference tendon length; at q=0 it is sqrt(2)*L.
            ET.SubElement(eq,'tendon',name=c['id']+'_angular',tendon1=c['id']+'_orthogonality',polycoef='0 0 0 0 0',**pars)
    act=ET.SubElement(root,'actuator')
    for a in cmg['actuators']:ET.SubElement(act,'motor',name=a['id'],joint=a['joint'],gear='1',ctrllimited='true',ctrlrange=fmt(a['force_bounds_N']),forcelimited='true',forcerange=fmt(a['force_bounds_N']))
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True);return path
