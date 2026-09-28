"""Standard full tree URDF plus explicit RoboIR loop/drive extension.

URDF alone cannot encode closed loops. Standard links, joints, inertias and
transmissions remain readable by tree importers; the namespaced mechanism
extension is necessary to reconstruct this constrained model.
"""
from pathlib import Path
import json,copy,warnings
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from mechanism_backend import fmt
NS='https://roboir.example/schema/mechanism/0.16'
ET.register_namespace('roboir',NS)

def rpy(R):
    # At a gimbal lock, SciPy chooses an equivalent non-unique RPY tuple.
    # The reconstructed rotation is verified by the roundtrip gate.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Gimbal lock detected.*',category=UserWarning)
        return Rotation.from_matrix(R).as_euler('xyz')

def write(c,path):
    root=ET.Element('robot',name='Kangaroo_published_cut_reconstruction')
    for b in c['bodies']:
        el=ET.SubElement(root,'link',name=b['id']);ie=ET.SubElement(el,'inertial');ET.SubElement(ie,'origin',xyz=fmt(b['com_m']),rpy='0 0 0');ET.SubElement(ie,'mass',value=str(b['mass_kg']));I=np.asarray(b['inertia_com_kg_m2']);ET.SubElement(ie,'inertia',**{k:str(I[i,j]) for k,i,j in [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]})
    for j in c['joints']:
        e=ET.SubElement(root,'joint',name=j['id'],type=j['type']);ET.SubElement(e,'parent',link=j['base_body']);ET.SubElement(e,'child',link=j['follower_body']);T=np.array(j['T_BJ']);ET.SubElement(e,'origin',xyz=fmt(T[:3,3]),rpy=fmt(rpy(T[:3,:3])))
        if not np.allclose(j['T_FJ'],np.eye(4),atol=1e-15):raise ValueError('Nonidentity follower attachment unsupported by tree URDF')
        if j['type']!='fixed':
            ET.SubElement(e,'axis',xyz=fmt(j['axis']));ET.SubElement(e,'limit',lower=str(j['limits']['lower']),upper=str(j['limits']['upper']),effort='5000' if j['type']=='prismatic' else '0',velocity='1000');ET.SubElement(e,'dynamics',damping=str(c['joint_dissipation'][j['id']]['damping']),friction=str(c['joint_dissipation'][j['id']]['frictionloss']))
    for a in c['actuators']:
        e=ET.SubElement(root,'transmission',name=a['id']);ET.SubElement(e,'type').text='transmission_interface/SimpleTransmission';jj=ET.SubElement(e,'joint',name=a['joint']);ET.SubElement(jj,'hardwareInterface').text='EffortJointInterface';aa=ET.SubElement(e,'actuator',name=a['id']);ET.SubElement(aa,'mechanicalReduction').text='1'
    metadata={k:v for k,v in c.items() if k not in ['bodies','joints']};metadata['fixed_axes']={j['id']:j['axis'] for j in c['joints'] if j['type']=='fixed'}
    metadata['urdf_limits_note']='The velocity=1000 and passive effort=0 attributes are serialization placeholders, not validated hardware ratings. Source position and motor force limits are authoritative for this benchmark.'
    ET.SubElement(root,f'{{{NS}}}mechanism',encoding='json').text=json.dumps(metadata,separators=(',',':'))
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def read(path):
    root=ET.parse(path).getroot();c=json.loads(root.find(f'{{{NS}}}mechanism').text);fixed=c.pop('fixed_axes');c.pop('urdf_limits_note');c['bodies']=[];c['joints']=[]
    def vec(el,key):return np.fromstring(el.get(key),sep=' ')
    for b in root.findall('link'):
        ie=b.find('inertial');ir=ie.find('inertia');vals={k:float(v) for k,v in ir.attrib.items()};I=[[vals['ixx'],vals['ixy'],vals['ixz']],[vals['ixy'],vals['iyy'],vals['iyz']],[vals['ixz'],vals['iyz'],vals['izz']]];c['bodies'].append(dict(id=b.get('name'),kind='rigid_body',mass_kg=float(ie.find('mass').get('value')),com_m=vec(ie.find('origin'),'xyz').tolist(),inertia_com_kg_m2=I))
    for j in root.findall('joint'):
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',vec(j.find('origin'),'rpy')).as_matrix();T[:3,3]=vec(j.find('origin'),'xyz');name=j.get('name');kind=j.get('type');record=dict(id=name,base_body=j.find('parent').get('link'),follower_body=j.find('child').get('link'),type=kind,T_BJ=T.tolist(),T_FJ=np.eye(4).tolist(),axis=fixed[name] if kind=='fixed' else vec(j.find('axis'),'xyz').tolist())
        if kind!='fixed':record.update(coordinate_unit='m' if kind=='prismatic' else 'rad',limits={x:float(j.find('limit').get(x)) for x in ['lower','upper']})
        c['joints'].append(record)
    return c

if __name__=='__main__':
    from reconstructed_model import whole_body
    root=Path(__file__).resolve().parent;c=whole_body();write(c,root/'models/kangaroo_full.urdf');r=read(root/'models/kangaroo_full.urdf');(root/'data/urdf_rebuilt_cmg.json').write_text(json.dumps(r,indent=2)+'\n')
