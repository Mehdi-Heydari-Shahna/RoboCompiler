"""URDF visual/collision tree plus explicit closed-loop/drive metadata.

A generic URDF importer receives an open tree. Consumers must implement the
RoboIR extension to recover physical loop constraints and drive dynamics.
"""
from pathlib import Path
import json,os
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import numpy as np
from reconstructed_model import whole_body
from contact_task import make_model,DEFAULTS
from import_full_model import UP,transform
from urdf_roundtrip import write,read,NS,rpy
from mechanism_backend import fmt
ROOT=Path(__file__).resolve().parent

def build():
    c=whole_body();path=ROOT/'models/kangaroo_contact.urdf';write(c,path)
    _,mjpath=make_model('contact_visual',.000025,DEFAULTS,visual=True)
    mj=ET.parse(mjpath).getroot();urdf=ET.parse(path);root=urdf.getroot();links={x.get('name'):x for x in root.findall('link')}
    meshes={x.get('name'):x for x in mj.findall('asset/mesh')};nv=nc=0
    for body in mj.iter('body'):
        for geom in body.findall('geom'):
            if geom.get('type')=='mesh':tag='visual';nv+=1
            elif geom.get('type')=='box' and geom.get('contype')=='1':tag='collision';nc+=1
            else:continue
            element=ET.SubElement(links[body.get('name')],tag);T=transform(geom)
            ET.SubElement(element,'origin',xyz=fmt(T[:3,3]),rpy=fmt(rpy(T[:3,:3])))
            geometry=ET.SubElement(element,'geometry')
            if tag=='visual':
                mesh=meshes[geom.get('mesh')];file=UP/'assets'/mesh.get('file')
                ET.SubElement(geometry,'mesh',filename=Path(os.path.relpath(file,path.parent)).as_posix(),scale=mesh.get('scale','1 1 1'))
                material=ET.SubElement(element,'material',name='source_visual_'+str(nv));ET.SubElement(material,'color',rgba=geom.get('rgba','.6 .65 .7 1'))
            else:ET.SubElement(geometry,'box',size=fmt(2*np.fromstring(geom.get('size'),sep=' ')))
    metadata=dict(schema='roboir-contact-benchmark-22',scope='Assumed simulation drive/contact model, not hardware parameters',
                  actuators=[dict(a,time_constant_s=DEFAULTS['actuator_time_constant_s'],command_slew_N_s=DEFAULTS['command_slew_N_s']) for a in c['actuators']],
                  contact=dict(foot_boxes_from_source=True,friction=DEFAULTS['friction'],time_constant_s=DEFAULTS['contact_time_constant_s']),
                  model=dict(floating_pelvis=True,self_collision=False,dry_friction_enabled=False),
                  note='Standard URDF does not enforce loop cuts, contact solref, or finite actuator response; import and implement both RoboIR metadata blocks.')
    ET.SubElement(root,f'{{{NS}}}contact_benchmark',encoding='json').text=json.dumps(metadata,separators=(',',':'))
    ET.indent(root);urdf.write(path,encoding='utf-8',xml_declaration=True)
    rebuilt=read(path)
    model_ok=rebuilt['closures']==c['closures'] and rebuilt['actuators']==c['actuators'] and len(rebuilt['bodies'])==78
    asset_ok=all((path.parent/m.get('filename')).is_file() for m in root.findall('.//visual/geometry/mesh'))
    result=dict(status='PASS' if model_ok and asset_ok and nc==2 else 'FAIL',visual_elements=nv,foot_collision_elements=nc,all_mesh_paths_exist=asset_ok,roboir_model_roundtrip=model_ok)
    (ROOT/'results/contact_urdf_export.json').write_text(json.dumps(result,indent=2)+'\n');print(result);return result

if __name__=='__main__':build()
