"""Source mechanics with free undercarriage and explicit granular contact.
CAD visuals/inertias unchanged. Collision proxies are massless and named.
"""
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from benchmark_model import ROOT,PORTS
SECTION=np.array([[-2.13268,.45688],[-1.92654,.33542],[-1.72419,.41405],[-1.52350,.55482],[-1.39611,.83593],[-1.49614,1.10182],[-1.96733,1.44613]])
BUCKET_X=(-.55256,.33623)
SOIL={'radius_m':.09,'particle_density_kg_m3':1800.,'friction':.65,'rolling_friction_m':.008,
 'grid':[5,5,3],'origin_m':[6.05,-.42,.096],'description':'Coarse noncohesive spherical aggregate; uncalibrated soil surrogate with rolling resistance'}
def fmt(x):return ' '.join(format(float(v),'.17g') for v in np.ravel(x))
def quat(R):return fmt(np.roll(Rotation.from_matrix(R).as_quat(),1))
def export_scene(path,*,dt=.0005,soil=True,bucket_contact=True,free_base=True,friction=.65,density_scale=1.,seed=21,solver="Newton",iterations=100):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);root=ET.parse(ROOT/'assets/visual_actual.xml').getroot();root.set('model','Excavator_full_body_digging_v21')
 root.find('compiler').set('meshdir',os.path.relpath(ROOT/'assets/meshes',path.parent))
 opt=root.find('option');opt.attrib.update(timestep=str(dt),gravity='0 0 -9.81',integrator='implicitfast',solver=solver,jacobian='sparse',cone='elliptic',iterations=str(iterations),ls_iterations='30',tolerance='1e-10')
 opt.find('flag').attrib.update(contact='enable',energy='enable',autoreset='disable',warmstart='enable')
 for eq in root.find('equality'):eq.set('solref','.004 1');eq.set('solimp','.9999 .9999 .001 .5 2')
 world=root.find('worldbody');bodies={b.get('name'):b for b in root.iter('body')}
 for x in list(world):
  if x.tag=='geom':world.remove(x)
 if free_base:ET.SubElement(bodies['body_53'],'freejoint',name='floating_undercarriage')
 default=ET.SubElement(root,'default');ET.SubElement(default,'geom',solref='.008 1',solimp='.99 .99 .001 .5 2',friction=f'{friction} .003 .008',condim='6',margin='0',gap='0')
 ET.SubElement(world,'geom',name='ground',type='plane',pos='0 0 0',size='30 30 .1',rgba='.43 .47 .44 1',contype='4',conaffinity='11',friction='1.0 .003 .008')
 for i,x in enumerate([-1.255,.35]):ET.SubElement(bodies['body_53'],'geom',name=f'track_collision_{i}',type='box',pos=fmt([x,2.1034,.22]),size='.31 1.62 .22',contype='8',conaffinity='5',group='3',rgba='.2 .2 .2 .25',mass='0',condim='3',friction='1.0 .003 .008')
 for bid,b in bodies.items():
  if bid in ['body_53','body_56']:continue
  cad=b.find("geom[@name='cad_"+bid+"']")
  if cad is not None:ET.SubElement(b,'geom',name='collision_'+bid,type='mesh',mesh=cad.get('mesh'),contype='8',conaffinity='5',group='3',rgba='.2 .3 .4 .15',mass='0',condim='3')
 bc=bodies['body_56'];asset=root.find('asset');common=dict(contype='2' if bucket_contact else '0',conaffinity='5' if bucket_contact else '0',group='3',rgba='.9 .2 .1 .3',mass='0',priority='1',friction='.25 .001 .002')
 width=BUCKET_X[1]-BUCKET_X[0];xc=sum(BUCKET_X)/2
 for k,(a,b) in enumerate(zip(SECTION[:-1],SECTION[1:])):
  length=np.linalg.norm(b-a);v=np.r_[0,(b-a)/length];R=np.column_stack(([1,0,0],v,np.cross([1,0,0],v)))
  ET.SubElement(bc,'geom',name=f'bucket_panel_{k}',type='box',pos=fmt(np.r_[xc,(a+b)/2]),quat=quat(R),size=fmt([width/2+.0175,length/2+.008,.0175]),**common)
 for i,x in enumerate(BUCKET_X):
  verts=np.array([[xx,y,z] for xx in [x-.0175,x+.0175] for y,z in SECTION]);mesh=f'bucket_side_proxy_{i}'
  ET.SubElement(asset,'mesh',name=mesh,vertex=fmt(verts));ET.SubElement(bc,'geom',name=f'bucket_side_{i}',type='mesh',mesh=mesh,**common)
 actuator=ET.SubElement(root,'actuator')
 for name in PORTS:ET.SubElement(actuator,'motor',name='effort_'+name,joint=name,gear='1',ctrllimited='false',forcelimited='false')
 rng=np.random.default_rng(seed);positions=[]
 if soil:
  nx,ny,nz=SOIL['grid'];rad=SOIL['radius_m'];origin=np.array(SOIL['origin_m'])
  for k in range(nz):
   for i in range(nx-k):
    for j in range(ny-k):
     # AB close packing; upper layers smaller. No particle receives a prescribed trajectory.
     p=origin+np.array([2*rad*(i+k//2)+rad*((j+k//2)%2)+rad*(k%2),np.sqrt(3)*rad*(j+k//2)+rad/np.sqrt(3)*(k%2),np.sqrt(8/3)*rad*k])*1.003
     p[:2]+=rng.uniform(-.0001,.0001,2);n=len(positions);positions.append(p);body=ET.SubElement(world,'body',name=f'grain_{n:03d}',pos=fmt(p));ET.SubElement(body,'freejoint',name=f'grain_joint_{n:03d}')
     mass=4*np.pi*rad**3/3*SOIL['particle_density_kg_m3']*density_scale;ET.SubElement(body,'inertial',pos='0 0 0',mass=str(mass),diaginertia=fmt([.4*mass*rad**2]*3))
     shade=.88+.14*rng.random();ET.SubElement(body,'geom',name=f'grain_geom_{n:03d}',type='sphere',size=str(rad),rgba=fmt(np.r_[shade*np.array([.53,.35,.19]),1]),contype='1',conaffinity='15',friction=f'{friction} .003 .008',condim='6')
 visual=root.find('visual');visual.find('global').set('offwidth','1440');visual.find('global').set('offheight','900');visual.find('quality').set('shadowsize','2048')
 root.find('statistic').set('center','3.8 0 1.2');root.find('statistic').set('extent','10');ET.indent(root,space='  ');ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
 return dict(soil=SOIL,particle_count=len(positions),particle_initial_positions_m=positions,free_base=free_base,bucket_contact=bucket_contact,friction=friction,density_scale=density_scale,collision_semantics='CAD-derived bucket cavity panels, two track patches and other-body convex proxies',original_inertias_modified=False,track_belt_dynamics=False,cohesion=False,soil_calibrated=False)
def bucket_inside(points_local,margin=.02):
 p=np.asarray(points_local);eq=ConvexHull(SECTION).equations
 return ((p[:,0]>BUCKET_X[0]+margin)&(p[:,0]<BUCKET_X[1]-margin)&np.all(p[:,1:]@eq[:,:2].T+eq[:,2]<-margin,axis=1))
