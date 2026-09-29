"""Accepted source mechanics and explicitly declared benchmark scope."""
from pathlib import Path
import json
import numpy as np
from cmg_io import load_cmg
from constrained_dynamics import ClosedDynamics
from actuation_maps import ActuationMaps
from extended_reference import ExtendedReference
from inverse_dynamics import SourceInverseDynamics
ROOT=Path(__file__).resolve().parent
PORTS=['p0','p1','p2','p3','p4','p5','q21','q23']
INDEPENDENT=['q23','q7','q4','q0','q1','q21','q22']
GRAVITY=np.array([0.,0.,-9.81])
GUARDS={'independent_rad':1.5,'passive_revolute_rad':3.,'passive_prismatic_m':1.8}
def context():
 c=load_cmg(ROOT/'data/accepted_cmg_v04.json');m=json.loads((ROOT/'data/accepted_mujoco_mapping.json').read_text())
 e=ClosedDynamics(c,ROOT/'data/accepted_mujoco_kinematic.xml',m);a=ActuationMaps(c,e.tree_ids);r=ExtendedReference(e,a,GUARDS)
 return c,m,e,a,r,SourceInverseDynamics(a)
def plain(x):
 if hasattr(x,'tolist'):return x.tolist()
 raise TypeError(type(x).__name__)
def save_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,default=plain,allow_nan=False)+'\n',encoding='utf-8')
def load_trace(name):
 """Decompress each array once; NpzFile itself does not cache repeated reads."""
 with np.load(ROOT/'results'/f'{name}.npz',allow_pickle=False) as archive:
  return {key:archive[key] for key in archive.files}
