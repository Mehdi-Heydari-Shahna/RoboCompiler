"""Accepted source mechanics and explicitly declared benchmark scope."""
from pathlib import Path
import json
import sys
import numpy as np
from cmg_io import load_cmg
from constrained_dynamics import ClosedDynamics
from actuation_maps import ActuationMaps
from extended_reference import ExtendedReference
from inverse_dynamics import SourceInverseDynamics
ROOT=Path(__file__).resolve().parent
# The excavator model is supplied at run time (see Excavator_CMG_PACDM/model_access/excavator_model.py).
_ACCESS=next(p/'model_access' for p in ROOT.parents if (p/'model_access'/'excavator_model.py').is_file())
if str(_ACCESS) not in sys.path:sys.path.append(str(_ACCESS))
import excavator_model  # noqa: E402
PORTS=['p0','p1','p2','p3','p4','p5','q21','q23']
INDEPENDENT=['q23','q7','q4','q0','q1','q21','q22']
GRAVITY=np.array([0.,0.,-9.81])
GUARDS={'independent_rad':1.5,'passive_revolute_rad':3.,'passive_prismatic_m':1.8}
def context():
 model=excavator_model.mirror
 c=load_cmg(model(ROOT/'data/accepted_cmg_v04.json'));m=json.loads(model(ROOT/'data/accepted_mujoco_mapping.json').read_text())
 e=ClosedDynamics(c,model(ROOT/'data/accepted_mujoco_kinematic.xml'),m);a=ActuationMaps(c,e.tree_ids);r=ExtendedReference(e,a,GUARDS)
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
