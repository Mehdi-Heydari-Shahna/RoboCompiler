"""Check imports and source asset paths without downloading or changing anything."""
from pathlib import Path
import importlib,sys
ROOT=Path(__file__).resolve().parent
failed=[]
for name in ['numpy','scipy','mujoco','pinocchio']:
    try:
        module=importlib.import_module(name);print(name,getattr(module,'__version__','available'))
    except Exception as e:failed.append(name);print('MISSING/FAILED',name,str(e))
for name in ['upstream/hucebot/kangaroo_mujoco/kangaroo.xml','data/whole_body_cmg.json','data/contact_reference.npz']:
    if not (ROOT/name).is_file():failed.append(name);print('MISSING',name)
if failed:
    print('Activate the robotics environment, or install requirements_validated.txt in a Python 3.12 environment.')
    sys.exit(1)
print('Core environment ready. Plotting/video additionally need requirements_visuals.txt and ffmpeg.')
