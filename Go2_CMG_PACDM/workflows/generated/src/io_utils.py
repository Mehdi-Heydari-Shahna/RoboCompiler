from __future__ import annotations
from pathlib import Path
import csv,hashlib,json,platform,sys,subprocess,os,importlib.metadata
import numpy as np
from threadpoolctl import threadpool_info


def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v)for v in x]
    return x

def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n',encoding='utf-8')

def save_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('',encoding='utf-8');return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:clean(v) for k,v in r.items()})

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def environment():
    cpu=platform.processor();source='platform.processor';lscpu=None
    if sys.platform.startswith('linux'):
        try:
            info=Path('/proc/cpuinfo').read_text()
            cpu=next(l.split(':',1)[1].strip()for l in info.splitlines()if l.startswith('model name'));source='/proc/cpuinfo (virtualized host-exposed identification)'
            lscpu=subprocess.run(['lscpu'],capture_output=True,text=True,timeout=3).stdout
        except (OSError,StopIteration,subprocess.SubprocessError):pass
    if sys.platform=='win32' and not cpu:
        try:
            cpu=subprocess.run(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],capture_output=True,text=True,timeout=10).stdout.strip();source='Windows CIM'
        except (OSError,subprocess.SubprocessError):pass
    versions={}
    for p in ('numpy','scipy','threadpoolctl'):
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]=None
    try:
        import pinocchio as pin
        versions['pinocchio_imported']=pin.__version__
    except ImportError:versions['pinocchio_imported']=None
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),
        processor_model=cpu or None,processor_source=source,visible_logical_cpus=os.cpu_count(),lscpu=lscpu,
        versions=versions,threadpools=threadpool_info(),note='Host-reported virtual CPU identification is not a dedicated bare-metal allocation. Timing repeats are descriptive.')


def timing_summary(rows,groups=('method',)):
    result=[];keys=sorted(set(tuple(r[k]for k in groups)for r in rows),key=str)
    for key in keys:
        a=[r for r in rows if tuple(r[k]for k in groups)==key];v=np.array([r['time_ms']for r in a])
        out=dict(zip(groups,key));out.update(attempts=len(a),accepted=sum(r['success']for r in a),
            median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),
            max_gap_m=max((r.get('max_gap_m',0)for r in a),default=0),
            median_evaluations=float(np.median([r.get('evaluations',0)for r in a])),
            median_solved_modules=float(np.median([r.get('solved_modules',0)for r in a])),
            fallback_count=sum(r.get('fallback',False)for r in a))
        result.append(out)
    return result
