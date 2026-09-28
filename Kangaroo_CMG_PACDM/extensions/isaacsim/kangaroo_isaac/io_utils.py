"""Atomic results and strict native name mapping."""
from pathlib import Path
import hashlib,json,os


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(obj):
        import numpy as np
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,Path):return str(obj)
        raise TypeError(type(obj).__name__)
    text=json.dumps(value,indent=2,default=convert,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');os.replace(tmp,path)


def name_map(source,native):
    import numpy as np
    native=list(native)
    if len(native)!=len(set(native)) or set(source)!=set(native):
        raise RuntimeError(f'Native names differ. Missing={set(source)-set(native)}, extra={set(native)-set(source)}')
    return np.array([native.index(n) for n in source],dtype=np.int64)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
