"""Verify the released files using only the Python standard library."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
failed=[]
for name,digest in manifest.items():
 p=root/name
 if not p.is_file():failed.append((name,'missing'))
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:failed.append((name,'changed'))
print(f'{len(manifest)-len(failed)}/{len(manifest)} release files verified')
for name,status in failed:print(status,name)
if failed:sys.exit(1)
