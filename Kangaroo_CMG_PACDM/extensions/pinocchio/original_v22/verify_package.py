"""Standard-library integrity check; no NumPy, MuJoCo or Pinocchio required."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    failed=[]
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file():failed.append((name,'missing'));continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:failed.append((name,'SHA-256 mismatch'))
    report=json.loads((ROOT/'results/complete_validation_v22.json').read_text())
    if report['status']!='PASS_RECONSTRUCTED_MODEL' or not all(x['passed'] for x in report['gates']):failed.append(('validation','required gate failure'))
    for name,why in failed:print('FAIL',name,why)
    print(f"{'PASS' if not failed else 'FAIL'}: {len(manifest['files'])} file checks; {report['gates_passed']}/{report['gates_total']} recorded numerical gates")
    if not failed:print('All files are intact. This verifies recorded evidence; it does not rerun the simulations.')
    return not failed

if __name__=='__main__':sys.exit(0 if main() else 1)
