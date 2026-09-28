"""Single-command full mechanics and contact-task reproduction."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,sys,hashlib,platform,time
ROOT=Path(__file__).resolve().parent

def assemble():
    names=['full_validation','contact_validation'];reports=[json.loads((ROOT/'results'/f'{n}.json').read_text()) for n in names]
    gates=[dict(g,suite=n+'/'+g.get('suite','')) for n,r in zip(names,reports) for g in r['gates']]
    exported=json.loads((ROOT/'results/contact_urdf_export.json').read_text())
    gates.append(dict(name='contact_urdf_visuals_collisions_and_roboir_roundtrip',suite='export',passed=exported['status']=='PASS'))
    result=dict(status='PASS_RECONSTRUCTED_MODEL' if all(g['passed'] for g in gates) else 'FAIL',gates=gates,gates_passed=sum(g['passed'] for g in gates),gates_total=len(gates),
        scope='Kangaroo published-cut full prototype reconstruction with native contact, finite mechanical force response, and explicit numerical limitations.',
        environment=dict(python=platform.python_version(),platform=platform.platform()),
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))})
    (ROOT/'results/complete_validation_v22.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],f"{result['gates_passed']}/{result['gates_total']} gates",flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--reuse-results',action='store_true',help='Recheck previously generated results without rerunning trajectories');p.add_argument('--visuals',action='store_true');args=p.parse_args()
    # Create output directories before model export and simulation.
    for folder in ['data','models','results','videos']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    if args.reuse_results and not (ROOT/'results/full_validation.json').is_file():
        p.error('No generated results found. Run python run_kangaroo_v22.py first.')
    if not args.reuse_results:
        from run_kangaroo_v20 import run as baseline
        from contact_reference import build_reference
        if not baseline():raise RuntimeError('Baseline mechanics failed')
        build_reference()
    from validate_contact_task import validate
    validate(args.reuse_results)
    from export_contact_urdf import build as export_urdf
    export_urdf()
    result=assemble()
    if args.visuals:
        from make_contact_report import build
        from render_contact_task import render
        build();render()
    return result['status']=='PASS_RECONSTRUCTED_MODEL'

if __name__=='__main__':sys.exit(0 if main() else 1)
