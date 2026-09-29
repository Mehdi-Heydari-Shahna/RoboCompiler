#!/usr/bin/env python
"""Panda CMG/PACDM: clean-process Isaac Sim 6.1 benchmark launcher.

The parent and --preflight use only the standard library. Never import pxr,
Pinocchio, MuJoCo, or simulator extensions before creating SimulationApp.
"""
from pathlib import Path
import argparse,datetime,faulthandler,hashlib,importlib.metadata as metadata
import json,os,platform,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
# Standard-library-only helper (no NumPy/USD import): retries os.replace when
# OneDrive/antivirus briefly locks a file instead of killing the run.
from cmg_isaac.fileio import write_json

CHILD_EXIT='child_exit.json'


def environment():
    versions={}
    for name in ('isaacsim','isaacsim-core','numpy','scipy','usd-core','openusd','pin','pinocchio','mujoco'):
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    issues=[];warnings=[]
    if sys.version_info[:2]!=(3,12):issues.append('This adapter targets Isaac Sim 6.1 with Python 3.12; current interpreter is '+sys.version.split()[0])
    version=versions['isaacsim'] or versions['isaacsim-core']
    if not version or not version.startswith('6.1.'):
        issues.append('Isaac Sim 6.1.x package metadata was not found in this interpreter. Activate the Isaac Sim 6.1 environment.')
    for name in ('numpy','scipy'):
        if not versions[name]:issues.append('Required package missing: '+name)
    for name in ('usd-core','openusd'):
        if versions[name]:warnings.append(name+' is installed separately. A different USD build can conflict with Isaac. No packages will be removed automatically.')
    if 'onedrive' in str(ROOT).lower():
        warnings.append('The package is inside a OneDrive folder. Sync can lock result files; writes are retried, '
                        'but a local folder such as C:\\panda_runs (use --output) is faster and safer.')
    gpu=None
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=15,check=False)
        gpu=result.stdout.strip() or result.stderr.strip()
        if result.returncode:issues.append('nvidia-smi failed; inspect GPU/driver setup before native execution.')
    except (OSError,subprocess.TimeoutExpired) as exc:
        issues.append('nvidia-smi unavailable: '+str(exc))
    return dict(executable=sys.executable,python=sys.version,platform=platform.platform(),versions=versions,
                gpu=gpu,issues=issues,warnings=warnings,passed=not issues,
                note='Metadata/driver visibility only. This is NOT a native engine or task validation.')


def checkpoint_factory(output):
    def checkpoint(label,details=None):
        item=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=label,details=details)
        # A progress marker must never abort a physics run (see fileio.py).
        write_json(output/'last_checkpoint.json',item,required=False)
        print('[PANDA] '+label+((' '+json.dumps(details)) if details else ''),flush=True)
    return checkpoint


def apply_overrides(case,args):
    """Diagnostic overrides are recorded in inputs.json; suites reject them."""
    if args.dt is not None:case['dt']=args.dt
    if args.solver is not None:case['solver']=args.solver
    if args.position_iterations is not None:case['position_iterations']=args.position_iterations
    if args.joint_offsets is not None:case['joint_zero_offsets']=args.joint_offsets=='on'
    if case['dt']<=0 or case['dt']>.001:raise ValueError('Use a positive dt <= 0.001 s; the source gains are stiff.')
    if not 1<=int(case['position_iterations'])<=255:raise ValueError('position_iterations must be in 1..255')
    return case


def child(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    faulthandler.enable(all_threads=True)
    checkpoint=checkpoint_factory(out)
    app=None;native=None;code=2;status='EXECUTION_FAILED'
    try:
        env=environment();write_json(out/'environment.json',env)
        if env['issues']:raise RuntimeError('; '.join(env['issues']))
        if any(k=='pxr' or k.startswith('pxr.') for k in sys.modules):
            raise RuntimeError('USD/pxr was imported before SimulationApp. Start a fresh Python process, not a notebook or Script Editor.')
        checkpoint('Creating SimulationApp')
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':not args.gui,'width':1280,'height':800,
                           'renderer':'RaytracedLighting','anti_aliasing':0,'multi_gpu':False,
                           'limit_cpu_threads':8,'sync_loads':True})
        checkpoint('SimulationApp created')
        # All numerical/native modules intentionally imported after app setup.
        from cmg_isaac.control import Reference
        from cmg_isaac.cases import get_case
        from cmg_isaac.native import NativePanda
        from cmg_isaac.simulation import run_mission
        from pxr import Usd
        write_json(out/'runtime_paths.json',dict(usd_version=list(Usd.GetVersion()),
            pxr_file=getattr(sys.modules.get('pxr'),'__file__',None),python_executable=sys.executable))
        cmg=json.loads((ROOT/'data/panda_cmg.json').read_text())
        ref=Reference(ROOT,args.reference)
        case=apply_overrides(get_case(args.mode,args.case),args)
        write_json(out/'inputs.json',dict(mode=args.mode,case=args.case,configuration=case,
            reference_path=str(ref.path),reference_sha256=hashlib.sha256(ref.path.read_bytes()).hexdigest(),
            cmg_sha256=hashlib.sha256((ROOT/'data/panda_cmg.json').read_bytes()).hexdigest(),
            pacdm_sha256=hashlib.sha256((ROOT/'vendor/pacdm_original.py').read_bytes()).hexdigest(),
            reference_source='PACDM-generated route, not measured robot state'))
        native=NativePanda(app,ROOT,cmg,ref,args.mode,case,out,args.gui,checkpoint)
        checkpoint('Checking native mechanics')
        mechanics=native.mechanics();write_json(out/'mechanics.json',mechanics)
        if not mechanics['passed']:
            write_json(out/'result.json',dict(status='NATIVE_MECHANICS_FAIL',passed=False,data_valid=False,mechanics=mechanics))
            status='NATIVE_MECHANICS_FAIL'
            raise RuntimeError('Native mechanics/coupling check failed. Mission not started; inspect mechanics.json.')
        result=run_mission(native,cmg,ref,args.mode,args.case,case,out,mechanics,args.smoke)
        status=result['status'];checkpoint(status)
        print(json.dumps(result['summary'],indent=2),flush=True)
        if args.smoke:code=0 if result['smoke_passed'] else 1
        elif args.case=='no_feedforward':code=0 if result['data_valid'] else 1
        else:code=0 if result['passed'] else 1
    except BaseException as exc:
        trace=traceback.format_exc();print(trace,file=sys.stderr,flush=True)
        write_json(out/'failure.json',dict(passed=False,error=repr(exc),traceback=trace),required=False)
        if not (out/'result.json').exists():
            write_json(out/'result.json',dict(status='EXECUTION_FAILED',passed=False,data_valid=False,error=repr(exc)),required=False)
        code=2
    finally:
        if native is not None:
            try:
                import numpy as np
                native.effort(np.zeros(9));native.timeline.stop()
            except Exception:pass
        # Kit's fast shutdown ends the process with status 0 inside app.close(),
        # so the intended status is handed to the parent through this file.
        write_json(out/CHILD_EXIT,dict(code=code,status=status,
                   note='Intended child exit code; the OS exit code is overwritten by Kit fast shutdown.'),required=False)
        sys.stdout.flush();sys.stderr.flush()
        if app is not None:
            try:app.close()
            except Exception as exc:print('SimulationApp shutdown error: '+repr(exc),file=sys.stderr,flush=True)
    return code


def resolve_exit(process_code,folder,attempts=20):
    """Combine the OS exit code with the child's own status file."""
    intended=None
    for attempt in range(attempts):
        path=folder/CHILD_EXIT
        if not path.exists():break
        try:
            intended=int(json.loads(path.read_text(encoding='utf-8'))['code']);break
        except OSError:time.sleep(.1)                 # transient OneDrive/antivirus lock
        except (ValueError,KeyError,TypeError):break
    if process_code!=0:return process_code,intended    # native crash / access violation wins
    if intended is None:return 2,None                    # exited "OK" without reporting: not trusted
    return intended,intended


def run_process(args,mode,name,folder):
    folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'run_isaac.py'),'--_child','--mode',mode,'--case',name,'--output',str(folder)]
    if args.gui:command.append('--gui')
    if args.smoke:command.append('--smoke')
    if args.reference:command+=['--reference',str(Path(args.reference).resolve())]
    if args.dt is not None:command+=['--dt',str(args.dt)]
    if args.solver is not None:command+=['--solver',args.solver]
    if args.position_iterations is not None:command+=['--position-iterations',str(args.position_iterations)]
    if args.joint_offsets is not None:command+=['--joint-offsets',args.joint_offsets]
    env=os.environ.copy()
    # Preserve the activated interpreter and DLL search configuration.
    # Leave PATH, sys.path and installed packages unchanged.
    env.setdefault('PYTHONNOUSERSITE','1')
    env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('OMP_NUM_THREADS','1')
    write_json(folder/'launcher.json',dict(command=command,executable=sys.executable,status='STARTING'),required=False)
    with (folder/'console.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,encoding='utf-8',errors='replace',env=env,bufsize=1)
        try:
            for line in p.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            process_code=p.wait()
        except KeyboardInterrupt:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            process_code=p.returncode if p.returncode not in (None,0) else 130
    code,intended=resolve_exit(process_code,folder)
    record=dict(command=command,returncode=code,process_returncode=process_code,child_reported_code=intended,
                windows_status_hex=hex(process_code & 0xffffffff),
                status='PROCESS_EXITED_OK' if code==0 else 'PROCESS_FAILED' if process_code!=0 or intended is None else 'CHILD_REPORTED_FAILURE')
    write_json(folder/'launcher.json',record,required=False)
    if code!=0 and not (folder/'failure.json').exists() and (process_code!=0 or intended is None):
        write_json(folder/'failure.json',dict(passed=False,returncode=code,process_returncode=process_code,
             note='Native process exited before Python could report the failure. Include console.log and last_checkpoint.json.',
             windows_status_hex=hex(process_code & 0xffffffff)),required=False)
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true',help='Inspect active interpreter and packages without launching Isaac')
    p.add_argument('--rebuild-reference',action='store_true',help='Rerun the original CMG/PACDM compiler in this process; no Isaac launch')
    p.add_argument('--mode',choices=['contact','wrench'],default='contact')
    p.add_argument('--case',default='nominal')
    p.add_argument('--suite',choices=['contact','wrench','all'])
    p.add_argument('--smoke',action='store_true',help='Native mechanics + 0.25 s only; never labels a task PASS')
    p.add_argument('--gui',action='store_true',help='Display the native simulation; default headless')
    p.add_argument('--dt',type=float,help='Explicit diagnostic timestep override; must divide 0.01 s')
    p.add_argument('--solver',choices=['PGS','TGS'],help='Diagnostic solver override (declared default PGS)')
    p.add_argument('--position-iterations',type=int,help='Diagnostic position-iteration override (declared default 4)')
    p.add_argument('--joint-offsets',choices=['on','off'],help='Diagnostic override of float32 joint-zero offsets (declared default on)')
    p.add_argument('--reference',help='Optional reference.npz regenerated using --rebuild-reference')
    p.add_argument('--output',help='Fresh output folder; default results/run_<UTC>')
    p.add_argument('--_child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args._child:return child(args)
    if args.preflight:
        env=environment();write_json(ROOT/'results/preflight.json',env);print(json.dumps(env,indent=2))
        return 0 if env['passed'] else 2
    if args.rebuild_reference:
        command=[sys.executable,str(ROOT/'tools/rebuild_reference.py')]
        if args.output:command+=['--output',args.output]
        return subprocess.call(command,cwd=str(ROOT))
    if args.suite and (args.gui or args.smoke or args.dt is not None or args.solver is not None
                       or args.position_iterations is not None or args.joint_offsets is not None):
        p.error('Suite runs require the declared solver settings, headless mode, and complete duration; run GUI/smoke/dt/solver/iteration/offset diagnostics separately.')
    out=Path(args.output).resolve() if args.output else ROOT/'results'/('run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if out.exists() and any(out.iterdir()):p.error('Output folder is not empty. Choose a new folder to prevent stale results being accepted.')
    out.mkdir(parents=True,exist_ok=True)
    env=environment();write_json(out/'preflight.json',env)
    for warning in env['warnings']:print('WARNING: '+warning,flush=True)
    if not env['passed']:
        print(json.dumps(env,indent=2));print('Preflight failed. No Isaac process launched.');return 2
    if args.suite:
        # Import after stdlib-only preflight; cases.py itself is dependency-free.
        from cmg_isaac.cases import CONTACT,WRENCH
        modes=['contact','wrench'] if args.suite=='all' else [args.suite]
        codes={}
        for mode in modes:
            for name in (CONTACT if mode=='contact' else WRENCH):
                print(f'\n===== {mode}/{name} =====',flush=True)
                code=run_process(args,mode,name,out/mode/name);codes[mode+'/'+name]=code
                # Native fatal startup is not retried over and over in a suite.
                if code not in (0,1):
                    write_json(out/'suite.json',dict(status='SUITE_ABORTED',passed=False,returncodes=codes))
                    print('Suite stopped after an execution failure; see '+str(out));return 2
        from cmg_isaac.report import aggregate
        report=aggregate(ROOT,out,modes,codes)
        print(report['status']+'; output: '+str(out))
        return 0 if report['passed'] else 1
    code=run_process(args,args.mode,args.case,out/args.mode/args.case)
    print('\nOutput folder: '+str(out))
    return code if code in (0,1,2) else 2


if __name__=='__main__':raise SystemExit(main())
