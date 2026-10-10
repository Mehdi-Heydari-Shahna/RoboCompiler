"""Regression tests for faithful failure and compatibility reporting."""
import json
from pathlib import Path
import subprocess
import sys
from kangaroo_isaac.model import ROOT
from kangaroo_isaac.runtime_reporting import classify_physics_log,make_failure_report
from kangaroo_isaac.report import suite_report

TGS=('[Warning] [omni.physx.plugin] Detected an articulation at /World/Kangaroo/Bodies/base_link '
     'with more than 4 velocity iterations being added to a TGS scene.The related behavior changed '
     'recently, please consult the changelog. This warning will only print once.')
BINDING='[Error] [omni.physx.tensors.plugin] Failed to get a valid attached USD stage id from PhysX simulation'


def test_observed_native_failure_cannot_be_hidden_by_compatibility_notice():
    r=classify_physics_log(TGS+'\n'+BINDING)
    assert r['failures']==[BINDING]
    assert r['compatibility_notices']==[TGS]


def test_only_exact_known_warning_is_nonfatal():
    for line in [TGS.replace('[Warning]','[Error]'),TGS+' Another issue.',
                 TGS.replace('base_link','unknown_robot'), '[Warning] [omni.physx.plugin] Solver failed',
                 '[Fatal] [omni.physics.core] Physics failure']:
        r=classify_physics_log(line)
        assert r['failures']==[line] and not r['compatibility_notices']


def test_inconsistent_tgs_iteration_notice_is_failure():
    assert classify_physics_log(TGS,high_tgs_iterations=False)['failures']==[TGS]


def test_previous_panda_crash_is_historical_not_current_failure():
    old="[Warning] [carb.crashreporter-breakpad.plugin] [previous crash] 'commandLine' = 'Panda_IsaacSim_Validation_R2'"
    r=classify_physics_log(old+'\n'+BINDING)
    assert r['historical_crash_notices']==[old] and r['failures']==[BINDING]


def test_failure_case_and_root_reports_exist(tmp_path):
    case=tmp_path/'nominal';case.mkdir()
    (case/'error.json').write_text(json.dumps({'failed_phase':'BINDING_MANAGED_PHYSICS_VIEW','error':'stage not attached'}))
    make_failure_report(case,'nominal',2)
    text=(case/'REPORT.md').read_text()
    assert 'NOT COMPLETED' in text and 'stage not attached' in text
    summary=suite_report(tmp_path,['nominal'])
    assert summary['overall_status']=='NATIVE_RUN_NOT_COMPLETED' and not summary['certified_ready']
    assert (tmp_path/'REPORT.md').is_file()


def test_timeout_does_not_need_python_traceback_for_report(tmp_path):
    (tmp_path/'supervisor_error.json').write_text(json.dumps({'status':'TIMEOUT'}))
    make_failure_report(tmp_path,'nominal',124)
    assert 'TIMEOUT' in (tmp_path/'REPORT.md').read_text()


def test_release_identifier_is_visible_without_isaac_import():
    p=subprocess.run([sys.executable,str(ROOT/'run_isaac.py'),'--version'],capture_output=True,text=True)
    assert p.returncode==0 and p.stdout.strip()=='Kangaroo Isaac Sim 23.0.5'
