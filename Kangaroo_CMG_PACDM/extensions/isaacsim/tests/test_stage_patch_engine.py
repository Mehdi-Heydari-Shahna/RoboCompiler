"""Test updater safety using tiny synthetic projects, not user directories."""
import base64
import importlib.util
import json
from pathlib import Path
import pytest
from kangaroo_isaac.model import ROOT
spec=importlib.util.spec_from_file_location('stage_patch_engine',ROOT/'tools/patch_engine.py')
patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)


def fixture(tmp_path):
    root=tmp_path/'project with spaces';(root/'data').mkdir(parents=True)
    baseline={'run_isaac.py':b'old runner','data/whole_body_cmg.json':b'{"unchanged":true}'}
    new={'run_isaac.py':b'new runner','kangaroo_isaac/runtime_bridge.py':b'new binding'}
    for n,d in baseline.items():(root/n).write_bytes(d)
    payload={'version':'23.0.2','base_hashes':{n:patch.digest(d) for n,d in baseline.items()},
        'files':{n:base64.b64encode(d).decode() for n,d in new.items()},
        'target_hashes':{n:patch.digest(d) for n,d in new.items()}}
    return root,payload,baseline,new


def test_updater_applies_backs_up_and_is_idempotent(tmp_path):
    root,payload,baseline,new=fixture(tmp_path)
    (root/'results').mkdir();(root/'results/keep.txt').write_text('previous result')
    result=patch.apply_patch(root,payload)
    assert result['status']=='INSTALLED'
    assert (Path(result['backup'])/'run_isaac.py').read_bytes()==baseline['run_isaac.py']
    for n,d in new.items():assert (root/n).read_bytes()==d
    assert (root/'results/keep.txt').read_text()=='previous result'
    assert patch.apply_patch(root,payload)['status']=='ALREADY_INSTALLED'


def test_dry_run_does_not_modify_anything(tmp_path):
    root,payload,baseline,new=fixture(tmp_path)
    before=sorted(str(p.relative_to(root)) for p in root.rglob('*'))
    assert patch.apply_patch(root,payload,dry_run=True)['status']=='DRY_RUN'
    assert (root/'run_isaac.py').read_bytes()==baseline['run_isaac.py']
    assert before==sorted(str(p.relative_to(root)) for p in root.rglob('*'))


@pytest.mark.parametrize('name',['run_isaac.py','data/whole_body_cmg.json'])
def test_custom_changes_are_not_overwritten(tmp_path,name):
    root,payload,baseline,new=fixture(tmp_path)
    (root/name).write_bytes(b'custom edits')
    with pytest.raises(ValueError,match='No files were changed'):patch.apply_patch(root,payload)
    assert (root/name).read_bytes()==b'custom edits'
    assert not (root/'_patch_backups').exists()


def test_update_rolls_back_on_failed_replace(tmp_path,monkeypatch):
    root,payload,baseline,new=fixture(tmp_path)
    original=patch.os.replace;calls=[]
    def fail(src,dst):
        calls.append(dst)
        if len(calls)==2:raise OSError('simulated disk failure')
        return original(src,dst)
    monkeypatch.setattr(patch.os,'replace',fail)
    with pytest.raises(OSError,match='simulated'):patch.apply_patch(root,payload)
    assert (root/'run_isaac.py').read_bytes()==baseline['run_isaac.py']
    assert not (root/'kangaroo_isaac/runtime_bridge.py').exists()
    assert not (root/'.kangaroo_patch.lock').exists()


@pytest.mark.parametrize('path',['../other.py','/other.py','C:/other.py','foo\\bar.py'])
def test_path_traversal_is_rejected(tmp_path,path):
    with pytest.raises(ValueError):patch.safe_path(tmp_path.resolve(),path)


def test_corrupt_payload_rejected_before_any_edit(tmp_path):
    root,payload,baseline,new=fixture(tmp_path)
    payload['target_hashes']['run_isaac.py']='bad'
    with pytest.raises(ValueError,match='integrity'):patch.apply_patch(root,payload)
    assert (root/'run_isaac.py').read_bytes()==baseline['run_isaac.py']


def test_wrong_project_rejected(tmp_path):
    with pytest.raises(ValueError,match='not the Kangaroo'):patch.inspect_patch(tmp_path,{})


def test_run_flag_executes_integrity_offline_then_native(tmp_path,monkeypatch):
    root,payload,baseline,new=fixture(tmp_path);commands=[]
    monkeypatch.setattr(patch.subprocess,'call',lambda cmd,cwd:(commands.append(cmd) or 0))
    assert patch.patch_main(payload,['--project',str(root),'--run'])==0
    assert 'verify_package.py' in commands[0][1]
    assert commands[1][-1]=='--offline-test'
    assert commands[2][-3:]==['--verify-native','--headless','--no-visuals']


def test_failed_offline_tests_prevent_native_launch(tmp_path,monkeypatch):
    root,payload,baseline,new=fixture(tmp_path);commands=[]
    def call(cmd,cwd):
        commands.append(cmd);return 1 if '--offline-test' in cmd else 0
    monkeypatch.setattr(patch.subprocess,'call',call)
    assert patch.patch_main(payload,['--project',str(root),'--run'])==1
    assert len(commands)==2
