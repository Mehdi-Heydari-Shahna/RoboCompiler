"""V2 CPU regressions. Synthetic fixtures are not simulator execution evidence."""
import ast
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from isaac_validation.compat import BodyView
from isaac_validation.model_checks import ReducedModel,check_reference,audit_case,physical_case_cmg
from isaac_validation.worker_check import check_worker

ROOT=Path(__file__).resolve().parents[1]


def test_fresh_newton_euler_matches_archived_reference():
    result=check_reference(ROOT)
    assert result['passed'],result


def test_reduced_dynamics_is_sensitive_to_actual_forces():
    cmg=physical_case_cmg(ROOT,'nominal')
    with np.load(ROOT/'baseline/reference.npz') as ref:
        seed=ref['q'][0]; q=ref['q'][300]; ld=ref['length_velocity'][300]
    model=ReducedModel(cmg,seed); a=model.terms(q,ld)
    np.testing.assert_allclose(a['N'][model.graph.active],np.eye(6),atol=1e-14)
    assert a['info']['rank_full']==36 and a['info']['rank_passive']==36
    assert a['info']['tangent_residual']<1e-8
    changed=np.linalg.solve(a['mass'],np.ones(6)*20)
    assert np.max(abs(changed))>.01
    # No fitted feedback gain enters the reduced force balance.
    np.testing.assert_allclose(a['mass']@changed,np.ones(6)*20,atol=1e-10)


def test_payload_mass_change_changes_dynamics():
    with np.load(ROOT/'baseline/reference.npz') as ref:seed=ref['q'][0]
    light=ReducedModel(physical_case_cmg(ROOT,'nominal'),seed).terms(seed,np.zeros(6))
    heavy=ReducedModel(physical_case_cmg(ROOT,'heavy_payload'),seed).terms(seed,np.zeros(6))
    assert np.max(abs(light['bias']-heavy['bias']))>1.
    assert np.max(abs(light['mass']-heavy['mass']))>.01


def test_archived_trajectory_cannot_pass_new_raw_body_audit():
    with np.load(ROOT/'baseline/nominal.npz') as z:data={k:z[k] for k in z.files}
    result=audit_case(data,physical_case_cmg(ROOT,'nominal'),data['q'][0])
    assert not result['passed']
    assert 'body_names' in result['error']


def test_worker_failure_artifact_overrides_exit_zero(tmp_path):
    (tmp_path/'nominal.failure.json').write_text(json.dumps(dict(error='PhysX could not attach stage')))
    r=check_worker(tmp_path,'nominal','test',0,22.,.002)
    assert not r['passed']
    assert any('PhysX could not attach' in s for s in r['errors'])


def test_missing_worker_outputs_fail_even_on_exit_zero(tmp_path):
    r=check_worker(tmp_path,'nominal','test',0,22.,.002)
    assert not r['passed'] and r['errors']


def _worker_fixture(directory,n=1001):
    # Explicitly synthetic shape/identity fixture, not physics data.
    t=np.arange(n)*.002
    np.savez(directory/'nominal.npz',time=t,engine_time=t,run_id=np.array('unit-test'),
             q=np.zeros((n,24)),velocity=np.zeros((n,24)),force=np.zeros((n,6)),
             wrench=np.zeros((n,6)),body_transforms=np.zeros((n,19,7)),body_velocities=np.zeros((n,19,6)))
    (directory/'nominal.json').write_text(json.dumps(dict(name='nominal',completed=True,run_id='unit-test')))


def test_worker_completed_recording_and_wrong_identity(tmp_path):
    _worker_fixture(tmp_path)
    assert check_worker(tmp_path,'nominal','unit-test',0,2.,.002)['passed']
    assert not check_worker(tmp_path,'nominal','wrong',0,2.,.002)['passed']
    assert not check_worker(tmp_path,'nominal','unit-test',0,22.,.002)['passed']
    assert not check_worker(tmp_path,'nominal','unit-test',1,2.,.002)['passed']


def test_runtime_does_not_overwrite_case_name_in_loop():
    tree=ast.parse((ROOT/'isaac_validation/runtime.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.For):
            assert not isinstance(node.target,ast.Name) or node.target.id!='name'


def test_numpy_body_view_preserves_boundary_dtypes():
    class Raw:
        prim_paths=['/body']
        def get_transforms(self):return np.zeros((1,7),np.float32)
        def apply_forces_and_torques_at_position(self,f,t,p,i,world):
            assert all(x.dtype==np.float32 for x in [f,t,p])
            assert i.dtype==np.int32 and world is True
            return True
    view=BodyView(Raw())
    assert view.frontend=='numpy'
    assert view.apply_forces_and_torques_at_position([[1,2,3]],[[0,0,0]],[[0,0,0]],[0],True)


def _synthetic_equilibrium():
    """Stationary algebra fixture, no engine and NEVER exported as run evidence."""
    cmg=physical_case_cmg(ROOT,'nominal')
    with np.load(ROOT/'baseline/reference.npz') as z:seed=z['q'][0].copy()
    model=ReducedModel(cmg,seed); a=model.terms(seed,np.zeros(6))
    names=['platform']+[f'leg_{i}_{part}' for i in range(6) for part in ['yoke','barrel','rod']]
    poses=np.array([np.r_[a['poses'][n][:3,3],Rotation.from_matrix(a['poses'][n][:3,:3]).as_quat()] for n in names])
    bodies={b['id']:b for b in cmg['bodies']}
    coms=np.array([bodies[n]['com_m'] for n in names],float)
    # Aggregate payload COM in the platform body frame, as in native rigid import.
    pp=a['poses']['platform']; py=a['poses']['payload']
    payload_local=pp[:3,:3].T@(py[:3,3]+py[:3,:3]@bodies['payload']['com_m']-pp[:3,3])
    mp=bodies['platform']['mass_kg']; my=bodies['payload']['mass_kg']
    coms[0]=(mp*coms[0]+my*payload_local)/(mp+my)
    count=11001
    data=dict(time=np.arange(count)*.002,body_names=np.array(names),body_coms=coms,
         body_transforms=np.broadcast_to(poses,(count,19,7)),body_velocities=np.zeros((count,19,6)),
         q=np.broadcast_to(seed,(count,24)),velocity=np.zeros((count,24)),
         force=np.broadcast_to(a['bias'],(count,6)).copy(),wrench=np.zeros((count,6)))
    return data,cmg,seed


def test_model_audit_equilibrium_and_force_negative_control():
    data,cmg,seed=_synthetic_equilibrium()
    a=audit_case(data,cmg,seed)
    assert a['passed'],a
    data['force']+=30.
    b=audit_case(data,cmg,seed)
    assert not b['passed']
    assert not b['checks']['leg_1_force_balance_rms']['passed']


def test_raw_body_mismatch_cannot_be_hidden_by_good_platform_tracking():
    data,cmg,seed=_synthetic_equilibrium()
    poses=data['body_transforms'].copy()
    poses[:,5,0]+=.001
    data['body_transforms']=poses
    result=audit_case(data,cmg,seed)
    assert not result['passed'] and not result['checks']['body_position']['passed']


def test_nonfinite_raw_samples_outside_audit_grid_still_fail():
    data,cmg,seed=_synthetic_equilibrium()
    data['body_velocities'][17,2,3]=np.nan
    result=audit_case(data,cmg,seed)
    assert not result['passed'] and 'Invalid raw schema' in result['error']
