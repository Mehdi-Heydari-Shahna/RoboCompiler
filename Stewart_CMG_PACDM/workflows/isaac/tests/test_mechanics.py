"""Independent engine-Jacobian tests of actor/COM mapping and actuator power."""
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
mujoco=pytest.importorskip('mujoco')
pytest.importorskip('pxr', reason='OpenUSD is required for the independently exported body properties')
from stewart.model import make_cmg, compile_mujoco
from isaac_validation.usd_scene import build_scene
from isaac_validation.mechanics import StateMapping
ROOT=Path(__file__).resolve().parents[1]


def fixture_state(tmp_path,q,qd):
    cmg=make_cmg()
    m=mujoco.MjModel.from_xml_path(str(compile_mujoco(cmg,tmp_path/'model.xml')))
    d=mujoco.MjData(m)
    jid=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in cmg['coordinate_ids']])
    d.qpos[m.jnt_qposadr[jid]]=q;d.qvel[m.jnt_dofadr[jid]]=qd
    mujoco.mj_forward(m,d)
    audit=build_scene(cmg,q)['audit']['expected_body_properties']
    names=sorted(audit)
    coms=np.array([audit[n]['com_m'] for n in names])
    transforms=[]; velocities=[]; jacs=[]
    for name,com in zip(names,coms):
        bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
        R=d.xmat[bi].reshape(3,3)
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        mujoco.mj_jacBody(m,d,jp,jr,bi)
        w=jr@d.qvel; arm=R@com
        transforms.append(np.r_[d.xpos[bi],Rotation.from_matrix(R).as_quat()])
        velocities.append(np.r_[jp@d.qvel+np.cross(w,arm),w])
        skew=np.array([[0,-arm[2],arm[1]],[arm[2],0,-arm[0]],[-arm[1],arm[0],0]])
        jacs.append((jp-skew@jr,jr))
    mapping=StateMapping(cmg,names,coms)
    state=mapping.read(np.array(transforms),np.array(velocities))
    return cmg,m,mapping,state,jid,jacs


def test_tensor_state_and_force_against_mujoco_jacobians(tmp_path):
    rng=np.random.default_rng(111)
    with np.load(ROOT/'baseline/reference.npz') as ref:
        for k in np.linspace(0,len(ref['time'])-1,13,dtype=int):
            q=ref['q'][k];qd=ref['velocity'][k]
            cmg,m,mapping,state,jid,jacs=fixture_state(tmp_path,q,qd)
            np.testing.assert_allclose(state['q'],q,atol=2e-12)
            np.testing.assert_allclose(state['velocity'],qd,atol=2e-12)
            assert state['joint_error_m']<1e-12
            assert state['joint_error_rad']<1e-12
            force=rng.normal(0,100,6)
            f,tau=mapping.forces(state,force,np.zeros(6))
            total=sum(jp.T@fi+jr.T@ti for (jp,jr),fi,ti in zip(jacs,f,tau))
            expected=np.zeros(24);expected[mapping.active]=force
            np.testing.assert_allclose(total[m.jnt_dofadr[jid]],expected,atol=1e-10)
            assert abs(np.sum(f*state['com_velocities'][:,:3])+np.sum(tau*state['com_velocities'][:,3:])-force@qd[mapping.active])<1e-10
            np.testing.assert_allclose(f.sum(axis=0),0,atol=1e-12)
            np.testing.assert_allclose((tau+np.cross(state['com'],f)).sum(axis=0),0,atol=1e-12)


def test_platform_external_force_at_origin_with_offset_com(tmp_path):
    with np.load(ROOT/'baseline/reference.npz') as r:q=r['q'][350];qd=r['velocity'][350]
    _,m,mapping,state,jid,jacs=fixture_state(tmp_path,q,qd)
    wrench=np.array([3.,9.,-20.,4.,2.,6.])
    f,tau=mapping.forces(state,np.zeros(6),wrench)
    ip=mapping.idx['platform']
    np.testing.assert_allclose(f.sum(axis=0),wrench[:3])
    np.testing.assert_allclose((tau+np.cross(state['com']-state['p'][ip],f)).sum(axis=0),wrench[3:],atol=1e-12)
