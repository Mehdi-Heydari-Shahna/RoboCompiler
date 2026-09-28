"""These tests exercise real numerical code, NOT a mocked PhysX simulation."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from kangaroo_isaac.model import axis_frame,inverse,ROOT,poses_from_xyzw
from kangaroo_isaac.pacdm import PACDM
from kangaroo_isaac.offline import audit_reference


def test_source_topology_mass_and_ports(model):
    assert (model.nb,model.n,len(model.edges),len(model.active))==(78,76,77,12)
    assert len(model.ca)==24 and len(model.universal)==8
    assert abs(model.mass.sum()-42.21486760806909)<1e-12
    assert np.count_nonzero(model.bounds[:,1]==2000)==10
    assert np.count_nonzero(model.bounds[:,1]==5000)==2
    assert np.all(model.bounds[:,0]==-model.bounds[:,1])
    assert np.count_nonzero(model.armature)==12
    assert np.all(model.armature[model.active]==.01)
    assert np.all(model.damping==3)
    assert sum(j['type']=='fixed' for j in model.c['joints'])==1
    assert np.all(model.lower<=model.upper)


def test_all_joint_axis_frames_and_offsets(model):
    for j in model.c['joints']:
        Q=axis_frame(j['axis']); A,B=model.joint_frames(j)
        np.testing.assert_allclose(Q.T@Q,np.eye(3),atol=1e-14)
        assert np.linalg.det(Q)==pytest.approx(1)
        np.testing.assert_allclose(Q[:,0],j['axis'],atol=1e-14)
        np.testing.assert_allclose(A[:3,3],np.array(j['T_BJ'])[:3,3],atol=1e-14)
        np.testing.assert_allclose(B[:3,3],np.array(j['T_FJ'])[:3,3],atol=1e-14)
        # Independently reconstruct a +X native joint motion at an arbitrary value.
        k=.217;S=np.eye(4);X=np.eye(4)
        if j['type']=='prismatic':
            S[:3,3]=np.asarray(j['axis'])*k;X[0,3]=k
        elif j['type']=='revolute':
            S[:3,:3]=Rotation.from_rotvec(np.asarray(j['axis'])*k).as_matrix()
            X[:3,:3]=Rotation.from_rotvec([k,0,0]).as_matrix()
        expected=np.asarray(j['T_BJ'])@S@inverse(np.asarray(j['T_FJ']))
        np.testing.assert_allclose(A@X@inverse(B),expected,atol=2e-14)


def test_all_body_principal_inertias(model):
    for I in model.I_local:
        eig,Q=np.linalg.eigh(I)
        if np.linalg.det(Q)<0:Q[:,0]*=-1
        assert eig.min()>0
        assert eig[2]<=eig[0]+eig[1]+1e-10
        np.testing.assert_allclose(Q@np.diag(eig)@Q.T,I,atol=2e-15)
        quat=Rotation.from_matrix(Q).as_quat().astype(np.float32)
        Qf=Rotation.from_quat(quat).as_matrix()
        np.testing.assert_allclose(Qf@np.diag(eig.astype(np.float32))@Qf.T,I,rtol=2e-5,atol=2e-9)


@pytest.mark.parametrize('index',[0,61,157,274,400])
def test_fk_energy_and_virtual_power_against_source(model,ref,dynamics,index):
    rng=np.random.default_rng(120+index)
    q=ref.data['q'][index];qd=rng.normal(0,.2,model.n)
    d=dynamics.evaluate(q,qd,np.array([0,0,-9.81]))
    P=model.fk(q)
    np.testing.assert_allclose(P,np.array([d['poses'][n] for n in model.names]),atol=4e-14)
    np.testing.assert_allclose(model.coordinates_from_poses(P),q,atol=4e-14)
    V=np.array([np.r_[d['body_com_jacobians'][n]@qd,d['jacobians'][n][3:]@qd] for n in model.names])
    K,U,rot=model.energy(P,V,qd)
    assert K.sum()==pytest.approx(d['kinetic_energy'],abs=2e-12)
    assert U.sum()==pytest.approx(d['potential_energy'],abs=2e-12)
    assert rot.sum()==pytest.approx(.5*np.sum(model.armature*qd**2),abs=1e-15)
    f=rng.uniform(-2000,2000,12)
    w,power=model.port_wrenches(P,V,f)
    assert power==pytest.approx(float(f@qd[model.active]),abs=3e-10)
    np.testing.assert_allclose(w[:,:3].sum(axis=0),0,atol=2e-12)
    moment=w[:,3:]+np.cross(model.com_world(P),w[:,:3])
    np.testing.assert_allclose(moment.sum(axis=0),0,atol=2e-12)


@pytest.mark.parametrize('index',[0,81,165,268,400])
def test_pacdm_jacobian_finite_difference_and_tangent(model,ref,graph,index):
    q=ref.data['qaug'][index];r,J,_=graph.residual(q)
    rng=np.random.default_rng(77+index);v=rng.normal(size=graph.n);v/=np.linalg.norm(v)
    h=2e-7
    fd=(graph.residual(q+h*v)[0]-graph.residual(q-h*v)[0])/(2*h)
    np.testing.assert_allclose(J@v,fd,atol=2e-8,rtol=2e-6)
    N,info=PACDM(graph).mapping(q)
    assert info['success'] and info['rank_full']==128 and info['rank_passive']==128
    np.testing.assert_allclose(N[graph.active],np.eye(12),atol=1e-15)
    np.testing.assert_allclose(J@N,0,atol=2e-11)
    np.testing.assert_allclose(N[:76],ref.data['tangent'][index],atol=1e-10,rtol=1e-10)


def test_pacdm_rejects_broken_closure(ref,graph):
    q=ref.data['qaug'][0].copy();q[graph.active[0]]+=.005
    N,info=PACDM(graph).mapping(q)
    assert N is None and not info['success']


def test_all_401_reference_samples_executed(tmp_path):
    result=audit_reference(tmp_path,verbose=False)
    assert result['status']=='PASS'
    assert result['samples_checked']==401
    assert result['rank_full_values']==result['rank_passive_values']==[128]
    assert result['unmodified_pacdm_core']
    assert result['maximum_errors']['physical_point_gap_m']<1e-10
    assert result['minimum_reference_universal_gimbal_margin_deg']>80
    assert result['isaac_sim_execution']=='NOT_RUN' and not result['certified_ready']


@pytest.mark.parametrize('index',[0,130,266,400])
def test_closures_invariant_to_floating_base(model,ref,index):
    q=ref.data['q'][index];P=model.fk(q)
    R=Rotation.from_rotvec([.43,-.24,.61]).as_matrix()
    W=model.fk(q,[.21,-.42,1.12],R)
    gap,dot,dist=model.closures(P);wgap,wdot,wdist=model.closures(W)
    np.testing.assert_allclose(wgap,gap@R.T,atol=3e-15)
    np.testing.assert_allclose(wdot,dot,atol=3e-15)
    np.testing.assert_allclose(wdist,dist,atol=3e-15)


@pytest.mark.parametrize('a,b',[(-.7,.5),(.4,-.6),(1.0,.3),(-.2,-.3)])
def test_universal_d6_requires_reversed_body_order(a,b):
    # Source convention B=A Rx(a) Ry(b). Thus A.X dot B.Y=0.
    A=Rotation.from_rotvec([.3,-.1,.4]).as_matrix()
    B=A@Rotation.from_rotvec([a,0,0]).as_matrix()@Rotation.from_rotvec([0,b,0]).as_matrix()
    assert abs(A[:,0]@B[:,1])<1e-14
    # PhysX free twist + swing1: actor1.X dot actor0.Y=0.
    actor0,actor1=B,A
    assert abs(actor1[:,0]@actor0[:,1])<1e-14
    # Keeping original actor order produces a genuinely different manifold.
    assert abs(B[:,0]@A[:,1])>1e-3


def test_sole_geometry_and_tensor_quaternions(model,ref):
    P=model.fk(ref.data['q'][0],ref.data['base'][0])
    assert abs(model.foot_points(P)[:,:,2].min())<1e-10
    xyzquat=np.c_[P[:,:3,3],Rotation.from_matrix(P[:,:3,:3]).as_quat()]
    np.testing.assert_allclose(poses_from_xyzw(xyzquat),P,atol=3e-14)
    assert len(model.visual['collisions'])==2


def test_all_visual_mesh_buffers_are_valid(model):
    paths=list((ROOT/'assets').glob('*.npz'));assert len(paths)==42
    for p in paths:
        with np.load(p,allow_pickle=False) as a:
            vertices=a['points'];tri=a['indices']
            assert vertices.ndim==2 and vertices.shape[1]==3
            assert np.isfinite(vertices).all() and len(vertices)>0
            assert np.issubdtype(tri.dtype,np.integer) and tri.size%3==0
            assert tri.min()>=0 and tri.max()<len(vertices)


def test_usd_float32_inertia_storage_and_per_body_sensitivity(model):
    from kangaroo_isaac.model import inertias_match
    all_I=[]
    for I in model.I_local:
        d,Q=np.linalg.eigh(I)
        if np.linalg.det(Q)<0:Q[:,0]*=-1
        Q=Rotation.from_quat(Rotation.from_matrix(Q).as_quat().astype(np.float32)).as_matrix()
        all_I.append(Q@np.diag(d.astype(np.float32))@Q.T)
    assert inertias_match(np.array(all_I),model.I_local)
    wrong=np.array(all_I);idx=np.argmin(np.max(abs(model.I_local),axis=(1,2)))
    wrong[idx]*=1.01
    assert not inertias_match(wrong,model.I_local)


def test_all_flattened_visual_faces_have_correct_counts():
    from kangaroo_isaac.model import triangular_face_counts
    for p in (ROOT/'assets').glob('*.npz'):
        with np.load(p,allow_pickle=False) as a:
            count=triangular_face_counts(a['indices'])
            assert int(count.sum())==a['indices'].size
            assert len(count)==a['indices'].size//3
    with pytest.raises(ValueError):triangular_face_counts(np.array([1,2]))
    with pytest.raises(ValueError):triangular_face_counts(np.array([0.,1.,2.]))
