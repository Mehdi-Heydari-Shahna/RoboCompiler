"""Executable API regression fixtures, NOT Isaac Sim physics validation.

The arrays below are prescribed synthetic readbacks. They validate ownership,
indexing, marshaling, rejection paths and worker control flow only. No result
from these fixtures is an empirical native tracking/performance measurement.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kangaroo_isaac.control import Config
from kangaroo_isaac.model import ROOT
from kangaroo_isaac.native import initialize_tensors
from kangaroo_isaac.runtime_bridge import (PhysicsBindingError, TensorIO, managed_simulation_view,
    fill_disturbance_buffers, require_methods)
from kangaroo_isaac.scene_paths import BODIES


class FixtureView:
    def __init__(self, art=None, valid=True):
        self.art = art
        self.valid = valid
        self.roots = []
        self.contacts = []
        self.kinematic_updates = 0
    def check(self): return self.valid
    def set_subspace_roots(self, path): self.roots.append(path)
    def create_articulation_view(self, path):
        self.art_path = path
        return self.art
    def create_rigid_contact_view(self, path, **kwargs):
        cv = SimpleNamespace(get_net_contact_forces=lambda dt: self.art.encode(np.zeros((1, 3), np.float32)))
        self.contacts.append((path, kwargs, cv))
        return cv
    def update_articulations_kinematic(self): self.kinematic_updates += 1


class FixtureWorld:
    def __init__(self, view=None, replacement=None):
        self.physics_sim_view = view
        self.replacement = replacement
        self.initializations = 0
        self.current_time = 0.
        self.current_time_step_index = 0
        self.dt = Config().dt_s
    def initialize_physics(self):
        self.initializations += 1
        self.physics_sim_view = self.replacement
    def get_physics_dt(self): return self.dt
    def is_playing(self): return True
    def step(self, render=False):
        self.current_time_step_index += 1
        self.current_time = self.current_time_step_index*self.dt


def manager_for(view=None, engine='physx'):
    return SimpleNamespace(get_active_physics_engine=lambda:engine,
        get_physics_simulation_view=lambda:view, get_physics_sim_view=lambda:view)


@pytest.mark.parametrize('origin', ['world', 'manager', 'legacy_manager'])
def test_uses_owned_view_without_ambient_factory_or_warmup(origin):
    v = FixtureView()
    w = FixtureWorld(v if origin == 'world' else None)
    m = manager_for(v if origin != 'world' else None)
    if origin == 'legacy_manager': del m.get_physics_simulation_view
    audit = {}
    assert managed_simulation_view(w, m, audit) is v
    assert audit['application_created_extra_simulation_view'] is False
    assert audit['application_warmup_steps'] == 0
    assert v.roots == ['/'] and w.initializations == 0
    assert audit['view_origin'].startswith('World' if origin == 'world' else 'SimulationManager')


@pytest.mark.parametrize('invalid', [False, True])
def test_exactly_one_documented_reinitialization_when_needed(invalid):
    v = FixtureView()
    w = FixtureWorld(FixtureView(valid=False) if invalid else None, replacement=v)
    audit = {}
    assert managed_simulation_view(w, manager_for(), audit) is v
    assert w.initializations == 1 and audit['explicit_reinitializations'] == 1


def test_missing_stage_view_fails_without_infinite_retry():
    w = FixtureWorld()
    with pytest.raises(PhysicsBindingError, match='did not produce'):
        managed_simulation_view(w, manager_for())
    assert w.initializations == 1


def test_invalid_world_view_uses_valid_manager_view():
    v = FixtureView()
    w = FixtureWorld(FixtureView(valid=False))
    assert managed_simulation_view(w, manager_for(v)) is v
    assert w.initializations == 0


@pytest.mark.parametrize('engine', ['newton', 'remotesim', 'unknown'])
def test_never_substitutes_a_different_physics_engine(engine):
    w = FixtureWorld(FixtureView())
    with pytest.raises(PhysicsBindingError, match='different engine'):
        managed_simulation_view(w, manager_for(engine=engine))
    assert w.initializations == 0


def test_missing_native_api_is_not_success():
    w = FixtureWorld(SimpleNamespace(check=lambda:True))
    with pytest.raises(PhysicsBindingError, match='required native methods'):
        managed_simulation_view(w, manager_for())


class FixtureWarpArray:
    """Type-shaped Warp fixture, explicitly not the Warp package."""
    __module__ = 'warp.fixture'
    def __init__(self, data, device='cpu', copy=False):
        self.data = np.array(data, copy=True) if copy else np.asarray(data)
        self.device = device
    def numpy(self): return self.data


def fixture_warp(monkeypatch, copies=False):
    m=types.ModuleType('warp')
    m.uint32=np.uint32; m.float32=np.float32
    m.from_numpy=lambda a,dtype,device:FixtureWarpArray(np.asarray(a,dtype=dtype),device,copy=copies)
    monkeypatch.setitem(sys.modules,'warp',m)
    return m


@pytest.mark.parametrize('frontend', ['numpy', 'warp_fixture'])
def test_tensor_read_write_float32_uint32_and_shapes(frontend, monkeypatch):
    fixture_warp(monkeypatch)
    sample=np.zeros((1, 76),np.float32)
    io=TensorIO(sample if frontend=='numpy' else FixtureWarpArray(sample.reshape(-1)))
    assert io.frontend==frontend.removesuffix('_fixture')
    a=io.array(np.arange(12,dtype=np.float64).reshape(1,12))
    np.testing.assert_array_equal(io.read(a,(1,12)),np.arange(12).reshape(1,12))
    assert io.read(a).dtype==np.float32
    index=io.array([0,3],indices=True)
    assert io.read(index).dtype==np.uint32
    assert io.read(sample.reshape(-1),(1,76)).shape==(1,76)


@pytest.mark.parametrize('value', [[-1], [0.1], [2**33]])
def test_illegal_native_indices_are_rejected(value):
    with pytest.raises(PhysicsBindingError,match='indices'):
        TensorIO(np.zeros(1)).array(value,indices=True)


@pytest.mark.parametrize('value', [np.array([np.nan]), np.array([np.inf]), np.array(['a']), np.array([1j])])
def test_nonfinite_and_nonreal_native_readbacks_are_rejected(value):
    with pytest.raises(PhysicsBindingError): TensorIO.read(value)
    with pytest.raises(PhysicsBindingError): TensorIO(np.zeros(1)).array(value)


def test_unexpected_array_size_is_rejected():
    with pytest.raises(PhysicsBindingError,match='shape'):
        TensorIO.read(np.zeros(5),(1,3))


def test_gpu_native_pipeline_not_silently_copied_to_cpu(monkeypatch):
    fixture_warp(monkeypatch)
    with pytest.raises(PhysicsBindingError,match='CPU'):
        TensorIO(FixtureWarpArray(np.zeros(5),'cuda:0'))


@pytest.mark.parametrize('copies', [False, True])
def test_command_buffers_never_submit_stale_values(monkeypatch,copies):
    fixture_warp(monkeypatch,copies)
    io=TensorIO(FixtureWarpArray(np.zeros(3)))
    host=np.zeros((1,76),np.float32)
    buf=io.buffer(host)
    assert buf.shared is (not copies)
    host[0,17]=1267.
    assert io.read(buf.native,(1,76))[0,17]==1267.
    host.fill(0)
    assert not io.read(buf.native).any()


def test_real_torch_cpu_frontend_when_installed():
    # Real Torch conversion, still NOT Isaac/PhysX execution.
    torch=pytest.importorskip('torch')
    io=TensorIO(torch.zeros((1,76)))
    host=np.zeros((1,76),np.float32);buf=io.buffer(host)
    assert io.frontend=='torch' and buf.shared
    host[0,8]=32.5
    assert buf.native[0,8].item()==32.5
    assert io.array([0],indices=True).dtype==torch.int32


class FixtureArticulation:
    """Synthetic source-consistent readbacks in a shuffled native order."""
    def __init__(self, model, ref, P, encode=lambda x:x):
        self.encode=encode
        self.count=1
        rng=np.random.default_rng(20260925)
        self.dofs=rng.permutation(model.n)
        self.links=rng.permutation(len(model.names))
        self.shared_metatype=SimpleNamespace(fixed_base=False,link_count=78,dof_count=76,
            dof_names=[model.ids[i] for i in self.dofs],link_names=[model.names[i] for i in self.links])
        com=np.zeros((78,7),np.float32);com[:,:3]=model.com_local;com[:,6]=1
        pose=np.column_stack((P[:,:3,3],Rotation.from_matrix(P[:,:3,:3]).as_quat())).astype(np.float32)
        self.data={
            'dof_positions':ref.data['q'][0,self.dofs][None].astype(np.float32),
            'dof_velocities':np.zeros((1,76),np.float32),
            'link_transforms':pose[self.links][None],
            'link_velocities':np.zeros((1,78,6),np.float32),
            'masses':model.mass[self.links][None].astype(np.float32),
            'coms':com[self.links][None],
            'inertias':model.I_local[self.links].reshape(1,78,9).astype(np.float32),
            'dof_dampings':model.damping[self.dofs][None].astype(np.float32),
            'dof_armatures':model.armature[self.dofs][None].astype(np.float32),
            'dof_limits':np.column_stack((model.lower,model.upper))[self.dofs][None].astype(np.float32),
            'dof_stiffnesses':np.zeros((1,76),np.float32),
            'dof_friction_properties':np.zeros((1,76,3),np.float32),
        }
        self.setters=[];self.wrenches=[]
    def __getattr__(self,name):
        if name.startswith('get_') and name[4:] in self.data:
            return lambda:self.encode(self.data[name[4:]].copy())
        if name.startswith('set_'):
            def setter(value,indices):
                value=TensorIO.read(value).copy();indices=TensorIO.read(indices)
                assert indices.tolist()==[0]
                self.setters.append((name,value))
                self.data[name[4:]]=value
            return setter
        raise AttributeError(name)
    def apply_forces_and_torques_at_position(self,force_data,torque_data,position_data,indices,is_global):
        for a in (force_data,torque_data,position_data):
            assert TensorIO.read(a).shape==(1,78,3)
        assert is_global is True
        self.wrenches.append(tuple(TensorIO.read(x).copy() for x in (force_data,torque_data,position_data)))


def initial_fixture(model,ref,encode=lambda x:x):
    P=model.fk(ref.data['q'][0],ref.data['base'][0],Rotation.from_rotvec(ref.data['rotvec'][0]).as_matrix())
    P[:,2,3]+=Config().drop_height_m
    art=FixtureArticulation(model,ref,P,encode)
    v=FixtureView(art)
    return P,art,v,FixtureWorld(v)


@pytest.mark.parametrize('frontend',['numpy','warp_fixture'])
def test_complete_tensor_initialization_and_readback(model,ref,monkeypatch,frontend):
    fixture_warp(monkeypatch)
    encode=(lambda x:x) if frontend=='numpy' else (lambda x:FixtureWarpArray(x))
    P,art,v,w=initial_fixture(model,ref,encode)
    binding={}
    result=initialize_tensors(w,model,P,ref,manager=manager_for(),binding_audit=binding)
    view,actual_art,contacts,qm,bm,idx,audit,io=result
    assert view is v and actual_art is art and len(contacts)==2
    assert audit['status']=='PASS' and all(audit['checks'].values())
    np.testing.assert_array_equal(art.dofs[qm],np.arange(76))
    np.testing.assert_array_equal(art.links[bm],np.arange(78))
    assert v.art_path==BODIES+'/'+model.c['root_body']
    assert w.initializations==0
    assert len([n for n,_ in art.setters if n in ('set_dof_positions','set_dof_velocities','set_root_transforms','set_root_velocities')])==4
    assert binding['tensor_frontend']==frontend.removesuffix('_fixture')


@pytest.mark.parametrize('defect',['mass','com','inertia','limits','poses','names','fixed_base','count'])
def test_bad_native_model_cannot_pass_readback(model,ref,defect):
    P,a,v,w=initial_fixture(model,ref)
    if defect=='mass':a.data['masses'][0,0]*=2
    if defect=='com':a.data['coms'][0,0,0]+=.1
    if defect=='inertia':a.data['inertias'][0,0]*=2
    if defect=='limits':a.data['dof_limits'][0,0,0]+=.1
    if defect=='poses':a.data['link_transforms'][0,0,2]+=.1
    if defect=='names':a.shared_metatype.dof_names[0]='unexpected_joint'
    if defect=='fixed_base':a.shared_metatype.fixed_base=True
    if defect=='count':a.shared_metatype.link_count=79
    with pytest.raises(RuntimeError):
        initialize_tensors(w,model,P,ref,manager=manager_for())


def test_contact_shape_mismatch_cannot_pass(model,ref):
    P,a,v,w=initial_fixture(model,ref)
    v.create_rigid_contact_view=lambda *args,**kwargs:SimpleNamespace(get_net_contact_forces=lambda dt:np.zeros((2,3)))
    with pytest.raises(PhysicsBindingError,match='shape'):
        initialize_tensors(w,model,P,ref,manager=manager_for())


@pytest.mark.parametrize('theta',[0.,.5,1.5])
def test_push_at_actual_com_not_actor_origin(model,ref,theta):
    P,art,v,w=initial_fixture(model,ref)
    R=Rotation.from_rotvec([theta,-theta*.3,theta*.7]).as_matrix()
    P[:,:3,:3]=R@P[:,:3,:3];P[:,:3,3]=P[:,:3,3]@R.T+np.array([.4,-.3,.6])
    bm=np.argsort(art.links)
    forces=np.zeros((1,78,3),np.float32);positions=np.zeros_like(forces)
    push=np.array([0,50.,0])
    fill_disturbance_buffers(model,P,bm,push,forces,positions)
    expected=P[model.torso,:3,3]+P[model.torso,:3,:3]@model.com_local[model.torso]
    np.testing.assert_allclose(positions[0,bm[model.torso]],expected,atol=2e-7)
    np.testing.assert_array_equal(forces[0,bm[model.torso]],push)
    assert np.count_nonzero(forces)==1
    np.testing.assert_allclose(np.cross(positions[0,bm[model.torso]]-expected,push),0,atol=1e-5)
    fill_disturbance_buffers(model,P,bm,np.zeros(3),forces,positions)
    assert not forces.any()


def install_worker_fixture(monkeypatch,model,ref,tmp_path,valid=True):
    """A fake physics API for executing worker Python logic; never real physics."""
    P,art,view,_=initial_fixture(model,ref)
    view.valid=valid
    instances=[]
    layer=SimpleNamespace(identifier='synthetic-test-only',Export=lambda path:Path(path).write_text('# API FIXTURE, NOT USD PHYSICS\n'))
    stage=SimpleNamespace(GetRootLayer=lambda:layer)
    class World(FixtureWorld):
        def __init__(self,**kwargs):
            super().__init__(view)
            self.dt=kwargs['physics_dt'];self.stage=stage
            instances.append(self)
        def reset(self): pass
    class App:
        def __init__(self,*args,**kwargs): self.closed=False
        def is_running(self): return not self.closed
        def close(self): self.closed=True
    modules={
        'isaacsim':{},
        'isaacsim.simulation_app':{'SimulationApp':App},
        'isaacsim.core':{},
        'isaacsim.core.api':{'World':World},
        'isaacsim.core.utils':{},
        'isaacsim.core.utils.extensions':{'enable_extension':lambda name:None},
        'isaacsim.core.simulation_manager':{'SimulationManager':manager_for(view)},
        'omni':{},
        'omni.usd':{'get_context':lambda:SimpleNamespace(get_stage=lambda:stage,get_stage_id=lambda:42)},
        'kangaroo_isaac.usd_builder':{'build':lambda *args,**kwargs:P,
                                    'audit_stage':lambda *args,**kwargs:{'status':'API_FIXTURE_ONLY'}},
    }
    for name,attrs in modules.items():
        m=types.ModuleType(name);m.__dict__.update(attrs);m.__path__=[]
        monkeypatch.setitem(sys.modules,name,m)
    for name in modules:
        if '.' in name:
            parent,child=name.rsplit('.',1)
            if parent in modules:setattr(sys.modules[parent],child,sys.modules[name])
    monkeypatch.setattr('kangaroo_isaac.solver_settings.read_stage_solver_settings',
        lambda stage,cfg: {'status':'API_FIXTURE_ONLY','native_physics_executed':False})
    return art,view,instances


def test_entire_worker_python_path_without_native_physics_claim(monkeypatch,model,ref,tmp_path):
    from kangaroo_isaac.native import run
    art,view,worlds=install_worker_fixture(monkeypatch,model,ref,tmp_path)
    cfg=replace(Config(),duration_s=.0001,push_start_s=0.)
    code=run('nominal',cfg,tmp_path,headless=True,visuals=False)
    # Stationary prescribed arrays cannot validate crouch/lateral movement.
    assert code==3
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['completed'] is True and result['certified_ready'] is False
    assert result['validation']['functional_status']=='FUNCTIONAL_GATES_FAILED'
    assert worlds[0].current_time_step_index==4
    assert len(art.wrenches)==4
    assert len([n for n,_ in art.setters if n=='set_dof_positions'])==1
    assert len([n for n,_ in art.setters if n=='set_root_transforms'])==1
    startup=json.loads((tmp_path/'startup_diagnostics.json').read_text())
    assert startup['release']=='23.0.5'
    assert startup['phase']=='NATIVE_STEPS_COMPLETED'
    assert startup['physics_binding']['view_origin']=='World.physics_sim_view'
    assert startup['command_buffers_share_host_memory'] is True
    assert not (tmp_path/'error.json').exists()


def test_worker_saves_binding_failure_and_does_not_step(monkeypatch,model,ref,tmp_path):
    from kangaroo_isaac.native import run
    art,view,worlds=install_worker_fixture(monkeypatch,model,ref,tmp_path,valid=False)
    code=run('nominal',replace(Config(),duration_s=.0001),tmp_path,headless=True,visuals=False)
    assert code==2
    error=json.loads((tmp_path/'error.json').read_text())
    assert error['failed_phase']=='BINDING_MANAGED_PHYSICS_VIEW'
    assert not error['completed'] and not error['certified_ready']
    assert worlds[0].current_time_step_index==0 and not art.wrenches
    assert not (tmp_path/'result.json').exists()
