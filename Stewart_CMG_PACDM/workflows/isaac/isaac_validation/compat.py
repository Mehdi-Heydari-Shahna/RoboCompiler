"""Isaac lifecycle adapter. Import this only after SimulationApp starts.

6.1: use the SimulationManager-owned view after USD attachment, not an
independent unbound tensor view. 4.5/5.x: retain the SimulationContext route.
The 6.1 integration is documentation-based and must be smoke-tested on Isaac.
"""
from __future__ import annotations
import numpy as np


def as_numpy(value):
    if hasattr(value,'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value,'numpy'):
        return value.numpy()
    return np.asarray(value)


class BodyView:
    """Convert only at the tensor boundary; preserve native frontend/device.

    Copies on reads prevent a later backend fetch from overwriting logged state.
    Validate before native calls, and retain write buffers until the next call.
    """
    def __init__(self,raw):
        self.raw=raw; self.prim_paths=list(raw.prim_paths)
        sample=raw.get_transforms(); module=type(sample).__module__
        self.frontend='warp' if module.startswith('warp') else 'torch' if module.startswith('torch') else 'numpy'
        self.device=getattr(sample,'device',None)
        self._write_buffers={}

    def _array(self,value,integer=False):
        value=np.ascontiguousarray(value,dtype=np.int32 if integer else np.float32)
        if self.frontend=='warp':
            import warp as wp
            return wp.array(value,dtype=wp.int32 if integer else wp.float32,device=self.device)
        if self.frontend=='torch':
            import torch
            return torch.as_tensor(value,dtype=torch.int32 if integer else torch.float32,device=self.device)
        return value

    def _indices(self,indices):
        a=np.asarray(indices)
        if a.ndim!=1 or a.dtype.kind not in 'iu' or not len(a):
            raise ValueError('Body indices must be a nonempty 1D integer array')
        if np.any(a<0) or np.any(a>=len(self.prim_paths)) or len(np.unique(a))!=len(a):
            raise ValueError('Body indices must be unique and in range')
        return self._array(a,True)

    def _values(self,value,width):
        a=np.asarray(value,dtype=float)
        if a.shape!=(len(self.prim_paths),width) or not np.all(np.isfinite(a)):
            raise ValueError(f'Native body values must be finite with shape {(len(self.prim_paths),width)}')
        # Reject overflow at the float32 native boundary, not after applying it.
        if np.max(abs(a))>np.finfo(np.float32).max:
            raise ValueError('Native body values overflow float32')
        return self._array(a)

    @staticmethod
    def _accepted(result,operation):
        # Some wrappers return None on success, others Python/NumPy bool.
        if isinstance(result,(bool,np.bool_)) and not bool(result):
            raise RuntimeError('Native rigid-body '+operation+' failed')
        return result

    def get_transforms(self):return as_numpy(self.raw.get_transforms()).copy()
    def get_velocities(self):return as_numpy(self.raw.get_velocities()).copy()
    def get_coms(self):return as_numpy(self.raw.get_coms()).copy()
    def get_masses(self):return as_numpy(self.raw.get_masses()).copy()
    def get_inertias(self):return as_numpy(self.raw.get_inertias()).copy()

    def set_transforms(self,value,indices):
        a=np.asarray(value,dtype=float)
        if a.shape!=(len(self.prim_paths),7) or not np.all(np.isfinite(a)):
            raise ValueError('Invalid native transform array')
        if np.max(abs(np.linalg.norm(a[:,3:],axis=1)-1))>1e-5:
            raise ValueError('Invalid native transform quaternion norm')
        buffers=(self._values(a,7),self._indices(indices))
        self._write_buffers['transforms']=buffers
        return self._accepted(self.raw.set_transforms(*buffers),'transform assignment')

    def set_velocities(self,value,indices):
        buffers=(self._values(value,6),self._indices(indices))
        self._write_buffers['velocities']=buffers
        return self._accepted(self.raw.set_velocities(*buffers),'velocity assignment')

    def apply_forces_and_torques_at_position(self,forces,torques,positions,indices,is_global):
        if not isinstance(is_global,(bool,np.bool_)):
            raise ValueError('is_global must be a boolean')
        buffers=(self._values(forces,3),self._values(torques,3),
                 self._values(positions,3),self._indices(indices))
        self._write_buffers['wrenches']=buffers
        return self._accepted(self.raw.apply_forces_and_torques_at_position(
            *buffers,bool(is_global)),'force application')


class ManagerContext:
    def __init__(self,app,manager):self.app=app;self.manager=manager
    @property
    def current_time(self):return float(self.manager.get_simulation_time())
    @property
    def current_time_step_index(self):return int(self.manager.get_num_physics_steps())
    def step(self,render=False):
        self.manager.step(steps=1,update_fabric=bool(self.manager.is_fabric_enabled()))
        if render:self.render()
    def render(self):
        import carb
        settings=carb.settings.get_settings(); key='/app/player/playSimulations'
        previous=settings.get(key); before=(self.current_time,self.current_time_step_index)
        try:
            settings.set_bool(key,False)
            self.app.update()
        finally:
            # This setting is normally True in Kit. Restore its previous value.
            settings.set_bool(key,True if previous is None else bool(previous))
        if before!=(self.current_time,self.current_time_step_index):
            raise RuntimeError('Render-only application update advanced the physics clock')
    def stop(self):
        import omni.timeline
        omni.timeline.get_timeline_interface().stop()
        self.manager.invalidate_physics()


def start_physics(app,dt):
    """Return a context and an already-initialized native simulation view."""
    import omni.timeline
    timeline=omni.timeline.get_timeline_interface()
    # Flush stage-open events while stopped. There is no measured rollout yet.
    timeline.stop(); app.update()
    try:
        from isaacsim.core.simulation_manager import SimulationManager as manager
    except ImportError:
        manager=None
    if manager is not None and all(hasattr(manager,k) for k in
            ('setup_simulation','get_physics_simulation_view','step','get_simulation_time')):
        active=manager.get_active_physics_engine()
        label=str(getattr(active,'value',active)).lower()
        if 'physx' not in label:
            if not manager.switch_physics_engine('physx'):
                raise RuntimeError(f'Cannot activate PhysX, active engine is {active}')
        manager.setup_simulation(dt=dt,device='cpu')
        if manager.get_physics_simulation_view() is None:
            manager.initialize_physics()
        view=manager.get_physics_simulation_view()
        if view is None:
            raise RuntimeError('SimulationManager did not create its attached PhysX simulation view')
        context=ManagerContext(app,manager)
        route='SimulationManager.setup_simulation/initialize_physics/get_physics_simulation_view'
    else:
        try:
            from isaacsim.core.api import SimulationContext
        except ImportError:
            from omni.isaac.core import SimulationContext
        context=SimulationContext(physics_dt=dt,rendering_dt=dt,stage_units_in_meters=1.,
            physics_prim_path='/World/PhysicsScene',backend='numpy',device='cpu',set_defaults=False)
        context.initialize_physics();context.play();app.update()
        view=getattr(context,'physics_sim_view',None)
        if view is None and manager is not None and hasattr(manager,'get_physics_sim_view'):
            view=manager.get_physics_sim_view()
        if view is None:
            import omni.physics.tensors as tensors
            view=tensors.create_simulation_view('numpy')
        route='legacy SimulationContext attached tensor view'
    view.set_subspace_roots('/')
    return context,view,route
