"""Isaac Sim 6.1 adapter. All state evolution is native CPU PhysX.

This module is only imported by the clean child process AFTER SimulationApp.
It deliberately does not import Pinocchio, MuJoCo or third-party USD wheels.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .scene import build_scene
from .rigid import RigidTree
from .contacts import ContactObserver
from .geometry import joint_zero_offsets


def numpy(value):
    return np.asarray(value.numpy() if hasattr(value,'numpy') else value,dtype=float)


def index_map(expected, actual):
    if len(actual)!=len(set(actual)) or set(expected)!=set(actual):
        raise RuntimeError(f'DOF name mismatch: expected {expected}; engine returned {actual}')
    return np.asarray([actual.index(name) for name in expected],dtype=np.int32)


class NativePanda:
    def __init__(self, app, root, cmg, reference, mode, case, output, gui, checkpoint):
        import omni.usd
        import omni.timeline
        import omni.kit.app
        from omni.physx import get_physx_interface
        # These extensions must be enabled before importing their Python APIs.
        manager=omni.kit.app.get_app().get_extension_manager()
        for ext in ('isaacsim.core.simulation_manager','isaacsim.core.experimental.prims',
                    'isaacsim.core.rendering_manager'):
            if not manager.set_extension_enabled_immediate(ext,True):
                # Some Kit builds return None on success; inspect actual state.
                if not manager.is_extension_enabled(ext):
                    raise RuntimeError('Could not enable required extension: '+ext)
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        self.app=app;self.cmg=cmg;self.ref=reference;self.mode=mode;self.case=case
        self.dt=case['dt'];self.gui=gui;self.output=Path(output);self.checkpoint=checkpoint
        self.tree=RigidTree(cmg);self.SM=SimulationManager;self.RM=RenderingManager
        self.physx=get_physx_interface();self.timeline=omni.timeline.get_timeline_interface()
        self.physics_errors=[]
        def physics_error(event):
            self.physics_errors.append(dict(type=int(event.type),payload=str(event.payload)))
        self.error_subscription=self.physx.get_error_event_stream().create_subscription_to_pop(physics_error)
        self.timeline.stop()
        omni.usd.get_context().new_stage()
        self.stage=omni.usd.get_context().get_stage()
        if not SimulationManager.switch_physics_engine('physx'):
            # switch can report False for an unavailable backend, not a license
            # to substitute Newton or a custom integrator.
            raise RuntimeError('Isaac Sim could not select the PhysX backend')
        q0=reference.at(0.)[0].copy()
        # q_cmg = q_native + offsets (float32 conditioning only; see geometry.py).
        self.offsets=joint_zero_offsets(cmg,reference.data['q'],case.get('joint_zero_offsets',True))
        self.scene=build_scene(root,self.stage,cmg,q0,mode,case,visuals=gui,joint_offsets=self.offsets)
        checkpoint('USD scene authored',dict(articulation_root=self.scene['root_api'],
            collider_count=self.scene['collision_count'],pad_count=self.scene['pad_count'],
            solver=case['solver'],position_iterations=case['position_iterations'],
            velocity_iterations=case['velocity_iterations'],joint_zero_offsets_rad=self.offsets.tolist()))
        SimulationManager.setup_simulation(dt=self.dt,device='cpu')
        RenderingManager.set_dt(self.dt)
        self.robot=Articulation(self.scene['root_api'])
        self.hand=RigidPrim(self.scene['bodies']['hand'])
        self.object=RigidPrim(self.scene['object']) if self.scene['object'] else None
        self.contacts=ContactObserver(self.scene['collider_labels'],self.dt) if mode=='contact' else None
        if gui:
            ViewportManager.set_camera(self.scene['camera'])
        self.stage.GetRootLayer().Export(str(self.output/'scene_initial.usda'))
        checkpoint('Starting native physics')
        self.timeline.play()
        # One boot update is permitted; benchmark initial conditions are set
        # after setup. Do not manually call initialize_physics a second time.
        app.update()
        if not self.robot.is_physics_tensor_entity_valid():
            raise RuntimeError('The articulation has no native physics tensor entity after timeline startup')
        if not self.hand.is_physics_tensor_entity_valid():
            raise RuntimeError('Native hand rigid-body view did not initialize')
        self.map=index_map(cmg['coordinate_ids'],list(self.robot.dof_names))
        self.robot.set_dof_gains(stiffnesses=np.zeros((1,9),np.float32),dampings=np.zeros((1,9),np.float32))
        self.robot.set_dof_armatures(np.asarray(cmg['armature'],np.float32)[None,:],dof_indices=self.map)
        self.initialized=True
        self.reset_initial()
        self.check_errors()
        checkpoint('Native tensors initialized',dict(engine_dof_names=list(self.robot.dof_names),
            cmg_to_engine_indices=self.map.tolist()))

    def check_errors(self):
        if self.physics_errors:
            raise RuntimeError('Native PhysX error: '+str(self.physics_errors[-1]))
        if self.contacts and self.contacts.error:
            raise RuntimeError(self.contacts.error)

    def state(self):
        q=numpy(self.robot.get_dof_positions())[0,self.map]+self.offsets
        v=numpy(self.robot.get_dof_velocities())[0,self.map]
        if q.shape!=(9,) or v.shape!=(9,) or not np.all(np.isfinite(np.r_[q,v])):
            raise FloatingPointError('Invalid/nonfinite native articulation state')
        return q,v

    def effort(self,tau):
        tau=np.asarray(tau,dtype=np.float32)
        if tau.shape!=(9,) or not np.all(np.isfinite(tau)):
            raise FloatingPointError('Invalid effort command')
        self.robot.set_dof_efforts(tau[None,:],dof_indices=self.map)

    def teleport_for_initialization(self,q,v=None):
        # Called ONLY before native probes/benchmark begin, never for tracking.
        native_q=np.asarray(q,float)-self.offsets
        self.robot.set_dof_positions(np.asarray(native_q,np.float32)[None,:],dof_indices=self.map)
        self.robot.set_dof_velocities(np.asarray(np.zeros(9) if v is None else v,np.float32)[None,:],dof_indices=self.map)

    @staticmethod
    def pose(view):
        p,q=view.get_world_poses()
        p=numpy(p)[0];q=numpy(q)[0]
        if not np.all(np.isfinite(np.r_[p,q])) or abs(np.linalg.norm(q)-1.)>1e-3:
            raise FloatingPointError('Invalid native body pose')
        R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        return p,R

    def tool_pose(self):
        p,R=self.pose(self.hand)
        T=np.asarray(self.cmg['tool']['T_body_tool'])
        return p+R@T[:3,3],R@T[:3,:3]

    def apply_wrench(self,w):
        w=np.asarray(w,dtype=np.float32)
        if not np.any(w):
            return  # PhysX resets nonpersistent external wrenches every step.
        if self.mode=='contact':
            p,R=self.pose(self.object)
            position=p+R@np.array([.003,0,0])  # actual authored COM, not body origin
            target=self.object
        else:
            position,_=self.tool_pose()
            target=self.hand
        target.apply_forces_and_torques_at_pos(forces=w[None,:3],torques=w[None,3:],
            positions=np.asarray(position,np.float32)[None,:],local_frame=False)

    def step(self):
        if self.contacts:self.contacts.clear()
        self.SM.step(steps=1,update_fabric=False)
        self.check_errors()

    def render(self):
        # Renderer cannot introduce another uncontrolled physics step.
        self.physx.update_transformations(False,True,False,False)
        self.RM.render()

    def reset_initial(self):
        q=self.ref.at(0.)[0].copy()
        if self.case['initial_offset']:
            q[:7]+=np.array([1.,-.6,.4,.7,-.5,.3,-.4])*.001
        self.teleport_for_initialization(q)
        self.effort(np.zeros(9))
        if self.object:
            self.object.set_world_poses(positions=np.array([[.45,-.20+self.case['offset'],.0702]],np.float32),
                                        orientations=np.array([[1,0,0,0]],np.float32))
            self.object.set_velocities(linear_velocities=np.zeros((1,3),np.float32),
                                       angular_velocities=np.zeros((1,3),np.float32))
        if self.contacts:self.contacts.clear()

    def mechanics(self):
        """Finite native gates; no simulated result is taken from stored NPZ."""
        report=dict(passed=False,checks={},details={},scope='Native finite-pose mechanics and asymmetric finger probe; not a full-task PASS')
        def gate(name,value,limit,relation='<='):
            value=float(value);ok=np.isfinite(value) and (value<=limit if relation=='<=' else value>=limit)
            report['checks'][name]=dict(passed=bool(ok),value=value if np.isfinite(value) else None,limit=limit,relation=relation)
        q0=self.ref.at(0.)[0].copy();q0[7:]=.02
        self.teleport_for_initialization(q0)
        self.effort(self.tree.rnea(q0,np.zeros(9),np.zeros(9)))
        self.step()
        q,v=self.state();p,R=self.tool_pose();T=self.tree.tool(q)
        gate('native_tool_translation_m',np.linalg.norm(p-T[:3,3]),2e-5)
        gate('native_tool_rotation_rad',Rotation.from_matrix(T[:3,:3].T@R).magnitude(),2e-4)
        arm=numpy(self.robot.get_dof_armatures())[0,self.map]
        gate('native_armature_readback',np.max(abs(arm-np.asarray(self.cmg['armature']))),1e-6)
        gains=self.robot.get_dof_gains()
        gate('zero_native_drive_gains',max(np.max(abs(numpy(x))) for x in gains),1e-12)
        # Set armature to zero for a *rigid-body* mass-matrix comparison. This
        # avoids guessing whether this SDK version includes it in this getter.
        self.robot.set_dof_armatures(np.zeros((1,9),np.float32))
        self.effort(self.tree.rnea(q,v,np.zeros(9)));self.step()
        q,v=self.state()
        Mnative=numpy(self.robot.get_mass_matrices())[0]
        if Mnative.shape!=(9,9):
            raise RuntimeError(f'Expected a fixed-base 9x9 mass matrix, got {Mnative.shape}')
        Mnative=Mnative[np.ix_(self.map,self.map)]
        M=self.tree.mass(q)
        gate('rigid_mass_matrix_relative_error',np.linalg.norm(Mnative-M)/np.linalg.norm(M),.003)
        gate('native_mass_matrix_min_eigenvalue',np.linalg.eigvalsh((Mnative+Mnative.T)/2).min(),1e-7,'>=')
        report['details']['mass_comparison']='Armature explicitly zero during this check; restored before probes and task.'
        self.robot.set_dof_armatures(np.asarray(self.cmg['armature'],np.float32)[None,:],dof_indices=self.map)
        # Independent acceleration probes check force direction, rotor inertia,
        # coupling and all physical coordinates at a configuration free of contact.
        S=np.zeros((9,8));S[:8,:8]=np.eye(8);S[8,7]=1.
        maxacc=0.;probe=[]
        for k in range(8):
            self.teleport_for_initialization(q0)
            self.effort(self.tree.rnea(q0,np.zeros(9),np.zeros(9)));self.step()
            q,v=self.state()
            ades=S[:,k]*(.3 if k<7 else .2)
            force=self.tree.rnea(q,v,ades)+np.asarray(self.cmg['armature'])*ades
            self.effort(force);self.step()
            _,v1=self.state();actual=(v1-v)/self.dt
            err=float(np.max(abs(actual-ades)));maxacc=max(maxacc,err)
            probe.append(dict(independent_coordinate=k,expected=ades.tolist(),measured=actual.tolist()))
        report['details']['acceleration_probes']=probe
        gate('native_acceleration_probe_max_error_SI',maxacc,.05)
        # A one-sided load must move BOTH fingers. Equal position commands
        # alone could let a missing mimic constraint pass unnoticed.
        self.teleport_for_initialization(q0)
        for _ in range(max(2,round(.020/self.dt))):
            q,v=self.state()
            force=self.tree.rnea(q,v,np.zeros(9))
            force[:7]+=300.*(q0[:7]-q[:7])-30.*v[:7]
            force[7]+=1.
            self.effort(force);self.step()
        q,v=self.state()
        gate('one_sided_load_finger_coupling_m',abs(q[7]-q[8]),2e-5)
        gate('one_sided_load_follower_displacement_m',q[8]-q0[8],1e-4,'>=')
        # Open-loop gravity compensation: no position drive hides a bad model.
        self.teleport_for_initialization(q0)
        for _ in range(max(2,round(.020/self.dt))):
            q,v=self.state()
            self.effort(self.tree.rnea(q,v,np.zeros(9))-np.asarray(self.cmg['damping'])*v)
            self.step()
        q,v=self.state()
        gate('gravity_hold_position_drift',np.max(abs(q-q0)),2e-5)
        gate('gravity_hold_max_speed',np.max(abs(v)),.002)
        # Low-speed integration probe. PhysX TGS advances articulation joint
        # angles once per position iteration in float32. If that increment is
        # below half a float32 spacing the joint stalls while PhysX still
        # reports the velocity (first native suite: 64 iterations, joints 4/6
        # stuck below ~0.03 rad/s). Coast at 0.02 rad/s with exact gravity and
        # Coriolis compensation and compare the displacement with the reported
        # velocities. The old setting fails this probe; the declared one passes.
        vslow=np.zeros(9);vslow[:7]=.02
        self.teleport_for_initialization(q0,vslow)
        qa,_=self.state();integrated=np.zeros(9)
        for _ in range(max(2,round(.05/self.dt))):
            q,v=self.state()
            self.effort(self.tree.rnea(q,v,np.zeros(9)));self.step()
            _,v1=self.state();integrated+=.5*(v+v1)*self.dt
        qb,_=self.state();moved=qb-qa
        relative=abs(moved[:7]-integrated[:7])/np.maximum(abs(integrated[:7]),1e-12)
        report['details']['slow_motion_probe']=dict(speed_rad_s=.02,duration_s=.05,
            displacement_rad=moved[:7].tolist(),integrated_velocity_rad=integrated[:7].tolist(),
            relative_error=relative.tolist())
        gate('slow_motion_position_integration_relative_error',np.max(relative),.05)
        report['details']['solver']=dict(type=self.case['solver'],position_iterations=self.case['position_iterations'],
            velocity_iterations=self.case['velocity_iterations'],dt_s=self.dt)
        report['details']['joint_zero_offsets_rad']=self.offsets.tolist()
        gate('native_physics_errors',len(self.physics_errors),0.)
        report['passed']=all(x['passed'] for x in report['checks'].values())
        self.reset_initial()
        return report
