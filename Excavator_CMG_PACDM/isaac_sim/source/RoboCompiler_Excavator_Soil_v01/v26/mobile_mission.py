"""Reference mission and bounded feedback for the articulated-track plant.

Only actuator commands are produced. No robot or material state is prescribed.
References are expressed about the original source world frame.
"""
from project import *
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from digging_export import export_scene, fmt
from digging_path import make_path, reference_at, LIP, HEEL
from track_model import add_tracks, TrackConfig
from track_servo import ShaftSpeedPI

STAGES = ['settle', 'approach pile', 'dig and lift', 'stabilize loaded bucket',
          'loaded reverse travel', 'loaded turn', 'position closed bucket',
          'open above receiver', 'settle delivered material', 'close bucket',
          'retract bucket', 'return turn', 'return travel', 'stop']
BOUNDARIES = np.array([0., 1., 6., 14., 16., 32., 39., 45., 51., 54., 60., 64., 71., 87., 90.])

def smooth(t, start, end):
    s = np.clip((t-start)/(end-start), 0., 1.)
    f = s**3*(10-15*s+6*s*s)
    df = 30*s*s*(1-s)**2/(end-start)
    ddf = 60*s*(1-3*s+2*s*s)/(end-start)**2
    return f, df, ddf

class Mission:
    def __init__(self, engine, reference, approach=.5, distance=2.0, yaw=.30):
        self.arm_path = make_path(engine, reference)
        self.approach = approach
        self.distance = distance
        self.yaw = yaw
        self.duration = float(BOUNDARIES[-1])
        self.departure = 16.
        self.loaded_arrival = 32.
        self.initial_tree = self.arm_path['closed_tree'][0].copy()
        self.transport = self.arm_path['independent'][5].copy()
        self.mouth_local = .5*(LIP+HEEL)
        active=np.array(engine.active)
        base=reference.nominal.copy()
        def mouth_observation(x):
            q=base.copy();q[active[1:4]]=x
            T=engine.source.evaluate(q,np.zeros(len(q)),[0,0,-9.81])['poses']['body_56']
            lip=T[:3,:3]@LIP+T[:3,3];heel=T[:3,:3]@HEEL+T[:3,3]
            mouth=T[:3,:3]@self.mouth_local+T[:3,3]
            return np.array([mouth[0],mouth[2],np.arctan2(lip[2]-heel[2],lip[0]-heel[0])])
        # Register a reachable receiver from the closed transport geometry,
        # before any material simulation. The former far endpoint required
        # opening while still travelling and could not be reached fully curled
        # within the unchanged joint guards.
        mouth_target=mouth_observation(self.transport[1:4])
        lower=base[active[1:4]]-1.49;upper=base[active[1:4]]+1.49
        parameters=np.linspace(0.,1.,41);curve=[];seed=self.transport[1:4].copy()
        for fraction in parameters:
            goal=mouth_target.copy();goal[2]=mouth_target[2]+fraction*(-.9-mouth_target[2])
            fit=least_squares(lambda x:mouth_observation(x)-goal,seed,bounds=(lower,upper),
                              xtol=1e-13,ftol=1e-13,gtol=1e-13)
            residual=float(np.max(abs(mouth_observation(fit.x)-goal)))
            if not fit.success or residual>1e-8:
                raise ValueError('Fixed-mouth receiver path is infeasible within source joint guards')
            u=self.transport.copy();u[0]+=.36;u[1:4]=fit.x;curve.append(u);seed=fit.x
        self.unload_curve=CubicSpline(parameters,np.asarray(curve),axis=0,extrapolate=False)
        self.receiver_closed=np.array(curve[0]);self.dump=np.array(curve[-1])
        max_residual=0.;minimum_margin=float('inf')
        for fraction in np.linspace(0.,1.,161):
            u=self.unload_curve(fraction);goal=mouth_target.copy()
            goal[2]=mouth_target[2]+fraction*(-.9-mouth_target[2])
            max_residual=max(max_residual,float(np.max(abs(mouth_observation(u[1:4])-goal))))
            minimum_margin=min(minimum_margin,float(np.min(np.r_[u[1:4]-lower,upper-u[1:4]])))
        if max_residual>2e-6 or minimum_margin<=0:
            raise ValueError('Interpolated receiver path violates geometric accuracy or joint guards')
        closed=reference.reconstruct(self.receiver_closed)
        T=engine.source.evaluate(closed,np.zeros(len(closed)),[0,0,-9.81])['poses']['body_56']
        source_mouth=T[:3,:3]@self.mouth_local+T[:3,3]
        self.depot_center=Rotation.from_euler('z',yaw).apply(source_mouth)+np.array([-distance,0,0])
        self.depot_half_size=np.array([1.15,1.15])
        self.receiver_plan=dict(registration='Vertical projection of fixed bucket mouth center, from feasible closed transport pose and commanded slew/base pose before any material simulation',
            mouth_local_m=self.mouth_local,source_unslewed_mouth_xz_m=mouth_target[:2],
            source_mouth_world_m=source_mouth,opening_pitch_rad=[float(mouth_target[2]),-.9],
            minimum_independent_guard_margin_rad=minimum_margin,maximum_interpolated_position_pitch_residual_SI=max_residual,
            closed_independent=self.receiver_closed,open_independent=self.dump,
            geometry='Closed positioning; fixed-mouth opening; settling; fixed-mouth reclosing; retraction')

    def _open_reference(self,t,start,end,reverse=False):
        f,df,ddf=smooth(t,start,end)
        if reverse:f,df,ddf=1-f,-df,-ddf
        q=self.unload_curve(f);dq=self.unload_curve(f,1);ddq=self.unload_curve(f,2)
        return q,dq*df,ddq*df**2+dq*ddf

    def stage(self,t):
        return min(int(np.searchsorted(BOUNDARIES, t, side='right')-1), len(STAGES)-1)

    def reference(self,t):
        # Source dig schedule is shifted six seconds; its first eight seconds
        # finish with material lifted and the original bucket curl retained.
        if t <= 6:
            arm = (self.arm_path['independent'][0].copy(), np.zeros(7), np.zeros(7))
        elif t <= 14:
            arm = reference_at(t-6, self.arm_path)
        elif t <= 39:
            arm = (self.transport.copy(), np.zeros(7), np.zeros(7))
        elif t <= 45:
            f,df,ddf=smooth(t,39,45);delta=self.receiver_closed-self.transport
            arm=(self.transport+f*delta,df*delta,ddf*delta)
        elif t <= 51:
            arm=self._open_reference(t,45,51)
        elif t <= 54:
            arm=(self.dump.copy(),np.zeros(7),np.zeros(7))
        elif t <= 60:
            arm=self._open_reference(t,54,60,reverse=True)
        elif t <= 64:
            f,df,ddf=smooth(t,60,64);delta=self.transport-self.receiver_closed
            arm=(self.receiver_closed+f*delta,df*delta,ddf*delta)
        else:
            arm=(self.transport.copy(),np.zeros(7),np.zeros(7))
        pose=np.array([-self.approach,0.,0.]);rate=np.zeros(3)
        if 1<t<=6:
            f,df,_=smooth(t,1,6);pose[0]+=self.approach*f;rate[0]=self.approach*df
        elif 6<t<=16:
            pose[:]=0
        elif 16<t<=32:
            f,df,_=smooth(t,16,32);pose[:]=[-self.distance*f,0,0];rate[0]=-self.distance*df
        elif 32<t<=39:
            f,df,_=smooth(t,32,39);pose[:]=[-self.distance,0,self.yaw*f];rate[2]=self.yaw*df
        elif 39<t<=64:
            pose[:]=[-self.distance,0,self.yaw]
        elif 64<t<=71:
            f,df,_=smooth(t,64,71);pose[:]=[-self.distance,0,self.yaw*(1-f)];rate[2]=-self.yaw*df
        elif 71<t<=87:
            f,df,_=smooth(t,71,87);pose[:]=[-self.distance*(1-f),0,0];rate[0]=self.distance*df
        elif t>87:
            pose[:]=0
        return arm,pose,rate

def build_scene(path, mission, *, dt=.0005, soil=True, bucket_contact=True,
                traction=.8, density_scale=1., track_config=None):
    meta=export_scene(path,dt=dt,soil=soil,bucket_contact=bucket_contact,density_scale=density_scale)
    root=ET.parse(path).getroot()
    root.set('model','Excavator_articulated_tracks_v26')
    # Newton may stop on small objective improvement before its force residual
    # meets the independent acceptance gate. Keep that gate unchanged and solve
    # the native contact/equality system more accurately.
    root.find('option').set('tolerance','1e-14')
    cfg=TrackConfig(**(track_config or {}))
    cfg.ground_friction=traction
    # Torque is supplied by explicit hydraulic dynamics. No native clamp may
    # silently change the effort used in the energy ledger.
    cfg.torque_limit=30000.
    meta['tracks']=add_tracks(root,cfg)
    world=root.find('worldbody')
    for body in world.findall('body'):
        if body.get('name')=='body_53' or body.get('name','').startswith('track_'):
            p=np.fromstring(body.get('pos','0 0 0'),sep=' ')
            p[0]-=mission.approach
            body.set('pos',fmt(p))
    # Hide the old static belt mesh while retaining all source inertial data.
    old=root.find(".//geom[@name='cad_body_53']")
    if old is not None:
        old.set('rgba','.25 .28 .30 0')
    base=root.find(".//body[@name='body_53']")
    ET.SubElement(base,'geom',name='chassis_visual',type='box',pos='-.453 2.1034 .40',size='.42 1.1 .18',mass='0',contype='0',conaffinity='0',rgba='.26 .30 .34 1')
    center=mission.depot_center
    # Thin visual target: deposition is measured by particle identity and
    # final position, not by a hidden attracting force or attached payload.
    ET.SubElement(world,'geom',name='depot_marker',type='box',pos=fmt([center[0],center[1],.002]),size=fmt([*mission.depot_half_size,.002]),mass='0',contype='0',conaffinity='0',rgba='.22 .47 .43 1')
    # A passive receiving bay retains delivered aggregate by ordinary native
    # contact. Its four walls enclose the pre-registered 2.30 m square; the
    # existing ground plane is the floor. No payload forces or attachments.
    wall_height=.40;wall_thickness=.10
    hx,hy=mission.depot_half_size
    walls=[('west',[-hx-wall_thickness/2,0],[wall_thickness/2,hy+wall_thickness,wall_height/2]),
           ('east',[hx+wall_thickness/2,0],[wall_thickness/2,hy+wall_thickness,wall_height/2]),
           ('south',[0,-hy-wall_thickness/2],[hx,wall_thickness/2,wall_height/2]),
           ('north',[0,hy+wall_thickness/2],[hx,wall_thickness/2,wall_height/2])]
    for label,offset,size in walls:
        ET.SubElement(world,'geom',name='receiver_wall_'+label,type='box',
                      pos=fmt([center[0]+offset[0],center[1]+offset[1],wall_height/2]),size=fmt(size),
                      contype='4',conaffinity='43',condim='6',friction='.65 .003 .008',
                      rgba='.30 .37 .38 1')
    meta['receiver_plan']=mission.receiver_plan
    meta['receiving_bay']=dict(native_contact=True,kinematic_or_payload_forcing=False,
        inner_half_size_m=mission.depot_half_size,wall_height_m=wall_height,wall_thickness_m=wall_thickness,
        wall_friction=[.65,.003,.008],floor='Existing native ground plane',
        description='Fixed open-top receiving bay registered from the feasible bucket mouth path before material tests')
    meta.update(original_inertias_modified=True, track_belt_dynamics=True,
                travel_controller=TravelController.specification(),
                receiver_plan=mission.receiver_plan,
                collision_semantics='Explicit articulated shoe-ground, shoe-roller, pin-tooth, guide-flange and nonadjacent shoe contact; native source arm, bucket and rigid aggregate contact.',
                original_undercarriage_inertia_partitioned=True,
                terrain_calibrated=False, scene_reference_translation_m=[-mission.approach,0,0],
                task_times_s=dict(departure=mission.departure,loaded_arrival=mission.loaded_arrival,end=mission.duration),
                depot_center_m=center.tolist(),depot_half_size_m=mission.depot_half_size.tolist())
    ET.indent(root,space='  ')
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return meta

class TravelController:
    """Bounded reference tracking and PI sprocket effort.

    Lateral steering scales with signed reference travel speed and vanishes
    during a stationary pivot. A tracked vehicle's load-dependent instantaneous
    centre of rotation need not coincide with this chassis marker; feeding that
    pivot displacement through sign(commanded_speed) caused positive feedback.
    Measured yaw-rate damping brakes the finite-inertia native plant. Neither
    the reference nor this controller prescribes any mechanical state.
    """
    REVISION = 'speed_scheduled_lateral_yaw_damping_multirate_v2'

    @classmethod
    def specification(cls):
        return dict(revision=cls.REVISION, outer_reference_period_s=.01,
                    inner_effort_period_s=.001,
                    longitudinal_gain_s_inv=.65, heading_gain_s_inv=1.3,
                    lateral_gain_m_inv_squared=3.0, yaw_rate_damping=.8,
                    shaft_Kp_Nm_s_rad=6500., shaft_Ki_Nm_rad=4500.,
                    integral_bound_rad=2.5, command_torque_bound_Nm=20000.,
                    forward_speed_bound_m_s=.30, yaw_rate_bound_rad_s=.14,
                    numerical_sample_rates_are_not_realtime_certification=True)

    def __init__(self,m,d,meta):
        self.m=m;self.base=m.body('body_53').id
        self.jids=np.array([m.joint(s).id for s in meta['tracks']['drive_joint_names']])
        self.vi=m.jnt_dofadr[self.jids]
        self.ai=np.array([m.actuator(s).id for s in meta['tracks']['drive_actuator_names']])
        self.radius=meta['tracks']['sprocket_pitch_radius_m']
        self.gauge=meta['tracks']['source_frame_left_x_m']-meta['tracks']['source_frame_right_x_m']
        self.shaft_servo=ShaftSpeedPI()
        # Source horizontal COM marker is insensitive to the inertial partition.
        self.marker=np.array(meta['tracks']['source_undercarriage_com_m'])
        self.reference_xy=np.array([2.1126661182113513, .45390271998685244])+np.array([-self.marker[1],self.marker[0]])
        self.last={}

    def pose(self,d):
        R=d.xmat[self.base].reshape(3,3)
        center=d.xpos[self.base]+R@self.marker
        forward=-R[:,1]
        yaw=np.arctan2(forward[1],forward[0])
        return np.r_[center[:2]-self.reference_xy,yaw]

    def evaluate(self,d,target,rate,dt,*,update_outer=True):
        """Update bounded shaft effort at the fixed 1 kHz simulated rate.

        The caller passes dt=.001 and sets ``update_outer`` on the 100 Hz
        pose/reference ticks. Between those ticks desired shaft speeds are held
        while current shaft velocities drive the PI regulator. The same control
        periods apply to every physics timestep used in the convergence study.
        This avoids sample-delay limit cycles when tooth contact unloads the
        low-inertia sprocket and rotor; it is not a real-time performance measure.
        """
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('Travel controller dt must be positive and finite')
        if update_outer or not self.last:
            pose=self.pose(d);yaw=pose[2]
            f=np.array([np.cos(yaw),np.sin(yaw)])
            lateral=np.array([-np.sin(yaw),np.cos(yaw)])
            error=target[:2]-pose[:2]
            heading=np.arctan2(np.sin(target[2]-yaw),np.cos(target[2]-yaw))
            reference_forward=np.array([np.cos(target[2]),np.sin(target[2])])
            reference_speed=float(rate[:2]@reference_forward)
            # cvel stores world-frame angular velocity first. Project its effect
            # on actual forward direction to obtain yaw rate even with base tilt.
            forward_3d=-d.xmat[self.base].reshape(3,3)[:,1]
            forward_dot=np.cross(d.cvel[self.base,:3],forward_3d)
            yaw_rate=float((forward_3d[0]*forward_dot[1]-forward_3d[1]*forward_dot[0]) /
                           max(1e-8,forward_3d[0]**2+forward_3d[1]**2))
            v=np.clip(rate[:2]@f+.65*(error@f),-.30,.30)
            lateral_steering=3.0*reference_speed*float(error@lateral)
            yaw_damping=.8*(float(rate[2])-yaw_rate)
            omega=np.clip(rate[2]+1.3*heading+lateral_steering+yaw_damping,-.14,.14)
            desired=np.array([v-.5*self.gauge*omega,v+.5*self.gauge*omega])/self.radius
            self.last=dict(pose=pose,target=target.copy(),desired_speed=desired,
                           error=error,heading_error=heading,reference_speed=reference_speed,
                           yaw_rate=yaw_rate,lateral_steering=lateral_steering,yaw_damping=yaw_damping)
        desired=self.last['desired_speed']
        speed=d.qvel[self.vi]
        limited=self.shaft_servo.evaluate(desired,speed,dt)
        self.last.update(speed=speed.copy(),command=limited.copy())
        return limited
