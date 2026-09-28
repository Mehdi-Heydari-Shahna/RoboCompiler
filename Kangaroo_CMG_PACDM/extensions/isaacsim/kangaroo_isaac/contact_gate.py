"""Planar contact-envelope estimate and causal 1-kHz controller telemetry.

This is NOT a native contact-count sensor or a contact-force/wrench correction.
The source MuJoCo task gates feedforward on d.ncon, not on force magnitude.
For this port's two box soles and fixed ground box, the new gate uses measured
sole corners and the sum of the existing collider contact offsets. It is a
stateless geometric approximation to contact presence, not an exact manifold
count. It can enable feedforward before positive load develops, inside the
contact envelope. It disables it when both soles leave that envelope.

Scope: at least one whole sole's XY footprint lies on the horizontal ground.
Partial overlap at a ground edge is deliberately NOT inferred as support.
No additional force, pose correction, smoothing, gain or threshold relaxation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .model import finite
from .scene_paths import GROUND_CENTER_M, GROUND_SIZE_M
from .io_utils import write_json


@dataclass(frozen=True)
class ContactDecision:
    selected: bool
    force_gate: bool
    envelope_gate: bool
    foot_in_envelope: np.ndarray
    foot_inside_floor_xy: np.ndarray
    sole_min_height_m: np.ndarray
    normal_force_N: np.ndarray


class ContactGate:
    def __init__(self,cfg):
        self.cfg=cfg.validate()
        self.center=np.asarray(GROUND_CENTER_M,dtype=float)
        self.half=np.asarray(GROUND_SIZE_M,dtype=float)/2
        self.top=float(self.center[2]+self.half[2])
        self.bottom=float(self.center[2]-self.half[2])
        # Same contact_offset is authored on BOTH the floor and sole colliders.
        self.contact_distance=2*cfg.contact_offset_m

    def evaluate(self,sole_points_world,foot_force_world):
        p=finite(sole_points_world,(2,8,3),'measured sole corners')
        force=finite(foot_force_world,(2,3),'measured foot force')
        height=p[:,:,2].min(axis=1)-self.top
        # This estimator is not valid for partial-edge or side-wall contact.
        inside=np.all(np.abs(p[:,:,:2]-self.center[:2])<=self.half[:2],axis=(1,2))
        above_bottom=p[:,:,2].max(axis=1)>=self.bottom
        near=(height<=self.contact_distance)&inside&above_bottom
        force_gate=bool(force[:,2].sum()>1.)
        envelope_gate=bool(near.any())
        selected=envelope_gate if self.cfg.feedforward_contact_mode=='contact_envelope' else force_gate
        if self.cfg.no_contact:selected=False
        return ContactDecision(bool(selected),force_gate,envelope_gate,near.copy(),inside.copy(),
                               height.copy(),force[:,2].copy())

    def description(self):
        return {'mode':self.cfg.feedforward_contact_mode,
            'measurement':'sole geometry from native body poses; NOT a native contact-count sensor',
            'contact_distance_m':self.contact_distance,
            'contact_distance_origin':'sum of the unchanged floor and sole contact offsets',
            'ground_center_m':list(GROUND_CENTER_M),'ground_size_m':list(GROUND_SIZE_M),
            'force_threshold_N_legacy_only':1.,
            'scope':'fixed horizontal ground; full XY footprint of at least one box sole within the ground',
            'stateful_latch':False,'force_injection':False,'native_manifold_equivalence_verified':False,
            'preload_note':'geometry mode may enable feedforward within the envelope before positive load develops'}


class ControllerTrace:
    """Record actual causal inputs/commands, not a decimated post-step guess.

    time_s is t_k before world.step. foot_normal_N is the impulse/dt read back
    from the preceding physics step. Initial force sample is explicitly zero.
    submitted_motor_force_N is the float32 effort held over the upcoming step;
    the actuator filter is advanced after that step, as in the previous port.
    """
    def __init__(self,cfg):
        self.cfg=cfg.validate();self.stride=round(cfg.control_period_s/cfg.dt_s)
        nsteps=round(cfg.duration_s/cfg.dt_s)
        self.capacity=(nsteps+self.stride-1)//self.stride
        self.count=0
        shapes={'step':(), 'time_s':(), 'contact_selected':(), 'legacy_force_gate':(),
            'envelope_gate':(), 'feedforward_enabled':(), 'foot_in_envelope':(2,),
            'foot_inside_floor_xy':(2,), 'sole_min_height_m':(2,), 'foot_normal_N':(2,),
            'motor_q_m':(12,), 'motor_qd_m_s':(12,), 'reference_q_m':(12,),
            'reference_qd_m_s':(12,), 'feedforward_N':(12,), 'raw_command_N':(12,),
            'command_N':(12,), 'submitted_motor_force_N':(12,)}
        bools={'contact_selected','legacy_force_gate','envelope_gate','feedforward_enabled',
               'foot_in_envelope','foot_inside_floor_xy'}
        self.data={k:np.empty((self.capacity,)+shape,dtype=np.int64 if k=='step' else
                    np.bool_ if k in bools else np.float64) for k,shape in shapes.items()}

    def record(self,k,contact,q,qd,controller):
        if not isinstance(k,(int,np.integer)) or k!=self.count*self.stride:
            raise ValueError('Controller trace must contain every causal control tick exactly once')
        if self.count>=self.capacity:raise ValueError('Controller trace capacity exceeded')
        if controller.update_count!=self.count+1:raise ValueError('Record only after the matching controller update')
        i=self.count
        enabled=bool(contact.selected and not (self.cfg.no_feedforward or self.cfg.passive))
        ref_i=min(i,len(controller.targets)-1)
        values={'step':k,'time_s':k*self.cfg.dt_s,'contact_selected':contact.selected,
            'legacy_force_gate':contact.force_gate,'envelope_gate':contact.envelope_gate,
            'feedforward_enabled':enabled,'foot_in_envelope':contact.foot_in_envelope,
            'foot_inside_floor_xy':contact.foot_inside_floor_xy,
            'sole_min_height_m':contact.sole_min_height_m,'foot_normal_N':contact.normal_force_N,
            'motor_q_m':q,'motor_qd_m_s':qd,'reference_q_m':controller.targets[ref_i],
            'reference_qd_m_s':controller.veltargets[ref_i],
            'feedforward_N':controller.feedforward[ref_i] if enabled else np.zeros(12),
            'raw_command_N':controller.last_raw,'command_N':controller.command,
            'submitted_motor_force_N':controller.force.astype(np.float32).astype(float)}
        for name,value in values.items():
            arr=np.asarray(value)
            if arr.shape!=self.data[name].shape[1:] or not np.isfinite(arr).all():
                raise ValueError('Invalid controller trace value: '+name)
        for name,value in values.items():self.data[name][i]=value
        self.count+=1

    def arrays(self):
        return {name:arr[:self.count].copy() for name,arr in self.data.items()}

    def summary(self):
        a={name:arr[:self.count] for name,arr in self.data.items()}
        switches=lambda x:int(np.count_nonzero(x[1:]!=x[:-1]))
        tail=a['time_s']>=self.cfg.duration_s-.5
        return {'mode':self.cfg.feedforward_contact_mode,'recorded_control_ticks':self.count,
            'expected_control_ticks':self.capacity,'all_control_ticks_recorded':self.count==self.capacity,
            'selected_contact_transitions':switches(a['contact_selected']),
            'legacy_force_gate_transitions':switches(a['legacy_force_gate']),
            'feedforward_transitions':switches(a['feedforward_enabled']),
            'supported_ticks_with_legacy_force_gate_off':int(np.count_nonzero(a['contact_selected']&~a['legacy_force_gate'])),
            'final_interval_ticks':int(tail.sum()),
            'final_interval_feedforward_disabled_ticks':int(np.count_nonzero(tail&~a['feedforward_enabled'])),
            'control_period_s':self.cfg.control_period_s,'native_contact_count_measured':False}

    def save(self,output,completed):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        prefix='' if completed else 'partial_'
        np.savez_compressed(output/(prefix+'controller_trace.npz'),**self.arrays())
        write_json(output/(prefix+'contact_gate_summary.json'),dict(self.summary(),
            completed=bool(completed),gate=ContactGate(self.cfg).description(),
            time_alignment='pre-step t_k; native force from preceding physics step; command just updated; effort before causal filter advance',
            status='TELEMETRY_ONLY_NOT_A_TASK_PASS'))
