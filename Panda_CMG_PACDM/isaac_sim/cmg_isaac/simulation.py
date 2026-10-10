"""The actual PhysX mission; logs are observations, never playback commands."""
from pathlib import Path
import csv,io,json,time
import numpy as np
from scipy.spatial.transform import Rotation
from .control import Controller,wrench_at
from .validation import validate_case
from .fileio import write_json,write_text,save_npz,remove


def native_backend_name(case):
    return 'Isaac Sim 6.1 / CPU PhysX '+str(case.get('solver','TGS'))


def settled_motion(times,positions,rotations,window=.5):
    """Path-averaged object speed over the final window of the 100 Hz log.

    Sum of sample-to-sample displacement (and rotation angle) divided by the
    window length. Oscillation is therefore counted, not averaged away.
    """
    t=np.asarray(times,float);keep=t>=t[-1]-window-1e-9
    p=np.asarray(positions,float)[keep];R=np.asarray(rotations,float)[keep];span=float(t[keep][-1]-t[keep][0])
    if len(p)<2 or span<=0:
        return dict(settled_speed_m_s=None,settled_angular_speed_rad_s=None,settled_window_s=span)
    linear=float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)))/span
    angular=float(np.sum((Rotation.from_matrix(R[:-1]).inv()*Rotation.from_matrix(R[1:])).magnitude()))/span
    return dict(settled_speed_m_s=linear,settled_angular_speed_rad_s=angular,settled_window_s=span)


def run_mission(native,cmg,ref,mode,name,case,output,mechanics,smoke=False):
    out=Path(output);dt=case['dt'];duration=.25 if smoke else ref.duration
    steps=round(duration/dt);stride=round(.01/dt)
    if abs(steps*dt-duration)>1e-10 or stride<1 or abs(stride*dt-.01)>1e-10:
        raise ValueError('dt must divide both mission duration and the 0.01 s logging interval')
    controller=Controller(cmg,ref,mode,case['feedforward'],case['grasp'])
    bounds={j['id']:j.get('limits') for j in cmg['joints'] if j['type']!='fixed'}
    low=np.array([bounds[k]['lower'] for k in cmg['coordinate_ids']])
    high=np.array([bounds[k]['upper'] for k in cmg['coordinate_ids']])
    lim=np.array([87,87,87,87,12,12,12],float)
    logs={};maxp=0.;maxr=0.;sats=0;clips=0;gear=0.;margin=1.;fmargin=1.;maxratio=0.
    unexpected=0;bad_examples=[];minair=1.;minclear=1.;slip=0.;startrel=None
    minleft=1e30;minright=1e30;bilateral=True;peaknormal=0.;lift=-1e30;support=[]
    max_joint_error=0.;max_actuator_ratio=0.;rms_sum=0.;rms_count=0;max_energy=0.;work=0.;work_act=0.;work_ext=0.;work_damp=0.
    corner=np.array([[x,y,z] for x in [-.015,.015] for y in [-case['width']/2,case['width']/2] for z in [-.035,.035]])
    start=time.perf_counter();q,v=native.state()
    E0=controller.tree.energy(q,v)+.5*float((np.asarray(cmg['armature'])*v)@v)
    cached=dict(left_normal_force=0.,right_normal_force=0.,normal_force=0.,pad_contacts=0,
                bilateral_contact=0,unexpected_contacts=0,pick_support_N=0.)
    completed=0
    contact_points_before=native.contacts.total_contact_points if mode=='contact' else 0
    contact_callbacks_before=native.contacts.callback_count if mode=='contact' else 0
    def save_partial():
        # Diagnostic only: a locked file (OneDrive/antivirus) must not end a run.
        if logs:
            save_npz(out/'trajectory_partial.npz',{k:np.asarray(a) for k,a in logs.items()},required=False)
    try:
        native.checkpoint('Native mission started',dict(mode=mode,case=name,steps=steps,dt_s=dt))
        for k in range(steps+1):
            t=k*dt;q,v=native.state()
            command=controller.evaluate(t,q,v)
            max_joint_error=max(max_joint_error,float(np.max(abs(q[:7]-command['q_ref'][:7]))))
            max_actuator_ratio=max(max_actuator_ratio,float(np.max(abs(command['actuator_effort'])/np.r_[lim,100.])))
            target_p,target_R=ref.target(t)
            p,R=native.tool_pose()
            pe=float(np.linalg.norm(p-target_p));ae=float(Rotation.from_matrix(target_R.T@R).magnitude())
            maxp=max(maxp,pe);maxr=max(maxr,ae)
            ge=abs(q[7]-q[8]);gear=max(gear,ge)
            margin=min(margin,float(np.min(np.r_[q[:7]-low[:7],high[:7]-q[:7]])))
            fmargin=min(fmargin,float(np.min(np.r_[q[7:]-low[7:],high[7:]-q[7:]])))
            if margin<-.002 or fmargin<-.002 or np.max(abs(v[:7]))>25 or np.max(abs(v[7:]))>2.:
                raise RuntimeError(f'Numerical safety stop at {t:.6f}s: joint bounds/speed exceeded')
            if k<steps:
                sats+=int(command['arm_saturated']);clips+=int(command['control_clipped'])
            maxratio=max(maxratio,float(np.max(abs(command['torque'][:7])/lim)))
            w=wrench_at(t,mode,case['mass'])
            E=controller.tree.energy(q,v)+.5*float((np.asarray(cmg['armature'])*v)@v) if mode=='wrench' or k==0 else 0.
            residual=E-E0-work if mode=='wrench' else 0.
            max_energy=max(max_energy,abs(residual))
            values=dict(time=t,q=q,v=v,q_ref=command['q_ref'],v_ref=command['v_ref'],a_ref=command['a_ref'],
                tool_pos=p,tool_R=R,target_pos=target_p,target_R=target_R,pose_error=pe,angle_error=ae,
                torque=command['torque'],actuator_effort=command['actuator_effort'],applied_effort=command['applied_effort'],
                ctrl=command['ctrl'],wrench=w,gear_error=ge)
            if mode=='wrench':
                values.update(energy=E,work=work,energy_balance=residual)
            if mode=='contact':
                op,oR=native.pose(native.object)
                clearance=float(np.min((op+(oR@corner.T).T)[:,2])-.20)
                lift=max(lift,op[2]-.0702)
                if startrel is None and t>=7.99-1e-10:
                    startrel=R.T@(op-p)
                if 8.-1e-10<=t<=16.+1e-10:
                    minair=min(minair,op[2]);minleft=min(minleft,cached['left_normal_force']);minright=min(minright,cached['right_normal_force'])
                    bilateral=bilateral and bool(cached['bilateral_contact'])
                    peaknormal=max(peaknormal,cached['normal_force'])
                    slip=max(slip,float(np.linalg.norm(R.T@(op-p)-startrel)))
                if 8.-1e-10<=t<=12.+1e-10:minclear=min(minclear,clearance)
                if 1.<=t<=2.:support.append(cached['pick_support_N'])
                unexpected+=int(cached['unexpected_contacts']>0)
                if cached['unexpected_contacts'] and len(bad_examples)<10:
                    bad_examples.append(dict(time=t,pairs=native.contacts.last_bad))
                values.update(object_pos=op,object_R=oR,barrier_clearance=clearance,**cached)
            if k%stride==0 or k==steps:
                for key,value in values.items():logs.setdefault(key,[]).append(np.asarray(value).copy())
                rms_sum+=pe*pe;rms_count+=1
            if k%max(1,round(1./dt))==0:
                native.checkpoint(f'Mission t={t:.2f}s',dict(tool_error_mm=pe*1000,finger_coupling_m=ge))
                save_partial()
            if k==steps:break
            native.effort(command['applied_effort']);native.apply_wrench(w);native.step()
            if mode=='contact':cached=native.contacts.values()
            else:
                q1,v1=native.state()
                actuator=.5*float(command['torque']@(v+v1))*dt
                dissipation=.5*float(np.asarray(cmg['damping'])@(v*v+v1*v1))*dt
                external=.5*float(w@(controller.tree.jacobian(q)@v+controller.tree.jacobian(q1)@v1))*dt
                work_act+=actuator;work_ext+=external;work_damp+=dissipation
                work=work_act+work_ext-work_damp
            completed=k+1
            if native.gui and completed%max(1,round((1./30)/dt))==0:
                native.render()
                if not native.app.is_running() or not native.timeline.is_playing():
                    raise RuntimeError('Window/timeline stopped before task completion')
        arrays={key:np.asarray(a) for key,a in logs.items()}
        save_npz(out/'trajectory.npz',arrays)
        remove(out/'trajectory_partial.npz')
        offsets=getattr(native,'offsets',None)
        s=dict(mode=mode,case=name,configuration=case,duration_s=completed*dt,physics_dt_s=dt,
               logging_dt_s=.01,physics_steps=completed,wall_seconds=time.perf_counter()-start,
               # The offline PhysX twin (tools/physx_twin.py) overrides this so its
               # output can never be aggregated as native Isaac evidence.
               native_backend=getattr(native,'backend_name',native_backend_name(case)),
               # Joint angles are advanced in float32 once per step with PGS and once
               # per position iteration (sub-step dt/N) with TGS.
               native_solver=dict(type=case.get('solver'),position_iterations=case.get('position_iterations'),
                                  velocity_iterations=case.get('velocity_iterations'),
                                  position_integration_step_s=dt if case.get('solver')=='PGS'
                                  else dt/max(1,int(case.get('position_iterations') or 1))),
               joint_zero_offsets_rad=None if offsets is None else np.asarray(offsets,float).tolist(),
               max_tool_position_error_m=maxp,max_tool_orientation_error_deg=float(np.rad2deg(maxr)),
               rms_tool_position_error_m=float(np.sqrt(rms_sum/max(1,rms_count))),
               max_finger_coupling_error_m=float(gear),min_arm_joint_margin_rad=margin,min_finger_joint_margin_m=fmargin,
               maximum_arm_torque_Nm=np.max(abs(arrays['torque'][:,:7]),axis=0).tolist(),
               max_arm_torque_ratio=maxratio,max_actuator_effort_ratio=max_actuator_ratio,
               max_arm_joint_error_deg=float(np.rad2deg(max_joint_error)),final_tool_error_m=pe,
               final_tool_orientation_error_deg=float(np.rad2deg(ae)),arm_saturation_steps=sats,arm_control_clip_steps=clips,
               physics_error_count=len(native.physics_errors),unexpected_contact_steps=unexpected,
               max_energy_balance_error_J=max_energy if mode=='wrench' else None,
               actuator_work_J=work_act if mode=='wrench' else None,external_work_J=work_ext if mode=='wrench' else None,
               dissipated_work_J=work_damp if mode=='wrench' else None,
               observation_note='States/tool/object transforms and pad contact impulses are native PhysX; reference is generated by preserved PACDM.',
               samples=len(arrays['time']))
        if mode=='contact':
            op,oR=native.pose(native.object);lv,av=native.object.get_velocities()
            from .native import numpy
            s.update(max_lift_m=float(lift),min_transfer_height_m=float(minair) if not smoke else None,
               max_grasp_slip_m=float(slip) if not smoke else None,
               minimum_payload_barrier_clearance_m=float(minclear) if not smoke else None,
               min_transfer_left_normal_N=float(minleft) if not smoke else None,min_transfer_right_normal_N=float(minright) if not smoke else None,
               bilateral_contact_all_transfer=bool(bilateral) if not smoke else None,peak_transfer_normal_N=peaknormal,
               final_object_position_m=op.tolist(),placement_xy_error_m=float(np.linalg.norm(op[:2]-[.45,.20])),
               placement_z_error_m=float(abs(op[2]-.070)),
               placement_angle_error_deg=float(np.rad2deg(Rotation.from_matrix(Rotation.from_euler('z',np.pi/2).as_matrix().T@oR).magnitude())),
               # Gated (validation.py): PhysX's instantaneous velocity after the last step.
               final_speed_m_s=float(np.linalg.norm(numpy(lv))),final_angular_speed_rad_s=float(np.linalg.norm(numpy(av))),
               # Diagnostic: actual motion of the logged pose over the final 0.5 s. A resting
               # body's solver velocity can carry a residual (0.5 mm/s with one TGS
               # iteration in the offline twin while the body moved 0.1 um/s).
               **settled_motion(arrays['time'],arrays['object_pos'],arrays['object_R']),
               initial_support_relative_error=float(abs(np.mean(support)-case['mass']*9.81)/(case['mass']*9.81)) if support else None,
               native_contact_points=native.contacts.total_contact_points-contact_points_before,
               contact_callback_count=native.contacts.callback_count-contact_callbacks_before,
               unexpected_contact_examples=bad_examples)
        result=validate_case(s,mechanics,smoke)
        result.update(summary=s,mechanics=mechanics)
        write_json(out/'summary.json',s);write_json(out/'result.json',result)
        rows=io.StringIO();writer=csv.writer(rows,lineterminator='\n');writer.writerow(['metric','value'])
        for key,val in s.items():
            if val is None or isinstance(val,(str,int,float,bool)):writer.writerow([key,val])
        write_text(out/'metrics.csv',rows.getvalue())
        write_text(out/'REPORT.md','# '+result['status']+'\n\n'+result['scope']+'\n\n'+
             '\n'.join(f'- {name}: {"PASS" if c["passed"] else "FAIL"}; value={c["value"]}; required {c["relation"]} {c["limit"]}'
             for name,c in result['checks'].items())+'\n')
        return result
    except BaseException:
        save_partial()
        raise
    finally:
        # This stops a simulator, not a physical robot stopping interface.
        native.effort(np.zeros(9))
        native.timeline.pause()
