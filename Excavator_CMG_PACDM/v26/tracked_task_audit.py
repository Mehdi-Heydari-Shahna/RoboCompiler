"""Particle-identity task evidence and all-body force/energy replay for v26.

This postprocessor does not simulate a new trajectory or change saved states.
Native dynamics are replayed from qpos/qvel/ctrl for sampled force diagnostics.
Particle qualification uses every recorded state (normally 0.02 s), independently
of the simulator's accumulated carried mask. Force replay normally uses 0.1 s.
MuJoCo tree interaction wrenches are not unique physical closed-loop pin loads.
"""
from project import ROOT, SOURCE, RESULTS, save_json
import argparse
import csv
import hashlib
import json
import time as clock
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from digging_export import bucket_inside

SOURCE_PROVENANCE_SCOPE = (
    'Code-only distribution integrity: the included v21 source and model/configuration '
    'files listed in SOURCE_CODE_SHA256.json. Historical generated results are outside '
    'this manifest; their original release manifest is retained in provenance/.'
)


def _sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _R(quat):
    return Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()


def _inertia_shift(mass, offset):
    return mass*(float(offset@offset)*np.eye(3)-np.outer(offset,offset))


def validate_trace(model,trace):
    """Validate finite states and the registered state-sampling interval."""
    times=np.asarray(trace['time'])
    if times.ndim!=1 or len(times)<2 or not np.all(np.isfinite(times)):
        raise ValueError('Trace time must be finite, one-dimensional, and contain at least two samples')
    differences=np.diff(times)
    if np.any(differences<=0):raise ValueError('Trace timestamps must be strictly increasing')
    if float(differences.max())>.0200001:
        raise ValueError('Particle retention audit requires the registered <=0.02s state sampling')
    for key,width in [('qpos',model.nq),('qvel',model.nv),('ctrl',model.nu)]:
        values=np.asarray(trace[key])
        if values.shape!=(len(times),width) or not np.all(np.isfinite(values)):
            raise ValueError(f'{key} must have finite shape ({len(times)},{width})')
    return dict(passed=True,samples=len(times),strictly_increasing=True,finite_states=True,
                maximum_interval_s=float(differences.max()),maximum_allowed_interval_s=.0200001)


def source_preservation(model, scene):
    """Check the scoped code-only manifest and compiled inertias independently."""
    manifest_path = SOURCE/'SOURCE_CODE_SHA256.json'
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('The code-only source manifest must be a nonempty path-to-SHA256 mapping')
    hashes = []
    for name, expected in manifest.items():
        path = SOURCE/name
        if (not isinstance(name, str) or Path(name).is_absolute()
                or not path.resolve().is_relative_to(SOURCE.resolve())
                or not isinstance(expected, str) or len(expected) != 64
                or any(character not in '0123456789abcdef' for character in expected)):
            raise ValueError('Invalid code-only source manifest entry: '+str(name))
        got = _sha(path) if path.is_file() else None
        hashes.append(dict(path=name,expected_sha256=expected,actual_sha256=got,passed=got==expected))
    cmg = json.loads((SOURCE/'data/accepted_cmg_v04.json').read_text())
    bodies = {b['id']:b for b in cmg['bodies'] if b['kind']=='rigid_body'}
    original = bodies['body_53']
    track = scene['tracks']
    components = ['body_53']+[c['body'] for c in track['components']]
    ids = np.array([model.body(name).id for name in components],dtype=int)
    if len(set(components)) != len(components):
        raise ValueError('Repeated body in undercarriage mass partition')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    base = model.body('body_53').id
    basis = data.xmat[base].reshape(3,3)
    coms = (data.xipos[ids]-data.xpos[base])@basis
    masses = model.body_mass[ids]
    mass = float(masses.sum())
    center = np.sum(masses[:,None]*coms,axis=0)/mass
    tensor = np.zeros((3,3))
    for bid,mb,pos in zip(ids,masses,coms):
        rotation = basis.T@data.ximat[bid].reshape(3,3)
        tensor += rotation@np.diag(model.body_inertia[bid])@rotation.T + _inertia_shift(mb,pos-center)
    unchanged = []
    for name,source in bodies.items():
        if name=='body_53':
            continue
        bid = model.body(name).id
        rotation = _R(model.body_iquat[bid])
        native_i = rotation@np.diag(model.body_inertia[bid])@rotation.T
        errors = dict(mass_kg=abs(float(model.body_mass[bid])-source['mass_kg']),
                      com_m=float(np.max(abs(model.body_ipos[bid]-source['com_m']))),
                      inertia_kg_m2=float(np.max(abs(native_i-source['inertia_com_kg_m2']))))
        unchanged.append(dict(body=name,errors=errors,passed=errors['mass_kg']<1e-8 and errors['com_m']<1e-10 and errors['inertia_kg_m2']<1e-7))
    robot_ids = [b for b in range(1,model.nbody) if not (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or '').startswith('grain_')]
    robot_mass = float(model.body_mass[robot_ids].sum())
    source_mass = float(sum(b['mass_kg'] for b in bodies.values()))
    errors = dict(mass_kg=abs(mass-original['mass_kg']),
                  com_m=float(np.max(abs(center-original['com_m']))),
                  inertia_kg_m2=float(np.max(abs(tensor-original['inertia_com_kg_m2']))),
                  whole_robot_mass_kg=abs(robot_mass-source_mass))
    limits = dict(mass_kg=1e-8,com_m=1e-10,inertia_kg_m2=1e-7,whole_robot_mass_kg=1e-8)
    return dict(passed=all(x['passed'] for x in hashes+unchanged) and all(errors[k]<=limits[k] for k in errors),
                source_files=dict(count=len(hashes),passed=all(x['passed'] for x in hashes),records=hashes,
                                  manifest=manifest_path.name,scope=SOURCE_PROVENANCE_SCOPE),
                unchanged_source_bodies=unchanged,
                undercarriage=dict(reference='Compiled qpos0 assembly, original body_53 coordinates',
                    bodies=components,mass_kg=mass,com_m=center,inertia_kg_m2=tensor,errors=errors,limits=limits,
                    metadata_reconstruction_errors={k:track.get(k) for k in ['mass_reconstruction_error_kg','com_reconstruction_error_m','inertia_reconstruction_error_kg_m2']}),
                source_robot_mass_kg=source_mass,compiled_robot_mass_kg=robot_mass,
                source_inertia_scope='Partition matches the original rigid assembly at reference; moving shoes and rotors intentionally change instantaneous robot inertia')


def particle_ledger(model, trace, scene, departure=14., arrival=30., lifted_z=.9, carry_distance=1., required_final_time=72.):
    trace_quality=validate_trace(model,trace)
    if not (0<=departure<arrival<=required_final_time):
        raise ValueError('Task times must satisfy 0 <= departure < loaded arrival <= mission end')
    times = trace['time']
    count = int(scene['particle_count'])
    grains = np.array([model.body(f'grain_{i:03d}').id for i in range(count)],dtype=int)
    masses = model.body_mass[grains]
    grain_velocity_addresses=np.array([model.jnt_dofadr[model.body_jntadr[b]] for b in grains],dtype=int)
    data = mujoco.MjData(model)
    bucket,base = model.body('body_56').id,model.body('body_53').id
    marker = np.asarray(scene['tracks']['source_undercarriage_com_m'])
    xyz = np.zeros((len(times),count,3))
    inside = np.zeros((len(times),count),dtype=bool)
    base_xy = np.zeros((len(times),2))
    for k,qpos in enumerate(trace['qpos']):
        data.qpos[:]=qpos
        mujoco.mj_kinematics(model,data)
        xyz[k] = data.xpos[grains]
        base_xy[k] = (data.xpos[base]+data.xmat[base].reshape(3,3)@marker)[:2]
        if count:
            local=(xyz[k]-data.xpos[bucket])@data.xmat[bucket].reshape(3,3)
            inside[k]=bucket_inside(local)
    lifted=xyz[:,:,2]>lifted_z
    qualified=inside&lifted
    dep_idx=int(np.argmin(abs(times-departure)))
    arr_idx=int(np.argmin(abs(times-arrival)))
    exact_dep=bool(abs(times[dep_idx]-departure)<1e-8)
    exact_arr=bool(abs(times[arr_idx]-arrival)<1e-8)
    transport_available=exact_dep and exact_arr
    full_mission_covered=bool(times[0]<=1e-8 and times[-1]>=required_final_time-1e-8)
    available=transport_available and full_mission_covered
    departed=qualified[dep_idx] if exact_dep else np.zeros(count,dtype=bool)
    retained=qualified[arr_idx] if exact_arr else np.zeros(count,dtype=bool)
    transported=np.zeros(count,dtype=bool)
    continuous=np.zeros(count,dtype=bool)
    maximum_displacement=np.zeros(count)
    inside_path=np.zeros(count)
    inside_seconds=np.zeros(count)
    interval=slice(dep_idx,arr_idx+1)
    if transport_available:
        for particle in range(count):
            mask=qualified[interval,particle]
            starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]])
            ends=np.flatnonzero(mask&~np.r_[mask[1:],False])
            for start,end in zip(starts,ends):
                segment=base_xy[dep_idx+start:dep_idx+end+1]
                if len(segment)>1:
                    # Diameter of an uninterrupted sampled carried segment;
                    # repeated small rocking does not accumulate transport distance.
                    delta=segment[:,None,:]-segment[None,:,:]
                    maximum_displacement[particle]=max(maximum_displacement[particle],float(np.sqrt(np.max(np.sum(delta*delta,axis=2)))))
            both=mask[:-1]&mask[1:]
            increments=np.linalg.norm(np.diff(base_xy[interval],axis=0),axis=1)
            inside_path[particle]=float(increments[both].sum())
            inside_seconds[particle]=float(np.diff(times[interval])[both].sum())
            continuous[particle]=bool(np.all(mask))
        transported=departed&retained&(maximum_displacement>=carry_distance)
    depot_center=np.asarray(scene['depot_center_m'])
    depot_half=np.asarray(scene['depot_half_size_m'])
    final_in_depot=np.all(abs(xyz[-1,:,:2]-depot_center[:2])<=depot_half,axis=1)&(xyz[-1,:,2]>=0.)&(xyz[-1,:,2]<.8)
    final_speed=np.array([np.linalg.norm(trace['qvel'][-1,start:start+3]) for start in grain_velocity_addresses])
    settled=final_speed<=.1
    deposited=transported&final_in_depot&~inside[-1]&settled
    loaded_base_displacement=float(np.linalg.norm(base_xy[arr_idx]-base_xy[dep_idx])) if transport_available else 0.
    rows=[]
    for particle in range(count):
        rows.append(dict(particle_id=particle,body_name=f'grain_{particle:03d}',mass_kg=float(masses[particle]),
                         inside_lifted_at_departure=bool(departed[particle]),inside_lifted_at_arrival=bool(retained[particle]),
                         continuously_inside_lifted_at_saved_samples=bool(continuous[particle]),
                         maximum_contiguous_carried_base_displacement_m=float(maximum_displacement[particle]),
                         sampled_inside_base_path_m=float(inside_path[particle]),sampled_inside_time_s=float(inside_seconds[particle]),
                         qualified_transported=bool(transported[particle]),final_inside_bucket=bool(inside[-1,particle]),
                         final_in_depot=bool(final_in_depot[particle]),qualified_delivered=bool(deposited[particle]),
                         final_linear_speed_m_s=float(final_speed[particle]),final_settled=bool(settled[particle]),
                         final_x_m=float(xyz[-1,particle,0]),final_y_m=float(xyz[-1,particle,1]),final_z_m=float(xyz[-1,particle,2])))
    summary=dict(available=available,transport_phase_available=transport_available,
                 full_mission_covered=full_mission_covered,required_final_time_s=required_final_time,
                 actual_final_time_s=float(times[-1]),departure_time_s=departure,arrival_time_s=arrival,
                 departure_sample_time_s=float(times[dep_idx]) if exact_dep else None,
                 arrival_sample_time_s=float(times[arr_idx]) if exact_arr else None,
                 particle_count=count,departure_particle_ids=np.flatnonzero(departed),arrival_particle_ids=np.flatnonzero(retained),
                 transported_particle_ids=np.flatnonzero(transported),delivered_particle_ids=np.flatnonzero(deposited),
                 departure_count=int(departed.sum()),transported_count=int(transported.sum()),delivered_count=int(deposited.sum()),
                 departure_mass_kg=float(masses[departed].sum()),transported_mass_kg=float(masses[transported].sum()),
                 delivered_mass_kg=float(masses[deposited].sum()),loaded_base_displacement_m=loaded_base_displacement,
                 lift_threshold_world_z_m=lifted_z,per_particle_carried_displacement_threshold_m=carry_distance,
                 depot_center_m=depot_center,depot_half_size_m=depot_half,final_depot_z_lower_m=0.,final_depot_z_upper_m=.8,
                 final_settled_linear_speed_upper_m_s=.1,
                 evidence_scope='Particle identity and bucket cavity at every saved state; retention between saved states is not evaluated',
                 trace_quality=trace_quality,trace_max_sample_interval_s=float(np.max(np.diff(times),initial=0)),
                 gates=dict(loaded_displacement=available and loaded_base_displacement>=1.,
                            at_least_three_transported=available and int(transported.sum())>=3,
                            at_least_two_transported_delivered=available and int(deposited.sum())>=2))
    summary['passed']=bool(available and all(summary['gates'].values()))
    arrays=dict(particle_time_s=times,particle_xyz_m=xyz,particle_inside_bucket=inside,
                particle_lifted=lifted,particle_mass_kg=masses,particle_base_xy_m=base_xy)
    return summary,rows,arrays


def _body_group(name):
    if name.startswith('grain_'):return 'material'
    if name.startswith('track_'):return 'track'
    return 'source_robot'


def body_replay(model,trace,sample_period=.1):
    """Replay all-body forces and contact point slip; no Pinocchio per frame."""
    m=model;d=mujoco.MjData(m)
    body_ids=np.arange(1,m.nbody,dtype=int)
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) for b in body_ids]
    groups=np.array([_body_group(n) for n in names])
    act_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)]
    raw_times=trace['time']
    wanted=np.arange(raw_times[0],raw_times[-1]+1e-9,sample_period)
    indices=np.unique(np.r_[np.array([np.argmin(abs(raw_times-t)) for t in wanted],dtype=int),len(raw_times)-1])
    fields=['time_s','body_com_world_m','body_com_velocity_world_m_s','body_angular_velocity_world_rad_s',
            'body_energy_J','body_contact_wrench_world_at_com','body_tree_interaction_wrench_world_at_com',
            'body_native_cfrc_int_torque_force','body_native_cframe_origin_m',
            'actuator_position','actuator_velocity','actuator_force','actuator_power_W',
            'ground_support_force_world_N','track_ground_force_world_N','track_slip_mean_m_s','track_slip_peak_m_s',
            'peak_contact_pair_force_N','native_energy_J','contact_mapping_relative','actuator_power_error_W','body_energy_sum_error_J']
    out={key:[] for key in fields}
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));raw=np.zeros(6)
    peak_pairs={}
    max_saved_contact_difference=0.
    for k in indices:
        mujoco.mj_resetData(m,d)
        d.time=float(raw_times[k]);d.qpos[:]=trace['qpos'][k];d.qvel[:]=trace['qvel'][k];d.ctrl[:]=trace['ctrl'][k]
        if m.na:
            if 'act' not in trace:raise ValueError('Native actuator activation state missing from trace')
            d.act[:]=trace['act'][k]
        # This benchmark disallows externally applied propulsion forces. A
        # future trace with any external forces must save and restore them.
        for field in ('qfrc_applied','xfrc_applied'):
            if field in trace:getattr(d,field)[:]=trace[field][k]
        mujoco.mj_forward(m,d)
        mujoco.mj_rnePostConstraint(m,d)
        velocity=np.zeros((m.nbody,3));omega=np.zeros_like(velocity)
        energy=np.zeros((m.nbody,2))
        for b in body_ids:
            mujoco.mj_jacBodyCom(m,d,jp,jr,int(b))
            v,w=jp@d.qvel,jr@d.qvel
            velocity[b],omega[b]=v,w
            wi=d.ximat[b].reshape(3,3).T@w
            energy[b]=[.5*m.body_mass[b]*float(v@v)+.5*float(m.body_inertia[b]@(wi*wi)),
                       -m.body_mass[b]*float(m.opt.gravity@d.xipos[b])]
        contacts=np.zeros((m.nbody,6));generalized=np.zeros(m.nv)
        ground=np.zeros((3,3)) # source robot, tracks, material
        track_ground=np.zeros((2,3)) # left,right
        slip_weight=np.zeros(2);slip_sum=np.zeros(2);slip_peak=np.zeros(2)
        frame_peak=0.
        for ci in range(d.ncon):
            con=d.contact[ci]
            if con.efc_address<0:continue
            mujoco.mj_contactForce(m,d,ci,raw)
            R=con.frame.reshape(3,3);f=R.T@raw[:3];torque=R.T@raw[3:]
            b1,b2=int(m.geom_bodyid[con.geom1]),int(m.geom_bodyid[con.geom2])
            for b,sgn in ((b1,-1.),(b2,1.)):
                contacts[b,:3]+=sgn*f
                contacts[b,3:]+=sgn*(torque+np.cross(con.pos-d.xipos[b],f))
                if b:mujoco.mj_applyFT(m,d,sgn*f,sgn*torque,con.pos,b,generalized)
            magnitude=float(np.linalg.norm(f));frame_peak=max(frame_peak,magnitude)
            pair=tuple(sorted([mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom1)) or f'geom_{con.geom1}',
                               mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom2)) or f'geom_{con.geom2}']))
            if pair not in peak_pairs or magnitude>peak_pairs[pair]['peak_force_N']:
                peak_pairs[pair]=dict(geom1=pair[0],geom2=pair[1],peak_force_N=magnitude,time_s=float(d.time))
            if (b1==0)!=(b2==0):
                b,sgn=(b2,1.) if b1==0 else (b1,-1.)
                name=names[b-1];group=_body_group(name)
                ground[{'source_robot':0,'track':1,'material':2}[group]]+=sgn*f
                if name.startswith('track_') and '_shoe_' in name:
                    side=0 if name.startswith('track_left_') else 1
                    track_ground[side]+=sgn*f
                    relative=velocity[b2]+np.cross(omega[b2],con.pos-d.xipos[b2])-velocity[b1]-np.cross(omega[b1],con.pos-d.xipos[b1])
                    tangent=relative-R[0]*float(R[0]@relative)
                    speed=float(np.linalg.norm(tangent));load=max(0.,float(raw[0]))
                    if load>1.:
                        slip_weight[side]+=load;slip_sum[side]+=load*speed;slip_peak[side]=max(slip_peak[side],speed)
        weights=np.zeros(d.nefc)
        mask=d.efc_type>=int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        weights[mask]=d.efc_force[mask]
        reference=np.zeros(m.nv);mujoco.mj_mulJacTVec(m,d,reference,weights)
        contact_relative=float(np.max(abs(generalized-reference),initial=0)/max(1.,np.max(abs(reference),initial=0)))
        origins=d.subtree_com[m.body_rootid[body_ids]]
        native=d.cfrc_int[body_ids].copy()
        # Native cfrc_int is torque:force about the corresponding free-tree
        # subtree CoM, world oriented. Shift moment to each body's own CoM.
        internal=np.c_[native[:,3:],native[:,:3]+np.cross(origins-d.xipos[body_ids],native[:,3:])]
        ports=d.actuator_force*d.actuator_velocity
        energy_error=float(abs(energy.sum()+.5*np.dot(m.dof_armature,d.qvel**2)-d.energy.sum()))
        if 'body_contact_wrenches' in trace:
            saved=trace['body_contact_wrenches'][k]
            max_saved_contact_difference=max(max_saved_contact_difference,float(np.max(abs(saved-contacts[body_ids]),initial=0)))
        values=dict(time_s=float(d.time),body_com_world_m=d.xipos[body_ids].copy(),body_com_velocity_world_m_s=velocity[body_ids],
                    body_angular_velocity_world_rad_s=omega[body_ids],body_energy_J=energy[body_ids],
                    body_contact_wrench_world_at_com=contacts[body_ids],body_tree_interaction_wrench_world_at_com=internal,
                    body_native_cfrc_int_torque_force=native,body_native_cframe_origin_m=origins.copy(),
                    actuator_position=d.actuator_length.copy(),actuator_velocity=d.actuator_velocity.copy(),
                    actuator_force=d.actuator_force.copy(),actuator_power_W=ports.copy(),
                    ground_support_force_world_N=ground,track_ground_force_world_N=track_ground,
                    track_slip_mean_m_s=np.divide(slip_sum,slip_weight,out=np.zeros(2),where=slip_weight>0),track_slip_peak_m_s=slip_peak,
                    peak_contact_pair_force_N=frame_peak,native_energy_J=d.energy.copy(),contact_mapping_relative=contact_relative,
                    actuator_power_error_W=abs(float(ports.sum()-d.qfrc_actuator@d.qvel)),body_energy_sum_error_J=energy_error)
        for key,value in values.items():out[key].append(value)
    arrays={key:np.asarray(value) for key,value in out.items()}
    arrays.update(body_names=np.asarray(names),body_groups=groups,body_ids=body_ids,body_mass_kg=m.body_mass[body_ids].copy(),
                  actuator_names=np.asarray(act_names),track_side_order=np.asarray(['left','right']),
                  ground_group_order=np.asarray(['source_robot','track','material']))
    actuator_units=[]
    actuator_summary=[]
    for i,name in enumerate(act_names):
        jid=int(m.actuator_trnid[i,0])
        linear=m.actuator_trntype[i]==int(mujoco.mjtTrn.mjTRN_JOINT) and m.jnt_type[jid]==int(mujoco.mjtJoint.mjJNT_SLIDE)
        unit='m' if linear else 'rad'
        force_unit='N' if linear else 'Nm'
        actuator_units.append(unit)
        actuator_summary.append(dict(name=name,position_unit=unit,effort_unit=force_unit,
                                     position_min=float(arrays['actuator_position'][:,i].min()),position_max=float(arrays['actuator_position'][:,i].max()),
                                     peak_abs_velocity=float(abs(arrays['actuator_velocity'][:,i]).max()),
                                     peak_abs_effort=float(abs(arrays['actuator_force'][:,i]).max()),
                                     peak_abs_mechanical_power_W=float(abs(arrays['actuator_power_W'][:,i]).max())))
    arrays['actuator_position_units']=np.asarray(actuator_units)
    pairs=sorted(peak_pairs.values(),key=lambda row:row['peak_force_N'],reverse=True)
    arrays.update(contact_pair_geom1=np.asarray([p['geom1'] for p in pairs]),contact_pair_geom2=np.asarray([p['geom2'] for p in pairs]),
                  contact_pair_peak_force_N=np.asarray([p['peak_force_N'] for p in pairs]),contact_pair_peak_time_s=np.asarray([p['time_s'] for p in pairs]))
    summaries=[]
    for i,name in enumerate(names):
        external=arrays['body_contact_wrench_world_at_com'][:,i]
        internal=arrays['body_tree_interaction_wrench_world_at_com'][:,i]
        ek=arrays['body_energy_J'][:,i,0];eu=arrays['body_energy_J'][:,i,1]
        summaries.append(dict(body_id=int(body_ids[i]),body_name=name,group=str(groups[i]),mass_kg=float(m.body_mass[body_ids[i]]),
                             peak_contact_force_N=float(np.linalg.norm(external[:,:3],axis=1).max()),
                             peak_contact_moment_Nm=float(np.linalg.norm(external[:,3:],axis=1).max()),
                             peak_tree_interaction_force_N=float(np.linalg.norm(internal[:,:3],axis=1).max()),
                             peak_tree_interaction_moment_Nm=float(np.linalg.norm(internal[:,3:],axis=1).max()),
                             peak_kinetic_energy_J=float(ek.max()),potential_energy_change_J=float(eu[-1]-eu[0])))
    max_contact=float(arrays['contact_mapping_relative'].max())
    max_energy=float(arrays['body_energy_sum_error_J'].max())
    max_power=float(arrays['actuator_power_error_W'].max())
    report=dict(samples=len(indices),sample_period_requested_s=sample_period,bodies=len(body_ids),actuators=m.nu,
                max_explicit_contact_mapping_relative=max_contact,max_body_energy_sum_error_J=max_energy,
                max_actuator_power_error_W=max_power,max_replayed_saved_contact_wrench_difference_SI=max_saved_contact_difference,
                replay_note='qacc warm-start is not saved; replay re-solves native constraints. Replay differences are numerical solver differences, not trajectory changes.',
                peak_contact_pair_force_N=float(arrays['peak_contact_pair_force_N'].max()),top_contact_pairs=pairs[:20],
                per_pair_peak_count=len(pairs),per_pair_full_data='contact_pair_* arrays in body audit NPZ',actuator_summary=actuator_summary,
                track_slip_mean_peak_m_s=arrays['track_slip_mean_m_s'].max(axis=0),track_slip_peak_m_s=arrays['track_slip_peak_m_s'].max(axis=0),
                track_slip_definition='Normal-force-weighted tangential relative point velocity of loaded (>1N) shoe-ground contacts; m/s, not a kinematic slip ratio',
                wrench_convention='World-oriented [Fx,Fy,Fz,Tx,Ty,Tz], moments about each body CoM. Native cfrc_int separately retained as torque:force about tree CoM.',
                tree_wrench_limitation='Native tree-parent interactions depend on the selected spanning tree and constraint representation; not unique physical closed-loop cut-pin loads',
                thresholds=dict(contact_mapping_relative=1e-8,body_energy_sum_error_J=1e-6,actuator_power_error_W=1e-7),
                passed=max_contact<=1e-8 and max_energy<=1e-6 and max_power<=1e-7)
    return report,arrays,summaries


def _write_csv(path,rows,default_header):
    with open(path,'w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else default_header)
        writer.writeheader();writer.writerows(rows)


def energy_decomposition(model,trace,source_report):
    """Separate quadrature, fluid integration and mechanical-port coupling.

    Accumulated work is read from the trace, independently of its JSON summary.
    Pressures and chamber positions independently reconstruct stored energy.
    Absolute-work normalizations, which cannot be recovered exactly from the
    downsampled trace, retain explicit simulator-summary provenance. Residuals
    are exposed without adding a new physical-accuracy acceptance threshold.
    """
    from hydraulic_actuators import PARAMETERS as arm_parameters
    from track_drive import PARAMETERS as drive_parameters
    mapping=json.loads((SOURCE/'data/accepted_mujoco_mapping.json').read_text())
    ports=[f'p{i}' for i in range(6)]
    positions=trace['qpos'][:,[model.jnt_qposadr[model.joint(name).id] for name in ports]]
    nominal=np.array([mapping['source_reference'][name] for name in ports])
    offset=positions-nominal
    arm_pressure=trace['arm_pressure'];drive_pressure=trace['drive_pressure']
    area_a=np.asarray(arm_parameters['area_A_m2']);area_b=np.asarray(arm_parameters['area_B_m2'])
    span=arm_parameters['coordinate_half_span_m'];beta=arm_parameters['bulk_modulus_Pa']
    volumes=arm_parameters['dead_volume_m3']+np.stack((area_a*(span+offset),area_b*(span-offset)),axis=-1)
    if np.any(volumes<=0):raise ValueError('Nonpositive hydraulic volume in recorded trace')
    arm_energy=np.sum(volumes*arm_pressure**2,axis=(1,2))/(2*beta)
    drive_energy=drive_parameters['effective_chamber_volume_m3']*np.sum(drive_pressure**2,axis=(1,2))/(2*drive_parameters['bulk_modulus_Pa'])
    energy_checks=dict(arm_stored_energy_reconstruction_max_error_J=float(np.max(abs(arm_energy-trace['arm_fluid_energy']))),
                       drive_stored_energy_reconstruction_max_error_J=float(np.max(abs(drive_energy-trace['drive_fluid_energy']))))
    order=source_report['power_order']
    increments=trace['work'][-1]-trace['work'][0]
    if len(order)!=len(increments) or not np.all(np.isfinite(increments)):
        raise ValueError('Invalid cumulative hydraulic work ordering or values')
    work=dict(zip(order,map(float,increments)))
    energy_checks['work_trace_report_max_error_J']=max(abs(work[k]-source_report['work_J'][k]) for k in order)
    d_arm=float(arm_energy[-1]-arm_energy[0]);d_drive=float(drive_energy[-1]-drive_energy[0])
    d_mechanical=float(trace['energy'][-1].sum()-trace['energy'][0].sum())
    arm_net=(work['arm_supply']-work['arm_throttle']-work['arm_leakage']-work['arm_relief']-work['arm_friction']
             +work['arm_compressibility_geometry']+work['arm_rotary'])
    drive_net=(work['drive_supply']-work['drive_throttle']-work['drive_leakage']-work['drive_relief']-work['drive_friction']
               -work['drive_numerical_storage_loss'])
    arm_fluid_defect=d_arm-(arm_net-work['arm_mechanical'])
    drive_fluid_defect=d_drive-(drive_net-work['drive_mechanical'])
    environmental_rectangle=work['equality']+work['contact']+work['passive']
    mechanical_rectangle_defect=d_mechanical-work['arm_mechanical']-work['drive_mechanical']-environmental_rectangle
    total_rectangle=d_mechanical+d_arm+d_drive-arm_net-drive_net-environmental_rectangle
    energy_checks['total_decomposition_identity_error_J']=abs(total_rectangle-mechanical_rectangle_defect-arm_fluid_defect-drive_fluid_defect)
    energy_checks['total_trace_report_error_J']=abs(total_rectangle-source_report['total_balance_defect_J'])
    midpoint_available='mechanical_midpoint_work' in trace
    midpoint=None;midpoint_mechanical_defect=None;total_midpoint_environment=None
    if midpoint_available:
        values=trace['mechanical_midpoint_work'][-1]-trace['mechanical_midpoint_work'][0]
        midpoint=dict(zip(['arm_mechanical','drive_mechanical','equality','contact','passive'],map(float,values)))
        midpoint_mechanical_defect=d_mechanical-sum(midpoint.values())
        total_midpoint_environment=d_mechanical+d_arm+d_drive-arm_net-drive_net-sum(midpoint[k] for k in ['equality','contact','passive'])
        energy_checks['midpoint_mechanical_trace_report_error_J']=abs(midpoint_mechanical_defect-source_report['mechanical_balance_defect_J'])
    displacements=np.asarray(source_report.get('port_displacement_work_J',[]),dtype=float)
    displacement_available=len(displacements)==10
    coupling=dict(displacement_work_available=displacement_available,
                  displacement_work_provenance='Simulator integrates held effort times actual per-step scalar-joint displacement; not recoverable exactly from 0.02s snapshots')
    if displacement_available:
        arm_displacement=float(displacements[:8].sum());drive_displacement=float(displacements[8:].sum())
        coupling.update(arm_displacement_work_J=arm_displacement,drive_displacement_work_J=drive_displacement,
                        arm_displacement_minus_held_speed_work_J=arm_displacement-work['arm_mechanical'],
                        drive_displacement_minus_held_speed_work_J=drive_displacement-work['drive_mechanical'],
                        arm_fluid_balance_using_displacement_defect_J=d_arm-arm_net+arm_displacement,
                        drive_fluid_balance_using_displacement_defect_J=d_drive-drive_net+drive_displacement)
        if midpoint_available:
            coupling.update(arm_midpoint_minus_held_speed_work_J=midpoint['arm_mechanical']-work['arm_mechanical'],
                            drive_midpoint_minus_held_speed_work_J=midpoint['drive_mechanical']-work['drive_mechanical'],
                            mechanical_displacement_and_midpoint_environment_defect_J=d_mechanical-arm_displacement-drive_displacement-sum(midpoint[k] for k in ['equality','contact','passive']))
    aw=source_report.get('absolute_work_J',{})
    maw=source_report.get('mechanical_midpoint_absolute_work_J',{})
    denominators=dict(regularization_floor_J=1.,
                      gross_hydraulic_supply_J=work['arm_supply']+work['drive_supply'],
                      gross_nonmechanical_ledger_activity_J=sum(v for k,v in aw.items() if k not in ['arm_mechanical','drive_mechanical']),
                      absolute_combined_mechanical_and_fluid_energy_change_J=abs(d_mechanical+d_arm+d_drive),
                      sum_absolute_mechanical_and_fluid_energy_changes_J=abs(d_mechanical)+abs(d_arm)+abs(d_drive),
                      held_speed_actuator_mechanical_activity_J=aw.get('arm_mechanical',0.)+aw.get('drive_mechanical',0.),
                      midpoint_mechanical_activity_J=sum(maw.values()) if midpoint_available else None,
                      absolute_activity_provenance='Native every-step accumulated absolute work from simulator JSON; other denominators reconstructed from trace')
    fractions={key:abs(total_rectangle)/max(1.,denominators[key]) for key in [
        'gross_hydraulic_supply_J','gross_nonmechanical_ledger_activity_J',
        'absolute_combined_mechanical_and_fluid_energy_change_J','sum_absolute_mechanical_and_fluid_energy_changes_J',
        'held_speed_actuator_mechanical_activity_J']}
    physical_loss_keys=[k for k in work if any(k.endswith('_'+suffix) for suffix in ['throttle','leakage','relief','friction'])]
    physical_loss_work={k:work[k] for k in physical_loss_keys}
    sampled_loss_min={k:float(np.min(trace['power'][:,order.index(k)])) for k in physical_loss_keys}
    return dict(available=True,accounting_consistency_passed=all(v<=1e-6 for v in energy_checks.values()),
                arithmetic_checks_J=energy_checks,arithmetic_limit_J=1e-6,
                energy_change_J=dict(mechanical=d_mechanical,arm_fluid=d_arm,drive_fluid=d_drive,combined=d_mechanical+d_arm+d_drive),
                arm_fluid=dict(integrator='Original accepted explicit Euler, variable chamber volumes',
                    net_input_after_held_shaft_work_J=arm_net-work['arm_mechanical'],stored_energy_change_J=d_arm,
                    balance_defect_J=arm_fluid_defect,absolute_balance_defect_J=abs(arm_fluid_defect),
                    relative_to_absolute_stored_energy_change=abs(arm_fluid_defect)/max(1.,abs(d_arm)),
                    relative_to_supply_work=abs(arm_fluid_defect)/max(1.,abs(work['arm_supply'])),
                    interpretation='Finite-step pressure/volume and coupling error; instantaneous differential identity does not imply zero accumulated storage defect'),
                drive_fluid=dict(integrator='Bounded backward Euler at held shaft speed, numerical storage dissipation separately accounted',
                    net_input_after_held_shaft_work_J=drive_net-work['drive_mechanical'],stored_energy_change_J=d_drive,
                    balance_defect_J=drive_fluid_defect,absolute_balance_defect_J=abs(drive_fluid_defect),
                    numerical_storage_dissipation_J=work['drive_numerical_storage_loss']),
                mechanical=dict(midpoint_available=midpoint_available,midpoint_balance_defect_J=midpoint_mechanical_defect,
                    rectangle_balance_defect_J=mechanical_rectangle_defect,midpoint_work_J=midpoint,
                    interpretation='Held pre-step generalized forces contracted with mean pre/post-step velocity; not proof of second-order time integration or hardware energy accuracy'),
                total=dict(rectangle_environment_balance_defect_J=total_rectangle,absolute_balance_defect_J=abs(total_rectangle),
                    midpoint_environment_balance_defect_J=total_midpoint_environment,
                    rectangle_decomposition='mechanical rectangle defect + arm fluid held-speed defect + drive fluid held-speed defect',
                    fractions_by_denominator=fractions),
                denominator_definitions=denominators,port_work_coupling=coupling,
                physical_losses=dict(accumulated_work_J=physical_loss_work,minimum_saved_power_W=sampled_loss_min,
                    all_accumulated_nonnegative=all(v>=-1e-7 for v in physical_loss_work.values()),
                    all_sampled_nonnegative=all(v>=-1e-7 for v in sampled_loss_min.values()),
                    sampling_scope='Saved instantaneous loss samples are 0.02s apart; accumulated works contain every integration step'),
                acceptance_scope='Arithmetic checks only. No new physical-energy accuracy threshold is introduced; timestep convergence and original mechanical gates are assessed separately.',
                energy_scope='Ideal-reservoir hydraulic work of an assumed model; not pump/engine fuel consumption, efficiency identification, or calibrated real-excavator energy.')


def audit_task(name,body_interval=.1):
    started=clock.perf_counter()
    source_report=json.loads((RESULTS/f'{name}.json').read_text())
    with np.load(RESULTS/f'{name}.npz',allow_pickle=False) as archive:
        trace={key:archive[key] for key in archive.files}
    xml=RESULTS/'assets'/f'{name}.xml'
    m=mujoco.MjModel.from_xml_path(str(xml))
    if not source_report.get('base_propulsion_wrench',True) is False:
        raise ValueError('Task replay requires declared zero base propulsion wrench')
    preservation=source_preservation(m,source_report['scene'])
    task_times=source_report['scene'].get('task_times_s',{})
    task,particles,particle_arrays=particle_ledger(m,trace,source_report['scene'],
        departure=float(task_times.get('departure',14.)),arrival=float(task_times.get('loaded_arrival',30.)),
        required_final_time=float(task_times.get('end',72.)))
    task['schedule_source']='Saved scene.task_times_s' if task_times else 'Original registered 14/30/72s mission (legacy trace)'
    body,body_arrays,body_rows=body_replay(m,trace,body_interval)
    energy=energy_decomposition(m,trace,source_report)
    body_path=RESULTS/f'{name}_body_audit.npz'
    np.savez_compressed(body_path,**body_arrays,**particle_arrays)
    _write_csv(RESULTS/f'{name}_particle_ledger.csv',particles,['particle_id','body_name','mass_kg','qualified_transported','qualified_delivered'])
    _write_csv(RESULTS/f'{name}_body_summary.csv',body_rows,['body_id','body_name'])
    report=dict(name=name,versions=dict(mujoco=mujoco.__version__),elapsed_s=clock.perf_counter()-started,
                input_model_sha256=_sha(xml),input_trace_sha256=_sha(RESULTS/f'{name}.npz'),
                input_report_sha256=_sha(RESULTS/f'{name}.json'),
                source_preservation=preservation,task=task,body_replay=body,energy_decomposition=energy,
                original_simulator_task_metrics={key:source_report.get(key) for key in ['carried_mass_kg','delivered_mass_kg','loaded_at_departure_mass_kg','loaded_base_displacement_m']},
                task_gate_available=task['available'],task_passed=task['passed'],
                technical_checks_passed=preservation['passed'] and body['passed'] and energy['accounting_consistency_passed'],
                passed=preservation['passed'] and body['passed'] and energy['accounting_consistency_passed'] and task['passed'],
                artifacts=dict(body_audit_npz=body_path.name,particle_ledger_csv=f'{name}_particle_ledger.csv',body_summary_csv=f'{name}_body_summary.csv'),
                sources=['https://mujoco.readthedocs.io/en/3.3.7/computation/index.html',
                         'https://mujoco.readthedocs.io/en/3.3.7/APIreference/APItypes.html',
                         'https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/src/engine/engine_core_smooth.c'])
    save_json(RESULTS/f'{name}_task_audit.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name');parser.add_argument('--sample-period',type=float,default=.1)
    args=parser.parse_args()
    report=audit_task(args.name,args.sample_period)
    print(json.dumps(dict(name=args.name,technical_checks_passed=report['technical_checks_passed'],task_gate_available=report['task_gate_available'],
                         task_passed=report['task_passed'],transported_count=report['task']['transported_count'],
                         delivered_count=report['task']['delivered_count'],elapsed_s=report['elapsed_s']),indent=2))
    raise SystemExit(0 if report['passed'] else 1)
