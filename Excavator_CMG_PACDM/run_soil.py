"""Run the actual MuJoCo soil prototype; save native states for video replay."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'v26'))
from project import save_json, save_npz
import argparse
import time
import traceback
import numpy as np
import mujoco
from benchmark_model import context, PORTS
from hydraulic_actuators import HydraulicBank
from digging_export import bucket_inside
from digging_path import LIP
from mobile_mission import TravelController
from tracked_arm import TrackedArmController
from track_drive import TrackDriveBank
from native_solve import accurate_forward, force_residual
from native_step import advance_after_forward, validate_model
from soil_scene import build_soil_scene, BOUNDS
from soil_mission import SoilMission


def run(args):
    out = ROOT/'outputs'/args.name; out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.dt <= 0 or abs(round(.001/args.dt)*args.dt-.001) > 1e-12:
        raise ValueError('dt must divide the 1 ms track servo period')
    cmg, mapping, e, a, r, inverse = context()
    mission = SoilMission(e, r, cut_depth=args.cut_depth, cut_policy=args.cut_policy)
    meta = build_soil_scene(out/'scene.xml', mission, dt=args.dt, radius=args.radius,
                            empty=args.empty, bucket_contact=not args.no_bucket_contact, cone=args.cone,
                            entry_slot=args.entry_slot)
    save_json(out/'scene_metadata.json', meta)
    m = mujoco.MjModel.from_xml_path(str(out/'scene.xml')); d = mujoco.MjData(m)
    validate_model(m)
    jids = np.array([m.joint(k).id for k in e.tree_ids]); qi=m.jnt_qposadr[jids]; vi=m.jnt_dofadr[jids]
    pi = np.array([e.tree_ids.index(k) for k in PORTS])
    ai = np.array([m.actuator('effort_'+k).id for k in PORTS])
    d.qpos[qi] = mission.initial_tree
    mujoco.mj_forward(m, d)
    arm = TrackedArmController(cmg, mapping, e, a, r, inverse, mission.initial_tree, m, d)
    travel = TravelController(m, d, meta); drive = TrackDriveBank()
    command = arm.evaluate(*mission.arm_reference(0.), d)
    bank = HydraulicBank(r.nominal[pi], command)
    grains = np.array([m.body(f'grain_{i:04d}').id for i in range(meta['particle_count'])], int)
    grain_qi = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in grains], int)
    grain_vi = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in grains], int)
    masses = m.body_mass[grains].copy(); total_mass = float(masses.sum())
    ever_lifted = np.zeros(len(grains), bool); deposited_dwell = np.zeros(len(grains))
    bucket = m.body('body_56').id
    eqnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or '' for i in range(m.neq)]
    arm_eq = [i for i, name in enumerate(eqnames) if not name.startswith('track_')]
    keys = ['time','qpos','qvel','ctrl','phase','bucket_mass','lifted_mass','deposited_mass',
            'source_mass','spill_mass','lip_position','tracking_error','closure_error',
            'force_residual','energy','mechanical_work','soil_bucket_force','soil_speed',
            'pressure_max','base_pose','contact_count','mass_partition_error']
    trace = {k: [] for k in keys}
    peak = dict(force_residual=0., closure_m=0., tracking_rad=0., bucket_mass_kg=0.,
                soil_bucket_force_N=0., native_warning_count=0, mass_partition_error_kg=0.)
    work = np.zeros(4); abswork = np.zeros(4); initial_energy = None
    warnings0 = np.array([w.number for w in d.warning]); last_command = np.zeros(2)
    error = None; status = 'INCOMPLETE'; next_print = 0.
    control_every=round(.01/args.dt); servo_every=round(.001/args.dt); sample_every=round(.04/args.dt)
    ma = np.zeros(m.nv)
    settled_state_saved = False
    try:
        for step in range(round(args.duration/args.dt)+1):
            t = float(d.time)
            desired = mission.arm_reference(t)
            q=d.qpos[qi].copy(); v=d.qvel[vi].copy()
            if step % control_every == 0:
                command = arm.evaluate(*desired, d)
            if step % servo_every == 0:
                last_command = travel.evaluate(d, np.zeros(3), np.zeros(3), .001,
                                               update_outer=step % control_every == 0)
            item=bank.evaluate(q[pi], v[pi], command, args.dt)
            di=drive.evaluate(d.qvel[travel.vi], last_command, args.dt)
            d.ctrl[ai]=item['effort']; d.ctrl[travel.ai]=di['effort']
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
                raise ValueError('Unexpected external force')
            solve=accurate_forward(m,d)
            if initial_energy is None: initial_energy=float(d.energy.sum())
            eqmask = d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            contactmask = d.efc_type >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
            armmask = eqmask & np.isin(d.efc_id, arm_eq)
            closure=float(np.max(abs(d.efc_pos[armmask]), initial=0.))
            tracking=float(np.max(abs(q[e.active[:6]]-desired[0][:6])))
            lip=d.xpos[bucket]+d.xmat[bucket].reshape(3,3)@LIP
            sampled = step % sample_every == 0 or step == round(args.duration/args.dt)
            if sampled:
                xyz=d.xpos[grains].copy()
                gv=d.qvel[grain_vi[:,None]+np.arange(3)] if len(grains) else np.zeros((0,3))
                speed=np.linalg.norm(gv, axis=1)
                local=(xyz-d.xpos[bucket])@d.xmat[bucket].reshape(3,3)
                inside=bucket_inside(local) if len(grains) else np.zeros(0,bool)
                ever_lifted |= inside & (xyz[:,2] > .5)
                receiver = np.all(abs(xyz[:,:2]-mission.depot_center[:2]) < mission.depot_half_size,axis=1)
                receiver &= (xyz[:,2] < .55) & (~inside) & ever_lifted & (speed < .12)
                deposited_dwell = np.where(receiver, deposited_dwell+.04, 0.)
                deposited = receiver & (deposited_dwell >= .5)
                in_source = np.all((xyz >= BOUNDS[:,0]) & (xyz <= BOUNDS[:,1]+[0,0,.12]),axis=1)
                in_source &= ~inside & ~deposited
                spilled = ~(inside | deposited | in_source)
                bmass=float(masses[inside].sum()); depmass=float(masses[deposited].sum())
                source_mass=float(masses[in_source].sum()); spillmass=float(masses[spilled].sum())
                mass_error=abs(total_mass-bmass-depmass-source_mass-spillmass)
                soil_speed=float(np.quantile(speed,.95)) if len(speed) else 0.
                force=np.zeros(3); cf=np.zeros(6)
                for ci in range(d.ncon):
                    con=d.contact[ci]; b1=m.geom_bodyid[con.geom1]; b2=m.geom_bodyid[con.geom2]
                    if (b1==bucket and b2 in grains) or (b2==bucket and b1 in grains):
                        mujoco.mj_contactForce(m,d,ci,cf)
                        force += (1 if b2==bucket else -1)*(con.frame.reshape(3,3).T@cf[:3])
                vals=dict(time=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),phase=mission.phase,
                          bucket_mass=bmass,lifted_mass=float(masses[ever_lifted].sum()),deposited_mass=depmass,
                          source_mass=source_mass,spill_mass=spillmass,lip_position=lip.copy(),tracking_error=tracking,
                          closure_error=closure,force_residual=solve['final_residual'],energy=d.energy.copy(),
                          mechanical_work=work.copy(),soil_bucket_force=force,soil_speed=soil_speed,
                          pressure_max=float(bank.pressure.max()),base_pose=travel.pose(d),contact_count=d.ncon,
                          mass_partition_error=mass_error)
                for k,val in vals.items(): trace[k].append(val)
                peak['bucket_mass_kg']=max(peak['bucket_mass_kg'],bmass)
                peak['soil_bucket_force_N']=max(peak['soil_bucket_force_N'],float(np.linalg.norm(force)))
                peak['mass_partition_error_kg']=max(peak['mass_partition_error_kg'],mass_error)
                old_phase=mission.phase
                mission.update(t,tracking,soil_speed,bmass,float(lip[2]),depmass,args.settle_only,lip_x=float(lip[0]))
                if old_phase==0 and (mission.phase!=0 or mission.done) and not mission.failures:
                    save_npz(out/'settled_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,
                             pressure=bank.pressure,rotary=bank.rotary,time=t)
                    settled_state_saved=True
                if mission.done:
                    status='TASK_FAILED' if mission.failures else 'COMPLETED'
                    break
            peak['force_residual']=max(peak['force_residual'],solve['final_residual'])
            peak['closure_m']=max(peak['closure_m'],closure);peak['tracking_rad']=max(peak['tracking_rad'],tracking)
            if t >= next_print:
                print(f"t={t:.2f}s {mission.names[mission.phase]} load={bmass:.1f}kg deposit={depmass:.1f}kg "
                      f"error={tracking:.3f}rad lip=({lip[0]:.2f},{lip[2]:.2f}) contact={d.ncon} wall={time.perf_counter()-start:.1f}s",flush=True)
                next_print += 1.
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                raise FloatingPointError('Nonfinite native state')
            if np.any(np.array([w.number for w in d.warning]) != warnings0):
                raise RuntimeError('MuJoCo warning; stop and inspect checkpoint')
            if step == round(args.duration/args.dt): break
            old_v=d.qvel.copy(); eqf=d.efc_force[eqmask].copy(); cof=d.efc_force[contactmask].copy()
            passive=d.qfrc_passive.copy(); tau=d.qfrc_actuator.copy()
            advance_after_forward(m,d)
            mid=.5*(old_v+d.qvel); jmid=np.zeros(d.nefc);mujoco.mj_mulJacVec(m,d,jmid,mid)
            powers=np.array([tau@mid, eqf@jmid[eqmask], cof@jmid[contactmask], passive@mid])
            work += args.dt*powers;abswork += args.dt*abs(powers)
            bank.advance(item,args.dt);drive.advance(di,args.dt)
    except (Exception, KeyboardInterrupt):
        error=traceback.format_exc();status='ERROR';print(error,flush=True)
    result={k:np.asarray(v) for k,v in trace.items()}
    save_npz(out/'trajectory.npz',**result)
    save_npz(out/'last_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time,
             pressure=bank.pressure,rotary=bank.rotary,qacc_warmstart=d.qacc_warmstart)
    warnings=np.array([w.number for w in d.warning])-warnings0
    peak['native_warning_count']=int(warnings.sum())
    final_energy=float(d.energy.sum());energy_defect=final_energy-(initial_energy or 0.)-float(work.sum())
    report=dict(status=status,error=error,scope='Stationary dry-granular prototype; no field calibration',
                elapsed_wall_s=time.perf_counter()-start,simulated_duration_s=float(d.time),
                args=vars(args),scene=meta,phase_names=mission.names,events=mission.events,
                failures=mission.failures,peak=peak,arm_controller=arm.diagnostics(),
                deposited_mass_kg=float(result['deposited_mass'][-1]) if len(result['time']) else 0.,
                settled_state_saved=settled_state_saved,mechanical_work_J=work,
                mechanical_energy_defect_J=energy_defect,
                mechanical_energy_relative_defect=abs(energy_defect)/max(1.,float(abswork.sum())),
                native_state_projection=False,external_forces=False,plant='MuJoCo '+mujoco.__version__,
                full_v26_validation_rerun=False,soil_parameter_calibration=False,
                resolution_and_timestep_convergence_tested=False)
    save_json(out/'report.json',report)
    print('Saved',out,'status',status,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='soil_demo');p.add_argument('--duration',type=float,default=65.)
    p.add_argument('--dt',type=float,default=.001);p.add_argument('--radius',type=float,default=.10)
    p.add_argument('--empty',action='store_true');p.add_argument('--no-bucket-contact',action='store_true')
    p.add_argument('--cone',choices=['pyramidal','elliptic'],default='pyramidal')
    p.add_argument('--entry-slot',type=float,default=0.)
    p.add_argument('--cut-depth',type=float,default=.11)
    p.add_argument('--cut-policy',choices=['load-aware','position-only'],default='position-only')
    p.add_argument('--settle-only',action='store_true')
    args=p.parse_args()
    if args.radius<.02 or args.radius>.10 or args.duration<=0:p.error('radius 0.02–0.10 m; positive duration required')
    if not 0<=args.entry_slot<=1.5 or not .01<=args.cut_depth<=.25:p.error('entry slot 0–1.5 m; cut depth 0.01–0.25 m')
    report=run(args)
    raise SystemExit(1 if report['status'] in ['ERROR','TASK_FAILED'] else 0)
