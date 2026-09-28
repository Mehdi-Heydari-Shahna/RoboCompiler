"""Independent force/energy checks and complete body/port data exports."""
from pathlib import Path
import csv,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from reconstructed_model import whole_body
from whole_body_dynamics import FloatingSource,PinFloating,add_drive_terms
from contact_task import HISTORY_COLUMNS

ROOT=Path(__file__).resolve().parent

def load_trace(name):
    with np.load(ROOT/'results'/f'{name}.npz') as a:return {k:a[k] for k in a.files}

def audit(name='landing_nominal',export_csv=True):
    a=load_trace(name);r=json.loads((ROOT/'results'/f'{name}.json').read_text());c=whole_body()
    m=mujoco.MjModel.from_xml_path(str(ROOT/'models'/f'{name}.xml'));d=mujoco.MjData(m)
    source=FloatingSource(c);pin=PinFloating(c)
    bodies=[m.body(b['id']).id for b in c['bodies']];jids=[m.joint(x).id for x in c['coordinate_ids']]
    qo=m.jnt_qposadr[jids];vo=m.jnt_dofadr[jids];order=np.r_[np.arange(6),vo];torso=m.body('torso').id
    motors=[m.joint(x).id for x in c['independent_ids']];mv=m.jnt_dofadr[motors]
    errors={key:0. for key in ['source_total_energy_J','pin_total_energy_J','source_body_energy_J',
        'body_energy_sum_J','contact_generalized_mapping_mixed_SI','actuator_generalized_mapping_mixed_SI','source_equation_relative','source_equation_absolute_mixed_SI',
        'native_equation_relative','replayed_acceleration','replayed_motor_force_N',
        'replayed_foot_force_N','source_mass_vs_pin','source_bias_vs_pin','raw_ledger_J',
        'source_body_force_balance_N','source_body_moment_balance_Nm','linear_impulse_balance_Ns']}
    def update(key,value):errors[key]=max(errors[key],float(value))
    out={key:[] for key in ['time','body_energy','armature_energy','body_tree_wrench_com',
        'body_contact_wrench_com','body_actuator_wrench_com','loop_generalized','contact_generalized',
        'limit_generalized','actuator_generalized','external_generalized','passive_generalized',
        'inertial_generalized','bias_generalized','linear_momentum']}
    raw_rows=[];samples=np.unique(np.r_[np.linspace(0,len(a['time'])-1,51).astype(int),np.arange(min(61,len(a['time'])))])
    detailed_indices=range(len(a['time'])) if export_csv else samples
    for k in detailed_indices:
        d.qpos[:]=a['q'][k];d.qvel[:]=a['v'][k];d.act[:]=a['act'][k];d.ctrl[:]=a['command'][k]
        d.xfrc_applied[:]=0;d.xfrc_applied[torso,:3]=a['push'][k]
        mujoco.mj_forward(m,d);mujoco.mj_rnePostConstraint(m,d)
        bodyenergy=[];tree=[];contactw=np.zeros((m.nbody,6));actw=np.zeros_like(contactw);momentum=np.zeros(3)
        native_mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,native_mass,d.qM)
        qparts={}
        for key,mask in [('loop',d.efc_type==0),('contact',d.efc_type>=5),('limit',(d.efc_type==3)|(d.efc_type==4))]:
            f=d.efc_force.copy();f[~mask]=0;v=np.zeros(m.nv);mujoco.mj_mulJacTVec(m,d,v,f);qparts[key]=v
        explicit_contact=np.zeros(m.nv)
        for n in range(d.ncon):
            con=d.contact[n];raw=np.zeros(6);mujoco.mj_contactForce(m,d,n,raw)
            force=con.frame.reshape(3,3).T@raw[:3];moment=con.frame.reshape(3,3).T@raw[3:]
            for bid,sign in [(m.geom_bodyid[con.geom1],-1),(m.geom_bodyid[con.geom2],1)]:
                if bid:
                    mujoco.mj_applyFT(m,d,sign*force,sign*moment,con.pos,bid,explicit_contact)
                    contactw[bid,:3]+=sign*force
                    contactw[bid,3:]+=sign*(moment+np.cross(con.pos-d.xipos[bid],force))
        explicit_actuator=np.zeros(m.nv)
        for j,force in zip(motors,d.actuator_force):
            child=m.jnt_bodyid[j];parent=m.body_parentid[child];F=d.xaxis[j]*force;point=d.xanchor[j]
            for bid,sign in [(child,1),(parent,-1)]:
                if bid:
                    mujoco.mj_applyFT(m,d,sign*F,np.zeros(3),point,bid,explicit_actuator)
                    actw[bid,:3]+=sign*F;actw[bid,3:]+=sign*np.cross(point-d.xipos[bid],F)
        external=np.zeros(m.nv);mujoco.mj_applyFT(m,d,a['push'][k],np.zeros(3),d.xipos[torso],torso,external)
        for bid in bodies:
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,d,jp,jr,bid)
            vc=jp@d.qvel;omega=jr@d.qvel;localomega=d.ximat[bid].reshape(3,3).T@omega
            momentum+=m.body_mass[bid]*vc
            kinetic=.5*m.body_mass[bid]*(vc@vc)+.5*np.dot(m.body_inertia[bid],localomega**2)
            potential=-m.body_mass[bid]*m.opt.gravity@d.xipos[bid];bodyenergy.append([kinetic,potential])
            # MuJoCo c-frame is centered at the root subtree COM. Shift the
            # reported interaction wrench to this body's COM; world axes.
            native=d.cfrc_int[bid];origin=d.subtree_com[m.body_rootid[bid]]
            tree.append(np.r_[native[3:],native[:3]+np.cross(origin-d.xipos[bid],native[3:])])
        bodyenergy=np.array(bodyenergy);armature=.5*np.dot(m.dof_armature,d.qvel**2)
        update('body_energy_sum_J',abs(bodyenergy.sum()+armature-d.energy.sum()))
        update('contact_generalized_mapping_mixed_SI',max(abs(explicit_contact-qparts['contact'])))
        update('actuator_generalized_mapping_mixed_SI',max(abs(explicit_actuator-d.qfrc_actuator)))
        load=d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint+external
        inertial=native_mass@d.qacc
        update('native_equation_relative',np.max(abs(inertial+d.qfrc_bias-load))/max(1.,np.max(abs(load))))
        update('replayed_acceleration',max(abs(d.qacc-a['a'][k])))
        update('replayed_motor_force_N',max(abs(d.actuator_force-a['act'][k])))
        feet=[m.body(x).id for x in ['left_ankle_roll','right_ankle_roll']]
        update('replayed_foot_force_N',np.max(abs(contactw[feet,:3]-a['foot_force'][k])))
        if k in samples:
            R=Rotation.from_quat(d.qpos[3:7][[1,2,3,0]]).as_matrix()
            T=np.eye(m.nv);T[3:6,3:6]=R.T
            v=np.r_[d.qvel[:3],R@d.qvel[3:6],d.qvel[vo]]
            acc=np.r_[d.qacc[:3],R@d.qacc[3:6],d.qacc[vo]]
            s=source.evaluate(d.qpos[qo],d.qpos[:3],R,v,m.opt.gravity)
            e=add_drive_terms(s,c);energy=e['potential_energy']+e['kinetic_energy']
            update('source_total_energy_J',abs(energy-d.energy.sum()))
            pp=pin.evaluate(d.qpos[qo],d.qpos[:3],R,v,m.opt.gravity)
            update('pin_total_energy_J',abs(pp['potential_energy']+pp['kinetic_energy']+armature-d.energy.sum()))
            update('source_mass_vs_pin',np.max(abs(s['mass_matrix']-pp['mass_matrix'])))
            update('source_bias_vs_pin',np.max(abs(s['bias_forces']-pp['bias_forces'])))
            rhs=T.T@(d.qfrc_actuator+d.qfrc_constraint+external)[order]
            residual=e['mass_matrix']@acc+e['bias_forces']-rhs
            # Scale by equation terms before cancellation, since net force can
            # approach zero in equilibrium. Native principal-axis serialization
            # differs slightly from the source inertia matrices.
            scale=max(1.,np.max(abs(rhs)),np.max(abs(e['bias_forces'])),np.max(abs(e['mass_matrix']@acc)))
            update('source_equation_relative',np.max(abs(residual))/scale)
            update('source_equation_absolute_mixed_SI',np.max(abs(residual)))
            for i,b in enumerate(c['bodies']):
                pose=s['poses'][b['id']];vc=s['body_com_jacobians'][b['id']]@v;w=s['body_velocities'][b['id']][3:]
                I=pose[:3,:3]@b['inertia_com_kg_m2']@pose[:3,:3].T
                be=[.5*b['mass_kg']*(vc@vc)+.5*w@I@w,-b['mass_kg']*m.opt.gravity@(pose[:3,3]+pose[:3,:3]@b['com_m'])]
                update('source_body_energy_J',np.max(abs(bodyenergy[i]-be)))
                bid=m.body(b['id']).id;children=np.where(m.body_parentid==bid)[0]
                net=d.cfrc_int[bid]+d.cfrc_ext[bid]-d.cfrc_int[children].sum(axis=0)
                origin=d.subtree_com[m.body_rootid[bid]]
                moment=net[:3]+np.cross(origin-d.xipos[bid],net[3:])
                acceleration=s['body_com_jacobians'][b['id']]@acc+s['body_com_jacobian_dots'][b['id']]@v
                alpha=s['jacobians'][b['id']][3:]@acc+s['body_jacobian_dots'][b['id']][3:]@v
                update('source_body_force_balance_N',max(abs(net[3:]+b['mass_kg']*m.opt.gravity-b['mass_kg']*acceleration)))
                update('source_body_moment_balance_Nm',max(abs(moment-I@alpha-np.cross(w,I@w))))
            raw_rows.append(dict(time_s=float(a['time'][k]),independent_energy_J=float(energy),native_energy_J=float(d.energy.sum()),equation_relative=float(np.max(abs(residual))/scale)))
        values=[a['time'][k],bodyenergy,armature,tree,contactw[bodies],actw[bodies],qparts['loop'],qparts['contact'],qparts['limit'],d.qfrc_actuator.copy(),external,d.qfrc_passive.copy(),inertial,d.qfrc_bias.copy(),momentum]
        for key,value in zip(out,values):out[key].append(value)
    out={k:np.array(v) for k,v in out.items()}
    h=a['history'];work=np.vstack([np.zeros(6),np.cumsum(.5*r['timestep_s']*(h[1:,3:9]+h[:-1,3:9]),axis=0)])
    ledger=h[:,1:3].sum(axis=1)-h[0,1:3].sum()-work.sum(axis=1)
    update('raw_ledger_J',abs(max(abs(ledger))-r['maximum_energy_ledger_error_J']))
    force=h[:,9:12]+h[:,12:15]+m.body_mass.sum()*m.opt.gravity
    impulse=np.vstack([np.zeros(3),np.cumsum(.5*r['timestep_s']*(force[1:]+force[:-1]),axis=0)])
    sample_ids=np.rint(out['time']/r['timestep_s']).astype(int)
    update('linear_impulse_balance_Ns',np.max(abs(out['linear_momentum']-out['linear_momentum'][0]-impulse[sample_ids])))
    np.savez_compressed(ROOT/'results'/f'{name}_body_forces.npz',**out)
    limits={'source_total_energy_J':1e-6,'pin_total_energy_J':1e-6,'source_body_energy_J':1e-7,'body_energy_sum_J':1e-7,
        'contact_generalized_mapping_mixed_SI':1e-7,'actuator_generalized_mapping_mixed_SI':1e-7,'source_equation_relative':1e-5,'source_equation_absolute_mixed_SI':.01,
        'native_equation_relative':1e-8,'replayed_acceleration':.01,'replayed_motor_force_N':1e-10,
        'replayed_foot_force_N':.01,'source_mass_vs_pin':1e-9,'source_bias_vs_pin':1e-8,'raw_ledger_J':1e-10,
        'source_body_force_balance_N':.005,'source_body_moment_balance_Nm':1e-6,'linear_impulse_balance_Ns':.05}
    result=dict(name=name,snapshots=len(out['time']),independent_snapshots=len(raw_rows),errors=errors,
                gates=[dict(name=key,passed=errors[key]<=lim,value=errors[key],limit=lim) for key,lim in limits.items()],records=raw_rows,
                body_order=[b['id'] for b in c['bodies']],generalized_coordinate_order=['base_vx','base_vy','base_vz','base_wx_local','base_wy_local','base_wz_local']+[m.joint(i).name for i in range(1,m.njnt)],
                wrench_convention='world axes; force XYZ N then moment XYZ Nm at each body COM; native tree-parent reaction is solver/model dependent, not a uniquely identified physical pin load')
    result['status']='PASS' if all(x['passed'] for x in result['gates']) else 'FAIL'
    (ROOT/'results'/f'{name}_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if export_csv:write_csv(a,out,c,name)
    print(name,'independent audit',result['status'],errors,flush=True);return result

def write_csv(a,out,c,name):
    folder=ROOT/'results/tables';folder.mkdir(exist_ok=True)
    def write(filename,header,rows):
        with (folder/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)
    # Downsampled readable telemetry; full physics-rate ledger is in the NPZ.
    step=max(1,round(.005/(a['history'][1,0]-a['history'][0,0])))
    write(name+'_telemetry.csv',HISTORY_COLUMNS,a['history'][::step])
    write(name+'_actuators.csv',['time_s','actuator','command_N','actual_force_N','velocity_m_s','power_W','work_J','positive_work_J','negative_work_J'],
          ([t,c['actuators'][j]['id'],a['command'][i,j],a['act'][i,j],a['motor_velocity'][i,j],a['motor_power'][i,j],a['motor_work'][i,j],a['motor_positive_work'][i,j],a['motor_negative_work'][i,j]] for i,t in enumerate(a['time']) for j in range(12)))
    write(name+'_bodies.csv',['time_s','body','kinetic_J','potential_J','parent_Fx_N','parent_Fy_N','parent_Fz_N','parent_Mx_Nm','parent_My_Nm','parent_Mz_Nm','contact_Fx_N','contact_Fy_N','contact_Fz_N','contact_Mx_Nm','contact_My_Nm','contact_Mz_Nm','actuator_Fx_N','actuator_Fy_N','actuator_Fz_N','actuator_Mx_Nm','actuator_My_Nm','actuator_Mz_Nm'],
          ([t,b['id'],*out['body_energy'][i,j],*out['body_tree_wrench_com'][i,j],*out['body_contact_wrench_com'][i,j],*out['body_actuator_wrench_com'][i,j]] for i,t in enumerate(out['time']) for j,b in enumerate(c['bodies'])))
    write(name+'_feet.csv',['time_s','foot','Fx_N','Fy_N','Fz_N','Mx_at_COM_Nm','My_at_COM_Nm','Mz_at_COM_Nm','x_m','y_m','z_m'],
          ([t,side,*a['foot_force'][i,j],*a['foot_moment'][i,j],*a['foot_position'][i,j]] for i,t in enumerate(a['time']) for j,side in enumerate(['left','right'])))
    # These are deliberately scenarios, not a calibrated battery model.
    Wplus=float(a['motor_positive_work'][-1].sum());Wminus=-float(a['motor_negative_work'][-1].sum())
    write('illustrative_drive_energy_scenarios.csv',['assumed_motoring_efficiency','assumed_regeneration_efficiency','positive_mechanical_work_J','absorbed_mechanical_work_J','illustrative_supply_energy_J','scope'],
          ([eta,regen,Wplus,Wminus,Wplus/eta-regen*Wminus,'Illustrative only; excludes idle power, electronics, temperature and battery losses'] for eta in [.7,.85,.95] for regen in [0,.5,.8]))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',default='landing_nominal');p.add_argument('--no-csv',action='store_true');args=p.parse_args();audit(args.name,not args.no_csv)
