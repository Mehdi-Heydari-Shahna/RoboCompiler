"""Kinematic receiver feasibility and conservative bucket/bay clearance.

These prescribed reference configurations are geometry checks only. The task
plant remains effort driven; successful deposition requires a native run.
"""
from project import *
import itertools
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from benchmark_model import context
from mobile_mission import Mission


def audit(model_path=RESULTS/'assets'/'coarse.xml'):
    cmg,mapping,e,a,r,inverse=context();mission=Mission(e,r)
    model=mujoco.MjModel.from_xml_path(str(model_path));data=mujoco.MjData(model)
    qadr=np.array([model.joint(s).qposadr[0] for s in e.tree_ids])
    bucket=model.body('body_56').id
    geoms=[i for i in range(model.ngeom) if model.geom_contype[i]!=0 and model.body(model.geom_bodyid[i]).name.startswith('body_')]
    bucket_geoms=[i for i in geoms if model.geom_bodyid[i]==bucket]
    vertices={}
    for gid in geoms:
        typ=model.geom_type[gid]
        if typ==mujoco.mjtGeom.mjGEOM_BOX:
            vertices[gid]=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*model.geom_size[gid]
        elif typ==mujoco.mjtGeom.mjGEOM_MESH:
            mid=model.geom_dataid[gid];start=model.mesh_vertadr[mid];number=model.mesh_vertnum[mid]
            vertices[gid]=np.array(model.mesh_vert[start:start+number])
        else:
            raise ValueError('Unsupported source collision geometry in clearance audit')
    source_mass=sum(b['mass_kg'] for b in cmg['bodies']);rows=[]
    center=mission.depot_center[:2];hx,hy=mission.depot_half_size
    walls=[(np.r_[center+offset,.2],np.asarray(size)) for offset,size in
           [([-hx-.05,0],[.05,hy+.10,.2]),([hx+.05,0],[.05,hy+.10,.2]),
            ([0,-hy-.05],[hx,.05,.2]),([0,hy+.05],[hx,.05,.2])]]
    minimum_wall_separation=float('inf');worst_wall_pair=None
    for t in np.linspace(32.,64.,161):
        arm,target,_=mission.reference(float(t));q=r.reconstruct(arm[0]);poses=e.source.evaluate(q,np.zeros(len(q)),[0,0,-9.81])['poses']
        com=sum(b['mass_kg']*(poses[b['id']][:3,:3]@np.asarray(b['com_m'])+poses[b['id']][:3,3]) for b in cmg['bodies'])/source_mass
        T=poses['body_56'];mouth=T[:3,:3]@mission.mouth_local+T[:3,3]
        loaded_com=(source_mass*com+77.*mouth)/(source_mass+77.)
        R=Rotation.from_euler('z',target[2]).as_matrix();data.qpos[:3]=R@np.array([2.1126661182113513,.45390271998685244,0])+np.r_[target[:2],0]
        data.qpos[3:7]=np.roll(Rotation.from_matrix(R@Rotation.from_euler('z',np.pi/2).as_matrix()).as_quat(),1)
        data.qpos[qadr]=q;mujoco.mj_kinematics(model,data)
        world_points={gid:vertices[gid]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid] for gid in geoms}
        world_vertices=np.vstack([world_points[gid] for gid in bucket_geoms])
        for gid,points in world_points.items():
            low=points.min(axis=0);high=points.max(axis=0)
            for wall_index,(wall_center,wall_half) in enumerate(walls):
                separation=np.maximum.reduce([wall_center-wall_half-high,low-(wall_center+wall_half),np.zeros(3)])
                lower_bound=float(np.linalg.norm(separation))
                if lower_bound<minimum_wall_separation:
                    minimum_wall_separation=lower_bound
                    worst_wall_pair=dict(time_s=float(t),geom=model.geom(gid).name,wall_index=wall_index,axis_separation_m=separation)
        rows.append(dict(time_s=float(t),bucket_minimum_z_m=float(world_vertices[:,2].min()),
                         source_whole_body_COM_m=com.tolist(),source_loaded_COM_m=loaded_com.tolist()))
    minimum=min(x['bucket_minimum_z_m'] for x in rows)
    coms=np.array([x['source_loaded_COM_m'] for x in rows])
    report=dict(scope='Kinematic pre-registration; conservative entire collision-bucket vertical clearance above a possible static 0.40 m receiving-bay wall. Native settling/contact validation is separate.',
        receiver_center_m=mission.depot_center,receiver_half_size_m=mission.depot_half_size,
        independent_guard_margin_rad=mission.receiver_plan['minimum_independent_guard_margin_rad'],
        mouth_curve_maximum_residual_SI=mission.receiver_plan['maximum_interpolated_position_pitch_residual_SI'],
        bucket_minimum_reference_z_m=minimum,proposed_wall_height_m=.4,
        minimum_reference_bucket_wall_vertical_clearance_m=minimum-.4,
        minimum_source_collision_AABB_wall_separation_m=minimum_wall_separation,worst_source_wall_pair=worst_wall_pair,
        maximum_loaded_COM_source_x_m=float(coms[:,0].max()),maximum_loaded_COM_source_abs_y_m=float(abs(coms[:,1]).max()),
        assumed_payload_for_static_COM_kg=77.,source_whole_body_mass_kg=source_mass,
        rows=rows)
    save_json(ROOT/'outputs/mobility_control/receiver_geometry_audit.json',report)
    return report

if __name__=='__main__':
    report=audit();print({k:v for k,v in report.items() if k!='rows'})
