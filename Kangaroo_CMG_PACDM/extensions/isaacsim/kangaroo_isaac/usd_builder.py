"""Author the source physical graph directly as USD/PhysX, not through URDF.

Import this module only after SimulationApp has started. Nothing in this file
sets joint positions during a running experiment. Universal cut body order is
intentionally reversed; see docs/PHYSICS_PORT.md and the D6 equivalence tests.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, PhysxSchema, Vt
from .model import ROOT, FOOT_NAMES, triangular_face_counts, inertias_match

from .scene_paths import ROBOT, BODIES, SCENE, GROUND, GROUND_CENTER_M, GROUND_SIZE_M


def vec(x): return Gf.Vec3f(*map(float,x))
def quat(R):
    x,y,z,w=Rotation.from_matrix(R).as_quat()
    return Gf.Quatf(float(w),Gf.Vec3f(float(x),float(y),float(z)))


def xform(prim,position,rotation,scale=None):
    x=UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(quat(rotation))
    if scale is not None:x.AddScaleOp().Set(vec(scale))


def set_frames(joint,b0,b1,A,B,excluded=False):
    joint.CreateBody0Rel().SetTargets([Sdf.Path(BODIES+'/'+b0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(BODIES+'/'+b1)])
    joint.CreateLocalPos0Attr(vec(A[:3,3]));joint.CreateLocalRot0Attr(quat(A[:3,:3]))
    joint.CreateLocalPos1Attr(vec(B[:3,3]));joint.CreateLocalRot1Attr(quat(B[:3,:3]))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(excluded)
    joint.CreateJointEnabledAttr(True)
    # No break threshold, motor, stabilization projection or mass scaling.


def physical_material(stage,cfg):
    material=UsdShade.Material.Define(stage,'/World/Materials/GroundContact')
    p=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    p.CreateStaticFrictionAttr(cfg.friction);p.CreateDynamicFrictionAttr(cfg.friction)
    p.CreateRestitutionAttr(0.)
    p=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    p.CreateFrictionCombineModeAttr('average');p.CreateRestitutionCombineModeAttr('min')
    if cfg.contact_model=='source_compliance':
        # Source MuJoCo soft contact as a PhysX implicit acceleration spring;
        # derivation and limits in contact_model.py. Fail closed if unsupported.
        # Stiffness is authored FIRST: omni.physx warns if damping or the
        # acceleration-spring flag is updated on a material whose stiffness is 0.
        for name,value in (('CompliantContactStiffness',float(cfg.contact_stiffness_per_s2)),
                           ('CompliantContactDamping',float(cfg.contact_damping_per_s)),
                           ('CompliantContactAccelerationSpring',True)):
            setter=getattr(p,'Create'+name+'Attr',None)
            if not callable(setter):
                raise RuntimeError(f'Installed PhysxSchema lacks Create{name}Attr; compliant source contact cannot be authored')
            setter(value)
    return material


def collision(prim,material,cfg):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(not cfg.no_contact)
    p=PhysxSchema.PhysxCollisionAPI.Apply(prim)
    p.CreateContactOffsetAttr(cfg.contact_offset_m);p.CreateRestOffsetAttr(cfg.rest_offset_m)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')


def build(stage,model,ref,cfg,visuals=True):
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage,1.)
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage,ROBOT);UsdGeom.Scope.Define(stage,BODIES)
    UsdGeom.Scope.Define(stage,ROBOT+'/Joints');UsdGeom.Scope.Define(stage,ROBOT+'/Loops')
    scene=UsdPhysics.Scene.Define(stage,SCENE)
    scene.CreateGravityDirectionAttr(vec([0,0,-1]));scene.CreateGravityMagnitudeAttr(9.81)
    ps=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps.CreateEnableGPUDynamicsAttr(False);ps.CreateBroadphaseTypeAttr('MBP')
    from .solver_settings import author_solver_settings
    author_solver_settings(ps,cfg)
    ps.CreateEnableStabilizationAttr(False)
    ps.CreateTimeStepsPerSecondAttr(round(1/cfg.dt_s))
    # Improve reproducibility without changing the physical masses or inertias.
    ps.CreateEnableEnhancedDeterminismAttr(True)
    base=ref.data['base'][0].copy();base[2]+=cfg.drop_height_m
    rotation=Rotation.from_rotvec(ref.data['rotvec'][0]).as_matrix()
    P=model.fk(ref.data['q'][0],base,rotation)
    for i,name in enumerate(model.names):
        prim=UsdGeom.Xform.Define(stage,BODIES+'/'+name).GetPrim()
        xform(prim,P[i,:3,3],P[i,:3,:3])
        rb=UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True);rb.CreateKinematicEnabledAttr(False)
        rb.CreateVelocityAttr(vec([0,0,0]));rb.CreateAngularVelocityAttr(vec([0,0,0]))
        mass=UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(model.mass[i]));mass.CreateCenterOfMassAttr(vec(model.com_local[i]))
        eig,Q=np.linalg.eigh(model.I_local[i])
        if np.linalg.det(Q)<0:Q[:,0]*=-1
        mass.CreateDiagonalInertiaAttr(vec(eig));mass.CreatePrincipalAxesAttr(quat(Q))
        px=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        px.CreateDisableGravityAttr(False)
        px.CreateLinearDampingAttr(0.);px.CreateAngularDampingAttr(0.)
        px.CreateEnableGyroscopicForcesAttr(True)
        px.CreateSleepThresholdAttr(0.)
        px.CreateSolverPositionIterationCountAttr(cfg.solver_position_iterations)
        px.CreateSolverVelocityIterationCountAttr(cfg.solver_velocity_iterations)
        # Source-scale bodies can move rapidly during impact. Do not silently
        # impose the comparatively low default angular-velocity cap.
        px.CreateMaxAngularVelocityAttr(1e7)
        prim.CreateAttribute('cmg:bodyId',Sdf.ValueTypeNames.String,custom=True).Set(name)
    root=stage.GetPrimAtPath(BODIES+'/'+model.c['root_body'])
    UsdPhysics.ArticulationRootAPI.Apply(root)
    ar=PhysxSchema.PhysxArticulationAPI.Apply(root)
    ar.CreateEnabledSelfCollisionsAttr(False)
    ar.CreateSolverPositionIterationCountAttr(cfg.solver_position_iterations)
    ar.CreateSolverVelocityIterationCountAttr(cfg.solver_velocity_iterations)
    ar.CreateSleepThresholdAttr(0.)
    for j in model.c['joints']:
        typ=j['type'];path=ROBOT+'/Joints/'+j['id']
        cls={'fixed':UsdPhysics.FixedJoint,'prismatic':UsdPhysics.PrismaticJoint,'revolute':UsdPhysics.RevoluteJoint}[typ]
        joint=cls.Define(stage,path)
        A,B=model.joint_frames(j)
        set_frames(joint,j['base_body'],j['follower_body'],A,B)
        if typ!='fixed':
            joint.CreateAxisAttr('X');index=model.qi[j['id']]
            units=1. if typ=='prismatic' else 180/np.pi
            joint.CreateLowerLimitAttr(float(model.lower[index]*units))
            joint.CreateUpperLimitAttr(float(model.upper[index]*units))
            p=PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
            p.CreateArmatureAttr(float(model.armature[index]))
            p.CreateJointFrictionAttr(0.)  # Effective v22 setting: disabled.
            p.CreateMaxJointVelocityAttr(1e6)
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear' if typ=='prismatic' else 'angular')
            drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(float(model.damping[index]/units))
            drive.CreateTargetVelocityAttr(0.);drive.CreateTargetPositionAttr(0.)
            drive.CreateMaxForceAttr(1e10)
            # The runner explicitly sets and reads back SI damping/armature
            # via the native tensor API before the first experiment step.
    if not cfg.no_loops:
        for k,cut in enumerate(model.c['closures']):
            path=ROBOT+'/Loops/cut_'+str(k).zfill(2)
            A=np.eye(4);B=np.eye(4)
            A[:3,3]=cut['point1_m'];B[:3,3]=cut['point2_m']
            if cut['type']=='universal':
                A[:3,:3]=cut['frame1_R'];B[:3,:3]=cut['frame2_R']
                joint=UsdPhysics.Joint.Define(stage,path)  # Native regular D6.
                # Free twist X + free swing Y constrains actor1.X dot actor0.Y.
                # Reverse source bodies so this equals source A.X dot B.Y.
                set_frames(joint,cut['body2'],cut['body1'],B,A,excluded=True)
                for axis in ('transX','transY','transZ','rotZ'):
                    limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
                    limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)  # low>high: locked.
                # No LimitAPI on rotX/rotY -> free in the USD generic-joint schema.
            else:
                joint=UsdPhysics.SphericalJoint.Define(stage,path)
                set_frames(joint,cut['body1'],cut['body2'],A,B,excluded=True)
            joint.GetPrim().CreateAttribute('cmg:closureIndex',Sdf.ValueTypeNames.Int,custom=True).Set(k)
    material=physical_material(stage,cfg)
    floor=UsdGeom.Cube.Define(stage,GROUND);floor.CreateSizeAttr(1.)
    xform(floor.GetPrim(),GROUND_CENTER_M,np.eye(3),GROUND_SIZE_M)
    floor.CreateDisplayColorAttr([vec([.22,.24,.27])]);collision(floor.GetPrim(),material,cfg)
    for geom in model.visual['collisions']:
        prim=UsdGeom.Cube.Define(stage,BODIES+'/'+geom['body']+'/SoleCollision')
        prim.CreateSizeAttr(1.);q=geom['quaternion_wxyz']
        R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
        xform(prim.GetPrim(),geom['position'],R,2*np.asarray(geom['half_extents']))
        collision(prim.GetPrim(),material,cfg)
        prim.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        contact=PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(BODIES+'/'+geom['body']))
        contact.CreateThresholdAttr(0.)
    if visuals:
        cache={}
        for k,geom in enumerate(model.visual['visuals']):
            name=geom['mesh']
            if name not in cache:
                with np.load(ROOT/'assets'/model.visual['meshes'][name]['cache']) as a:
                    cache[name]=(a['points'].copy(),a['indices'].copy())
            pts,tri=cache[name]
            mesh=UsdGeom.Mesh.Define(stage,BODIES+'/'+geom['body']+'/Visual_'+str(k))
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(triangular_face_counts(tri)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.ravel()))
            mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([vec(geom['rgba'][:3])])
            q=geom['quaternion_wxyz'];R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
            xform(mesh.GetPrim(),geom['position'],R)
        light=UsdLux.DomeLight.Define(stage,'/World/Lighting/Dome');light.CreateIntensityAttr(700.)
        sun=UsdLux.DistantLight.Define(stage,'/World/Lighting/Sun');sun.CreateIntensityAttr(2000.)
        xform(sun.GetPrim(),[0,0,4],Rotation.from_euler('xyz',[35,-30,15],degrees=True).as_matrix())
    return P


def audit_stage(stage,model,cfg):
    """Audit actual authored USD, before PhysX starts. Native readback is separate."""
    from .solver_settings import read_solver_settings
    solver_settings=read_solver_settings(PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(SCENE)),cfg)
    failures=[];body_count=0;tree_count=0;loop_count=0;native_mass=0.
    for name in model.names:
        p=stage.GetPrimAtPath(BODIES+'/'+name)
        if not p or not p.HasAPI(UsdPhysics.RigidBodyAPI):failures.append('Missing body '+name);continue
        body_count+=1;native_mass+=float(UsdPhysics.MassAPI(p).GetMassAttr().Get())
        i=model.bi[name];a=UsdPhysics.MassAPI(p)
        D=np.asarray(a.GetDiagonalInertiaAttr().Get());q=a.GetPrincipalAxesAttr().Get()
        Q=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
        if not inertias_match(Q@np.diag(D)@Q.T,model.I_local[i]):
            failures.append('USD inertia mismatch '+name)
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Joint):
            j=UsdPhysics.Joint(p);s=str(p.GetPath())
            if s.startswith(ROBOT+'/Joints/'):tree_count+=1
            elif s.startswith(ROBOT+'/Loops/'):
                loop_count+=1
                if not j.GetExcludeFromArticulationAttr().Get():failures.append('Loop not excluded '+s)
            if not j.GetBody0Rel().GetTargets() or not j.GetBody1Rel().GetTargets():
                failures.append('Unexpected world-fixed joint '+s)
    if body_count!=78 or tree_count!=77 or loop_count!=(0 if cfg.no_loops else 24):
        failures.append('Body/tree/loop count mismatch')
    if not np.isclose(native_mass,model.mass.sum(),rtol=2e-6):failures.append('Total mass mismatch')
    from .contact_model import describe
    material=stage.GetPrimAtPath('/World/Materials/GroundContact')
    authored={}
    if material:
        m=PhysxSchema.PhysxMaterialAPI(material)
        for name in ('CompliantContactAccelerationSpring','CompliantContactStiffness','CompliantContactDamping'):
            getter=getattr(m,'Get'+name+'Attr',None)
            attr=getter() if callable(getter) else None
            authored[name]=attr.Get() if attr is not None and attr.IsValid() and attr.HasAuthoredValue() else None
    else:failures.append('Missing ground contact material')
    if cfg.contact_model=='source_compliance':
        wanted={'CompliantContactAccelerationSpring':True,'CompliantContactStiffness':cfg.contact_stiffness_per_s2,
                'CompliantContactDamping':cfg.contact_damping_per_s}
        for name,value in wanted.items():
            got=authored.get(name)
            if got is None or (got!=value if isinstance(value,bool) else not np.isclose(float(got),value,rtol=1e-6)):
                failures.append(f'Compliant contact readback mismatch {name}: {got!r} != {value!r}')
    elif any(v not in (None,False,0.,0) for v in authored.values()):
        failures.append('Rigid contact model has compliant material attributes')
    result={'status':'PASS' if not failures else 'FAIL','body_count':body_count,
            'contact_model':describe(cfg),'authored_contact_material':authored,
            'tree_joint_count':tree_count,'closure_joint_count':loop_count,
            'mass_kg':native_mass,'universal_implementation':'D6 reversed-body zero-swing',
            'base_fixed':False,'solver_settings':solver_settings,'failures':failures}
    if failures:raise RuntimeError('USD audit failed: '+repr(result))
    return result
