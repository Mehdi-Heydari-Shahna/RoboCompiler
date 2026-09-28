"""CMG -> USD/PhysX scene. Import only AFTER SimulationApp exists.

No asset importer, downloaded Franka USD, or inferred link inertia is used.
All source mesh arrays are bundled. Physics state is never replayed from NPZ.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .rigid import RigidTree
from .geometry import joint_frames, collision_exclusions

# Contact offsets (rest offsets stay zero, so touching still means distance 0).
# PhysX generates contacts for a pair within the SUM of both offsets.
# Robot and cartridge keep v1's 0.2 mm, so cartridge-pad pairs stay at 0.4 mm:
# with 2.2 mm, speculative (not touching) pad points became friction anchors and
# locked a cartridge that had tipped 3.7 deg during the offset pick.
# Static task colliders use 1.8 mm, so cartridge-plinth/socket pairs reach 2 mm:
# with v1's 0.4 mm a slowly lowered, microscopically tilted cartridge kept the
# speculative 1-3 point manifold created at first detection (PCM and SAT), then
# tipped about that edge after release, sank ~0.4 mm into the plinth and was
# ejected. Reproduced with PhysX 5.3.1; pair distances >= 1 mm gave four points.
ROBOT_CONTACT_OFFSET = .0002
OBJECT_CONTACT_OFFSET = .0002
SCENE_CONTACT_OFFSET = .0018

# Pick-plinth friction, combined with 'min' so the pair value is 0.2 (v1: 1.0
# 'average' = 0.9 with the cartridge; MuJoCo source: 1.0). In the offset
# pickups one pad reaches the cartridge first and pushes it sideways 49-66 mm
# above the plinth; above a friction of half-width/height = 0.34 it can tip
# instead of sliding (twin: 2.4 deg for the tight-socket cartridge). The source
# MuJoCo model tips it as well (2.6 deg), but MuJoCo's soft friction realigns
# it in the grasp within ~7 s (0.25 deg at insertion). PhysX friction is rigid
# Coulomb friction: the pads' friction couple (2*mu*N*half-width ~0.36 N m)
# exceeds the jaws' aligning couple (N*17 mm ~0.16 N m), so the tilt stays and
# the cartridge jams in the 1 mm-clearance tight socket. At 0.2 it slides.
PICK_PLINTH_FRICTION = .2


def build_scene(root, stage, cmg, q0, mode, case, visuals=True, joint_offsets=None):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, UsdLux, PhysxSchema, Vt
    root = Path(root)
    ids = list(cmg['coordinate_ids'])
    offsets = np.zeros(len(ids)) if joint_offsets is None else np.asarray(joint_offsets, float)
    npos, nvel = int(case['position_iterations']), int(case['velocity_iterations'])
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    physics = UsdPhysics.Scene.Define(stage, '/World/physicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    physics.CreateGravityMagnitudeAttr(9.81)
    pscene = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    pscene.CreateEnableGPUDynamicsAttr(False)
    pscene.CreateBroadphaseTypeAttr('MBP')
    pscene.CreateSolverTypeAttr(case['solver'])
    pscene.CreateTimeStepsPerSecondAttr(round(1./case['dt']))
    pscene.CreateEnableCCDAttr(True)
    # Solver type and per-actor iteration counts come from cases.py. TGS would
    # advance articulation joint angles once per position iteration (sub-step
    # dt/N) in float32; PGS integrates positions once per step. The scene's
    # default min/max iteration clamps (1..255) keep the declared counts.
    # No Fabric dependency for headless tests. GUI explicitly synchronizes
    # PhysX transforms before a render-only application update.
    robot_path='/World/Panda'
    UsdGeom.Xform.Define(stage,robot_path)
    UsdGeom.Scope.Define(stage,robot_path+'/joints')
    bodies = {b['id']:b for b in cmg['bodies']}
    paths = {n:robot_path+'/'+n for n in bodies if n!='world'}
    poses=RigidTree(cmg).poses(q0)
    collider_labels={}
    material_cache={}

    def quat(R):
        q=Rotation.from_matrix(np.asarray(R)).as_quat()
        return Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3])))

    def transform(prim,T,scale=None):
        xf=UsdGeom.Xformable(prim)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float,T[:3,3])))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(quat(T[:3,:3]))
        if scale is not None:
            xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*map(float,scale)))

    def material(mu,combine='average'):
        key=(float(mu),combine)
        if key in material_cache:
            return material_cache[key]
        mat=UsdShade.Material.Define(stage,f'/World/Materials/physics_{len(material_cache)}')
        api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(mu));api.CreateDynamicFrictionAttr(float(mu))
        api.CreateRestitutionAttr(0.)
        pm=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        # "min" beats "average" in PhysX material-combination priority:
        # with the object's .8, the .5 low-friction pad remains .5 (and the
        # pick plinth .2). Average combination is NOT MuJoCo's max rule.
        pm.CreateFrictionCombineModeAttr(combine)
        pm.CreateRestitutionCombineModeAttr('min')
        material_cache[key]=mat
        return mat

    def collision(prim,label,mu=.8,pad=False,contact_offset=ROBOT_CONTACT_OFFSET,combine='average'):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(mode=='contact')
        api=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(float(contact_offset))
        api.CreateRestOffsetAttr(0.)
        if pad:
            api.CreateTorsionalPatchRadiusAttr(.005)
            api.CreateMinTorsionalPatchRadiusAttr(.005)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(mu,'min' if pad else combine),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose='physics')
        collider_labels[str(prim.GetPath())]=label

    def mass(prim,m,com,inertia):
        api=UsdPhysics.MassAPI.Apply(prim)
        d,R=np.linalg.eigh(np.asarray(inertia,float))
        if min(d)<=0. or m<=0.:
            raise ValueError('A dynamic rigid body needs positive mass and inertia')
        if np.linalg.det(R)<0.:
            R[:,0]*=-1.
        api.CreateMassAttr(float(m))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,com)))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,d)))
        api.CreatePrincipalAxesAttr(quat(R))

    def rigid(prim):
        rb=UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True);rb.CreateKinematicEnabledAttr(False)
        px=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        px.CreateLinearDampingAttr(0.);px.CreateAngularDampingAttr(0.)
        px.CreateSleepThresholdAttr(0.)
        px.CreateStabilizationThresholdAttr(0.)
        px.CreateMaxAngularVelocityAttr(1000.)
        px.CreateSolverPositionIterationCountAttr(npos)
        px.CreateSolverVelocityIterationCountAttr(nvel)
        px.CreateEnableGyroscopicForcesAttr(True)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)

    for name,path in paths.items():
        prim=UsdGeom.Xform.Define(stage,path).GetPrim()
        transform(prim,poses[name]);rigid(prim)
        b=bodies[name]
        mass(prim,b['mass_kg'],b['com_m'],b['inertia_kg_m2'])

    joint_paths={}
    for j in cmg['joints']:
        path=robot_path+'/joints/'+j['id'];joint_paths[j['id']]=path
        cls={'fixed':UsdPhysics.FixedJoint,'revolute':UsdPhysics.RevoluteJoint,
             'prismatic':UsdPhysics.PrismaticJoint}[j['type']]
        joint=cls.Define(stage,path)
        if j['base_body']!='world':
            joint.CreateBody0Rel().SetTargets([Sdf.Path(paths[j['base_body']])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(paths[j['follower_body']])])
        c=float(offsets[ids.index(j['id'])]) if j['type']!='fixed' else 0.
        # Native angle = CMG angle - c (c = 0 for fixed/prismatic joints).
        A,F=joint_frames(j,c)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float,A[:3,3])))
        joint.CreateLocalRot0Attr(quat(A[:3,:3]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float,F[:3,3])))
        joint.CreateLocalRot1Attr(quat(F[:3,:3]))
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(False)
        if j['type']!='fixed':
            joint.CreateAxisAttr('X')
            scale=180./np.pi if j['type']=='revolute' else 1.
            joint.CreateLowerLimitAttr(float((j['limits']['lower']-c)*scale))
            joint.CreateUpperLimitAttr(float((j['limits']['upper']-c)*scale))
            px=PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
            px.CreateJointFrictionAttr(0.)
            # Explicit efforts include both source actuator force and passive
            # damping; built-in drives must contribute exactly zero.
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular' if j['type']=='revolute' else 'linear')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.)
            drive.CreateMaxForceAttr(0.)
            # The experimental API sets/reads native armature again at start.
            joint.GetPrim().CreateAttribute('physxJoint:armature',Sdf.ValueTypeNames.Float).Set(
                float(cmg['armature'][cmg['coordinate_ids'].index(j['id'])]))
    # USD standard: root API on the fixed joint that anchors the tree to world.
    root_api_path=joint_paths['fixed_link0']
    rp=stage.GetPrimAtPath(root_api_path)
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    pa=PhysxSchema.PhysxArticulationAPI.Apply(rp)
    pa.CreateEnabledSelfCollisionsAttr(mode=='contact')
    pa.CreateSolverPositionIterationCountAttr(npos)
    pa.CreateSolverVelocityIterationCountAttr(nvel)
    pa.CreateSleepThresholdAttr(0.);pa.CreateStabilizationThresholdAttr(0.)
    mimic=PhysxSchema.PhysxMimicJointAPI.Apply(stage.GetPrimAtPath(joint_paths['finger_joint2']),'rotX')
    mimic.CreateReferenceJointRel().SetTargets([Sdf.Path(joint_paths['finger_joint1'])])
    mimic.CreateReferenceJointAxisAttr('rotX')
    mimic.CreateGearingAttr(-1.)
    mimic.CreateOffsetAttr(0.)
    # Do not replace mimic with frame-by-frame position copies or two drives.
    # Its asymmetric-load response is independently tested before the mission.
    for a,b in collision_exclusions(cmg):
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(paths[a])).CreateFilteredPairsRel().AddTarget(paths[b])

    specs=json.loads((root/'data/geometry.json').read_text())['geometries']
    with np.load(root/'data/mesh_arrays.npz',allow_pickle=False) as arrays:
        for g in specs:
            if not visuals and not g['collision']:
                continue
            path=paths[g['body']]+'/'+g['name']
            if g['kind']=='mesh':
                shape=UsdGeom.Mesh.Define(stage,path)
                points=arrays[g['mesh']+'_points'];faces=arrays[g['mesh']+'_faces']
                shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
                shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
                shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
                shape.CreateSubdivisionSchemeAttr('none')
                if g['collision']:
                    UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
                transform(shape.GetPrim(),np.asarray(g['T']))
            else:
                shape=UsdGeom.Cube.Define(stage,path);shape.CreateSizeAttr(2.)
                transform(shape.GetPrim(),np.asarray(g['T']),g['halfsize'])
            shape.CreateDisplayColorAttr([Gf.Vec3f(*map(float,g['rgba'][:3]))])
            if g['collision']:
                collision(shape.GetPrim(),{'kind':'robot','body':g['body'],'pad':g['pad']},
                          case['friction'] if g['pad'] else 1.,g['pad'])
                if not g['pad']:
                    shape.CreateVisibilityAttr('invisible')
                elif visuals:
                    shape.CreateDisplayColorAttr([Gf.Vec3f(.16,.18,.20)])

    def box(path,pos,halfsize,color,collide=True,mu=.8,label=None,contact_offset=SCENE_CONTACT_OFFSET,combine='average'):
        shape=UsdGeom.Cube.Define(stage,path);shape.CreateSizeAttr(2.)
        T=np.eye(4);T[:3,3]=pos
        transform(shape.GetPrim(),T,halfsize)
        shape.CreateDisplayColorAttr([Gf.Vec3f(*map(float,color))])
        if collide:
            collision(shape.GetPrim(),label or {'kind':'scene','name':path.rsplit('/',1)[-1]},mu,
                      contact_offset=contact_offset,combine=combine)
        return shape.GetPrim()

    object_path=None
    if mode=='contact':
        # A large, thin static floor has the same top surface as the source
        # infinite plane throughout this bounded task workspace.
        box('/World/Scene/floor',[0,0,-.025],[2.,2.,.025],[.11,.15,.20])
        for name,y in [('pick',-.20),('dock',.20)]:
            pick=name=='pick'
            box('/World/Scene/'+name+'_plinth',[.45,y,.0175],[.085,.075,.0175],[.20,.28,.36],
                mu=PICK_PLINTH_FRICTION if pick else 1.,combine='min' if pick else 'average')
        for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
            p=np.array([.45,.20,.044]);sz=np.array([.030,.023,.009])
            p[axis]+=sign*([.025,.018][axis]+.0025);sz[axis]=.0025
            name=f'socket_{axis}_{"neg" if sign<0 else "pos"}'
            box('/World/Scene/'+name,p,sz,[.1,.58,.65],mu=.5)
        box('/World/Scene/barrier',[.49,0,.10],[.145,.018,.10],[.28,.35,.43],mu=.6)
        box('/World/Scene/barrier_accent',[.49,0,.201],[.145,.018,.001],[1.,.62,.12],False)
        object_path='/World/cartridge'
        obj=UsdGeom.Xform.Define(stage,object_path).GetPrim()
        T=np.eye(4);T[:3,3]=[.45,-.20+case['offset'],.0702]
        transform(obj,T);rigid(obj)
        m,w=case['mass'],case['width']
        inertia=np.diag([m*(w*w+.070**2)/12,m*(.030**2+.070**2)/12,m*(.030**2+w*w)/12])
        mass(obj,m,[.003,0,0],inertia)
        PhysxSchema.PhysxRigidBodyAPI(obj).CreateEnableCCDAttr(True)
        box(object_path+'/collision',[0,0,0],[.015,w/2,.035],[.96,.58,.12],
            label={'kind':'object','name':'cartridge'},contact_offset=OBJECT_CONTACT_OFFSET)
        if visuals:
            for i,z in enumerate([-.026,.026]):
                box(object_path+f'/stripe{i}',[0,0,z],[.0152,w/2+.0002,.002],[.16,.22,.28],False)
            box(object_path+'/key',[.0153,0,.002],[.0002,.010,.012],[.95,.97,1.],False)
    elif visuals:
        box('/World/Scene/visual_floor',[0,0,-.025],[2,2,.025],[.11,.15,.20],False)

    if visuals:
        dome=UsdLux.DomeLight.Define(stage,'/World/Lighting/Dome')
        dome.CreateIntensityAttr(700.)
        light=UsdLux.DistantLight.Define(stage,'/World/Lighting/Key')
        light.CreateIntensityAttr(1800.)
        R=Rotation.from_euler('xyz',[30,-35,-30],degrees=True).as_matrix()
        T=np.eye(4);T[:3,:3]=R;transform(light.GetPrim(),T)
        camera=UsdGeom.Camera.Define(stage,'/World/Camera')
        eye=np.array([1.35,-1.35,.95]);target=np.array([.35,0,.25])
        z=eye-target;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        T=np.eye(4);T[:3,:3]=np.column_stack([x,y,z]);T[:3,3]=eye;transform(camera.GetPrim(),T)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.));camera.CreateFocalLengthAttr(30.)
    return dict(robot=robot_path,root_api=root_api_path,bodies=paths,joints=joint_paths,
                object=object_path,collider_labels=collider_labels,
                collision_count=len(collider_labels),pad_count=sum(g['pad'] for g in specs),
                physics_scene='/World/physicsScene',camera='/World/Camera')
