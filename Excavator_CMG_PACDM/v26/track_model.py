"""Explicit articulated steel-link track extension for the accepted excavator.

Uncalibrated engineering design, not recovered factory track parameters. Propulsion
comes only from rotary sprocket torque -> tooth/pin contact -> shoe/ground contact.
Each freely moving belt is a hinge tree closed with a two-site revolute constraint.
No belt-path constraints, prescribed surface velocity, or chassis forces are used.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class TrackConfig:
    teeth: int = 14
    straight_links: int = 18
    pitch_radius: float = .32
    shoe_width: float = .56
    shoe_thickness: float = .048
    shoe_mass: float = 12.
    sprocket_mass: float = 80.
    idler_mass: float = 65.
    road_roller_mass: float = 35.
    carrier_mass: float = 15.
    torque_limit: float = 30000.
    ground_friction: float = .8
    pin_damping: float = .1
    axle_damping: float = 2.
    sprocket_armature: float = 0.
    motor_rotor_mass: float = 10.
    motor_gear_ratio: float = 45.
    motor_rotor_axial_inertia: float = .008
    motor_rotor_transverse_inertia: float = .006
    contact_solref: str = '.004 1'
    closure_solref: str = '.002 1'
    center_y: float = 2.1034
    center_x: tuple = (-1.255, .35)


def fmt(v):
    return ' '.join(format(float(x), '.17g') for x in np.ravel(v))


def quat(R):
    return fmt(np.roll(Rotation.from_matrix(R).as_quat(), 1))


def rot_x(a):
    return Rotation.from_rotvec([a, 0, 0]).as_matrix()


def inertia_shift(c):
    return np.dot(c, c)*np.eye(3)-np.outer(c, c)


def cylinder_inertia(m, r, w):
    return np.diag([m*r*r/2, m*(3*r*r+w*w)/12, m*(3*r*r+w*w)/12])


def add_tracks(root, config=None):
    """Mutate a native v21 scene and return JSON-serializable provenance.

    Ground must have contype bit 4. Added shoes use bit32, bearing surfaces64,
    pins128 and teeth256. Drive names are track_drive_left/right; positive
    rotation is local +x. Left/right names follow world y at reference, viewed toward forward world +x.
    """
    c = config if isinstance(config, TrackConfig) else TrackConfig(**(config or {}))
    if c.teeth % 2:
        raise ValueError('An even sprocket tooth count is required')
    base = root.find(".//body[@name='body_53']")
    if base is None:
        raise ValueError('Source body_53 missing')
    if root.find(".//body[@name='track_left_shoe_000']") is not None:
        raise ValueError('Tracks already installed')
    world = root.find('worldbody')
    eq = root.find('equality')
    if eq is None:
        eq = ET.SubElement(root, 'equality')
    act = root.find('actuator')
    if act is None:
        act = ET.SubElement(root, 'actuator')
    for g in list(base.findall('geom')):
        if g.get('name', '').startswith('track_collision_'):
            base.remove(g)
    ground = world.find("geom[@name='ground']")
    if ground is not None:
        ground.set('conaffinity', str(int(ground.get('conaffinity', '0')) | 32))
    for g in world.findall(".//geom"):
        if g.get('name', '').startswith('grain_geom_'):
            g.set('conaffinity', str(int(g.get('conaffinity', '0')) | 32))
    if base.find('freejoint') is None:
        ET.SubElement(base, 'freejoint', name='floating_undercarriage')
    old = base.find('inertial')
    mass0 = float(old.get('mass'))
    com0 = np.fromstring(old.get('pos'), sep=' ')
    q = np.fromstring(old.get('quat', '1 0 0 0'), sep=' ')
    ri = Rotation.from_quat(np.roll(q, -1)).as_matrix()
    I0 = ri @ np.diag(np.fromstring(old.get('diaginertia'), sep=' ')) @ ri.T
    basep = np.fromstring(base.get('pos', '0 0 0'), sep=' ')
    baseq = np.fromstring(base.get('quat', '1 0 0 0'), sep=' ')
    baseR = Rotation.from_quat(np.roll(baseq, -1)).as_matrix()
    p = 2*c.pitch_radius*np.sin(np.pi/c.teeth)
    length = c.straight_links*p
    zc = c.pitch_radius + c.shoe_thickness/2
    rear = c.center_y-length/2
    front = c.center_y+length/2
    vertices = []
    # All links have exactly the same chord length, including both semicircles.
    for j in range(c.straight_links):
        vertices.append([rear+j*p, zc-c.pitch_radius])
    for j in range(c.teeth//2):
        a = -np.pi/2+j*2*np.pi/c.teeth
        vertices.append([front+c.pitch_radius*np.cos(a), zc+c.pitch_radius*np.sin(a)])
    for j in range(c.straight_links):
        vertices.append([front-j*p, zc+c.pitch_radius])
    for j in range(c.teeth//2):
        a = np.pi/2+j*2*np.pi/c.teeth
        vertices.append([rear+c.pitch_radius*np.cos(a), zc+c.pitch_radius*np.sin(a)])
    vertices = np.asarray(vertices)
    tangent = np.roll(vertices, -1, axis=0)-vertices
    assert np.max(np.abs(np.linalg.norm(tangent, axis=1)-p)) < 1e-12
    angles = np.arctan2(tangent[:, 1], tangent[:, 0])
    assigned = []
    side_meta = []
    common = dict(mass='0', margin='0', solref=c.contact_solref,
                  solimp='.98 .995 .001 .5 2')
    for side, x in zip(('right', 'left'), c.center_x):
        prefix = 'track_'+side
        names = []
        first = None
        previous = None
        shoeI = np.diag(c.shoe_mass/12*np.array([p*p+c.shoe_thickness**2,
                     c.shoe_width**2+c.shoe_thickness**2, c.shoe_width**2+p*p]))
        for j, (vertex, angle) in enumerate(zip(vertices, angles)):
            name = f'{prefix}_shoe_{j:03d}'
            names.append(name)
            R = rot_x(angle)
            point = np.r_[x, vertex]
            if j == 0:
                body = ET.SubElement(world, 'body', name=name,
                    pos=fmt(basep+baseR@point), quat=quat(baseR@R))
                ET.SubElement(body, 'freejoint', name=prefix+'_belt_free')
                first = body
            else:
                delta = np.arctan2(np.sin(angle-angles[j-1]), np.cos(angle-angles[j-1]))
                body = ET.SubElement(previous, 'body', name=name,
                    pos=fmt([0, p, 0]), quat=quat(rot_x(delta)))
                ET.SubElement(body, 'joint', name=f'{prefix}_pin_{j:03d}',
                    type='hinge', axis='1 0 0', limited='false', damping=str(c.pin_damping),
                    armature='0', frictionloss='0')
            ET.SubElement(body, 'inertial', pos=fmt([0, p/2, 0]), mass=str(c.shoe_mass),
                          diaginertia=fmt(np.diag(shoeI)))
            ET.SubElement(body, 'geom', name=name+'_pad', type='box',
                pos=fmt([0, p/2, 0]), size=fmt([c.shoe_width/2, p*.475, c.shoe_thickness/2]),
                rgba='.19 .22 .24 1', contype='32', conaffinity='111', condim='3',
                priority='2', friction=f'{c.ground_friction} .002 .001', **common)
            # Hardened transverse pin is the actual sprocket contact component.
            ET.SubElement(body, 'geom', name=name+'_pin', type='capsule',
                fromto=fmt([-.20, 0, 0, .20, 0, 0]), size='.018',
                rgba='.43 .46 .49 1', contype='128', conaffinity='256', condim='1', **common)
            # The hinge origin is the pin location; this site's position is diagnostic.
            ET.SubElement(body, 'site', name=name+'_hinge', pos='0 0 0', size='.003', rgba='0 0 0 0')
            center = point+R@np.array([0, p/2, 0])
            assigned.append((name, c.shoe_mass, center, R@shoeI@R.T))
            previous = body
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        # The seam is the same adjacent revolute pair as all native hinge pairs.
        ET.SubElement(contact, 'exclude', body1=names[-1], body2=names[0])
        for j, anchor_x in enumerate((-.20, .20)):
            a = f'{prefix}_closure_first_{j}'
            b = f'{prefix}_closure_last_{j}'
            ET.SubElement(first, 'site', name=a, pos=fmt([anchor_x, 0, 0]), size='.003', rgba='0 0 0 0')
            ET.SubElement(previous, 'site', name=b, pos=fmt([anchor_x, p, 0]), size='.003', rgba='0 0 0 0')
            ET.SubElement(eq, 'connect', name=f'{prefix}_closure_{j}', site1=a, site2=b,
                solref=c.closure_solref, solimp='.9999 .9999 .001 .5 2')
        def wheel(label, y, z, radius, mass, drive=False):
            name = prefix+'_'+label
            width = .42
            center = np.array([x, y, z])
            body = ET.SubElement(base, 'body', name=name, pos=fmt(center))
            joint = name+'_axle'
            ET.SubElement(body, 'joint', name=joint, type='hinge', axis='1 0 0',
                          limited='false', damping=str(c.axle_damping), armature=str(c.sprocket_armature if drive else 0.))
            I = cylinder_inertia(mass, radius, width)
            ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(mass), diaginertia=fmt(np.diag(I)))
            ET.SubElement(body, 'geom', name=name+'_rim', type='cylinder',
                          zaxis='1 0 0', size=fmt([radius, width/2]),
                          rgba='.36 .38 .41 1', contype='64', conaffinity='32',
                          condim='1' if drive else '3', priority='3',
                          friction='0 0 0' if drive else '.35 .001 .001', **common)
            # Physical retaining flanges; no hidden constraints hold the belts laterally.
            for k, sx in enumerate(() if label.startswith('road_') else (-.30, .30)):
                ET.SubElement(body, 'geom', name=f'{name}_flange_{k}', type='cylinder',
                    pos=fmt([sx, 0, 0]), zaxis='1 0 0', size=fmt([radius+.055, .012]),
                    rgba='.32 .34 .37 1', contype='64', conaffinity='32',
                    condim='1' if drive else '3', priority='3',
                    friction='0 0 0' if drive else '.15 .001 .001', **common)
            assigned.append((name, mass, center, I))
            if drive:
                for k in range(c.teeth):
                    a = np.pi/2+(k+.5)*2*np.pi/c.teeth
                    # Local y is the tooth radial axis, z its tangential axis.
                    pos = np.array([0, np.cos(a), np.sin(a)])*(c.pitch_radius-.028)
                    ET.SubElement(body, 'geom', name=f'{name}_tooth_{k:02d}', type='box',
                        pos=fmt(pos), quat=quat(rot_x(a)), size=fmt([.18, .054, .027]),
                        rgba='.65 .56 .25 1', contype='256', conaffinity='128', condim='1', **common)
                rotor_name = prefix+'_motor_rotor'
                rotor_center = center+np.array([.12, 0, 0])
                rotor = ET.SubElement(base, 'body', name=rotor_name, pos=fmt(rotor_center))
                rotor_joint = rotor_name+'_axle'
                ET.SubElement(rotor, 'joint', name=rotor_joint, type='hinge', axis='1 0 0',
                              limited='false', damping='0', armature='0')
                rotor_I = np.diag([c.motor_rotor_axial_inertia, c.motor_rotor_transverse_inertia,
                                   c.motor_rotor_transverse_inertia])
                ET.SubElement(rotor, 'inertial', pos='0 0 0', mass=str(c.motor_rotor_mass),
                              diaginertia=fmt(np.diag(rotor_I)))
                ET.SubElement(rotor, 'geom', name=rotor_name+'_visual', type='cylinder',
                              zaxis='1 0 0', size='.04 .0244948974', mass='0',
                              contype='0', conaffinity='0', rgba='.75 .2 .15 1')
                ET.SubElement(eq, 'joint', name=prefix+'_ideal_gear', joint1=rotor_joint,
                              joint2=joint, polycoef=fmt([0,c.motor_gear_ratio,0,0,0]),
                              solref=c.closure_solref, solimp='.9999 .9999 .001 .5 2')
                assigned.append((rotor_name, c.motor_rotor_mass, rotor_center, rotor_I))
                ET.SubElement(act, 'motor', name='track_drive_'+side, joint=joint, gear='1',
                    ctrllimited='true', ctrlrange=fmt([-c.torque_limit, c.torque_limit]),
                    forcelimited='true', forcerange=fmt([-c.torque_limit, c.torque_limit]))
            return name
        inner_r = c.pitch_radius*np.cos(np.pi/c.teeth)-c.shoe_thickness/2-.002
        wheel('sprocket', rear, zc, inner_r, c.sprocket_mass, True)
        wheel('idler', front, zc, inner_r, c.idler_mass)
        for j, frac in enumerate((-.33, -.11, .11, .33)):
            wheel(f'road_{j}', c.center_y+frac*length, c.shoe_thickness+.152, .15, c.road_roller_mass)
        for j, frac in enumerate((-.22, .22)):
            wheel(f'carrier_{j}', c.center_y+frac*length, zc+c.pitch_radius-c.shoe_thickness/2-.092, .09, c.carrier_mass)
        side_meta.append(dict(side=side, shoe_bodies=names, sprocket_joint=prefix+'_sprocket_axle',
                              actuator='track_drive_'+side, center_x_m=x))
    moved_mass = sum(t[1] for t in assigned)
    residual_mass = mass0-moved_mass
    if residual_mass <= 0:
        raise ValueError('Moving track components exceed source undercarriage mass')
    residual_com = (mass0*com0-sum(m*r for _, m, r, _ in assigned))/residual_mass
    origin_I = I0+mass0*inertia_shift(com0)
    residual_I = origin_I-sum(I+m*inertia_shift(r) for _, m, r, I in assigned)-residual_mass*inertia_shift(residual_com)
    eig = np.linalg.eigvalsh(residual_I)
    if eig[0] <= 0 or eig[2] >= eig[0]+eig[1]:
        raise ValueError(f'Residual chassis inertia is not physically realizable: {eig}')
    # Serialize the exact principal basis ourselves: MuJoCo's fullinertia
    # diagonalization uses a looser Jacobi stopping tolerance, which otherwise
    # perturbs the reconstructed 1e4 kg m^2 tensor by about 1e-5 kg m^2.
    principal, basis = np.linalg.eigh(residual_I)
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    old.attrib.clear()
    old.attrib.update(mass=str(residual_mass), pos=fmt(residual_com),
                      quat=quat(basis), diaginertia=fmt(principal))
    reconstructed_mass = residual_mass+moved_mass
    reconstructed_com = (residual_mass*residual_com+sum(m*r for _, m, r, _ in assigned))/mass0
    reconstructed_I = residual_I+residual_mass*inertia_shift(residual_com-com0)+sum(
        I+m*inertia_shift(r-com0) for _, m, r, I in assigned)
    meta = dict(config=asdict(c), pitch_m=p, pitch_radius_m=c.pitch_radius,
        straight_span_m=length, shoes_per_side=len(vertices), vertical_offset_m=0,
        drive_joint_names=['track_left_sprocket_axle', 'track_right_sprocket_axle'],
        drive_actuator_names=['track_drive_left', 'track_drive_right'],
        sprocket_pitch_radius_m=c.pitch_radius,
        source_frame_left_x_m=c.center_x[1], source_frame_right_x_m=c.center_x[0],
        positive_forward='world +x at reference; equal positive sprocket rotations',
        positive_yaw='right sprocket faster than left sprocket gives world +z yaw',
        sprocket_rotor_armature_kg_m2=c.sprocket_armature,
        explicit_motor_rotor_bodies=True, motor_rotor_gear_ratio=c.motor_gear_ratio,
        reflected_rotor_inertia_kg_m2=c.motor_rotor_axial_inertia*c.motor_gear_ratio**2,
        initial_ground_clearance_m=0., total_height_m=2*zc, sides=side_meta,
        source_undercarriage_mass_kg=mass0, chassis_residual_mass_kg=residual_mass,
        source_undercarriage_com_m=com0.tolist(), source_undercarriage_inertia_kg_m2=I0.tolist(),
        chassis_residual_com_m=residual_com.tolist(), residual_inertia_eigenvalues_kg_m2=eig.tolist(),
        mass_reconstruction_error_kg=float(abs(reconstructed_mass-mass0)),
        com_reconstruction_error_m=float(np.linalg.norm(reconstructed_com-com0)),
        inertia_reconstruction_error_kg_m2=float(np.linalg.norm(reconstructed_I-I0)),
        components=[dict(body=n, mass_kg=m, source_frame_com_m=r.tolist(), source_frame_inertia_kg_m2=I.tolist()) for n,m,r,I in assigned],
        prescribed_base_forces=False, prescribed_belt_motion=False, belt_path_constraints=False,
        geometry_calibrated_to_hardware=False, positive_drive='toothed sprocket contacting transverse shoe pins',
        assumptions=['Dimensions inferred from source track envelope; detailed drivetrain CAD unavailable.',
                     'Rigid articulated steel shoes, ideal revolute pins and compliant unilateral contacts.',
                     'Sprocket rim/flanges are frictionless bearings: positive drive requires tooth-pin engagement.',
                     'Mass/inertia partition exactly preserves source undercarriage aggregate at reference configuration.',
                     'Roller positions, shoe masses, drive torque bounds and friction are engineering assumptions.',
                     'Sprocket and idler use equal support radii; road rollers omit outer guide flanges to avoid end-wheel hardware intersections.',
                     'Belt lateral retention uses sprocket/idler flanges and carrier flanges; road rollers provide vertical load support.',
                     'Ground and material contact obey the native MuJoCo frictional contact model; no calibrated deformable soil.'])
    return meta


def isolated_scene(path, config=None, dt=.0005, payload_mass=9425.452625161175, friction=None):
    """Compile-ready free undercarriage carrying rigid surrogate upper-body ballast.

    This test is deliberately isolated; integrated moving articulated arm uses the
    full source model. Ballast adds known supported mass without hidden anchors.
    """
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    source=Path(__file__).parent/'source/Excavator_RoboIR_full_body_v21/assets/visual_actual.xml'
    src=ET.parse(source).getroot(); oldbase=src.find(".//body[@name='body_53']")
    root=ET.Element('mujoco', model='explicit_articulated_track_isolated_test')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false',autolimits='false',balanceinertia='false')
    op=ET.SubElement(root,'option',timestep=str(dt),gravity='0 0 -9.81',integrator='implicitfast',jacobian='sparse',solver='Newton',iterations='100',tolerance='1e-14',cone='elliptic')
    ET.SubElement(op,'flag',energy='enable',autoreset='disable')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='3 -4 8',dir='-1 1 -2')
    ET.SubElement(world,'geom',name='ground',type='plane',size='15 15 .1',contype='4',conaffinity='32',friction='.8 .002 .001',condim='3',rgba='.45 .5 .46 1')
    base=ET.SubElement(world,'body',**oldbase.attrib)
    base.append(copy.deepcopy(oldbase.find('inertial')))
    ET.SubElement(base,'freejoint',name='floating_undercarriage')
    ET.SubElement(base,'geom',type='box',pos='-.4525 2.1034 .6',size='.42 1.15 .2',mass='0',contype='0',conaffinity='0',rgba='.9 .7 .15 1')
    if payload_mass:
        ballast=ET.SubElement(base,'body',name='upper_body_ballast',pos='-.4525 2.1034 1.2')
        ET.SubElement(ballast,'inertial',mass=str(payload_mass),pos='0 0 0',diaginertia=fmt([payload_mass, payload_mass, payload_mass]))
        ET.SubElement(ballast,'geom',type='box',size='.65 .65 .3',mass='0',contype='0',conaffinity='0',rgba='.9 .72 .12 1')
    c=config if isinstance(config,TrackConfig) else TrackConfig(**(config or {}))
    if friction is not None:c.ground_friction=friction
    meta=add_tracks(root,c)
    ET.indent(root, space='  ');ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    path.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
    return meta
