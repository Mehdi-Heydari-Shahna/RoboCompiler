"""Native dry-granular excavation bed. All dimensions and properties are assumed."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from mobile_mission import build_scene
from digging_export import fmt

BOUNDS = np.array([[5.45, 7.95], [-.72, .72], [-.40, 0.]])


def build_soil_scene(path, mission, dt=.0005, radius=.08, empty=False,
                     bucket_contact=True, seed=22, cone='pyramidal', entry_slot=.9):
    path = Path(path)
    meta = build_scene(path, mission, dt=dt, soil=False, bucket_contact=True)
    root = ET.parse(path).getroot()
    root.set('model', 'RoboCompiler_soil_stationary_v01')
    # Explicit prototype modeling choice, not numerically identical to v26's
    # elliptic friction law. Keep it configurable and recorded in metadata.
    root.find('option').set('cone', cone)
    if not bucket_contact:
        # Disable only soil/bucket pairs. Bucket-ground collision is retained.
        for geom in root.findall(".//body[@name='body_56']/geom"):
            if geom.get('name', '').startswith(('bucket_panel_', 'bucket_side_')):
                geom.set('contype', '2'); geom.set('conaffinity', '4')
    world = root.find('worldbody')
    old = world.find("geom[@name='ground']")
    properties = dict(old.attrib)
    world.remove(old)
    x0, x1 = BOUNDS[0]; y0, y1 = BOUNDS[1]; bottom = BOUNDS[2, 0]

    def slab(name, low, high, ztop=0., thickness=.6):
        center = (np.asarray(low) + high) / 2
        half = (np.asarray(high) - low) / 2
        attr = dict(properties)
        attr.update(name=name, type='box', pos=fmt([*center, ztop-thickness/2]),
                    size=fmt([*half, thickness/2]), rgba='.40 .36 .28 1')
        ET.SubElement(world, 'geom', **attr)

    # Four surface slabs leave a real aperture. The bed bottom is lower.
    slab('support_west', [-25., -25.], [x0, 25.])
    slab('support_east', [x1, -25.], [25., 25.])
    slab('support_south', [x0, -25.], [x1, y0])
    slab('support_north', [x0, y1], [x1, 25.])
    slab('bed_bottom', [x0, y0], [x1, y1], ztop=bottom, thickness=.15)

    rng = np.random.default_rng(seed)
    positions = []
    rho = 2000.  # particle density, not bulk density or measured material data
    mass = 4*np.pi*radius**3*rho/3
    spacing = 2.015*radius
    if not empty:
        for layer, z in enumerate(np.arange(bottom+radius+.003, .001, np.sqrt(2/3)*spacing)):
            if z+radius > .055:
                break
            for row, y in enumerate(np.arange(y0+radius+.008, y1-radius, np.sqrt(3)/2*spacing)):
                y += radius/np.sqrt(3)*(layer % 2)
                if y+radius >= y1:
                    continue
                shift = radius*((row+layer) % 2)
                for x in np.arange(x0+radius+.008+shift, x1-entry_slot-radius, spacing):
                    p = np.array([x, y, z])
                    p[:2] += rng.uniform(-.0002, .0002, 2)
                    n = len(positions); positions.append(p)
                    body = ET.SubElement(world, 'body', name=f'grain_{n:04d}', pos=fmt(p))
                    ET.SubElement(body, 'freejoint', name=f'grain_joint_{n:04d}')
                    ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(mass),
                                  diaginertia=fmt(np.full(3, .4*mass*radius**2)))
                    shade = rng.uniform(.85, 1.12)
                    ET.SubElement(body, 'geom', name=f'soil_{n:04d}', type='sphere',
                                  size=str(radius), rgba=fmt([.46*shade, .30*shade, .14*shade, 1]),
                                  contype='1', conaffinity='47' if bucket_contact else '45', condim='6',
                                  friction='.65 .003 .006', solref='.008 1')
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    meta.update(particle_count=len(positions), particle_radius_m=radius,
                particle_density_kg_m3=rho, particle_mass_kg=mass,
                total_soil_mass_kg=mass*len(positions), seed=seed,
                bed_bounds_m=BOUNDS.tolist(), soil_calibrated=False,
                soil_model='Native rigid macro-particles: dry noncohesive prototype',
                original_surface_plane_removed=True, external_soil_forces=False,
                friction_cone=cone,
                cleared_entry_slot_m=entry_slot,
                bucket_soil_contact=bucket_contact, bucket_ground_contact=True,
                terrain_case=('Exposed soil face beside a cleared entry slot' if entry_slot > 0
                              else 'Filled granular excavation bed without an entry slot'),
                particle_initial_positions_m=positions)
    return meta
