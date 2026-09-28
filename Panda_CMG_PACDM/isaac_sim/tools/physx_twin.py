"""OPTIONAL offline PhysX twin (SAPIEN 3 / PhysX 5.3.1, CPU). NOT Isaac Sim evidence.

Why this exists
---------------
The first native Isaac suite failed in ways that could be reproduced without
Isaac: this tool builds the same CMG articulation, colliders and task scene in
SAPIEN's PhysX and drives it with the UNMODIFIED cmg_isaac.simulation.run_mission
loop (same controller, logging, metrics and acceptance gates). With the old
settings (TGS, 64/8 iterations, no joint offsets, 0.2 mm contact offsets,
pick plinth friction 1.0) it reproduces the native failure signature (2.008 mm
vs native 2.001 mm tool error, stalled joints 4/6, object creep); with the
declared settings it passes.

Differences from Isaac Sim 6.1 (results are indicative, never a native PASS):
PhysX 5.3.1 instead of the Isaac 6.1 PhysX build; the finger mimic joint is
emulated with a stiff fixed tendon; PhysX's 'min' friction-combine mode (pads,
pick plinth) is reproduced exactly for every cartridge pair through material
values; articulation self-collision is disabled; friction in every PGS
iteration is SAPIEN's default (Isaac's choice is undocumented, see
--friction-every-iteration).

Usage (needs:  pip install sapien==3.0.3  in a separate, non-Isaac environment)
    python tools/physx_twin.py --mode contact --case nominal
    python tools/physx_twin.py --mode wrench --case nominal --legacy   # old settings
Outputs go to results/physx_twin/<mode>/<case>/ with the usual trajectory,
summary, result and REPORT files. They are labelled as twin output and must not
be mixed into a native suite folder (report.aggregate rejects them anyway,
because their backend string is not the native Isaac backend).
"""
from pathlib import Path
import argparse, json, shutil, sys, time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TWIN_BACKEND = 'OFFLINE TWIN: SAPIEN 3 / PhysX 5.3.1 (not Isaac Sim)'
# v1 solver settings of the first native suite (the v1 scene is restored in TwinPanda).
LEGACY_SETTINGS = dict(solver='TGS', position_iterations=64, velocity_iterations=8, joint_zero_offsets=False)


def _pose(T):
    import sapien
    T = np.asarray(T, float)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return sapien.Pose(np.asarray(T[:3, 3], np.float32), np.asarray([q[3], q[0], q[1], q[2]], np.float32))


class _Timeline:
    def pause(self): pass
    def stop(self): pass
    def is_playing(self): return True


class _View:
    def __init__(self, comp): self.comp = comp
    def get_world_poses(self):
        pose = self.comp.entity_pose
        return np.asarray(pose.p, float)[None, :], np.asarray(pose.q, float)[None, :]
    def get_velocities(self):
        return np.asarray(self.comp.linear_velocity, float)[None, :], np.asarray(self.comp.angular_velocity, float)[None, :]


class _Contacts:
    """Same values() dictionary as cmg_isaac.contacts.ContactObserver."""
    def __init__(self, system, labels, dt):
        self.system, self.labels, self.dt = system, labels, dt
        self.error = None; self.total_contact_points = 0; self.callback_count = 0; self.last_bad = []
        self.clear()

    def clear(self):
        self.left = self.right = self.support = 0.
        self.pad_contacts = self.bad = self.contact_points = self.left_count = self.right_count = 0
        self.last_bad = []

    def collect(self):
        self.clear(); self.callback_count += 1
        for c in self.system.get_contacts():
            if not c.points:
                continue
            a, b = (self.labels[int(round(s.get_density() - 1000.))] for s in c.shapes)
            robot = [x for x in (a, b) if x['kind'] == 'robot']
            obj = any(x['kind'] == 'object' for x in (a, b))
            scene = next((x for x in (a, b) if x['kind'] == 'scene'), None)
            bad = (len(robot) == 2 or
                   bool(robot and scene and (scene['name'] != 'floor' or robot[0]['body'] != 'link0')) or
                   bool(obj and scene and scene['name'] == 'barrier') or
                   bool(obj and robot and robot[0]['body'] not in ('left_finger', 'right_finger')))
            if bad:
                self.bad += 1
                if len(self.last_bad) < 5: self.last_bad.append([a, b])
            for p in c.points:
                force = abs(float(np.asarray(p.normal, float) @ np.asarray(p.impulse, float))) / self.dt
                self.contact_points += 1; self.total_contact_points += 1
                if obj and robot and robot[0].get('pad', False):
                    self.pad_contacts += 1
                    if robot[0]['body'] == 'left_finger': self.left += force; self.left_count += 1
                    else: self.right += force; self.right_count += 1
                if obj and scene and scene['name'] == 'pick_plinth':
                    self.support += abs(float(np.asarray(p.impulse, float)[2])) / self.dt

    def values(self):
        return dict(left_normal_force=self.left, right_normal_force=self.right, normal_force=self.left + self.right,
                    pad_contacts=self.pad_contacts, bilateral_contact=int(self.left_count > 0 and self.right_count > 0),
                    unexpected_contacts=self.bad, pick_support_N=self.support)


class TwinPanda:
    """NativePanda-compatible interface on SAPIEN PhysX (used by run_mission)."""

    def __init__(self, cmg, reference, mode, case, legacy=False, friction_every_iteration=True):
        import sapien
        import sapien.physx as px
        from cmg_isaac.rigid import RigidTree
        from cmg_isaac.geometry import joint_frames, joint_zero_offsets
        from cmg_isaac.scene import ROBOT_CONTACT_OFFSET, OBJECT_CONTACT_OFFSET, SCENE_CONTACT_OFFSET, PICK_PLINTH_FRICTION
        self.cmg, self.ref, self.mode, self.case = cmg, reference, mode, case
        self.dt = case['dt']; self.gui = False; self.physics_errors = []; self.timeline = _Timeline()
        self.backend_name = TWIN_BACKEND
        self.checkpoint = lambda label, details=None: print('[TWIN] ' + label, flush=True)
        self.tree = RigidTree(cmg)
        # legacy=True restores the v1 scene (0.2 mm contact offsets everywhere,
        # pick plinth 1.0 'average'); the v1 solver settings arrive through
        # `case` (see LEGACY_SETTINGS in main()).
        npos = int(case['position_iterations']); nvel = int(case['velocity_iterations'])
        robot_co = ROBOT_CONTACT_OFFSET
        object_co = .0002 if legacy else OBJECT_CONTACT_OFFSET
        scene_co = .0002 if legacy else SCENE_CONTACT_OFFSET
        self.contact_offsets_m = dict(robot=robot_co, object=object_co, scene=scene_co)
        # PGS only (TGS always solves friction in every iteration). SAPIEN's
        # default is on; the PhysX SDK default is off (friction only in the
        # last 3 position iterations and in velocity iterations).
        self.friction_every_iteration = bool(friction_every_iteration)
        cfg = px.PhysxSceneConfig(); cfg.enable_tgs = case['solver'] == 'TGS'; cfg.enable_ccd = True
        cfg.enable_friction_every_iteration = self.friction_every_iteration
        cfg.gravity = np.array([0, 0, -9.81], np.float32); px.set_scene_config(cfg)
        bc = px.PhysxBodyConfig(); bc.solver_position_iterations = npos; bc.solver_velocity_iterations = nvel
        bc.sleep_threshold = 0.; px.set_body_config(bc)
        self.system = px.PhysxCpuSystem(); self.scene = sapien.Scene([self.system]); self.scene.set_timestep(self.dt)
        ids = cmg['coordinate_ids']
        self.offsets = joint_zero_offsets(cmg, reference.data['q'], case.get('joint_zero_offsets', True))
        q0 = reference.at(0.)[0].copy(); poses = self.tree.poses(q0)
        bodies = {b['id']: b for b in cmg['bodies']}; labels = []; mats = {}
        collide = mode == 'contact'

        # Friction. Native pair coefficients (scene.py): the pads and the pick
        # plinth use PhysX's 'min' combine mode, everything else 'average'. SAPIEN
        # only offers 'average', and every pair that can touch involves the
        # cartridge (0.8), so the cartridge gets friction 0 and every other shape
        # twice its native pair coefficient with the cartridge. This is exact for
        # all cartridge pairs; robot-scene pairs never touch (base on floor aside).
        obj_mu = .8
        obj_material = px.PhysxMaterial(0., 0., 0.)
        self.pick_plinth_friction = (1., 'average') if legacy else (PICK_PLINTH_FRICTION, 'min')

        def pair_mu(mu, combine='average'):
            return min(mu, obj_mu) if combine == 'min' else (mu + obj_mu) / 2

        def material(pair):
            return mats.setdefault(round(2 * pair, 6), px.PhysxMaterial(float(2 * pair), float(2 * pair), 0.))

        def label(shape, info, co, groups):
            shape.set_collision_groups(groups); shape.set_density(1000. + len(labels)); labels.append(info)
            shape.set_contact_offset(co); shape.set_rest_offset(0.)

        specs = json.loads((ROOT / 'data/geometry.json').read_text())['geometries']
        arrays = np.load(ROOT / 'data/mesh_arrays.npz', allow_pickle=False)
        joints = {j['follower_body']: j for j in cmg['joints']}
        self.links = {}; entities = []
        for name in self.tree.body_names[1:]:
            j = joints[name]
            link = px.PhysxArticulationLinkComponent(None if j['base_body'] == 'world' else self.links[j['base_body']])
            if collide:
                for g in (g for g in specs if g['body'] == name and g['collision']):
                    mu = pair_mu(case['friction'], 'min') if g['pad'] else pair_mu(1.)
                    if g['kind'] == 'mesh':
                        shape = px.PhysxCollisionShapeConvexMesh(vertices=np.asarray(arrays[g['mesh'] + '_points'], np.float32),
                                                                 scale=np.ones(3, np.float32), material=material(mu))
                    else:
                        shape = px.PhysxCollisionShapeBox(half_size=np.asarray(g['halfsize'], np.float32), material=material(mu))
                    shape.local_pose = _pose(g['T'])
                    label(shape, {'kind': 'robot', 'body': name, 'pad': g['pad']}, robot_co, [1, 1, 1, 7])
                    if g['pad']:
                        shape.set_patch_radius(.005); shape.set_min_patch_radius(.005)
                    link.attach(shape)
            b = bodies[name]; d, R = np.linalg.eigh(np.asarray(b['inertia_kg_m2'], float))
            if np.linalg.det(R) < 0: R[:, 0] *= -1
            Tc = np.eye(4); Tc[:3, :3] = R; Tc[:3, 3] = b['com_m']
            link.mass = float(b['mass_kg']); link.cmass_local_pose = _pose(Tc); link.inertia = np.asarray(d, np.float32)
            link.linear_damping = 0.; link.angular_damping = 0.; link.name = name
            k = ids.index(j['id']) if j['type'] != 'fixed' else None
            A, F = joint_frames(j, float(self.offsets[k]) if j['type'] == 'revolute' else 0.)
            link.joint.name = j['id']
            link.joint.type = {'fixed': 'fixed', 'revolute': 'revolute_unwrapped', 'prismatic': 'prismatic'}[j['type']]
            link.joint.pose_in_parent = _pose(A); link.joint.pose_in_child = _pose(F)
            if k is not None:
                link.joint.limit = np.array([[j['limits']['lower'] - self.offsets[k], j['limits']['upper'] - self.offsets[k]]], np.float32)
                link.joint.set_drive_property(0., 0.); link.joint.armature = np.array([cmg['armature'][k]], np.float32)
                link.joint.friction = 0.
            e = sapien.Entity(); e.add_component(link); e.name = name; e.pose = _pose(poses[name])
            self.links[name] = link; entities.append(e)
        self.art = self.links['link0'].articulation
        self.art.create_fixed_tendon([self.links['hand'], self.links['left_finger'], self.links['right_finger']],
                                     [0, 1, -1], [0, 1, -1], rest_length=0., stiffness=1e5, damping=0.)
        for e in entities: self.scene.add_entity(e)
        self.art.set_solver_position_iterations(npos); self.art.set_solver_velocity_iterations(nvel); self.art.set_sleep_threshold(0.)
        names = [jj.name for jj in self.art.get_active_joints()]
        self.map = np.array([names.index(n) for n in ids])
        self.hand = _View(self.links['hand']); self.object = None; self.contacts = None
        if collide:
            def box(name, pos, half, mu, combine='average', info=None, dynamic=False):
                comp = px.PhysxRigidDynamicComponent() if dynamic else px.PhysxRigidStaticComponent()
                shape = px.PhysxCollisionShapeBox(half_size=np.asarray(half, np.float32),
                                                  material=obj_material if dynamic else material(pair_mu(mu, combine)))
                label(shape, info or {'kind': 'scene', 'name': name}, object_co if dynamic else scene_co, [1, 1, 0, 0]); comp.attach(shape)
                e = sapien.Entity(); e.add_component(comp); T = np.eye(4); T[:3, 3] = pos; e.pose = _pose(T)
                return e, comp
            items = [('floor', [0, 0, -.025], [2., 2., .025], .8)]
            items += [('pick_plinth', [.45, -.20, .0175], [.085, .075, .0175], *self.pick_plinth_friction),
                      ('dock_plinth', [.45, .20, .0175], [.085, .075, .0175], 1.)]
            for axis, sign in [(0, -1), (0, 1), (1, -1), (1, 1)]:
                p = np.array([.45, .20, .044]); sz = np.array([.030, .023, .009])
                p[axis] += sign * ([.025, .018][axis] + .0025); sz[axis] = .0025
                items.append((f'socket_{axis}_{"neg" if sign < 0 else "pos"}', p, sz, .5))
            items.append(('barrier', [.49, 0, .10], [.145, .018, .10], .6))
            for it in items: self.scene.add_entity(box(*it)[0])
            m, w = case['mass'], case['width']
            e, comp = box('cartridge', [.45, -.20 + case['offset'], .0702], [.015, w / 2, .035], obj_mu,
                          info={'kind': 'object', 'name': 'cartridge'}, dynamic=True)
            comp.mass = m; comp.cmass_local_pose = _pose(np.block([[np.eye(3), np.array([[.003], [0], [0]])], [np.zeros((1, 3)), np.ones((1, 1))]]))
            comp.inertia = np.array([m * (w * w + .070 ** 2) / 12, m * (.030 ** 2 + .070 ** 2) / 12, m * (.030 ** 2 + w * w) / 12], np.float32)
            comp.set_solver_position_iterations(npos); comp.set_solver_velocity_iterations(nvel)
            comp.set_sleep_threshold(0.); comp.linear_damping = 0.; comp.angular_damping = 0.
            self.scene.add_entity(e); self.object_comp = comp; self.object = _View(comp)
            self.contacts = _Contacts(self.system, labels, self.dt)
        self.reset_initial()

    def state(self):
        return (np.asarray(self.art.get_qpos(), float)[self.map] + self.offsets,
                np.asarray(self.art.get_qvel(), float)[self.map])

    def effort(self, tau):
        out = np.zeros(9, np.float32); out[self.map] = np.asarray(tau, np.float32); self.art.set_qf(out)

    def teleport_for_initialization(self, q, v=None):
        qn = np.zeros(9, np.float32); qn[self.map] = np.asarray(q, float) - self.offsets
        vn = np.zeros(9, np.float32)
        if v is not None: vn[self.map] = v
        self.art.set_qpos(qn); self.art.set_qvel(vn)

    @staticmethod
    def pose(view):
        p, q = view.get_world_poses(); p, q = p[0], q[0]
        return p, Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

    def tool_pose(self):
        p, R = self.pose(self.hand); T = np.asarray(self.cmg['tool']['T_body_tool'])
        return p + R @ T[:3, 3], R @ T[:3, :3]

    def apply_wrench(self, w):
        w = np.asarray(w, float)
        if not np.any(w): return
        if self.mode == 'contact':
            p, R = self.pose(self.object); position = p + R @ np.array([.003, 0, 0]); comp = self.object_comp
        else:
            position, _ = self.tool_pose(); comp = self.links['hand']
        comp.add_force_at_point(np.asarray(w[:3], np.float32), np.asarray(position, np.float32))
        if np.any(w[3:]): comp.add_force_torque(np.zeros(3, np.float32), np.asarray(w[3:], np.float32))

    def step(self):
        self.scene.step()
        if self.contacts: self.contacts.collect()

    def render(self): pass

    def reset_initial(self):
        q = self.ref.at(0.)[0].copy()
        if self.case['initial_offset']: q[:7] += np.array([1., -.6, .4, .7, -.5, .3, -.4]) * .001
        self.teleport_for_initialization(q); self.effort(np.zeros(9))
        if self.object:
            T = np.eye(4); T[:3, 3] = [.45, -.20 + self.case['offset'], .0702]; self.object_comp.entity_pose = _pose(T)
            self.object_comp.linear_velocity = np.zeros(3, np.float32); self.object_comp.angular_velocity = np.zeros(3, np.float32)
        if self.contacts: self.contacts.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['contact', 'wrench'], default='contact'); p.add_argument('--case', default='nominal')
    p.add_argument('--legacy', action='store_true', help='Reproduce the first native suite settings (TGS 64/8, no joint offsets, 0.2 mm contact offsets, pick plinth 1.0)')
    p.add_argument('--duration', type=float, help='Shorter diagnostic run (s); acceptance then fails on duration')
    p.add_argument('--friction-every-iteration', choices=['on', 'off'], default='on',
                   help='PGS friction in every solver iteration (SAPIEN default: on). The PhysX SDK default is off '
                        '(friction only in the last 3 position iterations); Isaac does not document its choice, so check both')
    p.add_argument('--output', help='Default results/physx_twin/<mode>/<case>[_legacy][_fei_off]')
    a = p.parse_args()
    fei = a.friction_every_iteration == 'on'
    from cmg_isaac.control import Reference
    from cmg_isaac.cases import get_case
    from cmg_isaac.simulation import run_mission
    cmg = json.loads((ROOT / 'data/panda_cmg.json').read_text()); ref = Reference(ROOT)
    if a.duration: ref.duration = a.duration
    case = get_case(a.mode, a.case)
    if a.legacy:
        case.update(LEGACY_SETTINGS)
    out = Path(a.output) if a.output else ROOT / 'results/physx_twin' / a.mode / (
        a.case + ('_legacy' if a.legacy else '') + ('' if fei else '_fei_off'))
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    start = time.time()
    twin = TwinPanda(cmg, ref, a.mode, case, legacy=a.legacy, friction_every_iteration=fei)
    result = run_mission(twin, cmg, ref, a.mode, a.case, case, out, dict(passed=True, checks={}, note='twin: mechanics not probed'), False)
    report = out / 'REPORT.md'
    report.write_text('> OFFLINE PHYSX TWIN OUTPUT (SAPIEN 3 / PhysX 5.3.1). NOT an Isaac Sim result.\n\n' + report.read_text())
    settings = dict(solver=case['solver'], position_iterations=case['position_iterations'],
                    velocity_iterations=case['velocity_iterations'], joint_zero_offsets=case['joint_zero_offsets'],
                    dt=case['dt'], contact_offsets_m=twin.contact_offsets_m,
                    pick_plinth_friction=dict(value=twin.pick_plinth_friction[0], combine=twin.pick_plinth_friction[1]),
                    friction_every_iteration=twin.friction_every_iteration if case['solver'] == 'PGS' else 'n/a (TGS)',
                    duration_s=ref.duration)
    note = dict(backend=TWIN_BACKEND, legacy_settings=a.legacy, settings=settings,
                wall_seconds=time.time() - start, status=result['status'],
                failed_checks={k: v['value'] for k, v in result['checks'].items() if not v['passed']},
                warning='Offline PhysX twin only. Not an Isaac Sim result and not acceptable as native evidence.')
    (out / 'TWIN_NOTE.json').write_text(json.dumps(note, indent=2, allow_nan=False) + '\n')
    print(json.dumps(note, indent=2))


if __name__ == '__main__':
    main()
