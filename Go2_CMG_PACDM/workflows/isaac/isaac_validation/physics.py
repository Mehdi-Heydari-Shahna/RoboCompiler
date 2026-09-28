"""CPU PhysX tensor boundary. No controller dynamics are computed here."""
import numpy as np
from .coordinates import from_chart, to_chart, rotation
from .runtime import create_physx_numpy_view, current_stage_id


class PhysicsRobot:
    def __init__(self, metadata, cmg, joint_names, *, stage_id=None):
        # World.reset() must have initialized this exact stage. Do not ask the
        # legacy PhysX interface to guess it using the default stage_id=-1.
        self.stage_id = current_stage_id() if stage_id is None else stage_id
        self.sim = create_physx_numpy_view(self.stage_id)
        self.sim.set_subspace_roots('/')
        self.view = self.sim.create_articulation_view(metadata['base_path'])
        if self.view.count != 1 or self.view.max_dofs != 12 or self.view.max_links != 13:
            raise RuntimeError('Expected one floating Go2, 13 links and 12 DOFs')
        meta = self.view.shared_metatype
        self.dof_names = list(meta.dof_names)
        self.link_names = list(meta.link_names)
        self.order = np.array([self.dof_names.index(x) for x in joint_names])
        self.base_index = self.link_names.index('base')
        self.indices = np.array([0], dtype=np.int32)
        self.coms = np.asarray(self.view.get_coms()[0, :, :3], float)
        self.com = self.coms[self.base_index]
        self.feet = [(self.link_names.index(f['body']), np.array(f['point_m'])) for f in cmg['feet']]
        self.sim.update_articulations_kinematic()

    def set_state(self, q, v):
        pose, velocity = from_chart(q, v, self.com)
        jp = np.zeros((1, 12), dtype=np.float32)
        jv = np.zeros_like(jp)
        jp[0, self.order] = q[6:]
        jv[0, self.order] = v[6:]
        self.view.set_root_transforms(np.asarray([pose], np.float32), self.indices)
        self.view.set_root_velocities(np.asarray([velocity], np.float32), self.indices)
        self.view.set_dof_positions(jp, self.indices)
        self.view.set_dof_velocities(jv, self.indices)
        self.effort(np.zeros(12))
        self.sim.update_articulations_kinematic()

    def state(self):
        return to_chart(self.view.get_root_transforms()[0].copy(),
                        self.view.get_root_velocities()[0].copy(),
                        self.view.get_dof_positions()[0, self.order].copy(),
                        self.view.get_dof_velocities()[0, self.order].copy(), self.com)

    def foot_positions(self):
        poses = self.view.get_link_transforms()[0]
        return np.array([poses[i, :3] + rotation(poses[i, 3:7]) @ point for i, point in self.feet])

    def effort(self, tau):
        commands = np.zeros((1, 12), dtype=np.float32)
        commands[0, self.order] = tau
        self.view.set_dof_actuation_forces(commands, self.indices)

    def push(self, force):
        forces = np.zeros((1, 13, 3), dtype=np.float32)
        forces[0, self.base_index] = force
        poses = self.view.get_link_transforms()[0]
        positions = np.array([[p[:3] + rotation(p[3:7]) @ c for p, c in zip(poses, self.coms)]], dtype=np.float32)
        self.view.apply_forces_and_torques_at_position(forces, None, positions, self.indices, True)


    def close(self):
        """Drop tensor views while their native stage is still alive."""
        self.view = None
        self.sim = None


class ContactMonitor:
    def __init__(self, metadata, dt):
        from omni.physx import get_physx_simulation_interface
        self.dt = dt
        self.foot_paths = list(metadata['foot_collider_paths'].values())
        self.robot_root = metadata['robot_root'] + '/'
        self.normal = np.zeros(4)
        self.events = []
        self.bad = 0
        self.callbacks = 0
        self.points = 0
        self.error = None
        self.time = 0.
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.callback)

    def begin_step(self, time):
        self.time = time
        self.normal[:] = 0

    def callback(self, headers, data):
        from pxr import PhysicsSchemaTools
        try:
            self.callbacks += 1
            for h in headers:
                paths = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.collider0, h.collider1)]
                robots = [p.startswith(self.robot_root) for p in paths]
                if not any(robots):
                    continue
                is_self = all(robots)
                body_path = paths[robots.index(True)]
                normal = 0.
                for k in range(h.contact_data_offset, h.contact_data_offset+h.num_contact_data):
                    point = data[k]
                    impulse = np.asarray(point.impulse, float)
                    n = np.asarray(point.normal, float)
                    normal += abs(float(impulse @ n)) / self.dt
                    self.points += 1
                if not is_self and body_path in self.foot_paths:
                    self.normal[self.foot_paths.index(body_path)] += normal
                elif normal > 1.:
                    self.bad += 1
                    if len(self.events) < 100:
                        self.events.append(dict(time_s=self.time, colliders=paths, normal_force_N=normal))
        except Exception as exc:
            # PhysX callbacks may swallow Python exceptions; make them fatal in the main loop.
            self.error = repr(exc)

    def close(self):
        self.subscription = None
