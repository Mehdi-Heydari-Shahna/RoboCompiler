import numpy as np
import fake_world


def _inertia(manifest):
    from run_isaac import rotations
    ri = rotations([b['inertia_quat_wxyz'] for b in manifest['bodies']])
    diag = np.array([b['inertia_diagonal'] for b in manifest['bodies']])
    return np.einsum('nik,nk,njk->nij', ri, diag, ri)


class RigidBodyView:
    def __init__(self, w):
        self.w = w
        self.prim_paths = [w.paths[k] for k in w.view_order]
    def get_transforms(self): return self.w.transforms()
    def get_velocities(self): return self.w.velocities()
    def get_masses(self):
        return np.array([self.w.manifest['bodies'][k]['mass'] for k in self.w.view_order], np.float32)
    def get_coms(self):
        b = self.w.manifest['bodies']
        return np.array([[*b[k]['com_local'], 0, 0, 0, 1] for k in self.w.view_order], np.float32)
    def get_inertias(self):
        I = _inertia(self.w.manifest)[self.w.view_order]
        return I.transpose(0, 2, 1).reshape(-1, 9).astype(np.float32)
    def apply_forces_and_torques_at_position(self, force, torque, position, indices, is_global):
        if not is_global: raise RuntimeError('fake expects global wrenches')
        self.w.apply(force, torque, position, indices)


class ContactView:
    def __init__(self, w):
        self.w = w
        self.sensor_paths = [w.paths[k] for k in w.view_order]
    def get_net_contact_forces(self, dt): return self.w.contact_forces()


class ArticulationView:
    def __init__(self, count, links, dofs): self.count, self.max_links, self.max_dofs = count, links, dofs


class SimulationView:
    def __init__(self):
        self.w = fake_world.world()
        m = self.w.manifest
        root = next(b['id'] for b in m['bodies'] if b['path'] == m['articulation_root'])
        art = {root}
        for _ in range(len(m['bodies'])):
            before = len(art); art.update(j['body_id'] for j in m['joints'] if j['parent_id'] in art)
            if len(art) == before: break
        self.art = art; self.root = root
        self.dofs = sum(j['type'] in ('hinge', 'slide') and j['body_id'] in art for j in m['joints'])
        self.types = {}
        for b in m['bodies']:
            self.types[b['path']] = ('ObjectType.ArticulationRootLink' if b['id'] == root else
                                     'ObjectType.ArticulationLink' if b['id'] in art else 'ObjectType.RigidBody')
        for j in m['joints']:
            self.types[j['path']] = 'ObjectType.ArticulationJoint' if j['body_id'] in art else 'ObjectType.Invalid'
    def set_subspace_roots(self, root): pass
    def get_object_type(self, path): return self.types.get(path, 'ObjectType.Invalid')
    def create_rigid_body_view(self, pattern): return RigidBodyView(self.w)
    def create_rigid_contact_view(self, pattern):
        import os
        _CONTACT_VIEWS[0] += 1
        if os.environ.get('FAKE_CONTACT_VIEW_FAIL_AFTER_FIRST') and _CONTACT_VIEWS[0] > 1:
            raise RuntimeError('fake: contact view unavailable after first step')
        return ContactView(self.w)
    def create_articulation_view(self, root): return ArticulationView(1, len(self.art), self.dofs)


_CONTACT_VIEWS = [0]


def create_simulation_view(backend, stage_id=None): return SimulationView()
