"""Minimal data-driven pxr stand-in for run_isaac.py schema/probe checks (test only)."""
import json
from pathlib import Path
import numpy as np


class _Attr:
    def __init__(self, prim, key): self.prim, self.key = prim, key
    def Get(self): return self.prim.attrs.get(self.key)
    def Set(self, value): self.prim.attrs[self.key] = value; return True
    def IsValid(self): return True


class _Rel:
    def __init__(self, targets): self.targets = list(targets)
    def GetTargets(self): return list(self.targets)


class _Schema:
    def __init__(self, prim, *args): self.prim = prim
    def __bool__(self): return bool(self.prim)
    def __getattr__(self, name):
        if name.endswith('Attr') and name.startswith(('Get', 'Create')):
            key = name[3:-4] if name.startswith('Get') else name[6:-4]
            return lambda *a, **k: _Attr(self.prim, key)
        if name.startswith('Get') and name.endswith('Rel'):
            return lambda: _Rel(self.prim.rels.get(name[3:-3], []))
        raise AttributeError(name)


class _NS:
    pass


UsdPhysics = _NS()
for _n in ('RigidBodyAPI', 'CollisionAPI', 'ArticulationRootAPI', 'MassAPI', 'Scene', 'Joint',
           'RevoluteJoint', 'PrismaticJoint', 'FixedJoint', 'SphericalJoint'):
    setattr(UsdPhysics, _n, type(_n, (_Schema,), {}))
PhysxSchema = _NS()
for _n in ('PhysxMimicJointAPI', 'PhysxSceneAPI', 'PhysxRigidBodyAPI'):
    setattr(PhysxSchema, _n, type(_n, (_Schema,), {}))
UsdGeom = _NS()
UsdGeom.GetStageMetersPerUnit = lambda stage: 1.0


class _Id:
    def ToLongInt(self): return 9223002


class _StageCache:
    @staticmethod
    def Get(): return _StageCache()
    def GetId(self, stage): return _Id()


class _Id:
    def ToLongInt(self): return 9223002


UsdUtils = _NS()
UsdUtils.StageCache = _StageCache


class Prim:
    def __init__(self, path, types=(), apis=(), attrs=None, rels=None, schemas=()):
        self.path = path; self.types = set(types); self.apis = set(apis)
        self.attrs = dict(attrs or {}); self.rels = dict(rels or {}); self.schemas = list(schemas)
    def __bool__(self): return True
    def GetPath(self): return self.path
    def IsA(self, cls): return cls in self.types or (cls is UsdPhysics.Joint and bool(self.types & JOINTS))
    def HasAPI(self, api): return api in self.apis
    def GetAppliedSchemas(self): return list(self.schemas)
    def GetRelationship(self, name): return _Rel(self.rels.get(name, []))


class _Null:
    def __bool__(self): return False
    def HasAPI(self, api): return False
    def IsA(self, cls): return False


JOINTS = {UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint, UsdPhysics.FixedJoint, UsdPhysics.SphericalJoint}


class Stage:
    def __init__(self, scene_usda):
        manifest = json.loads((Path(scene_usda).parent/'manifest.json').read_text(encoding='utf-8'))
        self.prims = {}
        paths = {b['id']: b['path'] for b in manifest['bodies']}
        root = manifest['articulation_root']
        self.prims['/World'] = Prim('/World')
        self.prims['/World/Robot'] = Prim('/World/Robot')
        self.prims[manifest['physics_scene_path']] = Prim(manifest['physics_scene_path'], types=[UsdPhysics.Scene])
        for b in manifest['bodies']:
            apis = [UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, PhysxSchema.PhysxRigidBodyAPI]
            schemas = ['PhysicsRigidBodyAPI', 'PhysicsMassAPI', 'PhysxRigidBodyAPI', 'PhysxContactReportAPI']
            if b['path'] == root:
                apis.append(UsdPhysics.ArticulationRootAPI); schemas += ['PhysicsArticulationRootAPI', 'PhysxArticulationAPI']
            self.prims[b['path']] = Prim(b['path'], apis=apis, attrs={'KinematicEnabled': False}, schemas=schemas)
        kinds = {'hinge': UsdPhysics.RevoluteJoint, 'slide': UsdPhysics.PrismaticJoint, 'fixed': UsdPhysics.FixedJoint}
        for j in manifest['joints']:
            attrs = {'JointEnabled': True, 'Axis': 'X', 'ExcludeFromArticulation': False}
            if j.get('limits') is not None:
                limits = np.asarray(j['limits'], float)-j['q_initial']
                if j['type'] == 'hinge':
                    limits = np.rad2deg(limits)
                attrs['LowerLimit'], attrs['UpperLimit'] = float(limits[0]), float(limits[1])
            rels = {'Body0': [] if j['parent_id'] == 0 else [paths[j['parent_id']]], 'Body1': [paths[j['body_id']]]}
            self.prims[j['path']] = Prim(j['path'], types=[kinds[j['type']]], attrs=attrs, rels=rels)
        for c in manifest['closures']:
            self.prims[c['path']] = Prim(c['path'], types=[UsdPhysics.SphericalJoint],
                                         attrs={'ExcludeFromArticulation': True, 'JointEnabled': True})
        joint_paths = {j['id']: j['path'] for j in manifest['joints']}
        for g in manifest['gears']:
            prim = self.prims[g['path']]
            prim.schemas.append('PhysxMimicJointAPI:rotX')
            prim.attrs.update(Gearing=g['mimic_gearing'], Offset=g['mimic_offset_degrees'])
            prim.rels['ReferenceJoint'] = [joint_paths[g['sprocket_joint_id']]]
        self.session = 'session'; self.target = 'root'

    def GetPrimAtPath(self, path): return self.prims.get(str(path), _Null())
    def Traverse(self): return iter(list(self.prims.values()))
    def GetEditTarget(self): return self.target
    def SetEditTarget(self, target): self.target = target
    def GetSessionLayer(self): return self.session
