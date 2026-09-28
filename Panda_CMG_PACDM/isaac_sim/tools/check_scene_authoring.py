"""OPTIONAL offline check of the native USD authoring path (cmg_isaac/scene.py) without Isaac Sim.

Runs build_scene() for a contact and a wrench case on a real in-memory USD stage (pip package
usd-core provides pxr.Usd/UsdGeom/UsdPhysics/UsdShade/UsdLux/Gf/Sdf/Vt) and replaces NVIDIA's
PhysxSchema, which exists only inside Omniverse, with a stand-in that records every call.
It then checks what was authored: solver type and iteration counts, contact/rest offsets,
friction values and combine modes, joint limits shifted by the joint-zero offsets, and the
mimic joint. It catches Python/USD errors in the authoring code; it does NOT show that
Isaac/PhysX accepts or simulates the stage (only a native run does).

Run it in a separate, non-Isaac environment:   pip install usd-core numpy scipy
    python tools/check_scene_authoring.py
"""
from pathlib import Path
import json, math, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CALLS = []


class _Recorder:
    def __init__(self, api, path): self.api, self.path = api, path

    def __getattr__(self, name):
        def call(*args, **kw):
            CALLS.append((self.api, self.path, name, args))
            return _Recorder(self.api + '.' + name, self.path)
        return call


class _Api:
    def __init__(self, name): self.name = name

    def Apply(self, prim, *args):
        CALLS.append((self.name, str(prim.GetPath()), 'Apply', args)); return _Recorder(self.name, str(prim.GetPath()))

    def __call__(self, prim, *args):
        CALLS.append((self.name, str(prim.GetPath()), 'Get', args)); return _Recorder(self.name, str(prim.GetPath()))


class _PhysxSchemaStandIn:
    def __getattr__(self, name): return _Api(name)


def main():
    import pxr
    if hasattr(pxr, 'PhysxSchema') or 'pxr.PhysxSchema' in sys.modules:
        sys.exit('Run this in a non-Isaac environment (usd-core); Isaac already provides PhysxSchema.')
    pxr.PhysxSchema = _PhysxSchemaStandIn(); sys.modules['pxr.PhysxSchema'] = pxr.PhysxSchema
    from pxr import Usd, UsdPhysics, UsdShade
    import cmg_isaac.scene as scene
    from cmg_isaac.cases import get_case
    from cmg_isaac.control import Reference
    from cmg_isaac.geometry import joint_zero_offsets
    cmg = json.loads((ROOT / 'data/panda_cmg.json').read_text()); ref = Reference(ROOT)
    q0 = ref.at(0.)[0].copy(); offsets = joint_zero_offsets(cmg, ref.data['q'], True)
    ids = list(cmg['coordinate_ids']); report = {}

    def recorded(method):
        return [(p, args) for a, p, m, args in CALLS if m == method]

    for mode, name in [('contact', 'tight_socket'), ('wrench', 'nominal')]:
        CALLS.clear(); case = get_case(mode, name); stage = Usd.Stage.CreateInMemory()
        out = scene.build_scene(ROOT, stage, cmg, q0, mode, case, visuals=True, joint_offsets=offsets)
        r = dict(colliders=out['collision_count'], pads=out['pad_count'])
        r['solver'] = sorted({args[0] for p, args in recorded('CreateSolverTypeAttr')})
        r['position_iterations'] = sorted({args[0] for p, args in recorded('CreateSolverPositionIterationCountAttr')})
        r['velocity_iterations'] = sorted({args[0] for p, args in recorded('CreateSolverVelocityIterationCountAttr')})
        groups = {}
        for p, args in recorded('CreateContactOffsetAttr'):
            g = 'cartridge' if p.startswith('/World/cartridge') else 'scene' if p.startswith('/World/Scene') else 'robot'
            groups.setdefault(g, set()).add(round(args[0], 6))
        r['contact_offsets_m'] = {g: sorted(v) for g, v in groups.items()}
        r['rest_offsets_m'] = sorted({args[0] for p, args in recorded('CreateRestOffsetAttr')})
        combine = {p: args[0] for p, args in recorded('CreateFrictionCombineModeAttr')}
        friction = {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
                api = UsdPhysics.MaterialAPI(mat.GetPrim())
                friction[str(prim.GetPath())] = (round(api.GetStaticFrictionAttr().Get(), 6),
                                                 round(api.GetDynamicFrictionAttr().Get(), 6), combine[str(mat.GetPath())])
        if mode == 'contact':
            r['friction'] = {k.rsplit('/', 1)[-1] if 'Scene' in k else k: v for k, v in friction.items()
                             if 'Scene' in k or 'cartridge' in k}
            r['pad_friction'] = sorted({v for k, v in friction.items() if 'finger' in k and v[2] == 'min'})
        worst = 0.
        for j in cmg['joints']:
            if j['type'] == 'revolute':
                joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(out['joints'][j['id']])); c = offsets[ids.index(j['id'])]
                worst = max(worst, abs(joint.GetLowerLimitAttr().Get() - math.degrees(j['limits']['lower'] - c)),
                            abs(joint.GetUpperLimitAttr().Get() - math.degrees(j['limits']['upper'] - c)))
        r['max_joint_limit_difference_deg'] = worst
        r['mimic_joint_applied'] = any(a == 'PhysxMimicJointAPI' and m == 'Apply' for a, p, m, args in CALLS)
        checks = [r['solver'] == [case['solver']], r['position_iterations'] == [case['position_iterations']],
                  r['velocity_iterations'] == [case['velocity_iterations']], r['rest_offsets_m'] == [0.],
                  worst < 1e-4, r['mimic_joint_applied']]
        if mode == 'contact':
            checks += [r['contact_offsets_m'] == dict(robot=[scene.ROBOT_CONTACT_OFFSET], scene=[scene.SCENE_CONTACT_OFFSET],
                                                      cartridge=[scene.OBJECT_CONTACT_OFFSET]),
                       r['friction']['pick_plinth'] == (scene.PICK_PLINTH_FRICTION, scene.PICK_PLINTH_FRICTION, 'min'),
                       r['pad_friction'] == [(case['friction'], case['friction'], 'min')]]
        r['passed'] = all(checks); report[f'{mode}/{name}'] = r
        print(f'{mode}/{name}:', json.dumps(r, default=list))
    status = 'SCENE_AUTHORING_CHECK_PASS' if all(r['passed'] for r in report.values()) else 'SCENE_AUTHORING_CHECK_FAIL'
    print(status)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/scene_authoring_check.json').write_text(json.dumps(dict(
        status=status, usd_version='.'.join(map(str, Usd.GetVersion())), physx_schema='recording stand-in (not NVIDIA PhysxSchema)',
        scope='Offline execution of the native USD authoring code only; not an Isaac/PhysX run.', cases=report),
        indent=2, default=list) + '\n')
    return 0 if status.endswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
