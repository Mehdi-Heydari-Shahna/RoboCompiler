"""Create a physics-setting variant of an authored Kangaroo USD (development only)."""
from __future__ import annotations
import argparse, json
import ovphysx
from pxr import Plug
Plug.Registry().RegisterPlugins([str(p) for p in ovphysx.codeless_schema_paths()])
from pxr import Usd, Sdf  # noqa: E402

T = Sdf.ValueTypeNames
SCENE_TYPES = {'timeStepsPerSecond': T.UInt, 'minPositionIterationCount': T.UInt, 'maxPositionIterationCount': T.UInt,
               'minVelocityIterationCount': T.UInt, 'maxVelocityIterationCount': T.UInt, 'solverType': T.Token,
               'enableExternalForcesEveryIteration': T.Bool, 'solveArticulationContactLast': T.Bool,
               'frictionType': T.Token, 'enableStabilization': T.Bool, 'bounceThreshold': T.Float,
               'frictionOffsetThreshold': T.Float, 'frictionCorrelationDistance': T.Float,
               'maxBiasCoefficient': T.Float, 'enableEnhancedDeterminism': T.Bool, 'enableCCD': T.Bool}


def make(src, dst, scene=None, iterations=None, body=None, material=None, collision=None):
    stage = Usd.Stage.Open(src)
    sp = stage.GetPrimAtPath('/physicsScene')
    for k, v in (scene or {}).items():
        sp.CreateAttribute('physxScene:' + k, SCENE_TYPES[k]).Set(v)
    if iterations:
        pos, vel = iterations
        for k in ('minPositionIterationCount', 'maxPositionIterationCount'):
            sp.CreateAttribute('physxScene:' + k, T.UInt).Set(pos)
        for k in ('minVelocityIterationCount', 'maxVelocityIterationCount'):
            sp.CreateAttribute('physxScene:' + k, T.UInt).Set(vel)
    for prim in stage.Traverse():
        names = prim.GetAppliedSchemas()
        if 'PhysxRigidBodyAPI' in names:
            if iterations:
                prim.CreateAttribute('physxRigidBody:solverPositionIterationCount', T.Int).Set(iterations[0])
                prim.CreateAttribute('physxRigidBody:solverVelocityIterationCount', T.Int).Set(iterations[1])
            for k, v in (body or {}).items():
                prim.CreateAttribute('physxRigidBody:' + k, T.Float).Set(float(v))
        if 'PhysxArticulationAPI' in names and iterations:
            prim.CreateAttribute('physxArticulation:solverPositionIterationCount', T.Int).Set(iterations[0])
            prim.CreateAttribute('physxArticulation:solverVelocityIterationCount', T.Int).Set(iterations[1])
        if 'PhysxCollisionAPI' in names:
            for k, v in (collision or {}).items():
                prim.CreateAttribute('physxCollision:' + k, T.Float).Set(float(v))
        if 'PhysxMaterialAPI' in names:
            for k, v in (material or {}).items():
                typ = T.Bool if isinstance(v, bool) else T.Float if isinstance(v, (int, float)) else T.Token
                prim.CreateAttribute('physxMaterial:' + k, typ).Set(v)
    stage.GetRootLayer().Export(dst)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--spec', default='{}')
    a = ap.parse_args()
    spec = json.loads(a.spec)
    make(a.src, a.dst, **spec)
