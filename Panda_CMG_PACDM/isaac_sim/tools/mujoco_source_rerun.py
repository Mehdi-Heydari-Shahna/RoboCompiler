"""OPTIONAL diagnostic: rerun the archived SOURCE MuJoCo contact benchmark and report the
cartridge tilt history. Not part of the Isaac validation.

Why this exists
---------------
In the offset pickups (offset_pick, tight_socket) one finger pad reaches the cartridge first
and pushes it sideways, and the cartridge tips by a few degrees. This tool shows that the
source MuJoCo model tips it too, and that MuJoCo's soft friction then realigns it in the grasp
before insertion. PhysX friction is rigid Coulomb friction and keeps the tilt (see
cmg_isaac/scene.py, PICK_PLINTH_FRICTION).

What it runs
------------
source_evidence/mujoco/original_panda_simulation.py, unchanged, imported from a temporary
package with two stand-ins: `load_model` from panda/model.py (bundled source) and a
`PinBackend` whose inverse dynamics is cmg_isaac.rigid.RigidTree.rnea (the source's Pinocchio
backend is not bundled; the author's offline tests, which are not included in this
repository, showed the two agree to ~3e-12 N m on the archived samples). Results use the
installed MuJoCo version; archived results use MuJoCo 3.3.7.

Usage (separate environment; needs  pip install mujoco numpy scipy)
    python tools/mujoco_source_rerun.py                      # nominal, offset_pick, tight_socket, heavy_low_friction
    python tools/mujoco_source_rerun.py --cases tight_socket
Outputs: results/mujoco_source_rerun/ (summary.json plus the source's own json/npz per case).
"""
from pathlib import Path
import argparse, importlib, json, shutil, sys, tempfile
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = dict(nominal={}, offset_pick=dict(offset=.004), tight_socket=dict(width=.048, offset=-.003),
             heavy_low_friction=dict(mass=.30, friction=.5))
TIMES = [4.5, 5., 6., 8., 10., 12., 16., 18.5, 19.5, 22.]

SHIM_MODEL = 'from panda.model import load_model  # bundled source compiler\n'
SHIM_PIN = '''import json
from pathlib import Path
import numpy as np
from cmg_isaac.rigid import RigidTree
ROOT = Path(%r)
class PinBackend:
    """Stand-in for the source Pinocchio backend: RigidTree RNEA (M a + C v + g)."""
    def __init__(self, cmg):
        self.tree = RigidTree(json.loads((ROOT / 'data/panda_cmg.json').read_text()))
    def inverse(self, q, v, a):
        return self.tree.rnea(np.asarray(q, float), np.asarray(v, float), np.asarray(a, float))
''' % str(ROOT)


def tilt_history(npz):
    d = np.load(npz); t = d['time']; R = d['object_R']; T = d['tool_R']
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    i8, i16 = np.searchsorted(t, 8.), np.searchsorted(t, 16.)
    rel0 = T[i8].T @ R[i8]
    grasp_rot = max(np.degrees(Rotation.from_matrix(rel0.T @ (T[i].T @ R[i])).magnitude()) for i in range(i8, i16 + 1))
    return dict(tilt_from_vertical_deg={f'{x:g}s': float(tilt[min(np.searchsorted(t, x), len(t) - 1)]) for x in TIMES},
                max_tilt_4_to_6s_deg=float(tilt[(t >= 4) & (t <= 6)].max()),
                max_rotation_in_grasp_8_to_16s_deg=float(grasp_rot))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    a = p.parse_args()
    import mujoco
    out = ROOT / 'results/mujoco_source_rerun'; out.mkdir(parents=True, exist_ok=True)
    summary = dict(tool='tools/mujoco_source_rerun.py', mujoco_version=mujoco.__version__,
                   archived_source_mujoco_version='3.3.7',
                   feedforward='cmg_isaac.rigid.RigidTree.rnea in place of the unbundled Pinocchio backend',
                   scope='Diagnostic rerun of the source contact model; not Isaac evidence and not a validation gate.',
                   cases={})
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / 'mjsource'; pkg.mkdir()
        shutil.copy2(ROOT / 'source_evidence/mujoco/original_panda_simulation.py', pkg / 'sim.py')
        (pkg / '__init__.py').write_text(''); (pkg / 'model.py').write_text(SHIM_MODEL); (pkg / 'pin_backend.py').write_text(SHIM_PIN)
        sys.path.insert(0, tmp)
        sim = importlib.import_module('mjsource.sim')
        for name in a.cases:
            tag = 'mujoco_rerun_' + name            # the source writes results/<tag>.xml/.npz/.json
            s = sim.run_case(ROOT, tag, **CASES[name])
            for ext in ('npz', 'json'):
                src = ROOT / 'results' / f'{tag}.{ext}'
                if src.exists(): shutil.move(str(src), str(out / f'{name}.{ext}'))
            (ROOT / 'results' / f'{tag}.xml').unlink(missing_ok=True)   # generated scene; its mesh path is relative to results/
            summary['cases'][name] = dict(parameters=CASES[name], **tilt_history(out / f'{name}.npz'),
                                          **{k: s[k] for k in ('max_grasp_slip_m', 'placement_xy_error_m', 'placement_z_error_m',
                                                                'placement_angle_error_deg', 'min_transfer_left_normal_N',
                                                                'min_transfer_right_normal_N', 'max_tool_position_error_m')})
            print(name, json.dumps(summary['cases'][name]), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
