"""Open an interactive replay of the logged, validated native trajectory."""
import argparse,time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from reconstructed_model import whole_body
from native_model import export
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--standing',action='store_true');args=parser.parse_args();c=whole_body()
    path=export(c,ROOT/'models/viewer.xml',floating=args.standing,drive=True,visual=True,contact=args.standing,timestep=.000025 if args.standing else .00005)
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);arr=np.load(ROOT/'results'/('standing_2.5e-05.npz' if args.standing else 'motion_dt_5e-05.npz'))
    qo=np.array([m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in c['coordinate_ids']])
    with mujoco.viewer.launch_passive(m,d) as viewer:
        viewer.cam.lookat[:]=[-.01,0,.65 if args.standing else -.18];viewer.cam.distance=2.2;viewer.cam.azimuth=132;viewer.cam.elevation=-9;viewer.opt.sitegroup[:]=0;viewer.opt.tendongroup[:]=0
        t0=time.perf_counter()
        while viewer.is_running():
            t=((time.perf_counter()-t0)*.5)%arr['sample_t'][-1];i=int(np.argmin(abs(arr['sample_t']-t)))
            if args.standing:d.qpos[:]=arr['q'][i]
            else:d.qpos[qo]=arr['q'][i]
            d.ctrl[:]=arr['force'][i];mujoco.mj_forward(m,d);viewer.sync();time.sleep(1/60)
if __name__=='__main__':main()
