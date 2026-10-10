"""Generate the CMG graph and implementation diagram from the repository."""
from pathlib import Path
import runpy
from go2.model import build_model, save_model

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    save_model(build_model())
    for name in ('draw_go2_cmg.py', 'draw_go2_pipeline.py'):
        runpy.run_path(str(ROOT/name), run_name='__main__')
