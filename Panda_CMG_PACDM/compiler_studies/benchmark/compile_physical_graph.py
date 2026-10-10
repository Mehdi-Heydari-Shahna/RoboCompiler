import argparse,json
from pathlib import Path
from src.compiler import compile_graph
from src.io_utils import save_json
p=argparse.ArgumentParser();p.add_argument('--input',default='inputs/physical_graph.json');p.add_argument('--out',default='compiled_local');a=p.parse_args()
c=compile_graph(json.loads(Path(a.input).read_text()));out=Path(a.out)
save_json(out/'model.json',c.cmg);save_json(out/'plan.json',c.plan)
print('Compiled physical tree, exact affine lift and task-dependency plan:',out.resolve())
