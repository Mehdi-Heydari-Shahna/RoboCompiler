#!/usr/bin/env python
import argparse,json
from pathlib import Path
from src.compiler import compile_graph,support_sets
from src.io_utils import save_json
p=argparse.ArgumentParser();p.add_argument('--input',default='inputs/physical_graph.json');p.add_argument('--out',default='compiled_local');a=p.parse_args()
c=compile_graph(json.loads(Path(a.input).read_text(encoding='utf-8')));out=Path(a.out)
save_json(out/'model.json',c.cmg);save_json(out/'task_plan.json',c.plan);save_json(out/'support_plans.json',[c.support_plan(s)for s in support_sets()])
print('Compiled',len(c.cmg['coordinate_ids']),'physical scalar coordinates;',len(c.plan['conditional_modules']),'conditional task modules;',len(support_sets()),'support plans.')
