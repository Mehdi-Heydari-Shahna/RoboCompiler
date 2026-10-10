#!/usr/bin/env python
"""Compile the physical closed-chain input and write the generated CMG and plan.

    python compile_physical_graph.py --input PHYSICAL_GRAPH.json --partition joint_space --out compiled_local

Without --input, the physical graph of the excavator model is used
(compiler/inputs/physical_graph.json in the folder given with --model).
"""
import argparse, json
from pathlib import Path
from src.bootstrap import excavator_model
from src.compiler import compile_graph
from src.io_utils import save_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', default=None, help='physical graph JSON (default: the excavator model\'s)')
p.add_argument('--partition', default='joint_space')
p.add_argument('--out', default='compiled_local')
excavator_model.add_argument(p)
a = p.parse_args()
excavator_model.use(a.model)
source = Path(a.input) if a.input else excavator_model.path('compiler/inputs/physical_graph.json')
comp = compile_graph(json.loads(source.read_text()), a.partition)
out = Path(a.out)
save_json(out / 'compiled_cmg.json', comp.cmg)
save_json(out / 'plan.json', comp.plan)
print(json.dumps(comp.summary(), indent=2))
