#!/usr/bin/env python
"""Compile the physical closed-chain input and write the generated CMG and plan.

    python compile_physical_graph.py --input inputs/physical_graph.json --partition joint_space --out compiled_local
"""
import argparse, json
from pathlib import Path
from src.compiler import compile_graph
from src.io_utils import save_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', default='inputs/physical_graph.json')
p.add_argument('--partition', default='joint_space')
p.add_argument('--out', default='compiled_local')
a = p.parse_args()
comp = compile_graph(json.loads(Path(a.input).read_text()), a.partition)
out = Path(a.out)
save_json(out / 'compiled_cmg.json', comp.cmg)
save_json(out / 'plan.json', comp.plan)
print(json.dumps(comp.summary(), indent=2))
