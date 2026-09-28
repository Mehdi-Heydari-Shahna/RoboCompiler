#!/usr/bin/env python3
"""Compile a physical graph without loading the original robot-specific CMG."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from src.compiler import compile_graph, ModelError, canonical_hash


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use an empty/new output directory.')
    try:
        source=json.loads(args.input.read_text(encoding='utf-8'))
        model=compile_graph(source)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f'Compilation failed: {exc}',file=sys.stderr);return 2
    args.out.mkdir(parents=True,exist_ok=True)
    for name,value in [('physical_graph.json',source),('compiled_model.json',model.cmg),('topology_plan.json',model.plan)]:
        (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:model.plan[k] for k in ('physical_bodies','physical_joints','physical_cycles',
           'physical_tree_coordinates','augmented_coordinates','augmented_residual_rows','numerical_rank_status')},indent=2))
    print('Compiled files:',args.out.resolve())
    return 0

if __name__=='__main__':raise SystemExit(main())
