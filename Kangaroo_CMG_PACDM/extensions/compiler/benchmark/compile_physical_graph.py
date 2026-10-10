#!/usr/bin/env python
"""Compile a physical graph and write the generated CMG, plan and support plans (no benchmark run).

Usage:  python compile_physical_graph.py --input inputs/physical_graph.json --out compiled_local
"""
import argparse
import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

from src.compiler import compile_graph, support_sets  # noqa: E402
from src.io_utils import save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='inputs/physical_graph.json')
    p.add_argument('--out', default='compiled_local')
    a = p.parse_args()
    comp = compile_graph(json.loads(Path(a.input).read_text(encoding='utf-8')))
    out = Path(a.out)
    save_json(out / 'compiled_cmg.json', comp.cmg)
    save_json(out / 'plan.json', comp.plan)
    save_json(out / 'support_plans.json', {'+'.join(s) or 'floating': comp.support_plan(s) for s in [[]] + support_sets(comp)})
    save_json(out / 'chart_base_cmg.json', comp.chart_cmg())
    p_ = comp.plan
    print(f"Compiled {p_['physical_coordinates']} physical coordinates, {p_['loop_cuts']} loop cuts, "
          f"{p_['augmented_coordinates']} augmented coordinates; {len(p_['modules'])} modules with dependent sizes "
          f"{p_['module_sizes']}; {len(support_sets(comp))} sole-support plans. Output: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
