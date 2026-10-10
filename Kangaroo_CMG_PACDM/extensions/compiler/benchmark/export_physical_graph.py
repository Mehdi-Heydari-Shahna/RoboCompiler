#!/usr/bin/env python
"""Re-export inputs/physical_graph.json from the original v22 CMG and check it against the saved file.

The export is a one-time, provenance-preserving extraction (src/compiler.py:export_physical): bodies,
unordered joint edges, loop cuts, actuators, drive data, the named seed and the two sole sites (centre
of the accepted foot-box bottom corners, contact_reference.foot_corners). It contains no coordinate
order, paths, chart columns, partitions or modules.

Usage:  python export_physical_graph.py            (compare only)
        python export_physical_graph.py --write    (overwrite inputs/physical_graph.json)
"""
import argparse
import json
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from src import bootstrap  # noqa: E402,F401
from src.bootstrap import ROOT  # noqa: E402
from src.compiler import export_physical  # noqa: E402
from kangaroo_pin import legacy  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    a = p.parse_args()
    acc = legacy.load()
    cmg = json.loads((ROOT / 'original/original_v22/data/whole_body_cmg.json').read_text())
    corners = {}
    for body, local, _ in acc.foot_corners({n: np.eye(4) for n in acc.FOOT_NAMES}):
        corners.setdefault(body, []).append(local)
    text = json.dumps(export_physical(cmg, corners), indent=1) + '\n'
    target = ROOT / 'inputs/physical_graph.json'
    if a.write:
        target.write_text(text)
        print('written', target)
        return 0
    same = target.read_text() == text
    print(json.dumps({'identical_to_saved_input': same, 'path': str(target)}, indent=2))
    return 0 if same else 2


if __name__ == '__main__':
    raise SystemExit(main())
