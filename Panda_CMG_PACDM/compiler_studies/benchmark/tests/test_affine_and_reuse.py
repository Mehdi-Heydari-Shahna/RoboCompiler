"""Additional regression coverage for affine bounds and reuse transitions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bootstrap import ROOT
from src.compiler import compile_graph
from src.physics import validate
from src.task_graph import Solver


class TestAffineAndReuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((ROOT / 'inputs/physical_graph.json').read_text())

    def test_negative_affine_ratio_and_offset_preserve_bounds(self):
        source = deepcopy(self.source)
        coupling = source['affine_couplings'][0]
        coupling.update(multiplier=-2.0, offset=0.06)
        source['seed'][coupling['master']] = 0.02
        source['seed'][coupling['slave']] = 0.02
        compiled = compile_graph(source)
        master = compiled.plan['reduced_master_index']
        slave = compiled.plan['slave_index']
        limits = {joint['id']: joint.get('limits') for joint in source['joints']}
        self.assertAlmostEqual(compiled.lower[master], 0.01)
        self.assertAlmostEqual(compiled.upper[master], 0.03)
        for value in np.linspace(compiled.lower[master], compiled.upper[master], 9):
            reduced = compiled.reduced(compiled.q_seed)
            reduced[master] = value
            physical = compiled.physical(reduced)
            self.assertAlmostEqual(physical[slave], -2 * value + 0.06)
            for coordinate, q in zip(compiled.cmg['coordinate_ids'], physical):
                self.assertGreaterEqual(q, limits[coordinate]['lower'] - 1e-14)
                self.assertLessEqual(q, limits[coordinate]['upper'] + 1e-14)
            np.testing.assert_allclose(compiled.C @ physical, [0.06], atol=1e-14)

    def test_reuse_after_changed_target_matches_recomputed_mapping(self):
        compiled = compile_graph(self.source)
        cached = Solver(compiled, compiled.q_seed, 'compiled_reuse')
        recomputed = Solver(compiled, compiled.q_seed, 'compiled_pacdm')
        target = cached.x[cached.g.active].copy()
        sequence = [(0, 0.0004), (7, -0.005), (6, 0.0003), (7, 0.002)]
        for index, change in sequence:
            target[index] += change
            q_cached, mapping_cached, info = cached.step(target)
            q_full, mapping_full, _ = recomputed.step(target)
            self.assertEqual(info['reused'], index == 7)
            self.assertTrue(validate(compiled, q_cached, mapping_cached, target)['success'])
            self.assertTrue(validate(compiled, q_full, mapping_full, target)['success'])
            np.testing.assert_allclose(q_cached, q_full, atol=1e-8, rtol=0)
            np.testing.assert_allclose(mapping_cached, mapping_full, atol=1e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
