"""Small regression tests for benchmark bookkeeping and acceptance behavior."""
import json
import unittest
import numpy as np
from run_comparison import ROOT, PointGraph, PACDM, NumpyTree, inverse_seed, assessment, trial


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmg=json.loads((ROOT/'original/data/stewart.cmg.json').read_text())
        cls.q=inverse_seed(cls.cmg,cls.cmg['geometry']['nominal_pose'])
        cls.g=PointGraph(cls.cmg,cls.q);cls.b=NumpyTree(cls.cmg)
        cls.x=cls.g.lift(cls.q)

    def test_target_root_label_does_not_change_physical_acceptance(self):
        other=self.q.copy();other[0]+=.01
        result=assessment(self.g,self.b,self.x,self.q[self.g.active],other)
        self.assertTrue(result['complete_success'])
        self.assertFalse(result['root_match'])

    def test_tight_budget_is_reported_not_hidden(self):
        initial=self.x.copy();initial[0]+=.01
        row,_,_,_=trial('PACDM',self.g,self.b,self.q[self.g.active],initial,self.q,1,cold=True)
        self.assertFalse(row['success'])
        self.assertIn('EvaluationLimit',row['message'])
        self.assertEqual(row['combined_residual_jacobian_evaluations'],1)

    def test_every_cold_solver_handles_same_small_perturbation(self):
        initial=self.x.copy();initial[0]+=.003;initial[3]+=.01
        snapshot=initial.copy()
        for name in ('PACDM','PACDM_no_homotopy','TRF_augmented','TRF_points'):
            row,solved,_,_=trial(name,self.g,self.b,self.q[self.g.active],initial,self.q,1500,cold=True)
            self.assertTrue(row['success'],(name,row['message']))
            np.testing.assert_array_equal(initial,snapshot)
            np.testing.assert_allclose(solved[self.g.active],self.q[self.g.active],atol=1e-12,rtol=0)

    def test_nonfinite_and_incorrect_active_coordinates_rejected(self):
        for index,value in ((0,float('nan')),(int(self.g.active[0]),self.x[self.g.active[0]]+.001)):
            bad=self.x.copy();bad[index]=value
            self.assertFalse(assessment(self.g,self.b,bad,self.q[self.g.active],self.q)['complete_success'])


if __name__=='__main__':unittest.main(verbosity=2)
