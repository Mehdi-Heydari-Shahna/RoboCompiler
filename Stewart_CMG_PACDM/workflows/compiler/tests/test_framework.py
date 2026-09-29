"""Regression tests for the extension, separate from timed measurements."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib, json, unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from src.bootstrap import ROOT,PRIOR
from src.compiler import compile_graph,from_original_cmg,ModelError,canonical_hash
from src.experiments import make_state,ResidualOnly,structural_sparsity,world_poses,physical_acceptance
from numpy_backend import NumpyTree
from vendor.pacdm_original import PACDM


class FrameworkRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads((PRIOR/'original/data/stewart.cmg.json').read_text())
        cls.source=json.loads((ROOT/'inputs/physical_graph.json').read_text())
        cls.reference=NumpyTree(cls.original)

    def test_physical_input_has_no_compilation_fields(self):
        self.assertFalse({'tree','cuts','closures','coordinate_ids','jacobian','constraint_equations'} & self.source.keys())
        self.assertEqual(canonical_hash(self.source),canonical_hash(from_original_cmg(self.original)))

    def test_input_not_mutated(self):
        s=deepcopy(self.source);before=canonical_hash(s);compile_graph(s)
        self.assertEqual(before,canonical_hash(s))

    def test_expected_physical_and_compiled_counts(self):
        p=compile_graph(self.source).plan
        self.assertEqual((p['physical_bodies'],p['physical_joints'],p['physical_cycles']),(21,25,5))
        self.assertEqual((p['physical_tree_coordinates'],p['augmented_coordinates'],p['augmented_residual_rows']),(21,36,30))
        self.assertEqual((p['point_residual_rows'],p['dimensional_mobility'],p['generated_massless_frames']),(15,6,2))
        self.assertEqual(len(p['candidate_modules']),1)
        self.assertEqual(p['numerical_rank_status'],'not_checked_until_a_configuration_is_supplied')

    def test_deterministic_under_body_and_joint_order(self):
        s=deepcopy(self.source);s['bodies'].reverse();s['joints'].reverse()
        self.assertEqual(canonical_hash(compile_graph(self.source).cmg),canonical_hash(compile_graph(s).cmg))

    def test_all_six_tree_choices_preserve_physical_pose(self):
        for joint in sorted(j['id'] for j in self.source['joints'] if j['type']=='spherical'):
            with self.subTest(joint=joint):
                c=compile_graph(self.source,joint)
                qzero,_,_=make_state(c,self.original,self.reference,0.)
                q,_,W=make_state(c,self.original,self.reference,10.6)
                g=c.graph(qzero);check=physical_acceptance(g,NumpyTree(c.cmg),self.source,g.lift(q),q[g.active],W)
                self.assertTrue(check['success']);self.assertLess(check['pose_error'],1e-10)

    def test_generated_derivative_and_sparsity_off_manifold(self):
        c=compile_graph(self.source);q,_,_=make_state(c,self.original,self.reference,0.);g=c.graph(q)
        x=g.lift(q);rng=np.random.default_rng(5771);x[g.passive]+=.001*rng.normal(size=len(g.passive))
        f=ResidualOnly(g);r,J,_=g.residual(x);np.testing.assert_allclose(r,f(x),atol=1e-11,rtol=0)
        mask=structural_sparsity(g);self.assertLess(np.max(np.abs(J[~mask])),1e-12)
        d=rng.normal(size=g.n);d/=np.linalg.norm(d);h=1e-6
        np.testing.assert_allclose((f(x+h*d)-f(x-h*d))/(2*h),J@d,atol=1e-7,rtol=0)

    def test_unsupported_scalar_loop_rejected(self):
        s=deepcopy(self.source)
        a=deepcopy(next(j for j in s['joints'] if j['type']=='revolute'));a['id']='duplicate_scalar_edge';s['joints'].append(a)
        with self.assertRaisesRegex(ModelError,'spherical'):compile_graph(s)

    def test_nonunit_input_axis_rejected(self):
        s=deepcopy(self.source);next(j for j in s['joints'] if j['type']=='prismatic')['axis']=[0,0,2]
        with self.assertRaisesRegex(ModelError,'unit'):compile_graph(s)

    def test_bad_preferred_tree_rejected(self):
        with self.assertRaises(ModelError):compile_graph(self.source,'not_a_joint')

    def test_actuator_chart_requires_numerical_validation(self):
        s=deepcopy(self.source);s['actuators']=s['actuators'][:-1];c=compile_graph(s)
        q,_,_=make_state(c,self.original,self.reference,0.);g=c.graph(q);_,info=PACDM(g).mapping(g.lift(q))
        self.assertFalse(info['success'])

    def test_original_source_hashes_unchanged(self):
        records=json.loads((PRIOR/'ORIGINAL_SHA256.json').read_text())
        for name,expected in records.items():
            with self.subTest(name=name):self.assertEqual(hashlib.sha256((PRIOR/name).read_bytes()).hexdigest(),expected)

    def test_core_matches_production_launch_record(self):
        path=ROOT/'execution_evidence/production_launch.json'
        if not path.exists():self.skipTest('No recorded production launch record')
        for name,expected in json.loads(path.read_text())['source_file_hashes'].items():
            with self.subTest(name=name):self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)


if __name__=='__main__':unittest.main(verbosity=2)
