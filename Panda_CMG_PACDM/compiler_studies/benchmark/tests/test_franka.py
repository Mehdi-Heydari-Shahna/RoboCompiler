from pathlib import Path
import sys,unittest,json,hashlib
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.compiler import compile_graph,variant_inputs,ModelError
from src.task_graph import GeneratedTaskGraph,Solver,target_evaluate,target_kinematics
from src.physics import PhysicalReference,validate,curvature_witness,mx
from src.bootstrap import ROOT
from vendor.pacdm_original import PACDM

class TestFranka(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=json.loads((ROOT/'inputs/physical_graph.json').read_text());cls.c=compile_graph(cls.source)
    def test_01_core_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT/'original/vendor/pacdm_original.py').read_bytes()).hexdigest(),'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de')
    def test_02_affine_identity(self):
        c=self.c;self.assertEqual(c.S.shape,(9,8));self.assertEqual(mx(c.C@c.S),0);self.assertLess(mx(c.physical(c.reduced(c.q_seed))-c.q_seed),1e-12)
    def test_03_task_dimensions(self):
        a=GeneratedTaskGraph(self.c,True);b=GeneratedTaskGraph(self.c,False)
        self.assertEqual((a.n,len(a.passive),a.residual(a.lift(self.c.q_seed))[1].shape),(14,6,(6,14)))
        self.assertEqual((b.n,len(b.passive),b.residual(b.lift(self.c.q_seed))[1].shape),(15,7,(12,15)))
    def test_04_analytic_off_manifold(self):
        g=GeneratedTaskGraph(self.c);x=g.lift(self.c.q_seed);x[g.passive]+=.0008
        J=g.residual(x)[1];h=1e-6;FD=np.column_stack([(g.residual_only(x+h*np.eye(g.n)[i])-g.residual_only(x-h*np.eye(g.n)[i]))/(2*h)for i in range(g.n)])
        self.assertLess(mx(J-FD),2e-8)
    def test_05_target_consistency(self):
        a=np.array([.5,-.1,.4,.13,-.2,.07]);R=np.array(self.source['target_reference_rotation']);T,J=target_evaluate(a,R);U,K,_=target_kinematics(a,R)
        self.assertLess(mx(T-U),1e-14);self.assertLess(mx(J-K),1e-14)
    def test_06_frame_reexpression(self):
        v=dict(variant_inputs(self.source));c=compile_graph(v['body_frame_reexpression']);a=PhysicalReference(c.cmg).tool(c.q_seed)['T'];b=PhysicalReference(self.c.cmg).tool(self.c.q_seed)['T'];self.assertLess(mx(a-b),1e-12)
    def test_07_reverse_edges(self):
        c=compile_graph(dict(variant_inputs(self.source))['reversed_edge_directions']);self.assertLess(mx(PhysicalReference(c.cmg).tool(c.q_seed)['T']-PhysicalReference(self.c.cmg).tool(self.c.q_seed)['T']),1e-12)
    def test_08_opaque_names(self):
        c=compile_graph(dict(variant_inputs(self.source))['opaque_names']);g=GeneratedTaskGraph(c);N,info=PACDM(g).mapping(g.lift(c.q_seed));self.assertTrue(info['success'])
    def test_09_exact_gripper_reuse(self):
        s=Solver(self.c,self.c.q_seed,'compiled_reuse');a=s.x[s.g.active].copy();a[7]=.02;q,N,info=s.step(a);self.assertTrue(info['reused']);self.assertEqual(info['evaluations'],0);self.assertTrue(validate(self.c,q,N,a)['success'])
    def test_10_tool_invalidates(self):
        s=Solver(self.c,self.c.q_seed,'compiled_reuse');a=s.x[s.g.active].copy();a[0]+=.0005;q,N,info=s.step(a);self.assertFalse(info['reused']);self.assertTrue(validate(self.c,q,N,a)['success'])
    def test_11_redundancy_invalidates(self):
        s=Solver(self.c,self.c.q_seed,'compiled_reuse');a=s.x[s.g.active].copy();a[6]+=.001;q,N,info=s.step(a);self.assertFalse(info['reused']);self.assertTrue(validate(self.c,q,N,a)['success'])
    def test_12_invalid_gripper_input(self):
        s=Solver(self.c,self.c.q_seed,'compiled_reuse');a=s.x[s.g.active].copy();a[7]=.05
        with self.assertRaises(ValueError):s.step(a)
    def test_13_affine_ratio_and_bounds(self):
        s=deepcopy(self.source);c=s['affine_couplings'][0];c['multiplier']=2.;s['seed'][c['master']]=.015;s['seed'][c['slave']]=.03;g=compile_graph(s)
        self.assertEqual(g.S[g.plan['slave_index'],g.plan['reduced_master_index']],2);self.assertAlmostEqual(g.upper[g.plan['reduced_master_index']],.02);self.assertLess(mx(g.physical(g.reduced(g.q_seed))-g.q_seed),1e-14)
    def test_14_degenerate_coupling_rejected(self):
        s=deepcopy(self.source);s['affine_couplings'][0]['slave']=s['affine_couplings'][0]['master']
        with self.assertRaises(ModelError):compile_graph(s)
    def test_15_bad_axis(self):
        s=deepcopy(self.source);next(j for j in s['joints']if j['type']=='revolute')['axis']=[0,0,0]
        with self.assertRaises(ModelError):compile_graph(s)
    def test_16_bad_frame(self):
        s=deepcopy(self.source);s['tool']['T_body_tool'][0][0]=2.
        with self.assertRaises(ModelError):compile_graph(s)
    def test_17_bad_inertia(self):
        s=deepcopy(self.source);s['bodies'][2]['inertia_kg_m2'][0][0]=-1.
        with self.assertRaises(ModelError):compile_graph(s)
    def test_18_disconnected_endpoint(self):
        s=deepcopy(self.source);s['joints'][1]['body_b']='missing'
        with self.assertRaises(ModelError):compile_graph(s)
    def test_19_rank_deficient_actuation(self):
        s=deepcopy(self.source);s['actuators'][0]['transmission']={list(s['actuators'][0]['transmission'])[0]:0.}
        with self.assertRaises(ModelError):compile_graph(s)
    def test_20_invalid_chart_rejected(self):
        g=GeneratedTaskGraph(self.c,redundancy='joint2');_,info=PACDM(g).mapping(g.lift(self.c.q_seed));self.assertFalse(info['success'])
    def test_21_curvature_matters(self):
        r=curvature_witness(self.c,self.c.q_seed,np.array([.06,-.04,.05,.3,-.2,.12,.15,.01]),4)
        self.assertTrue(r['success']);self.assertGreater(r['omitted_linear_m_s2'],1e-3)
    def test_22_finger_tool_dependency(self):
        c=self.c;g=GeneratedTaskGraph(c);x=g.lift(c.q_seed);T=g.tool_pose(x);x[g.active[-1]]=.017;self.assertEqual(mx(g.tool_pose(x)-T),0.)
    def test_23_tool_attached_to_finger_rejected_for_cache(self):
        s=deepcopy(self.source);s['tool']['body']='left_finger'
        with self.assertRaises(ModelError):compile_graph(s)
    def test_24_nan_seed_rejected(self):
        s=deepcopy(self.source);s['seed']['joint1']=float('nan')
        with self.assertRaises(ModelError):compile_graph(s)

if __name__=='__main__':unittest.main()
