"""D170-R2S independent synthetic-only tests, no competitor imports or training."""
import math
import tempfile
import unittest
from tools.cw20_protocol import (
    CW10, CW20, QualificationError, seed_for, evaluation_plan, verify_original_orders)
from tools.cw20_evaluator import evaluate_episode, summarise, validate, write_jsonl
from tools.cw20_metrics import score_matrix_metrics

class Space:
    shape=(4,)
class Fake:
    action_space=Space()
    def __init__(self,success=True,missing=False):
        self.t=0;self.closed=False;self.success=success;self.missing=missing
    def reset(self):return [0.0]*12
    def step(self,act):
        self.t+=1
        return [0.0]*12,float(self.t),self.t==3,({} if self.missing else {"success":int(self.success)})
    def close(self):self.closed=True
class Tests(unittest.TestCase):
    def test_order_and_repetition(self):
        self.assertEqual(CW20,CW10+CW10)
        orig=[{"task":x,"hint":"a native description"} for x in CW20]
        self.assertTrue(verify_original_orders(orig,CW20)["repeated_task_positions_preserved"])
        bad=list(CW20);bad[0],bad[1]=bad[1],bad[0]
        with self.assertRaises(QualificationError):verify_original_orders(orig,bad)
        with self.assertRaises(QualificationError):verify_original_orders(orig[:10],CW20)
        with self.assertRaises(QualificationError):verify_original_orders([{"task":x} for x in CW20],CW20)
    def test_seed(self):
        self.assertEqual(seed_for(100110,0,0,"deterministic"),
                         seed_for(100110,10,0,"deterministic"))
        self.assertNotEqual(seed_for(100110,0,0,"deterministic"),
                            seed_for(100110,1,0,"deterministic"))
        self.assertNotEqual(seed_for(100110,0,0,"deterministic"),
                            seed_for(100110,0,0,"stochastic"))
        self.assertEqual(len(evaluation_plan(1,[0,10],2,["deterministic"])),4)
        with self.assertRaises(QualificationError):seed_for(1,20,0,"deterministic")
        with self.assertRaises(QualificationError):evaluation_plan(1,[10,0],1,["deterministic"])
    def test_episode_and_aggregation(self):
        spec=evaluation_plan(100110,[0],1,["deterministic"])[0]
        env=Fake()
        row=evaluate_episode(spec,lambda _:env,lambda o,s:[0.0]*4,
                             method="synthetic",source_sha="synthetic",
                             model_init_seconds=2,parameter_receipt={"count":5},
                             wrappers={"metaworld_sha":"0875192baaa91c43523708f55866d98eaf3facaf",
                                       "physics":"synthetic-test"},max_steps=3)
        self.assertTrue(env.closed)
        self.assertEqual((row["raw_reward"],row["success"],row["episode_length"]),(6,1,3))
        self.assertEqual(summarise([row])[0]["mean_raw_reward"],6)
        self.assertIsNotNone(row["steady_action_median_ms"])
        with tempfile.TemporaryDirectory() as tmp:write_jsonl(tmp+"/receipts.jsonl",[row])
        with self.assertRaises(QualificationError):validate([row,row])
    def test_fail_closed(self):
        spec=evaluation_plan(100110,[0],1,["deterministic"])[0]
        for action,missing in (([float("nan")]*4,False),([0.0]*3,False),([0.0]*4,True)):
            e=Fake(missing=missing)
            with self.assertRaises(QualificationError):
                evaluate_episode(spec,lambda _:e,lambda o,s:action,method="synthetic",
                                 source_sha="synthetic",model_init_seconds=0,
                                 parameter_receipt={},
                                 wrappers={"metaworld_sha":"0875192baaa91c43523708f55866d98eaf3facaf"},
                                 max_steps=3)
            self.assertTrue(e.closed)
    def test_metrics_revisit_and_forgetting(self):
        data=score_matrix_metrics(["A","B","A"],
             [[.1,.1],[.8,.2],[.5,.85],[.78,.65]])
        two=data["checkpoints"][1];three=data["checkpoints"][2]
        self.assertAlmostEqual(two["backward_transfer_others"],-.3)
        self.assertAlmostEqual(two["mean_forgetting_others"],.3)
        self.assertTrue(three["is_revisit"])
        self.assertEqual(three["unique_seen"],2)
        self.assertAlmostEqual(three["mean_retention_change"],-.11)
        self.assertAlmostEqual(three["mean_forgetting_all_seen"],.11)
        self.assertAlmostEqual(three["mean_forgetting_others"],.2)
        with self.assertRaises(QualificationError):score_matrix_metrics(["A","B"],[[.0],[.1]])
        with self.assertRaises(QualificationError):score_matrix_metrics(["A"],[[0.0],[math.nan]])
        with self.assertRaises(QualificationError):score_matrix_metrics(["A"],[[0.0],[2.0]])
    def test_restoration_pattern_on_exception(self):
        # Portable synthetic proof of lifecycle control before original PackNet integration.
        from tools.cw20_native import guarded_cw_evaluation
        class Agent:
            def __init__(self):self.x=[1,2];self.calls=[]
            def on_test_start(self,i):self.calls.append("start");self.x[0]=-1
            def on_test_end(self,i):self.calls.append("end");self.x[0]=1
        a=Agent()
        with self.assertRaises(RuntimeError):
            guarded_cw_evaluation(a,0,lambda: (_ for _ in ()).throw(RuntimeError("diagnostic")),
                                  lambda agent:tuple(agent.x))
        self.assertEqual(a.calls,["start","end"])
        self.assertEqual(a.x,[1,2])
        class Faulty(Agent):
            def on_test_end(self,i):self.calls.append("end")
        with self.assertRaises(QualificationError):
            guarded_cw_evaluation(Faulty(),0,lambda:None,lambda agent:tuple(agent.x))
    def test_packnet_scratch_not_part_of_live_model_invariant(self):
        from tools.cw20_native import cw_state
        class FakeTensor:
            def __init__(self, data):self.data=data
            def numpy(self):return self.data
        class FakeOwner:
            def __init__(self):
                self.name="weight:0"
            def deref(self):return self
        class FakeAgent:
            actor=None;critic1=None;critic2=None;target_critic1=None;target_critic2=None
            def __init__(self):
                self.current_view=FakeTensor(-1)
                self.freeze_biases_and_normalization=FakeTensor(False)
                self.saved_variables={FakeOwner():FakeTensor(0)}
                self.owner={}
        agent=FakeAgent()
        before=cw_state(agent)
        next(iter(agent.saved_variables.values())).data=12
        after=cw_state(agent)
        self.assertEqual(before,after,"Original PackNet scratch capture is not a model-weight mutation")
if __name__=="__main__":unittest.main()
