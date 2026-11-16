"""D170-R2U synthetic-only source-free journal tests. No gradients or training."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.d170_r2u_events import (
    EventJournal, JournalError, aggregate, read_jsonl, validate_journal,
    METHOD_SHA, SCHEMA)


class FakeClock:
    def __init__(self):
        self.time = 100
    def __call__(self):
        self.time += 10
        return self.time


class JournalTests(unittest.TestCase):
    def journal(self, path, execution="pilot-synthetic", method="nevo"):
        clk=FakeClock()
        return EventJournal(path, execution, method, wall_clock=clk,
                            cpu_clock=clk)

    def test_repeated_events_separate_members_and_stages(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events.jsonl"
            j=self.journal(p)
            a=j.start(stage_index=0,operation="population_update",
                      scope="population",population_member=0)
            j.complete(a,actual_invocations=1,parameter_updates=0,batch_elements=256)
            b=j.start(stage_index=0,operation="population_update",
                      scope="population",population_member=0)
            j.complete(b,actual_invocations=2,parameter_updates=0,batch_elements=512)
            c=j.start(stage_index=10,operation="critic_optimizer",
                      scope="critic",population_member=2)
            j.complete(c,actual_invocations=1,parameter_updates=0,batch_elements=128)
            result=aggregate(read_jsonl(p))
            self.assertEqual(result["completed_events"],3)
            self.assertEqual(len(result["groups"]),2)
            first=result["groups"][0]
            self.assertEqual(first["actual_invocations"],3)
            self.assertEqual(first["batch_elements"],768)
            self.assertEqual(first["parameter_updates"],0)
            self.assertEqual(first["stage_index"],0)
            self.assertEqual(first["completed_events"],2)
            self.assertTrue(result["unresolved_operations_are_unmeasured"])

    def test_duplicate_terminal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.journal(Path(d)/"events")
            key=j.start(stage_index=0,operation="simulator_step",scope="simulator")
            completed=j.complete(key,actual_invocations=1,parameter_updates=0)
            with self.assertRaises(JournalError):
                j.complete(key)
            duplicate=j.records+[completed]
            with self.assertRaises(JournalError):
                validate_journal(duplicate)

    def test_missing_terminal_stays_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events.jsonl"
            j=self.journal(p)
            good=j.start(stage_index=1,operation="evaluation")
            j.complete(good)
            pending=j.start(stage_index=2,operation="nevo_distillation")
            before=aggregate(read_jsonl(p))
            self.assertEqual(before["completed_events"],1)
            self.assertEqual(before["open_event_ids"],[pending])
            j2=self.journal(p)
            self.assertEqual(j2.interrupted_recovery(),[pending])
            final=aggregate(read_jsonl(p))
            self.assertEqual(final["completed_events"],1)
            self.assertEqual(final["interrupted_event_ids"],[pending])
            self.assertFalse(final["open_event_ids"])

    def test_recovery_is_idempotent_and_sequence_continues(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events"
            j=self.journal(p)
            key=j.start(stage_index=0,operation="initialization")
            j2=self.journal(p)
            self.assertEqual(j2.interrupted_recovery(),[key])
            self.assertEqual(j2.interrupted_recovery(),[])
            n=j2.start(stage_index=0,operation="inference")
            self.assertNotEqual(n,key)
            j2.complete(n)
            self.assertEqual(aggregate(read_jsonl(p))["completed_events"],1)

    def test_exception_safe_context(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.journal(Path(d)/"events")
            with self.assertRaises(ZeroDivisionError):
                with j.measure(stage_index=0,operation="evaluation"):
                    1/0
            outcome=aggregate(j.records)
            self.assertEqual(outcome["completed_events"],0)
            self.assertEqual(len(outcome["interrupted_event_ids"]),1)

    def test_source_stage_and_population_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events"
            j=self.journal(p)
            with self.assertRaises(JournalError):
                j.start(stage_index=20,operation="inference")
            with self.assertRaises(JournalError):
                j.start(stage_index=0,operation="unapproved-work")
            with self.assertRaises(JournalError):
                j.start(stage_index=0,operation="inference",population_member=5)
            self.assertFalse(p.exists())
            cw=self.journal(p,method="ewc")
            with self.assertRaises(JournalError):
                cw.start(stage_index=0,operation="inference",population_member=0)
            self.assertFalse(p.exists())

    def test_prevent_unapproved_parameter_updates(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.journal(Path(d)/"events")
            id=j.start(stage_index=0,operation="actor_optimizer",scope="actor")
            with self.assertRaises(JournalError):
                j.complete(id,actual_invocations=1,parameter_updates=1,
                           batch_elements=256)
            state=aggregate(j.records)
            self.assertEqual(state["open_event_ids"],[id])
            self.assertEqual(state["completed_events"],0)
            j.interrupt(id,"unauthorized update blocked")
            self.assertEqual(aggregate(j.records)["completed_events"],0)

    def test_wrong_source_and_stage_identity(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.journal(Path(d)/"events")
            id=j.start(stage_index=0,operation="evaluation")
            forged=dict(j.records[0],source_revision="0"*40)
            with self.assertRaises(JournalError):
                validate_journal([forged])
            forged=dict(j.records[0],task_identity="other")
            with self.assertRaises(JournalError):
                validate_journal([forged])
            j.interrupt(id,"synthetic test")

    def test_crash_truncated_jsonl_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events"
            j=self.journal(p)
            j.start(stage_index=0,operation="inference")
            with p.open("ab") as output:
                output.write(b'{"partial":')
            with self.assertRaises(JournalError):
                read_jsonl(p)
            with self.assertRaises(JournalError):
                self.journal(p)

    def test_changed_execution_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events"
            j=self.journal(p,execution="first")
            key=j.start(stage_index=0,operation="evaluation")
            j.complete(key)
            with self.assertRaises(JournalError):
                self.journal(p,execution="second")

    def test_aggregation_order_independent_of_group_order(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.journal(Path(d)/"events")
            for stage in (2,0,1):
                event=j.start(stage_index=stage,operation="evaluation")
                j.complete(event)
            outcome=aggregate(j.records)
            self.assertEqual([g["stage_index"] for g in outcome["groups"]],
                             [0,1,2])

    def test_unknown_counts_not_converted_to_zero(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.journal(Path(d)/"events")
            event=j.start(stage_index=0,operation="ewc_importance",
                          scope="task_transition")
            j.complete(event,actual_invocations=1,parameter_updates=None,
                       batch_elements=None)
            row=aggregate(j.records)["groups"][0]
            self.assertIsNone(row["parameter_updates"])
            self.assertIsNone(row["batch_elements"])
            self.assertEqual(row["unknown_updates_events"],1)

if __name__=="__main__":
    unittest.main()
