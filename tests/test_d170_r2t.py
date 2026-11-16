"""D170-R2T fail-closed audit and neutral cost tests. Python stdlib only."""
import unittest
from tools.d170_r2t_static import AuditFailure, parse_defaults, selected_cw_paper_commands
from tools.d170_r2t_cost import (
    CostError, entry, validate_ledger, original_schedule_config,
    source_configured_ledger, comparison_gate)

N="fe0456e1ae5382213d12421270209eb3f47a9a0f"
C="73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"

class ContractTests(unittest.TestCase):
    def test_independent_scheduled_work(self):
        s=original_schedule_config()
        self.assertEqual(s["nevo"]["nevo_distillation"],760000)
        self.assertEqual(s["ewc"]["ewc_importance"],190)
        self.assertEqual(s["packnet"]["packnet_retrain"],1900000)
        self.assertEqual(s["nevo"]["simulator_step"],20000000)

    def test_unknown_real_work_not_zero(self):
        r=source_configured_ledger("packnet",C)
        retrain=next(x for x in r["configured_operations"] if x["operation"]=="packnet_retrain")
        self.assertEqual(retrain["configured_iterations"],1900000)
        self.assertIsNone(retrain["actual_invocations"])
        self.assertIsNone(retrain["parameter_updates"])
        self.assertIsNone(retrain["cpu_seconds"])

    def test_fail_closed_false_measured_work(self):
        with self.assertRaises(CostError):
            entry("nevo_distillation",configured_iterations=760000,
                  actual_invocations=760000,attribution="configured_only")
        with self.assertRaises(CostError):
            entry("packnet_retrain",attribution="observed",actual_invocations=None)
        with self.assertRaises(CostError):
            entry("ewc_importance",attribution="observed",actual_invocations=1,
                  cpu_seconds=float("nan"))
        with self.assertRaises(CostError):
            entry("actor_optimizer",attribution="observed",actual_invocations=1,
                  cpu_seconds=.1,parameter_updates=-1)

    def test_no_training_guard(self):
        row=entry("actor_optimizer",configured_iterations=3,actual_invocations=3,
                  parameter_updates=3,batch_elements=384,cpu_seconds=2.4,
                  attribution="observed")
        with self.assertRaises(CostError):
            validate_ledger([row],N,"nevo",False)

    def test_resource_gate(self):
        reports={name:source_configured_ledger(name,N if name=="nevo" else C)
                 for name in ("nevo","ewc","packnet")}
        gate=comparison_gate(reports)
        self.assertEqual(gate["result"],"NO_GO")
        self.assertFalse(gate["gates"]["actual_operation_costs_measured"])
        self.assertFalse(comparison_gate(reports,success_metrics_verified=True,
                      equal_physics=True,authorized_training=True,
                      full_native_protocols=True)["go"])

    def test_no_false_common_update_count(self):
        self.assertNotEqual(original_schedule_config()["nevo"]["nevo_distillation"],
                            original_schedule_config()["packnet"]["packnet_retrain"])
        for name in ("nevo","ewc","packnet"):
            self.assertTrue(source_configured_ledger(name,N if name=="nevo" else C)
                            ["no_actual_training_invocations_measured"])

    def test_original_command_scanner_negative(self):
        good="python3 run_cl.py --tasks=CW20 --seed=0 --cl_method=ewc --cl_reg_coef=10000.0"
        with self.assertRaises(AuditFailure):
            selected_cw_paper_commands(good)
        with self.assertRaises(AuditFailure):
            selected_cw_paper_commands(good.replace("10000.0","0.0"))

    def test_default_parser_negative(self):
        code="flags.DEFINE_integer('distill_steps', int(4e4), 'original')"
        self.assertEqual(parse_defaults(code,True)["distill_steps"],40000)
        self.assertNotEqual(parse_defaults(
            code.replace("4e4","8e4"),True)["distill_steps"],40000)

if __name__=="__main__":
    unittest.main()
