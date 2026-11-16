"""D170-R2T fail-closed, algorithm-neutral computational cost ledger.

Configured operations, actual invocations, parameter updates, batch elements
and measured CPU seconds are distinct. No source assumptions about equivalence.
"""
import math

KINDS = (
 "simulator_step","actor_optimizer","critic_optimizer","temperature_optimizer",
 "population_update","ewc_importance","packnet_prune",
 "packnet_retrain","nevo_distillation","nevo_evolution",
 "nevo_mask_reduce","nevo_mask_replace","evaluation",
 "checkpoint","initialization","inference"
)
class CostError(ValueError):
    pass
def require(ok,msg):
    if not ok:raise CostError(msg)

def entry(kind, *, configured_iterations=None, actual_invocations=None,
          parameter_updates=None, batch_elements=None, cpu_seconds=None,
          device="cpu", attribution="observed"):
    require(kind in KINDS,"Unknown work category: "+str(kind))
    require(device=="cpu","Only free CPU accounting enabled")
    require(attribution in ("observed","configured_only"),"Invalid evidence type")
    optional=(configured_iterations,actual_invocations,parameter_updates,batch_elements)
    for value in optional:
        require(value is None or (type(value) is int and value>=0),
                "Counts must be nonnegative integers or unknown")
    require(cpu_seconds is None or
            (isinstance(cpu_seconds,(int,float)) and math.isfinite(cpu_seconds) and cpu_seconds>=0),
            "CPU seconds must be finite nonnegative or unknown")
    if attribution=="configured_only":
        require(all(v is None for v in (actual_invocations,parameter_updates,batch_elements,cpu_seconds)),
                "Configured-only work must not pretend to have measured work")
    else:
        require(actual_invocations is not None and cpu_seconds is not None,
                "Observed work must have actual invocations and measured elapsed seconds")
    if parameter_updates is not None:
        require(actual_invocations is not None and
                parameter_updates>=0,
                "Parameter updates require observed invocation evidence")
    return {"operation":kind,"configured_iterations":configured_iterations,
            "actual_invocations":actual_invocations,
            "parameter_updates":parameter_updates,
            "batch_elements":batch_elements,
            "cpu_seconds":cpu_seconds,"device":device,
            "evidence_kind":attribution}

def validate_ledger(records, source_revision, method, training_authorized=False):
    require(method in ("nevo","ewc","packnet"),"Unknown comparator")
    require(isinstance(source_revision,str) and len(source_revision)==40,
            "Missing pinned source SHA")
    require(isinstance(records,list) and bool(records),"Empty operation ledger")
    seen=set()
    for x in records:
        check=entry(x["operation"],configured_iterations=x["configured_iterations"],
          actual_invocations=x["actual_invocations"],parameter_updates=x["parameter_updates"],
          batch_elements=x["batch_elements"],cpu_seconds=x["cpu_seconds"],
          device=x["device"],attribution=x["evidence_kind"])
        require(x==check,"Unexpected ledger metadata")
        require(x["operation"] not in seen,"Duplicate work category")
        seen.add(x["operation"])
        if not training_authorized:
            require(x["parameter_updates"] in (None,0),"No neural training permitted")
    return True

def original_schedule_config():
    return {
      "nevo":{
       "selected_run":"README original CW20 command; rnd_explore flag inherited True",
       "simulator_step":20*1000000,
       "nevo_distillation":19*40000,
       "population_update":None,
       "nevo_evolution":None,"nevo_mask_reduce":None,
       "nevo_mask_replace":None},
      "ewc":{"selected_run":"Original 20-seed paper CW20 EWC protocol",
       "simulator_step":20*1000000,
       "ewc_importance":19*10},
      "packnet":{"selected_run":"Original 20-seed paper CW20 PackNet protocol",
       "simulator_step":20*1000000,
       "packnet_prune":19,"packnet_retrain":19*100000}
    }

def source_configured_ledger(method,sha):
    schedule=original_schedule_config()[method]
    items=[]
    for kind,count in schedule.items():
        if kind=="selected_run":continue
        items.append(entry(kind,configured_iterations=count,attribution="configured_only"))
    validate_ledger(items,sha,method,training_authorized=False)
    return {"schema":"D170-R2T/cost-ledger-v1",
            "method":method,"source_sha":sha,
            "native_selected_experiment":schedule["selected_run"],
            "configured_operations":items,
            "no_actual_training_invocations_measured":True,
            "actual_trained_capability":None,
            "cost_comparability":"NOT_ESTABLISHED"}

def comparison_gate(reports,success_metrics_verified=False,
                    equal_physics=False,authorized_training=False,
                    full_native_protocols=False):
    require(set(reports)=={"nevo","ewc","packnet"},"Missing comparator")
    for name,item in reports.items():
        validate_ledger(item["configured_operations"],item["source_sha"],name,False)
    requirements={"equal_physics":equal_physics,
      "verified_acquisition_retention":success_metrics_verified,
      "authorized_training":authorized_training,
      "full_original_learning_protocols":full_native_protocols,
      "actual_operation_costs_measured":all(
        not x.get("no_actual_training_invocations_measured",True) for x in reports.values())}
    return {"go":all(requirements.values()),
       "gates":requirements,"result":"GO" if all(requirements.values()) else "NO_GO",
       "no_single_equivalent_update_metric":True}
