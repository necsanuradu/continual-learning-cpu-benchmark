"""D170-R2T pinned, fail-closed public training lifecycle audit. No training."""
import ast
import hashlib
import json
import pathlib
import shlex
import subprocess

NEVO_SHA = "fe0456e1ae5382213d12421270209eb3f47a9a0f"
CW_SHA = "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"
NEVO_PATHS = ("README.md","CW/train_our_ep.py","CW/configs/sac_cotasp.yaml")
CW_PATHS = ("examples/paper_cl_experiments.sh","input_args.py","run_cl.py",
 "continualworld/sac/sac.py","continualworld/methods/regularization.py",
 "continualworld/methods/ewc.py","continualworld/methods/packnet.py",
 "continualworld/tasks.py")

class AuditFailure(ValueError):
    pass

def check(value, reason):
    if not value:
        raise AuditFailure(reason)

def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root)] + list(args),
                                   text=True).strip()

def read_pinned(root, sha, paths):
    root = pathlib.Path(root)
    check(root.is_dir(), "Original source checkout absent")
    check(git(root,"rev-parse","HEAD")==sha,"Wrong pinned source revision")
    check(not git(root,"status","--porcelain","--untracked-files=no"),
          "Upstream checkout altered")
    texts, receipts = {}, {}
    for path in paths:
        target = root / path
        check(target.is_file(), "Missing original source "+path)
        blob = git(root,"rev-parse","HEAD:"+path)
        check(blob==git(root,"hash-object",path),"Modified source content "+path)
        content=target.read_bytes()
        texts[path]=content.decode("utf-8")
        receipts[path]={"git_blob":blob,"sha256":hashlib.sha256(content).hexdigest()}
    return texts,receipts

def literal(node):
    try:
        return ast.literal_eval(node)
    except (SyntaxError,TypeError,ValueError):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="int" and len(node.args)==1:
            value=literal(node.args[0])
            check(isinstance(value,(int,float)),"Invalid int default")
            return int(value)
        raise AuditFailure("Dynamic config value prevents static certification")

def parse_defaults(source, nevo=False):
    result={}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):
            continue
        if nevo and node.func.attr.startswith("DEFINE_") and len(node.args)>=2:
            name, value = node.args[:2]
        elif not nevo and node.func.attr=="add_argument" and node.args:
            fields=[k.value for k in node.keywords if k.arg=="default"]
            if not fields:continue
            name,value=node.args[0],fields[0]
        else:continue
        try:
            key=literal(name); default=literal(value)
        except AuditFailure:
            continue
        if isinstance(key,str):
            result[key.lstrip("-")]=default
    return result

def selected_cw_paper_commands(source):
    seen={"ewc":set(),"packnet":set()}
    for line in source.splitlines():
        if not line.strip().startswith("python3 run_cl.py"):
            continue
        flags={}
        for token in shlex.split(line)[2:]:
            if token.startswith("--") and "=" in token:
                key,value=token[2:].split("=",1)
                flags[key]=value
        method=flags.get("cl_method")
        if method not in seen:continue
        seed=flags.get("seed")
        check(seed is not None and seed not in seen[method],"Duplicate paper seed")
        check(flags.get("tasks")=="CW20","Non-CW20 paper configuration")
        seen[method].add(seed)
        if method=="ewc":
            check(float(flags.get("cl_reg_coef","nan"))==10000.0,
                  "Published EWC regularization coefficient mismatch")
        else:
            check(int(flags.get("packnet_retrain_steps","0"))==100000,
                  "Published PackNet retrain-steps mismatch")
            check(float(flags.get("clipnorm","nan"))==0.00002,
                  "Published PackNet clipnorm mismatch")
    for method, seeds in seen.items():
        check(seeds==set(map(str,range(20))),"Published seed schedule mismatch "+method)
    return {key:len(value) for key,value in seen.items()}

def audit_protocol(nevo, cw):
    n=nevo["CW/train_our_ep.py"]
    nf=parse_defaults(n,True)
    c=cw["continualworld/sac/sac.py"]
    cf=parse_defaults(cw["input_args.py"],False)
    for key,value in {"population_size":5,"max_step":1000000,
      "start_training":10000,"distill_steps":40000,"rnd_explore":True,
      "updates_per_step":1}.items():
        check(nf.get(key)==value,"Nevo flag changed: "+key)
    for key,value in {"steps_per_task":1000000,"batch_size":128,
      "hidden_sizes":[256]*4,"multihead_archs":True,
      "reset_buffer_on_task_change":True,"reset_optimizer_on_task_change":True}.items():
        check(cf.get(key)==value,"CW default changed: "+key)
    readme=nevo["README.md"]
    check("CUDA_VISIBLE_DEVICES=0 python train_our_ep.py" in readme,
          "Canonical Nevo CW command missing")
    selected=readme.split("CUDA_VISIBLE_DEVICES=0 python train_our_ep.py",1)[1].split("### Brax",1)[0]
    for value in ("--env_name=cw20","--population_size=5","--use_critic_pop=True",
      "--evo_interval=200000","--sample_ratio=0.25",
      "--start_steps=800000","--end_steps=900000"):
        check(value in selected,"Nevo selected experiment changed: "+value)
    check("--rnd_explore" not in selected and nf["rnd_explore"] is True,
          "Cannot prove selected native distillation enabled")
    for marker in ("if task_idx > 0 and FLAGS.rnd_explore:",
      "for i in range(FLAGS.distill_steps):","agent.rand_net_distill(",
      "agent.start_task(","agent.end_task(","agent.update(",
      "agent.reduce_mask(","agent.set_new_mask(","evolution(",
      "for _ in range(FLAGS.updates_per_step* train_nums):",
      "one_task_env_step - previous_evo_step > FLAGS.evo_interval",
      "one_task_env_step < 800000"):
        check(marker in n,"Nevo native learning operation absent: "+marker)
    dims=nevo["CW/configs/sac_cotasp.yaml"]
    check(dims.count("  - 1024")==4 and dims.count("  - 256")==4,
          "Nevo actor/critic dimension drift")
    for marker in ("self.on_task_end(","self.learn_on_batch(",
      "self._handle_task_change(","self.test_agent("):
        check(marker in c,"CW SAC lifecycle absent: "+marker)
    reg=cw["continualworld/methods/regularization.py"]
    ewc=cw["continualworld/methods/ewc.py"]
    packnet=cw["continualworld/methods/packnet.py"]
    for marker in ("def on_task_start(","self._update_reg_weights(",
      "batches_num: int = 10, batch_size: int = 256",
      "self._merge_weights("):
        check(marker in reg,"EWC task-transition work missing: "+marker)
    check("g.jacobian(" in ewc and "def _get_importance_weights(" in ewc,
          "EWC native Fisher path missing")
    for marker in ("def on_task_end(","self._prune(",
      "for _ in range(self.retrain_steps):","self.learn_on_batch(",
      "def on_test_start(","def on_test_end("):
        check(marker in packnet,"PackNet original learning pathway missing: "+marker)
    seeds=selected_cw_paper_commands(cw["examples/paper_cl_experiments.sh"])
    return {
      "audit":"PASS_ORIGINAL_SOURCE_STATIC_ONLY",
      "selected_experiments":{"nevo":"pinned official README CW20 command",
                              "ewc":"pinned original paper script",
                              "packnet":"pinned original paper script"},
      "nevo":{"population":5,"critic_population_enabled":True,
        "distillation_enabled":True,"distillation_iterations_per_transition":40000,
        "configured_distillation_loop_iterations":760000,
        "evolution_hardcoded_cutoff":800000,
        "mask_evolution_and_reduction_present":True},
      "ewc":{"paper_seed_count":seeds["ewc"],"cl_reg_coef":10000.0,
        "importance_batches_per_transition":10,"importance_batch_size":256,
        "configured_importance_sample_elements":48640},
      "packnet":{"paper_seed_count":seeds["packnet"],
        "retrain_iterations_per_transition":100000,
        "configured_postpruning_retrain_calls":1900000,
        "clipnorm":0.00002},
      "all_counts_configured_not_executed":True,
      "neural_optimizer_updates":0}

def run(nevo_root,cw_root):
    nevo,na=read_pinned(nevo_root,NEVO_SHA,NEVO_PATHS)
    cw,ca=read_pinned(cw_root,CW_SHA,CW_PATHS)
    check(not (pathlib.Path(cw_root)/"LICENSE").is_file(),
          "Continual World license status changed; review rights")
    return {"schema":"D170-R2T/source-training-audit-v1",
      "source_shas":{"nevo":NEVO_SHA,"continual_world":CW_SHA},
      "sources_checked":{"nevo":na,"continual_world":ca},
      "protocol":audit_protocol(nevo,cw),
      "upstream_code_redistributed":False,
      "training_performed":False}
