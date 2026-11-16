#!/usr/bin/env python3
"""D170-R2S original-method common CW20 evaluator (untrained CPU only).

CW: EWC/PackNet original 32-wide one-hot env + native test lifecycle.
Nevo: original 12-wide env, pinned SentenceTransformer, full-size native
CoTASPLearner, task-0 masks and sample_actions. No update/gradient calls.
"""
import argparse
import json
import os
import pathlib
import random
import runpy
import sys
import time
import traceback

ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from tools.cw20_protocol import (CW20,METAWORLD_SHA,NEVO_SHA,CONTINUAL_WORLD_SHA,
                                 evaluation_plan,verify_original_orders)
from tools.cw20_evaluator import evaluate_episode,validate,summarise,write_jsonl
from tools.cw20_native import (cw_state,guarded_cw_evaluation,cw_policy_action,
                              cw_param_receipt,seed_native_env)

def log(marker,payload):
    print(marker+" "+json.dumps(payload,sort_keys=True,allow_nan=False),flush=True)

def original_source(path,sha):
    import subprocess
    got=subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    assert got==sha,(got,sha)
    return got

def use_cw(method):
    os.environ.update({"MUJOCO_GL":"osmesa","CUDA_VISIBLE_DEVICES":"-1"})
    base=ROOT/"cw-original"
    original_source(base,CONTINUAL_WORLD_SHA)
    sys.path.insert(0,str(base))
    import numpy as np
    import tensorflow as tf
    import gym
    import mujoco_py
    from continualworld.tasks import TASK_SEQS as CWTASKS
    from continualworld.envs import get_single_env
    start=time.perf_counter()
    # Existing successful original-constructor diagnostic leaves model in scope
    # after calling no training methods; recreating it at original dimensions.
    sys.argv=["scripts/cw_native_cpu.py",method]
    scope=runpy.run_path(str(ROOT/"scripts"/"cw_native_cpu.py"),run_name="__main__")
    agent=scope["native"]
    original_source(base,CONTINUAL_WORLD_SHA)
    original_source(ROOT/"competitor-src",NEVO_SHA) if (ROOT/"competitor-src").is_dir() else None
    from continualworld.tasks import TASK_SEQS as cw
    # Independently read the immutable original Nevo task order without
    # importing its separate JAX dependency environment.
    names=cw["CW20"]
    if tuple(names)!=CW20:raise RuntimeError("Original CW20 changed")
    init=time.perf_counter()-start
    stats=cw_param_receipt(agent)
    rows=[]
    modes=("deterministic",)
    for item in evaluation_plan(100110,[0,10],2,modes):
        idx=item["stage_index"]
        def create(spec):
            environment=get_single_env(spec["task_identity"],one_hot_idx=idx,
                                       one_hot_len=20,randomization="random_init_all")
            seed_native_env(spec["evaluation_seed"],environment,tensorflow=True)
            return environment
        def run():
            return evaluate_episode(item,create,
                lambda obs,s:cw_policy_action(agent,obs,s),
                method=method,source_sha=CONTINUAL_WORLD_SHA,
                model_init_seconds=init,parameter_receipt=stats,
                wrappers={"metaworld_sha":METAWORLD_SHA,"physics":"MuJoCo-2.0",
                    "mujoco_py":"2.0.2.8","wrapper":"original-ContinualWorld",
                    "randomization":"random_init_all","native_observation_width":32,
                    "side_information":"20-position-task-one-hot",
                    "task_label_position":idx},max_steps=5)
        row=guarded_cw_evaluation(agent,idx,run,lambda a:cw_state(a))
        rows.append(row)
    validate(rows)
    return rows

def use_nevo():
    os.environ.update({"CUDA_VISIBLE_DEVICES":"-1","JAX_PLATFORM_NAME":"cpu",
        "JAX_PLATFORMS":"cpu","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
        "XLA_PYTHON_CLIENT_PREALLOCATE":"false"})
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    base=ROOT/"competitor-src"
    original_source(base,NEVO_SHA)
    sys.path.insert(0,str(base/"CW"))
    from continual_world import get_single_env,TASK_SEQS
    from tools.nevo_encoder_integrity_lock import LOCK,verify
    assert LOCK.is_file()
    verify(json.loads(LOCK.read_text()))
    start=time.perf_counter()
    scope=runpy.run_path(str(ROOT/"scripts"/"nevo_full_native_cpu.py"),run_name="__main__")
    learner=scope["learner"]
    assert learner.step==0 and len(learner.actor_pop)==5
    init=time.perf_counter()-start
    stats={"population_size":len(learner.actor_pop),"actor_hidden":[1024]*4,
        "actor_population_parameter_elements":sum(sum(int(v.size) for v in
         flatten_dict(a.params).values()) for a in learner.actor_pop),
        "critic_parameter_elements":sum(sum(int(v.size) for v in
         flatten_dict(q.params).values()) for q in learner.critic_pop),
        "sentence_encoder_elements":sum(int(p.numel()) for p in
         learner.task_encoder.parameters())}
    rows=[]
    # Only task 0 has its original initial masks installed. Evaluating later
    # stage indices without prior sequential training would be fictitious.
    for item in evaluation_plan(100110,[0],2,("stochastic",)):
        def create(spec):
            env=get_single_env(spec["task_identity"],spec["evaluation_seed"],
                randomization="random_init_all",normalize_reward=False)
            seed_native_env(spec["evaluation_seed"],env,tensorflow=False)
            return env
        def action(obs,spec):
            learner.rng=jax.random.PRNGKey(spec["evaluation_seed"]+counter[0])
            counter[0]+=1
            out=learner.sample_actions(0,jnp.asarray(np.asarray(obs,dtype=np.float32)[None,:]),
                                       spec["stage_index"],1.0)
            return np.asarray(out)[0]
        counter=[0]
        rows.append(evaluate_episode(item,create,action,
                method="nevo",source_sha=NEVO_SHA,
                model_init_seconds=init,parameter_receipt=stats,
                wrappers={"metaworld_sha":METAWORLD_SHA,"physics":"MuJoCo-2.0",
                    "mujoco_py":"2.0.2.8","wrapper":"original-Nevo",
                    "randomization":"random_init_all","native_observation_width":12,
                    "side_information":"384-dimensional-task-hint-encoder-initialized",
                    "model_revision":"a50ef00143b4d5391434df20ae11632588ac25be"},
                max_steps=5))
    verify(json.loads(LOCK.read_text()))
    assert learner.step==0 and all(a.step==0 for a in learner.actor_pop)
    validate(rows)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",required=True,choices=("nevo","ewc","packnet"))
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    rows=use_nevo() if args.method=="nevo" else use_cw(args.method)
    write_jsonl(args.output,rows)
    summaries=summarise(rows)
    log("CW20_EVALUATOR_NATIVE_PASS",{
        "method":args.method,"episodes":len(rows),
        "stage_indices":[r["stage_index"] for r in rows],
        "task_identities":[r["task_identity"] for r in rows],
        "evaluation_seeds":[r["evaluation_seed"] for r in rows],
        "success_flags":[r["success"] for r in rows],
        "sum_simulator_interactions":sum(r["simulator_interactions"] for r in rows),
        "inference_calls":sum(r["action_calls"] for r in rows),
        "policy_interfaces_native":True,
        "training_updates":0,"source_modified":False,
        "rows_file":args.output,"mean_results":summaries})
if __name__=="__main__":
    try:main()
    except Exception as e:
        log("CW20_EVALUATOR_NATIVE_BLOCKER",{"type":type(e).__name__,
          "reason":str(e)[:800],"traceback":traceback.format_exc(limit=5)[-1600:],
          "training_updates":0})
        raise
