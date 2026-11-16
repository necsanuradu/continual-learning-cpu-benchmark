"""Unmodified-native policy interfaces; common CW20 episode receipt schema.

Pure episode engine never feeds altered observations or invokes train/update.
"""
import json
import math
import os
import pathlib
import resource
import statistics
import time

from tools.cw20_protocol import CW20, QualificationError, METAWORLD_SHA


def _fail(text):
    raise QualificationError(text)


def _action_vector(value, dim):
    try:
        v = value.tolist() if hasattr(value, "tolist") else list(value)
        if not isinstance(v, list) or len(v) != dim or any(isinstance(x, (list, tuple)) for x in v):
            _fail("Native action is not a flat original action-space vector")
        v = [float(x) for x in v]
    except (TypeError, ValueError) as exc:
        raise QualificationError("Native action conversion failed: " + str(exc))
    if not all(math.isfinite(x) for x in v):
        _fail("Native policy returned nonfinite action")
    return v


def evaluate_episode(item, make_env, act, *, method, source_sha,
                     model_init_seconds, parameter_receipt,
                     wrappers, max_steps=200):
    stage = item["stage_index"]
    if not 0 <= stage < 20 or item["task_identity"] != CW20[stage]:
        _fail("CW20 stage/task mismatch")
    if max_steps < 1 or max_steps > 200:
        _fail("Original MetaWorld evaluation max horizon is 200")
    if not source_sha or not wrappers or wrappers.get("metaworld_sha") != METAWORLD_SHA:
        _fail("Original source/wrapper provenance missing or mismatched")
    env = None
    interactions = 0
    action_time_ms = []
    simulator_ms = []
    reward_total = 0.0
    episode_success = False
    done = False
    create_ms = reset_ms = close_ms = 0.0
    episode_started = time.perf_counter_ns()
    try:
        t = time.perf_counter_ns()
        env = make_env(item)
        create_ms = (time.perf_counter_ns() - t) / 1e6
        dim = tuple(env.action_space.shape)
        if len(dim) != 1 or dim[0] != 4:
            _fail("Original MetaWorld action dimension must be 4")
        t = time.perf_counter_ns()
        observation = env.reset()
        reset_ms = (time.perf_counter_ns() - t) / 1e6
        while interactions < max_steps:
            t = time.perf_counter_ns()
            action = _action_vector(act(observation, item), dim[0])
            action_time_ms.append((time.perf_counter_ns() - t) / 1e6)
            t = time.perf_counter_ns()
            step = env.step(action)
            simulator_ms.append((time.perf_counter_ns() - t) / 1e6)
            if not isinstance(step, tuple) or len(step) != 4:
                _fail("Original Gym step() expected (obs,reward,done,info)")
            observation, reward, done, info = step
            if type(info) is not dict or "success" not in info:
                _fail("Raw original MetaWorld success signal missing")
            success = float(info["success"])
            if success not in (0.0, 1.0) or not math.isfinite(success):
                _fail("Invalid native MetaWorld success flag")
            if not math.isfinite(float(reward)):
                _fail("Nonfinite original environment reward")
            episode_success |= bool(success)
            reward_total += float(reward)
            interactions += 1
            if done:
                break
    except Exception as exc:
        print("CW20_EVALUATION_FAIL_CLOSED " + json.dumps({
            "method": method, "stage": stage, "evaluation_seed": item["evaluation_seed"],
            "exception": type(exc).__name__, "error": str(exc)[:600],
            "steps_before_failure": interactions, "neural_training_updates": 0
        }, sort_keys=True), flush=True)
        raise
    finally:
        if env is not None:
            t = time.perf_counter_ns()
            env.close()
            close_ms = (time.perf_counter_ns() - t) / 1e6

    if interactions < 1:
        _fail("No simulator steps executed")
    elapsed_ms = (time.perf_counter_ns() - episode_started) / 1e6
    # ru_maxrss on Linux is a lifetime process peak in KiB, not incremental
    return {
        "schema": "D170-R2S/episode-v1", "status": "untrained-diagnostic",
        "method": method, "source_sha": source_sha,
        "stage_index": stage, "task_identity": CW20[stage],
        "unique_task_index": stage % 10,
        "evaluation_seed": item["evaluation_seed"], "episode_index": item["episode_index"],
        "policy_mode": item["policy_mode"], "raw_reward": reward_total,
        "success": int(episode_success), "episode_length": interactions,
        "environment_done": bool(done),
        "truncated_at_eval_limit": bool(not done and interactions == max_steps),
        "environment": dict(wrappers),
        "model_parameters": dict(parameter_receipt),
        "initialization_seconds": float(model_init_seconds),
        "environment_create_ms": create_ms, "environment_reset_ms": reset_ms,
        "first_action_ms": action_time_ms[0],
        "steady_action_median_ms": statistics.median(action_time_ms[1:]) if len(action_time_ms) > 1 else None,
        "action_total_ms": sum(action_time_ms), "simulator_step_total_ms": sum(simulator_ms),
        "environment_close_ms": close_ms, "episode_total_ms": elapsed_ms,
        "action_calls": len(action_time_ms),
        "simulator_interactions": interactions,
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "optimizer_updates": 0,
    }


def validate(rows):
    if not rows:
        _fail("Missing episodes")
    seen = set()
    for row in rows:
        if row.get("schema") != "D170-R2S/episode-v1":
            _fail("Wrong row schema")
        index = row["stage_index"]
        if not isinstance(index, int) or not 0 <= index < 20 or row["task_identity"] != CW20[index]:
            _fail("Task order violation")
        key = (row["method"], index, row["evaluation_seed"], row["episode_index"], row["policy_mode"])
        if key in seen:
            _fail("Duplicate episode receipt")
        seen.add(key)
        if row["success"] not in (0, 1):
            _fail("Invalid success flag")
        if row["episode_length"] != row["simulator_interactions"] or row["episode_length"] < 1:
            _fail("Simulator-step ledger mismatch")
        if row["optimizer_updates"] != 0:
            _fail("Training evidence cannot enter untrained diagnostic data")
        if not math.isfinite(row["raw_reward"]):
            _fail("Invalid episode reward")
    return True


def summarise(rows):
    validate(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["stage_index"], row["policy_mode"]), []).append(row)
    result = []
    for (method, stage, mode), group in sorted(groups.items()):
        result.append({
            "method": method, "stage_index": stage, "task_identity": CW20[stage],
            "policy_mode": mode, "n_episodes": len(group),
            "mean_raw_reward": statistics.mean(x["raw_reward"] for x in group),
            "success_rate": statistics.mean(x["success"] for x in group),
            "mean_episode_length": statistics.mean(x["episode_length"] for x in group),
            "total_simulator_interactions": sum(x["simulator_interactions"] for x in group),
            "total_action_calls": sum(x["action_calls"] for x in group),
        })
    return result


def write_jsonl(path, rows):
    validate(rows)
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    with open(str(temp), "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(str(temp), str(dest))
