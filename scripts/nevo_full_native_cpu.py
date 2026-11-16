"""Nevo full-size native learner qualification. PUBLIC competitor sources only.

Original constructor/algorithm unmodified; original full 4x1024 actor, 4x256
critic, population five, task descriptions and pinned public sentence encoder.
No gradient calculation, optimizer update, learner.update, or training.
"""
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import traceback

os.environ.update({
    "CUDA_VISIBLE_DEVICES": "-1",
    "JAX_PLATFORM_NAME": "cpu",
    "JAX_PLATFORMS": "cpu",
    "WANDB_MODE": "offline",
    "WANDB_DISABLED": "true",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "D4RL_SUPPRESS_IMPORT_ERROR": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
})
root = pathlib.Path(__file__).resolve().parent.parent / "competitor-src"
src = root / "CW"
pin = "fe0456e1ae5382213d12421270209eb3f47a9a0f"
sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert sha == pin
os.chdir(str(src))
sys.path.insert(0, str(src))
print("NEVO_NATIVE_SOURCE", json.dumps({
    "source_sha": sha, "python": platform.python_version(),
    "competitor_source_changed": False, "optimizer_updates": 0
}, sort_keys=True), flush=True)
try:
    import gym
    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax
    import yaml
    import torch
    import sentence_transformers
    import transformers
    import huggingface_hub
    from flax.traverse_util import flatten_dict
    assert jax.default_backend() == "cpu"
    assert not torch.cuda.is_available()
    random.seed(110)
    np.random.seed(110)

    from continual_world import TASK_SEQS, get_single_env
    from jaxrl.agents.sac.sac_learner import CoTASPLearner

    cfg = yaml.unsafe_load(pathlib.Path("configs/sac_cotasp.yaml").read_text())
    assert list(cfg["actor_configs"]["hidden_dims"]) == [1024]*4
    assert list(cfg["critic_configs"]["hidden_dims"]) == [256]*4
    first = TASK_SEQS["cw20"][0]
    env = get_single_env(first["task"], 110, randomization="random_init_all")
    observations = np.asarray(env.observation_space.sample(), dtype=np.float32)[np.newaxis]
    actions = np.asarray(env.action_space.sample(), dtype=np.float32)[np.newaxis]
    env.close()
    assert observations.shape == (1,12), observations.shape
    assert actions.shape == (1,4), actions.shape

    print("NEVO_NATIVE_DEPENDENCY_RECEIPT", json.dumps({
      "python": platform.python_version(),
      "jax": jax.__version__, "jaxlib": jax.lib.__version__,
      "flax": flax.__version__, "gym": gym.__version__,
      "torch": torch.__version__, "transformers": transformers.__version__,
      "sentence_transformers": sentence_transformers.__version__,
      "huggingface_hub": huggingface_hub.__version__,
      "backend": jax.default_backend(), "source_sha": sha,
      "original_config": "configs/sac_cotasp.yaml",
      "actor_hidden": list(cfg["actor_configs"]["hidden_dims"]),
      "critic_hidden": list(cfg["critic_configs"]["hidden_dims"]),
      "population_size": 5, "use_critic_pop": False,
      "task_count": len(TASK_SEQS["cw20"]),
      "environment": first["task"],
      "observations_shape": list(observations.shape),
      "actions_shape": list(actions.shape),
      "optimizer_updates": 0
    }, sort_keys=True), flush=True)

    print("NEVO_NATIVE_CONSTRUCTOR_ATTEMPT", flush=True)
    learner = CoTASPLearner(5, False, 110,
                            jnp.asarray(observations), jnp.asarray(actions),
                            len(TASK_SEQS["cw20"]), **cfg)
    assert learner.step == 0
    assert len(learner.actor_pop) == 5
    assert len(learner.critic_pop) == 1
    assert len(learner.layer_sampler) == 4
    assert all(actor.step == 0 for actor in learner.actor_pop)

    def tree_shapes(tree):
        return {"/".join(str(p) for p in path): list(v.shape)
                for path,v in flatten_dict(tree).items()
                if hasattr(v, "shape")}
    actor_shapes = tree_shapes(learner.actor_pop[0].params)
    assert all(tree_shapes(x.params)==actor_shapes for x in learner.actor_pop)
    print("NEVO_NATIVE_LEARNER_CONSTRUCTOR_PASS", json.dumps({
       "source_sha": sha, "class": "CoTASPLearner",
       "actor_shapes": actor_shapes,
       "critic_shapes": tree_shapes(learner.critic_pop[0].params),
       "cumulative_mask_shapes": tree_shapes(learner.cumul_masks),
       "population": len(learner.actor_pop), "actor_steps": [int(x.step) for x in learner.actor_pop],
       "learner_step": int(learner.step), "gradient_updates": 0
    }, sort_keys=True), flush=True)

    emb = np.asarray(learner.task_encoder.encode(first["hint"]))
    assert emb.shape == (384,) and np.isfinite(emb).all()
    print("NEVO_NATIVE_TASK_EMBEDDING_PASS", json.dumps({
       "shape": list(emb.shape), "finite": True,
       "model": "sentence-transformers/all-MiniLM-L12-v2",
       "model_revision": "a50ef00143b4d5391434df20ae11632588ac25be",
       "description_source": "pinned CW/continual_world.py cw20[0]['hint']",
       "gradient_updates": 0
    }, sort_keys=True), flush=True)

    masks = learner.start_task(0.1, 0, first["hint"])
    assert len(masks) == 5
    widths = [[int(np.asarray(m).size) for m in mm] for mm in masks]
    active = [[int(np.count_nonzero(np.asarray(m))) for m in mm] for mm in masks]
    assert all(x==[1024]*4 for x in widths), widths
    assert all(x==[102]*4 for x in active), active
    print("NEVO_NATIVE_TASK_MASKS_PASS", json.dumps({
       "per_population_layer_widths": widths, "per_population_active_neurons": active,
       "native_start_task": True, "gradient_updates": 0
    },sort_keys=True),flush=True)

    output = np.asarray(learner.sample_actions(0,jnp.asarray(observations),0))
    assert output.shape == (1,4), output.shape
    assert np.isfinite(output).all()
    assert learner.step == 0
    assert all(a.step == 0 for a in learner.actor_pop)
    print("NEVO_NATIVE_LEARNER_FORWARD_PASS", json.dumps({
       "source_sha": sha, "output_shape": list(output.shape), "finite": True,
       "population_size": len(learner.actor_pop), "native_mask_active": True,
       "learner_step": int(learner.step), "gradient_updates": 0,
       "optimizer_updates": 0, "neural_training": False
    },sort_keys=True),flush=True)
except Exception as exc:
    print("NEVO_NATIVE_LEARNER_BLOCKER",json.dumps({
       "source_sha": sha, "type": type(exc).__name__,
       "reason": str(exc)[:1400],
       "stack_tail": traceback.format_exc(limit=7)[-3600:],
       "neural_training": False, "optimizer_updates": 0,
       "original_algorithm_changed": False
    },sort_keys=True),flush=True)
    raise
