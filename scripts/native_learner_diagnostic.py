"""Pinned PUBLIC native constructor diagnostics: no training or optimizer steps."""
import os, sys, json, pathlib, subprocess, platform, traceback
os.environ.update({"CUDA_VISIBLE_DEVICES":"-1","JAX_PLATFORM_NAME":"cpu",
"JAX_PLATFORMS":"cpu","WANDB_MODE":"offline","WANDB_DISABLED":"true",
"HF_HUB_DISABLE_TELEMETRY":"1"})
name=sys.argv[1]
assert name in ("nevo","wave")
root=pathlib.Path(__file__).resolve().parent.parent/"competitor-src"
src=root/"CW" if name=="nevo" else root
pins={"nevo":"fe0456e1ae5382213d12421270209eb3f47a9a0f",
"wave":"7814979cb1f8f4324fc3a48881591b1d0c7d7989"}
actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
assert actual==pins[name]
os.chdir(src)
sys.path.insert(0,str(src))
print("NATIVE_SOURCE_RECEIPT",json.dumps({"competitor":name,"source_sha":actual,
"python":platform.python_version(),"neural_updates":0,"source_modified":False}),flush=True)
try:
    import jax, flax, yaml, numpy as np, jax.numpy as jnp
    assert jax.default_backend()=="cpu"
    obs=jnp.zeros((1,12),dtype=jnp.float32)
    act=jnp.zeros((1,4),dtype=jnp.float32)
    cfg=yaml.unsafe_load((src/"configs/sac_cotasp.yaml").read_text())
    print("NATIVE_CONSTRUCTOR_ATTEMPT",json.dumps({"competitor":name,
    "jax":jax.__version__,"flax":flax.__version__,
    "constructor":"CoTASPLearner" if name=="nevo" else "MaskCombinationLearner",
    "obs_shape":list(obs.shape),"act_shape":list(act.shape),
    "config":"configs/sac_cotasp.yaml","neural_updates":0}),flush=True)
    if name=="wave":
        missing=["trained_soms/som_layer_%d.npy"%i for i in range(4)
                 if not (src/"trained_soms"/("som_layer_%d.npy"%i)).is_file()]
        print("WAVE_ORIGINAL_SOM_WEIGHTS",json.dumps({"missing":missing,
        "replacement":False,"som_train_updates":0}),flush=True)
        from jaxrl.agents.sac.sac_mask_combination import MaskCombinationLearner
        learner=MaskCombinationLearner(110,obs,act,20,**cfg)
        result=np.asarray(learner.sample_actions(obs,0))
    else:
        from jaxrl.agents.sac.sac_learner import CoTASPLearner
        learner=CoTASPLearner(5,True,110,obs,act,20,**cfg)
        result=np.asarray(learner.sample_actions(0,obs,0))
    assert result.shape==(1,4) and np.isfinite(result).all()
    assert learner.step==0
    print("NATIVE_LEARNER_FORWARD_PASS",json.dumps({
       "competitor":name,"sha":actual,"output_shape":list(result.shape),
       "finite":True,"learner_step":int(learner.step),"neural_updates":0,
       "som_training":0,"environment_steps":0}),flush=True)
except Exception as e:
    print("NATIVE_LEARNER_BLOCKER",json.dumps({"competitor":name,"sha":actual,
       "exception":type(e).__name__,"message":str(e)[:900],
       "stack_tail":traceback.format_exc(limit=5)[-2000:],
       "neural_updates":0,"source_modified":False,
       "replacement_som_weights":False}),flush=True)
    raise
