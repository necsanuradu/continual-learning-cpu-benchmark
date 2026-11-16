"""Original pinned MetaWorld MT50 simulator reset and step, no neural training."""
import os,sys,json,subprocess,platform,traceback,pathlib
os.environ.update({"CUDA_VISIBLE_DEVICES":"-1","JAX_PLATFORM_NAME":"cpu","JAX_PLATFORMS":"cpu","WANDB_MODE":"offline","MUJOCO_GL":"osmesa"})
name=sys.argv[1]
root=pathlib.Path(__file__).resolve().parent.parent/"competitor-src"
source=root/"CW" if name=="nevo" else root
pin={"nevo":"fe0456e1ae5382213d12421270209eb3f47a9a0f","wave":"7814979cb1f8f4324fc3a48881591b1d0c7d7989"}[name]
actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
assert actual==pin
os.chdir(source);sys.path.insert(0,str(source))
print("ORIGINAL_SIMULATOR_ATTEMPT",json.dumps({"competitor":name,"source_sha":actual,
"python":platform.python_version(),"metaworld_pin":"0875192baaa91c43523708f55866d98eaf3facaf",
"training_updates":0,"original_source_modified":False},sort_keys=True),flush=True)
try:
 import gym,numpy as np
 print("HISTORICAL_GYM_VERSION",gym.__version__,flush=True)
 import mujoco_py
 print("MUJOCO_PY_IMPORT_PASS",getattr(mujoco_py,"__version__","unknown"),flush=True)
 from continual_world import TASK_SEQS,get_single_env
 env_id=TASK_SEQS["cw10"][0]["task"]
 env=get_single_env(env_id,seed=110,randomization="random_init_all")
 obs=np.asarray(env.reset())
 action=np.zeros(env.action_space.shape,dtype=np.float32)
 out=env.step(action)
 assert len(out)==4
 next_obs,reward,done,info=out
 assert np.isfinite(obs).all() and np.isfinite(next_obs).all() and np.isfinite(reward)
 print("ORIGINAL_METAWORLD_RESET_STEP_PASS",json.dumps({"competitor":name,
 "env_id":env_id,"obs_shape":list(obs.shape),"action_shape":list(action.shape),
 "reward":float(reward),"done":bool(done),"source_sha":actual,
 "simulator_steps":1,"neural_training_updates":0},sort_keys=True),flush=True)
 env.close()
except Exception as exc:
 print("ORIGINAL_SIMULATOR_BLOCKED",json.dumps({"competitor":name,"source_sha":actual,
 "type":type(exc).__name__,"reason":str(exc)[:1200],
 "trace_tail":traceback.format_exc(limit=6)[-2800:],
 "neural_updates":0,"physics_substitution":False},sort_keys=True),flush=True)
 raise
