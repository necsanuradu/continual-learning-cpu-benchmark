"""Original Continual World EWC_SAC / PackNet_SAC construction on CPU.

Original upstream source is fetched at exact git SHA on the GitHub runner,
not copied into this public benchmark repository. No learning, gradient, 
optimiser step, pruning or task-end callbacks are invoked.
"""
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["OMP_NUM_THREADS"] = "2"
name = sys.argv[1]
assert name in ("ewc","packnet"), name
root = (pathlib.Path(__file__).resolve().parent.parent / "cw-original").resolve()
pin = "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"
sha = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
assert sha == pin
sys.path.insert(0,str(root))
os.chdir(str(root))
print("CW_NATIVE_SOURCE_PIN_PASS",json.dumps({
  "repository":"awarelab/continual_world", "sha":sha,
  "candidate":name,"source_modified":False,
  "redistributed":False,"neural_training_updates":0
},sort_keys=True),flush=True)
try:
    import gym
    import numpy as np
    import tensorflow as tf
    import metaworld
    import mujoco_py
    from continualworld.envs import get_cl_env, get_single_env
    from continualworld.tasks import TASK_SEQS
    from continualworld.sac.models import MlpActor
    from continualworld.utils.run_utils import get_sac_class
    from continualworld.utils.enums import BufferType
    from continualworld.utils.utils import get_activation_from_str
    from continualworld.sac.utils.logx import EpochLogger
    assert not tf.config.list_physical_devices("GPU")
    seq = TASK_SEQS["CW20"]
    assert len(seq) == 20 and seq[:10] == seq[10:]
    # Small simulator step-budget for initialization; NOT an altered learning budget.
    # No learning or episode training is executed.
    train_env = get_cl_env(seq, steps_per_task=100, randomization="random_init_all")
    sample_env = get_single_env(seq[0],one_hot_idx=0,one_hot_len=len(seq))
    observation = np.asarray(sample_env.reset(),dtype=np.float32)
    assert train_env.observation_space.shape == (32,),train_env.observation_space.shape
    assert observation.shape == (32,), observation.shape
    assert train_env.action_space.shape == (4,), train_env.action_space.shape
    a = np.zeros((4,),dtype=np.float32)
    step_data = sample_env.step(a)
    assert len(step_data) == 4
    obs2, reward, done, info = step_data
    assert np.isfinite(np.asarray(obs2)).all()
    assert np.isfinite(reward)
    print("CW_NATIVE_ORIGINAL_SIMULATOR_PASS",json.dumps({
        "candidate":name,"cw20_length":len(seq),"onehot_dimension":20,
        "observation_shape":list(observation.shape),
        "action_shape":list(a.shape),"step_reward":float(reward),
        "original_mujoco_python":getattr(mujoco_py,"__version__","unknown"),
        "tensorflow":tf.__version__, "gym":gym.__version__,
        "python":platform.python_version(), "environment_steps":1,
        "training_steps":0
    },sort_keys=True),flush=True)
    sample_env.close()
    actor_kwargs=dict(hidden_sizes=[256]*4,
                      activation=get_activation_from_str("lrelu"),
                      use_layer_norm=True,num_heads=20,hide_task_id=True)
    critic_kwargs=dict(hidden_sizes=[256]*4,
                       activation=get_activation_from_str("lrelu"),
                       use_layer_norm=True,num_heads=20,hide_task_id=True)
    with tempfile.TemporaryDirectory() as tmp:
        logger = EpochLogger(["tsv"],config={"mode":"public-no-training",
                             "candidate":name},group_id="cw_native",
                             output_dir=tmp)
        kwargs=dict(
          env=train_env,test_envs=[],logger=logger,seed=110,
          steps=2000,log_every=20000,
          replay_size=1000,batch_size=128,
          actor_cl=MlpActor,actor_kwargs=actor_kwargs,
          critic_kwargs=critic_kwargs,buffer_type=BufferType.FIFO,
          reset_buffer_on_task_change=True,
          reset_optimizer_on_task_change=True,
          lr=1e-3,alpha="auto",reset_critic_on_task_change=False,
          gamma=.99,target_output_std=.089,
          agent_policy_exploration=False
        )
        Class=get_sac_class(name)
        if name=="ewc":
            native=Class(**kwargs,cl_reg_coef=1.0,regularize_critic=False)
        else:
            native=Class(**kwargs,regularize_critic=False,retrain_steps=0)
        assert native.obs_dim==32 and native.act_dim==4
        assert native.num_tasks==20
        def shapes(variables):
            return {v.name:list(v.shape) for v in variables}
        print("CW_NATIVE_CONSTRUCTOR_PASS",json.dumps({
            "class":Class.__name__,
            "candidate":name,"original_source_sha":sha,
            "actor_parameter_shapes":shapes(native.actor.trainable_variables),
            "critic_parameter_shapes":shapes(native.critic1.trainable_variables),
            "num_tasks":native.num_tasks, "task_onehot_width":20,
            "optimizer_steps":0,"training_steps":0
        },sort_keys=True),flush=True)
        action = np.asarray(native.get_action(
            tf.convert_to_tensor(observation,dtype=tf.float32),
            tf.constant(True)))
        assert action.shape==(4,),action.shape
        assert np.isfinite(action).all()
        print("CW_NATIVE_FORWARD_PASS",json.dumps({
            "candidate":name,"source_sha":sha,
            "action_shape":list(action.shape),"finite":True,
            "neural_updates":0,"gradient_calls":0,
            "on_task_end_called":False,"pruning_called":False
        },sort_keys=True),flush=True)
    train_env.close()
except Exception as e:
    print("CW_NATIVE_BLOCKER",json.dumps({
        "candidate":name,"source_sha":sha,
        "exception":type(e).__name__,"reason":str(e)[:1100],
        "trace":traceback.format_exc(limit=7)[-3200:],
        "neural_optimizer_updates":0,"competitor_source_modified":False
    },sort_keys=True),flush=True)
    raise
