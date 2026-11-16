"""D170-R2S common evaluation native adapters and state-restoration lifecycle.

The evaluation engine only consumes original Gym environment outputs. Native
model input and inference remain source-specific; no training/update is called.
"""
import hashlib
import json
import random
import time
from tools.cw20_protocol import QualificationError

def _hash_array(x):
    import numpy as np
    a=np.asarray(x)
    h=hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def cw_state(agent):
    """Entire accessible TF model, optimizer-owned model variables and PackNet
    saved views; values hashed, not exported."""
    data={}
    for name in ("actor","critic1","critic2","target_critic1","target_critic2"):
        model=getattr(agent,name,None)
        if model is not None:
            for i, v in enumerate(model.variables):
                data[name+"/"+str(i)+"/"+v.name]=_hash_array(v.numpy())
    # PackNet saved_variables is an internal scratch buffer intentionally
    # populated by on_test_start and retained after on_test_end. It is not
    # a model parameter. Verify all LIVE network variables, ownership masks,
    # and active-view controls; never demand scratch-buffer equality.
    for name in ("owner",):
        table=getattr(agent,name,None)
        if table:
            for i,ref in enumerate(sorted(table,key=lambda r:str(r.deref().name))):
                data[name+"/"+str(i)]=_hash_array(table[ref].numpy())
    for name in ("current_view","freeze_biases_and_normalization"):
        if hasattr(agent,name):
            data[name]=_hash_array(getattr(agent,name).numpy())
    return data

def guarded_cw_evaluation(agent, seq_idx, callback, state_digest=cw_state):
    """Original on_test_start -> policy evaluation -> on_test_end, always.

    Ensures PackNet restores its original weights and temporary ownership view
    even if the environment/policy raises. Does not call any training hook.
    """
    before=state_digest(agent)
    entered=False
    original_error=None
    result=None
    try:
        entered=True
        agent.on_test_start(seq_idx)
        result=callback()
    except BaseException as exc:
        original_error=exc
    finally:
        if entered:
            try:
                agent.on_test_end(seq_idx)
            except BaseException as e:
                raise QualificationError("Native on_test_end failed: "+repr(e)) from e
        after=state_digest(agent)
        if after != before:
            if isinstance(before, dict) and isinstance(after, dict):
                changed = sorted(k for k in (set(before)|set(after)) if before.get(k) != after.get(k))
                details = ",".join(changed[:12])
            else:
                details = "non-mapping snapshot differs"
            raise QualificationError("Native EWC/PackNet LIVE model state not restored: " + details)
    if original_error is not None:
        raise original_error
    return result

def seed_native_env(seed, env, *, tensorflow=False):
    """Best-effort original seeding; record verifier result rather than assert
    that distinct wrappers share identical random goal trajectories."""
    import numpy as np
    seed=int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tensorflow:
        import tensorflow as tf
        tf.random.set_seed(seed)
    raw=env.unwrapped
    if hasattr(raw,"seed"):
        raw.seed(seed)
    if hasattr(env.action_space,"seed"):
        env.action_space.seed(seed)
    return {"seed":seed,"seeded_global_numpy":True,"seeded_python_random":True,
            "seeded_original_raw_env":hasattr(raw,"seed"),
            "tensorflow_rng_seeded":bool(tensorflow)}

def cw_policy_action(agent, observation, spec):
    import tensorflow as tf
    return agent.get_action_test(
        tf.convert_to_tensor(observation,dtype=tf.float32),
        tf.constant(spec["policy_mode"]=="deterministic")
    ).numpy()

def cw_param_receipt(agent):
    network_count=0
    for name in ("actor","critic1","critic2","target_critic1","target_critic2"):
        network=getattr(agent,name,None)
        if network is not None:
            network_count+=sum(int(x.shape.num_elements()) for x in network.variables)
    extras=0
    for tab in ("owner","saved_variables"):
        table=getattr(agent,tab,None)
        if table:extras+=sum(int(v.shape.num_elements()) for v in table.values())
    return {"network_parameters_with_target_critics":network_count,
            "extra_PackNet_owner_and_saved_elements":extras,
            "includes_encoder":False,
            "buffer_and_optimizer_state": "not_included_in_network_count"}
