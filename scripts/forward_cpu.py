"""Original public policy source, CPU forward-only diagnostic.

No optimizer objects, gradient calls, environment interactions, or training.
A passing result is NOT a full algorithm or simulator qualification.
"""
import json
import pathlib
import sys

competitor = sys.argv[1] if len(sys.argv) > 1 else ""
if competitor not in ("nevo", "wave"):
    raise SystemExit("Usage: forward_cpu.py nevo|wave")

root = pathlib.Path(__file__).resolve().parent.parent
source = root / "competitor-src"
module_dir = source / "CW" if competitor == "nevo" else source
if not module_dir.is_dir():
    raise RuntimeError("Expected pinned public source path is missing: " + str(module_dir))

sys.path.insert(0, str(module_dir))

import jax
import jax.numpy as jnp
import numpy as np
import flax

backend = jax.default_backend()
print("CPU_ENVIRONMENT_RECEIPT", json.dumps({
    "competitor": competitor,
    "jax": jax.__version__,
    "flax": flax.__version__,
    "backend": backend,
    "devices": [str(device) for device in jax.devices()],
    "training_updates": 0,
}, sort_keys=True), flush=True)
if backend != "cpu":
    raise RuntimeError("CPU-only execution required")

from jaxrl.networks.policies import MSEPolicy

obs = jnp.arange(39, dtype=jnp.float32).reshape(1, 39) / 100
policy = MSEPolicy(hidden_dims=(32, 32), action_dim=4)
parameters = policy.init(jax.random.PRNGKey(2026), obs)
actions = np.asarray(policy.apply(parameters, obs))
if actions.shape != (1, 4) or not np.isfinite(actions).all():
    raise AssertionError("Policy produced wrong shape or nonfinite values")

print("CPU_ORIGINAL_SOURCE_POLICY_FORWARD_PASS", json.dumps({
    "status": "PASS",
    "shape": list(actions.shape),
    "actions": actions.tolist(),
    "parameter_count": sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(parameters)),
    "training_updates": 0,
}, sort_keys=True), flush=True)
