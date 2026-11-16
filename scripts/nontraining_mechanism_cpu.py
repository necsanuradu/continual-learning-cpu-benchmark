"""Non-training checks of ORIGINAL PUBLIC competitor mask mechanisms.

Never calls a neural optimizer, training loop, environment trajectory, or WAVE SOM.train.
All masks/fitness/SOM weights here are synthetic diagnostics, NOT learned assets.
"""
import copy
import json
import pathlib
import random
import sys
import tempfile

import numpy as np

competitor = sys.argv[1] if len(sys.argv) > 1 else ""
root = pathlib.Path(__file__).resolve().parent.parent / "competitor-src"
if competitor not in ("nevo", "wave") or not root.is_dir():
    raise SystemExit("Usage: nontraining_mechanism_cpu.py nevo|wave with pinned public source")
random.seed(2026)
np.random.seed(2026)

if competitor == "nevo":
    sys.path.insert(0, str(root / "CW"))
    from EA import evolution, mutate_mask_equal_flip

    # Single non-training operator invocation with fixture masks and fixture fitness.
    left = np.array([1] * 8 + [0] * 8, dtype=np.int8)
    right = 1 - left
    patterns = [left, right, np.roll(left, 2), np.roll(right, 3), np.roll(left, 4)]
    population = [[p.copy(), np.roll(p, 1)] for p in patterns]
    original = copy.deepcopy(population)
    activity = [[np.linspace(0.001, 0.025, 16), np.linspace(0.025, 0.001, 16)] for _ in population]
    positions, children, ancestors, swaps, common, uncommon = evolution(
        0.25, population, activity, [10.0, 9.0, 8.0, 7.0, 6.0]
    )
    assert set(positions) == {3, 4}
    assert len(children) == 2 and len(ancestors) == 2
    for child in children:
        assert len(child) == 2
        for layer in child:
            arr = np.asarray(layer)
            assert arr.shape == (16,) and np.isin(arr, [0, 1]).all()
    for before, after in zip(original, population):
        for p, q in zip(before, after):
            assert np.array_equal(p, q), "evolution mutated its provided population"
    altered = mutate_mask_equal_flip(activity[0], original[0], rng=np.random.default_rng(2026))
    assert len(altered) == 2
    assert all(np.isin(x, [0, 1]).all() for x in altered)
    print("NEVO_ORIGINAL_EA_OPERATOR_PASS", json.dumps({
        "source": "CW/EA.py", "mode": "synthetic_single_operator_only",
        "population": len(population), "candidate_count": len(children),
        "replacement_indices": sorted(positions), "swap_count": int(swaps),
        "mutation_operator": "mutate_mask_equal_flip",
        "neural_optimizer_steps": 0, "environment_interactions": 0
    }, sort_keys=True), flush=True)

elif competitor == "wave":
    sys.path.insert(0, str(root))
    from som_mask.som_network import SOMNetwork
    from som_mask.mask_interface import MaskInterface

    # WAVE's original SOM get_mask inference, with RANDOM INITIAL WEIGHTS.
    # No SOM training or reinforcement and no learned historical checkpoints.
    som = SOMNetwork(input_dim=8, som_size=(4, 4), random_seed=2026)
    vector = np.linspace(-0.5, 0.5, 8, dtype=np.float64)
    initial = som.som.get_weights().copy()
    direct = np.asarray(som.get_mask(vector, k=3, mask_dim=64))
    assert direct.shape == (64,) and int(np.sum(direct)) == 12
    assert np.isin(direct, [0, 1]).all()
    assert np.array_equal(initial, som.som.get_weights())

    # MaskInterface loads synthetic initial weights, demonstrating file-based
    # routing wiring, WITHOUT treating the weights as trained SOM evidence.
    with tempfile.TemporaryDirectory(prefix="public_som_diagnostic_") as tmp:
        np.save(pathlib.Path(tmp) / "som_layer_0.npy", initial)
        interface = MaskInterface(input_dim=8, num_layers=1, som_size=(4, 4),
                                  mask_dim=64, weight_dir=tmp)
        routed = np.asarray(interface.get_mask(0, vector, topk=3))
    assert np.array_equal(direct, routed)
    assert np.array_equal(initial, som.som.get_weights())
    print("WAVE_ORIGINAL_SOM_INFERENCE_PASS", json.dumps({
        "sources": ["som_mask/som_network.py", "som_mask/mask_interface.py"],
        "weight_provenance": "synthetic_random_initial_weights_not_learned",
        "mask_size": int(routed.size), "mask_active": int(routed.sum()),
        "top_k": 3, "neural_optimizer_steps": 0, "som_training_steps": 0,
        "environment_interactions": 0
    }, sort_keys=True), flush=True)
