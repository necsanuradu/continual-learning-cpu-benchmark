"""D170-R2S prospective original CW20 task identity and deterministic seed contract.

Standard library only. No model source, neural training, or simulator substitution.
"""
import hashlib
import json

CW10 = (
    "hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1",
    "stick-pull-v1", "handle-press-side-v1", "push-v1", "shelf-place-v1",
    "window-close-v1", "peg-unplug-side-v1",
)
CW20 = CW10 + CW10
METAWORLD_SHA = "0875192baaa91c43523708f55866d98eaf3facaf"
NEVO_SHA = "fe0456e1ae5382213d12421270209eb3f47a9a0f"
CONTINUAL_WORLD_SHA = "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"


class QualificationError(ValueError):
    pass


def verify_original_orders(nevo_tasks, cw_tasks):
    a = tuple(x["task"] for x in nevo_tasks)
    b = tuple(cw_tasks)
    if len(a) != 20 or a != CW20:
        raise QualificationError("Pinned original Nevo CW20 differs from frozen task order")
    if len(b) != 20 or b != CW20:
        raise QualificationError("Pinned original Continual World CW20 differs from frozen order")
    if not all(isinstance(t.get("hint"), str) and t["hint"] for t in nevo_tasks):
        raise QualificationError("Pinned original Nevo natural-language task hints missing")
    h = hashlib.sha256(json.dumps(CW20, separators=(",", ":")).encode()).hexdigest()
    return {"task_order_sha256": h, "stages": 20, "unique_task_names": 10,
            "repeated_task_positions_preserved": True,
            "native_inputs_differ": True}


def seed_for(base_seed, stage_index, episode_index, mode):
    """Repeat stage j+10 uses the same evaluation seed as task j.

    This supports same-goal evaluation across repeated task visits while
    preserving distinct stage indices and one-hot routing positions.
    """
    if type(base_seed) is not int or base_seed < 0:
        raise QualificationError("Invalid base seed")
    if type(stage_index) is not int or not 0 <= stage_index < 20:
        raise QualificationError("Invalid CW20 stage")
    if type(episode_index) is not int or episode_index < 0:
        raise QualificationError("Invalid episode index")
    if mode not in ("deterministic", "stochastic"):
        raise QualificationError("Invalid native inference mode")
    identity = stage_index % 10
    src = json.dumps(["cw20-eval-v1", base_seed, identity, CW10[identity],
                      episode_index, mode], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(src.encode()).digest()[:4], "big") % (2**31 - 2) + 1


def evaluation_plan(base_seed, stages, episodes, modes):
    if type(episodes) is not int or episodes < 1:
        raise QualificationError("Invalid episode count")
    stages = list(stages)
    if not stages or stages != sorted(set(stages)):
        raise QualificationError("Requested stages must be unique and in original order")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise QualificationError("Invalid or repeated inference modes")
    result = []
    for stage in stages:
        if stage < 0 or stage >= 20:
            raise QualificationError("Out-of-bounds stage")
        for mode in modes:
            for episode in range(episodes):
                result.append({
                    "stage_index": stage, "task_identity": CW20[stage],
                    "unique_task_index": stage % 10, "episode_index": episode,
                    "policy_mode": mode,
                    "evaluation_seed": seed_for(base_seed, stage, episode, mode),
                })
    return result
