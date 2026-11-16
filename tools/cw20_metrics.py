"""Neutral acquisition, retention and forgetting on SYNTHETIC matrices.

Rows: baseline then after each stage. Columns: unique task names in first
appearance order. Repeated tasks revisit a skill; never count a new skill.
"""
import math
import statistics
from tools.cw20_protocol import QualificationError


def score_matrix_metrics(stage_tasks, rows):
    names = tuple(stage_tasks)
    unique = tuple(dict.fromkeys(names))
    if not names or not unique:
        raise QualificationError("Empty task sequence")
    if len(rows) != len(names) + 1 or any(len(row) != len(unique) for row in rows):
        raise QualificationError("Matrix must contain baseline and every stage")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for row in rows for v in row):
        raise QualificationError("All score entries must be finite in [0,1]")
    first = {name: names.index(name) + 1 for name in unique}
    cols = {name: i for i, name in enumerate(unique)}
    checkpoints = []
    for stage in range(1, len(names) + 1):
        seen = [name for name in unique if first[name] <= stage]
        previously_seen = [name for name in seen if name != names[stage - 1]]
        def entry(name):
            col = cols[name]
            start = rows[first[name]][col]
            now = rows[stage][col]
            best = max(rows[t][col] for t in range(first[name], stage + 1))
            return {"task_identity": name, "first_acquisition": start,
                    "current": now, "change_from_first": now - start,
                    "forgetting_from_best": max(0.0, best - now)}
        all_scores = [entry(x) for x in seen]
        other_scores = [entry(x) for x in previously_seen]
        checkpoints.append({
            "stage": stage, "stage_task": names[stage - 1],
            "is_revisit": first[names[stage - 1]] < stage,
            "unique_seen": len(seen),
            "current_stage_success": rows[stage][cols[names[stage - 1]]],
            "mean_retention_change": statistics.mean(e["change_from_first"] for e in all_scores),
            "mean_forgetting_all_seen": statistics.mean(e["forgetting_from_best"] for e in all_scores),
            "backward_transfer_others": statistics.mean(e["change_from_first"] for e in other_scores) if other_scores else None,
            "mean_forgetting_others": statistics.mean(e["forgetting_from_best"] for e in other_scores) if other_scores else None,
            "tasks": all_scores,
        })
    return {"source": "synthetic-or-explicit-score-matrix-not-inferred-training",
            "unique_tasks": list(unique), "first_acquisition_stage": first,
            "checkpoints": checkpoints}
