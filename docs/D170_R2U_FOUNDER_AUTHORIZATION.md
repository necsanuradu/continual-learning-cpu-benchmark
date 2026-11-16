# D170-R2U — Founder decision package: smallest informative learning pilot (NOT AUTHORIZED)

**Decision currently requested:** approve or reject a *future* bounded experimental plan only after CTO review. This document neither requests expenditure nor grants training permission. The current authorization remains **NO-GO to all optimizer updates and neural training**.

## What is already verified

Public original Nevo-CRL, EWC and PackNet can be initialized on free CPU; original MetaWorld and MuJoCo 2.0 can reset/step; common untrained evaluation records and source/static cost checks pass. None of these verifies acquired competence, retained competence, authentic full training behavior, completed task-boundary procedures or an advantage in CPU efficiency. D170-R2U synthetic journal testing, when verified, will establish *accounting behavior only*, not neural training observability.

## Proposed first authorization, only after prerequisite checks: bounded two-task sequential feasibility pilot

**Purpose:** test whether genuinely native learner updates can execute within a strict £0 CPU boundary, whether each originally required stage transition executes, and whether we can collect an interpretable *first-acquisition → second-acquisition → first-task-retention* trajectory. Not a final algorithm ranking or a full CW20 paper reproduction.

Proposed pilot structure: one CPU runner, one method at a time; use the first two distinct original CW20 task positions 0 (hammer-v1) and 1 (push-wall-v1), historical MuJoCo 2.0; one predeclared development seed (110), 10,000 native simulator transitions **per task** maximum and up to ten evaluation episodes per task at baseline, after stage 1 and after stage 2. Preserve original native observation/hint interfaces, published network sizes and required transition operations, including full native EWC importance and PackNet prune/retrain and Nevo distillation where those operations occur. The 10,000 cap is an **explicitly adapted computational pilot**, not an unchanged paper experiment. Do not suppress native transition work to fit runtime limits: if full task-boundary code exceeds the cap, the job must stop and report unresolved resource feasibility.

Predeclare independent evaluation seeds and a bounded evaluation episode horizon of 200; evaluate both tasks after each stage, including first task retention after learning second. Record success rates, raw rewards, confidence across episodes, model snapshots and their identity, parameter/allocation changes, exact simulator transitions, separate actor/critic/temperature/population update counts, EWC Jacobian and replay sampling, PackNet pruning/retraining, Nevo evolution/mask/distillation, checkpoint/evaluation overhead and CPU/memory use.

**Limit:** 25 minutes hard wall-clock per method/job including initialization and original task-transition work, ≤2 CPU runner threads, <=4 GiB peak RSS, and £0 paid spending. Actual GitHub-hosted free minutes, account quota and job availability must be checked before launching; do not assume free capacity. Bound output artifacts to ≤100 MiB per method and no proprietary files, weights or logs published. If memory or CPU limits exceed these proposed constraints, terminate and report a censored pilot rather than silently reducing native model size or skipping transition steps.

**Science threshold:** This pilot's primary success criterion is valid original learner calls, full original task transition fidelity, complete event accounting, stable restored checkpoints, repeatable evaluation and bounded resource usage. Secondary *learning evidence* is improvement versus the untrained baseline on both tasks, with stage-2 retention reported even if poor; no >80% competence is expected from this deliberately short budget. If no measurable acquisition occurs, the result is still a valid **feasibility** result, but **not** a demonstration of continual learning or efficient retention.

**Stopping conditions:** unlicensed redistribution or other unapproved external rights use; any paid spend; unexpected dependencies altering original physics; missing certified model assets; incomplete timestamp/event integrity; unsupported training operations; unintended source or checkpoint changes; unsupported episode/task information; task-mismatched source; nonfinite losses, reproducibility failure, peak memory over cap or CPU wall-time budget exceeded. On interruption mark partial events unknown, save exact completed counts, never call partial outcomes PASS.

## Separate venture-defining acceptance: expansion with retained capabilities

Even if both stages learn and retain skills, this does **not** demonstrate that a network **grows** to support additional tasks. The commercial thesis requires a **single evolving network lineage** that can add capacity and acquire new capability while retaining previously acquired tasks. That must be measured in a separate, privately governed four-stage A→B→C→D capability-expansion experiment with independently useful new input channels, oracle-scored full-task competence, frozen checkpoints before/after each addition and retained whole-task evaluation. Predeclare ≥80% out-of-sample acquired capability and ≤5 percentage-point decline on previously competent tasks as prospective thresholds, with multiple independent seeds. Structural additions, parameter counts, routing and checkpoint continuity require independent verification. No such proof exists yet, and nothing private will be put on public GitHub.

**Founder decision boundary:** authorize, in a subsequent explicit order, *only the limited native learning feasibility pilot* after confirming free compute availability and legal/commercial rights review; authorizing that pilot **does not** authorize full CW20 training or private architecture evaluation, which each require separate approvals and separate private infrastructure/guardrails.

## Outstanding gates

1. Authoritative source-license/permission clearance for public baseline code, redistribution and intended commercial testing; absence of GitHub license metadata is unresolved, not automatic permission.
2. Confirm original training entry points and full task-boundary callback integration can be instrumented externally without changing native algorithm behavior; synthetic journal validation is insufficient by itself.
3. Independently validate the selected evaluation/physics environment is identical for source-original and adapted-budget runs; document task side-information asymmetry.
4. Establish actual free CPU quota, memory and disk limits, and firm fail-closed stop conditions before training permission.
5. Obtain explicit Founder/CTO written authorization for the exact pilot, seed, task order, target machine and cost ceiling; until then, zero updates.

**Recommendation:** CONDITIONAL GO for non-training engineering and a *future* tightly bounded two-stage feasibility-pilot proposal. NO-GO for training today. No public or private benchmark can yet substantiate computational advantage, knowledge retention or architectural expansion.
