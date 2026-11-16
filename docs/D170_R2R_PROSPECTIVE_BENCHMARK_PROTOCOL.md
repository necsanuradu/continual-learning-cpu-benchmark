# D170-R2R — prospective public-only sequential-learning benchmark protocol

**Status:** PROPOSED / FROZEN BEFORE ALL COMPARATIVE TRAINING. No comparative training authorised.

## Exact public environment and task unit

Use the original MetaWorld revision `0875192baaa91c43523708f55866d98eaf3facaf` with **MuJoCo 2.0**, and canonical Continual World CW10 in its declared order, repeated for CW20: hammer-v1, push-wall-v1, faucet-close-v1, push-back-v1, stick-pull-v1, handle-press-side-v1, push-v1, shelf-place-v1, window-close-v1, peg-unplug-side-v1, and repeat those 10. Task *positions* 1–20 are stages; repeated task names remain separate stages in sequential measurement. The original ecosystem can use different task-description strings and randomisation wrappers: report the exact source and all differences. **Never treat matching task names alone as an equal interface.**

Original CW20 academic reference uses 1,000,000 simulator steps **per stage** (20,000,000 steps total). The primary scientific budget must respect this level unless the CTO freezes a different common budget **before** observing comparative outcomes. Proposed 100,000 steps/stage and 1,000,000 steps/stage budget tiers are explicitly separate studies, never merged. A CPU feasibility pilot is not a substitute for full acquisition results. Evaluation is separate from training and receives zero optimiser steps.

## Precommitted deterministic experiment choices

Training seeds: 110, 211, 317, 419, 523 (five independent runs). Evaluation seeds: 100110, 100211, 100317, 100419, 100523; evaluators do not reuse training episodes. Fix task randomisation mode `random_init_all`, physics version, episode horizon 200, robot action limits, reward transformation (prefer native raw reward plus original success `info['success']`), and goal distribution; record achieved goal/task identity and environment wrappers. Use 10 deterministic and 10 stochastic episodes per task at each checkpoint, report both without cherry-picking; reward scale and success rate separate. If an original framework's stochastic draw differs, log this explicitly. Do not set a common inference-time task oracle unless every method is contractually allowed equivalent information.

At stage `k`, evaluate **all 20 task positions** on frozen evaluation seeds, and record pre-stage and post-stage scores where feasible. An unseen-task score is zero-shot, not trained competency. Index evaluation by task identity *and stage index*. Preserve all intermediate checkpoints and exact source/asset hashes. At a minimum evaluate after every stage and every 100,000 environment interactions during each stage; use independently seeded resets and identical evaluation budgets.

## Core predeclared metrics (algorithm neutral)

**Acquisition:** success_{k,k} and learning curve integral over the within-stage training budget; record raw episodic returns and trained-stage success. Competence pass threshold: mean deterministic success >=0.8 at final check for each unique task, plus repeated-task stage accounting. If this threshold is unmet, report failure even if efficiency is attractive.

**Retention:** `R(k)=mean_{j<=k}(S(k,j)-S(j,j))` for task identities measured after their first acquisition; also report the worst single-task difference, average forgetting `F(k)=mean_{j<k} max(0,max_{t<=k} S(t,j)-S(k,j))`, backward transfer and final retained success. Correctly distinguish revisits to a previously seen task from a first acquisition. Failed acquisition cannot be relabelled retained competence.

**Growth and resources:** total parameter count, trainable count, frozen count, newly allocated count, live executor count, per-stage architecture inventory and source/config references; zero architectural growth is a valid result for a fixed-capacity baseline. Do not invent growth for static algorithms. Report memory of model, optimizer state, replay buffers and ancillary pretrained encoders separately, with byte totals and peak resident memory.

**Compute:** count simulator transitions and true optimiser update steps; CPU wall seconds split into model init, encoder init, action inference, simulator, backpropagation, evaluation, checkpoint and cumulative total. Record peak RSS, hardware SKU, core count, CPU threads, OS, exact dependency versions and power/energy only when metered. Report wall time and updates required to cross success thresholds (0.5 and 0.8), censored if not achieved. Report deterministic and stochastic inference latency (median, p95) after warm-up with batching and actual active model resources. No ratios on censored or unsuccessful acquisition; never call smaller parameter count alone a computational win.

**Statistical method:** publish every seed and every run, paired stage/task comparison only when same evaluation identities and environment physics are verified; mean, median, seed dispersion, and 95% uncertainty intervals using seed-level bootstrap. Do not claim significance for overlapping or censored results. Every result identifies whether it is native-original or compatibility-adapted.

## Original-source / common-evaluator distinction

Keep *native reproduction* untouched: Nevo obtains original text hints and 384-dimensional MiniLM embeddings; Continual World uses its original task-index one-hot wrapper (20 positions) and may hide the task ID from selected network components while still routing multi-heads using the one-hot vector. Record input widths, side information, action scaling, reward normalisation, success criterion, and task randomisation. Do not remove or add task information to pretend native methods receive identical observations.

Common evaluation compares the same underlying MetaWorld task distributions, success flags, reward units and fixed seeds; separately report a **task-information comparability audit** indicating whether a method received a task label/index, natural-language hint, or neither. Claims of controlled compute advantage require equivalent competence and explicit stratification of side information. An aligned-interface ablation would be a separate intervention, not authentic-original execution.

## Controls and go/no-go

No training until Founder/CTO separately authorises and the source rights, simulator, model assets, baseline authenticity, measurement implementation and fixed budget are independently verified. Failed checks, missing trained SOM, unavailable source licenses or source substitutions fail closed. No PathLight code, architecture or protected benchmark results may be committed to this public repository. Budget £0, free CPU only.
