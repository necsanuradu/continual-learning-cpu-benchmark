# D170-R2S — Original CW20 evaluation and CPU feasibility (no training)

**Scope:** Shared episode record, fixed seed schedule, original source-specific policy input and inference, original MetaWorld success/reward, synthetic-only retention metrics. No training or claims about learned competence.

Native lanes are distinct. Nevo uses 12D observations, task-description encoder and initial task-zero masks in full 5×1024×4 CoTASPLearner; only stage 0 is runnable before sequential learning has created later task masks. The initial diagnostic is stochastic through original `sample_actions` (not falsely called deterministic). Continual World uses 32D observations with 20-element stage-index one-hot; its original 4×256 EWC/PackNet native implementations are evaluated through `on_test_start(index) -> get_action_test() -> on_test_end(index)`. Original PackNet weights, saved variables and ownership state are hashed before and after the **whole** lifecycle and fail closed if not restored, including on exceptions.

### Original task order, pair seeds, environment differences

Both original pinned source CW20 sequences must equal the frozen 10 tasks repeated in position order. Unique task name and stage position are both recorded. Stage j and j+10 are issued identical seed values, **but** their original task-one-hot vectors differ. Each environment is created independently, seeded prior to reset, and the original physics/horizon retained. The original Nevo and Continual World MT50 constructors seed differently, so equal numerical seeds do not prove identical initial states. A repeated seed probe documents any divergence rather than claiming shared goals.

Episode schema includes source SHA, raw reward, success signal, timestep count, stage position, evaluation seed, inference mode, original wrapper, model parameter count, model initialisation time, first-action time, steady-action time, simulator step time, environment reset/create/close time, process-wide peak RSS and zero optimizer updates. A 5-step cap is *diagnostic only*, not a success-rate estimate for full 200-step CW20 benchmarks. Failure on missing success flags, wrong task order, mismatched action dimensions, unverified source or altered post-evaluation state blocks acceptance.

### CPU feasibility and decision gates

There are 20 original stage positions. Original Continual World reference budget of **1,000,000 steps/stage** means 20 million training simulator transitions for one method/seed. With five seeds and three competitors, that is **300 million training transitions**, before evaluation. Simulator interaction time, optimizer-update time and per-model memory must be measured rather than inferred from cold-start GitHub minutes. For original benchmarks, 20-task evaluations at 20 episodes/task and 200 steps/episode have an upper bound of **80,000 extra environment steps per checkpoint/method/seed**. Checking all 20 stages would cost up to 1.6 million additional evaluation steps per method/seed. These are upper bounds, not observed run metrics.

A £0 GitHub CPU smoke qualification proves source portability and determines runtime, not that 300 million transitions plus optimization fit free usage budgets. Future feasibility pilot (e.g. 1000–10000 transitions/task, only after separate training authorization) must report actual learner gradient-update throughput and peak memory. The science-grade run requires consistent task acquisition/retention and whole-pipeline resource costs; no success or efficiency verdict from untrained policies. Authorisation remains **NO-GO for neural training**.

### Records and supersession

The original `evidence/D170_R2R_PUBLIC_REPRODUCIBLE_MANIFEST.json` is an immutable historical snapshot with `new_independent_integrity_gate:PENDING_EXECUTION` and `continualworld.authentic_status:UNTESTED`. The later **success-verified** `evidence/D170_R2R_QUALIFICATION_RESULTS.json` explicitly supersedes those two status assertions and links CI runs 43882825412 and 43883076410. Neither previous file is rewritten or removed. The independent supersession record below stores both Git blob SHA IDs and exact acceptance run IDs, ensuring old claims remain auditable.

### Licensing and integrity

Continual World's pinned upstream root has no confirmed redistribution license. CI checks out upstream source transiently, not into the public repository. Do not publish original files or model weights. No PathLight source, training data, protected results, algorithm details, weights or checkpoints in this repository.
