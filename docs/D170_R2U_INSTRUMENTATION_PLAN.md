# D170-R2U — Future native training instrumentation integration (design only)

**Authorization state:** non-training engineering permitted; any gradient training or model update remains prohibited. All source code in this repository is independent and public-only. No PathLight architecture, source, checkpoints, private benchmark data or proprietary algorithm information is included.

## Instrumented scope

Public original-source pins: Nevo-CRL `fe0456e1ae5382213d12421270209eb3f47a9a0f` and Continual World EWC/PackNet `73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa`. Original physics pin `0875192baaa91c43523708f55866d98eaf3facaf`, MuJoCo 2.0. Keep all original upstream files byte-for-byte unchanged; any eventual instrumentation is a separate external driver around original public call sites, and must be rechecked for scientific observability and no behavioral interference before any training permission.

The `tools/d170_r2u_events.py` schema records one **start** and one **complete** or **interrupted** terminal for every identified operation. Fields include original method and immutable source SHA, execution ID, unique event ID, stage position and task identity, operation category, actor/critic/temperature/encoder/replay/simulator/task-transition scope, Nevo population member, configured iterations, actual invocations, distinct parameter updates, processed batch elements, wall/CPU nanoseconds, observed process RSS peak and device (CPU). Repeated operations generate distinct event IDs and group deterministically. Missing counts remain null. Journal writes are append-only, fsynced at each transition, and validated on reload. An unclosed start after a crash is **not** considered successfully measured, and recovery marks it interrupted rather than inventing zero work. The journal currently represents a single-writer control; future multi-process scheduling needs independent files and a provenance-preserving merge, not unguarded shared writes.

## Future wrappers / authentic call sites (not executed)

- **Nevo-CRL:** instrument native per-stage simulator transitions, agent actor and critic member optimizer operations, population selection and mask mutation, mask reduction, evolution, 40,000-iteration distillation at applicable transitions, evaluation and checkpoint. Log each member separately when source allows; count actual parameter tensor writes separately from high-level `agent.update` invocations. Respect original 1,024×4 actor and 256×4 critic sizes and published enabled options.
- **Continual World EWC:** instrument native SAC environment interactions, actor and twin-critic optimizer activity, temperature and replay behavior, EWC Fisher Jacobian and importance accumulation (original 10 replay batches ×256 at each eligible boundary), evaluation/checkpoint and task-index observation routing. Never misclassify Jacobian work as parameter updates merely because it computes derivatives.
- **Continual World PackNet:** instrument original SAC optimizer, each task-boundary prune operation, its 100,000 post-prune retraining iterations for each applicable boundary, ownership mask changes, original `on_test_start -> get_action_test -> on_test_end` lifecycle, evaluation overhead and model-state restoration. The original temporary saved-weight scratch cache is not a permanent model modification.

The measurement boundary is outside each original call; no changes to optimizer semantics, seeds, replay samples, stochastic ordering, scheduling, gradient graph or weights may be made to obtain instrumented outputs. When original code cannot expose individual member optimizer operations without source edits, mark those metrics **unverified** until an independently reviewed, clearly labeled compatibility instrumentation protocol exists. Never present an unsupported count as measured. Overlapping nested timers must not be summed to infer non-overlapping whole-job CPU time: expose inclusive component times and explicitly measured job total.

## Scientific views frozen

**Published native reproduction:** use each original method's selected official training configuration and task transition mechanism, with original Continual World 20-seed experimental commands (0–19), original native task information and full method-specific costs. Reproduction of all 20 original seeds remains a distinct obligation, and no unmeasured or excluded transition cost is permissible.

**Prospective controlled study:** a separate five-seed comparison (110, 211, 317, 419, 523) under shared MetaWorld task order, physics, evaluation distribution, observation-policy side-information audit and predeclared resource budgets. Those five seeds are **not** a reproduction of the 20 published Continual World seeds. Any change in training budget or interface is an adapted method and must be labeled explicitly; unchanged source files alone do not make a run methodologically original.

## Evidence and limitation

A successful synthetic journal test certifies that the proposed interface detects duplicate counting, preserves completed records across interruptions and rejects false update claims. It does **not** establish feasibility, completeness or zero perturbation when attached to actual actor/critic updates. Actual CPU time, memory and native learning throughput remain unmeasured. Independent review of future native instrumentation, along with explicit training permission, is mandatory.

## Rights and resources

The inspected GitHub API currently identifies no repository license for `yeshenpy/Nevo-CRL`, `awarelab/continual_world` or `ggtz/WAVE` (this is not proof no permissions exist outside GitHub). Public availability does not grant republication/commercial redistribution rights. Preserve original upstream retrieval on disposable CI only, no copied original source. Separate any publicly available model asset licenses from code-license rights. No PathLight exposure, paid compute or protected-release modifications. Every experiment must fail closed on rights uncertainty, unauthorised workload, incompatible physics, missing checkpoints, or model-state corruption.
