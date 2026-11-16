# D170-R2S — Common CW20 evaluation harness validation report

## Independent executable release

**Full workflow run [43890292105](https://github.com/necsanuradu/continual-learning-cpu-benchmark/actions/runs/43890292105) — SUCCESS (four of four jobs).** Evaluator commit `c12bddc12945c425530767d846410cbf494225a8`.

| Tested component | Status | GitHub job | Artifact |
|---|---|---|---|
| Python stdlib synthetic metrics, episode and lifecycle suite (7/7) | PASS | 133379331746 | Console receipts |
| Original EWC_SAC native evaluation and live-state restoration | PASS | 133379332182 | cw20-ewc-nontraining-episodes |
| Original PackNet_SAC native evaluation and live-state restoration | PASS | 133379332383 | cw20-packnet-nontraining-episodes |
| Original full-size Nevo CoTASPLearner native evaluation | PASS | 133379332478 | cw20-nevo-nontraining-episodes |

Machine-readable official results: `evidence/D170_R2S_VALIDATION_RESULTS.json`. Full per-episode rows with source/seed/task, raw reward, success, length, init time, inference time, simulator time, memory, parameters and zero-updates fields are held in the GitHub Actions artifacts associated with the run. Artifacts were uploaded successfully.

## Coverage and scientific limits

This was a **five-step, untrained diagnostic**, not a complete CW20 evaluation or fair performance comparison. Native EWC and PackNet executed **four episodes each**, two for stage index 0 and two for stage index 10 (both `hammer-v1`, distinguished by 20-index original one-hot task position). Nevo executed **two episodes at stage index 0 only**, where its original initial mask exists. Running Nevo at stage 10 without sequential training and task mask initialization would manufacture a task history; it is deliberately excluded.

Native CWorld inference uses original `on_test_start -> get_action_test -> on_test_end`; original PackNet's intended scratch `saved_variables` is excluded from *live-model-state* equality while all live network parameters, ownership masks and active-view flags are hashed across evaluation. This controls the earlier false-positive restoration failure; no original algorithm is edited.

The original MetaWorld raw `info['success']` was present. Every reported short diagnostic episode had success zero; these values have no acquisition or retention interpretation.

## Reproducibility

The identical-seed replay probe executed under each original model and produced **exactly matching** reward, length and success for the tested short episode; it does not certify complete trajectory identity across competitor APIs, longer sessions or sequential training. Evaluation seeds, names and stage indices are included in row records and console receipts. Task order and invalid inputs fail closed in the portable tests.

## CPU feasibility, restrictions and acceptance

The executable and machine-readable common evaluator is now qualified for benchmark engineering; no neural training was run. A full 20-stage × 1,000,000-transition × five-seed × three-method study implies 300 million training simulator transitions **before** evaluation and optimization overhead, and the free CPU feasible pilot still needs measured optimizer throughput and authorization. No comparative savings claim can be derived from these smoke tests.

Continual World's original repository lacks confirmed redistribution permission. Its sources are fetched only onto ephemeral runners at exact pinned SHA, and never copied into the public benchmark repository. The original Nevo encoder remains immutable and fail-closed according to D170-R2R.

**Engineer A disposition: D170-R2S common evaluator engineering PASS; CONDITIONAL GO for further non-training benchmark preparation; NO-GO for sequential comparative training until founder authorization, baseline rights, full fidelity, task-side-information comparability and empirical CPU cost estimates are satisfied.**
