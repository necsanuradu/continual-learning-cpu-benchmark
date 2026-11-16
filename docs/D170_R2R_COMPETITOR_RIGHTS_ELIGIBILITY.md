# D170-R2R — authentic public competitor eligibility and licensing audit

## Nevo-CRL

Original `yeshenpy/Nevo-CRL@fe0456e1ae5382213d12421270209eb3f47a9a0f` native full-size CoTASPLearner constructor, task embedding, mask and inference PASS: GitHub run 43794300420. Original MetaWorld MuJoCo2 reset/step PASS: runs 43791220667 and 43792710546. Native learner execution is not sequential learning.

Model encoder originally loads `SentenceTransformer('all-MiniLM-L12-v2')` by mutable alias. D170-R2R adds separate fail-closed verification that the *actual* assets are those from immutable public revision `a50ef00143b4d5391434df20ae11632588ac25be`; until a new CI run passes, this specific asset-integrity requirement is PENDING.

## WAVE: BLOCKED

Original `ggtz/WAVE@7814979cb1f8f4324fc3a48881591b1d0c7d7989` lacks original trained `trained_soms/som_layer_0.npy` through `som_layer_3.npy`. The original SOM recipe's shape compatibility does not establish the existence of learned routing. Do not substitute random weights for authentic trained WAVE. Historical dependencies have incompatible MuJoCo constraints. Status **NO-GO** as a trained authentic comparator.

## Continual World: preferred baseline family, RIGHTS AND EXECUTION PENDING

Original `awarelab/continual_world@73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa` contains TensorFlow native `EWC_SAC` and `PackNet_SAC`, uses original CW20 task order and the same pinned MetaWorld Git revision.

**Rights boundary:** No root LICENSE was identified and GitHub lists no repository license. The README makes source available but does not itself grant redistribution. Do not copy or bundle source into this public repository, do not publish derivative code, and request an explicit license grant from maintainers before redistribution. Public benchmark contains only our independent diagnostic scripts that fetch exact upstream sources at runtime. Research/reproduction without redistribution must still respect applicable law and GitHub terms.

**Scientific distinction:** Original Continual World wraps observations with task-index one-hot vectors, whose total length is 20 for CW20; Nevo uses natural-language hints and pretrained transformer embeddings. Original Continual World `hide_task_id` may apply to the actor backbone but the multi-head selector relies on task ID. A fair benchmark must report this side-information asymmetry; matching environment labels alone is insufficient.

**Eligibility gates:** pinned source check PASS at commit, non-training import/constructor/forward PASS for both EWC and PackNet with actual original `ContinualLearningEnv`, no algorithm modifications, rights clearance, common evaluation protocol, prospective training authority. Pending at document creation.

## CTO decision

**CONDITIONAL GO for benchmark engineering, NO-GO for comparative training now.** Continue native non-training qualification of EWC/PackNet and fail-closed Nevo encoder. No computation-efficiency or retention claims until original sequential learning at equivalent budgets yields actual data.
