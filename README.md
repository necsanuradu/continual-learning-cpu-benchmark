# Public continual-learning CPU compatibility checks

This public, non-training diagnostic checks CPU execution of two existing research implementations:

- [Nevo-CRL](https://github.com/yeshenpy/Nevo-CRL), pinned to `fe0456e1ae5382213d12421270209eb3f47a9a0f`
- [WAVE](https://github.com/ggtz/WAVE), pinned to `7814979cb1f8f4324fc3a48881591b1d0c7d7989`

The workflow acquires original public source at these exact revisions and attempts policy initialization plus one forward pass, with original JAX versions adapted to CPU-only `jaxlib` wheels. No model training, optimizer steps, simulator trajectories, private assets, or third-party uploads are performed.

## Qualification

Open [GitHub Actions](https://github.com/necsanuradu/continual-learning-cpu-benchmark/actions) and find **Public competitor CPU qualification**. The workflow runs on its first commit and can be started manually afterward. It uses independent Python 3.8 and 3.10 jobs on standard free Linux CPU runners.

A failed install or import identifies a dependency blocker. A successful original-source MSE policy forward pass establishes only base policy numerical execution; it is **not** evidence that the full Nevo-CRL evolutionary mask procedure or WAVE SOM mechanism, MuJoCo simulator, checkpoint restoration, training, or any head-to-head benchmark has executed. No efficiency advantage can be claimed from this test.

All repository contents and workflow logs are public. Do not commit confidential files. This repository runs only on standard public-repository runners with repository permissions restricted to read during execution.