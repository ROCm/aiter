# Remaining PR 5989 failures

This experiment is separate from the original PR 2699 replay. `pins.json`
records the image, AITER, ATOM, installed packages and model revisions from
run 36712689029. Model arguments and accuracy thresholds come from that exact
ATOM checkout, not its current main branch.

Run on the experimental branch only:

```sh
gh workflow run atom-test.yaml --repo ROCm/aiter --ref leo/atom-ci-triage \
  -f triage_remaining=true -f triage_model=Kimi-K2.7-Code-MXFP4 \
  -f triage_mode=baseline
```

Modes: `audit` checks the environment without running a model; `baseline`
uses unchanged runtime settings; `affinity` limits compiler workers to the
runner's 20 allocated CPUs; `prebuild` builds the slow CK tile GEMM module
before the unchanged startup deadline; `ipc-copy` tests the existing copy-in
graph path instead of registering captured input pointers. These are
experiments, not confirmed fixes. Each mode starts a fresh runtime container.

The driver also works directly in an equivalent manual runner pod: place the
pinned ATOM checkout at the working directory, the diagnostic scripts beneath
it, and invoke `run.py` with the same arguments. It requires the CI device
assignment file and refuses model execution if nested Docker exposes more
GPUs than the pod was allocated. Do not create pods without the cluster's
normal GPU resource requests, Kueue queue and bin-packing scheduler.

The shared model cache is mounted read-only. Missing or changed revisions
abort the experiment rather than downloading over other jobs' models.
Diagnostics are separate from server/client logs, so they cannot keep the
hang detector alive. Both the 45-minute startup deadline and original
90-minute launch/evaluation step budget remain unchanged. Success requires
all 1319 GSM8K samples and the original accuracy threshold. No test mode
automatically retries a failed run.
