# ATOM timeout investigation

Experimental branch: `leo/atom-ci-triage`. No kernel/model changes, cluster
changes, accuracy-threshold changes, or production watchdog changes.

## Replay

```sh
gh workflow run atom-test.yaml --repo ROCm/aiter --ref leo/atom-ci-triage \
  -f triage=true -f triage_model=Kimi-K2.7-Code-MXFP4
```

The replay pins the inputs of failed job `108985135719` from PR #2699:

- AITER: `9bd0ae391992afb7e1c34c1fbd6eef2bc2caae18`
- ATOM, including model configuration: `aa0c5c3a131d98124726b41ab62342bb8cafb207`
- Image: `rocm/atom-dev@sha256:7ab6365ab92d78eea5a517b5ab4ea8fc579f9d88bb753ca8ccb8b2f7c1c60686`
- Runner: `linux-aiter-do-mi350x-4`
- Model: `amd/Kimi-K2.7-Code-MXFP4`, original TP4 arguments and 0.92 threshold

The baseline uses ATOM's original startup and accuracy scripts unchanged.
Resource recording writes a separate file so it cannot reset the watchdog's
log-based progress detector. If the baseline watchdog fails, the experiment
preserves the logs and observes unfinished JIT builds for up to 20 minutes
without changing the server. It only starts a warm-cache comparison after the
original client has exited, avoiding concurrent evaluation clients.

A failed baseline remains a failed GitHub job, even if the continued build or
warm-cache comparison succeeds. Startup failures and non-watchdog failures are
not automatically retried. Diagnostics are uploaded before container cleanup.

## Evidence to collect

- Whether the compiler's cumulative CPU time and Ninja output advance during
  the 18 silent watchdog polls.
- Whether that exact unfinished module finishes after the watchdog exits.
- Whether evaluation succeeds once the module is available, without changing
  model arguments, kernels, or the accuracy threshold.
- CPU affinity/quota, memory pressure and OOM evidence. Runner container
  resources alone do not establish the nested Docker workload's effective
  limits.

The original logs end at JIT compilation, but that alone does not prove a
false-positive hang. The replay run is
https://github.com/ROCm/aiter/actions/runs/36595656162.

## Local checks

```sh
python3 .github/scripts/atom-ci/test_diagnostics.py
actionlint .github/workflows/atom-test.yaml
shellcheck .github/scripts/atom-ci/collect.sh
```
