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

## Confirmed results

The original logs alone were inconclusive. The controlled replay confirmed a
false-positive watchdog failure, not a kernel deadlock:

- Run https://github.com/ROCm/aiter/actions/runs/36595656162 reproduced the
  18-poll hang at 16:53:06 UTC on September 29.
- The unfinished module was `module_gemm_a8w8_bpreshuffle`. Its build completed
  normally in 263.5 seconds, about 70 seconds after the failed accuracy step.
- Compiler CPU time advanced during the quiet interval. In the last sampled
  minute before failure, 45 surviving clang processes accumulated 577.63 CPU
  seconds. The container had 20 CPUs in its affinity mask and no recorded OOM
  or memory-pressure events. Host `os.cpu_count()` reported 192, so the default
  build parallelism was substantially above the allocated CPU affinity.
- The unchanged warm-cache evaluation passed all 1319 GSM8K samples at
  0.941622 against the original 0.92 threshold.

Two fresh-image tests then moved the implicated build into a separate bounded
prebuild step, following the existing AITER vLLM prebuild pattern. The helper
uses the existing AITER build arguments and limits its workers to the available
CPU affinity. It checks that the expected gfx950 binary was produced.

| Test | Run | Accuracy | Threshold | Result |
| --- | --- | --- | --- | --- |
| Kimi-K2.7-Code-MXFP4 | https://github.com/ROCm/aiter/actions/runs/36603509351 | 0.946171 | 0.92 | PASS |
| DeepSeek-V4-Pro | https://github.com/ROCm/aiter/actions/runs/36603618544 | 0.952237 | 0.94 | PASS |

Both used all 1319 samples and the original ATOM watchdog, model arguments,
and accuracy thresholds. The prebuild helper is byte-identical to the helper
on the clean fix branch `leo/atom-ci-prebuild`.

DeepSeek's original job used image digest
`sha256:a88cc66af5c4f255a90de38a0c560cefdb9d67abee331158182d45966540766b`,
different from Kimi's digest despite both using `latest`. Its validation used
that exact digest. An earlier queued DeepSeek run was cancelled and replaced
to correct the image pin before running the validation.

Add `-f triage_prebuild=true` to the replay command to test the mitigation.
The model-to-module mapping also covers the three other original watchdog
failures using these same modules, but those three models were not rerun.
Kimi-K3's startup timeout and the DSpark Group32 assertion remain separate
issues; this change does not claim to fix them.

No kernel/model code, cluster configuration, production branch, PR, or accuracy
threshold was changed. The clean fix branch removes experimental replay pins
and retries and adds prebuilding plus retained failure diagnostics only.

## Compiler logging alternative

The `AITER_LOG_MORE=1` alternative was tested without prebuilding. Both runs
used the same pinned revisions, image digests, model arguments, watchdog, and
accuracy thresholds as above. Before launch, each verified that the implicated
module's `.so` did not exist and that `AITER_LOG_MORE` was `1`.

| Test | Run | Accuracy | Threshold | Cold module build | Result |
| --- | --- | --- | --- | --- | --- |
| Kimi-K2.7-Code-MXFP4 | https://github.com/ROCm/aiter/actions/runs/36618677804 | 0.956785 | 0.92 | 275.7 s | PASS |
| DeepSeek-V4-Pro | https://github.com/ROCm/aiter/actions/runs/36618683679 | 0.944655 | 0.94 | 390.8 s | PASS |

Both evaluated all 1319 GSM8K samples. Replay with `-f triage_log_more=true`
and `-f triage_prebuild=false`; the workflow rejects combining the two modes.

**Do not ship this flag alone.** A separate accelerated-time test used the
unchanged pinned ATOM watchdog and the actual pinned AITER `getLogger()`
formatter, with a stalled client/server and only the shared-memory idle warning
every 60 seconds. Normal logging triggered the 18-poll hang detector; verbose
logging instead reached the full 30-minute timeout. The verbose formatter puts
metadata on a separate line, which survives the watchdog's idle-warning filter
and falsely counts as progress. With no warnings, hang detection still worked;
GPU fault detection also still returned exit 2. This is a simulated regression
test, not a deliberately hung GPU run.

The next minimal alternative should enable compiler output without changing
the global logger format. Neither the prebuild branch nor `main` was changed
by this experiment.

## Local checks

```sh
python3 .github/scripts/atom-ci/test_diagnostics.py
actionlint .github/workflows/atom-test.yaml
shellcheck .github/scripts/atom-ci/collect.sh
```
