# Uncontrolled GPU execution during ticket 01 test development

This is an execution incident, not acceptance evidence. No result from these
runs contributes to ticket completion, numerical acceptance or calibration.

During the red runtime-device test, mocks were attached to the parent tuner
instance. The existing `spawn` supervisor constructed new tuners in its children
and did not inherit those mocks. The test therefore reached real GPU allocation
and kernel execution before its expected preflight error. A second test command
was started while the first command had not yet returned; this also violated the
serialized GPU-task constraint. This was an implementer error.

## Known execution details

Both commands ran in `/root/workspace/tmp_dev/aiter-mxfp4-01`:

```bash
PATH=/root/workspace/tmp_dev/mxfp4-port-env/bin:$PATH \
CK_DIR=/root/workspace/tmp_dev/aiter/3rdparty/composable_kernel \
AITER_USE_SYSTEM_TRITON=1 \
/root/workspace/tmp_dev/mxfp4-port-env/bin/python -m pytest -q \
  op_tests/tuning_tests/test_mxfp4_tuner_lifecycle.py -x
```

| Run | Tool session | Parent PID | Child PIDs | Printed logical assignments | Pytest elapsed |
| --- | --- | --- | --- | --- | --- |
| First | 57578 | 232324 | 232532, 232533 | token4→GPU0, token8→GPU1 | 90.87 s |
| Second | 60489 | 240648 | 244497, 244498 | token4→GPU0, token8→GPU1 | 68.77 s |

The associated resource-tracker PIDs were 232531 and 244496. Both test commands
exited with status 1. A later `kill -TERM` targeting only these known PIDs found
all already exited. The process checks after result retrieval found no owned
`spawn_main`, resource-tracker or lifecycle-pytest processes.

UTC start/end were not recorded. `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`
and `CUDA_VISIBLE_DEVICES` were not supplied in either command and were not
snapshotted at launch. A later environment check found them all unset; this is
not a launch-time identity proof. No physical UUID/PCI mapping or idle sample was
recorded. Physical card identities must therefore be treated as unknown. The
printed logical GPU0/GPU1 assignments do not establish GPU7 use.

The test supplied `SimpleNamespace(mp=2, timeout=0,
mxfp4_search_mode="prune")`. Child output showed real topksoftmax kernel loading,
MXFP4 auxiliary code generation and `_port_e2e` cosine checks. Every candidate
failed when accessing missing `args.errRatio`; profiling/winner acceptance did
not complete. This does not undo the real allocation/kernel work already done.

## Log retention limitation

The commands were initially started without shell redirection. Their output was
retained by the execution-tool sessions and later retrieved into the conversation;
the two returned outputs reported 74826 and 74550 original tokens and were
truncated by the tool. The sessions closed after retrieval. A complete raw log
was not written to disk and cannot be truthfully reconstructed. The available
verbatim excerpts are retained in `incident-console-excerpts.txt`; they are
partial console evidence, not complete raw logs. This evidence limitation is
explicit and must remain visible in the final delivery.

## Correction and isolation audit

The runtime-preflight red test now replaces `_run_shapes_isolated` with a failing
guard before calling `tune`. An autouse fixture guards `_run_shapes_isolated`
and `_prepare_case` for every MXFP4 lifecycle case. Multi-slot tests must supply
their own synchronous controlled supervisor and can never reach a real spawn
supervisor. The module still requires ROCm/FlyDSL imports; those imports are not
described as a GPU-free environment.

All new test launches were stopped when the coordinator requested an isolation
review. Tests already started before that instruction completed and exited.
Further test execution waits for the coordinator's review.
