---
applyTo: "aiter/configs/**"
---

# AITER shared tuning configs

In version-controlled shared tuning configs, do not add or retune entries to
select hipBLASLt (`libtype=hipblaslt`). Flag those changes, not unchanged
entries. Use a valid non-hipBLASLt candidate, or leave the workload untuned if
its existing fallback supports it.

The saved `solidx` is build-specific: in another build, it can select the wrong
kernel and give wrong results, even with `kernelName` saved. Runtime hipBLASLt
use, such as dynamic fallback selection, is unaffected.
