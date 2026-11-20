---
applyTo: "aiter/configs/**"
---

# AITER shared tuning configs

Do not add or retune an entry to select hipBLASLt (`libtype=hipblaslt`) in a
version-controlled shared tuning CSV or config. In PR review, flag these
changes, not unchanged legacy rows. Use a valid non-hipBLASLt candidate or
leave the workload untuned if the existing fallback supports it.

AITER saves the hipBLASLt solution index (`solidx`) and uses it to select the
kernel. Across builds or releases, the same index can select the wrong kernel
and give wrong results. Saving the kernel name (`kernelName`) does not make the
index portable. This rule does not restrict runtime hipBLASLt use, including
dynamic selection in an existing fallback.
