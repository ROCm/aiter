---
applyTo: "**/*.csv,**/*.json,**/*.yaml,**/*.yml,**/*.toml"
---

# Tuning workload coverage

For paired tuning CSVs, check both directions on newly changed workloads:
1. Newly added workload-specific tuned entries must have their full workload key
   in the applicable untuned input (or ask for it when it is known to be
   required but missing).
2. Newly added untuned workloads must have a matching tuned entry in an
   associated tuned file, and tuned workloads must not be removed while still
   present in the untuned input (unless retained in another associated tuned
   file, including runtime-merged model CSVs, or moved).

Do not flag unchanged legacy entries, generic defaults, unknown pairings, or
ambiguous/unsupported tuning types and formats.
