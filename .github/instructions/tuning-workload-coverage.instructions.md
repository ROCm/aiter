---
applyTo: "**/*.csv,**/*.json,**/*.yaml,**/*.yml,**/*.toml"
---

# Tuning workload coverage

For each newly added workload-specific tuned entry, check that its full workload
key is in the applicable untuned input. Ask for the missing input or a documented
exception. Do not flag unchanged legacy entries or generic defaults.
