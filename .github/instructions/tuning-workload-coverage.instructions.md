---
applyTo: "**/*.csv,**/*.json,**/*.yaml,**/*.yml,**/*.toml"
---

# Tuning workload coverage

Where an applicable untuned input exists, check that each newly added
workload-specific tuned entry has its full workload key in that input, or ask
for a documented exception. Do not flag unchanged legacy entries, generic
defaults, or tuning types/formats that have no untuned input.
