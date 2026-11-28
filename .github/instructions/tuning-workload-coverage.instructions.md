---
applyTo: "**/*tuned*.csv"
---

# Tuned/untuned rows move together

When a PR adds tuned rows for some workloads, the paired untuned CSV should get
the same input rows; when it removes rows on one side, it should remove them on
the other. Comment only on rows the PR itself adds or removes. A row may
legitimately live in another tuned file the runtime merges, or in a shared
untuned input, so check before asking. Do not flag untouched legacy rows.
