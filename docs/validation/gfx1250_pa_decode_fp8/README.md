# gfx1250 FP8 PA decode evidence

These compact results support the [optimization report](../../gfx1250_pa_decode_fp8_optimization.md).
The baseline was a saved workspace kernel based on
`fae4e8c61ebc7e41d205c6764240e13027cc8160`, including the earlier D128
optimization. It was not a clean checkout of that commit.

- [Bandwidth JSON](bandwidth-summary.json) and [CSV](bandwidth-summary.csv)
  preserve physical read samples, graph latency, and bandwidth calculations.
- Fixed-length paired timings: [B64/64K](final-b64-c65536.json),
  [B128/64K](final-b128-c65536.json), [B64/128K](final-b64-c131072.json), and
  [B128/128K](final-b128-c131072.json).
- Variable-length controls: [D128/log-uniform](varlen-d128-log-uniform.json),
  [D256/log-uniform](varlen-d256-log-uniform.json), and
  [D256/bimodal](varlen-d256-bimodal.json).
- [Counter capture status](counter-status.json) records successful completion
  of all eight physical-read captures.
- [ISA resource comparison](resource-diff.json) records increased VGPR/LDS
  usage and reduced SGPR usage. Its resource regression verdict does not
  measure execution latency.
- ATT instruction hotspots: [baseline](hotspots-final-baseline.txt) and
  [optimized](hotspots-final-optimized.txt). The analyzer incorrectly identifies
  gfx1250 as gfx942. Use instruction stall samples; its occupancy estimates and
  split-wait classifications do not apply to gfx1250.

Large raw traces, source snapshots, experimental variants, and the local
comparison/profiling drivers are excluded. The optimization report provides a
command to benchmark the current kernel using the committed benchmark CLI.
