# Q-A/KV WMMA assembly optimization, 2026-10-20 UTC
Deadline 01:55 UTC (20:55 America/Chicago October 19).
Scope: same isolated Q-A(4096->1024)+KV(4096->512)+first weighted RMSNorm as qkv-atomic-ab; no Q-B/RoPE/serving claim.
Baseline assembly: four-wave 16x32 tile, BK128, split8; operand conversions stage through LDS with 4 barrier pairs inside each K loop plus final output conversion. Only one output row is useful.
Plan: sweep split-K and N tiles; change operand load layouts to linear equivalents of WMMA dot layouts, avoiding LDS round trips; preserve BF16 operand and output rounding, FP32 accumulate and norm. Test fused completion after compute optimizations.
Ownership: producer(tile,split) exclusively writes Part[split,output]. Separate norm launch orders reads and needs no counters. Fused versions publish full tile sumsq only after K reduction, group-last ticket finalizes norm. Per-invocation initialization/reset included. No concurrent scratch use.
All timings two rounds, second reported, interleaved variants and graph replay; correctness vs FP64 reference and unchanged 0.001 relative RMS gate.
