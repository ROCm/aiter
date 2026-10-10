# Targeted alternating selected-pair replay

All10 focused tasks passed on the prescribed physicalGPU7, with0 overlaps and60 normal main kernel-duration-sum profiling samples. Each task used one prepared generated input/routes/weights fixture, matched its calibration replay hashes and kept invocation limit/output flags stable. Pair order wasleft/right, right/left, left/right; each pair had3 samples. Every sampled output passed finite Torch cosine_diff≤0.1. No tuning search was added.

| Comparison | Left median μs | Left σ μs | Right median μs | Right σ μs | Right/left difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| minimax-64-default-full-vs-prune | 75.3692 | 0.1047 | 76.0187 | 0.1019 | +0.862% |
| minimax-128-default-prune-vs-staged | 81.5081 | 0.0957 | 82.2694 | 0.0671 | +0.934% |
| minimax-512-default-prune-vs-staged | 96.8353 | 0.0766 | 96.5958 | 0.0706 | -0.247% |
| minimax-2048-default-prune-vs-staged | 174.0129 | 0.3085 | 179.6429 | 0.0494 | +3.235% |
| gpt-2048-default-prune-vs-staged | 117.2948 | 0.0798 | 121.6763 | 0.1012 | +3.735% |
| gpt-2048-fp8-prune-vs-staged | 102.2395 | 0.1746 | 110.4411 | 0.0985 | +8.022% |
| gpt-2048-limit5-prune-vs-staged | 116.4180 | 0.2254 | 121.6371 | 0.1973 | +4.483% |
| gpt-2048-combined5-prune-vs-staged | 105.3016 | 0.2485 | 111.2634 | 0.1350 | +5.662% |
| gpt-2048-fp4-prune-vs-staged | 110.9414 | 0.0602 | 110.8273 | 0.1181 | -0.103% |
| minimax-8192-default-prune-vs-staged | 407.0264 | 0.7047 | 407.4209 | 0.4053 | +0.097% |

Left/right meanings and concrete motivating observations are retained in targeted-replay-plan.json. These medians compare the two selected pairs under a warmed fixed-fixture execution; they are not re-search results or universal optima. CSV search kernel-duration sums and cold tuning wall totals remain unchanged.

Full/prune64 reversed its initial negative subset comparison: the selectedfull winner was0.862% faster in alternating replay, supporting noise/execution-condition sensitivity rather than a pruning benefit. MiniMax128 also reversed its small negative initial difference; MiniMax512 reversed its initial+3.545% staged difference to−0.247%, so its discardedG1 rank4 alone does not establish a stable performance loss.

MiniMax2048 andGPTdefault/FP8/limit5/combined5 retained slower staged-selected pair medians under alternating replay. Their prune winners were outsideK2 atG1 rank5 (MiniMax2048), atomicG2 rank3 (GPTdefault/limit5), and non-atomicG2 rank7 (FP8/combined5). This is concrete shortlist loss for these measured contexts. It does not justify changing the agreedK, fallback or groups automatically; the port reports the measured approximation cost and has no hard regret target.

GPTFP4 repeats the exact same selected pair and measures variation; MiniMax8192 uses retained/validated prune winner alternatives and quantifies near-zero ordering. Samples, min/max/dispersion, error, actual settings, names, hashes and task identities are retained in targeted-summary.json and original task directories. No remaining selected-winner discrepancy requires another tuning search. FinalStandards/Spec review remains required before closing ticket04.
