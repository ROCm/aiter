# mHC native RMS correction and error rerun

Changed only the mHC output RMS denominator to use FP32 stream-mixture squares before BF16 rounding. The numerator still uses the BF16-rounded mixture, matching ATOM. Original source and results remain unchanged in mhc-optimized and atom-actual-microbench. Corrected source is candidate/tdm_tail.py.

137 inputs (9 checkpoint embeddings and128 deterministic random residuals), positions1024/1536/2047:411 comparisons per round, two rounds. Tables use round2. mHC values repeat across positions. Threshold relative RMS0.001; no tolerance changes.

| Full-region checkpoint | Maximum before | Maximum after | Failures before / after (411 cases) |
|---|---:|---:|---:|
| x | 0.001061097 | 0.000145058 | 3 / 0 |
| post | 0.000000704 | 0.000000704 | 0 / 0 |
| comb | 0.000000149 | 0.000000149 | 0 / 0 |
| q | 0.021466093 | 0.008461183 | 411 / 411 |
| kv | 0.008723231 | 0.001814947 | 301 / 190 |
| attention | 0.009175642 | 0.004848986 | 411 / 411 |

mHC normalized output now passes all137 distinct inputs. It is not bitwise equal. Full mHC-through-attention remains above the native-equivalence threshold; no production acceptance or performance claim. All outputs finite and all mailbox checks pass.

Independent unchanged-QKV rerun reproduces216/222 failures: maxQ0.00927003, KV0.00181375. Independent attention rerun reproduces native/prototype differences (~0.0021). These isolate remaining differences from the mHC correction.

All four GPUs used: GPU0 before, GPU1 after, GPU2 isolatedQKV, GPU3 isolatedattention. No performance measurements were rerun. Corrected resource metadata/ISA saved in after/.

Run check.py before or after inside existing container with one exclusive GPU. /work is the host dsv4-monokernel mount; actual ATOM wrappers remain in sibling atom-actual-microbench. Scripts: check.py, qkv-check/validate.py, attention-check/validate.py. Both rounds and every case retained. Copied candidate folder contains historical optimization logs; only this root report and before/after validation files describe this correction.
