DSV4 Pro TP4 hybrid monokernel release
Base: rocm/fw-bringup-mlperf:gfx1250-atom-20261015
Tag: rocm/fw-bringup-mlperf:gfx1250-atom-20261018-mk_tp

Includes the first-round mHC+FP8 quant fusion and small QKV projection, plus
round-two attention projection tuning (multicast / LDS pipeline configurations).
MoE and the baseline LL collective remain unchanged. Experimental prefetch and
semaphore overrides are not installed. Flash experiments are not included.

Launch with four gfx1250 GPUs, /dev/kfd and /dev/dri access, host IPC/network,
and /data mounted read-only containing DeepSeek-V4-Pro. Mount an empty writable
experiment directory at /work; run bash /app/scripts/serve_dsv4_mk_tp4.sh.
In a separate shell in the same container, run:
  bash /app/scripts/bench_dsv4_mk_tp4.sh 1 8 16 32 64 128
Report only results/mk_tp_run2_*.json; run1 is the full client warmup.
Do not share ATOM compilation caches between baseline and hybrid modes.
For baseline use all three ATOM_DSV4_HYBRID{,_GEMM,_ATTN}=0 and a fresh cache.
For the round-one control retain HYBRID=1,GEMM=1 and set ATTN=0.

Reports, source snapshots, kernel averages, metrics and experiment evidence:
  /opt/dsv4-mk-tp/reproduction/dsv4-monokernel/REPORT.txt
  /opt/dsv4-mk-tp/reproduction/dsv4-monokernel/round2/REPORT.txt
The round-two change improved C64 TPOT by 9.22% versus the first-round hybrid.
C128 showed no dependable end-to-end improvement; all repetitions are retained.
Exact real projection checks were performed on all four TP ranks.
These are hybrid regions, not a single fused full-layer kernel.
