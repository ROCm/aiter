# gfx950 gluon vs Triton — cross-version, right-sized split-KV decode (3.6.0 / 3.7.0 / 3.8.0)

The **same** gluon kernel runs on all three Triton versions; the only version-dependent difference is
the **KV-load layout path**, chosen by native gating (`ASYNC_COPY_SUPPORTS_DISTRIBUTED`, ported from
`attention/fp8_mqa_logits.py` — it introspects `async_copy.global_load_to_shared`):

| version | KV-load path | why |
|---|---|---|
| 3.6.0 | BlockedLayout | `TRITON_BEYOND_37 = False` |
| 3.7.0 | BlockedLayout | async_copy requires Blocked (introspection → fallback) |
| 3.8.0 (ToT `9cf558d1`) | **distributed offset_bases** | async_copy accepts DistributedLayout (introspection → native) |

Method: 512 MB L2/Infinity-cache flush **every iteration**, torch.profiler, per-kernel-name filter,
30 iters, bf16, HEAD_SIZE=128, causal. Decode = split-KV attention + Triton `reduce_segments`
(identical reduce on both sides). Decode is double-buffered (**nb=2**) on all three — `gl.thread_barrier`
(the single-buffer 32 KB scheme) is still absent on 3.7/3.8, so nb=2 is used uniformly for an
apples-to-apples table. Triton = same-version reference at its own `select_3d_config` split count;
gluon uses the **right-sized** split count described next.

## Split right-sizing (the dominant decode lever)

Triton's `NUM_SEGMENTS_PER_SEQ` heuristic is tuned for Triton's decode occupancy (num_warps=2). The
gluon decode runs **num_warps=1** (BLOCK_M=16, 16×16 MFMA), so each workgroup occupies fewer resources
and **fewer, fatter splits** both fill the GPU and cut the cross-split reduce cost. Right-size to just
enough workgroups to saturate the CUs:

```
NUM_SPLITS = clamp( round( CU*4 / (num_seqs * num_kv_heads) ),  1,  num_tiles )
                          └ ~4 WGs/CU target ┘                 │    └ ≥1 KV tile per split ┘
                                                          no-split floor
```

- **S = 1 → no split, no reduce.** When `num_seqs*num_kv_heads` already ≳ CU*4 there are enough
  workgroups without splitting, so the kernel takes the **non-split path**: it writes the output
  directly (`store_output`) and **launches no reduce kernel at all**. This is the fastest decode mode
  and is why **C128 64/8** (128·8 = 1024 WGs ≈ CU·4) is the biggest win — it pays zero reduce.
  *(Harness note: a bug used to still run the reduce over unwritten partials at S=1 → garbage xcheck
  (2e+05); fixed — S=1 now bypasses the partial buffers and the reduce entirely.)*
- **Cap at num_tiles.** A split must own ≥1 KV tile, so small-ctx + small-batch MQA (e.g. C16/C32 8/1
  at ctx1024, num_tiles = 1024/64 = 16) is split-capped — no headroom, so right-sized ≈ heuristic and
  there is little to gain.
- **Version-independent.** The split count is a host launch parameter, identical across 3.6/3.7/3.8
  (see the shared `S` column below).

## Right-sized gluon vs same-version Triton

gluon GB/s (decode) / TFLOP/s (prefill); parenthetical = speedup vs **same-version** Triton.

| workload | S | Triton 3.6.0 | Triton 3.7.0 | **Triton 3.8.0 (ToT)** |
|---|:--:|--:|--:|--:|
| Prefill b1 8k 64/8       | –  | 852 TF (1.16×) | 823 TF (1.10×) | **959 TF (1.36×)** |
| Decode C64 ctx8192 64/8  | 2  | 6422 (1.13×) | 6515 (1.12×) | 6534 (1.14×) |
| Decode C128 ctx8192 64/8 | 1  | 6611 (1.17×) | 6742 (1.19×) | 6642 (1.17×) |
| Decode C32 ctx8192 8/1   | 32 | 4437 (1.16×) | 4348 (1.13×) | **4601 (1.41×)** |
| Decode C16 ctx8192 8/1   | 64 | 3189 (1.04×) | 3135 (1.00×) | **3360 (1.22×)** |
| Decode C64 ctx1024 64/8  | 2  | 5446 (1.21×) | 5418 (1.20×) | 5546 (1.29×) |
| Decode C128 ctx1024 64/8 | 1  | 6519 (1.41×) | 6533 (1.43×) | **6541 (1.50×)** |
| Decode C32 ctx1024 8/1   | 16 | 1451 (0.97×) | 1465 (0.97×) | 1487 (0.98×) |
| Decode C16 ctx1024 8/1   | 16 | 865 (1.00×)  | 881 (1.01×)  | 895 (1.04×) |

(All shapes 64/8 = GQA, 8/1 = MQA/TP8. xcheck ≤ 5e-4 on every cell, including S=1.)

## Takeaways

- **Right-sizing wins on every version.** GQA C128 → S1 (no split, no reduce) **1.17–1.50×**, C64 → S2
  1.12–1.29×; MQA 1.00–1.41×. The only non-win is C32 ctx1024 8/1 (~0.97–0.98×, split-capped tiny
  batch).
- **gluon absolute perf is ~identical on 3.6.0 and 3.7.0** (both Blocked path). **3.8.0 (distributed
  path) is faster** on the compute-/index-bound shapes: prefill 959 vs ~823–852 TF (**+13–16%**), MQA
  decode +3–7%. GQA decode is BW-saturated (~6.5 TB/s) so it is ~flat across versions.
- **Caveat on the large 3.8.0 MQA speedups (1.22–1.41×):** part of that ratio is a *Triton* regression
  on 3.8.0 for MQA decode (Triton 3267 / 2761 GB/s on 3.8 vs 3820 / 3078 on 3.6), not only gluon
  gains. The fair cross-version metric is gluon's **absolute GB/s** (the numbers, not the ×), which
  rises modestly 3.6 → 3.8.
- Correctness held (xcheck ≤5e-4) on all three versions and all shapes, including the S=1 non-split
  path.
- Remaining item: `gl.thread_barrier` is absent on 3.7/3.8, so the single-buffer 32 KB decode (nb=1,
  2 wg/CU occupancy) needs a shim there; this table is **nb=2** throughout so the three versions are
  directly comparable.

_Env restored to ToT Triton 3.8.0 (editable, `/home/mekaymak/triton`). Cross-version numbers were
taken by directory-swapping `triton.bak.3.6.0` / `triton.bak_3.7.0` in dist-packages (instant renames,
no pip). Revert to a clean 3.6.0 wheel with `bench_gluon_ua/revert_triton_360.sh`; the per-version raw
logs are `bench_gluon_ua/compare_371_<ver>.txt`._
