# Persistent GEMM2 buffer-scatter experiment — 2026-11-19

**The original sections below describe commit5e22c9bb4.** Follow-up experiments
retain a faster full-N row writer, still slower than TDM. One run per buffer
variant gives:

| Buffer variant | GEMM2 us | E2E/layer us |
|---|---:|---:|
| Original coalesced compute-wave rows | 729.77 | 1685.21 |
| Precomputed row addresses | 703.63 | 1686.20 |
| Precomputed addresses + four-row grouping | 696.44 | 1668.43 |
| Full-N rows + four-row grouping (retained) | 669.89 | 1650.96 |

The retained Flash writer uses64 buffer_store_b128 instructions, zero tensor
stores, and one exit-only storecnt at `.LBB0_51`, line4619 in its saved ISA.
Rows/addresses use zero descriptor bounds for invalid routes. Cross-wave LDS
reads use the existing raw dscnt + signal/wait barrier. Tests passed at257 and
16384/rank. Raw per-GPU results are in `results_ep_buffer_fullrow4_20261119.json`;
preaddr/group4 results have corresponding JSON files. The optional NT cache
policy passed correctness only and is preserved in `ep_buffer_nt_experiment.patch`.

LLVM branches and raw barriers implicitly drain XCNT. Comparing only explicit
XCNT instruction counts misses these drains; neither count proves a stall cost.
The newer [interleaved TDM comparison](RESULTS_EP_TDM_PERSIST_20261119.md)
selects persistent256 TDM for both GEMM2 and E2E. Do not pool that fresh paired
comparison with the single-run history below.

The buffer-store path satisfies the assembly requirement but is slower. Keep
`AITER_MOE_EP_BUFFER_STORE=0` (default) for the best measured E2E result.

Workload: EP4, 16384 tokens/rank = 65536 global, E384, topk6, H5120, I2304,
a4w4_mxfp4, balanced routing, original gather/preshuffled S1F, P96/C160 with
3:5 affinity. Unique pull, source packing and linear scatter are disabled.
GEMM2 uses M256/N256/K256, 2x2 waves, four buffers and cluster1x4.
All variants use the same CSV; only GEMM2 scheduling/writeback changes.

One successful timing run per variant, five warmups and32 iterations.
Numbers average the four GPU medians. GEMM stages use explicit HIP graph-event
nodes. E2E is timed separately on the uninstrumented eight-layer graph, then
divided by eight; do not add the instrumented stages to reconstruct E2E.
The persistent-TDM run was completed earlier in this experiment and reused.
The original-default and coalesced-buffer runs are fresh. This is a balanced
workload comparison, not a claim about random-routing performance.

| Variant | GEMM1 us | GEMM2 us | Plan us | Ready us | E2E/layer us |
|---|---:|---:|---:|---:|---:|
| Original TDM scatter, default grid | 537.94 | 578.35 | 37.24 | 12.16 | 1553.67 |
| TDM scatter, persistent256 | 541.88 | 553.17 | 40.86 | 13.74 | 1574.99 |
| Buffer rows, persistent256 | 525.08 | 729.77 | 36.65 | 11.91 | 1685.21 |

Buffer rows increase GEMM2 latency by 26.18%
and E2E by 8.47% versus the default.
Against persistent-TDM alone, GEMM2 is 31.92% slower.
Small GEMM1 differences are not attributed to a GEMM1 optimization: its code and
settings did not change. The superseded direct-register buffer variant took
6701.37us for GEMM2 and7899.10us E2E; coalescing removes most of that regression,
but still does not beat TDM scatter.

## Actual assembly

[Audit](ep_buffer_assembly_audit.json), [excerpt](ep_buffer_assembly_excerpt.s),
and [audit tool](audit_ep_buffer_asm.py). The audit parses branch edges and
strongly connected components, not just instruction text order.

- 128 static `buffer_store_b64 ... scope:SCOPE_SYS` instructions; no tensor stores.
- Exactly one `s_wait_storecnt 0`, in exit block `.LBB0_310` at line10100.
- That block is outside all CFG cycles. The store-to-next-task edge goes to
 `.LBB0_2`, whose synchronization is `s_wait_dscnt 0` plus raw
 `s_barrier_signal -1` / `s_barrier_wait -1`.
- 35 static `s_wait_xcnt` instructions remain. LLVM uses X_CNT for XNACK
 address-translation/source-operand protection; it is distinct from STORE_CNT.
 These required waits were retained.

## Why removing the wait is insufficient

The new epilogue stages weighted bf16 results in the existing LDS layout.
Each wave reads only its own M128/N128 region;32 lanes each write four adjacent
bf16 values, so a wave writes256 contiguous bytes per row. Output stores no
longer consume the tensor counter. The task-ending LDS barrier protects input
buffer reuse without draining remote writes; the exit store wait and existing
stream-ordered combine/barrier preserve output completion.

However, scalar row-route broadcast, peer/slot address construction, LDS loads
and128 buffer-store instructions replace bulk TDM scatter. Generated code is
substantially larger, and per-row LDS-read dependencies remain. This is direct
static evidence of extra instruction work, not a cycle-level attribution of the
regression. Removing a counter dependency alone cannot guarantee a speedup;
the new writeback must also issue output efficiently.

## Validation and reproduction

Small257 and full16384 tokens/rank passed six graph replays on four GPUs,
comparing against the existing push/TDM output with `rtol=atol=0.01`. Inputs,
weights and routing change across replays; tests include random/hot experts,
disjoint ownership, empty experts and unequal tails. Metadata and dispatched
payload checks also pass. This is a differential check against the existing
kernel, not a new full fp32 MoE oracle. No GPU reset or timeout occurred.

The final post-test cleanup changes formatting, comments and one unused import
only. The runner records commit, source diff, exact command/environment, CSV,
three review passes, idle check, boot IDs and raw results for each run.

```bash
# Coalesced buffer candidate.
BENCH_EP_BUFFER_STORE=1 BENCH_GEMM2_WORKERS=256 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_buffer_rows_NEW
# Persistent-only control.
BENCH_GEMM2_WORKERS=256 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_tdm_persist_NEW
# Original default.
BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_original_NEW
# Correctness plus actual assembly. Use257 for the small shape.
CHECK_TPR=16384 CHECK_EP_BUFFER=1 BENCH_EP_BUFFER_STORE=1 BENCH_GEMM2_WORKERS=256 BENCH_DUMP_ASM=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_ep_buffer_NEW
```

Raw per-GPU medians, symbols and aggregates are retained in
[results_ep_buffer_20261119.json](results_ep_buffer_20261119.json).
Successful run directories (under this directory):

- `ipflegacy_p96_check_src_ep_buffer_rows_small_02`
- `ipflegacy_p96_check_src_ep_buffer_rows_flash_01`
- `ipflegacy_p96_events_ep_buffer_rows_flash_01`
- `ipflegacy_p96_events_ep_original_flash_01`
- `ipflegacy_p96_events_ep_tdm_persist_flash_01`

The first row-writeback small attempt failed during compilation because of a
mixed legacy/typed select; it is excluded. Large JIT caches and full compiler
dumps remain local rather than being committed.
