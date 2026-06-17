# FlyDSL port — CK Unified Attention handoff

This is the entry point for porting the CK `unified_attention` (UA) kernel to
FlyDSL. The CK kernel is the **reference implementation**: it is correct
(263/263 in the test matrix) and tuned on gfx950 (MI350/MI355) across fp8 / bf16
/ fp16, prefill + decode, paged + contiguous KV. Match its numerics and contract;
use it to A/B both correctness and performance as you port.

Read this top-to-bottom once, then keep the three linked READMEs open as you work.

---

## 1. Get the exact source

The work spans two repos: `ROCm/aiter` (test harness + tooling) and its pinned
`ROCm/composable_kernel` **submodule** (the kernel). The aiter branch pins CK to
the matching commit, so a **recursive** checkout gets both in sync — never check
out the CK branch by hand.

| repo | branch | role |
|---|---|---|
| `ROCm/aiter` | `jukorhon/unified-attention-ck-fav4` | UA op, test harness, analysis tooling |
| `ROCm/composable_kernel` (submodule) | `jukorhon/fa4-k-preread` | the kernel (auto-selected by the submodule pin) |

```bash
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout jukorhon/unified-attention-ck-fav4
git submodule update --init --recursive 3rdparty/composable_kernel
# sanity: the submodule should be on the pinned UA kernel commit
git -C 3rdparty/composable_kernel log --oneline -1
#   -> ...  CK-UA: freeze docs + comment cleanup (+ gated decode-ring scaffolding)
```

The kernel source lives in
`3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/`, and the
compiled shape/dtype **instances** in
`3rdparty/composable_kernel/example/ck_tile/42_unified_attention/instances/`.

---

## 2. Build & run the reference (`op_tests/test_unified_attention_ck.py`)

The kernel ships as an aiter JIT module. **Critical: the `.so` is NOT rebuilt
automatically when you edit kernel source** — always force a clean rebuild before
trusting any correctness or perf number, or you may be testing a stale binary.

```bash
# Pick an idle GPU (its index = HIP_VISIBLE_DEVICES). python3 = the env aiter is installed in.
rocm-smi --showuse

# --- clean rebuild + stale-proof smoke matrix + regression fixtures (~3 min) ---
# Hard-deletes the .so + build dir, forces AITER_REBUILD=1, and verifies the .so
# was actually regenerated, then runs a prefill/decode/split-KV/long-context gate.
HIP_VISIBLE_DEVICES=2 ua-test-scripts/rebuild_and_test.sh
#   expect: "ALL GREEN"

# --- full correctness matrix (~290 configs; the authoritative gate) ---
# Run AFTER a rebuild (rebuild_and_test.sh, or wipe the .so) so it isn't stale.
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py --full
#   expect: 263/263 PASS (32 expected skips: fp8 with block_size < 32), 0 fail
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py --quick   # smoke subset

# --- one shape vs the torch reference ---
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 2 -sq 8192 -sk 8192 --num-heads 12,2 --head-size 128 \
    --block-size 64 --dtype fp8 --num-blocks auto --no-triton --seed 42
#   -> "CK vs ref: PASS" + max abs delta

# --- one shape, CK vs Triton UA (the perf baseline) ---
HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py \
    -b 1 -sq 75600 -sk 75600 --num-heads 5,5 --head-size 128 \
    --block-size 128 --dtype bf16 --num-blocks auto --mask-type 2 --triton --seed 42
#   -> "UA vs Triton = N.NNx"
```

If you edit kernel source and run the test directly (instead of
`rebuild_and_test.sh`), force the rebuild manually:

```bash
rm -rf aiter/jit/build/module_unified_attention aiter/jit/module_unified_attention.so
AITER_REBUILD=1 HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py ...
```

Useful flags: `--dtype fp8|bf16|fp16`, `--num-heads HQ,HK`, `--block-size` (KV
page size; fp8 needs ≥32), `--mask-type 0` (non-causal) / `2` (causal),
`--contiguous` (non-paged/THD), `--full`, `--quick`, `--no-triton`,
`--no-reference`. For the production-shape sweep vs Triton, see
`ua-test-scripts/sweep_amir_shapes.py`.

---

## 3. Reference docs (read in this order)

1. **Kernel architecture** —
   [`3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/README.md`](../3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/README.md).
   The port spec: file map, per-CTA work assignment (grid → kv_head/seq/q_block/
   split), online-softmax math + host scale fusion (`scale_s = sm_scale ·
   q_descale · k_descale · log2(e)`; deferred `v_descale`), the two pipeline
   regimes (FA4 matrix‖softmax overlap for prefill vs single-warp-group serial
   decode), paged-KV tiers, split-KV, and the tuning-knob / failed-experiment
   table.
2. **Status + how to test/bench** — [`ua-test-scripts/README.md`](README.md).
   Current correctness/perf standing, all run commands, and the fresh-build rule.
3. **Deep-dive tooling** — [`ua-test-scripts/analysis/README.md`](analysis/README.md).
   When you need to understand *why* something is fast/slow: the JIT-free
   standalone driver (stamped builds — no staleness ambiguity), the headless
   attviewer overlap timeline, VGPR/ISA inspection, and rocprof phase profiling.

---

## 4. What to know going in

- **One operator, two regimes.** Prefill uses the FA4 2-warp-group matrix‖softmax
  overlap; decode uses a single-warp-group serial deferred-PV pipeline. Both run
  the *same* deferred-PV math (`alu1(prev) → PV(prev) → QK(cur) → alu0(cur) →
  D_upd`); they differ only in how the work is scheduled across warps.
- **Numerics contract** mirrors Triton's `unified_attention`: base-2 online
  softmax (`exp2`), fp8 Q/K descales folded into `scale_s`, fp8 V descale deferred
  to the post-loop `o_acc · v_descale / l`. GQA ratio is a runtime value.
- **Paged KV** is the perf-sensitive part: constexpr page-size instances
  (`ps16/32/64/128`), single-page SRD rebase, and a Tier-2 LDS-resident
  page-table cache keep page-index resolution off the critical path. Start the
  port from the contiguous (`kIsPaged == false`) path — all paging math compiles
  out there — then add paging.
- **Known standing:** prefill wins ~1.3× over Triton; long-context decode is
  HBM-bandwidth bound (~0.92× vs Triton) and is the open optimization item — see
  `ua-test-scripts/decode_pipeline_*.md`. There is no correctness gap to chase.
- **Validate every step** against the CK reference at matching shapes: full matrix
  (263/263) for correctness, `sweep_amir_shapes.py` for the perf bands.
