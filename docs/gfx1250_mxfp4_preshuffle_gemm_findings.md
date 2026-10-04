# gfx1250 MXFP4 preshuffle GEMM: gap to roofline at M=1536, N=7168, K=16384

Kernel: `gemm_afp4wfp4_preshuffle` → Gluon `gemm_mxfp4_preshuffle_gfx1250`, bf16 output, on gfx1250
(256 CUs, wave32, 320 KB LDS per CU).

All times use the tuning harness's method: a CUDA graph over rotating cold copies (1 GB cold
set, 24 calls × 25 replays, median), one GPU per run. Run-to-run noise is about ±1 µs. Every
variant below was checked bit-exact against the existing kernel.

## Results

| M=1536, N=7168, K=16384 | µs | vs. before |
|---|---|---|
| Before: installed config (256×256×256, `NUM_BUFFERS=3`), default scales | 52.2 | — |
| Retuned M≤2048 bucket: 2×4 multicast cluster (512×1024×256, `num_ctas=8`, `NUM_BUFFERS=3`), default scales | 45.8 | −12% |
| New kernel, `scale_layout="gfx1250_tile"`, same config | 41.6 | −20% |

Across the M≤2048 bucket:

| M | old config, default scales | new config, default scales | new config, `gfx1250_tile` |
|---|---|---|---|
| 1056 | 48.6 | 42.4 | 37.8 |
| 1280 | 48.1 | 43.1 | 39.5 |
| 1536 | 51.8 | 45.7 | 41.6 |
| 1792 | 58.7 | 55.0 | 51.0 |
| 2048 | 63.2 | 56.2 | 53.3 |

## What is on this branch

- **Config**: `configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/GEMM-AFP4WFP4_PRESHUFFLED-N=7168-K=16384.json`.
  The only change from the tuned file of #6028/#6069 is the `M_LEQ_2048` bucket (2×4 cluster).
- **Kernel**: `gemm_mxfp4_preshuffle_gfx1250_tiled_scales`, appended to
  `_gluon_kernels/gfx1250/gemm/basic/gemm_mxfp4.py`. The existing kernel is unchanged.
- **Wrapper**: `gemm_afp4wfp4_preshuffle(..., scale_layout="shuffle_scale" | "gfx1250_tile")`.
  The default is unchanged. `_tiled_scales_num_slots` sizes the LDS ring.
- **Test**: `test_gemm_mxfp4_preshuffled_gfx1250_tiled_scales`. It checks that output is
  bit-exact against the default path and within tolerance of the torch reference.

### Dependencies

This branch is cut from main, but the changes were made and measured on top of unmerged work.
It runs once that work lands, so rebase onto it.

- **Multicast support for this kernel** (`cluster_shape`, `num_ctas` in
  `get_gemm_afp4wfp4_preshuffle_layouts`, and the wrapper's `num_ctas` handling). On plain main:
  - `scale_layout="gfx1250_tile"` raises `ImportError` for `cluster_shape`.
  - Main's wrapper ignores `num_ctas`, so the new `M_LEQ_2048` bucket would run as a single
    512×1024 CTA. With 3 buffers that needs over 600 KB of LDS, against 320 KB available.
- **The tuned N=7168-K=16384 file** is itself new to main (#6028/#6069). Expect an add/add
  conflict on rebase: keep that file and re-apply the `M_LEQ_2048` bucket.
- **The tiled dispatch block recomputes `num_ctas` and the cluster shape** only so that it
  applies cleanly to main. After the rebase it can reuse the multicast block's values.

## Using the tiled scale layout

```python
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm


def tile_scales_gfx1250(scales):  # (rows, K // 32) e8m0, un-shuffled
    pad = (-scales.shape[0]) % 32
    if pad:
        scales = torch.cat([scales, scales.new_zeros(pad, scales.shape[1])])
    return shuffle_scale_gemm(
        scales.contiguous(), arch="gfx1250", preshuffle_factor=32, scale_kwidth=8
    )


x_scales_t = x_scales if M < 32 else tile_scales_gfx1250(x_scales)  # M < 32: plain row-major
w_scales_t = tile_scales_gfx1250(w_scales)  # once, offline, like the weight preshuffle
y = gemm_afp4wfp4_preshuffle(
    x_fp4, w_preshuf, x_scales_t, w_scales_t, torch.bfloat16, scale_layout="gfx1250_tile"
)
```

- **Opt-in by design.** Callers pass a different scale format: the same 32×8 tile the gfx1250
  MoE path uses.
- **Activation scales are the catch.** Weight scales can be tiled once, offline. Activation scales
  change every call, so the activation-quant kernel should write this layout directly. Shuffling
  inside the wrapper would cost more than the ~4 µs it saves.
- **Same bits.** Output is bit-identical to the default path: same tiles, same per-tile K order.

## Roofline

- **Work:** 2·M·N·K = 3.61e14 FLOP.
- **Matrix rate:** a register-resident microbenchmark issues one scaled FP4 WMMA (32×16×128) every
  8.25 clocks per SIMD. That is 39 PF/s at 2.4 GHz, so the dense roofline is **9.3 µs**. The
  18 µs target corresponds to about half that rate (20 PF/s).
- **Power limits the clock under load.** Pure WMMA on all 256 CUs settles at about 1.7 GHz, a
  **~13 µs** floor. The power cap is already at its 2500 W maximum.

| CUs running pure WMMA | 1 | 32 | 128 | 256 |
|---|---|---|---|---|
| Clock (GHz) | 2.38 | 2.28 | 1.94–2.03 | 1.66–1.71 |

- **The real kernel slows the same way.** The original kernel ran at 2.27 GHz with 56 CTAs, 2.04 GHz
  with 168 (this shape) and 1.49 GHz with 252.
- **A loop with no global→LDS traffic** (ablation below) delivers 13.9 PF/s at this occupancy,
  about 26 µs. That is roughly where incremental fixes to this design top out. 18 µs needs a
  dataflow that spends less energy per FLOP.

## Where the time goes

### Original kernel (52.2 µs)

Per-CTA cycle and realtime stamps were taken inside the harness's CUDA-graph replay (168 CTAs,
~2.04 GHz). Each layer was isolated by ablating a scratch copy of the kernel. There are 64
K-steps of BK=256 per tile.

| Component | Clocks per K-step | ≈ µs |
|---|---|---|
| WMMA at its ideal rate | 528 | 16.3 |
| LDS operand reads, scale byte gathers, register pressure | +201 | 6.2 |
| TDM loads: issue, LDS writes, compiler-inserted CTA barriers | +285 | 8.8 |
| Cold-cache memory traffic | +330 | 10.2 |
| Prologue 2.4 + epilogue 2.2 + launch gap ~2.5 | — | ~7 |

- **Scale byte gathers.** The default `shuffle_scale` layout forces byte-wise LDS gathers for the
  scales.
- **Idle CUs aren't the first lever.** Only 168 of 256 CUs get a tile. A stream-K proxy (the same
  FLOPs spread over 252 CTAs) gained only 2.5%, because the clock drops as CUs fill.
- **Latency is already hidden.** Time blocked waiting for TDM data is ~5% of the loop. Memory cost
  shows up as throughput, not stalls.

### New kernel (41.6 µs)

Ablations on the shipped kernel and config:

| Variant | µs |
|---|---|
| Full kernel | 41.6 |
| TDM loads kept, always from the same K offset (warm cache) | 35.4 |
| No global→LDS loads in the K loop | 25.9 |
| Scales only | 26.1 |
| A only | 33.5 |
| A + scales | 34.6 |
| B + scales | 38.2 |

- **Moving A and B into LDS costs 15.7 µs:** about 9.5 µs of TDM and LDS-write throughput even from
  a warm cache, plus about 6.2 µs of cold-memory traffic.
- **Synchronization is cheap.** Loading only the scales keeps the same issue/wait/barrier structure
  and costs 0.1 µs, so the cost is the bytes, not the sync.
- **B costs more than A.** In the 2×4 cluster each A block is multicast to 4 CTAs, but each B block
  goes to only 2.
- **Per-CTA phases** (measured before the XCD remap): prologue 2.3 µs, loop 37.2 µs at 2.07 GHz
  (1193 clocks per K-step against 528 ideal), epilogue 3.5 µs.
- **Registers:** 857 of 1024 VGPRs, no spills.

## What worked

1. **2×4 multicast cluster config** (config only): 52.2 → 45.8 µs. The installed config predated
   multicast support, and the new one is faster at every M in the bucket (−6% to −13%). Other
   tile and `num_ctas` choices, measured with the tiled kernel:

   | Tile, `num_ctas` | µs |
   |---|---|
   | 512×1024, 8 | 42.3 |
   | 512×512, 4 | 45.0 |
   | 256×512, 2 | 46.3 |
   | 512×256, 2 | 47.4 |
   | 512×2048, 16 (with K sub-tiling) | 50.2 |
2. **gfx1250-native scale layout** (`gfx1250_tile`). Scales are stored as 32-row stripes of
   row-major [32 rows × 8 K-groups] chunks, which the kernel reads with wide LDS loads instead of
   byte gathers.
   - With the 256×256 config it halves the loop's instruction count (553 → 271).
   - It frees about 90–130 VGPRs.
   - With the 2×4 cluster it goes from 46.2 to 42.3 µs.
3. **XCD-aware tile order.** Each XCD takes a contiguous, column-major run of tiles, so it streams
   whole B panels. This brought 42.3–43.2 µs down to 41.6–41.9 µs.

Design note on slots: the new kernel derives each LDS slot from the K-step, which lets the compiler
prove the slot being written differs from the slots being read. That keeps `NUM_SLOTS - 1` stages
in flight, so it allocates `NUM_BUFFERS + 1` slots to match the original kernel's depth. For
reference, 3 slots measured 52.1 µs and 4 slots 46.2 µs (2×4 cluster, default scales). LDS capacity
caps the slot count.

## What did not help

M=1536, tiled kernel, 2×4 cluster, unless noted.

| Idea | µs | Note |
|---|---|---|
| Direct register store of C (`STORE_MODE=1`, cluster-aware) | 44.4–45.4 vs 41.8–42.3 | 1023 VGPRs. It was neutral on the original kernel (52.3). |
| Double-buffered C | not applicable | Each CTA computes one tile, so no store overlaps a next tile. Revisit once the kernel is persistent. |
| Per-slot mbarriers instead of CTA barriers | 66.0–68.0 (256×256, no cluster) vs 50.1–51.5 | Correct and bit-exact, but 27–68 spills |
| K sub-tiling (slice K in 2) | 44.9 vs 42.3 | 1010 VGPRs |
| `BLOCK_SIZE_K=512` (2 slots) | 64.6–65.4 | |
| 8 warps | 56.1 | 93 spills |
| L2 prefetch (`tdm.prefetch`), 1 or 3 stages ahead | 43.6–43.7 vs 42.3–43.2 | |
| Fused TDM copies split across warps | 42.2–43.2 vs 42.3–43.2 | Neutral |
| Padded B LDS layout (pad interval 256–1024 B) | 41.6–42.0 vs 41.6 | Neutral; a 2048 B interval fails to compile |
| 5 LDS slots | — | Needs 350 KB of LDS; 320 KB is available |
| Row-major un-shuffled scales through TDM (original kernel) | 68.4 | TDM is slow on 8-byte rows |
| Storing straight from the WMMA layout (original kernel) | 59.7 | Uncoalesced |
| Stream-K | −2.5% (proxy) | Power-limited clock; revisit when the loop is leaner |
| Triton's gfx1250 MXFP4 Gluon example kernel | 50.9 (warm cache) | This kernel: 43.5 under the same warm-cache timing |

## Next steps, toward ~25 µs

1. **Load B straight into registers.** This skips one LDS write and read for the operand that
   costs most. It needs register headroom, and the kernel uses 857 of 1024 VGPRs.
2. **Make the kernel persistent.** Stream-K gives 10,752 K-steps ÷ 256 CUs = 42 per CU. Combine it
   with `STORE_MODE=1` and double-buffered C to hide the ~6 µs of prologue and epilogue per tile.
   This pays off once step 1 makes the loop leaner.
3. **Integration.** Emit tiled activation scales from the quant kernel, and teach the tuning harness
   the `gfx1250_tile` path.
4. **Below ~25 µs the kernel is power-bound.** Each % of energy per FLOP saved is roughly % of speed:
   fewer non-WMMA instructions, more multicast reuse, less LDS traffic.

## Validation

Run on gfx1250 on top of the multicast work:

- **New test:** `test_gemm_mxfp4_preshuffled_gfx1250_tiled_scales`, 8/8 pass.
- **Existing suite:** `test_gemm_mxfp4_preshuffled_gfx1250` (excluding the tiled test), 60/60 pass.
- **After the final refactor:** 14/14 pass. Timing was re-checked at 41.2–41.5 µs (tiled) and
  45.1–45.5 µs (default).
- **Pre-existing failure, unrelated:** `test_gemm_afp4_wfp4_preshuffle_splitk` raises
  `KeyError: 'NUM_BUFFERS'`. It forces a Triton split-K config into the gfx1250 Gluon path and
  fails the same way without these changes.

Commands as run, pinned to one GPU and holding that GPU's lock:

```bash
HIP_VISIBLE_DEVICES=0 flock -n -E 75 /tmp/gpu-0.lock python3 -m pytest -q \
  op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py -k test_gemm_mxfp4_preshuffled_gfx1250

# timing of the installed config, from aiter/ops/triton/utils/_triton/tuning/ (cfg.json: [null])
HIP_VISIBLE_DEVICES=0 flock -n -E 75 /tmp/gpu-0.lock python3 harness.py gemm_afp4wfp4_preshuffle \
  --M 1536 --N 7168 --K 16384 --configs cfg.json --out out.jsonl
```

The `gfx1250_tile` timings wrapped a `scale_layout="gfx1250_tile"` call in the same harness helpers,
`make_cold_copies(inputs, 1024)` and `time_with_cuda_graph(call, config, copies, 24, 25)`. These
helpers come from the refactored tuning harness, which is not on main yet.

**Profiling.** Hardware counters weren't usable, because rocprofv3 counter collection aborts on this
stack (`aqlprofile`). Clocks and phase times come from per-CTA `s_get_shader_cycles_u64` and
realtime stamps. The realtime counters differ by up to 30 µs between XCCs, so only stamps from the
same CTA were compared.
