# gfx1250 mHC fused post/pre optimization notes

## Scope

This work targets `mhc_fused_post_pre_gemm_sqrsum_kernel` on gfx1250 with
256 CUs, wave32, packed BF16 weights, and shuffled residuals. The primary
decode shapes use `hidden_size` 4096 or 7168 and `M <= 1024`.

The representative correctness and end-to-end performance command is:

```bash
python3 op_tests/test_mhc.py \
    -n 7168 \
    -m 512 \
    --fuse_rmsnorm \
    --w_preshuffle_bf16 \
    --res_shuffle
```

ATT traces can be collected with:

```bash
rocprofv3 -i ../inputs.json -d ./att-logs -- \
    python3 op_tests/test_mhc.py \
        -n 7168 \
        -m 512 \
        --fuse_rmsnorm \
        --w_preshuffle_bf16 \
        --res_shuffle
```

## XCNT analysis

The original direct-store implementation generated a large number of
`s_wait_xcnt` stalls. The main issue was not an overlap with the destination
registers of the FN loads. It was reuse of the VGPRs used as the vector address
of those loads before the corresponding XACK completed.

A representative steady-state sequence was:

```asm
buffer_load_b128 v[2:5],   v14,  ...
...
buffer_load_b128 v[18:21], v212, ...
...
s_wait_xcnt 0
ds_load_b128 v[212:215], ...
```

The FN load uses `v212` as an address operand, and a later LDS load wants to
overwrite the same VGPR range. The compiler inserts `s_wait_xcnt 0` before that
overwrite. A similar dependency appears in the tail, where FN addresses in
`v34/v35` are overwritten by register moves.

Selecting the LDS/TDM path for the affected shapes reduces the measured XCNT
stall cost from 57,557 cycles in the original direct-store trace to 614 cycles:

- Steady state: 8 hits, 100 stall cycles.
- Tail: 4 hits, 514 stall cycles.
- Reduction: approximately 98.9%.

The remaining waits protect real address-register lifetimes and have not been
removed by this change.

## FN register pinning investigation

`pin_reg_report.html` describes the newer `amdgpu_pin_vgpr` attribute. The
clang version in `jun_att2` is AMD clang 23 based on revision
`aa451e1...+PATCHED:440716...`; it reports:

```text
warning: unknown attribute 'amdgpu_pin_vgpr' ignored
```

Consequently, using that attribute would silently produce an unpinned kernel.

Two alternatives were evaluated but not retained:

1. Extending the lifetime of the FN vector-offset VGPRs until tile computation
   completed. This moved the waits rather than removing them and increased VGPR
   allocation from approximately 252 to 256.
2. Splitting the FN address into a loop-invariant VGPR offset plus an SGPR
   offset. The compiler then aggressively reused the SGPR offset and introduced
   additional `s_wait_xcnt 0/1/2/3` instructions.

Old-style scalar fixed-register variables such as `register int x asm("v20")`
work, but wide vector values cannot be allocated reliably this way. Fully
pinning the FN address and data register ranges would require rewriting the
loads register by register, constraining total VGPR allocation, and auditing all
live ranges. It also does not remove the XACK dependency when the pinned address
register is updated for the next iteration. No pinning code is included in the
final change.

## Current pipeline dispatch

For packed BF16 plus shuffled residuals, the selected mid-M interval uses the
LDS/TDM pipeline:

```cpp
const bool use_tdm = (hidden_size == 4096 || hidden_size == 7168)
                     && m >= 512 && m <= 768;
```

The optimized direct-store specialization is selected only when all of the
following are true:

- Packed BF16 weights and shuffled residuals are enabled.
- `WARP_SIZE == 32`.
- The device has 256 CUs.
- `1 <= M <= 1024`.
- `hidden_size` is 4096 or 7168.
- `tile_n == 32` and `tile_k == 32`.
- `tile_m` is 16 or 32.
- `M` is outside the inclusive interval `[512, 768]`.

Other configurations use the existing default dispatch. For the mid-M shapes
above, that means `decode_direct_store=false`, which enables the LDS/TDM path on
gfx1250.

## Existing performance evidence

The following results came from two direct-versus-forced-TDM runs. Positive
numbers mean TDM was faster. Both paths used the same configuration selected by
the existing Python policy, so these results are directional evidence rather
than a completed joint pipeline/configuration tune.

| M | N=4096 | N=7168 |
|---:|---:|---:|
| 1 | -12.8% | -7.8% |
| 32 | -5.8% | -6.1% |
| 64 | -6.0% | -8.6% |
| 96 | -9.3% | -8.1% |
| 128 | -10.3% | -9.8% |
| 256 | -1.6% | +0.1% |
| 384 | -1.3% | +2.8% |
| 512 | +2.8% | +6.6% |
| 640 | -14.2% | +3.7% |
| 768 | +4.6% | +16.3% |
| 1023 | +3.8% | +1.0% |

For M=512, three-run end-to-end medians showed improvement in both epilogue
variants:

| N | Epilogue | Direct | TDM | Latency reduction |
|---:|---|---:|---:|---:|
| 4096 | RMSNorm | 19.058 us | 18.163 us | 4.7% |
| 7168 | RMSNorm | 27.224 us | 23.363 us | 14.2% |
| 4096 | no RMSNorm | 18.923 us | 17.532 us | 7.4% |
| 7168 | no RMSNorm | 26.715 us | 24.331 us | 8.9% |

All corresponding correctness checks passed.

## Configuration coupling and required retuning

Pipeline selection cannot be tuned independently from the configuration
returned by `get_mhc_fused_post_pre_config`. End-to-end cost depends on at least:

- `hidden_size`.
- `split_k`.
- `tile_m` and `tile_k`.
- `ceil(M / tile_m) * split_k`, which controls grid fill.
- `hidden_size / (split_k * tile_k)`, which controls K-loop depth.
- The reduction following the GEMM, whose cost grows with `split_k`.
- RMSNorm versus non-RMSNorm epilogue cost.

The current forced direct/TDM comparison reused one config for both pipelines.
For example, `N=4096, M=640` used `(split_k, tile_m, tile_n, tile_k) =
(64, 32, 32, 32)`. Its TDM regression must not be treated as proof that every
TDM configuration loses at that shape.

When the GPU is available, the `[512, 768]` interval should be retuned jointly:

1. Sweep M continuously, preferably in steps of 32 with boundary probes around
   any transition.
2. Test direct and TDM independently.
3. Sweep `tile_m` in `{16, 32}` and all useful legal `split_k` values.
4. For N=4096 and `tile_k=32`, legal split candidates divide 128.
5. For N=7168 and `tile_k=32`, legal split candidates divide 224.
6. Measure the full fused operation, not only the GEMM/sqrsum kernel.
7. Run both RMSNorm and non-RMSNorm epilogues.
8. Require all correctness checks to pass before accepting a configuration.

The final policy should use ranges derived from stable regions in grid fill and
K-loop depth. The current inclusive `[512, 768]` interval is the requested
interim dispatch policy, not the result of that complete joint sweep.

## Verification status

- `gfx1250` compilation of `module_mhc` succeeds.
- The generated module contains both `decode_direct_store=true` and
  `decode_direct_store=false` specializations for `tile_m` 16 and 32.
- Host disassembly checks `hidden_size` 4096/7168 and implements the inclusive
  M interval with an unsigned range comparison corresponding to 512 through
  768.
- `git diff --check` passes.
- Final runtime benchmarking and new ATT capture are pending because the GPU
  driver is currently stuck in `amdgpu_mes_reg_write_reg_wait`; new `rocminfo`
  processes enter uninterruptible sleep.

## Artifacts

- Benchmark and trace collection:
  `/home/ljin1/jun/jun_claude/artifacts/mhc_opt_20260917/`
- Final compile-only module:
  `/home/ljin1/jun/jun_claude/artifacts/mhc_opt_20260917/precise_dispatch_compile/module_mhc.so`
- Baseline ATT:
  `/home/ljin1/jun/aiter/att-logs-codex-baseline/`
- TDM ATT:
  `/home/ljin1/jun/aiter/att-logs-codex-tdm-m512/`
