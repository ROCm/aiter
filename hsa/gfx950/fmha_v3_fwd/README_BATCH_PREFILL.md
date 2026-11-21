# Static-page batch prefill

`fmha_batch_prefill.csv` describes the page size, layout, dtype, ABI and launch
schedule of each paged ASM kernel. `min_seqlen_q`/`max_seqlen_q` are inclusive;
zero means unbounded. `ts_qo` is the kernel's Q tile size, while
`qtiles_per_workgroup` determines how many such tiles each workgroup processes.
The latter must match the compiled schedule, not simply the attention mask.

For causal FP8-to-BF16 HD256 PS64, dispatch uses:

| Maximum Q length | Q tile | Q tiles/workgroup | Schedule |
|---|---:|---:|---|
| 1–1024 | 64 | 1 | Full grid |
| 1025–4096 | 64 | 2 | Paired causal tiles |
| 4097+ | 256 | 2 | Existing QT256 |

Non-causal and BF16 kernels retain their existing schedules. Unsupported
manifest combinations fall through to CK when CK supports that configuration;
the kernel's compiled page size must match the physical cache page size.

## Causal QT64 code objects updated September 14, 2026

Source: `f8_fmha_prefill_hd256_qt64_kv128_16x16_1buf.sp3` from poc_kl,
source commit `a6e288e` (repository checkpoint `7eeda9dac30a611b0a4ee9a2f64fd6c7631c82d3`).
Source SHA256: `e7a1146a433f2ef27325743db1f3ad1ac0c84fe2dc7fdf302786c747ed7d6eb7`.
The source includes Q/K/V LDS swizzles, overlapping K/V loads, wider output
stores, V-load scattering and factored V addresses. The hybrid exponential
policy is unchanged: exact exponentials for masked/dominant tiles and the
existing approximation for low-share fully valid tiles.

Build: native SP3 `asic=MI350`, followed by `mha_cvt.py` and ROCm 7.2.3 clang
assembly targeting gfx950. Converter SHA256:
`42b6019b852156834ff39c7b80201699be2226bb482e6c717b06f7466daf108d`.
Common constants: `CAUSAL_MASK=1`, `GROUP_MODE=0`, `PAGED=1`, `PAGED_VARLEN=1`,
QT64/KV128, page size 64. `CAUSAL_TILE_PAIR=0` for full-grid and `1` for paired.
Converter resource settings: `LDS_SIZE=65536`, `KERNARG_SIZE=688`,
`KERNEL_VGPR=144`, `KERNEL_ACCVGPR=32`, `WORKGROUP_SIZE=256` (SGPR metadata 96).

| Code object | SHA256 |
|---|---|
| `fwd_hd256_fp8_causal_qt64_fullgrid_paged_varlen.co` | `6b62aa3f4815a65cfddc6de5e76d85ddcb7f2c635c569949111890f1567fbad2` |
| `fwd_hd256_fp8_causal_qt64_paged_varlen.co` | `46cc7ba3fba3dc0333092c1e56991f0895de31b948e1ec57bcab561fdede864a` |

Both production-named code objects have byte-identical `.text` to their
hardware-validated benchmark equivalents. The ABI is the 688-byte
`paged_varlen_v3_ext`; block size is 256, LDS is 64 KiB, and the register
allocation is 144 architectural VGPR plus 32 AGPR (176 total).

## FP8-to-BF16 benchmark

gfx950, ROCm 7.2.3, PyTorch 2.10.0; causal, batch 1, linear SGLang layout.
All implementations were run on the same device. Hq is the query-head count,
Hkv the KV-head count, and HD the head dimension.

| Q=KV | Hq | Hkv | HD | CK PS16 | Previous PR QT64 PS64 | Updated QT64 PS64 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 1 | 256 | 69.82 us | 40.60 us | 32.47 us |
| 2048 | 8 | 1 | 256 | 117.64 us | 59.99 us | 53.42 us |

Median graph-replay time per AITER invocation: seven alternating-order samples,
100 calls per graph, ten ordinary warmups and one graph warmup. Allocation,
reference computation and capture are outside the timed region. Timing includes
auxiliary GPU work submitted by AITER, not just the FMHA dispatch. Ordinary
Python-loop ASM measurements remain around 87–90 us due to submission overhead.
Q/K/V use uniform [-0.5,0.5) data cast to FP8 with unit descales; performance
and crossover points are workload-dependent.

Numerical validation covers partial tiles/pages, shuffled page tables and
randomized multi-request inputs. For the benchmark inputs, max absolute error
was 0.0093163 at 1K and 0.0092374 at 2K, against the float32 reference (threshold
0.06). The tests in `op_tests/test_batch_prefill.py` exercise manifest dispatch
around the 1K schedule boundary and the QT64/QT256 boundary.

Production-path validation on asrock-1w300-h4-3 (gfx950), using a fresh isolated
JIT and no benchmark routing overrides: 32 hardware cases passed, 40 were
intentionally skipped by configuration/range guards. Manifest/code-generation
tests: 8 passed. Loaded-kernel logs confirmed both new causal QT64 schedules
and the existing BF16, non-causal FP8 and QT256 paths.
