# A16W8 MXFP8 GEMM (gfx942)

`gemm_a16w8_mxfp8_asm` computes `out = A @ dequant(B, B_scale).T` for decode on gfx942 (MI300X,
MI325X): BF16 activations, OCP MX weights with E4M3 elements and one E8M0 exponent per 32 weights
along K, BF16 output. The weights are dequantized in registers and the products accumulate in fp32;
the activations are never quantized. Unsupported inputs raise instead of falling back.

## Scope

- gfx942 only; other architectures raise.
- `A` [M, K] contiguous BF16 with M from 1 to 64; `out` [M, N] contiguous BF16.
- `B` [N, K] `torch.float8_e4m3fnuz` and `B_scale` [N, K / 32] `uint8`, both from
  `gemm_a16w8_mxfp8_prepare_weight`; N % 16 == 0, K % 32 == 0.
- 32 × 32 block-scale tensors are not read directly; callers pass the per-row expansion.
- The E5M2 MXFP8 variant is not covered.

## APIs

```python
B, B_scale = gemm_a16w8_mxfp8_prepare_weight(weight, weight_scale)  # once, at load
out = gemm_a16w8_mxfp8_asm(A, B, B_scale, out=None, kernelName=None, splitK=None)
cfg = get_gemm_a16w8_mxfp8_config(M, N, K)   # (kernelName, splitK) or None
ok = is_gemm_a16w8_mxfp8_tuned(M, N, K)
```

`prepare_weight` converts an OCP checkpoint to what gfx942 reads: the fp8 conversion reads
e4m3fnuz, whose bits mean half the e4m3fn value and whose 0x80 is NaN. It maps 0x80 (-0) to 0x00
and adds 1 to every exponent; both steps are exact. e4m3fn NaN weights (0x7F, 0xFF) and exponents
of 254 or 255 have no exact equivalent and raise.

## Kernel Selection

The tuned table is `AITER_CONFIGS.AITER_CONFIG_GEMM_A16W8_MXFP8_ASM_FILE` (`aiter/jit/core.py`):
the family's default file `aiter/configs/a16w8_mxfp8_asm_tuned_gemm.csv` (header only) merged with
every `aiter/configs/model_configs/*a16w8_mxfp8_asm_tuned_gemm*.csv`. The rows shipped today are in
`model_configs/dsv41_a16w8_mxfp8_asm_tuned_gemm.csv`: DeepSeek-V4.1-Flash at tensor parallel 4,
7 per-rank (N, K) shapes, M buckets 1, 12, 24, 36, 48 and 64, tuned on MI325X (gfx942, 304 CUs);
`model_configs/dsv41_a16w8_mxfp8_asm_untuned_gemm.csv` lists the shapes they were tuned from.
`AITER_CONFIG_GEMM_A16W8_MXFP8_ASM` replaces the set (one path, or several joined with `:`); a path
that does not exist raises. `AITER_LOG_TUNED_CONFIG=1` logs each lookup.

Rows use AITER's tuned-GEMM columns `gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio`
(`kernelId` is the row of the kernel in the registry csv, `us` the graph-timed kernel time, `bw`
GB/s including the scale bytes). A call uses the entry with the smallest tuned M at or above its
row count for its `(gfx, cu_num, N, K)`. A row is listed only where the kernel beat BF16 hipBLASLt on
the dequantized weights. Engines should gate on `is_gemm_a16w8_mxfp8_tuned`; other shapes run the
default kernel `tr<ceil(M/16)>_tn1_nw4`, which is correct but not tuned. `kernelName` / `splitK`
override the choice.

The tuner `csrc/gemm_a16w8_mxfp8/gemm_a16w8_mxfp8_tune.py` (AITER's `GemmCommonTuner` and
`mp_tuner`) writes these rows from an untuned `M,N,K` list: it times every registered kernel that
admits the shape (split-K kernels at splits 2, 3, 4, 6, 8) and checks each against a float64
reference, times the 8 fastest again in CUDA graphs (as decode runs them) interleaved with BF16 and
with the weights rotated out of the Infinity Cache, and writes the fastest one only when it beats
BF16. The re-tune command is in `csrc/gemm_a16w8_mxfp8/README.md`.

Kernel names encode the tile: `tr` 16-row tiles (M <= 16·tr), `tn` 16-column tiles per workgroup
(N % (16·tn) == 0), `nw` waves splitting K inside the workgroup. `sk` kernels also split K over up
to 8 workgroups; `xl` stages X through LDS; `wd2` / `wd3` prefetch W two or three K steps ahead.

## Split-K Contract

Each split writes an fp32 partial tile and increments a per-tile arrival counter; the last split
sums the partials in split order (deterministic), stores BF16 and resets the counter to 0. The op
keeps one workspace (16 MB) and counter array (256 KiB) per (device, stream); calls sharing them
must be stream-ordered. Non-split kernels get empty buffers.

## Compile And ABI Rules

1. Kernel choice, workspace and launch run inside the custom op `aiter::gemm_a16w8_mxfp8_asm_out`
   (mutates `out`, no-op fake); `gemm_a16w8_mxfp8_asm` is traceable Python around it and works
   under `torch.compile(fullgraph=True)` and in CUDA graphs.
2. Kernel arguments follow AITER's asm layout: one 16-byte slot per argument (pointer + `p2`,
   uint32 + `p3`); the kernels build their buffer descriptors from the pointers and sizes (144
   bytes, 192 with split-K). Rows past M are dropped by the buffer range checks.
3. Keep `Optional[T]` in public, fake and custom-op declarations.

## Validation

Run `python op_tests/test_gemm_a16w8_mxfp8_asm.py`: 7 tuned and 4 untuned shapes × M
1, 2, 7, 12, 16, 17, 24, 36, 48, 64 in `exact` mode (bit-exact against a float64 reference rounded
to BF16, split-K included) and `randn` mode (rtol 1e-2, atol 1e-3), timed against BF16 `torch.mm`
on the dequantized weights; plus guard checks, CUDA-graph replay and `torch.compile` parity. Kernel
changes additionally require validation of every built variant (on the tuned shapes, M 1–64,
with negative controls) and graph-timed benchmarks with the weights rotated out of
cache.

Key implementation locations:

- Python op: `aiter/ops/gemm_op_a16w8_mxfp8.py`
- Host launcher: `csrc/py_itfs_cu/asm_gemm_a16w8_mxfp8.cu`
- Registry and binaries: `hsa/gfx942/a16w8gemm/` (`a16w8gemm_mxfp8.csv`)
- Tuned tables: `aiter/configs/a16w8_mxfp8_asm_{tuned,untuned}_gemm.csv` (defaults, header only),
  `aiter/configs/model_configs/dsv41_a16w8_mxfp8_asm_{tuned,untuned}_gemm.csv` (DeepSeek-V4.1-Flash)
- Tuner: `csrc/gemm_a16w8_mxfp8/gemm_a16w8_mxfp8_tune.py` (`README.md` next to it)
