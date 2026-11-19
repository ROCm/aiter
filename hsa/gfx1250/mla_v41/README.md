# MLA V4.1 H64 fused A8W4

`mla_sparse_prefill_v41(..., backend="asm")` supports H=64, D=512 on gfx1250 B0+.
Q/SWA use FP8 with E8M0 scales, CSA uses NVFP4 with E4M3 scales, and output is BF16.

- Inputs must be contiguous. CSA data `[P,256]` and scales `[P,32]` are separate
  arrays. Convert interleaved 288-byte rows to these arrays before calling the API.
- `softmax_scale` must equal the FP32 value of `1/sqrt(512)`.
  Sink values must be finite or negative infinity. `attn_sink=None` allocates
  and fills a negative-infinity tensor; prepare the tensor before kernel timing.
- `num_kv_splits` accepts 1..16. With `None`, N > 64 uses 1 split. For N <= 64,
  an average of 512 CSA prefix rows uses 1 split; 1024 uses 4 splits at N <= 32
  and 2 otherwise. Other prefix lengths use the ASM fallback heuristic. SWA
  rows do not count toward the prefix average.
- Grid `(S,N,1)`, block `(128,1,1)`; S>1 uses cluster `(S,1,1)` and fused reduction.
  The API allocates FP32 workspaces `[N,S,65,512]` and `[N,S,64]` for S>1.
- Code object `fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41.co`.
- Symbol `fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41`; 120-byte arguments, 327680-byte static LDS.
- CO SHA256: `ab4c3201944a87bd17e57ca8d9674a9e5318ca96ccfbf6f2fde0c14a429f59c4`.

## Build alignment

The instruction image (65,536 bytes), AMDHSA metadata and ABI are unchanged.
The `.text` file offset / virtual address moves from `0x1000` / `0x2000` to
`0x4000` / `0x4000`, with section and executable-segment alignment raised from
4 KiB to 16 KiB. The descriptor's relative entry offset follows the new layout.

In the complete generated assembly, change only the kernel-entry alignment:

```asm
.globl    fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41
.type     fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41,@function
.p2align  14
fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41:
```

Keep descriptor/internal-label alignment and tail padding, including
`.balignl 4096, 0xbf9f0000`. Assemble/link with gfx1250-capable ROCm LLVM:

```bash
clang-24 -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 kernel.s -o kernel.co
llvm-readelf --sections --segments --symbols --wide kernel.co
```

Require `.text` and entry addresses divisible by `0x4000`, and `.text`
`sh_addralign` / executable `PT_LOAD.p_align` at least `0x4000`. Check segment
bounds, `p_offset % p_align == p_vaddr % p_align`, and the unchanged ABI/resources.
The recorded build used LLVM from `rocm-10.2.0a20261013+bkc.20261019`.

## A/B verification

Use separate checkouts of base `5ff09acd601d5b95b1281dccae518ac3bc42750d` and this
PR, the same dependencies, and an otherwise idle gfx1250 GPU. Record
`hipconfig --full`, `rocminfo`, and the loaded HIP/HSA library paths and versions
(or SHA256s, using `/proc/<pid>/maps`). Run from each checkout root:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" AITER_META_DIR="$PWD" \
AITER_JIT_DIR="$PWD/.jit-mla-v41-ab" \
python3 op_tests/test_mla_sparse_prefill.py --backend asm --prec v41_fp4 \
    --h_q 64 -n 96 --total_pages 65536 --topk 512 --window 128 \
    --num_kv_splits auto --seed 0
```

Use three fresh processes per version in each order (base/updated and
updated/base), so every measurement reloads the module. Keep the UT's two
warmups, 101 profiler iterations, first-iteration removal and existing IQR
filter. Keep correctness checks enabled; report CO hashes and median/min/max
of the six profiler means per version. Do not mix profiler and Graph timings.

For controlled placements, use diagnostic assembly copies with `.p2align 18`,
then `.zero 0x22000` or `.zero 0x32000` immediately before the original entry
alignment (`.p2align 8` for base, `.p2align 14` for updated); use the same link
command. Verify the actual HSA kernel entry modulo `0x40000`: base positions
are `0x22000` / `0x32000`, updated positions are `0x24000` / `0x34000`.
Compare instruction bytes starting at the entry; prefix padding is not executed.
These are diagnostic copies, not the shipped CO. Fresh-process CLI runs alone
do not guarantee either controlled placement.

Recorded N96/H64/D512, CSA65536/topk512, causal SWA128, S1 results used gfx1250,
PyTorch `2.11.0+rocm10.1.0a20260811-bkc.20260820` and HIP `7.16.26315`.
Each row summarizes six profiler means across three processes and both orders:

| CO / actual entry modulo 256 KiB | Median (us) | Min?max (us) |
|---|---:|---:|
| Base / `0x22000` | 9.516 | 9.480?9.573 |
| Base / `0x32000` | 9.765 | 9.750?9.837 |
| Updated / `0x24000` | 9.543 | 9.445?9.575 |
| Updated / `0x34000` | 9.499 | 9.455?9.564 |

The 164-configuration study found no stable regression above 1% at the updated
placements. Individual/order-dependent variation remains; this is not a bound
on every sample or every possible entry address.
