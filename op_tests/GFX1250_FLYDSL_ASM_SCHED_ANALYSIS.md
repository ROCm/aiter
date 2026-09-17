# gfx1250 F8 GEMM: native ASM versus FlyDSL

## Scope

This note compares the native kernel from
`origin/gfx1250/bench_asm_f8gemm` at `996b11ddfc25` with the FlyDSL
256x256 kernel for:

```text
M=512, N=65536, K=1536, BF16 output, split-K=1
```

The quoted commands do not initially exercise equivalent scale formats:

- Native ASM is MX32 and receives pre-shuffled A, B, A-scale, and B-scale data.
- `test_gemm_a8w8_blockscale.py --flydsl --ck_preshuffle True --apre True`
  is MX128 and receives logical-layout scales.

That distinction is material to both the instruction stream and the timing.

## Measurements

Measurements were collected on gfx1250 on 2026-09-18. Native values are the
six-call reproduction driver's accepted GPU-event means. FlyDSL values are
from `run_perftest`; the MX32 row also includes a `rocprofv3 --stats
--kernel-trace` recheck.

| Kernel/input | Time (us) | Difference from native constant |
|---|---:|---:|
| Native ASM MX32, constant (16.59, 16.62, 16.82) | 16.6767 | baseline |
| Native ASM MX32, uniform (17.63, 17.51, 17.32) | 17.4867 | n/a |
| FlyDSL MX128, constant | 19.6980 | +3.0213 (+18.1%) |
| FlyDSL MX128, uniform | 22.7493 | n/a |
| FlyDSL MX32/native-format scales, constant | 19.0303 | +2.3536 (+14.1%) |

The FlyDSL MX32 profiler recheck dispatched the target kernel 121 times and
reported 19.5184 us average, 18.027 us minimum. Earlier warm runs measured
18.89-18.92 us by GPU event and 19.2000 us by `rocprofv3` average.

Matching the native scale layout therefore recovers about 0.67-0.81 us, but it
does not explain the remaining roughly 2.2-2.4 us constant-input gap.

The exact FlyDSL control command remained correct for constant input. Uniform
input produced the same accepted BF16 warning as the CK references: 0.1% of
elements differed at the configured tolerance, with maximum absolute delta 20.

## ISA findings

### Scale-format overhead

The MX128 FlyDSL path must broadcast scale bytes before each group of WMMAs. A
representative sequence is:

```text
s_wait_dscnt 18
v_perm_b32
s_wait_dscnt 1
v_perm_b32
s_wait_dscnt 0
v_perm_b32 ...
s_wait_dscnt 8
v_wmma_scale_...
```

The native MX32 kernel consumes packed scale words through WMMA scale selectors
and reaches the first WMMA after a single `s_wait_dscnt 8`.

A scheduling barrier after the MX128 scale loads did not hide this work. It
changed the dependency waits to approximately 34/33/32/8, added another wait,
and regressed event timing from about 19.65 to 19.73 us. Feeding packed MX32
words directly to the MX128 path is not valid: constant inputs produced a
half-scale result and random inputs failed badly. Those experiments are not in
the final source.

### VGPR-bank and wait-count differences

For each 64-WMMA steady-state block, native has a repeatable schedule:

| Kernel | Block | `s_set_vgpr_msb` | `s_wait_dscnt` | `s_wait_alu` |
|---|---:|---:|---:|---:|
| Native MX32 | 0-3 | 10 | 4 | 0 |
| FlyDSL MX32 | 0 | 27 | 6 | 2 |
| FlyDSL MX32 | 1 | 21 | 6 | 5 |
| FlyDSL MX32 | 2 | 52 | 6 | 3 |
| FlyDSL MX32 | 3 | 21 | 6 | 2 |

Block 2 is the largest remaining outlier. Native keeps a stable operand-bank
tuple through long WMMA runs. FlyDSL still changes the A operand bank 11, 2,
12, and 2 times in blocks 0-3 respectively, despite pinning accumulator and
operand groups to reduce allocator freedom.

The first FlyDSL MX32 WMMA is also preceded by a split dependency chain:

```text
s_wait_dscnt 8
s_wait_alu depctr_va_vdst(0)
ds_load_b128 x4
s_set_vgpr_msb
s_wait_dscnt 4
v_wmma_scale_...
```

Native schedules all initial fragment reads before its single
`s_wait_dscnt 8`, then starts WMMA. Aligning this prologue is the clearest next
target. Native also issues three initial tensor loads, waits with
`s_wait_tensorcnt(2)`, then issues the fourth; FlyDSL currently issues four and
waits with `s_wait_tensorcnt(3)`.

## Branch changes

The experimental kernel state in this branch:

- pins accumulator quadrants and A/B fragments to intended physical VGPR banks;
- removes the duplicated runtime parity schedule and retains the parity-0
  traversal;
- introduces explicit LDS scheduling boundaries and native-like DS wait points;
- moves the tensor fence wait earlier to leave independent WMMAs in front of it;
- uses iterative-ILP scheduling with one wave per EU and a 1024-register cap.

The changes are intentionally scoped to the gfx1250 FlyDSL kernel. Failed scale
reinterpretation and scale-barrier experiments were reverted.

## Reproduction

FlyDSL MX128 control:

```bash
ENABLE_CK=0 FLYDSL_COMPILE_LLVM_DIR=/app/llvm-pin-tools \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --ck_preshuffle True --apre True \
  -m 512 -nk 65536,1536
```

GPU-only trace:

```bash
ENABLE_CK=0 FLYDSL_COMPILE_LLVM_DIR=/app/llvm-pin-tools \
rocprofv3 --stats --kernel-trace -f csv -o /tmp/gfx1250_fly_mx128 -- \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --ck_preshuffle True --apre True \
  -m 512 -nk 65536,1536
```

Native reproduction from `origin/gfx1250/bench_asm_f8gemm`:

```bash
python -m op_tests.test_mxfp8fp4gemm_perf \
  --cases wq_b --data-init constant uniform
```
