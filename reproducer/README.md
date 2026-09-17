# `packed-fp32-ops` on the Gluon fp8 MQA-logits kernel, Triton 3.7.0 vs 3.8.0

Self-contained reproducer for two regressions on `_gluon_fp8_mqa_logits_kernel`
(gfx950, glm-shaped: 32 query heads, head_size 128, one `1x8192x32768` launch).

**1. Disabling `packed-fp32-ops` roughly doubles the loop's VALU count on
3.8.0, far beyond what unpacking accounts for.** Unpacking a `v_pk_fma_f32`
into two scalar ops costs exactly one extra instruction, so the arithmetic is
fixed: 3.7.0 lands **+1.00 VALU per MFMA** past that prediction, 3.8.0 lands
**+13.12** — a 62% overshoot, about 105 instructions per loop body that nothing
about the flag asks for. The same flag is a **1.03x win on 3.7.0** and a
**0.78x loss on 3.8.0**.

**2. 3.8.0 is 9% slower than 3.7.0 on this kernel at default settings**, before
any flag is touched. Cause not identified — see *What this is not* below, which
rules out the obvious candidate with data.

## Results

`1x8192x32768`, 32 heads, head_size 128, `clean_logits=True`. Kernel-only µs
from the profiler, median of 3 alternating A/B rounds × 3 reps × 50 iters.
Outputs verified bit-identical between arms in every run.

| triton | `packed-fp32-ops` | µs | TFLOP/s | `v_pk_*` | VGPR | AGPR | spill |
|---|---|---:|---:|---:|---:|---:|---:|
| 3.7.0 | enabled (default) | 1034.6 | 1860 | 84 | 168 | 84 | 0 |
| 3.7.0 | **disabled** | **1007.8** | **1909** | 0 | 138 | 54 | 0 |
| 3.8.0 | enabled (default) | 1127.4 | 1707 | 90 | 256 | 0 | 5 |
| 3.8.0 | **disabled** | 1454.8 | 1323 | 0 | 256 | 0 | 8 |

| comparison | effect |
|---|---|
| disabling packing, on 3.7.0 | **1.027x faster** |
| disabling packing, on 3.8.0 | **0.775x — 29% slower** |
| 3.7.0 → 3.8.0, both at default | **0.918x — 9% slower** |
| 3.7.0 best vs 3.8.0 best | **1.119x in 3.7.0's favour** |

### Steady-state loop body — where the regression is visible

Counted on the longest span closed by a backward branch. The arms can land on
different unroll factors (3.7.0 unrolls the unpacked loop 2x; 3.8.0 does not),
so raw counts are not comparable and the per-MFMA view is the one to read.

| | 3.7.0 on | 3.7.0 off | 3.8.0 on | 3.8.0 off |
|---|---:|---:|---:|---:|
| loop lines | 468 | 1110 | 495 | 758 |
| `v_mfma` | 8 | 16 | 8 | 8 |
| `v_pk_*` | 34 | 0 | 34 | 0 |
| VALU (other) | 100 | 352 | 101 | 274 |
| **VALU total** | **134** | **352** | **135** | **274** |
| `buffer_load` | 10 | 12 | 10 | 10 |
| `s_waitcnt` | 7 | 15 | 7 | 5 |

| VALU per `v_mfma` | packing on | packing off | predicted from unpacking | **excess** |
|---|---:|---:|---:|---:|
| **3.7.0** | 16.75 | 22.00 | 21.00 | **+1.00** |
| **3.8.0** | 16.88 | **34.25** | 21.12 | **+13.12** |

The prediction is not a model, it is arithmetic: a `v_pk_fma_f32` does two lanes
of work, scalarising it yields two `v_fma_f32`, so the loop should gain exactly
one instruction per packed op removed. 3.7.0 lands within 5% of that. 3.8.0
emits **2.03x** the VALU for the same 8 MFMAs, of which only a quarter is the
unpacking itself.

Note `s_waitcnt` *falls* 7 → 5 on 3.8.0 while VALU doubles. The loop is not
waiting on memory more; it is executing far more arithmetic between the same
MFMAs, and on gfx950 a packed FP32 op is the one VALU form that can never
co-issue in an MFMA shadow — so this arithmetic competes for issue slots the
shadow was supposed to absorb.

Per-round times were tight enough to rule out noise (3.8.0 default:
1135.5 / 1127.4 / 1126.9; 3.8.0 disabled: 1454.8 / 1456.1 / 1450.0), which
matters because this machine is shared — see *Caveats*.

## What this is *not*: the AGPR attribute

3.8.0 emits a function attribute that 3.7.0 does not, and it is tempting to
stop there. Its own source says what it is:

```python
# Workaround, remove once the LLVM fix lands
# With this set, waves_per_eu >= 2 uses no AGPRs; waves_per_eu = 1 stills gets 256.
kernel_fn.add_fn_attr("amdgpu-agpr-alloc", "0")
```

This kernel compiles at `waves_per_eu=2`, so it gets **zero AGPRs**, sits at the
256-VGPR ceiling, and spills — against 84 AGPRs / 168 VGPRs / no spills on
3.7.0.

**It does not explain either regression.** `force_agpr.py` overrides the
attribute (the backend applies `options.llvm_fn_attrs` *after* that line, with
`remove_fn_attr` first, so it is a genuine override):

| `amdgpu-agpr-alloc` | VGPR | AGPR | spill | µs, packing on | µs, packing off |
|---|---:|---:|---:|---:|---:|
| `"0"` (3.8.0 default) | 256 | 0 | 5 | 1133.5 | 1459.1 |
| absent (as 3.7.0) | 256 | 128 | 36 | — | — |
| `"0,256"` | 256 | 0 | 5 | 1137.5 | — |
| `"64,256"` | 256 | 64 | **0** | 1135.6 | — |
| `"96,256"` | 256 | 96 | **0** | 1130.6 | 1466.7 |

Restoring AGPRs and eliminating **every** spill moves the time by 0.3% — inside
noise — and leaves the packing inversion exactly where it was (1459 → 1467 µs).
The loop VALU count is 135 / 274 in every one of these rows regardless of AGPR
state.

So the spills are a side effect, not the mechanism. The quantity that tracks
the slowdown is the VALU count, and that is the thing to explain. Worth noting
separately that `min=0` is a permission, not a request — only a non-zero
minimum (`"64,256"`) actually gets AGPRs allocated.

## Why `packed-fp32-ops` matters at all

It is the LLVM AMDGPU feature bit gating selection of `v_pk_fma_f32` /
`v_pk_mul_f32` / `v_pk_add_f32` / `v_pk_mov_b32` (on by default for gfx90a,
gfx942, gfx950). LLVM's SLP vectorizer pairs this kernel's independent per-row
FP32 reduction chains into `v_pk_fma_f32`. Turning the feature off leaves ISel
nothing to select and makes the SLP cost model see no cheap `<2 x float>` op, so
the pairing never forms. It also relaxes register allocation, because packed ops
need even-aligned register pairs — visible on 3.7.0 as VGPRs 168 → 138.

On gfx950 a packed FP32 op cannot be hidden in an MFMA shadow, so removing them
is the right call for work meant to be covered. That is what 3.7.0 delivers.

## Running it

```bash
cd reproducer
python3 repro_fp8_mqa_disable_pk.py --both --rounds 3   # both arms, alternating
python3 repro_fp8_mqa_disable_pk.py --packing off       # one arm
python3 repro_fp8_mqa_disable_pk.py --packing on --agpr # + undo agpr-alloc=0
```

The script detects the installed Triton and writes to `<version>_triton/`, so
run it once per version with nothing to edit in between. Swapping versions:

```bash
pip install --extra-index-url \
  "https://pypi.amd.com/triton/_release/rocm-7.2.0/simple/" "triton==3.8.0"
```

That index no longer serves 3.7.0, so 3.7.0 has to come from a wheel you
already have (`pip install --force-reinstall --no-deps <wheel>`).

**Each arm runs in its own process with its own `TRITON_CACHE_DIR`**, and that
is load-bearing rather than tidy: Triton's cache key does not include the
target-feature change, so a shared cache hands back whichever binary compiled
first and the flag appears to do nothing — in both directions.

## Layout

```
reproducer/
├── README.md
├── repro_fp8_mqa_disable_pk.py     the reproducer
├── no_packed_fp32.py              disables packed-fp32-ops for named kernels
├── force_agpr.py                  undoes 3.8.0's amdgpu-agpr-alloc="0"
├── 3.7.0_triton/
│   ├── result_{with,without}_packing.json
│   ├── with_packing/     _gluon_fp8_mqa_logits_kernel_{source,ttgir,llir,amdgcn}.txt
│   └── without_packing/  same
└── 3.8.0_triton/         same, plus with_packing_agpr/
```

Both helpers wrap `llvm.optimize_module` and use Triton's own per-function API
(`add_fn_target_feature` / `add_fn_attr`, the same calls the AMD backend makes),
so they append rather than replacing the whole feature string, and they match on
kernel name — unrelated kernels in the same process are untouched. No aiter or
Triton source is modified.

## Where to look in the IRs

TTGIR is identical between the two versions apart from one `#loc` line number,
so nothing upstream of the LLVM backend differs:

```bash
diff 3.7.0_triton/with_packing/*_ttgir.txt 3.8.0_triton/with_packing/*_ttgir.txt
grep -E "\.vgpr_count|\.agpr_count|\.vgpr_spill_count" \
     */*/_gluon_fp8_mqa_logits_kernel_amdgcn.txt
grep -o 'amdgpu-agpr-alloc"="[^"]*"' */*/_gluon_fp8_mqa_logits_kernel_llir.txt
```

The divergence is therefore inside the backend, between LLIR and ISA.

## Caveats

- **Shared machine.** A vLLM server holds memory on all eight GPUs of this node
  and its load varies; identical shapes have measured 30% apart here within half
  an hour. The A/B alternates arms across rounds so drift lands on both, and the
  per-round spread is far under the effect size — but absolute µs should be
  re-taken on a quiet box before being quoted.
- `3.8.0` is `3.8.0+amd.rocm7.2.0.git111ff227`, the wheel aiter CI pins — **not**
  a from-source top-of-tree build. `3.7.0` is `3.7.0+amd.rocm7.2.0.git89002410`.
- ROCm 7.2.3, torch 2.12.0, gfx950 (MI355X), 256 CUs.
