# `packed-fp32-ops` regression on the Gluon fp8 MQA-logits kernel

Reproducer for a Triton codegen regression on `_gluon_fp8_mqa_logits_kernel`, gfx950, glm-shaped (32 query heads, head_size 128), one `1x8192x32768` launch.

Two compilers, from `pip show triton`:

| label | version | source |
|---|---|---|
| **3.7.0** | `3.7.0+amd.rocm7.2.0.git89002410` | AMD release wheel |
| **tot** | `3.8.0+gitb63e34c521` | `triton-lang/triton` @ `b63e34c521dfc55a5f9e419cfeaa64df8263891f`, built from source |

ROCm 7.2.3, torch 2.12.0, python 3.12.13, gfx950 (MI355X, 256 CUs).

The KV loop unroll factor is pinned, so all four arms compile to the same **4-`v_mfma` loop body** and the only variables are the compiler and the flag.

**Disabling `packed-fp32-ops` roughly doubles the loop's VALU count on ToT.** Unpacking a `v_pk_fma_f32` yields two `v_fma_f32`, so removing 17 packed ops must add exactly 17 instructions. 3.7.0 obeys that — it emits 83 against a predicted 86. ToT emits **150**, an excess of **+14.75 VALU per MFMA**, about 59 instructions per loop body that the flag does not ask for. The same flag is a **1.02x win on 3.7.0** and a **0.78x loss on ToT**.

**ToT is also 11% slower than 3.7.0 at default settings**, before any flag is touched. Cause not identified.

## Results

Kernel-only µs from the profiler, median of 3 alternating A/B rounds × 3 reps × 50 iters. Outputs bit-identical between arms in every run.

| triton | `packed-fp32-ops` | µs | TFLOP/s | loop VALU | `v_pk_*` | VGPR | AGPR | spill |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3.7.0 | enabled (default) | 1060.7 | 1814 | 69 | 17 | 167 | 83 | 0 |
| 3.7.0 | **disabled** | **1042.0** | **1847** | 83 | 0 | 132 | 48 | 0 |
| tot | enabled (default) | 1192.1 | 1614 | 74 | 17 | 256 | 128 | 9 |
| tot | **disabled** | 1522.0 | 1264 | **150** | 0 | 256 | 128 | 7 |

| comparison | effect |
|---|---|
| disabling packing, 3.7.0 | **1.018x faster** |
| disabling packing, ToT | **0.783x — 28% slower** |
| 3.7.0 → ToT, both default | **0.890x — 11% slower** |

## The loop body

Longest span closed by a backward branch, identical shape in all four arms.

| | 3.7.0 on | 3.7.0 off | tot on | tot off |
|---|---:|---:|---:|---:|
| loop lines | 232 | 277 | 266 | 426 |
| `v_mfma` | 4 | 4 | 4 | 4 |
| `v_pk_*` | 17 | 0 | 17 | 0 |
| VALU (other) | 52 | 83 | 53 | 145 |
| `v_accvgpr` move | 0 | 0 | 4 | 5 |
| **VALU total** | **69** | **83** | **74** | **150** |
| `buffer_load` | 5 | 5 | 5 | 5 |
| `s_waitcnt` | 4 | 4 | 4 | 2 |

| VALU per `v_mfma` | on | off | predicted from unpacking | **excess** |
|---|---:|---:|---:|---:|
| **3.7.0** | 17.25 | 20.75 | 21.50 | **−0.75** |
| **tot** | 18.50 | 37.50 | 22.75 | **+14.75** |

Both compilers are handed the same loop with 17 of its VALU packed. 3.7.0 unpacks and lands slightly under the arithmetic prediction. ToT emits **2.03x** the VALU against an expected +17 instructions.

`s_waitcnt` *falls* 4 → 2 on ToT while VALU doubles, so the loop is not waiting on memory more — it is executing more arithmetic between the same MFMAs. On gfx950 a packed FP32 op is the one VALU form that can never co-issue in an MFMA shadow, which is why removing them is worth doing at all, and what 3.7.0 delivers.

## Not register allocation

ToT gives the kernel **more** registers than 3.7.0, not fewer — 128 AGPRs against 83 — and is still 11% slower at default with the same excess when packing is disabled. Register pressure and spilling do not track the slowdown; the loop VALU count does.

TTGIR is identical between the two compilers apart from one `#loc` line number, so nothing upstream of the LLVM backend differs. The divergence is between LLIR and ISA.

## Running it

```bash
cd reproducer
python3 repro_fp8_mqa_disable_pk.py --both --rounds 3       # both arms, alternating
python3 repro_fp8_mqa_disable_pk.py --packing off           # one arm
python3 repro_fp8_mqa_disable_pk.py --both --unroll 2       # as aiter ships
python3 repro_fp8_mqa_disable_pk.py --both --label tot      # name the output folder
```

Run it once per compiler; it detects the installed Triton and writes to `<version>_triton/`, or to `<label>_triton/` when the version string does not say which build it is.

```bash
# upstream ToT
git clone https://github.com/triton-lang/triton && cd triton
pip install -r python/requirements.txt && MAX_JOBS=96 pip install --no-build-isolation .
# 3.7.0, from a wheel you already have -- the AMD index no longer serves it
pip install --force-reinstall --no-deps <triton-3.7.0-...whl>
```

Each arm runs in its own process with its own `TRITON_CACHE_DIR`. That is load-bearing: Triton's cache key does not include the target-feature change, so a shared cache returns whichever binary compiled first and the flag appears to do nothing, in both directions.

`--unroll` defaults to 1, not aiter's shipped 2, because at 2 the arms do not share a loop body — on 3.7.0 packing-on gave 8 `v_mfma` per body and packing-off gave 16. At 1 every arm gives 4. The effect is present either way; at `UNROLL=2` the excess is +13.12 VALU/MFMA against +1.00.

## Layout

```
reproducer/
├── repro_fp8_mqa_disable_pk.py     the reproducer
├── no_packed_fp32.py               disables packed-fp32-ops for named kernels
├── force_unroll.py                 pins the KV loop unroll factor
├── 3.7.0_triton/
│   ├── result_{with,without}_packing.json
│   ├── with_packing/     _gluon_fp8_mqa_logits_kernel_{source,ttgir,llir,amdgcn}.txt
│   └── without_packing/  same
└── tot_triton/                     same, plus a `_glir.txt` stage 3.7.0 does not emit
```

The helpers hook `llvm.optimize_module` or the kernel's launch and use Triton's own per-function API, matching on kernel name. Unrelated kernels in the same process are untouched, and no aiter or Triton source is modified.
