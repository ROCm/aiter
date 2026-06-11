# i8fp8 ASM FMHA kernel — runbook for matching its perf from UA

**Purpose.** Capture everything needed to *build, run, profile, and read* the
hand-tuned **i8fp8 ASM attention forward kernel** (int8 Q/K, fp8 V, bf16 out) so
the UA (unified-attention CK) work can target the same throughput. This is the
kernel behind the `+60T` Sage attention work; it is a separate codebase from UA
but solves the same prefill attention problem at `hd128`.

Written by a sibling agent that set up the toolchain. Nothing here has been run
to a final TFLOPs number yet — the last setup step (building aiter from the
i8fp8 branch) was in progress when handed off. The "Remaining steps" section is
the short path to a number.

---

## 0. TL;DR status

| Piece | State | Where |
|---|---|---|
| ASM kernel source (pyisa) | ✅ present | `/root/diffusion-models-inference-private` (branch `initial-asm-kernels`), `asm/fmha_sage_fwd/gfx950/mi350_fmha_hd128_i8fp8.py` |
| `pyisa` assembler DSL | ✅ installed | cloned `ROCm/PyISA` → `/root/PyISA`, symlinked to `~/pyisa` (kernels hardcode `~/pyisa`) |
| `.s` → `.co` build | ✅ verified working | emits + assembles cleanly (see Part A) |
| i8fp8 aiter host integration | ✅ located | `/root/aiter` branch **`origin/i8fp8_fmha_gfx950`** (and `..._sparse`) |
| Isolated run env (worktree + venv) | 🟡 in progress | worktree `/root/aiter-i8fp8`, venv `/root/i8fp8-venv` |
| Measured TFLOPs | ❌ not yet | run `bench_sage.py --kernel aiter_i8fp8` once aiter builds |

**The whole point of the isolation:** your active UA checkout at `/root/aiter`
(branch `jukorhon/unified-attention-ck-fav4`) and its editable venv
(`/root/workspace/aiter-venv`) are **left untouched**. The i8fp8 branch differs
from your UA branch by ~676 files and needs a different compiled `fmha_v3_fwd`
op, so it must run from a separate worktree + venv.

---

## 1. Target shape & the TFLOPs formula

The reference shape is the one from `att_std_d128_fp8_noncausal_sq75600_h5`
(your `ua_trace 75600 5 5 128 0 3`):

```
batch=1, hq=hk=5, sq=sk=75600, d=dv=128, non-causal
```

`bench_sage.py` computes (dense, both matmuls):

```
total_flops = 2 * B * hq * Sq * Sk * (d + dv)
            = 2 * 1 * 5 * 75600 * 75600 * (128 + 128)
            ≈ 1.4631e13 FLOP per attention call   (~14.63 TFLOP)

TFLOPS = total_flops / kernel_ms * 1e-9  =  14631.3 / kernel_ms
```

So a kernel time of e.g. 20 ms ⇒ ~732 TFLOPS, 15 ms ⇒ ~975 TFLOPS. Use this to
sanity-check whatever the bench prints, and to compare directly against a UA
`ua_trace`/CK number at the same shape (identical FLOP count).

---

## 2. Asset inventory (absolute paths)

```
ASM kernel sources (pyisa):  /root/diffusion-models-inference-private/asm/fmha_sage_fwd/
  gfx950/mi350_fmha_hd128_i8fp8.py        # the kernel of interest (MI350/CDNA4)
  gfx950/mi350_fmha_hd128_i8fp8_sparse.py # block-sparse variant
  gfx950/mi350_fmha_hd128_{fp8,mxfp4,mxfp6}.py + _sparse
  gfx942/mi300_fmha_hd128_{fp8,i8fp8}.py  # MI300/CDNA3 ports
  tools/build_deploy.sh                   # emit -> assemble -> deploy -> cosine, one shot
  tools/README.md                         # profiling toolchain (rocprof + rcv_* headless tools)
  tools/rollback/validate_rollback_i8fp8.py # canonical i8fp8 run/validate harness (READ THIS)

Raw SP3 bf16 kernels (different family): asm/fmha_v3_fwd/{mi300,mi350}/*.sp3

pyisa:                /root/PyISA  (== ~/pyisa via symlink)
i8fp8 aiter worktree: /root/aiter-i8fp8        (branch i8fp8_run -> origin/i8fp8_fmha_gfx950)
i8fp8 run venv:       /root/i8fp8-venv         (layered on /root/workspace/aiter-venv deps)
```

---

## 3. Part A — Build the kernel: `.py` → `.s` → `.co`

This is the "line on how to generate .s and compile to .co" — it lives in
`asm/fmha_sage_fwd/tools/build_deploy.sh`. Minimal form:

```bash
# pyisa must be importable as ~/pyisa (already symlinked: /root/pyisa -> /root/PyISA)
cd /root/diffusion-models-inference-private

# 1) emit GCN assembly via pyisa
python3 asm/fmha_sage_fwd/gfx950/mi350_fmha_hd128_i8fp8.py /tmp/i8fp8.s \
  --symbol _ZN5aiter28fmha_fwd_hd128_i8fp8_gfx950E

# 2) assemble to an HSA code object
/opt/rocm/llvm/bin/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx950 \
  -x assembler /tmp/i8fp8.s -o /tmp/i8fp8.co
```

Verified resources of the freshly-built `.co` (read with the Triton llvm, since
`/opt/rocm/llvm/bin/llvm-readelf` is absent here):

```bash
/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/llvm-readobj --notes /tmp/i8fp8.co \
  | grep -E "vgpr_count|sgpr_count|group_segment_fixed_size|private_segment_fixed_size|kernarg_segment_size|\.name:"
# symbol _ZN5aiter28fmha_fwd_hd128_i8fp8_gfx950E
# .vgpr_count 224  .sgpr_count 95  LDS(group_segment) 99840 B
# .kernarg_segment_size 656  .private_segment_fixed_size 0  (no spills)
```

> Note: the i8fp8 aiter branch **already ships a prebuilt, tuned**
> `hsa/gfx950/fmha_v3_fwd/fwd_hd128_i8fp8.co` (the `+60T` waves-balance build,
> ~33 KB). You only need to rebuild from source if you want to *modify* the
> kernel (e.g. toggle `_USE_ROLLBACK` / `_USE_EXP2_APPROX` at the top of the
> `.py`) or compare against your own variant. To deploy a rebuilt one:
> `cp /tmp/i8fp8.co /root/aiter-i8fp8/hsa/gfx950/fmha_v3_fwd/fwd_hd128_i8fp8.co`

---

## 4. Part B — How the kernel is dispatched at runtime

The host path is **pure Python over the existing ASM launcher**, plus a small
C++ change:

- `aiter/ops/mha.py::flash_attn_i8fp8_pertensor_func(q,k,v,q_descale,k_descale,v_descale,...)`
  takes **int8** Q/K + **fp8** V + fp32 per-tensor descales, calls
  `_flash_attn_forward`, which calls the compiled op `fmha_v3_fwd`.
- `csrc/py_itfs_cu/asm_mha_fwd.cu::fmha_v3_fwd` was extended to accept
  `int8 q/k + fp8 v` and set `dtype_str = "i8fp8bf16"` (output forced bf16).
  **This C++ delta is why your UA-branch build cannot run i8fp8** — its launcher
  rejects int8 Q/K. aiter is JIT (`PREBUILD_KERNELS` unset), so this recompiles
  the one fmha module on first call from the worktree source.
- Lookup: `hsa/gfx950/fmha_v3_fwd/fmha_fwd.csv` line maps the kernel:
  ```
  i8fp8bf16,128,128,0,0,0,256,128,_ZN5aiter28fmha_fwd_hd128_i8fp8_gfx950E,fwd_hd128_i8fp8.co
  ```
  i.e. dtype `i8fp8bf16`, hd 128, tile 256x128 → symbol + `fwd_hd128_i8fp8.co`.
- `bench_sage.py` (i8fp8 branch) adds: `--kernel aiter_i8fp8`, `i8fp8_quantize()`,
  and `generate_test_tensors()`.

Kernarg layout is the 656-byte `fmha_fwd_v3_args` blob (same as the fp8 kernel).

---

## 5. Part C — Isolated run environment (already set up)

Created so your UA env is untouched:

```bash
# worktree at the i8fp8 branch (does NOT change /root/aiter's checked-out UA tree)
cd /root/aiter
git worktree add -b i8fp8_run /root/aiter-i8fp8 origin/i8fp8_fmha_gfx950

# reuse the existing CK submodule checkout instead of re-cloning (huge)
ln -sfn /root/aiter/3rdparty/composable_kernel \
        /root/aiter-i8fp8/3rdparty/composable_kernel

# a venv layered on the working venv's heavy deps (torch rocm7 + triton 3.7)
/usr/bin/python3.12 -m venv /root/i8fp8-venv
echo "/root/workspace/aiter-venv/lib/python3.12/site-packages" \
  > /root/i8fp8-venv/lib/python3.12/site-packages/_deps.pth
# sanity: torch + triton visible
/root/i8fp8-venv/bin/python -c "import torch,triton;print(torch.__version__,triton.__version__)"
```

### ⚠️ Gotcha that cost time: aiter's setup.py auto-installs triton

`setup.py` runs `.github/scripts/install_triton.sh`, which `pip install`s triton
from `https://pypi.amd.com/triton/...` into a hardcoded `/venv` and **hangs**
(network + wrong target). **Always set `AITER_USE_SYSTEM_TRITON=1`** so it keeps
the already-present triton and skips that hook:

```bash
cd /root/aiter-i8fp8
AITER_USE_SYSTEM_TRITON=1 /root/i8fp8-venv/bin/pip install -e . \
  --no-deps --no-build-isolation
```

(`--no-deps` so it doesn't try to reinstall the custom ROCm torch.) The
worktree's `aiter/jit/` starts empty, so the **first** kernel call JIT-compiles
`module_aiter_core` + the `fmha_v3_fwd` module — expect a few minutes of one-time
compile before the first timing.

---

## 6. Part D — Remaining steps to a TFLOPs number

```bash
cd /root/aiter-i8fp8/op_tests/op_benchmarks/triton

# (a) finish the editable install (Part C, with AITER_USE_SYSTEM_TRITON=1)

# (b) run i8fp8 at the reference shape, non-causal
AITER_USE_SYSTEM_TRITON=1 HIP_VISIBLE_DEVICES=0 \
  /root/i8fp8-venv/bin/python bench_sage.py \
    --kernel aiter_i8fp8 --b 1 --hq 5 --sq 75600 --d 128 --layout bshd
# add --compare-to-ref to also check cosine vs torch reference

# bench prints a throughput(TFLOPS) column; that is the number to compare to UA.
```

Notes:
- The GPUs are **gfx950 (MI350)**, multiple present; pick an idle one via
  `HIP_VISIBLE_DEVICES`. `rocm-smi --showuse` index ≠ `HIP_VISIBLE_DEVICES`
  index — verify utilization before trusting a number.
- For a contention-robust A/B, the diffusion repo's
  `asm/fmha_sage_fwd/tools/bench_ab.sh` does interleaved median-of-N rounds.
- `tools/rollback/validate_rollback_i8fp8.py` is the reference harness; it
  builds baseline vs rollback, deploys to the slot, and prints `ms / TFLOPS /
  cosine` — read it to mirror exact quantization (`i8fp8_quantize`) and timing
  (`triton.testing.do_bench(warmup=25, rep=100)`).

---

## 7. Reading the ASM kernel

- Top ~50 lines of `mi350_fmha_hd128_i8fp8.py` are an architecture banner (math,
  512-thread/8-wave layout with K-loader vs V-loader roles, LDS layout, latency
  hiding, numerics). Worth reading first.
- Tunable toggles at the top of the file: `_USE_EXP2_APPROX` (Schraudolph 2^x
  softmax), `_USE_ROLLBACK` (speculative max-freezing softmax with rollback),
  `CAUSAL_MASK`, `GROUP_MODE`. Setting `_USE_ROLLBACK=False` gives a
  data-independent, byte-stable baseline.
- Body is near-1:1 GCN/CDNA4 instructions via pyisa (`v_mfma_*`,
  `ds_read_b64_tr_b8`, `buffer_load ... lds:1`, `v_readfirstlane_b32`, ...). The
  i8fp8 variant runs the QK matmul in int8 and the PV matmul in fp8; P is packed
  to fp8 e4m3 between them.
- For perf attribution use the headless tools in `asm/fmha_sage_fwd/tools/`
  (`rcv_hotspot.py`, `rcv_timeshare.py`, `rcv_wave_balance.py`,
  `rcv_simd_util.py`, `rpc_mix.py`) on a `rocprofv3 --att` capture — see
  `tools/README.md`. These are directly comparable to your UA ATT analysis under
  `ua-test-scripts/rocprof_analysis/`.

---

## 8. What to compare against UA

Same shape, same FLOP count ⇒ TFLOPS is directly comparable. The i8fp8 ASM
kernel is the perf target for the UA CK prefill at `hd128`. Useful comparisons:
- i8fp8 ASM `aiter_i8fp8` vs your UA `ua_trace` at `b1/h5/sq75600/d128` non-causal.
- The ASM kernel's static budget (224 VGPR, 99,840 B LDS, 2 waves/SIMD target,
  0 spills) vs the UA CK pipeline's occupancy — a likely lever if UA is behind.

## 9. Cleanup (when done, to free the worktree)

```bash
cd /root/aiter
git worktree remove /root/aiter-i8fp8        # add --force if the venv symlink blocks it
git branch -D i8fp8_run
rm -rf /root/i8fp8-venv
# /root/PyISA + the ~/pyisa symlink + /root/diffusion-models-inference-private can stay
```
