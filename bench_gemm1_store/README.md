# gemm1 epilogue store: what does it actually cost?

FlyDSL grouped MoE stage1 (`tile_m=64`, TDM), gfx1250, DSv4 shape.
The gemm1 output store is **0.22% of the kernel's load traffic but ~2.3% of its
time** — roughly 10x its share of bytes. This reproduces that.

## Result

DS shape, `--data-init constant`, `--iters 16`, `--scenario kernel`, 8 rounds
with the two cases alternating. Bandwidth counts **loads only**.

| | us (median, n=8) | TB/s (load-only) |
|---|---:|---:|
| store ON (baseline) | 121.90 | 18.441 |
| store OFF | 119.10 | 18.875 |

Removing the store: **+2.46% mean / +2.27% median, 95% CI +1.92..+2.99%, 8/8 rounds.**

Bytes, for K=7168:

- load `17/16 x (96*6144*7168*0.5 + 512*7168*0.5)` = 2,247,999,488 B
- store = 5,013,504 B = **0.223%** of load. gemm1 runs with
  `stage1_quant_out=1` (kernel name carries `_q1r4`), so the output is
  quantized to **fp4** in the epilogue — 0.5 B/elem, not bf16.
  C tile `3072 x 1536 B` + e8m0 scales `3072 x 96 B`.

## Run it

```bash
N=8 bench_gemm1_store/sweep.sh > store.log
bench_gemm1_store/stats.py store.log
```

One round resolves nothing (per-round sd is ~0.8%); 8 alternating rounds do.

## How the store is removed

`AITER_TDM_NO_STORE=1` forces the epilogue store extent to a **runtime** zero,
so every store falls fully out of bounds and writes nothing.

This is a measurement instrument, not an optimization:

- **It breaks correctness on purpose** — `logits_diff = 1.0`. The trace table
  prints before the sanity assert, which is why the timing is still readable.
- It is **not upstreamable** as-is.
- The instruction stream is unchanged, so the delta is the store's runtime cost
  and not dead-code elimination. Both builds emit identical final ISA:
  `wmma=64`, `tensor_store=1`, `global_store=4`, `s_wait_storecnt=1`; the
  no-store build is 3 lines longer, for the select that produces the zero.

## Not established

Why the store costs ~10x its byte share is **not answered here**. The suspicion
is the epilogue's `s_wait_storecnt(0)` + `tensor_wait(0)` delaying workgroup
retirement — this kernel is not persistent, so a CU cycles ~13.5 workgroups
(3456 WGs over 256 CUs) and pays that tail each time. **This was not verified;**
it needs a ttrace.

Measured on a shared GPU (other KFD clients present), fclk 1950 MHz,
sclk ~2267 MHz. No reset or clock pinning.
