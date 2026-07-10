# gfx950 gluon: 16x16 vs 32x32 MFMA

- bf16, HEAD_SIZE=128, causal; Triton = reference. speedup vs triton in ().
- profiler 30 iters, 512MB flush.

## Prefill (TFLOP/s)

| shape | variant | us | TFLOP/s | xcheck |
|---|---|--:|--:|--:|
| b1 8192/8192 64/8 | triton 2d | 1534.0 | 717 | 0.0e+00 |
| b1 8192/8192 64/8 | gluon 32x32 BM128/nw4 (1.19x) | 1292.1 | 851 | 7.8e-03 |
| b1 8192/8192 64/8 | gluon 16x16 BM64/nw4 (0.56x) | 2744.7 | 401 | 7.8e-03 |
| b1 8192/8192 64/8 | gluon 16x16 BM128/nw8 (0.54x) | 2826.3 | 389 | 7.8e-03 |
| b8 1024/1024 8/1 | triton 2d | 50.8 | 339 | 0.0e+00 |
| b8 1024/1024 8/1 | gluon 32x32 BM128/nw4 (1.06x) | 47.9 | 359 | 7.8e-03 |
| b8 1024/1024 8/1 | gluon 16x16 BM64/nw4 (0.69x) | 73.6 | 234 | 7.8e-03 |
| b8 1024/1024 8/1 | gluon 16x16 BM128/nw8 (0.56x) | 90.0 | 191 | 7.8e-03 |

## Decode (attn+reduce, GB/s)

| shape | S | variant | us | GB/s | xcheck |
|---|--:|---|--:|--:|--:|
| C16 ctx1024 64/8 | 16 | triton 3d+red | 20.9 | 3240 | 0.0e+00 |
| C16 ctx1024 64/8 | 16 | gluon 32x32 BM32/nw1 (0.76x) | 27.4 | 2467 | 4.9e-04 |
| C16 ctx1024 64/8 | 16 | gluon 16x16 BM16/nw1 (0.78x) | 26.7 | 2528 | 4.9e-04 |
| C64 ctx8192 64/8 | 8 | triton 3d+red | 374.9 | 5734 | 0.0e+00 |
| C64 ctx8192 64/8 | 8 | gluon 32x32 BM32/nw1 (0.96x) | 388.6 | 5532 | 1.2e-04 |
| C64 ctx8192 64/8 | 8 | gluon 16x16 BM16/nw1 (0.95x) | 395.7 | 5432 | 1.2e-04 |
| C128 ctx8192 64/8 | 8 | triton 3d+red | 759.2 | 5663 | 0.0e+00 |
| C128 ctx8192 64/8 | 8 | gluon 32x32 BM32/nw1 (0.94x) | 808.2 | 5320 | 2.4e-04 |
| C128 ctx8192 64/8 | 8 | gluon 16x16 BM16/nw1 (0.94x) | 810.1 | 5307 | 2.4e-04 |
| C64 ctx8192 8/1 | 64 | triton 3d+red | 58.2 | 4616 | 0.0e+00 |
| C64 ctx8192 8/1 | 64 | gluon 32x32 BM32/nw1 (0.91x) | 64.3 | 4178 | 1.2e-04 |
| C64 ctx8192 8/1 | 64 | gluon 16x16 BM16/nw1 (0.95x) | 61.3 | 4381 | 1.2e-04 |
