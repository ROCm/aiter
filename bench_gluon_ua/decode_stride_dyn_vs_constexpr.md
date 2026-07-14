# gfx950 gluon decode — older dynamic-stride vs new constexpr-stride kernel (triton 3.6.0 & 3.8.0)

**old** = pre-refactor kernel (HEAD `033550e66`): KV/block-table strides are runtime `gl.int32`
args threaded through the loader. **new** = current kernel: strides are `gl.constexpr` on
`AttentionConfig` (gfx1250-style, baked in). Decode grid, right-sized split, **nb=2** (double-buffer;
the config the dynamic kernel can compile in), 512 MB L2 flush, torch.profiler. VGPRs from the
compiled kernel (0 spills). Δ% = constexpr vs dynamic bandwidth.

> **Why nb=2 / HEAD:** the dynamic-stride variant of the *new* structure (constexpr routing +
> the single-buffer/ALL_DECODE decode-path changes) **fails to compile on both 3.6.0 and 3.8.0**
> — `builtin.unrealized_conversion_cast` fails to lower. Constexpr strides are what let the current
> kernel compile. HEAD is the last dynamic-stride version that compiles, so it's the baseline here.

## triton 3.8.0 (ToT) — dynamic (old) vs constexpr (new), nb=2
_bandwidth mean Δ +1.4%; constexpr uses ~9–13 fewer VGPRs_

| shape | S | dyn GB/s | cxpr GB/s | Δ% | dyn VGPR | cxpr VGPR |
|---|--:|--:|--:|--:|--:|--:|
| C16 ctx1024 64/8 | 8 | 3480 | 3527 | +1.3 | 173 | 164 |
| C32 ctx1024 64/8 | 4 | 4347 | 4442 | +2.2 | 173 | 164 |
| C64 ctx1024 64/8 | 2 | 5512 | 5558 | +0.8 | 173 | 164 |
| C128 ctx1024 64/8 | 1 | 6547 | 6711 | +2.5 | 175 | 162 |
| C16 ctx1024 8/1 | 16 | 905 | 928 | +2.6 | 173 | 164 |
| C32 ctx1024 8/1 | 16 | 1505 | 1550 | +3.0 | 173 | 164 |
| C64 ctx1024 8/1 | 16 | 2080 | 2098 | +0.8 | 173 | 164 |
| C128 ctx1024 8/1 | 8 | 3468 | 3487 | +0.5 | 173 | 164 |
| C16 ctx8192 64/8 | 8 | 5934 | 5967 | +0.5 | – | 164 |
| C32 ctx8192 64/8 | 4 | 6110 | 6103 | -0.1 | – | 164 |
| C64 ctx8192 64/8 | 2 | 6551 | 6555 | +0.1 | – | 164 |
| C128 ctx8192 64/8 | 1 | 6640 | 6647 | +0.1 | 175 | 162 |
| C16 ctx8192 8/1 | 64 | 3344 | 3453 | +3.3 | 173 | 164 |
| C32 ctx8192 8/1 | 32 | 4576 | 4710 | +2.9 | 173 | 164 |
| C64 ctx8192 8/1 | 16 | 5626 | 5701 | +1.3 | – | 164 |
| C128 ctx8192 8/1 | 8 | 6216 | 6262 | +0.7 | – | 164 |

## triton 3.6.0 — dynamic (old) vs constexpr (new), nb=2
_bandwidth mean Δ +1.1%; constexpr uses ~33–35 fewer VGPRs_

| shape | S | dyn GB/s | cxpr GB/s | Δ% | dyn VGPR | cxpr VGPR |
|---|--:|--:|--:|--:|--:|--:|
| C16 ctx1024 64/8 | 8 | 3444 | 3506 | +1.8 | 236 | 201 |
| C32 ctx1024 64/8 | 4 | 4229 | 4264 | +0.8 | 236 | 201 |
| C64 ctx1024 64/8 | 2 | 5501 | 5522 | +0.4 | 236 | 201 |
| C128 ctx1024 64/8 | 1 | 6577 | 6548 | -0.4 | 222 | 189 |
| C16 ctx1024 8/1 | 16 | 881 | 901 | +2.3 | 236 | 201 |
| C32 ctx1024 8/1 | 16 | 1468 | 1500 | +2.2 | 236 | 201 |
| C64 ctx1024 8/1 | 16 | 1991 | 2080 | +4.4 | 236 | 201 |
| C128 ctx1024 8/1 | 8 | 3198 | 3317 | +3.7 | 236 | 201 |
| C16 ctx8192 64/8 | 8 | 5938 | 5983 | +0.8 | – | 201 |
| C32 ctx8192 64/8 | 4 | 5990 | 5973 | -0.3 | – | 201 |
| C64 ctx8192 64/8 | 2 | 6419 | 6420 | +0.0 | – | 201 |
| C128 ctx8192 64/8 | 1 | 6602 | 6626 | +0.4 | 222 | 189 |
| C16 ctx8192 8/1 | 64 | 3097 | 3167 | +2.3 | 236 | 201 |
| C32 ctx8192 8/1 | 32 | 4443 | 4272 | -3.8 | 236 | 201 |
| C64 ctx8192 8/1 | 16 | 5167 | 5309 | +2.7 | – | 201 |
| C128 ctx8192 8/1 | 8 | 6148 | 6188 | +0.6 | – | 201 |

## Reference: new constexpr kernel at the shipping decode default (nb=1, single-buffer)

| shape | S | 3.6.0 GB/s / VGPR | 3.8.0 GB/s / VGPR |
|---|--:|--:|--:|
| C16 ctx1024 64/8 | 8 | 3366 / 181 | 3515 / 156 |
| C32 ctx1024 64/8 | 4 | 4117 / 181 | 4271 / 156 |
| C64 ctx1024 64/8 | 2 | 5157 / 181 | 5122 / 156 |
| C128 ctx1024 64/8 | 1 | 6487 / 177 | 6475 / 160 |
| C16 ctx1024 8/1 | 16 | 861 / 181 | 879 / 156 |
| C32 ctx1024 8/1 | 16 | 1436 / 181 | 1465 / 156 |
| C64 ctx1024 8/1 | 16 | 2224 / 181 | 2342 / 156 |
| C128 ctx1024 8/1 | 8 | 3262 / 181 | 3481 / 156 |
| C16 ctx8192 64/8 | 8 | 5721 / 181 | 5710 / 156 |
| C32 ctx8192 64/8 | 4 | 5847 / 181 | 5927 / 156 |
| C64 ctx8192 64/8 | 2 | 6158 / 181 | 6182 / 156 |
| C128 ctx8192 64/8 | 1 | 6815 / 177 | 7022 / 160 |
| C16 ctx8192 8/1 | 64 | 3339 / 181 | 3550 / 156 |
| C32 ctx8192 8/1 | 32 | 4114 / 181 | 4378 / 156 |
| C64 ctx8192 8/1 | 16 | 5391 / 181 | 5503 / 156 |
| C128 ctx8192 8/1 | 8 | 6098 / 181 | 6094 / 156 |

## Takeaways

- **Bandwidth is unchanged** (mean Δ within noise on both versions) — decode is bandwidth-bound
  with LDS-limited occupancy, so baking the strides in doesn't move throughput.
- **Constexpr uses fewer VGPRs**: ~11 fewer on 3.8.0 (≈175→164) and ~30 fewer on 3.6.0
  (≈222→190). The 3.6 saving is larger because its (older) compiler spilled the runtime strides
  into more registers. (This delta is the *net* old→new: constexpr strides + the decode-path
  simplifications; the isolated pure-stride effect measured earlier was ~neutral / −8 VGPR.)
- **Constexpr is required by the current kernel**, not just an optimization: the dynamic-stride
  variant of the new structure won't compile on 3.6 or 3.8 (`unrealized_conversion_cast`). Keeping
  strides constexpr (gfx1250-style) is the working, and correct-on-large-KV, choice.

