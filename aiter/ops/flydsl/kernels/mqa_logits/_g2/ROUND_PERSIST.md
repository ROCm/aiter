# Round PERSIST: persistent bf16 r32 prefill blocks (opt-in)

Base = champion `adea96d5` (kernel sha `9428b7ef…`). No push, no GPU work, no
frozen-directory changes. **Opt-in until measured**: default routing and every
existing specialization are unchanged (source-identical functions, byte-identical
final ISA versus the merge evidence).

## Switch

| env | default | effect |
|---|---|---|
| `QSA_BF16_PERSIST` | `0` | `1` routes bf16 r32 TDM prefill with `columns >= 64*QSA_PERSIST_COLS` to the persistent kernel |
| `QSA_PERSIST_COLS` | `4` | chunks (64-column tiles) per block: `2`, `4` or `8` |

Raw/cop access, unaligned cache bases (-> raw r16), r16 decode/draft, r64 opt-in,
fp8 and every narrower-than-one-span call keep their champion paths. Builder
private API: `_build_wmma_kernel(div, 32, "persist", chunks)`; wrapper call text is
unchanged (`k_load = "persist"` is set only by the opt-in branch).

## Kernel `..._r32_c64_flydsl_tdm_persist{P}`

Grid `ceil(columns / (64P)) x ceil(rows / 32)`, 128 threads, LDS 2 x 17,408 B
(272 B pitch) = 34,816 B. Per full single-request block (same `_r32_guard`):

```
issue(0 -> buf0); wait(0); barrier
for k in 0..P-1 (unrolled):
    issue(k+1 -> buf (k+1)%2)        # 2 x 8-row gathers per wave, SGPR-only D#
    consume(k  <- buf k%2)           # unchanged _pair_values (WMMA + stores)
    wait(0); barrier                 # (k+1 < P): publishes k+1, retires buf k%2
```

One wait(0) + one barrier per chunk (no more than the single-tile kernel), but
chunk k+1's TDM gathers are in flight while chunk k's ds_loads/WMMA/stores run.
Q fragments, row metadata, the 32-row reuse guard and plane descriptors are now
amortised over P tiles. Issue is skipped (SGPR-uniform branch; wait/barrier stay
unconditional) for chunks with `col0 >= columns` or `col0 >= max(ctx,0)/4` — every
row's visibility is below that bound, so those chunks store only -inf masks.
Mixed/partial row blocks run the exact raw r16 per-M-tile fallback per chunk.

## Evidence

* `_g2/test_persist_cpu.py` (16 tests): base source identity, two-buffer plan,
  AST interpretation of the real kernel body == plan for P = 2/4/8, per-iteration
  wait/barrier order, 4-wave adversarial interleaving simulator (+ negative
  controls), byte-exact K fragments per chunk/wave/lane/n-tile/k-chunk through the
  real producer helper vs per-lane global reference, store exactly-once over the
  real grid, LDS bounds, SGPR-only guards, routing/env validation.
* `_g2/summarize_persist_cpu.py`: no-device final-ISA gate + byte identity.
* `_g2/host_verify_persist_cpu.sh`: compile (auto/raw/tdm/p2/p4/p8) + every CPU gate.
