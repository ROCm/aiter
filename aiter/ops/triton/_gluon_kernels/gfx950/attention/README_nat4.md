# DSA V4 fwd — natural-order + K-prefetch, int32 offsets, deferred rescale

Research kernels branched from the committed early-gather kernel (`dsa_fwd_v4_gluon.py`, c4b07fe5a).
These are NOT wired into the production dispatch yet — standalone modules for evaluation.

## Kernels
- `dsa_fwd_v4_gluon_nat.py`  — natural-order compute (QK→softmax→PV, softmax exposed) with K
  prefetched into registers during PV (K_rd‖PV) and V read during QK (V_rd‖QK); 2 KV buffers,
  topk dedup + far-prefetch. `sparse_mla_fwd_v4_gluon_nat(...)`.
- `dsa_fwd_v4_gluon_nat4.py` — nat + two no-tile-change cycle cuts:
  1. **int32 KV-gather offsets** (halves the gather-address VALU; host-guard: safe while
     total_tokens·D_QK < 2^31, ~3.7M positions for D_QK=576 — a single 1M-context request fits).
  2. **deferred rescale** (frame reference m_frame, flush acc/l only when the running max drifts
     >8; removes the per-tile [BH,512] `acc*=alpha` 128-mul and shortens the acc chain).
  `sparse_mla_fwd_v4_gluon_nat4(...)`.

## Measured (cv350 MI350 proxy, GPU6 idle @ ~2.1 GHz, T=4096 H=128 TOPK=1152, BH64/TK32)
| kernel | ms | TFLOP/s | n_regs (vgpr/agpr) |
|---|---|---|---|
| orig early-gather (`dsa_fwd_v4_gluon.py`) | 3.029 | 434 | 404 (256+148) |
| nat | 2.944 | 446 | 501 (256+245) |
| **nat4 (int32 + deferred rescale)** | **2.748** | **478** | 465 (256+209) |

nat4 = **−9.3% vs orig, −6.6% vs nat**, 0 spill, lower register pressure (deferred rescale freed
~36 AGPR of overflow).

## Correctness (validated vs fp32 reference)
nat4's error vs fp32 ≈ nat's (ratio 1.08–1.54; max ~6e-3, mean ~2e-4, LSE ~1e-6) — bf16-level,
deferred rescale confirmed essentially lossless. Accuracy knob: lowering the flush threshold 8→4
caps P≤e^4 for tighter bf16 precision at ~no speed cost.

## Notes / next
- int32 needs a host-side guard for extreme flattened multi-million-position KV caches.
- Open: in-place-acc (drop the dst≠srcC ping-pong) to free ~50 VGPR for a V/K-coexistent PV‖K_rd
  interleave — register-feasible on nat4 (465−50+64<512) but needs measuring (in-place PV cost +
  whether the scheduler fills the bubbles). See dsa_dev/docs/NATURAL_ORDER_SCHEDULE_PROPOSAL.md.
