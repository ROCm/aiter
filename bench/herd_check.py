"""Check whether HERD activates for the DeepSeek-V4-Flash routing shape on this GPU (uses the GPU).

Run once per env setting; HERD flags are read at module import time.

[flydsl topk_gating] is aiter's automatic HERD inside topk_gating, which aiter enables on gfx950 only.
On gfx1250 the FlyDSL HERD is driven by ATOM#2713 instead (Top-(K+1) routing + fused_moe(herd_topk=K)),
so this line reads False there even when HERD works; check the server log for "FlyDSL HERD route".
"""
import os

import torch

import aiter.ops.topk as topk_mod
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.moe_routing import routing as routing_mod

M, E, K, SCALE = 64, 256, 6, 1.5

torch.manual_seed(0)
logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1

print(f"gfx={get_gfx()} AITER_FLYDSL_USE_HERD={os.environ.get('AITER_FLYDSL_USE_HERD', '')} "
      f"AITER_TRITON_USE_HERD={os.environ.get('AITER_TRITON_USE_HERD', '')}")

# FlyDSL side (ATOM_USE_TRITON_MOE=0): select_experts -> aiter.topk_gating.
calls = []
real_herd = None
try:
    from aiter.ops.flydsl import herd_topk as herd_mod

    real_herd = herd_mod.herd_topk_gating

    def spy(*args, **kwargs):
        calls.append(1)
        return real_herd(*args, **kwargs)

    herd_mod.herd_topk_gating = spy
except ImportError:
    pass

w = torch.empty(M, K, device="cuda", dtype=torch.float32)
ids = torch.empty(M, K, device="cuda", dtype=torch.int32)
use_herd = topk_mod._use_flydsl_herd(w, ids, logits, bias, "sqrtsoftplus", True)
topk_mod.topk_gating(w, ids, logits, bias, True, SCALE, score_func="sqrtsoftplus")
torch.cuda.synchronize()
print(f"[flydsl topk_gating, auto HERD (gfx950 only)] _use_flydsl_herd={use_herd} herd_topk_gating_calls={len(calls)} "
      f"unique_experts_in_batch={ids.unique().numel()}")

# Triton side (ATOM_USE_TRITON_MOE=1, decode): routing(score_mode='sqrtsoftplus').
herd_window = routing_mod._USE_HERD and routing_mod._HERD_MIN_M <= M <= routing_mod._HERD_MAX_M
rd, gather_idx, scatter_idx = routing_mod.routing(
    logits, K, score_mode="sqrtsoftplus", bias=bias, renorm=True, routed_scaling_factor=SCALE
)
torch.cuda.synchronize()
hist = rd.expt_hist
print(f"[triton routing] herd_branch={herd_window} active_experts_in_batch={int((hist > 0).sum())} "
      f"routed_slots={int(hist.sum())}")
