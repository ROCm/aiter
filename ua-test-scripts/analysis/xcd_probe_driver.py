"""Minimal real-kernel driver for XCD-balance profiling.

Calls the production `unified_attention_fwd` (the SAME path aiter ships, so the
XCD-swizzled decode grid is exercised) in a tight loop. No torch reference, no
Triton, no correctness check -> rocprofv3 only has to instrument the one CK
decode kernel (+ combine), which keeps a PMC pass to a few seconds.

Env:
  HEADS=hq,hk   (e.g. 64,8 or 5,5)     B=batch   SK=context_len
  ITERS=iters   AITER_UA_FORCE_SPLITS=N (optional; else heuristic)
"""
import os
import sys

import torch

sys.argv = [sys.argv[0]]  # neutralize test-module argparse at import
_AITER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_AITER, "op_tests"))
import test_unified_attention_ck as T  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.ops.unified_attention import (  # noqa: E402
    _pick_num_splits,
    unified_attention_fwd,
)

heads = tuple(int(x) for x in os.environ.get("HEADS", "64,8").split(","))
b = int(os.environ.get("B", "4"))
sk = int(os.environ.get("SK", "75600"))
iters = int(os.environ.get("ITERS", "20"))
bs = 128
device = "cuda"

blocks_per_seq = (sk + bs - 1) // bs
num_blocks = 2 * b * blocks_per_seq  # mirrors test's --num-blocks auto

cfg = T.CaseConfig(
    seq_lens=[(1, sk)] * b,
    num_heads=heads,
    head_size=128,
    block_size=bs,
    dtype=torch.bfloat16,
    q_dtype="fp8",
    num_blocks=num_blocks,
    mask_type=2,
)
t = T._make_inputs(cfg, device, seed=0)
out = torch.empty(t["total_q"], heads[0], 128, dtype=dtypes.bf16, device=device)

ns = _pick_num_splits(t["q_fp8"], t["k_fp8"], t["kv_lens"], t["block_tables"])
forced = os.environ.get("AITER_UA_FORCE_SPLITS")
print(
    f"[driver] heads={heads} b={b} sk={sk} num_blocks={num_blocks} "
    f"num_splits={'forced=' + forced if forced else ns} iters={iters}",
    flush=True,
)


def one():
    unified_attention_fwd(
        out, t["q_fp8"], t["k_fp8"], t["v_fp8"],
        t["block_tables"], t["kv_lens"], t["cu_seqlens_q"],
        mask_type=2, scale_s=t["scale"],
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        cache_ptr_int32_overflow_possible=False,
        allow_splitkv=True,
        q_descale=float(t["q_descale"]), k_descale=float(t["k_descale"]),
        v_descale=float(t["v_descale"]),
        max_seqlen_q=t["max_query_len"],
        window_size_left=-1, window_size_right=-1,
        is_paged=True, kv_start_len=None,
    )


for _ in range(3):  # warmup / JIT
    one()
torch.cuda.synchronize()
for _ in range(iters):
    one()
torch.cuda.synchronize()
print("[driver] done", flush=True)
