"""Minimal reproducer for the gfx1250 a8w4 mxfp4 grouped-MoE crash.

Bug: running several *different* token sizes through the grouped a8w4 MoE path
in one process eventually triggers a GPU memory access fault
(HSA_STATUS_ERROR_EXCEPTION code 0x1016 / "Memory access fault ... Page not
present"). Each token size on its own -- or the same token size repeated -- is
fine; the fault only appears once a large-token call has churned the caching
allocator and a subsequent call's kernel over-reads onto an unmapped page.

Because the failure is a hard GPU abort (SIGABRT), it cannot be caught in-process
with pytest.raises. Instead each scenario runs in its own subprocess and we
assert on the exit code:

    * control  (same token repeated)      -> exits 0
    * repro    (mixed token sizes)        -> aborts (negative / 134 return code)

Run directly:
    python op_tests/test_grouped_a8w4_crash_repro.py 16 512 16   # aborts
    python op_tests/test_grouped_a8w4_crash_repro.py 16 16 16    # ok
Or under pytest:
    pytest -q op_tests/test_grouped_a8w4_crash_repro.py
"""

import os
import sys
import subprocess

# ---- shape (DSv4 decode-ish; grouped a8w4 mxfp4, EP=8) ----
MODEL_DIM, INTER_DIM, E, TOPK, EP = 7168, 3072, 384, 6, 8
MAX_TOKENS = 4096 * 4

_CHILD_ENV = {
    "ENABLE_CK": "0",  # ck_tile has no gfx1250 target; build via the shim
    "AITER_FORCE_A8W4": "1",
    "AITER_USE_GROUPED_GEMM": "1",
    "AITER_BF16_FP8_MOE_BOUND": "0",
}


def _run_moe_sweep(tokens):
    """Call fused_moe once per token size, sharing one static weight set."""
    import torch
    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_topk, fused_moe
    from aiter.ops.shuffle import shuffle_weight, moe_shuffle_scale
    from aiter.ops.flydsl.moe_common import GateMode

    if aiter.get_gfx() != "gfx1250":
        print(f"[repro] skip: needs gfx1250, got {aiter.get_gfx()}", flush=True)
        return

    dtype = dtypes.bf16
    ep_id = EP - 1
    expert_mask = torch.zeros((E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // EP) : (ep_id + 1) * E // EP] = 1
    local_E = int(expert_mask.sum().item())
    expert_mask[-1] = 0
    expert_mask[E:-1] = 1

    def quant(w):
        w_qt, w_scale = aiter.get_torch_quant(QuantType.per_1x32)(
            w, quant_dtype=dtypes.fp4x2
        )
        return w_qt.view(w.shape[0], w.shape[1], w.shape[2] // 2), w_scale

    w1 = torch.randn((local_E, INTER_DIM * 2, MODEL_DIM), dtype=dtype, device="cuda") / 10
    w2 = torch.randn((local_E, MODEL_DIM, INTER_DIM), dtype=dtype, device="cuda") / 10
    w1_qt, w1_scale = quant(w1)
    w2_qt, w2_scale = quant(w2)
    w1_a = shuffle_weight(w1_qt.view(torch.uint8), layout=(16, 16))
    w2_a = shuffle_weight(w2_qt.view(torch.uint8), layout=(16, 16))
    w1_s = moe_shuffle_scale(w1_scale.contiguous(), experts_cnt=local_E)
    w2_s = moe_shuffle_scale(w2_scale.contiguous(), experts_cnt=local_E)

    for tok in tokens:
        inp = torch.randn((tok, MODEL_DIM), dtype=dtype, device="cuda")
        score = torch.randn((tok, E), dtype=dtype, device="cuda")
        # DSv4 layout: TOPK routed columns + 1 trailing "fake" expert column
        # (global id == E) that the EP mask always drops. This dropped-route
        # column is what triggers the fault; TOPK real columns alone do not.
        ids = torch.empty((MAX_TOKENS, TOPK + 1), dtype=dtypes.i32, device="cuda")
        w = torch.empty((MAX_TOKENS, TOPK + 1), dtype=dtypes.fp32, device="cuda")
        fused_topk(inp, score, TOPK, True, ids[:, :TOPK], w[:, :TOPK])
        ids[:, TOPK] = E  # fake/shared expert -> dropped by expert_mask[E]==0
        w[:, TOPK] = 0.1
        fused_moe(
            inp, w1_a, w2_a, w[:tok], ids[:tok],
            expert_mask=expert_mask, activation=ActivationType.Silu,
            gate_mode=GateMode.SEPARATED.value, quant_type=QuantType.per_1x32,
            w1_scale=w1_s, w2_scale=w2_s,
        )
        torch.cuda.synchronize()
        print(f"[repro] token={tok} OK", flush=True)
    print("[repro] ALL OK", flush=True)


def _spawn(tokens):
    env = {**os.environ, **_CHILD_ENV}
    return subprocess.run(
        [sys.executable, __file__, *map(str, tokens)], env=env
    ).returncode


def test_single_call_ok():
    """Control: a single fused_moe call in a fresh process is safe.

    This is what run_dsv4_moe_ep.sh's PER_TOKEN=1 relies on (one call per
    process), and it passes reliably.
    """
    assert _spawn([16]) == 0


def test_mixed_token_sizes_crash():
    """Repro: several different token sizes in ONE process aborts.

    ``16 512 16`` reproduces reliably: the large (512) call churns the caching
    allocator, then the following small (16) call's grouped a8w4 kernel over-reads
    onto an unmapped page -> HSA_STATUS_ERROR_EXCEPTION (code 0x1016). The fault
    is allocator-layout dependent, so mixing sizes (rather than repeating one) is
    what makes it deterministic.
    """
    rc = _spawn([16, 512, 16])
    assert rc != 0, "expected a GPU memory-access-fault abort, but it exited cleanly"


if __name__ == "__main__":
    _run_moe_sweep([int(a) for a in sys.argv[1:]] or [16, 512, 16])
