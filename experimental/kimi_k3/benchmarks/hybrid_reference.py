"""AITER expert invocation and independent unfused reference for K3 hybrid."""

import torch
from validate_complete_moe import situ, tp_sum
from k3_gluon_hybrid_b8.host import expert_weights as expert_weights, run_experts


def reference(x, ref, expert, rank, world, precision="a4w4"):
    scores = (x.float() @ ref["router"].float().T).sigmoid()
    ids = (
        torch.argsort(scores + ref["bias"], descending=True, stable=True)[:, :16]
        .int()
        .contiguous()
    )
    weights = scores.gather(1, ids.long())
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    latent = (x.float() @ ref["down"].float().T).bfloat16()
    local = run_experts(latent, expert, ids, weights, precision=precision)
    routed = tp_sum(local, world)
    rr = routed.float()
    norm = (
        rr
        * torch.rsqrt(rr.square().mean(-1, keepdim=True) + 1.0e-5)
        * ref["gain"].float()
    ).bfloat16()
    gu = (x.float() @ ref["shared"].float().T).bfloat16()
    partial = (situ(gu).float() @ ref["sd"].float().T).bfloat16()
    shard = slice(rank * (7168 // world), (rank + 1) * (7168 // world))
    partial[:, shard] = (
        partial[:, shard].float() + norm.float() @ ref["up"].float().T
    ).bfloat16()
    return dict(
        output=tp_sum(partial, world),
        ids=ids,
        weights=weights,
        latent=latent,
        routed=routed,
    )


def check(x, state, ref, expert, rank, world, precision="a4w4"):
    expected = reference(x, ref, expert, rank, world, precision)
    torch.testing.assert_close(state["IDs"], expected["ids"], rtol=0, atol=0)
    torch.testing.assert_close(
        state["Weights"], expected["weights"], rtol=2.0e-5, atol=2.0e-6
    )
    diff = state["Out"].float() - expected["output"].float()
    nrmse = (
        diff.square().mean().sqrt()
        / expected["output"].float().square().mean().sqrt().clamp_min(1.0e-12)
    ).item()
    assert nrmse < 0.025, nrmse
    assert not state["Errors"].count_nonzero().item()
    assert torch.all(state["Epochs"] == state["Epochs"][0])
    return dict(
        output_full_nrmse=nrmse, max_abs=diff.abs().max().item(), mailbox_errors=0
    )
