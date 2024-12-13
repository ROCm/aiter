import torch
import torch.nn.functional as F
import ater
from ater.test_common import checkAllclose, perftest


@perftest()
def test_nofuse(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

    ater.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
        False
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


@perftest()
def test_fuse(hidden_states: torch.Tensor,
              gating_output: torch.Tensor,
              topk: int,
              renormalize: bool):
    from ater.fused_moe_gelu import fused_topk
    return fused_topk(hidden_states, gating_output, topk, renormalize)


def test_topk_softmax(dtype, m, n, E, topk):
    dim = (m, n)
    hidden_states = torch.randn(dim, dtype=dtype, device="cuda")
    gating_output = torch.randn((m, E), dtype=dtype, device="cuda")

    (topk_weights_a, topk_ids_a), avg_a = test_nofuse(
        hidden_states, gating_output, topk, True)
    (topk_weights_b, topk_ids_b), avg_b = test_fuse(
        hidden_states, gating_output, topk, True)
    msg = f"[perf] {m=}, {n=}, {E=}, {topk=}, dtype: {dtype}, ref avg: {avg_a:<8.2f} us, b avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(topk_weights_a, topk_weights_b,
                  atol=0.03, msg=msg)
    checkAllclose(topk_ids_a, topk_ids_b,
                  atol=0, msg='topk_ids')


for dtype in [torch.float16, torch.bfloat16]:
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256][-2:-1]:
        for n in [4096, 8192, 16384, 32768, 65536][1:2]:
            test_topk_softmax(dtype, m, n, 32, 5)
