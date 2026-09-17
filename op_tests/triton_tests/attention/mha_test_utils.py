import pytest
import torch

from aiter.ops.triton.attention.mha import gluon_forward_unsupported_reason
from aiter.ops.triton.utils._triton import arch_info


def skip_if_gluon_unsupported(backend: str, **feature_flags):
    """
    Skip forward tests the Gluon backend can't run.
    """
    if backend != "gluon":
        return
    reason = gluon_forward_unsupported_reason(**feature_flags)
    if reason:
        pytest.skip(reason)


def skip_if_triton_padded_head_miscompiled(
    backend: str, HEAD_SZ: int, CAUSAL: bool, SEQLEN_K: int, NUM_K_HEADS: int
):
    """
    Skip Triton forward tests miscompiled by the pinned ROCm Triton build.

    Remove this function once the Triton compiler pinned by AITER CI is updated.
    """
    if (
        backend == "triton"
        and HEAD_SZ == 33
        and CAUSAL
        and SEQLEN_K >= 2048
        and NUM_K_HEADS == 1
        and arch_info.get_arch() == "gfx950"
    ):
        pytest.skip(
            "gfx950: padded head size 33 is miscompiled by the pinned ROCm "
            "Triton build for causal long-sequence single-KV-head cases; "
            "passes on triton-lang/triton@2074a1b"
        )


def pad_rearrange_dropout_mask(
    S_dmask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqlen_q,
    seqlen_k,
    num_q_heads,
):
    batch_size = cu_seqlens_q.numel() - 1

    padded_dropout_mask = torch.ones(
        (batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda"
    )
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
            padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[
                b, h, :, :
            ]

    return padded_dropout_mask
