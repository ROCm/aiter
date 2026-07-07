from aiter.ops.triton.utils.common_utils import switch_to_contiguous_if_needed
from op_tests.triton_tests.attention.test_hstu_attn import (
    generate_sparse_seq_len,
    apply_SL,
    sanity_check_attention,
)
import pytest
import torch

from aiter.ops.flydsl.hstu_attention_kernels import (
    flydsl_hstu_attention_fwd,
)
from op_tests.triton_tests.utils.hstu_attention_ref import (
    torch_hstu_attention,
)


@pytest.mark.parametrize(
    "batch_size,max_seq_len,sparsity,"
    "max_attn_len,contextual_seq_len,target_size,"
    "attn_dim,hidden_dim",
    [
        (256, 1024, 0.5, 0, 0, 0, 128, 128),
        # target_size > 0
        (256, 1024, 0.5, 0, 0, 20, 128, 128),
        # max_attn_len > 0
        (256, 1024, 0.5, 64, 0, 0, 128, 128),
        # contextual_seq_len > 0
        (256, 1024, 0.5, 0, 64, 0, 128, 128),
        # symmetric and dims %64 != 0
        (256, 1024, 0.5, 0, 0, 0, 96, 96),
        # not symmetric
        (256, 1024, 0.5, 0, 0, 0, 128, 64),
        # not symmetric and dims %64 != 0
        (256, 1024, 0.5, 0, 0, 0, 96, 192),
    ],
)
def test_flydsl_hstu_attention(
    batch_size: int,
    max_seq_len: int,
    sparsity: float,
    max_attn_len: int,
    contextual_seq_len: int,
    target_size: int,
    attn_dim: int,
    hidden_dim: int,
    heads: int = 4,
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    # In prod, BF16 is used by HSTU attention
    dtype = torch.bfloat16
    invalid_attn_mask_type = "lower_triangular"
    causal = True
    alpha = 1.0 / attn_dim * 10000

    # generate inputs
    torch.manual_seed(1001)  # for reproducibility
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        device=torch.device("cuda"),
    )
    lengths = apply_SL(lengths, 0.2, max_seq_len=max_seq_len)

    num_targets = None
    if target_size > 0:
        num_targets = torch.randint(
            1,
            target_size + 1,
            (batch_size,),
            device=lengths.device,
            dtype=lengths.dtype,
        )
        num_targets = torch.where(num_targets > lengths, lengths, num_targets)

    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)

    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)

    sanity_check_attention(
        max_seq_len=max_seq_len,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        invalid_attn_mask_type=invalid_attn_mask_type,
        dropout_pr=0.0,
        attn_bias=None,
        max_attn_len=None if max_attn_len == 0 else max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )

    def flydsl_attn():
        return flydsl_hstu_attention_fwd(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            num_targets,
            max_attn_len,
            contextual_seq_len,
        )

    def torch_attn():
        return torch_hstu_attention(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            dropout_pr=0.0,
            training=False,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=0,
        )

    out = flydsl_attn() * max_seq_len
    out_ref = torch_attn() * max_seq_len
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=0)
