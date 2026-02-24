# Copyright (C) 2024, Tri Dao.
#
# AIter causal_conv1d tests
#
# Issues fixed:
# 1. float16/bfloat16 support added to causal_conv1d_channellast_fwd_kernel_traits
#    - Modified kernel traits to support templated input_t and weight_t
#    - Now correctly handles FP16/BF16 data types
# 2. When has_bias=False, need to pass a zero tensor with shape (dim,) instead of None
#

import torch
import torch.nn.functional as F
import pytest
import aiter


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
):
    """
    Wrapper function to match the original causal_conv1d_update signature.
    Adapts the call to aiter's causal_conv1d_update interface.
    """
    batch, dim, seqlen = x.shape

    # Create output tensor (initialize to zero to handle padding slots correctly)
    # When conv_state_indices[i] == pad_slot_id, the kernel will skip processing
    # and leave the output unchanged, so we need to initialize to zero
    out = torch.zeros_like(x)

    # Convert weight to input dtype if needed
    weight_tensor = weight.to(dtype=x.dtype) if weight.dtype != x.dtype else weight

    # Convert bias to empty tensor if None, otherwise convert to input dtype
    if bias is None:
        bias_tensor = torch.empty(0, dtype=x.dtype, device=x.device)
    else:
        bias_tensor = bias.to(dtype=x.dtype) if bias.dtype != x.dtype else bias

    # Convert cache_seqlens to empty tensor if None
    if cache_seqlens is None:
        cache_seqlens_tensor = torch.empty(0, dtype=torch.int32, device=x.device)
    else:
        cache_seqlens_tensor = cache_seqlens

    # Convert conv_state_indices to empty tensor if None
    if conv_state_indices is None:
        conv_state_indices_tensor = torch.empty(0, dtype=torch.int32, device=x.device)
    else:
        conv_state_indices_tensor = conv_state_indices

    # Convert activation string to bool
    use_silu = activation in ["silu", "swish"]

    # Call aiter's causal_conv1d_update
    aiter.causal_conv1d_update(
        x,
        conv_state,
        weight_tensor,
        bias_tensor,
        out,
        use_silu,
        cache_seqlens_tensor,
        conv_state_indices_tensor,
        -1,  # pad_slot_id
    )

    return out


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update(
    dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype
):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()
    conv_state = torch.randn(
        batch, state_len, dim, device=device, dtype=itype
    ).transpose(-1, -2)
    weight = torch.randn(
        dim, width, device=device, dtype=torch.float32, requires_grad=True
    )
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (
        torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
        if has_cache_seqlens
        else None
    )
    out = causal_conv1d_update(
        x, conv_state, weight, bias, activation=activation, cache_seqlens=cache_seqlens
    )
    out_ref = causal_conv1d_update_ref(
        x,
        conv_state_ref,
        weight,
        bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update_with_batch_gather(
    dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype
):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()

    total_entries = 10 * batch
    conv_state = torch.randn(
        total_entries, state_len, dim, device=device, dtype=itype
    ).transpose(-1, -2)
    conv_state_indices = torch.randperm(total_entries)[:batch].to(
        dtype=torch.int32, device=device
    )

    weight = torch.randn(
        dim, width, device=device, dtype=torch.float32, requires_grad=True
    )
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (
        torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
        if has_cache_seqlens
        else None
    )
    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )
    out_ref = causal_conv1d_update_ref(
        x,
        conv_state_ref,
        weight,
        bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
