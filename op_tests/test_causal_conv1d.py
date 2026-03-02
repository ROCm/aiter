# Copyright (C) 2024, Tri Dao.
# AIter causal_conv1d benchmark tests with @perftest and @benchmark

import argparse
import itertools
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter.test_common import benchmark, perftest

PAD_SLOT_ID = -1


def seed_everything(seed: int = 0) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    pad_slot_id=-1,
):
    """Wrapper for aiter.causal_conv1d_update."""
    batch, dim, seqlen = x.shape
    out = torch.zeros_like(x)
    weight_tensor = weight.to(dtype=x.dtype) if weight.dtype != x.dtype else weight
    if bias is None:
        bias_tensor = torch.empty(0, dtype=x.dtype, device=x.device)
    else:
        bias_tensor = bias.to(dtype=x.dtype) if bias.dtype != x.dtype else bias
    if cache_seqlens is None:
        cache_seqlens_tensor = torch.empty(0, dtype=torch.int32, device=x.device)
    else:
        cache_seqlens_tensor = cache_seqlens
    if conv_state_indices is None:
        conv_state_indices_tensor = torch.empty(0, dtype=torch.int32, device=x.device)
    else:
        conv_state_indices_tensor = conv_state_indices
    use_silu = activation in ["silu", "swish"]

    aiter.causal_conv1d_update(
        x,
        conv_state,
        weight_tensor,
        bias_tensor,
        out,
        use_silu,
        cache_seqlens_tensor,
        conv_state_indices_tensor,
        pad_slot_id,
    )
    return out


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """
    Reference for causal_conv1d_update with cache_seqlens (circular buffer).
    conv_state: (batch, dim, state_len). cache_seqlens: (batch,), dtype int32.
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
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
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


def causal_conv1d_update_with_indices_ref(
    x, conv_state, weight, bias=None, activation=None, conv_state_indices=None
):
    """Reference implementation with conv_state_indices (linear shift)."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    num_cache_lines = conv_state.shape[0]
    assert weight.shape == (dim, width)
    assert state_len == width - 1

    if conv_state_indices is None:
        assert conv_state.shape == (batch, dim, state_len)
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        assert conv_state_indices.shape == (batch,)
        assert num_cache_lines >= conv_state_indices.max().item() + 1
        x_new_list = []
        for i in range(batch):
            slot = int(conv_state_indices[i].item())
            state_slot = conv_state[slot : slot + 1, :, :].clone()
            x_new_i = torch.cat(
                [state_slot.to(weight.dtype), x[i : i + 1, :, :].to(weight.dtype)],
                dim=-1,
            )
            conv_state[slot, :, :] = torch.cat(
                [state_slot[:, :, 1:], x[i : i + 1, :, :]], dim=-1
            ).to(conv_state.dtype)[:, :, -state_len:]
            x_new_list.append(x_new_i)
        x_new = torch.cat(x_new_list, dim=0)

    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@perftest()
def run_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    conv_state_indices: torch.Tensor,
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """Run causal_conv1d_update kernel for perf measurement."""
    return causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        pad_slot_id=pad_slot_id,
    )


@perftest()
def run_causal_conv1d_update_batch_gather(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    cache_seqlens: torch.Tensor,
    conv_state_indices: torch.Tensor,
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """Run causal_conv1d_update with cache_seqlens + conv_state_indices for perf measurement."""
    return causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        pad_slot_id=pad_slot_id,
    )


@perftest()
def run_causal_conv1d_update_cache_seqlens(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Run causal_conv1d_update kernel with cache_seqlens for perf measurement."""
    return causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )


@benchmark()
def test_causal_conv1d_update_with_indices_decode(
    batch: int,
    dim: int,
    width: int,
    seqlen: int,
    has_bias: bool = False,
    silu_activation: bool = True,
    itype: torch.dtype = torch.bfloat16,
    num_cache_lines: int = 128,
    with_padding: bool = False,
    padding: int = 5,
) -> dict:
    """Test causal_conv1d_update with conv_state_indices. Uses @perftest for timing."""
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    seed_everything(0)
    padded_batch = batch + (padding if with_padding else 0)
    x = torch.randn(padded_batch, dim, seqlen, device=device, dtype=itype)
    # Need num_slots >= batch so we can assign batch distinct indices
    num_slots = max(num_cache_lines, batch)
    conv_state_initial = (
        torch.randn(num_slots, width - 1, dim, device=device, dtype=itype)
        .contiguous()
        .transpose(1, 2)
    )
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = (
        torch.randn(dim, device=device, dtype=itype)
        if has_bias
        else torch.empty(0, dtype=itype, device=device)
    )
    activation = None if not silu_activation else "silu"
    valid_indices = torch.randperm(num_slots, device=device, dtype=torch.int32)[:batch]
    if with_padding:
        conv_state_indices = torch.cat(
            [
                valid_indices,
                torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            ],
            dim=0,
        )
        pad_slot_id = PAD_SLOT_ID
    else:
        conv_state_indices = valid_indices
        pad_slot_id = -1

    conv_state = conv_state_initial.clone()
    conv_state_ref = conv_state_initial.clone()
    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        pad_slot_id=pad_slot_id,
    )
    x_ref = x[:batch]
    conv_state_indices_ref = conv_state_indices[:batch]
    out_ref = causal_conv1d_update_with_indices_ref(
        x_ref,
        conv_state_ref,
        weight,
        bias if has_bias else None,
        activation=activation,
        conv_state_indices=conv_state_indices_ref,
    )

    conv_state_t = conv_state_initial.clone()
    _, us = run_causal_conv1d_update(
        x,
        conv_state_t,
        weight,
        bias,
        activation,
        conv_state_indices,
        pad_slot_id,
    )

    all_close_state = torch.equal(conv_state, conv_state_ref)
    all_close_out = torch.allclose(out[:batch], out_ref, rtol=rtol, atol=atol)
    all_close = all_close_state and all_close_out

    return {
        "batch": batch,
        "dim": dim,
        "width": width,
        "seqlen": seqlen,
        "dtype": str(itype),
        "all_close": all_close,
        "us": us,
    }


@benchmark()
def test_causal_conv1d_update_cache_seqlens_decode(
    batch: int,
    dim: int,
    width: int,
    seqlen: int,
    has_bias: bool = False,
    silu_activation: bool = True,
    itype: torch.dtype = torch.bfloat16,
    has_cache_seqlens: bool = False,
) -> dict:
    """Test causal_conv1d_update with cache_seqlens (circular buffer). Uses @perftest for timing."""
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    seed_everything(0)
    state_len = width - 1
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    conv_state_initial = (
        torch.randn(batch, state_len, dim, device=device, dtype=itype)
        .contiguous()
        .transpose(1, 2)
    )
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = (
        torch.randn(dim, device=device, dtype=itype)
        if has_bias
        else torch.empty(0, dtype=itype, device=device)
    )
    activation = None if not silu_activation else "silu"
    cache_seqlens = (
        torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
        if has_cache_seqlens
        else None
    )

    conv_state = conv_state_initial.clone()
    conv_state_ref = conv_state_initial.clone()
    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )
    out_ref = causal_conv1d_update_ref(
        x,
        conv_state_ref,
        weight,
        bias if has_bias else None,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )

    conv_state_t = conv_state_initial.clone()
    cache_seqlens_for_perf = (
        cache_seqlens
        if has_cache_seqlens
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    _, us = run_causal_conv1d_update_cache_seqlens(
        x,
        conv_state_t,
        weight,
        bias,
        activation,
        cache_seqlens_for_perf,
    )

    all_close_state = torch.equal(conv_state, conv_state_ref)
    all_close_out = torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    all_close = all_close_state and all_close_out

    return {
        "batch": batch,
        "dim": dim,
        "width": width,
        "seqlen": seqlen,
        "has_cache_seqlens": has_cache_seqlens,
        "dtype": str(itype),
        "all_close": all_close,
        "us": us,
    }


@benchmark()
def test_causal_conv1d_update_with_batch_gather_decode(
    batch_size: int,
    dim: int,
    width: int,
    seqlen: int,
    has_bias: bool = False,
    silu_activation: bool = True,
    itype: torch.dtype = torch.bfloat16,
    has_cache_seqlens: bool = False,
    with_padding: bool = False,
    padding: int = 5,
    total_entries_scale: int = 10,
) -> dict:
    """Test causal_conv1d_update with conv_state_indices + cache_seqlens + padding (batch_gather)."""
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    seed_everything(0)
    pad_cnt = padding if with_padding else 0
    padded_batch_size = batch_size + pad_cnt
    total_entries = total_entries_scale * batch_size
    state_len = width - 1

    x = torch.randn(padded_batch_size, dim, seqlen, device=device, dtype=itype)
    x_ref = x.clone()

    conv_state_indices = torch.randperm(
        total_entries, device=device, dtype=torch.int32
    )[:batch_size]
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[conv_state_indices] = False
    padded_state_indices = (
        torch.cat(
            [
                conv_state_indices,
                torch.full((pad_cnt,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            ],
            dim=0,
        )
        if with_padding
        else conv_state_indices
    )

    conv_state_initial = (
        torch.randn(total_entries, state_len, dim, device=device, dtype=itype)
        .contiguous()
        .transpose(1, 2)
    )
    conv_state = conv_state_initial.clone()
    conv_state_initial_for_unused = conv_state_initial.clone() if with_padding else None

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = (
        torch.randn(dim, device=device, dtype=itype)
        if has_bias
        else torch.empty(0, dtype=itype, device=device)
    )
    conv_state_ref = conv_state[conv_state_indices, :].clone()
    activation = None if not silu_activation else "silu"

    if has_cache_seqlens:
        cache_seqlens_real = torch.randint(
            0, 1024, (batch_size,), dtype=torch.int32, device=device
        )
        if with_padding:
            cache_seqlens = torch.cat(
                [
                    cache_seqlens_real,
                    torch.zeros(pad_cnt, dtype=torch.int32, device=device),
                ],
                dim=0,
            )
        else:
            cache_seqlens = cache_seqlens_real
        cache_seqlens_for_kernel = cache_seqlens
    else:
        cache_seqlens = None
        cache_seqlens_for_kernel = None

    pad_slot_id = PAD_SLOT_ID if with_padding else -1

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        cache_seqlens=cache_seqlens_for_kernel,
        conv_state_indices=padded_state_indices,
        pad_slot_id=pad_slot_id,
    )

    cache_seqlens_ref = (
        cache_seqlens[:batch_size]
        if has_cache_seqlens and cache_seqlens is not None
        else None
    )
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size],
        conv_state_ref,
        weight,
        bias if has_bias else None,
        activation=activation,
        cache_seqlens=cache_seqlens_ref,
    )

    all_close_state = torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    if with_padding:
        all_close_unused = torch.equal(
            conv_state[unused_states_bool],
            conv_state_initial_for_unused[unused_states_bool],
        )
    else:
        all_close_unused = True
    all_close_out = torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)
    all_close = all_close_state and all_close_unused and all_close_out

    conv_state_t = conv_state_initial.clone()
    cache_seqlens_perf = (
        cache_seqlens
        if has_cache_seqlens
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    _, us = run_causal_conv1d_update_batch_gather(
        x,
        conv_state_t,
        weight,
        bias,
        activation,
        cache_seqlens_perf,
        padded_state_indices,
        pad_slot_id,
    )

    return {
        "batch": batch_size,
        "dim": dim,
        "width": width,
        "seqlen": seqlen,
        "has_cache_seqlens": has_cache_seqlens,
        "with_padding": with_padding,
        "dtype": str(itype),
        "all_close": all_close,
        "us": us,
    }


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Benchmark causal_conv1d_update",
)
parser.add_argument("-b", "--batch", type=int, default=[1, 64, 256], nargs="+")
parser.add_argument("-d", "--dim", type=int, default=[2048, 4096], nargs="+")
parser.add_argument("-w", "--width", type=int, default=[2, 3, 4], nargs="+")
parser.add_argument("-s", "--seqlen", type=int, default=[1], nargs="+")
parser.add_argument(
    "--has-bias",
    nargs="*",
    default=["true", "false"],
    choices=["true", "false"],
    help="Use bias (default: both)",
)
parser.add_argument(
    "--silu-activation",
    nargs="*",
    default=["true", "false"],
    choices=["true", "false"],
    help="Use SiLU activation (default: both)",
)
parser.add_argument(
    "--itype",
    type=str,
    default="bf16",
    choices=["bf16", "fp16", "fp32"],
    help="Input dtype",
)
parser.add_argument(
    "-n",
    "--num-cache-lines",
    type=int,
    default=512,
    help="Number of conv_state cache lines (indices mode)",
)
parser.add_argument(
    "--with-padding",
    nargs="*",
    default=["true", "false"],
    choices=["true", "false"],
    help="Add padding rows (indices/batch_gather mode, default: both)",
)
parser.add_argument(
    "--padding",
    type=int,
    default=5,
    help="Number of padding rows when --with-padding",
)
parser.add_argument(
    "--has-cache-seqlens",
    nargs="*",
    default=["true"],
    choices=["true", "false"],
    help="Use cache_seqlens (default: both)",
)
parser.add_argument(
    "--mode",
    type=str,
    nargs="*",
    default=["indices", "cache_seqlens", "batch_gather"],
    choices=["indices", "cache_seqlens", "batch_gather"],
    help="Test mode(s), default: all three",
)
parser.add_argument(
    "--total-entries-scale",
    type=int,
    default=10,
    help="total_entries = scale * batch (batch_gather mode)",
)
args = parser.parse_args()

itype = _DTYPE_MAP[args.itype]


def _parse_bool_list(lst):
    return [v.lower() == "true" for v in (lst or ["true", "false"])]


has_bias_list = _parse_bool_list(args.has_bias)
silu_activation_list = _parse_bool_list(args.silu_activation)
with_padding_list = _parse_bool_list(args.with_padding)
has_cache_seqlens_list = _parse_bool_list(args.has_cache_seqlens)
mode_list = args.mode if args.mode else ["indices", "cache_seqlens", "batch_gather"]


def _mode_param_pairs(mode):
    """Yield (with_padding, has_cache_seqlens) for each mode - only relevant flags."""
    if mode == "indices":
        for wp in with_padding_list:
            yield (wp, False)
    elif mode == "cache_seqlens":
        for hc in has_cache_seqlens_list:
            yield (False, hc)
    else:
        for wp, hc in itertools.product(with_padding_list, has_cache_seqlens_list):
            yield (wp, hc)


df = []
for mode in mode_list:
    for batch in args.batch:
        for dim in args.dim:
            for width in args.width:
                for seqlen in args.seqlen:
                    for has_bias in has_bias_list:
                        for silu_activation in silu_activation_list:
                            for with_padding, has_cache_seqlens in _mode_param_pairs(
                                mode
                            ):
                                if mode == "indices":
                                    ret = test_causal_conv1d_update_with_indices_decode(
                                        batch=batch,
                                        dim=dim,
                                        width=width,
                                        seqlen=seqlen,
                                        has_bias=has_bias,
                                        silu_activation=silu_activation,
                                        itype=itype,
                                        num_cache_lines=args.num_cache_lines,
                                        with_padding=with_padding,
                                        padding=args.padding,
                                    )
                                elif mode == "batch_gather":
                                    ret = test_causal_conv1d_update_with_batch_gather_decode(
                                        batch_size=batch,
                                        dim=dim,
                                        width=width,
                                        seqlen=seqlen,
                                        has_bias=has_bias,
                                        silu_activation=silu_activation,
                                        itype=itype,
                                        has_cache_seqlens=has_cache_seqlens,
                                        with_padding=with_padding,
                                        padding=args.padding,
                                        total_entries_scale=args.total_entries_scale,
                                    )
                                else:
                                    ret = (
                                        test_causal_conv1d_update_cache_seqlens_decode(
                                            batch=batch,
                                            dim=dim,
                                            width=width,
                                            seqlen=seqlen,
                                            has_bias=has_bias,
                                            silu_activation=silu_activation,
                                            itype=itype,
                                            has_cache_seqlens=has_cache_seqlens,
                                        )
                                    )
                                ret["mode"] = mode
                                df.append(ret)

df = pd.DataFrame(df)
# Remove duplicate rows (e.g. from benchmark/log_args)
dedup_cols = ["batch", "dim", "width", "seqlen", "has_bias", "silu_activation", "mode"]
for c in ["with_padding", "has_cache_seqlens"]:
    if c in df.columns:
        dedup_cols.append(c)
df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
try:
    df_md = df.to_markdown(index=False)
except ImportError:
    df_md = df.to_string(index=False)
aiter.logger.info("causal_conv1d_update summary (modes=%s):\n%s", mode_list, df_md)
