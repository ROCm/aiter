# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from typing import Optional

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest

HEAD_DIM = 128


def make_cos_sin_cache(
    max_pos: int, rotary_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    base = 5_000_000.0
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_pos, dtype=torch.float32, device="cuda")
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps) * (1.0 + weight.float())


def apply_rope_neox_partial(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    half = rotary_dim // 2
    cos_sin = cos_sin_cache[positions].float()
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)

    rot = x[..., :rotary_dim]
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    out = x.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def norm_rope_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    normed = gemma_rmsnorm(x.float(), weight, eps)
    return apply_rope_neox_partial(normed, positions, cos_sin_cache, rotary_dim).to(
        dtype
    )


def make_case(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    block_size: int = 16,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    num_index_heads: int = 4,
    rotary_dim: int = 64,
    seed: int = 123,
):
    torch.manual_seed(seed)
    eps = 1e-6
    max_pos = 4096

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    iq_w = (
        torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
        if num_index_heads > 0
        else None
    )
    ik_w = (
        torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
        if num_index_heads > 0
        else None
    )
    cos_sin = make_cos_sin_cache(max_pos, rotary_dim, dtype)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device="cuda"
    )

    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    iq_size = num_index_heads * HEAD_DIM
    ik_size = HEAD_DIM if num_index_heads > 0 else 0
    qkv = torch.randn(
        num_tokens,
        q_size + 2 * kv_size + iq_size + ik_size,
        dtype=dtype,
        device="cuda",
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]
    index_slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]

    return {
        "qkv": qkv,
        "q_norm_weight": q_w,
        "k_norm_weight": k_w,
        "index_q_norm_weight": iq_w,
        "index_k_norm_weight": ik_w,
        "cos_sin_cache": cos_sin,
        "positions": positions,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_index_heads": num_index_heads,
        "rotary_dim": rotary_dim,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "slot_mapping": slot_mapping,
        "index_slot_mapping": index_slot_mapping,
        "sizes": (q_size, kv_size, kv_size, iq_size, ik_size),
        "eps": eps,
        "dtype": dtype,
    }


def make_refs(case: dict, qkv_orig: torch.Tensor):
    q_size, kv_size, _, iq_size, ik_size = case["sizes"]
    split_sizes = [q_size, kv_size, kv_size]
    if case["num_index_heads"] > 0:
        split_sizes.extend([iq_size, ik_size])
    splits = qkv_orig.split(split_sizes, dim=-1)
    q_in, k_in, v_in = splits[:3]
    num_tokens = qkv_orig.size(0)

    q_ref = norm_rope_ref(
        q_in.view(num_tokens, case["num_heads"], HEAD_DIM),
        case["q_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["rotary_dim"],
        case["eps"],
        case["dtype"],
    ).view(num_tokens, q_size)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, case["num_kv_heads"], HEAD_DIM),
        case["k_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["rotary_dim"],
        case["eps"],
        case["dtype"],
    )
    v_ref = v_in.view(num_tokens, case["num_kv_heads"], HEAD_DIM)

    refs = {"q": q_ref, "k": k_ref, "v": v_ref}
    if case["num_index_heads"] > 0:
        iq_in, ik_in = splits[3:]
        refs["index_q"] = norm_rope_ref(
            iq_in.view(num_tokens, case["num_index_heads"], HEAD_DIM),
            case["index_q_norm_weight"],
            case["positions"],
            case["cos_sin_cache"],
            case["rotary_dim"],
            case["eps"],
            case["dtype"],
        ).view(num_tokens, iq_size)
        refs["index_k"] = norm_rope_ref(
            ik_in.view(num_tokens, 1, HEAD_DIM),
            case["index_k_norm_weight"],
            case["positions"],
            case["cos_sin_cache"],
            case["rotary_dim"],
            case["eps"],
            case["dtype"],
        ).view(num_tokens, HEAD_DIM)
    return refs


def split_qkv(case: dict, qkv: torch.Tensor):
    q_size, kv_size, _, iq_size, ik_size = case["sizes"]
    split_sizes = [q_size, kv_size, kv_size]
    if case["num_index_heads"] > 0:
        split_sizes.extend([iq_size, ik_size])
    return qkv.split(split_sizes, dim=-1)


def make_insert_outputs(case: dict, *, kv_cache_dtype: Optional[torch.dtype] = None):
    q_size, _, _, iq_size, _ = case["sizes"]
    q_out = torch.empty(case["qkv"].size(0), q_size, dtype=case["dtype"], device="cuda")
    index_q_out = torch.empty(
        case["qkv"].size(0), iq_size, dtype=case["dtype"], device="cuda"
    )
    kv_cache = torch.zeros(
        case["num_blocks"],
        2,
        case["block_size"],
        case["num_kv_heads"],
        HEAD_DIM,
        dtype=kv_cache_dtype or case["dtype"],
        device="cuda",
    )
    index_cache = torch.zeros(
        case["num_blocks"],
        case["block_size"],
        HEAD_DIM,
        dtype=case["dtype"],
        device="cuda",
    )
    return q_out, index_q_out, kv_cache, index_cache


def gather_cache_outputs(
    case: dict,
    kv_cache: torch.Tensor,
    index_cache: Optional[torch.Tensor],
    *,
    index_slot_mapping: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
):
    index_slots = (
        index_slot_mapping if index_slot_mapping is not None else case["slot_mapping"]
    )
    k_outs = []
    v_outs = []
    index_k_outs = []

    for token in range(case["qkv"].size(0)):
        slot = case["slot_mapping"][token].item()
        block, offset = divmod(slot, case["block_size"])
        k_out = kv_cache[block, 0, offset]
        v_out = kv_cache[block, 1, offset]
        if k_scale is not None and v_scale is not None:
            k_out = maybe_view_fp8(k_out)
            v_out = maybe_view_fp8(v_out)
            k_out = k_out.float() * k_scale
            v_out = v_out.float() * v_scale
        k_outs.append(k_out)
        v_outs.append(v_out)

        if index_cache is not None:
            index_slot = index_slots[token].item()
            index_k_outs.append(index_cache.view(-1, HEAD_DIM)[index_slot])

    index_k = torch.stack(index_k_outs) if index_k_outs else None
    return torch.stack(k_outs), torch.stack(v_outs), index_k


def check_close(actual, expected, *, msg: str, rtol: float, atol: float):
    err = checkAllclose(actual, expected, msg=msg, rtol=rtol, atol=atol)
    if err != 0:
        raise AssertionError(f"{msg} mismatch ratio: {err}")


def fp8_cache_dtype() -> Optional[torch.dtype]:
    if dtypes.fp8 is not torch.uint8:
        return dtypes.fp8
    return getattr(torch, "float8_e4m3fnuz", getattr(torch, "float8_e4m3fn", None))


def maybe_view_fp8(x: torch.Tensor) -> torch.Tensor:
    if x.dtype is not torch.uint8:
        return x
    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    return x.view(fp8_dtype)


def fp8_cache_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    return (x.float() / scale).to(fp8_dtype).float() * scale


@perftest(num_iters=10, num_warmup=1, num_rotate_args=1)
def run_fused_qknorm_idxrqknorm(
    case: dict,
    mode: str,
    use_index_slot_mapping: bool,
    use_fp8_kv_cache: bool,
):
    qkv = case["qkv"].clone()
    if mode == "inplace":
        aiter.fused_qknorm_idxrqknorm(
            qkv,
            case["q_norm_weight"],
            case["k_norm_weight"],
            case["cos_sin_cache"],
            case["positions"],
            case["num_heads"],
            case["num_kv_heads"],
            case["rotary_dim"],
            case["eps"],
            case["index_q_norm_weight"],
            case["index_k_norm_weight"],
            case["num_index_heads"],
        )
        return qkv

    use_uint8_kv_cache = mode == "fp8_kv_cache_uint8"
    kv_cache_dtype = None
    if use_fp8_kv_cache:
        kv_cache_dtype = torch.uint8 if use_uint8_kv_cache else dtypes.fp8
    q_out, index_q_out, kv_cache, index_cache = make_insert_outputs(
        case,
        kv_cache_dtype=kv_cache_dtype,
    )
    index_slot_mapping = case["index_slot_mapping"] if use_index_slot_mapping else None
    index_q_out_arg = index_q_out
    index_cache_arg = index_cache
    if case["num_index_heads"] == 0:
        index_q_out_arg = None
        index_cache_arg = None
        index_slot_mapping = None
    k_scale = None
    v_scale = None
    kv_cache_dtype_arg = "auto"
    if use_fp8_kv_cache:
        k_scale = torch.tensor(0.75, dtype=torch.float32, device="cuda")
        v_scale = torch.tensor(1.25, dtype=torch.float32, device="cuda")
        kv_cache_dtype_arg = "fp8_e4m3"

    aiter.fused_qknorm_idxrqknorm(
        qkv,
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        case["num_heads"],
        case["num_kv_heads"],
        case["rotary_dim"],
        case["eps"],
        case["index_q_norm_weight"],
        case["index_k_norm_weight"],
        case["num_index_heads"],
        case["slot_mapping"],
        kv_cache,
        index_cache_arg,
        case["block_size"],
        q_out,
        index_q_out_arg,
        index_slot_mapping,
        kv_cache_dtype=kv_cache_dtype_arg,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return (
        q_out,
        index_q_out_arg,
        kv_cache,
        index_cache_arg,
        index_slot_mapping,
        k_scale,
        v_scale,
    )


@benchmark()
def test_fused_qknorm_idxrqknorm(
    mode: str,
    dtype: torch.dtype,
    num_tokens: int,
    block_size: int,
    rotary_dim: int,
    num_index_heads: int = 4,
):
    use_fp8_kv_cache = mode.startswith("fp8_kv_cache")
    if use_fp8_kv_cache and fp8_cache_dtype() is None:
        aiter.logger.info("Skip fp8_kv_cache: torch FP8 dtype is unavailable")
        return {
            "dtype": str(dtype),
            "num_tokens": num_tokens,
            "block_size": block_size,
            "rotary_dim": rotary_dim,
            "num_index_heads": num_index_heads,
            "status": "skipped",
        }

    case = make_case(
        dtype=dtype,
        num_tokens=num_tokens,
        block_size=block_size,
        num_index_heads=num_index_heads,
        rotary_dim=rotary_dim,
    )
    refs = make_refs(case, case["qkv"])
    rtol = 1e-2
    atol = 1e-2
    use_index_slot_mapping = mode != "slot_mapping_fallback"
    result, avg_opt = run_fused_qknorm_idxrqknorm(
        case,
        mode,
        use_index_slot_mapping,
        use_fp8_kv_cache,
    )

    info = (
        f"mode:{mode}, dtype:{dtype}, tokens:{num_tokens}, block:{block_size}, "
        f"rotary:{rotary_dim}, index_heads:{num_index_heads}"
    )
    msg = f"[perf] === {info} === fused_kernel avg: {avg_opt:<8.2f} us "

    if mode == "inplace":
        qkv_out = result
        q_out, k_out, v_out, *index_outs = split_qkv(case, qkv_out)
        _, _, v_orig, *_ = split_qkv(case, case["qkv"])
        check_close(q_out, refs["q"], msg=f"{msg}(q)", rtol=rtol, atol=atol)
        check_close(
            k_out.view(num_tokens, case["num_kv_heads"], HEAD_DIM),
            refs["k"],
            msg=f"{msg}(k)",
            rtol=rtol,
            atol=atol,
        )
        check_close(v_out, v_orig, msg=f"{msg}(v)", rtol=0, atol=0)
        if num_index_heads > 0:
            index_q_out, index_k_out = index_outs
            check_close(
                index_q_out,
                refs["index_q"],
                msg=f"{msg}(index_q)",
                rtol=rtol,
                atol=atol,
            )
            check_close(
                index_k_out,
                refs["index_k"],
                msg=f"{msg}(index_k)",
                rtol=rtol,
                atol=atol,
            )
    else:
        (
            q_out,
            index_q_out,
            kv_cache,
            index_cache,
            index_slot_mapping,
            k_scale,
            v_scale,
        ) = result
        check_close(q_out, refs["q"], msg=f"{msg}(q_out)", rtol=rtol, atol=atol)
        if num_index_heads > 0:
            check_close(
                index_q_out,
                refs["index_q"],
                msg=f"{msg}(index_q_out)",
                rtol=rtol,
                atol=atol,
            )

        k_out, v_out, index_k_out = gather_cache_outputs(
            case,
            kv_cache,
            index_cache,
            index_slot_mapping=index_slot_mapping,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        if use_fp8_kv_cache:
            k_ref = fp8_cache_ref(refs["k"], k_scale)
            v_ref = fp8_cache_ref(refs["v"], v_scale)
        else:
            k_ref = refs["k"]
            v_ref = refs["v"]
        check_close(k_out, k_ref, msg=f"{msg}(k_cache)", rtol=rtol, atol=atol)
        check_close(v_out, v_ref, msg=f"{msg}(v_cache)", rtol=rtol, atol=atol)
        if num_index_heads > 0:
            check_close(
                index_k_out,
                refs["index_k"],
                msg=f"{msg}(index_cache)",
                rtol=rtol,
                atol=atol,
            )

    return {
        "mode": mode,
        "dtype": str(dtype),
        "num_tokens": num_tokens,
        "block_size": block_size,
        "rotary_dim": rotary_dim,
        "num_index_heads": num_index_heads,
        "fused_kernel_us": avg_opt,
        "status": "passed",
    }


DEFAULT_CASES = [
    ("insert", "bf16", 1, 16, 64, 4),
    ("insert", "bf16", 17, 16, 64, 4),
    ("insert", "bf16", 19, 16, 96, 4),
    ("insert", "fp16", 33, 8, 128, 4),
    ("dense_insert", "bf16", 13, 8, 96, 0),
    ("slot_mapping_fallback", "bf16", 9, 8, 64, 4),
    ("inplace", "bf16", 11, 16, 64, 0),
    ("inplace", "bf16", 11, 16, 64, 4),
    ("inplace", "fp16", 11, 16, 64, 4),
    ("fp8_kv_cache", "bf16", 17, 16, 64, 4),
    ("fp8_kv_cache_uint8", "bf16", 17, 16, 64, 4),
]

l_mode = [
    "insert",
    "dense_insert",
    "slot_mapping_fallback",
    "inplace",
    "fp8_kv_cache",
    "fp8_kv_cache_uint8",
]
l_dtype = ["fp16", "bf16"]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test fused_qknorm_idxrqknorm op",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=l_mode,
    nargs="*",
    default=None,
    help="Mode(s) to test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="*",
    default=None,
    help="Data type(s). e.g. -d bf16 or -d bf16 fp16",
)
parser.add_argument("--num_tokens", type=int, nargs="*", default=None)
parser.add_argument("--block_size", type=int, nargs="*", default=None)
parser.add_argument("--rotary_dim", type=int, nargs="*", default=None)
parser.add_argument("--num_index_heads", type=int, nargs="*", default=None)
args = parser.parse_args()

selected_cases = []
for (
    mode,
    dtype_name,
    num_tokens,
    block_size,
    rotary_dim,
    num_index_heads,
) in DEFAULT_CASES:
    if args.mode is not None and mode not in args.mode:
        continue
    if args.dtype is not None and dtype_name not in args.dtype:
        continue
    if args.num_tokens is not None and num_tokens not in args.num_tokens:
        continue
    if args.block_size is not None and block_size not in args.block_size:
        continue
    if args.rotary_dim is not None and rotary_dim not in args.rotary_dim:
        continue
    if args.num_index_heads is not None and num_index_heads not in args.num_index_heads:
        continue
    selected_cases.append(
        (mode, dtype_name, num_tokens, block_size, rotary_dim, num_index_heads)
    )

df = []
for (
    mode,
    dtype_name,
    num_tokens,
    block_size,
    rotary_dim,
    num_index_heads,
) in selected_cases:
    ret = test_fused_qknorm_idxrqknorm(
        mode=mode,
        dtype=dtypes.d_dtypes[dtype_name],
        num_tokens=num_tokens,
        block_size=block_size,
        rotary_dim=rotary_dim,
        num_index_heads=num_index_heads,
    )
    df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("fused_qknorm_idxrqknorm summary (markdown):\n%s", df_md)
