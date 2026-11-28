# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.attn_res import (
    _attn_res_consume_kernel,
    _attn_res_kernel,
    _attn_res_produce_kernel,
)
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)


def _config(key, T=0, residual_added=False):
    cfg = load_config_json(
        f"{resolve_config_dir('fusions', 'ATTN_RES', backend='gluon')}/DEFAULT.json"
    )
    if key == "ahead_k":
        return cfg[key]
    kind = (
        "plain" if key[0] == "n" else "consume" if key.startswith("cons") else "produce"
    )
    key = "h" if key[0] == "h" else key
    tuned = select_leq_config(cfg["default"], T, prefix="T_LEQ_").get(key, {})
    if residual_added:
        tuned = {
            **tuned,
            **select_leq_config(cfg["residual_added"], T, prefix="T_LEQ_").get(key, {}),
        }
    return {**cfg[kind], **tuned}


def _out(prefix, quant_dtype):
    out = torch.empty_like(
        prefix, dtype=quant_dtype or prefix.dtype, memory_format=torch.contiguous_format
    )
    if quant_dtype is None:
        return out, None, 0.0
    scale = torch.empty((prefix.shape[0], 1), device=prefix.device, dtype=torch.float32)
    return out, scale, float(torch.finfo(quant_dtype).max)


def _pieces(nch, chunk):
    ps, pw, start = [], [], 0
    for bit in reversed(range(nch.bit_length())):
        if nch >> bit & 1:
            ps.append(start)
            pw.append(chunk << bit)
            start += chunk << bit
    return ps, pw


def _block_tiles(num_blocks, hidden, nch, chunk, lds_max):
    nbp = triton.next_power_of_2(max(num_blocks, 1))
    bd = triton.next_power_of_2(hidden)
    if nbp * bd * 2 * 2 > lds_max // 2:
        return nbp, bd, _pieces(nch, chunk)
    return nbp, bd, ([0], [bd])


def _check(
    prefix, delta, blocks, num_blocks, block_write_idx, quant_dtype, residual_added
):
    assert (
        prefix.stride(-1) == 1
        and blocks.stride(-1) == 1
        and blocks.stride(1) == prefix.shape[1]
    )
    assert num_blocks <= blocks.shape[1] and block_write_idx < blocks.shape[1]
    assert block_write_idx < 0 or block_write_idx >= num_blocks
    assert delta is None or delta.stride(-1) == 1
    assert not (residual_added and delta is not None), "residual_added: pass delta=None"
    assert quant_dtype in (None, torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def attn_res_produce(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    w_next=None,
    quant_dtype=None,
    store_prefix=True,
    residual_added=False,
    launch=None,
):
    """One call; with w_next [K, H] it also returns the mixtures [T, K, H] and stats [T, K, 2] of the next K calls."""
    _check(
        prefix, delta, blocks, num_blocks, block_write_idx, quant_dtype, residual_added
    )
    T, H = prefix.shape
    ahead = w_next is not None and w_next.shape[0] > 0
    assert not ahead or (
        num_blocks > 0 and w_next.dtype == torch.float32 and w_next.is_contiguous()
    )
    k = w_next.shape[0] if ahead else 0
    out, scale, qmax = _out(prefix, quant_dtype)
    mix = torch.empty(
        (T, k, H) if ahead else (1,), device=prefix.device, dtype=torch.float32
    )
    stats = torch.empty(
        (T, k, 2) if ahead else (1,), device=prefix.device, dtype=torch.float32
    )
    if launch is None:
        launch = _config(f"h{num_blocks}" if ahead else "produce", T, residual_added)
    nslot, num_warps, ctas_per_cu = launch["nslot"], launch["num_warps"], launch["cpc"]
    epl = 32 // num_warps
    chunk = 32 * epl * num_warps
    nch = triton.cdiv(H, chunk)
    lds_max = triton.runtime.driver.active.utils.get_device_properties(
        prefix.device.index
    )["max_shared_mem"]
    nbp, bd, (ps, pw) = _block_tiles(num_blocks, H, nch, chunk, lds_max)
    lds = (
        nslot
        * 2
        * ((1 + (delta is not None)) * bd + (nbp * sum(pw) if num_blocks else 0))
    )
    cpc = ctas_per_cu or max(1, min(4, lds_max // lds))
    cus = torch.cuda.get_device_properties(prefix.device).multi_processor_count
    ks = launch.get("ks", 1) if T * launch.get("ks", 1) <= cus else 1
    grid = min(T, cpc * cus) * ks
    _attn_res_produce_kernel[(grid,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        scale,
        w_next if ahead else norm_weight,
        mix,
        stats,
        T,
        k,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        out.stride(0),
        eps,
        output_norm_eps,
        HIDDEN=H,
        NUM_BLOCKS=num_blocks,
        BLOCK_WRITE_IDX=max(block_write_idx, 0),
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        STORE_PREFIX=store_prefix,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        QUANT_MAX=qmax,
        NBP=nbp,
        EPL=epl,
        C=chunk,
        NCH=nch,
        BD=bd,
        PS=tuple(ps),
        PW=tuple(pw),
        NSLOT=nslot,
        AHEAD=ahead,
        NUM_WARPS=num_warps,
        EARLY=launch.get("early", 0),
        UB=launch["ub"],
        KSPLIT=ks,
        num_warps=num_warps,
    )
    res = (out,) if scale is None else (out, scale)
    return res + ((mix, stats) if ahead else ())


def attn_res_consume(
    prefix,
    delta,
    blocks,
    mix,
    stats,
    norm_weight,
    qk_weight,
    output_norm_weight,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
    store_prefix=True,
    residual_added=False,
):
    """One call that reads one mixture row mix [T, H] and stats [T, 2] instead of the blocks."""
    T, H = prefix.shape
    assert not (residual_added and delta is not None), "residual_added: pass delta=None"
    c = _config("consf" if quant_dtype is not None else "consb", T, residual_added)
    nslot, num_warps = c["nslot"], c["num_warps"]
    assert prefix.stride(-1) == 1 and (delta is None or delta.stride(-1) == 1)
    assert mix.shape == prefix.shape and mix.stride(-1) == 1 and stats.stride(-1) == 1
    out, scale, qmax = _out(prefix, quant_dtype)
    epl = 32 // num_warps
    chunk = 32 * epl * num_warps
    assert H % chunk == 0
    nch = H // chunk
    ps, pw = _pieces(nch, chunk)
    pbd = triton.next_power_of_2(H)
    cpc = c["cpc"] if delta is not None else c["cpc_no_delta"]
    lds_max = triton.runtime.driver.active.utils.get_device_properties(
        prefix.device.index
    )["max_shared_mem"]
    assert cpc * nslot * ((1 + (delta is not None)) * 2 + 4) * pbd <= lds_max, "LDS"
    grid = min(
        T, cpc * torch.cuda.get_device_properties(prefix.device).multi_processor_count
    )
    _attn_res_consume_kernel[(grid,)](
        prefix,
        delta,
        blocks,
        mix,
        stats,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        scale,
        T,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        mix.stride(0),
        stats.stride(0),
        out.stride(0),
        eps,
        output_norm_eps,
        HIDDEN=H,
        BLOCK_WRITE_IDX=max(block_write_idx, 0),
        WRITE_BLOCK=block_write_idx >= 0,
        HAS_DELTA=delta is not None,
        STORE_PREFIX=store_prefix,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        QUANT_MAX=qmax,
        EPL=epl,
        C=chunk,
        NCH=nch,
        PS=tuple(ps),
        PW=tuple(pw),
        NSLOT=nslot,
        PBD=pbd,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out if scale is None else (out, scale)


def attn_res(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
    store_prefix=True,
    residual_added=False,
):
    """One AttnRes call. Returns out, or (out, scale) with quant_dtype."""
    _check(
        prefix, delta, blocks, num_blocks, block_write_idx, quant_dtype, residual_added
    )
    T, H = prefix.shape
    if T == 0:
        out, scale, _ = _out(prefix, quant_dtype)
        return out if scale is None else (out, scale)
    c = _config(
        f"n{num_blocks}{'f' if quant_dtype is not None else 'b'}", T, residual_added
    )
    nslot, early, ub, num_warps, cpc = (
        c["nslot"],
        c["early"],
        c["ub"],
        c["num_warps"],
        c["cpc"],
    )
    if c["kernel"] == "produce":
        res = attn_res_produce(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            block_write_idx,
            eps,
            output_norm_eps,
            quant_dtype=quant_dtype,
            store_prefix=store_prefix,
            residual_added=residual_added,
            launch=c,
        )
        return res[0] if quant_dtype is None else res
    out, scale, qmax = _out(prefix, quant_dtype)
    epl = 32 // num_warps
    chunk = 32 * epl * num_warps
    nch = triton.cdiv(H, chunk)
    lds_max = triton.runtime.driver.active.utils.get_device_properties(
        prefix.device.index
    )["max_shared_mem"]
    nbp, bd, (ps, pw) = _block_tiles(num_blocks, H, nch, chunk, lds_max)
    lds = (
        nslot
        * 2
        * ((1 + (delta is not None)) * bd + (nbp * sum(pw) if num_blocks else 0))
    )
    cpc = cpc or max(1, min(4, lds_max // lds))
    grid = min(
        T, cpc * torch.cuda.get_device_properties(prefix.device).multi_processor_count
    )
    _attn_res_kernel[(grid,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        scale,
        T,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        out.stride(0),
        eps,
        output_norm_eps,
        HIDDEN=H,
        NUM_BLOCKS=num_blocks,
        BLOCK_WRITE_IDX=max(block_write_idx, 0),
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        STORE_PREFIX=store_prefix,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        QUANT_MAX=qmax,
        NBP=nbp,
        EPL=epl,
        C=chunk,
        NCH=nch,
        BD=bd,
        PS=tuple(ps),
        PW=tuple(pw),
        NSLOT=nslot,
        NUM_WARPS=num_warps,
        EARLY=early,
        UB=ub,
        num_warps=num_warps,
    )
    return out if scale is None else (out, scale)


def attn_res_fwd(
    query, residuals, rms_weight, output_rms_weight, rms_eps, *, hidden=7168
):
    """fusions.attn_res_fwd, packed layout. Returns None for inputs these kernels do not serve."""
    if not (
        isinstance(residuals, torch.Tensor)
        and residuals.dim() >= 3
        and residuals.is_contiguous()
    ):
        return None
    L, D = residuals.shape[-2:]
    if not (
        residuals.is_cuda
        and residuals.dtype == torch.bfloat16
        and D == hidden == query.numel()
        and L <= 9
    ):
        return None
    x = residuals.view(-1, L, D)
    onw = None if output_rms_weight is None else output_rms_weight.flatten()
    out = attn_res(
        x[:, L - 1],
        None,
        x,
        rms_weight.flatten(),
        query.flatten(),
        onw,
        L - 1,
        -1,
        rms_eps,
        rms_eps,
        store_prefix=False,
    )
    return out.view(residuals.shape[:-2] + (D,))


def attn_res_gate(
    prefix, blocks, score_weight, eps, output_rms_weight, output_rms_eps, *, hidden=7168
):
    """fusions.attn_res_gate without add, close_block or quant. Returns None for inputs these kernels do not serve."""
    if not (
        prefix.dim() == 2
        and blocks.dim() == 3
        and prefix.stride(-1) == blocks.stride(-1) == 1
    ):
        return None
    if not (
        prefix.is_cuda
        and prefix.dtype == blocks.dtype == torch.bfloat16
        and prefix.shape[-1] == hidden
    ):
        return None
    if blocks.stride(-2) != hidden or blocks.shape[-2] > 8:
        return None
    args = (
        prefix,
        None,
        blocks,
        score_weight.flatten(),
        None,
        output_rms_weight,
        blocks.shape[-2],
        -1,
        eps,
    )
    return attn_res(*args, output_rms_eps, store_prefix=False)


def attn_res_schedule(calls, use_ahead=True, *, residual_added=False):
    """Mix-ahead plan and state for one model forward; pass it to attn_res_inplace with each call's index."""
    ahead_k = {int(n): k for n, k in _config("ahead_k").items()}
    role, src, groups, i = ["plain"] * len(calls), [None] * len(calls), [], 0
    while i < len(calls):
        n, j = calls[i]["n"], i
        while j < len(calls) and calls[j]["n"] == n:
            j += 1
        for g in (
            range(i, j, ahead_k.get(n, 0) + 1) if use_ahead and n in ahead_k else ()
        ):
            e = min(j, g + ahead_k[n] + 1)
            if e - g >= 2:
                role[g], src[g] = "produce", (len(groups), -1)
                w = [
                    calls[c]["norm_weight"].float() * calls[c]["qk_weight"].float()
                    for c in range(g + 1, e)
                ]
                groups.append((g + 1, e, torch.stack(w).contiguous()))
                for c in range(g + 1, e):
                    role[c], src[c] = "consume", (len(groups) - 1, c - g - 1)
        i = j
    return {
        "calls": calls,
        "residual_added": residual_added,
        "role": role,
        "src": src,
        "groups": groups,
        "mix": {},
        "last": -1,
    }


def attn_res_inplace(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
    store_prefix=True,
    residual_added=False,
    schedule=None,
    call_idx=None,
    hidden=7168,
):
    """In-place AttnRes call, through the schedule when it names this call. None for unsupported inputs."""
    if not (
        prefix.is_cuda
        and prefix.dtype == blocks.dtype == torch.bfloat16
        and prefix.shape[-1] == hidden
    ):
        return None
    if num_blocks > 8:
        return None
    i = call_idx
    c = (
        schedule["calls"][i]
        if schedule is not None and i is not None and 0 <= i < len(schedule["calls"])
        else None
    )
    if c is None or (c["n"], c["block_write_idx"], c["quant_dtype"]) != (
        num_blocks,
        block_write_idx,
        quant_dtype,
    ):
        args = (
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            block_write_idx,
        )
        return attn_res(
            *args,
            eps,
            output_norm_eps,
            quant_dtype=quant_dtype,
            store_prefix=store_prefix,
            residual_added=residual_added,
        )
    if i <= schedule["last"]:
        schedule["mix"].clear()
    schedule["last"] = i
    T = prefix.shape[0]
    role, src, groups = schedule["role"], schedule["src"], schedule["groups"]
    w = (c["norm_weight"], c["qk_weight"], c["output_norm_weight"])
    kw = {
        "quant_dtype": c["quant_dtype"],
        "store_prefix": c["store_prefix"],
        "residual_added": schedule["residual_added"],
    }
    args = (c["block_write_idx"], c["eps"], c["output_norm_eps"])
    r = role[i]
    if r == "consume":
        g, s = src[i]
        ent = schedule["mix"].get(g)
        if ent is None or ent[2] != blocks.data_ptr() or ent[3] != T:
            r = "plain"
    if r == "plain":
        out = attn_res(prefix, delta, blocks, *w, c["n"], *args, **kw)
    elif r == "produce":
        g = src[i][0]
        res = attn_res_produce(
            prefix, delta, blocks, *w, c["n"], *args, w_next=groups[g][2], **kw
        )
        schedule["mix"][g] = (res[-2], res[-1], blocks.data_ptr(), T)
        out = res[0] if c["quant_dtype"] is None else (res[0], res[1])
    else:
        mix, stats = ent[0], ent[1]
        out = attn_res_consume(
            prefix, delta, blocks, mix[:, s], stats[:, s], *w, *args, **kw
        )
        if s == mix.shape[1] - 1:
            del schedule["mix"][g]
    return out
