# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 unified attention: public dispatch and focused direct-launch regressions.

Timings use warm, fixed buffers; TB/s is logical traffic, not measured HBM bandwidth.
The run-only check covers warm JIT-cache reuse, not wheel AOT packaging.
"""

import argparse
import itertools
import os
import sys
from functools import partial
from unittest import mock

import pandas as pd
import torch

# Plain-script CI invocation must select this checkout, not another editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
import aiter.ops.triton.attention.unified_attention as ua
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

PAGE, H, HKV, D = 64, 64, 4, 128


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(x.device)


def ref_paged_attn(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    out_dtype=torch.float32,
    sliding_window=None,
    soft_cap=None,
    sinks=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    output_scale=None,
    causal=1,
):
    # Local copy of the established paged oracle; never import a test suite.
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    start_idx = 0
    query = query.to(torch.float32)
    key_cache = key_cache.to(torch.float32)
    value_cache = value_cache.to(torch.float32)
    if q_descale is not None:
        query = query * q_descale
    if k_descale is not None:
        key_cache = key_cache * k_descale
    if v_descale is not None:
        value_cache = value_cache * v_descale
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if causal:
            attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks[:, None, None].repeat_interleave(attn.shape[-2], dim=-2)
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sinks is not None:
            attn = attn[..., :-1]
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
        start_idx += query_len
    out = torch.cat(outputs, dim=0)
    if output_scale is not None:
        out = out / output_scale
    return out.to(out_dtype)


def make_case(query_lens, kv_lens, dtype, causal=True, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [max(0, (n + PAGE - 1) // PAGE) for n in kv_lens]
    n_pages = max(1, sum(pages))
    q, qs = q8(
        torch.randn(
            sum(query_lens), H, D, device="cuda", dtype=dtypes.bf16, generator=g
        )
    )
    k, ks = q8(
        torch.randn(
            n_pages, PAGE, HKV, D, device="cuda", dtype=dtypes.bf16, generator=g
        )
    )
    v, vs = q8(
        torch.randn(
            n_pages, PAGE, HKV, D, device="cuda", dtype=dtypes.bf16, generator=g
        )
    )
    perm = torch.randperm(
        n_pages, generator=torch.Generator().manual_seed(seed + 1)
    ).tolist()
    bt = torch.zeros(len(pages), max(1, max(pages)), device="cuda", dtype=torch.int32)
    offset = 0
    for i, count in enumerate(pages):
        bt[i, :count] = torch.tensor(
            perm[offset : offset + count], device="cuda", dtype=torch.int32
        )
        offset += count
    cu_q = torch.tensor(
        [0] + list(itertools.accumulate(query_lens)), device="cuda", dtype=torch.int32
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.full(q.shape, float("nan"), device="cuda", dtype=dtype),
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": max(query_lens),
        "seqused_k": torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": D**-0.5,
        "causal": causal,
        "window_size": (-1, -1),
        "block_table": bt,
        "softcap": 0,
        "q_descale": qs,
        "k_descale": ks,
        "v_descale": vs,
    }


def reference(case, query_lens, kv_lens):
    return ref_paged_attn(
        case["q"],
        case["k"],
        case["v"],
        query_lens,
        kv_lens,
        case["block_table"],
        case["softmax_scale"],
        q_descale=case["q_descale"],
        k_descale=case["k_descale"],
        v_descale=case["v_descale"],
        causal=case["causal"],
        soft_cap=case["softcap"],
        sliding_window=(
            None if case["window_size"][0] < 0 else case["window_size"][0] + 1
        ),
    )


def shuffle_kv(k, v):
    nb, page, hkv, d = k.shape
    k = k.permute(0, 2, 3, 1).contiguous().view(nb, hkv, d // 16, 16, page)
    v = v.permute(0, 2, 3, 1).contiguous().view(nb, hkv, d, page // 16, 16)
    return k.permute(0, 1, 2, 4, 3).contiguous(), v.permute(0, 1, 3, 2, 4).contiguous()


def direct_candidate(case, kv_lens, splits=1, packed=False):
    from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import (
        dualwave_splitk_workspace_elems,
    )
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module,
    )

    mod = build_flash_attn_dualwave_swp_fp8_module(
        num_heads=H,
        head_dim=D,
        causal=case["causal"],
        dtype_str="fp8",
        out_dtype_str="bf16" if case["out"].dtype == dtypes.bf16 else "f16",
        num_kv_heads=HKV,
        paged=True,
        varlen=True,
        num_kv_splits=splits,
        gqa_pack_m=packed,
        kv_cache_layout="vectorized" if case["k"].ndim == 5 else "linear",
    )
    b, max_q = len(kv_lens), case["max_seqlen_q"]
    cu_kv = torch.tensor(
        [0] + list(itertools.accumulate(kv_lens)), device="cuda", dtype=torch.int32
    )
    ws = None
    if splits > 1:
        ws = torch.zeros(
            dualwave_splitk_workspace_elems(b, H, max_q, splits, D),
            device="cuda",
            dtype=dtypes.fp32,
        )

    def launch():
        mod(
            case["q"].reshape(-1),
            case["k"].reshape(-1),
            case["v"].reshape(-1),
            case["out"].reshape(-1),
            b,
            max_q,
            workspace=ws,
            cu_seqlens_q=case["cu_seqlens_q"],
            cu_seqlens_kv=cu_kv,
            block_table=case["block_table"].reshape(-1),
            block_table_stride=case["block_table"].stride(0),
            q_descale=case["q_descale"],
            k_descale=case["k_descale"],
            v_descale=case["v_descale"],
            stream=torch.cuda.current_stream().cuda_stream,
        )
        return case["out"]

    return launch


def compare(want, got, atol, name):
    want, got = want.float(), got.float()
    err = checkAllclose(want, got, rtol=0, atol=atol, tol_err_ratio=0, msg=name)
    assert err == 0, f"{name}: {err:.3%} elements exceed atol={atol}"
    return err


def measure(candidates, case, want, query_lens, kv_lens, atol=None, bad_rows=False):
    # Preserve the old adapter's 8% global-scale gate, not element-relative rtol.
    if atol is None:
        atol = 0.08 * want.abs().max().item()
    flops = 4 * H * D * sum(q * k for q, k in zip(query_lens, kv_lens))
    nbytes = sum(query_lens) * H * D * (1 + case["out"].element_size())
    nbytes += 2 * sum(kv_lens) * HKV * D
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        got = fn()
        err = compare(want, got, atol, name)
        if bad_rows:
            bad = int(
                (~torch.isclose(want.float(), got.float(), rtol=0, atol=atol))
                .any(dim=-1)
                .any(dim=-1)
                .sum()
                .item()
            )
            assert bad == 0, f"{name}: {bad} bad rows"
            ret[f"{name} bad_rows"] = bad
        _, us = run_perftest(fn, num_rotate_args=1)
        assert us > 0, f"{name}: empty timing"
        compare(want, got, atol, f"{name} after timing")
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_prefill(dtype, causal):
    query_lens, kv_lens = [512, 256], [512, 256]
    case = make_case(query_lens, kv_lens, dtype, causal)
    want = reference(case, query_lens, kv_lens)
    candidates = {"flydsl": partial(ua.unified_attention, **case, backend="flydsl")}
    return measure(candidates, case, want, query_lens, kv_lens)


@benchmark()
def test_decode(dtype, depth, layout):
    query_lens = [1] * 8
    kv_lens = [depth, max(1, depth // 2 + 3), depth, 65, depth, 1, depth, 0]
    case = make_case(query_lens, kv_lens, dtype)
    want = reference(case, query_lens, kv_lens)
    if layout == "vectorized":
        case["k"], case["v"] = shuffle_kv(case["k"], case["v"])
    candidates = {"flydsl": partial(ua.unified_attention, **case, backend="flydsl")}
    ret = measure(candidates, case, want, query_lens, kv_lens)
    compare(torch.zeros_like(want[-1]), case["out"][-1], 0, "empty KV")
    return ret


@benchmark()
def test_mixed_batch(causal, splits):
    query_lens = [128] + [1] * 8
    kv_lens = [16384, 4096, 8192, 1024, 6000, 300, 12000, 2048, 512]
    case = make_case(query_lens, kv_lens, dtypes.bf16, causal)
    want = reference(case, query_lens, kv_lens)
    storage = torch.full(
        (sum(query_lens) + 64, H, D), float("nan"), device="cuda", dtype=dtypes.bf16
    )
    case["out"] = storage[: sum(query_lens)]
    candidates = {"packed_splitk": direct_candidate(case, kv_lens, splits, packed=True)}
    ret = measure(candidates, case, want, query_lens, kv_lens, atol=0.1, bad_rows=True)
    assert torch.isnan(
        storage[sum(query_lens) :]
    ).all(), "combine wrote beyond packed output"
    return ret


@benchmark()
def test_cross_attention_mask(query_len, kv_len):
    case = make_case([query_len], [kv_len], dtypes.bf16)
    want = reference(case, [query_len], [kv_len])
    candidates = {"flydsl": partial(ua.unified_attention, **case, backend="flydsl")}
    return measure(candidates, case, want, [query_len], [kv_len], atol=0.1)


@benchmark()
def test_splitk_combine(seq_len, splits):
    case = make_case([seq_len], [seq_len], dtypes.bf16, causal=False)
    want = reference(case, [seq_len], [seq_len])
    candidates = {"splitk": direct_candidate(case, [seq_len], splits)}
    return measure(
        candidates, case, want, [seq_len], [seq_len], atol=0.1, bad_rows=True
    )


@benchmark()
def test_paged_addressing(pool_blocks, layout):
    query_lens, kv_lens = [128], [256]
    original = make_case(query_lens, kv_lens, dtypes.bf16)
    want = reference(original, query_lens, kv_lens)
    candidates, cases = {}, {}
    count = original["k"].shape[0]
    for name, ids in (
        ("identity", list(range(count))),
        ("reversed", list(reversed(range(count)))),
    ):
        case = dict(original, out=torch.empty_like(original["out"]))
        ids = [pool_blocks - count + i for i in ids]
        for key in ("k", "v"):
            pool = torch.zeros(
                pool_blocks, PAGE, HKV, D, device="cuda", dtype=original[key].dtype
            )
            for logical, pid in enumerate(ids):
                pool[pid] = original[key][int(original["block_table"][0, logical])]
            case[key] = pool
        case["block_table"] = torch.tensor([ids], device="cuda", dtype=torch.int32)
        if layout == "vectorized":
            case["k"], case["v"] = shuffle_kv(case["k"], case["v"])
        cases[name] = case
        candidates[name] = direct_candidate(case, kv_lens)
    ret = measure(candidates, cases["identity"], want, query_lens, kv_lens, atol=0.1)
    compare(
        cases["identity"]["out"],
        cases["reversed"]["out"],
        0,
        "page permutation invariance",
    )
    return ret


@benchmark()
def test_routing_backend_gate(config):
    import aiter.ops.flydsl.unified_attention_kernels as adapter

    query_lens, kv_lens = [256], [256]
    case = make_case(query_lens, kv_lens, dtypes.bf16)
    if config == "softcap":
        case["softcap"] = 30.0
    elif config == "sliding_window":
        case["window_size"] = (127, 0)
    want = reference(case, query_lens, kv_lens)
    explicit = partial(ua.unified_attention, **case, backend="flydsl")
    if config != "supported":
        try:
            explicit()
        except RuntimeError as exc:
            assert "does not support this configuration" in str(exc)
        else:
            raise AssertionError("explicit FlyDSL silently accepted a declined config")
    real = adapter.flydsl_unified_attention
    with mock.patch.object(adapter, "flydsl_unified_attention", wraps=real) as spy:
        auto = ua.unified_attention(**case).clone()
        assert spy.call_count > 0, "automatic dispatch never offered the call to FlyDSL"
    compare(want, auto, 0.08 * want.abs().max().item(), "automatic routing")
    candidates = {"triton": partial(ua.unified_attention, **case, backend="triton")}
    with mock.patch.object(
        adapter,
        "flydsl_unified_attention",
        side_effect=AssertionError("explicit backend selected FlyDSL"),
    ):
        ret = measure(candidates, case, want, query_lens, kv_lens)
        if config != "supported":
            compare(auto, case["out"], 0, "declined config vs Triton bitwise")
        if ua._is_gluon_available():
            ret.update(
                measure(
                    {"gluon": partial(ua.unified_attention, **case, backend="gluon")},
                    case,
                    want,
                    query_lens,
                    kv_lens,
                )
            )
        else:
            try:
                ua.unified_attention(**case, backend="gluon")
            except AssertionError as exc:
                assert "Gluon backend requires" in str(exc)
            else:
                raise AssertionError("unsupported Gluon backend did not raise")
            ret["gluon gate"] = "rejected: unsupported arch (no FlyDSL call)"
    if config == "supported":
        ret.update(measure({"flydsl": explicit}, case, want, query_lens, kv_lens))
    return ret


@benchmark()
def test_warm_cache_run_only(path):
    from aiter.aot.flydsl.common import run_only_env

    query_lens, kv_lens = ([256], [256]) if path == "prefill" else ([1] * 8, [4096] * 8)
    case = make_case(query_lens, kv_lens, dtypes.bf16, seed=17)
    want = reference(case, query_lens, kv_lens)
    call = partial(ua.unified_attention, **case, backend="flydsl")
    call()
    torch.cuda.synchronize()
    case["out"].fill_(float("nan"))
    # Missing artifacts must raise rather than quietly compile a replacement.
    with run_only_env():
        return measure({"warm_run_only": call}, case, want, query_lens, kv_lens)


def main():
    arch = get_gfx_runtime()
    if get_gfx() != "gfx950" or arch != "gfx950":
        aiter.logger.warning(
            "FlyDSL unified attention requires gfx950; skipping (build=%s, attached=%s)",
            get_gfx(),
            arch,
        )
        return
    from aiter.ops.flydsl.unified_attention_kernels import is_flydsl_available

    if (
        not is_flydsl_available(torch.cuda.current_device())
        or torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        != 256
    ):
        aiter.logger.warning(
            "FlyDSL unified attention requires FlyDSL and a full-chip gfx950; skipping"
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FP8 unified attention correctness and warm-buffer timing sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        default=[dtypes.bf16, dtypes.fp16],
        choices=[dtypes.bf16, dtypes.fp16],
    )
    parser.add_argument("--causal", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    parser.add_argument("--decode-depth", type=int, nargs="+", default=[64, 8192])
    parser.add_argument(
        "--layout",
        choices=["linear", "vectorized"],
        nargs="+",
        default=["linear", "vectorized"],
    )
    parser.add_argument(
        "--splits", type=int, choices=[2, 4, 8], nargs="+", default=[2, 4]
    )
    parser.add_argument("--seq-len", type=int, nargs="+", default=[1024])
    args = parser.parse_args()
    sweeps = [
        ("prefill", test_prefill, itertools.product(args.dtype, args.causal)),
        (
            "decode",
            test_decode,
            itertools.product(args.dtype, args.decode_depth, args.layout),
        ),
        ("mixed batch", test_mixed_batch, itertools.product(args.causal, args.splits)),
        (
            "cross-attention mask",
            test_cross_attention_mask,
            itertools.product([320], [1024]),
        ),
        (
            "split-K combine",
            test_splitk_combine,
            itertools.product(args.seq_len, args.splits),
        ),
        # 65535 pages stay below the dualwave launcher's signed-i32 element limit.
        (
            "paged addressing",
            test_paged_addressing,
            itertools.product([65535], args.layout),
        ),
        (
            "routing/backend gate",
            test_routing_backend_gate,
            itertools.product(["supported", "softcap", "sliding_window"]),
        ),
        (
            "warm-cache run-only (not full AOT)",
            test_warm_cache_run_only,
            itertools.product(["prefill", "decode"]),
        ),
    ]
    for name, fn, parameters in sweeps:
        rows = [fn(*values) for values in parameters]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "%s summary (markdown):\n%s", name, df.to_markdown(index=False)
        )
    aiter.logger.info(
        "PASS: all eight unified-attention test groups; all candidate timings non-zero"
    )


if __name__ == "__main__":
    main()
