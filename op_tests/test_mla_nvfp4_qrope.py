# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 Q RoPE/quantization fusion: python -m pytest -q this_file.py."""

import os
import subprocess
import sys

import pytest
import torch
from op_tests.test_mla_decode_pagesize64 import (
    _check_nvfp4_page64_case,
    _make_nvfp4_page64_case,
    _tail_guarded,
)

import aiter


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
        0
    ).gcnArchName.startswith("gfx1250"):
        pytest.skip("requires gfx1250")


def rotate_reference(q, cos, sin, positions):
    """Independent adjacent-pair rotation with BF16 MUL then BF16 FMA."""
    x = q.cpu().double().reshape(*q.shape[:-1], 32, 2)
    c = cos.cpu().reshape(-1, 32)[positions.cpu()].double()[:, None, :, None]
    s = sin.cpu().reshape(-1, 32)[positions.cpu()].double()[:, None, :, None]
    pair = torch.stack((-x[..., 1], x[..., 0]), dim=-1)
    term = (pair * s).to(torch.bfloat16).double()
    return (x * c + term).to(torch.bfloat16).reshape(q.shape).to(q.device)


def make_case(
    lengths,
    layout="projection",
    compact_scale=False,
    identity=False,
    table_rows=257,
    positions=None,
    shared_tables=False,
    table_singletons=0,
):
    case = _make_nvfp4_page64_case(lengths, "randn")
    a = case["args"]
    b = len(lengths)
    width = {"projection": 192, "full_q": 576, "compact": 64}[layout]
    projection = torch.full(
        (b, 128, width), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    raw = projection[..., -64:]
    raw.copy_(case["q"][..., 512:])
    # Exercise zero blocks, signed zero, tiny scales and scale saturation.
    if b:
        raw[:, 0].zero_()
        raw[:, 1, ::2] = -0.0
        raw[:, 2] *= 2**-12
        raw[:, 3] *= 2**12
    theta = torch.rand((table_rows, 32), device="cuda") * 6.28 - 3.14
    cos, sin = theta.cos().bfloat16(), theta.sin().bfloat16()
    if shared_tables:
        # Aliased planes retain compact 64-byte rows, including under compile.
        cs = torch.stack((cos, sin))
        cos, sin = cs[0], cs[1]
    table_shape = (table_rows, *(1 for _ in range(table_singletons)), 32)
    cos, sin = cos.view(table_shape), sin.view(table_shape)
    if identity:
        cos.fill_(1)
        sin.zero_()
    if positions is None:
        positions = torch.randint(0, table_rows, (b,), dtype=torch.int64, device="cuda")
    else:
        positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
    # The separate path uses the existing public quantizer and pure decoder.
    rotated = rotate_reference(raw, cos, sin, positions)
    case["q"][..., 512:] = rotated
    qn, qa = aiter.mla_nvfp4_quantize_q(case["q"])
    a["q"], a["q_aux"] = qn, qa
    scale = qa[..., 64:96]
    if compact_scale:
        scale = scale.contiguous()
    output_guard = torch.full_like(case["guards"][0], float("nan"))
    args = {
        "q_nope": qn,
        "q_scale": scale,
        "q_rope": raw,
        "kv_nope": a["kv_buffer"],
        "kv_aux": a["kv_aux"],
        "out": output_guard[:-256].view(b, 128, 512),
        "kv_indptr": a["kv_indptr"],
        "kv_indices": a["kv_indices"],
        "kv_last_page_lens": a["kv_last_page_lens"],
        "params": a["nvfp4_params"],
        "positions": positions,
        "cos": cos,
        "sin": sin,
    }
    return case, args, output_guard


def separate(case):
    a = case["args"]
    return aiter.mla_decode_nvfp4_fwd(
        a["q"],
        a["q_aux"],
        a["kv_buffer"],
        a["kv_aux"],
        a["o"],
        a["kv_indptr"],
        a["kv_indices"],
        a["kv_last_page_lens"],
        params=a["nvfp4_params"],
        lse=a["nvfp4_lse"],
    )[0]


def check(case, args, guard):
    expected = separate(case).clone()
    # In particular, the fused kernel must not write Q AUX through a scale view.
    before = {
        k: args[k].contiguous().view(torch.uint8).clone()
        for k in ("q_nope", "q_scale", "q_rope", "cos", "sin", "positions")
    }
    for _ in range(3):
        out, lse = aiter.mla_decode_nvfp4_qrope_fwd(**args)
        assert out is args["out"] and lse is None
        assert torch.equal(out, expected)
        assert torch.isfinite(out).all()
    for k, value in before.items():
        assert torch.equal(args[k].contiguous().view(torch.uint8), value), k
    assert torch.isnan(guard[-256:]).all()
    assert torch.isnan(case["guards"][0][-256:]).all()
    assert (case["guards"][1] == 12345).all()
    return expected


@pytest.mark.parametrize("layout", ["projection", "full_q", "compact"])
@pytest.mark.parametrize("compact_scale", [False, True])
@pytest.mark.parametrize("identity", [False, True])
def test_layouts_and_page_tails(layout, compact_scale, identity):
    lengths = [
        0,
        1,
        16,
        32,
        63,
        64,
        65,
        96,
        127,
        128,
        129,
        255,
        256,
        257,
        513,
        1025,
        5120,
    ]
    case, args, guard = make_case(lengths, layout, compact_scale, identity)
    check(case, args, guard)


@pytest.mark.parametrize("table_rows", [1, 2, 17, 257, 163840])
@pytest.mark.parametrize("shared_tables", [False, True])
@pytest.mark.parametrize("table_singletons", [0, 1, 2])
def test_compact_table_boundaries(table_rows, shared_tables, table_singletons):
    case, args, guard = make_case(
        [1, 63, 65, 129],
        table_rows=table_rows,
        shared_tables=shared_tables,
        table_singletons=table_singletons,
        positions=[0, table_rows - 1, table_rows // 2, table_rows - 1],
    )
    assert (
        args["cos"].stride()
        == args["sin"].stride()
        == (32,) * (table_singletons + 1) + (1,)
    )
    check(case, args, guard)


def test_all_partial_page_lengths():
    case, args, guard = make_case(list(range(65)))
    check(case, args, guard)


def test_bf16_attention_reference():
    case, args, guard = make_case([0, 1, 65, 129, 513])
    # The saturation fixture tests the specified quantizer, not operator accuracy.
    args["q_rope"][:, 2:4] *= torch.tensor(
        [2**12, 2**-12], device="cuda", dtype=torch.bfloat16
    )[None, :, None]
    case["q"][..., 512:] = rotate_reference(
        args["q_rope"], args["cos"], args["sin"], args["positions"]
    )
    qn, qa = aiter.mla_nvfp4_quantize_q(case["q"])
    case["args"]["q"], case["args"]["q_aux"] = qn, qa
    args["q_nope"], args["q_scale"] = qn, qa[..., 64:96]
    check(case, args, guard)
    case["args"]["o"] = args["out"]
    _check_nvfp4_page64_case(case)


def test_empty_batch():
    case, args, guard = make_case([])
    check(case, args, guard)


def test_stream_graph_compile():
    case, args, guard = make_case([0, 65, 129], table_singletons=2)
    expected = check(case, args, guard)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        aiter.mla_decode_nvfp4_qrope_fwd(**args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            aiter.mla_decode_nvfp4_qrope_fwd(**args)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(args["out"], expected)
    args["kv_indptr"].zero_()
    args["kv_last_page_lens"].zero_()
    args["positions"].fill_(-1)  # Empty rows must not dereference the RoPE tables.
    args["q_rope"].fill_(float("nan"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert (args["out"] == 0).all() and torch.isnan(guard[-256:]).all()
    assert (case["guards"][1] == 12345).all()
    case, args, guard = make_case([65], shared_tables=True, table_singletons=2)
    expected = check(case, args, guard)
    before = {
        k: args[k].clone() for k in ("q_scale", "q_rope", "cos", "sin", "positions")
    }
    compiled = torch.compile(aiter.mla_decode_nvfp4_qrope_fwd, fullgraph=True)
    for _ in range(2):
        out, lse = compiled(**args)
        assert torch.equal(out, expected) and lse is None
    for k, value in before.items():
        assert torch.equal(args[k], value), k


def test_contracts():
    _, args, _ = make_case([65])
    changes = [
        {"q_scale": args["q_scale"].float()},
        {"q_rope": args["q_rope"].float()},
        {"q_rope": args["q_rope"][..., ::2]},
        {"q_rope": args["q_rope"].expand(2, -1, -1)},
        {"positions": args["positions"].int()},
        {"positions": args["positions"].cpu()},
        {"cos": torch.empty((257, 64), dtype=torch.bfloat16, device="cuda")[:, :32]},
        {"sin": torch.empty((257, 64), dtype=torch.bfloat16, device="cuda")[:, :32]},
        {"cos": args["cos"][:0], "sin": args["sin"][:0]},
        {"sin": args["sin"][:-1]},
        {"cos": args["cos"].view(257, 2, 16)},
        {"sin": args["sin"].reshape(-1)},
        {"params": args["params"].float()},
        {"kv_indices": args["kv_indices"].long()},
        {"out": args["out"].float()},
        {"return_lse": True},
    ]
    misaligned = torch.empty(128 * 64 + 1, device="cuda", dtype=torch.bfloat16)
    changes.append({"q_rope": misaligned[1:].view(1, 128, 64)})
    for change in changes:
        with pytest.raises((RuntimeError, ValueError)):
            aiter.mla_decode_nvfp4_qrope_fwd(**(args | change))
    _, empty, _ = make_case([])
    for strides in ((2**32, 64, 1), (8192, 2**27, 1), (0, 64, 1)):
        bad = torch.as_strided(empty["q_rope"], (0, 128, 64), strides)
        with pytest.raises(RuntimeError, match="stride"):
            aiter.mla_decode_nvfp4_qrope_fwd(**(empty | {"q_rope": bad}))


@pytest.mark.skipif(
    os.getenv("AITER_NVFP4_TEST_LARGE") != "1", reason="large GPU allocation"
)
def test_cache_addresses_above_4gib():
    case, args, guard = make_case([65, 129])
    expected = check(case, args, guard)
    pages = 524305
    n = args["kv_nope"].shape[0]
    target = torch.arange(pages - n, pages, device="cuda")
    for key, width in (("kv_nope", 16384), ("kv_aux", 8192)):
        pool = torch.empty((pages, width), dtype=torch.uint8, device="cuda")
        pool[target] = args[key]
        args[key] = pool
    args["kv_indices"] = target[args["kv_indices"].long()].int()
    args["params"] = aiter.prepare_mla_nvfp4_params(
        device="cuda", physical_pages=pages, sm_scale=576**-0.5
    )
    out, _ = aiter.mla_decode_nvfp4_qrope_fwd(**args)
    assert torch.equal(out, expected) and torch.isnan(guard[-256:]).all()


@pytest.mark.skipif(
    os.getenv("AITER_NVFP4_TEST_LARGE") != "1", reason="large GPU allocation"
)
def test_raw_query_and_scale_addresses_above_4gib():
    case, args, guard = make_case([1, 65, 129])
    for key, stride, dtype in (
        ("q_rope", (2**30 + 128, 192, 1), torch.bfloat16),
        ("q_scale", (2**31 + 128, 128, 1), torch.uint8),
    ):
        sparse = torch.empty_strided(
            args[key].shape, stride, dtype=dtype, device="cuda"
        )
        sparse.copy_(args[key])
        assert sparse.stride(0) * sparse.element_size() < 2**32
        assert 2 * sparse.stride(0) * sparse.element_size() > 2**32
        args[key] = sparse
    check(case, args, guard)


@pytest.mark.skipif(
    os.getenv("AITER_NVFP4_TEST_LARGE") != "1", reason="large GPU allocation"
)
def test_table_addresses_above_4gib():
    case, args, guard = make_case([1, 65, 129])
    expected = separate(case).clone()
    rows = 2**26 + 2
    positions = torch.tensor([2**26 - 1, 2**26, rows - 1], device="cuda")
    for key in ("cos", "sin"):
        table = torch.empty((rows, 32), dtype=torch.bfloat16, device="cuda")
        table[positions] = args[key][args["positions"]]
        args[key] = table
    args["positions"] = positions
    out, _ = aiter.mla_decode_nvfp4_qrope_fwd(**args)
    assert torch.equal(out, expected) and torch.isnan(guard[-256:]).all()


def _guard_case(ctx_len):
    assert os.environ.get("PYTORCH_NO_HIP_MEMORY_CACHING") == "1"
    case, args, _ = make_case(
        [0, ctx_len, ctx_len], compact_scale=True, positions=[0, 0, 256]
    )
    expected = separate(case).clone()
    # Preserve the real projection stride while placing the final raw-Q row
    # at its own allocation's end. Guard all other complete tensor buffers.
    physical_q = torch.full(
        (3, 128, 192), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    physical_q[..., 128:] = args["q_rope"]
    args["q_rope"] = _tail_guarded(physical_q)[..., 128:]
    for name in args:
        if name != "q_rope":
            args[name] = _tail_guarded(args[name])
    out, _ = aiter.mla_decode_nvfp4_qrope_fwd(**args)
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    print("COMPLETED guarded compact Q RoPE", flush=True)


@pytest.mark.skipif(
    os.getenv("AITER_NVFP4_TEST_GUARDS") != "1",
    reason="isolated allocation-tail checks",
)
@pytest.mark.parametrize("ctx_len", [1, 31, 63, 64, 65, 127, 128, 129, 257, 513])
def test_guarded_compact_inputs(ctx_len):
    env = os.environ | {"PYTORCH_NO_HIP_MEMORY_CACHING": "1"}
    result = subprocess.run(
        [sys.executable, __file__, "--guard-case", str(ctx_len)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETED guarded compact Q RoPE" in result.stdout


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--guard-case":
        _guard_case(int(sys.argv[2]))
    else:
        sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
