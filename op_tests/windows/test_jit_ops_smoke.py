# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Smoke test for JIT-built HIP modules against a torch reference.

Each op runs in its own subprocess, so a failed module build or a GPU fault in
one op is reported against that op instead of taking down the whole run.

    pytest op_tests/windows/test_jit_ops_smoke.py
    python op_tests/windows/test_jit_ops_smoke.py [op ...]
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _is_gfx1201_64cu():
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    try:
        return get_gfx() == "gfx1201" and get_cu_num() == 64
    except Exception:  # noqa: BLE001
        return False


def _rms_norm_ref(x, weight, eps):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return out.to(x.dtype)


def _assert_close(name, out, ref, **tol):
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"{name}: max_abs_diff={diff:.3e} tol={tol}")
    torch.testing.assert_close(out, ref, **tol)


def check_rms_norm():
    import aiter

    # bf16, 2-D, hidden <= 8192 dispatches to module_rmsnorm_quant.
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    out = aiter.rms_norm(x, weight, 1e-6)
    _assert_close("rms_norm", out, _rms_norm_ref(x, weight, 1e-6), atol=2e-2, rtol=2e-2)


def check_rmsnorm2d_fwd():
    import aiter

    # fp32 dispatches to the opus kernel in module_rmsnorm.
    x = torch.randn(64, 1024, dtype=torch.float32, device="cuda")
    weight = torch.randn(1024, dtype=torch.float32, device="cuda")
    out = aiter.rmsnorm2d_fwd(x, weight, 1e-6)
    _assert_close(
        "rmsnorm2d_fwd", out, _rms_norm_ref(x, weight, 1e-6), atol=1e-4, rtol=1e-4
    )


def check_silu_and_mul():
    import aiter

    d = 2048
    x = torch.randn(64, 2 * d, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(64, d, dtype=torch.bfloat16, device="cuda")
    aiter.silu_and_mul(out, x)
    ref = (F.silu(x[:, :d].float()) * x[:, d:].float()).to(x.dtype)
    _assert_close("silu_and_mul", out, ref, atol=2e-2, rtol=2e-2)


def check_topk_softmax():
    import aiter

    num_tokens, num_experts, topk = 512, 64, 8
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    for need_renorm in (False, True):
        weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
        ids = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        token_expert_ids = torch.empty_like(ids)
        aiter.topk_softmax(weights, ids, token_expert_ids, gating, need_renorm)

        ref_weights, ref_ids = gating.softmax(-1).topk(topk, dim=-1)
        if need_renorm:
            ref_weights = ref_weights / ref_weights.sum(-1, keepdim=True)
        # The kernel does not promise any order within the selected top-k.
        order = ids.argsort(-1)
        suffix = f"renorm={need_renorm}"
        _assert_close(
            f"topk_softmax_ids[{suffix}]",
            ids.gather(-1, order),
            ref_ids.int().sort(-1)[0],
        )
        ref_order = ref_ids.argsort(-1)
        _assert_close(
            f"topk_softmax_weights[{suffix}]",
            weights.gather(-1, order),
            ref_weights.gather(-1, ref_order),
            atol=1e-4,
            rtol=1e-4,
        )


def _rope_ref(x, cos, sin, is_neox):
    xf = x.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    if is_neox:
        x1, x2 = xf.chunk(2, dim=-1)
    else:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox:
        out = torch.cat((o1, o2), dim=-1)
    else:
        out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out.to(x.dtype)


def check_rotary_embedding_fwd():
    import aiter

    num_tokens, num_heads, num_kv_heads, head_size, max_pos = 32, 8, 2, 128, 256
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
    )
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    cos_cache = freqs.cos().to(torch.bfloat16).cuda()
    sin_cache = freqs.sin().to(torch.bfloat16).cuda()
    positions = torch.randint(0, max_pos, (num_tokens,), device="cuda")

    for is_neox in (True, False):
        # The kernel takes the flattened [num_tokens, num_heads * head_size] layout.
        q = torch.randn(
            num_tokens, num_heads * head_size, dtype=torch.bfloat16, device="cuda"
        )
        k = torch.randn(
            num_tokens, num_kv_heads * head_size, dtype=torch.bfloat16, device="cuda"
        )
        cos, sin = cos_cache[positions], sin_cache[positions]
        ref_q = _rope_ref(q.view(num_tokens, num_heads, head_size), cos, sin, is_neox)
        ref_k = _rope_ref(
            k.view(num_tokens, num_kv_heads, head_size), cos, sin, is_neox
        )
        aiter.rotary_embedding_fwd(
            positions, q, k, head_size, cos_cache, sin_cache, is_neox, False
        )
        suffix = f"is_neox={is_neox}"
        _assert_close(
            f"rotary_embedding_fwd_q[{suffix}]",
            q,
            ref_q.flatten(1),
            atol=2e-2,
            rtol=2e-2,
        )
        _assert_close(
            f"rotary_embedding_fwd_k[{suffix}]",
            k,
            ref_k.flatten(1),
            atol=2e-2,
            rtol=2e-2,
        )


def _make_bf16_g1u1_moe_case(tokens):
    import aiter

    model_dim = 2048
    inter_dim = 128
    num_experts = 256
    topk = 8
    hidden = torch.randn((tokens, model_dim), dtype=aiter.dtypes.bf16, device="cuda")
    w1 = torch.randn(
        (num_experts, 2 * inter_dim, model_dim),
        dtype=aiter.dtypes.bf16,
        device="cuda",
    )
    w2 = torch.randn(
        (num_experts, model_dim, inter_dim),
        dtype=aiter.dtypes.bf16,
        device="cuda",
    )
    ids = torch.randint(
        0, num_experts, (tokens, topk), dtype=torch.int32, device="cuda"
    )
    weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (
        hidden,
        w1.contiguous(),
        w2.contiguous(),
        weights.contiguous(),
        ids.contiguous(),
    )


def check_moe_gemm_a16w16():
    if not _is_gfx1201_64cu():
        print("moe_gemm_a16w16: SKIP (requires gfx1201 64-CU target)")
        return

    import aiter
    from aiter.fused_moe import fused_moe, torch_moe
    from aiter.test_common import checkAllclose

    torch.manual_seed(5059)
    hidden, w1, w2, weights, ids = _make_bf16_g1u1_moe_case(1024)
    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=aiter.dtypes.bf16,
    )
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    assert checkAllclose(actual, expected, rtol=1e-2, atol=128) == 0


def check_moe_ck2stage_bf16():
    if not _is_gfx1201_64cu():
        print("moe_ck2stage_bf16: SKIP (requires gfx1201 64-CU target)")
        return

    os.environ["AITER_DISABLE_GFX1201_BF16_G1U1_TRITON"] = "1"

    import aiter
    from aiter.fused_moe import fused_moe, get_2stage_cfgs, get_padded_M
    from aiter.ops.enum import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.moe_sorting_opus import moe_sorting_opus_get_workspace_size
    from aiter.ops.shuffle import shuffle_weight

    torch.manual_seed(4815)
    token = 2050
    model_dim = 2048
    inter_dim = 128
    num_experts = 256
    topk = 8

    assert get_padded_M(token) == 4096
    assert moe_sorting_opus_get_workspace_size(token, num_experts, topk, 0, 0) >= 0
    metadata = get_2stage_cfgs(
        get_padded_M(token),
        model_dim,
        inter_dim,
        num_experts,
        topk,
        aiter.dtypes.bf16,
        aiter.dtypes.bf16,
        aiter.dtypes.bf16,
        QuantType.No,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.SEPARATED.value,
        opus_weights_shuffled=True,
        input_dtype=aiter.dtypes.bf16,
    )
    assert metadata.stage1.keywords["kernelName"].startswith("moe_ck2stages_gemm1_")
    assert metadata.stage2.keywords["kernelName"].startswith("moe_ck2stages_gemm2_")

    hidden, w1, w2, weights, ids = _make_bf16_g1u1_moe_case(token)
    actual = fused_moe(
        hidden,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=aiter.dtypes.bf16,
        gate_mode=GateMode.SEPARATED.value,
    )
    assert actual.shape == (token, model_dim)
    assert actual.dtype == aiter.dtypes.bf16
    assert torch.isfinite(actual).all()


CHECKS = {
    "rms_norm": check_rms_norm,
    "rmsnorm2d_fwd": check_rmsnorm2d_fwd,
    "silu_and_mul": check_silu_and_mul,
    "topk_softmax": check_topk_softmax,
    "rotary_embedding_fwd": check_rotary_embedding_fwd,
    "moe_gemm_a16w16": check_moe_gemm_a16w16,
    "moe_ck2stage_bf16": check_moe_ck2stage_bf16,
}


TIMEOUT_S = 1800


def _run_isolated(op):
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        _REPO_ROOT
        if not env.get("PYTHONPATH")
        else f"{_REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )
    try:
        return subprocess.run(
            [sys.executable, os.path.abspath(__file__), op],
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired carries bytes even when text=True.
        stdout = e.stdout.decode(errors="replace") if e.stdout else ""
        return subprocess.CompletedProcess(
            e.cmd, -1, stdout, f"timed out after {TIMEOUT_S}s"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("op", list(CHECKS))
def test_jit_op(op):
    proc = _run_isolated(op)
    print(proc.stdout, end="")
    assert proc.returncode == 0, f"{op} failed:\n{proc.stdout}\n{proc.stderr}"


def main(ops):
    if len(ops) == 1:
        torch.manual_seed(0)
        CHECKS[ops[0]]()
        print(f"{ops[0]}: OK")
        return 0
    failed = []
    for op in ops:
        proc = _run_isolated(op)
        if proc.returncode == 0:
            print(proc.stdout, end="")
        else:
            failed.append(op)
            print(f"{op}: FAILED\n{proc.stdout}\n{proc.stderr}")
    return 1 if failed else 0


if __name__ == "__main__":
    unknown = [op for op in sys.argv[1:] if op not in CHECKS]
    if unknown:
        sys.exit(
            f"unknown op(s): {', '.join(unknown)}; choose from {', '.join(CHECKS)}"
        )
    sys.exit(main(sys.argv[1:] or list(CHECKS)))
