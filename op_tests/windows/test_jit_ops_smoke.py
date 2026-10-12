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


def _rms_norm_ref(x, weight, eps):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return out.to(x.dtype)


def check_rms_norm():
    import aiter

    # bf16, 2-D, hidden <= 8192 dispatches to module_rmsnorm_quant.
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    out = aiter.rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(
        out, _rms_norm_ref(x, weight, 1e-6), atol=2e-2, rtol=2e-2
    )


def check_rmsnorm2d_fwd():
    import aiter

    # fp32 dispatches to the opus kernel in module_rmsnorm.
    x = torch.randn(64, 1024, dtype=torch.float32, device="cuda")
    weight = torch.randn(1024, dtype=torch.float32, device="cuda")
    out = aiter.rmsnorm2d_fwd(x, weight, 1e-6)
    torch.testing.assert_close(
        out, _rms_norm_ref(x, weight, 1e-6), atol=1e-4, rtol=1e-4
    )


def check_silu_and_mul():
    import aiter

    d = 2048
    x = torch.randn(64, 2 * d, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(64, d, dtype=torch.bfloat16, device="cuda")
    aiter.silu_and_mul(out, x)
    ref = (F.silu(x[:, :d].float()) * x[:, d:].float()).to(x.dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


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
        torch.testing.assert_close(ids.gather(-1, order), ref_ids.int().sort(-1)[0])
        ref_order = ref_ids.argsort(-1)
        torch.testing.assert_close(
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
        torch.testing.assert_close(q, ref_q.flatten(1), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k, ref_k.flatten(1), atol=2e-2, rtol=2e-2)


CHECKS = {
    "rms_norm": check_rms_norm,
    "rmsnorm2d_fwd": check_rmsnorm2d_fwd,
    "silu_and_mul": check_silu_and_mul,
    "topk_softmax": check_topk_softmax,
    "rotary_embedding_fwd": check_rotary_embedding_fwd,
}


TIMEOUT_S = 1800


def _run_isolated(op):
    try:
        return subprocess.run(
            [sys.executable, os.path.abspath(__file__), op],
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_S,
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
            print(f"{op}: OK")
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
