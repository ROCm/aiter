"""Expanded per-call stride regression on actual AITER; no video speed claim."""

import json
from functools import partial
from unittest.mock import patch

import torch

from aiter.ops.triton.attention import mha


def tensors(lengths, heads, dim, strided=False, grad=False):
    n = sum(lengths)
    if strided:
        x = torch.randn(n * 2 + 3, heads, dim + 8, device="cuda", dtype=torch.bfloat16)[
            1 : 1 + 2 * n : 2, :, :dim
        ]
    else:
        x = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    return x.detach().requires_grad_(grad)


def reference(q, k, v, qlens, klens, causal, window):
    output = []
    qo = ko = 0
    for nq, nk in zip(qlens, klens):
        qq, kk, vv = [
            x.transpose(0, 1).float()
            for x in (q[qo : qo + nq], k[ko : ko + nk], v[ko : ko + nk])
        ]
        if qq.shape[0] != kk.shape[0]:
            repeats = qq.shape[0] // kk.shape[0]
            kk, vv = kk.repeat_interleave(repeats, 0), vv.repeat_interleave(repeats, 0)
        qi = torch.arange(nq, device="cuda")[:, None] + nk - nq
        ki = torch.arange(nk, device="cuda")[None, :]
        mask = torch.ones(nq, nk, device="cuda", dtype=torch.bool)
        if causal:
            mask &= ki <= qi
        if window[0] >= 0:
            mask &= ki >= qi - window[0]
        if window[1] >= 0:
            mask &= ki <= qi + window[1]
        output.append(
            torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, attn_mask=mask
            ).transpose(0, 1)
        )
        qo += nq
        ko += nk
    return torch.cat(output)


torch.manual_seed(20260913)
original_kernel = mha._attn_fwd
widths = []


class RecordWidth:
    def __getitem__(self, grid):
        launch = original_kernel[grid]

        def run(*args, **kwargs):
            widths.append(kwargs["USE_INT64_STRIDES"])
            return launch(*args, **kwargs)

        return run


cases = [
    ([1, 127, 129], [1, 127, 129], 4, 4, 128, False, False, (-1, -1)),
    ([17, 129], [33, 257], 4, 4, 128, True, False, (-1, -1)),
    ([31, 129], [31, 129], 4, 4, 128, False, True, (-1, -1)),
    ([65, 257], [65, 257], 8, 2, 128, True, False, (-1, -1)),
    ([65, 129], [65, 129], 4, 4, 80, False, True, (-1, -1)),
    ([129, 257], [129, 257], 4, 4, 128, True, False, (31, -1)),
]
for case_id, (qlens, klens, qh, kh, dim, causal, strided, window) in enumerate(cases):
    q, k, v = (
        tensors(qlens, qh, dim, strided),
        tensors(klens, kh, dim, strided),
        tensors(klens, kh, dim, strided),
    )
    cuq = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cuk = torch.tensor(
        [0, *torch.tensor(klens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )

    invoke = partial(
        mha.flash_attn_varlen_func,
        q,
        k,
        v,
        cuq,
        cuk,
        max(qlens),
        max(klens),
        causal=causal,
        window_size=window,
    )

    with torch.inference_mode():
        widths.clear()
        with patch.object(mha, "_attn_fwd", RecordWidth()):
            a, b = invoke(prefer_int32_strides=False), invoke(prefer_int32_strides=True)
        assert widths == [True, False], widths
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        ref = reference(q, k, v, qlens, klens, causal, window)
        torch.testing.assert_close(b.float(), ref, atol=0.02, rtol=0.03)
        assert mha._USE_INT64_STRIDES is True
        print(
            json.dumps(
                {
                    "case": case_id,
                    "qlens": qlens,
                    "klens": klens,
                    "qh": qh,
                    "kh": kh,
                    "dim": dim,
                    "strided": strided,
                    "causal": causal,
                    "window": window,
                    "parity_reference": "pass",
                }
            ),
            flush=True,
        )

# Training must retain 64-bit forward and identical backward, even with the opt-in.
for prefer in (False, True):
    torch.manual_seed(42)
    q, k, v = [tensors([65, 129], 4, 128, grad=True) for _ in range(3)]
    cu = torch.tensor([0, 65, 194], device="cuda", dtype=torch.int32)
    widths.clear()
    with patch.object(mha, "_attn_fwd", RecordWidth()):
        out = mha.flash_attn_varlen_func(
            q, k, v, cu, cu, 129, 129, prefer_int32_strides=prefer
        )
    assert widths == [True], widths
    out.float().square().sum().backward()
    result = (out.detach(), q.grad, k.grad, v.grad)
    if not prefer:
        control = [x.clone() for x in result]
    else:
        for x, y in zip(result, control):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
for prefer in (False, True):
    try:
        mha.flash_attn_varlen_func(
            q, k, v, cu, cu, 129, 129, window_size=(31, 7), prefer_int32_strides=prefer
        )
    except ValueError as error:
        assert "window_size_right" in str(error)
    else:
        raise AssertionError("Unsupported right window must not silently run")
print("H3_LAYOUT_CAUSAL_GQA_WINDOW_GRADIENT_REGRESSION_PASS", flush=True)
