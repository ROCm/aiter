# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Exercise short-query split-K dispatch without GPU execution or compilation."""

from contextlib import nullcontext

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize(
    "q_len,causal,cross,splits,expected",
    [
        (1, False, True, 4, 4),
        (70, False, True, 4, 4),
        (70, False, True, None, 4),
        (70, False, True, 1, 1),
        (383, False, True, 4, 4),
        (70, True, True, 4, None),
        (70, True, False, 4, None),
        (70, False, False, 4, None),
        (70, True, True, None, 1),
        (70, False, False, None, 1),
        (384, True, False, 4, 4),
    ],
)
def test_splitk_dispatch(monkeypatch, varlen, q_len, causal, cross, splits, expected):
    builds, launches = [], []

    def build(**kwargs):
        builds.append(kwargs)
        return lambda *args, **kw: launches.append(kw)

    monkeypatch.setattr(fa, "_build_fp8", build)
    monkeypatch.setattr(fa, "_gpu_arch", lambda device: "gfx950")
    monkeypatch.setattr(fa, "_num_cu", lambda device: 256)
    monkeypatch.setattr(fa, "_fp8_auto_kv_splits", lambda *args, **kw: 4)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)

    with FakeTensorMode() as mode:

        def fake(shape, dtype=torch.float8_e4m3fn):
            return FakeTensor(
                mode,
                torch.empty(shape, dtype=dtype, device="meta"),
                torch.device("cuda:0"),
            )

        kv_len = 16384 if cross else q_len
        q_shape = (q_len, 12, 192) if varlen else (1, q_len, 12, 192)
        k_shape = (kv_len, 12, 192) if varlen else (1, kv_len, 12, 192)
        q, k, v = fake(q_shape), fake(k_shape), fake((*k_shape[:-1], 128))
        descale = fake((1,), torch.float32)
        kwargs = {
            "q_descale": descale,
            "k_descale": descale,
            "v_descale": descale,
            "causal": causal,
            "num_kv_splits": splits,
            "softmax_scale": 0.137,
            "return_lse": True,
        }
        if varlen:
            kwargs.update(
                cu_seqlens_q=fake((2,), torch.int32),
                cu_seqlens_kv=fake((2,), torch.int32),
                max_seqlen_q=q_len,
                max_seqlen_kv=kv_len,
                cross_seqlen=cross,
            )
        if expected is None:
            with pytest.raises(ValueError, match="split-K requires"):
                fa.flydsl_flash_attn_fp8_func(q, k, v, **kwargs)
            assert not builds and not launches
            return

        out, lse = fa.flydsl_flash_attn_fp8_func(q, k, v, **kwargs)
        assert out.shape == (*q_shape[:-1], 128)
        assert lse.shape == ((12, q_len) if varlen else (1, 12, q_len))
        assert builds[-1]["num_kv_splits"] == expected
        assert builds[-1]["cross_seqlen"] == cross
        assert launches[-1]["softmax_scale"] == 0.137
        assert ("workspace" in launches[-1]) == (expected > 1)

        if splits is None and expected > 1:
            # Autotuning must still respect its existing workspace budget.
            monkeypatch.setattr(fa, "_FP8_AUTOSPLIT_MAX_WS_BYTES", 0)
            fa.flydsl_flash_attn_fp8_func(q, k, v, **kwargs)
            assert builds[-1]["num_kv_splits"] == 1
            assert "workspace" not in launches[-1]
