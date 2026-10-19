# SPDX-License-Identifier: MIT
"""Public dispatcher tests with meta tensors; kernel numerics have a GPU gate."""

import importlib

import pytest
import torch

ua = importlib.import_module("aiter.ops.triton.attention.unified_attention")


class CaptureKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))

        return launch


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(ua, "DEVICE_ARCH", "gfx950")
    monkeypatch.setattr(ua.arch_info, "get_arch", lambda: "gfx950")
    monkeypatch.setattr(ua, "get_num_sms", lambda: 256)
    kernel = CaptureKernel()
    monkeypatch.setattr(ua, "_unified_attention_kernel_gfx950", kernel)
    monkeypatch.setattr(ua, "_reduce_segments_triton", lambda *a, **kw: None)

    def forbid(*args, **kwargs):
        raise AssertionError("Triton attention fallback")

    monkeypatch.setattr(ua, "_unified_attention_2d_triton", forbid)
    monkeypatch.setattr(ua, "_unified_attention_3d_triton", forbid)
    return kernel


def invoke(batch=1, qlen=4, context=16385, heads=16, kv=1, page=128):
    q = torch.empty(batch * qlen, heads, 128, device="meta", dtype=torch.bfloat16)
    k = torch.empty(8, kv, 8, page, 16, device="meta", dtype=ua.e4m3_dtype)
    v = torch.empty(8, kv, page // 16, 128, 16, device="meta", dtype=k.dtype)
    scales = torch.empty(8, kv, page, device="meta", dtype=torch.float32)
    ua.unified_attention(
        q,
        k,
        v,
        torch.empty_like(q),
        cu_seqlens_q=torch.empty(batch + 1, device="meta", dtype=torch.int32),
        seqused_k=torch.empty(batch, device="meta", dtype=torch.int32),
        max_seqlen_q=qlen,
        max_seqlen_k=context,
        softmax_scale=128**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.empty(batch, 2048, device="meta", dtype=torch.int32),
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        shuffled_kv_cache=True,
        k_scale_cache=scales,
        v_scale_cache=scales,
        backend="gluon",
    )


@pytest.mark.parametrize("qlen", [1, 2, 4])
@pytest.mark.parametrize("kv", [1, 2, 4, 8, 16])
def test_short_queries_split_context(capture, qlen, kv):
    invoke(qlen=qlen, kv=kv)
    assert len(capture.calls) == 1
    grid, kw = capture.calls[0]
    assert len(grid) == 3 and grid[-1] > 1
    assert not kw["ALL_DECODE"]
    assert kw["PER_TOKEN_KV_SCALES"] and kw["TILE_SIZE"] == 128
    assert kw["NUM_SPLITS"] == grid[-1]
    assert kw["NUM_BUFFERS"] == 1 and kw["num_warps"] == 4


def test_large_prefill_retains_query_parallelism(capture):
    invoke(batch=32, qlen=1024, context=2049, heads=64, kv=16)
    grid, kw = capture.calls[0]
    assert len(grid) == 2 and kw["NUM_SPLITS"] == 1


def test_unsupported_scaled_page_does_not_silently_fallback(capture):
    with pytest.raises(ValueError, match="Scaled Gluon"):
        invoke(page=64)
    assert capture.calls == []


def test_missing_gluon_kernel_is_an_error(capture, monkeypatch):
    monkeypatch.setattr(ua, "_unified_attention_kernel_gfx950", None)
    with pytest.raises(RuntimeError, match="refusing fallback"):
        invoke()
