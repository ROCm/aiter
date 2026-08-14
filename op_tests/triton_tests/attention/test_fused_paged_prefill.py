# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.fused_paged_prefill import (
    SUPPORTED_HEAD_DIMS,
    is_supported,
    mha_batch_prefill_func,
)

POOL_PAGES = 50000

# (batch, q_len, kv_len) -- kv_len > q_len is the extend/chunked-prefill case, where
# the causal diagonal is bottom-right aligned by delta = kv_len - q_len.
SHAPES = [
    (1, 128, 128),  # delta 0: the whole tile is on the diagonal
    (1, 128, 256),
    (2, 256, 1024),
    (2, 1024, 4096),
    (3, 300, 1500),  # nothing divides BLOCK_M/BLOCK_N
    (1, 512, 32768),  # long shared prefix, the shape this was written for
]

# delta = kv_len - q_len straddling BLOCK_N. A sibling implementation of this kernel
# returned NaN for every delta < BLOCK_N - 1, because the unmasked BULK range is empty
# there and the first tile was flushed without ever being masked. Cheap to keep pinned.
DELTAS = [0, 1, 16, 32, 62, 63, 64, 65, 128, 1024]


def ref_attention(q, k_cache, v_cache, cu_q, kv_indptr, kv_pages, sm_scale):
    """fp32 reference, one (request, head) at a time, chunked over kv to bound memory."""
    out = torch.empty_like(q)
    bs = cu_q.shape[0] - 1
    H = q.shape[1]
    for b in range(bs):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(kv_indptr[b]), int(kv_indptr[b + 1])
        q_len, kv_len = qe - qs, ke - ks
        pages = kv_pages[ks:ke].long()
        k = k_cache[pages, 0].float()  # [kv_len, D]
        v = v_cache[pages, 0].float()
        delta = kv_len - q_len
        qi = q[qs:qe].float()  # [q_len, H, D]
        pos = torch.arange(q_len, device=q.device)[:, None] + delta
        mask = torch.arange(kv_len, device=q.device)[None, :] <= pos
        for h in range(H):
            s = (qi[:, h] @ k.T) * sm_scale
            s = s.masked_fill(~mask, float("-inf"))
            p = torch.softmax(s, dim=-1)
            out[qs:qe, h] = (p @ v).to(out.dtype)
    return out


def make_inputs(bs, q_len, kv_len, H, D, Hkv=1, seed=0, device="cuda",
                dtype=torch.bfloat16, dist="normal"):
    """Shared-prefix page layout: every request shares the first (kv_len - q_len)
    pages and owns a unique tail, which is what prefix caching produces."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    prefix = kv_len - q_len
    perm = torch.randperm(POOL_PAGES, generator=g)
    shared = perm[:prefix]
    tails = perm[prefix : prefix + bs * q_len]
    idx = [torch.cat([shared, tails[i * q_len : (i + 1) * q_len]]) for i in range(bs)]
    cu_q = torch.arange(bs + 1, dtype=torch.int32) * q_len
    kv_indptr = torch.arange(bs + 1, dtype=torch.int32) * kv_len

    def mk(*sh):
        t = torch.randn(*sh, generator=g) * 0.5
        if dist == "spikes":
            # large-magnitude values in the LAST tenth of the pool, so the running
            # softmax max is revised late and the rescale path is exercised hard
            t[int(t.shape[0] * 0.9) :] *= 12.0
        return t.to(dtype).to(device)

    return (
        mk(bs * q_len, H, D),
        mk(POOL_PAGES, Hkv, D),
        mk(POOL_PAGES, Hkv, D),
        cu_q.to(device),
        kv_indptr.to(device),
        torch.cat(idx).to(torch.int32).to(device),
    )


def rms_rel(got, ref):
    got, ref = got.float(), ref.float()
    return ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("bs,q_len,kv_len", SHAPES)
@pytest.mark.parametrize("D", SUPPORTED_HEAD_DIMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_correctness(bs, q_len, kv_len, D, dtype):
    H = 8
    q, k, v, cu_q, kv_indptr, pages = make_inputs(bs, q_len, kv_len, H, D, dtype=dtype)
    sm_scale = D**-0.5
    got = mha_batch_prefill_func(
        q, k, v, cu_q, kv_indptr, pages, q_len, kv_len, softmax_scale=sm_scale
    )
    assert torch.isfinite(got).all(), "output contains NaN/Inf"
    ref = ref_attention(q, k, v, cu_q, kv_indptr, pages, sm_scale)
    # bf16 inputs and a bf16 output: ~2e-3 is the rounding floor, not slack
    assert rms_rel(got, ref) < 2e-2


@pytest.mark.parametrize("delta", DELTAS)
def test_delta_boundary_is_finite(delta):
    """delta < BLOCK_N must not poison the first tile. Regression pin."""
    q_len, D, H = 128, 256, 4
    q, k, v, cu_q, kv_indptr, pages = make_inputs(1, q_len, q_len + delta, H, D)
    sm_scale = D**-0.5
    got = mha_batch_prefill_func(
        q, k, v, cu_q, kv_indptr, pages, q_len, q_len + delta, softmax_scale=sm_scale
    )
    assert torch.isfinite(got).all(), f"NaN/Inf at delta={delta}"
    ref = ref_attention(q, k, v, cu_q, kv_indptr, pages, sm_scale)
    assert rms_rel(got, ref) < 2e-2


@pytest.mark.parametrize("prescale", [0, 1, 2])
def test_prescale_modes_agree(prescale, monkeypatch):
    """All three q-scaling modes must land inside the same tolerance. Mode 1 folds the
    full sm_scale*log2e into q (inexact in bf16), mode 2 folds only a power of two."""
    import json

    monkeypatch.setenv("AITER_FPP_CFG", json.dumps({"PRESCALE_Q": prescale}))
    q_len, kv_len, D, H = 512, 4096, 256, 8
    q, k, v, cu_q, kv_indptr, pages = make_inputs(2, q_len, kv_len, H, D)
    sm_scale = D**-0.5
    got = mha_batch_prefill_func(
        q, k, v, cu_q, kv_indptr, pages, q_len, kv_len, softmax_scale=sm_scale
    )
    ref = ref_attention(q, k, v, cu_q, kv_indptr, pages, sm_scale)
    assert rms_rel(got, ref) < 2e-2


def test_default_scaling_mode_is_exact():
    """The default must stay on a bf16-exact scaling mode.

    Mode 1 is 3.4% faster but its rounding error is data-dependent: on a distribution
    with large late kv values it measured 7.0x the CK-tile path's error against fp32
    while still passing the 2e-2 budget above, so that budget alone does not protect
    this choice.
    """
    from aiter.ops.triton.attention.fused_paged_prefill import _DEFAULT_CFG

    assert _DEFAULT_CFG["PRESCALE_Q"] in (0, 2)


@pytest.mark.parametrize("prescale", [1, 2])
def test_exact_mode_is_robust_to_late_kv_spikes(prescale, monkeypatch):
    """Pins the measurement behind the default: on the spiked distribution the exact
    mode holds near the bf16 floor while the folded mode degrades by ~7x."""
    import json

    monkeypatch.setenv("AITER_FPP_CFG", json.dumps({"PRESCALE_Q": prescale}))
    q_len, kv_len, D, H = 512, 8192, 256, 8
    q, k, v, cu_q, kv_indptr, pages = make_inputs(2, q_len, kv_len, H, D, dist="spikes")
    sm_scale = D**-0.5
    got = mha_batch_prefill_func(
        q, k, v, cu_q, kv_indptr, pages, q_len, kv_len, softmax_scale=sm_scale
    )
    ref = ref_attention(q, k, v, cu_q, kv_indptr, pages, sm_scale)
    err = rms_rel(got, ref)
    assert torch.isfinite(got).all()
    if prescale == 2:
        assert err < 2e-3, f"exact mode degraded on spiked kv: {err:.3e}"
    else:
        assert err < 2e-2


def test_unsupported_calls_are_rejected():
    """Every condition in is_supported() must actually be reported, in particular the
    two that would otherwise be SILENT: >1 kv head and an unsupported head dim."""
    q, k, *_ = make_inputs(1, 128, 256, 8, 256)

    assert is_supported(q, k)[0]

    # more than one kv head: the kernel has no kv-head offset, so this must be refused
    _, k2, *_ = make_inputs(1, 128, 256, 8, 256, Hkv=2)
    ok, why = is_supported(q, k2)
    assert not ok and "kv head" in why

    # head dim 128 is correct but slower than CK-tile, so it is out of scope by policy
    q128, k128, *_ = make_inputs(1, 128, 256, 8, 128)
    ok, why = is_supported(q128, k128)
    assert not ok and "head_dim" in why

    for kwargs, expect in [
        ({"causal": False}, "causal"),
        ({"logits_soft_cap": 30.0}, "soft_cap"),
        ({"window_size": (1024, 0)}, "window"),
        ({"return_lse": True}, "return_lse"),
        ({"alibi_slopes": torch.zeros(8, device=q.device)}, "alibi"),
        ({"sink_ptr": torch.zeros(8, device=q.device)}, "sink"),
        ({"k_descale": torch.ones(1, device=q.device)}, "descale"),
    ]:
        ok, why = is_supported(q, k, **kwargs)
        assert not ok and expect in why, f"{kwargs} -> ({ok}, {why!r})"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_adversarial_distribution(dtype):
    """Large-magnitude values late in the kv range force the online-softmax running
    max to be revised on a late tile, which is where a rescale bug would show."""
    bs, q_len, kv_len, H, D = 2, 512, 8192, 8, 256
    q, k, v, cu_q, kv_indptr, pages = make_inputs(
        bs, q_len, kv_len, H, D, dtype=dtype, dist="spikes"
    )
    sm_scale = D**-0.5
    got = mha_batch_prefill_func(
        q, k, v, cu_q, kv_indptr, pages, q_len, kv_len, softmax_scale=sm_scale
    )
    assert torch.isfinite(got).all(), "non-finite output on the spiked distribution"
    ref = ref_attention(q, k, v, cu_q, kv_indptr, pages, sm_scale)
    assert rms_rel(got, ref) < 2e-2


def test_large_shape():
    """One shape at the scale this kernel is actually for: a long shared prefix."""
    bs, q_len, kv_len, H, D = 2, 2048, 40960, 8, 256
    q, k, v, cu_q, kv_indptr, pages = make_inputs(bs, q_len, kv_len, H, D)
    sm_scale = D**-0.5
    got = mha_batch_prefill_func(
        q, k, v, cu_q, kv_indptr, pages, q_len, kv_len, softmax_scale=sm_scale
    )
    assert torch.isfinite(got).all()
    ref = ref_attention(q, k, v, cu_q, kv_indptr, pages, sm_scale)
    assert rms_rel(got, ref) < 2e-2
