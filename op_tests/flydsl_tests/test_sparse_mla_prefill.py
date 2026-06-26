# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for ``flydsl_sparse_mla_prefill`` (gfx942, Phase A).

Run:
    cd /home/AMD/samremes/dev/aiter
    python op_tests/flydsl_tests/test_sparse_mla_prefill.py
or:
    pytest op_tests/flydsl_tests/test_sparse_mla_prefill.py -v

Reference: a minimal reimplementation of vLLM's ``reference_mla_sparse_prefill``
(``vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py``). The kernel
computes both GEMMs in native fp8 (e4m3fnuz) MFMA, so inputs are fp8-rounded
before the reference dot to isolate fp8 quantization from kernel error.
"""

import math
import os
import sys

import torch

# ---- Bootstrap import paths for FlyDSL runtime (``flydsl`` package only).
# Kernel source lives in ``aiter/ops/flydsl/kernels/sparse_mla_prefill.py``.
_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if os.path.isdir(os.path.join(_AITER_ROOT, "aiter")) and _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)
_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 3)))
_flydsl_root = os.path.join(_DEV, "FlyDSL")


def _flydsl_imports_ok() -> bool:
    try:
        import flydsl  # noqa: F401
        from flydsl._mlir import ir  # noqa: F401

        return True
    except Exception:
        return False


def _ensure_flydsl_importable() -> bool:
    if _flydsl_imports_ok():
        return True
    # Candidate dirs that may hold a *built* flydsl (with compiled _mlir):
    #   - $FLYDSL_PKGS / $FLYDSL_REPO env vars
    #   - a built-package dir next to the repos (e.g. dev/.r1_flydsl_pkgs)
    #   - FlyDSL/build-fly/python_packages (source build output)
    #   - FlyDSL/python (only if it carries compiled bindings)
    cands = []
    for env in ("FLYDSL_PKGS", "FLYDSL_REPO", "FLYDSL_HOME"):
        v = os.environ.get(env)
        if v:
            cands.append(v)
    cands += [
        os.path.join(_DEV, ".r1_flydsl_pkgs"),
        os.path.join(_flydsl_root, "build-fly", "python_packages"),
        os.path.join(_flydsl_root, "python"),
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "flydsl")) and c not in sys.path:
            sys.path.insert(0, c)
            if _flydsl_imports_ok():
                return True
    return _flydsl_imports_ok()


_HAS_FLYDSL = _ensure_flydsl_importable()


def _is_gfx942() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return arch.lower().split(":")[0].startswith("gfx942")


LOG2E = math.log2(math.e)


def reference_prefill(q_ref, kv_ref, indices_dense, scale, d_v):
    """Dense oracle. q_ref: [sq,H,D] f32; kv_ref: [skv,D] f32;
    indices_dense: [sq,topk] int. Returns out [sq,H,d_v] f32."""
    sq, H, D = q_ref.shape
    skv = kv_ref.shape[0]
    invalid = (indices_dense < 0) | (indices_dense >= skv)
    idx = indices_dense.clone()
    idx[invalid] = 0
    kvs = kv_ref[idx]  # [sq, topk, D]
    score = torch.einsum("qhd,qkd->qhk", q_ref, kvs)  # [sq,H,topk]
    score = score.masked_fill(invalid.unsqueeze(1), float("-inf"))
    score = score * (scale * LOG2E)
    m = score.max(dim=-1, keepdim=True).values
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.exp2(score - m)
    denom = p.sum(dim=-1, keepdim=True)
    p = torch.where(denom > 0, p / denom, torch.zeros_like(p))
    out = torch.einsum("qhk,qkd->qhd", p, kvs[:, :, :d_v])
    return out


def _fp8_round_bf16(x_bf16):
    """Round a bf16 tensor through e4m3fnuz and back to bf16 (idempotent cast)."""
    return x_bf16.to(torch.float8_e4m3fnuz).to(torch.bfloat16)


def _build_inputs(sq, topk, skv, H=128, D=512, seed=0, device="cuda", invalid_ratio=0.0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.randn(sq, H, D, generator=g, dtype=torch.bfloat16, device=device) * 0.3)
    kv = (torch.randn(skv, D, generator=g, dtype=torch.bfloat16, device=device) * 0.3)

    # fp8-round inputs so the kernel's internal cast is idempotent and the
    # reference dot uses the same fnuz-dequantized values.
    q_in = _fp8_round_bf16(q)
    kv_fp8 = kv.to(torch.float8_e4m3fnuz)
    kv_ref = kv_fp8.to(torch.float32)
    q_ref = q_in.to(torch.float32)

    indices = torch.randint(0, skv, (sq, topk), generator=g, dtype=torch.int32, device=device)
    if invalid_ratio > 0.0:
        mask = torch.rand(sq, topk, generator=g, device=device) < invalid_ratio
        indices[mask] = -1
    return q_in, kv_fp8, kv_ref, q_ref, indices


def _run_csr(q_in, kv_fp8, indices_dense, indptr, scale, num_queries, H=128, D=512):
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    out = torch.empty(num_queries, H, D, dtype=torch.bfloat16, device=q_in.device)
    kv3d = kv_fp8.reshape(kv_fp8.shape[0], 1, kv_fp8.shape[1])
    flydsl_sparse_mla_prefill(
        q_in,
        kv3d,
        indices_dense.reshape(-1),
        indptr,
        out,
    )
    return out


def _metrics(out, ref):
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    cos = torch.nn.functional.cosine_similarity(o, r, dim=-1)
    # guard rows where ref is ~0 (cosine undefined) -> treat as matching iff out ~0
    ref_norm = r.norm(dim=-1)
    nonzero = ref_norm > 1e-4
    cos_nz = cos[nonzero] if nonzero.any() else torch.ones(1, device=o.device)
    max_abs = (o - r).abs().max().item()
    return cos_nz.mean().item(), cos_nz.min().item(), max_abs


def test_basic():
    sq, topk, skv = 4, 256, 1024
    scale = 1.0 / math.sqrt(512)
    q_in, kv_fp8, kv_ref, q_ref, indices = _build_inputs(sq, topk, skv, seed=1)
    indptr = (torch.arange(sq + 1, dtype=torch.int32, device=q_in.device) * topk)
    out = _run_csr(q_in, kv_fp8, indices, indptr, scale, sq)
    ref = reference_prefill(q_ref, kv_ref, indices.long(), scale, 512)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[basic] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_all_invalid():
    sq, topk, skv = 2, 128, 512
    scale = 1.0 / math.sqrt(512)
    q_in, kv_fp8, kv_ref, q_ref, indices = _build_inputs(sq, topk, skv, seed=2)
    indices[:] = -1
    indptr = (torch.arange(sq + 1, dtype=torch.int32, device=q_in.device) * topk)
    out = _run_csr(q_in, kv_fp8, indices, indptr, scale, sq)
    print(f"[all_invalid] max_abs_out={out.abs().max().item():.6f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.abs().max().item() == 0.0, "all-invalid must produce zero output"


def test_empty_kv_len():
    # 3 queries: q0 normal, q1 empty (kv_len=0), q2 normal.
    topk, skv = 128, 512
    scale = 1.0 / math.sqrt(512)
    q_in, kv_fp8, kv_ref, q_ref, indices = _build_inputs(3, topk, skv, seed=3)
    # CSR with a zero-length middle segment.
    idx0 = indices[0]
    idx2 = indices[2]
    flat = torch.cat([idx0, idx2]).contiguous()
    indptr = torch.tensor([0, topk, topk, 2 * topk], dtype=torch.int32, device=q_in.device)

    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    out = torch.empty(3, 128, 512, dtype=torch.bfloat16, device=q_in.device)
    kv3d = kv_fp8.reshape(skv, 1, 512)
    flydsl_sparse_mla_prefill(q_in, kv3d, flat, indptr, out)

    assert not torch.isnan(out).any(), "NaN in output"
    assert out[1].abs().max().item() == 0.0, "empty query row must be zero"

    # Verify q0/q2 against the oracle.
    dense = torch.stack([idx0, idx2]).long()
    ref = reference_prefill(
        torch.stack([q_ref[0], q_ref[2]]), kv_ref, dense, scale, 512
    )
    got = torch.stack([out[0], out[2]])
    cos_mean, cos_min, max_abs = _metrics(got, ref)
    print(f"[empty] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


# ===========================================================================
# Phase B packed fp8_ds_mla cache coverage.
#
# The packer / reader byte layout is copied from vLLM's
# ``tests/kernels/attention/test_rocm_triton_attn_dsv4.py`` (NOT imported, to
# avoid a vllm dependency).  See docs/sparse-mla-prefill/01 Section 6.1.
#
# The FlyDSL B-phase kernel computes every GEMM in native fp8 (e4m3fnuz) MFMA:
#   - NoPE (448): region0 fnuz bytes used directly; region1 OCP bytes decoded
#     and re-encoded to fnuz (x2 exponent correction); UE8M0 (power-of-2) scale
#     folded as an exponent shift.
#   - RoPE (64): bf16 cache tail re-quantized to fnuz fp8 ("block 7").
#   - Q (512): bf16 -> fnuz fp8 in-kernel.
# The references below replicate that exact fp8 rounding so the test isolates
# kernel-logic error from fp8 quantization error (the "fp8-rounded ref" allowed
# by docs/sparse-mla-prefill/03).
# ===========================================================================

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
PACKED_HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM  # 512
_FNUZ = torch.float8_e4m3fnuz
_OCP = torch.float8_e4m3fn


def _pack_fp8_ds_mla_cache(kv, block_size, is_extra=False, scale_byte=127):
    """Pack [num_tokens, 512] bf16 KV into the fp8_ds_mla uint8 cache.

    Layout (per docs Section 6.1): cache[num_blocks, block_size, 584] uint8.
      token_base = block_idx*block_size*584 + pos*576
        [token_base            : +448]  448 fp8 NoPE  (OCP if is_extra else fnuz)
        [token_base+448        : +128]  64 bf16 RoPE  (= 128 bytes)
      scale_base = block_idx*block_size*584 + block_size*576 + pos*8
        [scale_base : scale_base+7]     7 UE8M0 exponent bytes (+1 pad)

    ``scale_byte`` may be an int (filled into all 7) or a length-7 iterable for
    per-64-block scales (UE8M0 != 1 testing).
    """
    assert kv.shape[-1] == PACKED_HEAD_DIM
    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros((num_blocks, block_size, 584), dtype=torch.uint8, device=kv.device)
    cache_flat = cache.view(torch.uint8).flatten()
    nope_dtype = _OCP if is_extra else _FNUZ
    kv_nope_fp8 = kv[:, :NOPE_HEAD_DIM].to(nope_dtype).view(torch.uint8)
    kv_rope_u8 = kv[:, NOPE_HEAD_DIM:].contiguous().view(torch.uint8)
    for slot in range(num_tokens):
        block_idx = slot // block_size
        pos = slot % block_size
        block_base = block_idx * cache.stride(0)
        token_base = block_base + pos * 576
        scale_base = block_base + block_size * 576 + pos * 8
        cache_flat[token_base : token_base + NOPE_HEAD_DIM].copy_(kv_nope_fp8[slot])
        cache_flat[token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2].copy_(
            kv_rope_u8[slot]
        )
        if isinstance(scale_byte, int):
            cache_flat[scale_base : scale_base + 7].fill_(scale_byte)
        else:
            sb = torch.tensor(list(scale_byte), dtype=torch.uint8, device=kv.device)
            cache_flat[scale_base : scale_base + 7].copy_(sb)
    return cache


def _dequant_row_like_kernel(cache, slot, block_size, is_extra=False):
    """Read one cache row and reproduce the FlyDSL kernel's fp8 rounding.

    Returns f32 [512] = the effective KV vector the kernel feeds into MFMA.
    """
    cache_flat = cache.view(torch.uint8).flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576
    scale_base = block_base + block_size * 576 + pos * 8

    nope_u8 = cache_flat[token_base : token_base + NOPE_HEAD_DIM]
    if is_extra:
        # OCP bytes -> true OCP value -> (kernel re-encodes to fnuz).
        nope = nope_u8.view(_OCP).to(torch.float32)
    else:
        nope = nope_u8.view(_FNUZ).to(torch.float32)

    enc = cache_flat[scale_base : scale_base + 7].to(torch.float32)  # 7 UE8M0 bytes
    blk_scale = torch.exp2(enc - 127.0)  # [7] per-64-col block
    nope = nope * blk_scale.repeat_interleave(64)
    if is_extra:
        # kernel bakes the scaled value back into fnuz fp8 in LDS.
        nope = nope.to(_FNUZ).to(torch.float32)
    else:
        # fnuz convert path also re-encodes; lossless for power-of-2 scales.
        nope = nope.to(_FNUZ).to(torch.float32)

    rope_u8 = cache_flat[token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2]
    rope = rope_u8.view(torch.bfloat16).to(torch.float32)
    rope = rope.to(_FNUZ).to(torch.float32)  # kernel re-quantizes RoPE to fnuz fp8
    return torch.cat([nope, rope])


def _ragged_from_rows(rows, device):
    flat = [slot for row in rows for slot in row]
    indptr = [0]
    for row in rows:
        indptr.append(indptr[-1] + len(row))
    return (
        torch.tensor(flat, dtype=torch.int32, device=device),
        torch.tensor(indptr, dtype=torch.int32, device=device),
    )


def _ref_prefill_packed(q, regions, scale, attn_sink, block_size):
    """f32 oracle over packed caches, reproducing the kernel's fp8 rounding.

    q: [sq, H, 512] bf16. regions: list of (cache, rows_list, is_extra).
    rows_list[query] = list of slot ids for that query in this region (may have
    invalid slots that the caller already filtered, or be empty).
    """
    sq, H, D = q.shape
    q_eff = q.to(_FNUZ).to(torch.float32)  # kernel quantizes all of Q to fnuz
    out = torch.zeros(sq, H, D, dtype=torch.float32, device=q.device)
    for qi in range(sq):
        row_kv = []
        for cache, rows_list, is_extra in regions:
            for slot in rows_list[qi]:
                row_kv.append(_dequant_row_like_kernel(cache, int(slot), block_size, is_extra))
        if not row_kv:
            continue
        kv = torch.stack(row_kv).to(q.device)  # [k, 512]
        for h in range(H):
            scores = torch.mv(kv, q_eff[qi, h]) * scale
            if attn_sink is not None:
                scores_s = torch.cat([scores, attn_sink[h].float().reshape(1)])
                probs = torch.softmax(scores_s, dim=0)[:-1]
            else:
                probs = torch.softmax(scores, dim=0)
            out[qi, h] = torch.sum(probs[:, None] * kv, dim=0)
    return out.to(torch.bfloat16)


def _dequant_row_bf16_rope(cache, slot, block_size, is_extra=False):
    """bf16-RoPE contract dequant (vLLM-faithful for the QK dot).

    Returns ``(kv_score, kv_value)`` f32 [512]:
      - ``kv_score`` feeds the QK dot: NoPE is fp8-rounded (kernel fp8 MFMA), but
        the 64 RoPE dims stay **bf16** (NOT re-quantized to fp8) -- the P0.2 fix.
      - ``kv_value`` feeds P@V (output): identical NoPE, and the RoPE tail is the
        fp8 re-quant the kernel still keeps in cache block 7 for the V read.
    """
    cache_flat = cache.view(torch.uint8).flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576
    scale_base = block_base + block_size * 576 + pos * 8

    nope_u8 = cache_flat[token_base : token_base + NOPE_HEAD_DIM]
    nope = (nope_u8.view(_OCP) if is_extra else nope_u8.view(_FNUZ)).to(torch.float32)
    enc = cache_flat[scale_base : scale_base + 7].to(torch.float32)
    blk_scale = torch.exp2(enc - 127.0)
    nope = nope * blk_scale.repeat_interleave(64)
    nope = nope.to(_FNUZ).to(torch.float32)  # kernel re-encodes NoPE to fnuz in LDS

    rope_u8 = cache_flat[token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2]
    rope_bf16 = rope_u8.view(torch.bfloat16).to(torch.float32)  # QK dot: stays bf16
    rope_fp8 = rope_bf16.to(_FNUZ).to(torch.float32)  # V read: kernel keeps fp8 block 7
    return torch.cat([nope, rope_bf16]), torch.cat([nope, rope_fp8])


def _ref_prefill_packed_bf16rope(q, regions, scale, attn_sink, block_size):
    """bf16-RoPE oracle: QK dots the RoPE tail in bf16 (Q RoPE stays bf16 too),
    NoPE in fp8; P@V uses the fp8-rope V (matching the kernel's unchanged GEMM2).
    """
    sq, H, D = q.shape
    q_nope = q[:, :, :NOPE_HEAD_DIM].to(_FNUZ).to(torch.float32)  # Q NoPE -> fp8
    q_rope = q[:, :, NOPE_HEAD_DIM:].to(torch.float32)  # Q RoPE stays bf16
    q_score = torch.cat([q_nope, q_rope], dim=-1)
    out = torch.zeros(sq, H, D, dtype=torch.float32, device=q.device)
    for qi in range(sq):
        kv_score, kv_value = [], []
        for cache, rows_list, is_extra in regions:
            for slot in rows_list[qi]:
                ks, kv = _dequant_row_bf16_rope(cache, int(slot), block_size, is_extra)
                kv_score.append(ks)
                kv_value.append(kv)
        if not kv_score:
            continue
        ks = torch.stack(kv_score).to(q.device)  # [k, 512]
        kvv = torch.stack(kv_value).to(q.device)
        for h in range(H):
            scores = torch.mv(ks, q_score[qi, h]) * scale
            if attn_sink is not None:
                scores_s = torch.cat([scores, attn_sink[h].float().reshape(1)])
                probs = torch.softmax(scores_s, dim=0)[:-1]
            else:
                probs = torch.softmax(scores, dim=0)
            out[qi, h] = torch.sum(probs[:, None] * kvv, dim=0)
    return out.to(torch.bfloat16)


def _identity_block_table(num_slots, block_size, device):
    """block_table[req=0][b] = b (logical block == physical block)."""
    num_blocks = (num_slots + block_size - 1) // block_size
    return torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(1, num_blocks)


def _gen_kv(num_tokens, seed, device="cuda", scale=0.125):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(num_tokens, PACKED_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * scale)


def _gen_q(sq, H, seed, device="cuda", scale=0.125):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(sq, H, PACKED_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * scale)


def test_b2_two_region():
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    main_tokens, extra_tokens = 400, 300
    main_kv = _gen_kv(main_tokens, seed=51)
    extra_kv = _gen_kv(extra_tokens, seed=52)
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, is_extra=False)   # fnuz
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)  # OCP
    main_rows = [[5, 200, 7, 64], [0, 1, 2, 3, 4]]
    extra_rows = [[10, 11, 12], [50, 51, 52, 53, 100, 299]]
    sq = len(main_rows)
    q = _gen_q(sq, H, seed=53)
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)
    extra_indices, extra_indptr = _ragged_from_rows(extra_rows, device)
    main_bt = _identity_block_table(main_tokens, block_size, device)
    extra_bt = _identity_block_table(extra_tokens, block_size, device)
    sink = (torch.randn(H, dtype=torch.float32, device=device) * 0.4)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill_2region(
        q, out,
        main_cache, main_indices, main_indptr, main_bt,
        extra_cache, extra_indices, extra_indptr, extra_bt,
        block_size=block_size, attn_sink=sink,
    )
    ref = _ref_prefill_packed(
        q, [(main_cache, main_rows, False), (extra_cache, extra_rows, True)], scale, sink, block_size
    )
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[b2_two_region] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.97, f"cosine too low: {cos_mean}"


def test_b2_ue8m0_non_unity():
    """Non-unity per-64-block UE8M0 scale bytes in each region.  region0 (fnuz
    SWA) needs ``main_scale_mode='ue8m0'`` to route through the convert load;
    region1 (OCP compressed) always reads its scale bytes via convert."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    main_tokens, extra_tokens = 300, 300
    main_kv = _gen_kv(main_tokens, seed=81, scale=0.0625)
    extra_kv = _gen_kv(extra_tokens, seed=82, scale=0.0625)
    # per-64-block scale bytes that differ across the 7 NoPE blocks (UE8M0 != 1)
    region0_scales = [127, 128, 126, 128, 127, 125, 128]
    region1_scales = [128, 126, 127, 128, 127, 128, 126]
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, is_extra=False, scale_byte=region0_scales)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True, scale_byte=region1_scales)
    main_rows = [[5, 6, 7, 64, 200], [1, 2, 3]]
    extra_rows = [[10, 11, 12, 70], [50, 51]]
    sq = len(main_rows)
    q = _gen_q(sq, H, seed=83, scale=0.0625)
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)
    extra_indices, extra_indptr = _ragged_from_rows(extra_rows, device)
    main_bt = _identity_block_table(main_tokens, block_size, device)
    extra_bt = _identity_block_table(extra_tokens, block_size, device)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill_2region(
        q, out,
        main_cache, main_indices, main_indptr, main_bt,
        extra_cache, extra_indices, extra_indptr, extra_bt,
        block_size=block_size, main_scale_mode="ue8m0",
    )
    ref = _ref_prefill_packed(
        q, [(main_cache, main_rows, False), (extra_cache, extra_rows, True)], scale, None, block_size
    )
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[b2_ue8m0] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.97, f"cosine too low: {cos_mean}"


def test_b2_region0_empty_rejected():
    """Region0 (main/SWA) must be non-empty per query: the B2 kernel seeds the
    shared softmax state from region0 tile 0, so an empty region0 either faults
    (fully empty main_indices) or silently skips region1.  The host wrapper must
    reject both rather than launch."""
    import pytest

    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region

    device = "cuda"
    block_size = 64
    H = 128
    main_tokens = extra_tokens = 400
    main_kv = _gen_kv(main_tokens, seed=71)
    extra_kv = _gen_kv(extra_tokens, seed=72)
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, is_extra=False)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)
    main_bt = _identity_block_table(main_tokens, block_size, device)
    extra_bt = _identity_block_table(extra_tokens, block_size, device)

    def _run(main_rows, extra_rows):
        sq = len(main_rows)
        q = _gen_q(sq, H, seed=73)
        m_idx, m_iptr = _ragged_from_rows(main_rows, device)
        e_idx, e_iptr = _ragged_from_rows(extra_rows, device)
        out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
        flydsl_sparse_mla_prefill_2region(
            q, out,
            main_cache, m_idx, m_iptr, main_bt,
            extra_cache, e_idx, e_iptr, extra_bt,
            block_size=block_size,
        )

    # fully empty region0 buffer (would fault on the GPU)
    with pytest.raises(ValueError):
        _run([[]], [[10, 11, 12, 13]])
    # per-query empty region0 segment with a non-empty buffer (would drop region1)
    with pytest.raises(ValueError):
        _run([[], [5, 6]], [[10, 11, 12], []])
    print("[b2_region0_empty_rejected] host guard OK")


def test_b2_multitile():
    """B2 where BOTH regions span multiple tiles (>32 entries each), so the
    per-tile region select exercises region0 tiles [1, n0) AND region1 tiles."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    main_tokens, extra_tokens = 500, 400
    main_kv = _gen_kv(main_tokens, seed=61)
    extra_kv = _gen_kv(extra_tokens, seed=62)
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, is_extra=False)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)
    # q0: 70 main (3 tiles) + 40 extra (2 tiles); q1: 33 main (2 tiles) + 95 extra (3 tiles)
    g = torch.Generator(device=device).manual_seed(63)
    main_rows = [
        torch.randint(0, main_tokens, (70,), generator=g, device=device).tolist(),
        torch.randint(0, main_tokens, (33,), generator=g, device=device).tolist(),
    ]
    extra_rows = [
        torch.randint(0, extra_tokens, (40,), generator=g, device=device).tolist(),
        torch.randint(0, extra_tokens, (95,), generator=g, device=device).tolist(),
    ]
    sq = len(main_rows)
    q = _gen_q(sq, H, seed=64)
    main_indices, main_indptr = _ragged_from_rows(main_rows, device)
    extra_indices, extra_indptr = _ragged_from_rows(extra_rows, device)
    main_bt = _identity_block_table(main_tokens, block_size, device)
    extra_bt = _identity_block_table(extra_tokens, block_size, device)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill_2region(
        q, out,
        main_cache, main_indices, main_indptr, main_bt,
        extra_cache, extra_indices, extra_indptr, extra_bt,
        block_size=block_size,
    )
    ref = _ref_prefill_packed(
        q, [(main_cache, main_rows, False), (extra_cache, extra_rows, True)], scale, None, block_size
    )
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[b2_multitile] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.97, f"cosine too low: {cos_mean}"


# ===========================================================================
# P0.2 fix: bf16-RoPE split QK dot.  NoPE (448) stays fp8 MFMA; the 64 RoPE
# dims are dotted in bf16 (Q RoPE stays bf16, K RoPE read from the bf16 cache
# tail -- NOT re-quantized to fp8).  Gated behind the non-default ``rope_bf16``
# compile flag, so all the fp8-RoPE tests above are unchanged.  These tests use
# a vLLM-contract reference (``_ref_prefill_packed_bf16rope``) that does NOT
# re-quantize RoPE, and a RoPE-dominant adversary so the difference is visible.
# ===========================================================================


def _abs_metrics(out, ref):
    o = out.float().reshape(-1)
    r = ref.float().reshape(-1)
    return (o - r).abs().max().item(), (o - r).abs().mean().item()


def test_b1_rope_bf16():
    """Single-region DSv4 packed path with rope_bf16=True vs the bf16-RoPE
    reference.  RoPE-dominant adversary (near-zero NoPE) so the bf16-vs-fp8 RoPE
    contract difference dominates the score."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    num_tokens = 320

    g = torch.Generator(device=device).manual_seed(401)
    kv = torch.empty(num_tokens, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv[:, :NOPE_HEAD_DIM] = torch.randn(num_tokens, NOPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 0.004
    kv[:, NOPE_HEAD_DIM:] = torch.randn(num_tokens, ROPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 1.2
    q = torch.empty(3, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    q[:, :, :NOPE_HEAD_DIM] = torch.randn(3, H, NOPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 0.004
    q[:, :, NOPE_HEAD_DIM:] = torch.randn(3, H, ROPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 1.2

    cache = _pack_fp8_ds_mla_cache(kv, block_size, is_extra=False)
    # Moderate KV-row counts: softmax is selective enough that the systematic
    # fp8-RoPE rounding bias shifts weights (separating the two references),
    # but not so one-hot that per-element output error blows past the envelope.
    rows = [
        torch.randint(0, num_tokens, (12,), generator=g, device=device).tolist(),
        torch.randint(0, num_tokens, (10,), generator=g, device=device).tolist(),
        torch.randint(0, num_tokens, (11,), generator=g, device=device).tolist(),
    ]
    sq = len(rows)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True, rope_bf16=True,
    )
    regions = [(cache, rows, False)]
    ref_bf16 = _ref_prefill_packed_bf16rope(q, regions, scale, None, block_size)
    ref_fp8 = _ref_prefill_packed(q, regions, scale, None, block_size)

    cos_mean, cos_min, _ = _metrics(out, ref_bf16)
    max_abs_bf16, mean_abs_bf16 = _abs_metrics(out, ref_bf16)
    max_abs_fp8, mean_abs_fp8 = _abs_metrics(out, ref_fp8)
    ref_div_max, ref_div_mean = _abs_metrics(ref_fp8, ref_bf16)
    print(
        f"[b1_rope_bf16] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} "
        f"mean_abs(vs bf16-ref)={mean_abs_bf16:.5f} mean_abs(vs fp8-ref)={mean_abs_fp8:.5f} "
        f"| fp8-vs-bf16 ref divergence: max={ref_div_max:.4f} mean={ref_div_mean:.5f}"
    )
    assert not torch.isnan(out).any(), "NaN in output"
    # Contract (assert_close, not cosine-only): the bf16 kernel tracks the
    # bf16-RoPE reference within the fp8/bf16 output envelope of a deliberately
    # RoPE-dominant adversary (mean error ~1e-3; a few elements at fp8 step).
    torch.testing.assert_close(out.float(), ref_bf16.float(), atol=6.0e-2, rtol=5.0e-2)
    # Primary discriminating gate: the kernel is closer to the bf16-RoPE
    # reference than to the old fp8-RoPE reference (proves it does NOT requant
    # RoPE to fp8), and the adversary genuinely separates the two contracts.
    assert mean_abs_bf16 < mean_abs_fp8, (
        f"rope_bf16 kernel not closer to bf16 ref ({mean_abs_bf16}) than fp8 ref ({mean_abs_fp8})"
    )
    assert ref_div_mean > 5e-4, "adversary failed to separate fp8 vs bf16 RoPE references"


def test_b2_rope_bf16():
    """Two-region (SWA fnuz + compressed OCP) with rope_bf16=True vs the
    bf16-RoPE reference, including attn_sink and a RoPE-dominant adversary."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    main_tokens, extra_tokens = 400, 300

    def _adv_kv(n, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        kv = torch.empty(n, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
        kv[:, :NOPE_HEAD_DIM] = torch.randn(n, NOPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 0.005
        kv[:, NOPE_HEAD_DIM:] = torch.randn(n, ROPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 1.2
        return kv

    main_kv = _adv_kv(main_tokens, 451)
    extra_kv = _adv_kv(extra_tokens, 452)
    main_cache = _pack_fp8_ds_mla_cache(main_kv, block_size, is_extra=False)
    extra_cache = _pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)
    g = torch.Generator(device=device).manual_seed(453)
    main_rows = [
        torch.randint(0, main_tokens, (10,), generator=g, device=device).tolist(),
        torch.randint(0, main_tokens, (8,), generator=g, device=device).tolist(),
    ]
    extra_rows = [
        torch.randint(0, extra_tokens, (6,), generator=g, device=device).tolist(),
        torch.randint(0, extra_tokens, (12,), generator=g, device=device).tolist(),
    ]
    sq = len(main_rows)
    q = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    q[:, :, :NOPE_HEAD_DIM] = torch.randn(sq, H, NOPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 0.005
    q[:, :, NOPE_HEAD_DIM:] = torch.randn(sq, H, ROPE_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * 1.2

    main_indices, main_indptr = _ragged_from_rows(main_rows, device)
    extra_indices, extra_indptr = _ragged_from_rows(extra_rows, device)
    main_bt = _identity_block_table(main_tokens, block_size, device)
    extra_bt = _identity_block_table(extra_tokens, block_size, device)
    sink = (torch.randn(H, dtype=torch.float32, device=device) * 0.3)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill_2region(
        q, out,
        main_cache, main_indices, main_indptr, main_bt,
        extra_cache, extra_indices, extra_indptr, extra_bt,
        block_size=block_size, attn_sink=sink, rope_bf16=True,
    )
    regions = [(main_cache, main_rows, False), (extra_cache, extra_rows, True)]
    ref_bf16 = _ref_prefill_packed_bf16rope(q, regions, scale, sink, block_size)
    ref_fp8 = _ref_prefill_packed(q, regions, scale, sink, block_size)

    cos_mean, cos_min, _ = _metrics(out, ref_bf16)
    max_abs_bf16, mean_abs_bf16 = _abs_metrics(out, ref_bf16)
    max_abs_fp8, mean_abs_fp8 = _abs_metrics(out, ref_fp8)
    ref_div_max, ref_div_mean = _abs_metrics(ref_fp8, ref_bf16)
    print(
        f"[b2_rope_bf16] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} "
        f"mean_abs(vs bf16-ref)={mean_abs_bf16:.5f} mean_abs(vs fp8-ref)={mean_abs_fp8:.5f} "
        f"| fp8-vs-bf16 ref divergence: max={ref_div_max:.4f} mean={ref_div_mean:.5f}"
    )
    assert not torch.isnan(out).any(), "NaN in output"
    torch.testing.assert_close(out.float(), ref_bf16.float(), atol=6.0e-2, rtol=5.0e-2)
    assert mean_abs_bf16 < mean_abs_fp8, (
        f"rope_bf16 kernel not closer to bf16 ref ({mean_abs_bf16}) than fp8 ref ({mean_abs_fp8})"
    )
    assert ref_div_mean > 5e-4, "adversary failed to separate fp8 vs bf16 RoPE references"


# ===========================================================================
# GLM-5 / DeepSeek-V3.2 (ROCM_AITER_MLA_SPARSE): single-region flat fp8 cache,
# head_dim=576 (512 latent + 64 rope, both fp8), per-tensor scale, no sink.
# Oracle helpers live in sparse_mla_prefill_ref.py (shared with the bench).
# ===========================================================================
from op_tests.flydsl_tests.sparse_mla_prefill_ref import (  # noqa: E402
    GLM_HEAD_DIM,
    GLM_V_DIM,
    default_scale_glm,
    gen_kv_glm,
    gen_q_glm,
    pack_glm_fp8_cache,
    ref_prefill_glm,
)


def test_glm576_paged_basic():
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 64
    H = 128
    scale = default_scale_glm()
    num_tokens = 600
    kv = gen_kv_glm(num_tokens, seed=71)
    cache = pack_glm_fp8_cache(kv, block_size)  # kv_scale=1.0
    rows = [[5, 200, 7, 400, 63, 64, 599, 128], [0, 1, 2, 3], [300, 301, 302, 303, 304]]
    sq = len(rows)
    q = gen_q_glm(sq, H, seed=72)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.empty(sq, H, GLM_V_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
        scale_mode="per_tensor",
    )
    ref = ref_prefill_glm(q, cache, rows, scale, block_size)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[glm576_basic] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.shape[-1] == GLM_V_DIM
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_glm576_per_tensor_scale():
    """Non-trivial per-tensor kv_scale (!= 1) folded into QK scale + output."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 32
    H = 128
    scale = default_scale_glm()
    kv_scale = 2.0
    num_tokens = 300
    # Larger true values; packer stores kv/kv_scale to keep fp8 in range.
    kv = gen_kv_glm(num_tokens, seed=81, scale=0.25)
    cache = pack_glm_fp8_cache(kv, block_size, kv_scale=kv_scale)
    rows = [[5, 200, 7, 64, 63, 31, 32, 33], [10, 11, 12, 13, 14, 15, 16]]
    sq = len(rows)
    q = gen_q_glm(sq, H, seed=82, scale=0.25)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.empty(sq, H, GLM_V_DIM, dtype=torch.bfloat16, device=device)
    kv_scale_t = torch.tensor([kv_scale], dtype=torch.float32, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
        scale_mode="per_tensor", kv_scale=kv_scale_t,
    )
    ref = ref_prefill_glm(q, cache, rows, scale, block_size, kv_scale=kv_scale)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[glm576_pertensor] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_glm576_q_scale():
    """Non-trivial per-tensor q_scale applied during bf16->fp8 Q quant."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 32
    H = 128
    scale = default_scale_glm()
    q_scale = 1.5
    num_tokens = 256
    kv = gen_kv_glm(num_tokens, seed=91, scale=0.2)
    cache = pack_glm_fp8_cache(kv, block_size)
    rows = [[5, 20, 40, 60, 80], [1, 2, 3, 4, 5, 6]]
    sq = len(rows)
    q = gen_q_glm(sq, H, seed=92, scale=0.2)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.empty(sq, H, GLM_V_DIM, dtype=torch.bfloat16, device=device)
    q_scale_t = torch.tensor([q_scale], dtype=torch.float32, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
        scale_mode="per_tensor", q_scale=q_scale_t,
    )
    ref = ref_prefill_glm(q, cache, rows, scale, block_size, q_scale=q_scale)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[glm576_qscale] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_glm576_two_tile():
    """Exactly two tiles per query (BLOCK_N=32 -> 33..64 entries).  The GLM
    multi-tile path's first/last tile handling must work when the middle loop
    has zero iterations (n0_tiles == 2)."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 64
    H = 128
    scale = default_scale_glm()
    num_tokens = 700
    kv = gen_kv_glm(num_tokens, seed=171)
    cache = pack_glm_fp8_cache(kv, block_size)
    g = torch.Generator(device=device).manual_seed(172)
    # row lengths landing in (BLOCK_N, 2*BLOCK_N] -> exactly two 32-row tiles
    rows = [
        torch.randint(0, num_tokens, (33,), generator=g, device=device).tolist(),
        torch.randint(0, num_tokens, (64,), generator=g, device=device).tolist(),
        torch.randint(0, num_tokens, (40,), generator=g, device=device).tolist(),
    ]
    sq = len(rows)
    q = gen_q_glm(sq, H, seed=173)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.empty(sq, H, GLM_V_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
        scale_mode="per_tensor",
    )
    ref = ref_prefill_glm(q, cache, rows, scale, block_size)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[glm576_two_tile] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_glm576_edge_empty_invalid():
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 64
    H = 128
    scale = default_scale_glm()
    num_tokens = 200
    kv = gen_kv_glm(num_tokens, seed=91)
    cache = pack_glm_fp8_cache(kv, block_size)
    # q0 normal, q1 empty, q2 all-invalid, q3 mixed valid/invalid (>= num_tokens)
    rows_full = [[5, 7, 9, 64, 65], [], [-1, -1, -1], [10, -1, 11, 199, 5000]]
    sq = len(rows_full)
    q = gen_q_glm(sq, H, seed=92)
    indices, indptr = _ragged_from_rows(rows_full, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.full((sq, H, GLM_V_DIM), 7.0, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
        scale_mode="per_tensor",
    )
    assert not torch.isnan(out).any(), "NaN in output"
    assert out[1].abs().max().item() == 0.0, "empty query must be zero"
    assert out[2].abs().max().item() == 0.0, "all-invalid query must be zero"
    rows_valid = [[s for s in r if 0 <= s < num_tokens] for r in rows_full]
    ref = ref_prefill_glm(q, cache, rows_valid, scale, block_size)
    cos_mean, cos_min, max_abs = _metrics(out[[0, 3]], ref[[0, 3]])
    print(f"[glm576_edge] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def _main():
    if not _HAS_FLYDSL:
        print("[SKIP] flydsl not importable")
        return 0
    if not _is_gfx942():
        print("[SKIP] not a gfx942 device")
        return 0
    # Phase A
    test_basic()
    test_all_invalid()
    test_empty_kv_len()
    # Phase B2
    test_b2_two_region()
    test_b2_multitile()
    test_b2_ue8m0_non_unity()
    test_b2_region0_empty_rejected()
    # P0.2 bf16-RoPE split dot (non-default rope_bf16 flag)
    test_b1_rope_bf16()
    test_b2_rope_bf16()
    # GLM-5 / DSv3.2 (single-region, head_dim=576, per-tensor scale)
    test_glm576_paged_basic()
    test_glm576_per_tensor_scale()
    test_glm576_q_scale()
    test_glm576_two_tile()
    test_glm576_edge_empty_invalid()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
