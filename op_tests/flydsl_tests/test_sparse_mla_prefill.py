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
# Phase B (B1 paged single-region + B2 two-region) -- packed fp8_ds_mla cache.
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


def test_b1_paged_basic():
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    num_tokens = 600
    kv = _gen_kv(num_tokens, seed=11)
    cache = _pack_fp8_ds_mla_cache(kv, block_size)
    rows = [[5, 200, 7, 400, 63, 64, 599, 128], [0, 1, 2, 3], [300, 301, 302, 303, 304, 305, 306]]
    sq = len(rows)
    q = _gen_q(sq, H, seed=12)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
    )
    ref = _ref_prefill_packed(q, [(cache, rows, False)], scale, None, block_size)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[b1_paged_basic] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_b1_sink():
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    num_tokens = 300
    kv = _gen_kv(num_tokens, seed=21)
    cache = _pack_fp8_ds_mla_cache(kv, block_size)
    rows = [[5, 200, 7, 64, 63], [10, 11, 12, 13, 14, 15, 16, 17]]
    sq = len(rows)
    q = _gen_q(sq, H, seed=22)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)
    sink = (torch.randn(H, dtype=torch.float32, device=device) * 0.5 - 0.2)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True, attn_sink=sink,
    )
    ref = _ref_prefill_packed(q, [(cache, rows, False)], scale, sink, block_size)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[b1_sink] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_b1_edge_empty_invalid():
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 64
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    num_tokens = 200
    kv = _gen_kv(num_tokens, seed=31)
    cache = _pack_fp8_ds_mla_cache(kv, block_size)
    # q0 normal, q1 empty, q2 all-invalid (-1), q3 mixed valid/invalid
    rows_full = [[5, 7, 9, 64], [], [-1, -1, -1], [10, -1, 11, 199, 5000]]
    sq = len(rows_full)
    q = _gen_q(sq, H, seed=32)
    indices, indptr = _ragged_from_rows(rows_full, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.full((sq, H, PACKED_HEAD_DIM), 7.0, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
    )
    assert not torch.isnan(out).any(), "NaN in output"
    assert out[1].abs().max().item() == 0.0, "empty query must be zero"
    assert out[2].abs().max().item() == 0.0, "all-invalid query must be zero"
    # reference filters invalid slots (slot<0 or >=num_tokens).
    rows_valid = [[s for s in r if 0 <= s < num_tokens] for r in rows_full]
    ref = _ref_prefill_packed(q, [(cache, rows_valid, False)], scale, None, block_size)
    cos_mean, cos_min, max_abs = _metrics(out[[0, 3]], ref[[0, 3]])
    print(f"[b1_edge] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert cos_mean > 0.98, f"cosine too low: {cos_mean}"


def test_b1_ue8m0_nontrivial():
    """UE8M0 != 1: per-64-block power-of-2 scales folded by the convert path."""
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    device = "cuda"
    block_size = 16
    H = 128
    scale = PACKED_HEAD_DIM ** -0.5
    num_tokens = 64
    kv = _gen_kv(num_tokens, seed=41, scale=0.25)
    # 7 per-block exponents around 127 (so exp2(enc-127) in {1/4..4}); small to
    # keep scaled values inside the fnuz range.
    scale_bytes = [125, 126, 127, 128, 129, 127, 126]
    cache = _pack_fp8_ds_mla_cache(kv, block_size, scale_byte=scale_bytes)
    rows = [[1, 5, 9, 16, 17, 33], [0, 2, 4, 6, 8, 10, 12]]
    sq = len(rows)
    q = _gen_q(sq, H, seed=42, scale=0.25)
    indices, indptr = _ragged_from_rows(rows, device)
    block_table = _identity_block_table(num_tokens, block_size, device)

    out = torch.empty(sq, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flydsl_sparse_mla_prefill(
        q, cache, indices, indptr, out,
        block_table=block_table, block_size=block_size, packed=True,
        scale_mode="ue8m0",
    )
    ref = _ref_prefill_packed(q, [(cache, rows, False)], scale, None, block_size)
    cos_mean, cos_min, max_abs = _metrics(out, ref)
    print(f"[b1_ue8m0] cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} max_abs={max_abs:.4f}")
    assert not torch.isnan(out).any(), "NaN in output"
    assert cos_mean > 0.97, f"cosine too low: {cos_mean}"


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
    # Phase B1
    test_b1_paged_basic()
    test_b1_sink()
    test_b1_edge_empty_invalid()
    test_b1_ue8m0_nontrivial()
    # Phase B2
    test_b2_two_region()
    test_b2_multitile()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
