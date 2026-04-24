import torch
import pytest
from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope
from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (
    fused_qkv_split_qk_norm_rope_cache,
)
from op_tests.triton_tests.fusions.test_fused_qk_concat import (
    generate_rope_cached_freqs,
)
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle


def generate_qkv_inputs(
    B: int,
    QH_PER_KH: int,
    KH: int,
    D: int,
    nope: bool,
    attn_output_gate,
    dtype,
    qkv_layout: str = "interleaved",
):
    """Build a flat ``qkv`` tensor for fused split / norm+RoPE+cache tests.

    Args:
        B: Sequence length (number of tokens); first dimension of ``qkv``.
        QH_PER_KH: Query heads per key head; total query heads ``QH = QH_PER_KH * KH``.
        KH: Number of key (and value) heads.
        D: Head dimension (no RoPE/Nope split here; ``nope`` scales logical width).
        nope: If True, build doubled logical width per head for Nope-style layouts
            (unused by the cache test path, which passes ``False``).
        attn_output_gate: If True, append a per-query-head gate next to Q in ``qkv``.
        dtype: Tensor dtype (typically ``torch.bfloat16``).
        qkv_layout: When ``attn_output_gate`` is True, how Q and gate are packed:
            ``"interleaved"`` — ``[q_h0||g_h0||q_h1||g_h1||...]`` then K then V;
            ``"blocked"`` — ``[all Q flat][all gate flat]`` then K then V.
            Non-gated QKV is always logically interleaved; only gated uses
            ``blocked``.
    """
    QH = QH_PER_KH * KH
    kv_size = KH * D
    d_flat = D * (2 if nope else 1)
    if attn_output_gate and not nope:
        q = torch.randn(B, QH, D, dtype=dtype, device="cuda")
        gate = torch.randn(B, QH, D, dtype=dtype, device="cuda")
        if qkv_layout == "interleaved":
            qg = torch.stack([q, gate], dim=2).reshape(B, QH * 2 * D)
        elif qkv_layout == "blocked":
            qg = torch.cat([q.reshape(B, QH * D), gate.reshape(B, QH * D)], dim=-1)
        else:
            raise ValueError(qkv_layout)
        k = torch.randn(B, KH, D, dtype=dtype, device="cuda").reshape(B, kv_size)
        v = torch.randn(B, KH, D, dtype=dtype, device="cuda").reshape(B, kv_size)
        return torch.cat([qg, k, v], dim=-1)
    qkv = torch.randn(
        (
            B,
            (QH * (2 if attn_output_gate else 1) + 2 * KH) * d_flat,
        ),
        dtype=dtype,
        device="cuda",
    )
    return qkv


def run_torch(
    qkv,
    QH_PER_KH,
    KH,
    D,
    ref_freqs,
    reuse_freqs_front_part,
    nope,
    nope_first,
    rotate_style,
):
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    q = ref_rope_sbhd_fwd(
        q,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    k = ref_rope_sbhd_fwd(
        k,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )

    return q, k, v


# @pytest.mark.parametrize("B", [32])
# @pytest.mark.parametrize("QH_PER_KH", [8])
# @pytest.mark.parametrize("KH", [8])
# @pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("B", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("QH_PER_KH", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("KH", [1, 4])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("max_embed_positions", [131072])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qkv_split_qk_rope(
    B: int,
    QH_PER_KH: int,
    KH: int,
    D: int,
    rotate_style: int,
    max_embed_positions: int,
    nope: bool,
    nope_first: bool,
    reuse_freqs_front_part: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(1)
    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, False, dtype)

    # Full-head RoPE (rotary_dim == head dim). Partial rotary_dim < D is covered in
    # test_fused_qkv_split_qk_rope_with_cache (fused_qkv_split_qk_norm_rope_cache kernel).
    head_dim = D * (2 if nope else 1)
    freqs_last_dim = (head_dim // 2) if reuse_freqs_front_part else head_dim
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, freqs_last_dim, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)

    q_triton, k_triton, v_triton = fused_qkv_split_qk_rope(
        qkv,
        cos,
        sin,
        pos,
        QH_PER_KH * KH,
        KH,
        (D * (2 if nope else 1)),
        is_neox=(rotate_style == RotateStyle.NEOX),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    q_torch, k_torch, v_torch = run_torch(
        qkv,
        QH_PER_KH,
        KH,
        (D * (2 if nope else 1)),
        ref_freqs,
        reuse_freqs_front_part,
        nope,
        nope_first,
        rotate_style,
    )

    torch.testing.assert_close(q_torch, q_triton)
    torch.testing.assert_close(k_torch, k_triton)
    torch.testing.assert_close(v_torch, v_triton)


# 2. RMS Norm
def rms_norm(x, w, eps):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x


def run_torch_with_cache(
    qkv,
    q_weight,
    k_weight,
    QH_PER_KH,
    KH,
    D,
    attn_output_gate,
    ref_freqs,
    rotate_style,
    reuse_freqs_front_part,
    eps,
    slot_mapping,
    num_blocks,
    block_size,
    k_scale,
    v_scale,
    qkv_layout: str = "interleaved",
):
    """Reference: split QKV, RMSNorm Q/K, RoPE, optional KV de-scale, paged KV write.

    Mirrors ``fused_qkv_split_qk_norm_rope_cache`` for assertions in
    ``test_fused_qkv_split_qk_rope_with_cache``.

    Args:
        qkv: Flat packed ``[B, q_part + k_part + v_part]`` (see ``generate_qkv_inputs``).
        q_weight: Per-dim RMSNorm gamma for Q, shape ``(D,)``.
        k_weight: Per-dim RMSNorm gamma for K, shape ``(D,)``.
        QH_PER_KH: Query heads per key head.
        KH: Key/value head count.
        D: Head dimension (RoPE applied over ``rotary_dim`` implied by ``ref_freqs``).
        attn_output_gate: Whether ``qkv`` includes a gate tensor after Q.
        ref_freqs: RoPE frequencies for ``ref_rope_sbhd_fwd`` (already indexed by position).
        rotate_style: ``RotateStyle.GPTJ`` or ``RotateStyle.NEOX``.
        reuse_freqs_front_part: Matches kernel / table layout for partial RoPE.
        eps: RMSNorm epsilon.
        slot_mapping: Int tensor ``[B]``, token ``i`` maps cache cell ``slot_mapping[i]``.
        num_blocks: Number of paged blocks in the reference cache tensors.
        block_size: Slots per block in paged layout (must match cache strides / kernel).
        k_scale: Optional scalar scale for K before cache write (``None`` => 1).
        v_scale: Optional scalar scale for V before cache write (``None`` => 1).
        qkv_layout: ``"interleaved"`` or ``"blocked"`` gated layout; ignored when not gated.
    """
    QH = QH_PER_KH * KH
    q_size = QH * D
    kv_size = KH * D
    # 1. Split (gated: interleaved or blocked [all Q][all gate], matching kernel)
    if attn_output_gate:
        qg_len = 2 * q_size
        if qkv_layout == "interleaved":
            qg = qkv[:, :qg_len].reshape(-1, QH, 2, D)
            q = qg[:, :, 0, :].contiguous()
            gate = qg[:, :, 1, :].contiguous()
        elif qkv_layout == "blocked":
            q = qkv[:, :q_size].reshape(-1, QH, D).contiguous()
            gate = qkv[:, q_size:qg_len].reshape(-1, QH, D).contiguous()
        else:
            raise ValueError(qkv_layout)
        k, v = qkv[:, qg_len:].split([kv_size, kv_size], dim=-1)
    else:
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.reshape(-1, QH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    q = rms_norm(q, q_weight, eps)
    k = rms_norm(k, k_weight, eps)

    # 3. RoPE
    q = ref_rope_sbhd_fwd(
        q,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    )
    k = ref_rope_sbhd_fwd(
        k,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    )

    if k_scale is None:
        k_scale = 1
    if v_scale is None:
        v_scale = 1

    k_scale_rcprl = 1 / k_scale
    k_descaled = k * k_scale_rcprl

    v_scale_rcprl = 1 / v_scale
    v_descaled = v * v_scale_rcprl

    # 4. Reference Caching (Paged)
    k_cache = torch.zeros(
        (num_blocks, KH, block_size, D), dtype=qkv.dtype, device="cuda"
    )
    v_cache = torch.zeros(
        (num_blocks, KH, block_size, D), dtype=qkv.dtype, device="cuda"
    )

    for i in range(qkv.shape[0]):
        slot = slot_mapping[i].item()
        if slot >= 0:
            b = slot // block_size
            s = slot % block_size
            k_cache[b, :, s, :] = k_descaled[i]
            v_cache[b, :, s, :] = v_descaled[i]

    if attn_output_gate:
        return q, gate, k, v, k_cache, v_cache
    else:
        return q, k, v, k_cache, v_cache


# Parametrize grid for ``test_fused_qkv_split_qk_rope_with_cache``; see that test's
# docstring for the meaning of each argument.
@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("QH_PER_KH", [1, 2, 4])
@pytest.mark.parametrize("KH", [1, 4])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("max_embed_positions", [131072])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("attn_output_gate", [False, True])
@pytest.mark.parametrize("use_kv_scale", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("qkv_layout", ["interleaved", "blocked"])
@pytest.mark.parametrize(
    "rotary_dim",
    [None, 32],
    ids=["full", "32"],
)
def test_fused_qkv_split_qk_rope_with_cache(
    B,
    QH_PER_KH,
    KH,
    D,
    block_size,
    rotate_style,
    max_embed_positions,
    reuse_freqs_front_part,
    attn_output_gate,
    use_kv_scale,
    dtype,
    qkv_layout,
    rotary_dim,
):
    """E2E ``fused_qkv_split_qk_norm_rope_cache`` vs torch reference (norm, RoPE, paged KV).

    Parametrize values (lines above) map to:

    Args:
        B: Batch / sequence length for synthetic ``qkv`` and position ids.
        QH_PER_KH: Query heads per key head (``QH = QH_PER_KH * KH``).
        KH: Key and value head count.
        D: Per-head feature dimension (head_dim for RMSNorm and cache).
        block_size: Paged KV block length (fixed ``16`` in the grid).
        rotate_style: RoPE layout, GPT-J vs NeoX (``RotateStyle`` enum).
        max_embed_positions: RoPE table length passed to ``generate_rope_cached_freqs``.
        reuse_freqs_front_part: Whether cos/sin pack the low half first; partial
            ``rotary_dim < D`` cases require ``True`` (skipped otherwise).
        attn_output_gate: Include gate in ``qkv`` and in op outputs when True.
        use_kv_scale: Pass random scalar ``k_scale`` / ``v_scale`` into kernel vs ``None``.
        dtype: Tensor dtype (grid uses ``torch.bfloat16`` only).
        qkv_layout: Gated packing: ``"interleaved"`` or ``"blocked"``; non-gated
            ``blocked`` cases are skipped.
        rotary_dim: RoPE span in features along the head dim; ``None`` means full ``D``,
            ``32`` exercises partial RoPE when ``D`` is 64 or 128.
    """
    if not attn_output_gate and qkv_layout == "blocked":
        pytest.skip("blocked QKV layout applies only when attn_output_gate=True")

    rd = D if rotary_dim is None else rotary_dim
    if rd > D:
        pytest.skip(f"rotary_dim={rd} exceeds head_dim D={D}")
    if rd <= 0 or rd % 2 != 0:
        pytest.skip(f"invalid rotary_dim={rd} (must be positive and even)")
    rotary_dim = rd

    partial_rope = rotary_dim < D
    if partial_rope and not reuse_freqs_front_part:
        pytest.skip(
            "partial rotary_dim tested with reuse_freqs_front_part=True "
            "(kernel/table layout matches ref_rope in that mode)"
        )

    eps = 1e-6
    QH = QH_PER_KH * KH
    torch.manual_seed(1)

    freqs_last_dim = (rotary_dim // 2) if reuse_freqs_front_part else rotary_dim

    qkv = generate_qkv_inputs(
        B, QH_PER_KH, KH, D, False, attn_output_gate, dtype, qkv_layout=qkv_layout
    )

    if use_kv_scale:
        k_scale = torch.randn((), dtype=torch.float32, device="cuda")
        v_scale = torch.randn((), dtype=torch.float32, device="cuda")
    else:
        k_scale = v_scale = None

    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, freqs_last_dim, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)
    q_weight = torch.randn((D,), dtype=dtype, device="cuda")
    k_weight = torch.randn((D,), dtype=dtype, device="cuda")

    # Setup Paged Cache
    num_blocks = (B + block_size - 1) // block_size + 2  # Extra blocks for safety
    k_cache = torch.zeros((num_blocks, KH, block_size, D), dtype=dtype, device="cuda")
    v_cache = torch.zeros((num_blocks, KH, block_size, D), dtype=dtype, device="cuda")

    # Random slot mapping (shuffled unique slots)
    slot_mapping = torch.randperm(num_blocks * block_size)[:B].to(torch.int32).cuda()

    # Triton Call
    tri_result = fused_qkv_split_qk_norm_rope_cache(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        pos,
        k_cache,
        v_cache,
        slot_mapping,
        QH,
        KH,
        D,
        is_neox=(rotate_style == RotateStyle.NEOX),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        eps=eps,
        k_scale=k_scale,
        v_scale=v_scale,
        attn_output_gate=attn_output_gate,
        gated_qkv_layout=qkv_layout,
    )
    if attn_output_gate:
        q_tri, gate_tri, k_tri, v_tri = tri_result
    else:
        q_tri, k_tri, v_tri = tri_result

    # Torch Reference
    ref_result = run_torch_with_cache(
        qkv,
        q_weight,
        k_weight,
        QH_PER_KH,
        KH,
        D,
        attn_output_gate,
        ref_freqs,
        rotate_style,
        reuse_freqs_front_part,
        eps,
        slot_mapping,
        num_blocks,
        block_size,
        k_scale,
        v_scale,
        qkv_layout=qkv_layout,
    )

    if attn_output_gate:
        q_ref, gate_ref, k_ref, v_ref, k_cache_ref, v_cache_ref = ref_result
    else:
        q_ref, k_ref, v_ref, k_cache_ref, v_cache_ref = ref_result

    # BF16 vs float ref: occasional ~0.011 drift (incl. partial rotary_dim < D)
    atol, rtol = 2e-2, 2e-2

    # Verify Contiguous Outputs
    if attn_output_gate:
        torch.testing.assert_close(gate_tri, gate_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q_tri, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_tri, k_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_tri, v_ref, atol=atol, rtol=rtol)

    # Verify Paged Cache
    torch.testing.assert_close(k_cache, k_cache_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_cache, v_cache_ref, atol=atol, rtol=rtol)
