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
    B: int, QH_PER_KH: int, KH: int, D: int, nope: bool, attn_output_gate, dtype
):
    qkv = torch.randn(
        (
            B,
            (QH_PER_KH * KH * (2 if attn_output_gate else 1) + 2 * KH)
            * (D * (2 if nope else 1)),
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

    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
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
):
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    # 1. Split
    if attn_output_gate:
        q, gate, k, v = qkv.split([q_size, q_size, kv_size, kv_size], dim=-1)
        gate = gate.view(-1, QH_PER_KH * KH, D).contiguous()
    else:
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
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
            k_cache[b, :, s, :] = k[i]
            v_cache[b, :, s, :] = v[i]

    if attn_output_gate:
        return q, gate, k, v, k_cache, v_cache
    else:
        return q, k, v, k_cache, v_cache


@pytest.mark.parametrize("B", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("QH_PER_KH", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("KH", [1, 4])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("max_embed_positions", [131072])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("attn_output_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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
    dtype,
):
    eps = 1e-5
    QH = QH_PER_KH * KH
    torch.manual_seed(1)

    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, False, attn_output_gate, dtype)
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
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
        attn_output_gate=attn_output_gate,
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
    )

    if attn_output_gate:
        q_ref, gate_ref, k_ref, v_ref, k_cache_ref, v_cache_ref = ref_result
    else:
        q_ref, k_ref, v_ref, k_cache_ref, v_cache_ref = ref_result

    # Verify Contiguous Outputs
    if attn_output_gate:
        torch.testing.assert_close(gate_tri, gate_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q_tri, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_tri, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_tri, v_ref, atol=1e-2, rtol=1e-2)

    # Verify Paged Cache
    torch.testing.assert_close(k_cache, k_cache_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_cache, v_cache_ref, atol=1e-2, rtol=1e-2)
