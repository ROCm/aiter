import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_qk_norm_mrope import (
    HEAD_DIM,
    ROPE_HALF,
    fused_qk_norm_mrope_gfx1250,
    fused_qk_norm_mrope_tdm_gfx1250,
    get_fused_qk_norm_mrope_layouts,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "FUSED_QK_NORM_MROPE"


def _get_config(num_rows: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, num_rows, prefix="ROWS_LEQ_")


def _launch(
    q,
    k,
    v,
    q_weight,
    k_weight,
    eps,
    positions,
    cos,
    sin,
    mrope_section,
    k_cache,
    v_cache,
    slot_mapping,
    cache,
):
    assert arch_info.get_arch() == "gfx1250"
    num_tokens = q.shape[0]
    q_heads = q.shape[1] // HEAD_DIM.value
    k_heads = k.shape[1] // HEAD_DIM.value
    assert q.shape[1] == q_heads * HEAD_DIM.value
    assert k.shape == (num_tokens, k_heads * HEAD_DIM.value)
    assert q.stride(1) == 1 and k.stride(1) == 1
    assert positions.shape == (3, num_tokens) and positions.stride(1) == 1
    assert cos.shape[1] == ROPE_HALF.value and cos.stride(1) == 1
    assert sin.stride() == cos.stride()
    num_rows = num_tokens * (q_heads + k_heads)
    config = _get_config(num_rows)

    q_out = torch.empty((num_tokens, q.shape[1]), dtype=q.dtype, device=q.device)
    k_out = torch.empty((num_tokens, k.shape[1]), dtype=k.dtype, device=k.device)
    # The cache write makes the row kernel faster up to a larger row count.
    tdm = config["TDM_CACHE"] if cache else config["TDM"]
    _LOGGER.info(
        "FUSED_QK_NORM_MROPE: T=%d Hq=%d Hk=%d cache=%s TDM=%s",
        num_tokens,
        q_heads,
        k_heads,
        cache,
        tdm,
    )

    num_warps = config["num_warps"]
    args = (q, k, q_weight, k_weight, positions, cos, sin, q_out, k_out)
    strides = (q.stride(0), k.stride(0), positions.stride(0), cos.stride(0))
    cache_args = (v, slot_mapping, v.stride(0), slot_mapping.shape[0], k_cache, v_cache)
    if tdm:
        # A program does block_m tokens, so that the K tile has one row a wave.
        assert q_heads & (q_heads - 1) == 0 and k_heads & (k_heads - 1) == 0
        assert q.stride(0) % 8 == 0 and k.stride(0) % 8 == 0 and v.stride(0) % 8 == 0
        block_m = max(1, num_warps // k_heads)
        fused_qk_norm_mrope_tdm_gfx1250[(triton.cdiv(num_tokens, block_m),)](
            *args,
            num_tokens,
            *strides,
            *cache_args,
            EPS=eps,
            Q_HEADS=q_heads,
            K_HEADS=k_heads,
            SECTION_H=mrope_section[1],
            SECTION_W=mrope_section[2],
            BLOCK_M=block_m,
            CACHE=cache,
            BLOCK_SIZE=k_cache.shape[3] if cache else 1,
            X=k_cache.shape[4] if cache else 1,
            num_warps=num_warps,
            **get_fused_qk_norm_mrope_layouts(num_warps),
        )
    else:
        # One row per wave. The rows of token t are its Q heads, then its K heads.
        fused_qk_norm_mrope_gfx1250[(triton.cdiv(num_rows, num_warps),)](
            *args,
            num_rows,
            *strides,
            *cache_args,
            EPS=eps,
            Q_HEADS=q_heads,
            K_HEADS=k_heads,
            SECTION_H=mrope_section[1],
            SECTION_W=mrope_section[2],
            BLOCK_M=num_warps,
            CACHE=cache,
            BLOCK_SIZE=k_cache.shape[3] if cache else 1,
            X=k_cache.shape[4] if cache else 1,
            num_warps=num_warps,
            **get_fused_qk_norm_mrope_layouts(num_warps),
        )
    return q_out, k_out


def fused_qk_norm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
):
    """Q/K Gemma RMSNorm and interleaved MRoPE.

    Each head of 256 values gets x / rms(x) * (1 + weight). Values [0, 64) then get
    Neox RoPE: values i and i + 32 rotate with cos/sin[pos, i]. pos is the H position
    when i % 3 == 1 and i < 3 * mrope_section[1], the W position when i % 3 == 2 and
    i < 3 * mrope_section[2], else the T position.

    Args:
        q: [T, Hq * 256] bf16. k: [T, Hk * 256] bf16. Rows can be strided.
        q_weight, k_weight: [256] Gemma RMSNorm weights.
        eps: RMSNorm epsilon.
        positions: [3, T] T, H and W positions. Each must be in [0, max_pos).
        cos, sin: [max_pos, 32] fp32.
        mrope_section: T, H and W section sizes.

    Returns:
        q_out [T, Hq * 256] and k_out [T, Hk * 256], contiguous.

    Above the row count of the smallest config bucket, a TDM kernel moves the Q and K rows
    of several tokens as tiles. It needs head counts that are powers of 2 and row strides
    that are multiples of 16 bytes.
    """
    # Without the cache write the kernel does not read v, the caches or the slots.
    return _launch(
        q,
        k,
        k,
        q_weight,
        k_weight,
        eps,
        positions,
        cos,
        sin,
        mrope_section,
        k,
        k,
        positions,
        False,
    )


def fused_qk_norm_mrope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """``fused_qk_norm_mrope``, and the K and V rows into the paged KV cache.

    The cache has the layout of ``reshape_and_cache(asm_layout=True)``: value d of
    K head h of the token in slot s goes to k_cache[s // block, h, d // x, s % block,
    d % x], and of V to v_cache[s // block, h, (s % block) // x, d, s % x].

    Args:
        q, k, q_weight, k_weight, eps, positions, cos, sin, mrope_section: as in
            ``fused_qk_norm_mrope``.
        v: [T, Hk * 256] bf16. Rows can be strided.
        k_cache: [blocks, Hk, 256 // x, block, x], contiguous, dtype of q, x values in 16 bytes.
        v_cache: [blocks, Hk, block // x, 256, x], contiguous, dtype of q.
        slot_mapping: [S] int32 or int64. Token t < S goes to slot slot_mapping[t].
            Tokens with t >= S or a negative slot are padding: they are not written.

    Returns:
        q_out [T, Hq * 256] and k_out [T, Hk * 256], contiguous.

    The TDM kernel also does the cache write, above a larger row count than in
    ``fused_qk_norm_mrope``. The V rows then also need a stride that is a multiple of
    16 bytes.
    """
    num_blocks, k_heads, _, block, x = k_cache.shape
    assert k_cache.shape == (num_blocks, k_heads, HEAD_DIM.value // x, block, x)
    assert v_cache.shape == (num_blocks, k_heads, block // x, HEAD_DIM.value, x)
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert (
        k_cache.dtype == v_cache.dtype == q.dtype and x * k_cache.element_size() == 16
    )
    assert v.shape == k.shape and v.stride(1) == 1
    assert slot_mapping.dim() == 1
    return _launch(
        q,
        k,
        v,
        q_weight,
        k_weight,
        eps,
        positions,
        cos,
        sin,
        mrope_section,
        k_cache,
        v_cache,
        slot_mapping,
        True,
    )
