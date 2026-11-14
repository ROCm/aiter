import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_qk_norm_mrope import (
    HEAD_DIM,
    ROPE_HALF,
    fused_qk_norm_mrope_gfx1250,
    fused_qk_norm_mrope_k4v8_gfx1250,
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
# One config per op: the cache write keeps the row kernel faster up to a larger row count.
_CONFIG_NAME = "FUSED_QK_NORM_MROPE"
_CACHE_CONFIG_NAME = "FUSED_QK_NORM_MROPE_CACHE"
# A8K4V8: from this many tokens on, fused_qk_norm_mrope_k4v8_gfx1250 (K, Q and V in separate CTAs, whole
# 16-token K tiles written with TDM stores) runs the whole call; below it (and for decode) the row kernel.
_K4V8_PAGE_MIN_TOKENS = 4096
# fused_qk_norm_mrope_k4v8_gfx1250's launch: warps (one launch, so shared by the K / Q / V sections), Q heads a Q
# CTA (QH_PER), tokens a Q / V CTA (BLOCK_M_Q / _V; a K CTA is one 16-row NVFP4 tile). From the sweep of num_warps
# {4, 8, 16} x BLOCK_M_Q / BLOCK_M_V {16, 32, 64, 128} on Qwen3.5 EP4 (16 / 2 heads) and TP4 (4 / 1), pages 64 and
# 128: 16 warps (one row a wave), except TP4 prefills of 8192+ tokens: 8 warps.
_K4V8_CONFIG = None  # tests / sweeps: a fixed launch config instead of the selection


def _k4v8_config(num_tokens: int, q_heads: int) -> dict:
    if _K4V8_CONFIG is not None:
        cfg = dict(_K4V8_CONFIG)
    elif q_heads >= 16:
        # 8 warps / 32 Q tokens wins only on page-aligned batches and has a ~45 us floor on mid-page ones
        cfg = {"num_warps": 16, "QH_PER": 4, "BLOCK_M_Q": 16, "BLOCK_M_V": 32}
    else:
        cfg = (
            {"num_warps": 8, "QH_PER": 4, "BLOCK_M_Q": 16, "BLOCK_M_V": 16}
            if num_tokens >= 8192
            else {"num_warps": 16, "QH_PER": 4, "BLOCK_M_Q": 16, "BLOCK_M_V": 16}
        )
    if q_heads % cfg["QH_PER"]:
        cfg["QH_PER"] = 1
    return cfg


def _get_config(config_name: str, num_rows: int) -> dict:
    config_dir = resolve_config_dir("fusions", config_name, backend="gluon")
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
    kv_scale=None,
    kv_loc=None,
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
    config = _get_config(_CACHE_CONFIG_NAME if cache else _CONFIG_NAME, num_rows)

    fp8 = kv_scale is not None
    # A8K4V8: uint8 NVFP4 K, fp8 V; Q goes out in V's fp8
    k_nvfp4 = cache and k_cache.dtype == torch.uint8
    cache_dtype = v_cache.dtype if k_nvfp4 else k_cache.dtype
    q_out = torch.empty(
        (num_tokens, q.shape[1]),
        dtype=cache_dtype if fp8 else q.dtype,
        device=q.device,
    )
    k_out = torch.empty((num_tokens, k.shape[1]), dtype=k.dtype, device=k.device)
    # Only the row kernel writes an fp8 cache (and so an A8K4V8 one, which is always fp8).
    tdm = config["TDM"] and not fp8
    _LOGGER.info(
        "FUSED_QK_NORM_MROPE: T=%d Hq=%d Hk=%d cache=%s TDM=%s fp8=%s",
        num_tokens,
        q_heads,
        k_heads,
        cache,
        tdm,
        fp8,
    )

    if k_nvfp4 and 0 < _K4V8_PAGE_MIN_TOKENS <= num_tokens:
        page_block = k_cache.shape[2]
        cfg = _k4v8_config(num_tokens, q_heads)
        qh_per, bm_q, bm_v = cfg["QH_PER"], cfg["BLOCK_M_Q"], cfg["BLOCK_M_V"]
        grid = (
            triton.cdiv(num_tokens, 16)
            + (q_heads // qh_per) * triton.cdiv(num_tokens, bm_q)
            + k_heads * triton.cdiv(num_tokens, bm_v),
        )
        fused_qk_norm_mrope_k4v8_gfx1250[grid](
            q,
            k,
            v,
            q_weight,
            k_weight,
            positions,
            cos,
            sin,
            q_out,
            k_out,
            slot_mapping,
            k_cache if kv_loc is None else kv_loc[0],
            v_cache if kv_loc is None else kv_loc[1],
            num_tokens,
            slot_mapping.shape[0],
            cos.shape[0],
            q.stride(0),
            k.stride(0),
            v.stride(0),
            positions.stride(0),
            cos.stride(0),
            1.0 / kv_scale,
            None if kv_loc is None else kv_loc[2],
            EPS=eps,
            Q_HEADS=q_heads,
            K_HEADS=k_heads,
            SECTION_H=mrope_section[1],
            SECTION_W=mrope_section[2],
            BLOCK_SIZE=page_block,
            SCALE_PRESHUFFLE=min(128, page_block),
            FP8_MAX=torch.finfo(cache_dtype).max,
            QH_PER=qh_per,
            BLOCK_M_Q=bm_q,
            BLOCK_M_V=bm_v,
            KV_LOC=kv_loc is not None,
            num_warps=cfg["num_warps"],
        )
        return q_out, k_out
    num_warps = config["num_warps"]
    args = (q, k, q_weight, k_weight, positions, cos, sin, q_out, k_out)
    strides = (q.stride(0), k.stride(0), positions.stride(0), cos.stride(0))
    if kv_loc is None:
        cache_args = (
            v,
            slot_mapping,
            v.stride(0),
            slot_mapping.shape[0],
            k_cache,
            v_cache,
        )
    else:
        cache_args = (
            v,
            slot_mapping,
            v.stride(0),
            slot_mapping.shape[0],
            kv_loc[0],
            kv_loc[1],
        )
    fp8_kwargs = {}
    if tdm:
        # A program does block_m tokens, so that the K tile has one row a wave.
        assert q_heads & (q_heads - 1) == 0 and k_heads & (k_heads - 1) == 0
        assert q.stride(0) % 8 == 0 and k.stride(0) % 8 == 0 and v.stride(0) % 8 == 0
        kernel, num = fused_qk_norm_mrope_tdm_gfx1250, num_tokens
        block_m = max(1, num_warps // k_heads)
    else:
        # One row per wave. The rows of token t are its Q heads, then its K heads.
        kernel, num, block_m = fused_qk_norm_mrope_gfx1250, num_rows, num_warps
        cache_args += (
            1.0 / kv_scale if fp8 else 1.0,
            None if kv_loc is None else kv_loc[2],
        )
        fp8_kwargs = {
            "FP8": fp8,
            "FP8_MAX": torch.finfo(cache_dtype).max if fp8 else 0.0,
            "K_NVFP4": k_nvfp4,
            "SCALE_PRESHUFFLE": min(128, k_cache.shape[2]) if k_nvfp4 else 128,
            "KV_LOC": kv_loc is not None,
        }
    kernel[(triton.cdiv(num, block_m),)](
        *args,
        num,
        *strides,
        *cache_args,
        EPS=eps,
        Q_HEADS=q_heads,
        K_HEADS=k_heads,
        SECTION_H=mrope_section[1],
        SECTION_W=mrope_section[2],
        BLOCK_M=block_m,
        CACHE=cache,
        BLOCK_SIZE=(k_cache.shape[2] if k_nvfp4 else k_cache.shape[3]) if cache else 1,
        **fp8_kwargs,
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
    kv_scale: float | None = None,
    kv_loc: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
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
        kv_scale: for an fp8 cache (and the fp8 V of an A8K4V8 one), its one per-tensor scale. K and V are stored as
            value / kv_scale clamped to the fp8 range, as ``reshape_and_cache(kv_cache_dtype="fp8")``
            stores them after the caller's clamp, and Q is returned in the cache dtype,
            clamped, without a scale. None for a cache in the dtype of q.
        kv_loc: A8K4V8 only, block 128 and 2 KV heads: (k_base uint8, v_base in the dtype of v_cache, x_table
            uint8 [codes]) of one layer's IOD-local cache (``aiter.ops.kv_iod``). The block id slot // 128 is
            a code c. The bytes of each (block, head) page go to the placed addresses of c in k_base / v_base,
            not to k_cache / v_cache. k_cache and v_cache then only give the layout and are not written.

    Returns:
        q_out [T, Hq * 256] (fp8 when kv_scale is given) and k_out [T, Hk * 256] in the
        dtype of q, contiguous.

    The TDM kernel also does the cache write, above a larger row count than in
    ``fused_qk_norm_mrope``. The V rows then also need a stride that is a multiple of
    16 bytes.

    A8K4V8 (fp8 Q, NVFP4 K, fp8 V for aiter's unified attention): k_cache uint8
    [blocks, Hk, block, 256 // 2 + 256 // 16] (per page and head: the e2m1 pairs of each 16-token tile in
    (16, 16)-byte tiles, then the e4m3 per-16 scales preshuffled in min(128, block)-row groups), v_cache fp8
    as above (x = 16), kv_scale required; block 64, 128 or a multiple of 128. K is quantized without a scale
    (k_descale = 1), V as for an fp8 cache, Q returned in fp8. Written by the row kernel, or from
    _K4V8_PAGE_MIN_TOKENS tokens on by fused_qk_norm_mrope_k4v8_gfx1250 (the same bytes).
    """
    if k_cache.dtype == torch.uint8:
        # A8K4V8: NVFP4 K [blocks, Hk, block, 256 // 2 + 256 // 16] (e2m1 bytes in (16, 16) tiles, then
        # the e4m3 scales preshuffled in min(128, block)-row groups), fp8 V as below; Q comes back in fp8
        num_blocks, k_heads, block, k_row = k_cache.shape
        assert k_row == HEAD_DIM.value // 2 + HEAD_DIM.value // 16, k_cache.shape
        assert v_cache.shape == (
            num_blocks,
            k_heads,
            block // 16,
            HEAD_DIM.value,
            16,
        ), v_cache.shape
        assert v_cache.element_size() == 1 and kv_scale is not None and kv_scale > 0
        assert block >= 64 and (block <= 128 or block % 128 == 0), block
        assert k_cache.is_contiguous() and v_cache.is_contiguous()
    else:
        num_blocks, k_heads, _, block, x = k_cache.shape
        assert k_cache.shape == (num_blocks, k_heads, HEAD_DIM.value // x, block, x)
        assert v_cache.shape == (num_blocks, k_heads, block // x, HEAD_DIM.value, x)
        assert k_cache.is_contiguous() and v_cache.is_contiguous()
        assert k_cache.dtype == v_cache.dtype and x * k_cache.element_size() == 16
        if kv_scale is None:
            assert k_cache.dtype == q.dtype
        else:
            assert k_cache.element_size() == 1 and kv_scale > 0
    assert v.shape == k.shape and v.stride(1) == 1
    assert slot_mapping.dim() == 1
    if kv_loc is not None:
        k_base, v_base, x_table = kv_loc
        assert k_cache.dtype == torch.uint8, "kv_loc: A8K4V8 only"
        assert k_cache.shape[1:3] == (2, 128), k_cache.shape
        assert (
            k_base.dtype == torch.uint8
            and v_base.dtype == v_cache.dtype
            and x_table.dtype == torch.uint8
        )
        assert k_base.data_ptr() % (2 << 20) == 0 and v_base.data_ptr() % (2 << 20) == 0
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
        kv_scale,
        kv_loc,
    )
