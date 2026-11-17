import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_qk_norm_mrope import (
    HEAD_DIM,
    ROPE_HALF,
    fused_qk_norm_mrope_gfx1250,
    fused_qk_norm_mrope_block16_kv_fp4_gfx1250,
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
# A8K4V8: from this many tokens on, fused_qk_norm_mrope_block16_kv_fp4_gfx1250 (K, Q and V in separate CTAs, whole
# 16-token K tiles written with TDM stores) runs the whole call; below it (and for decode) the row kernel.
_K4V8_PAGE_MIN_TOKENS = 4096
# fused_qk_norm_mrope_block16_kv_fp4_gfx1250's launch: warps (one launch, so shared by the K / Q / V sections), Q heads a Q
# CTA (QH_PER), tokens a Q / V CTA (BLOCK_M_Q / _V; a K CTA is one 16-row NVFP4 tile). From the sweep of num_warps
# {4, 8, 16} x BLOCK_M_Q / BLOCK_M_V {16, 32, 64, 128} on Qwen3.5 EP4 (16 / 2 heads) and TP4 (4 / 1), pages 64 and
# 128: 16 warps (one row a wave), except TP4 prefills of 8192+ tokens: 8 warps.
_K4V8_CONFIG = None  # tests / sweeps: a fixed launch config instead of the selection
# A8K4V4PT with the fp4pt V fused (fused_qk_norm_mrope_cache_k4v4pt with v / v_cache): from this many tokens
# on the K / Q / V kernel (V in 16-token windows, full tiles TDM-stored when _K4V4PT_V_TDM), below it the row
# kernel (one token a V row). Picked with the 2026-11-16 microbench (CUDA-graph GPU time, Qwen 16/2 and 4/1 heads,
# decode- and prefill-like slots; k4v4pt_writer/RESULTS.md): the row kernel wins up to 1024 tokens, the K / Q / V
# kernel from 2048 on. The full-tile TDM store measured no faster than the masked row stores (equal within ~2%), so
# it is off by default; it is tested bit-exact either way.
_K4V4PT_PAGE_MIN_TOKENS = 2048
_K4V4PT_V_TDM = False
# From this many Q heads the row kernel runs every A8K4V8 call: with 16 Q / 2 K heads (Qwen3.6-35B-A3B on
# MI455X, instruction prefetch on) it beats the k4v8 kernel at every size measured, 4096 tokens 48 -> 25 us,
# 8192 91 -> 47, 16384 177 -> 95 (the k4v8 kernel's 78 KB of code does not fit the instruction cache).
# Fewer heads keep _K4V8_PAGE_MIN_TOKENS.
_K4V8_ROWS_MIN_Q_HEADS = 16


def _k4v8_min_tokens(q_heads: int) -> int:
    """Tokens from which the k4v8 kernel runs the call (0: never)."""
    if _K4V8_ROWS_MIN_Q_HEADS and q_heads >= _K4V8_ROWS_MIN_Q_HEADS:
        return 0
    return _K4V8_PAGE_MIN_TOKENS


def _k4v4pt_min_tokens(q_heads: int) -> int:
    """Tokens from which the K / Q / V kernel runs an A8K4V4PT call (0: never). From _K4V8_ROWS_MIN_Q_HEADS Q heads
    the row kernel runs every call, as for A8K4V8: with 16 Q / 2 K heads it is 1.06-1.10x the A8K4V8 row kernel and
    beats the K / Q / V kernel at every size (2048 tokens 19.8 -> 15.8 us, 4096 33.0 -> 24.4, 16384 110.3 -> 84.5;
    CUDA-graph GPU time on MI455X). Fewer heads keep _K4V4PT_PAGE_MIN_TOKENS."""
    if _K4V8_ROWS_MIN_Q_HEADS and q_heads >= _K4V8_ROWS_MIN_Q_HEADS:
        return 0
    return _K4V4PT_PAGE_MIN_TOKENS


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
    v_store=True,
    v_fp4pt=False,
    num_tokens=None,
):
    assert arch_info.get_arch() == "gfx1250"
    rows = q.shape[0]
    # Only the first num_tokens rows are computed (the rest are a graph bucket's pad rows); the outputs
    # keep all the rows, the pad ones unwritten.
    num_tokens = rows if num_tokens is None else max(0, min(int(num_tokens), rows))
    q_heads = q.shape[1] // HEAD_DIM.value
    k_heads = k.shape[1] // HEAD_DIM.value
    assert q.shape[1] == q_heads * HEAD_DIM.value
    assert k.shape == (rows, k_heads * HEAD_DIM.value)
    assert q.stride(1) == 1 and k.stride(1) == 1
    assert positions.shape == (3, rows) and positions.stride(1) == 1
    assert cos.shape[1] == ROPE_HALF.value and cos.stride(1) == 1
    assert sin.stride() == cos.stride()
    num_rows = num_tokens * (q_heads + k_heads)
    config = _get_config(_CACHE_CONFIG_NAME if cache else _CONFIG_NAME, num_rows)

    fp8 = kv_scale is not None
    # A8K4V8: uint8 NVFP4 K, fp8 V; Q goes out in V's fp8
    k_nvfp4 = cache and k_cache.dtype == torch.uint8
    cache_dtype = v_cache.dtype if k_nvfp4 else k_cache.dtype
    if v_fp4pt:
        # A8K4V4PT: Q goes out in fp8 (the V cache is the uint8 fp4pt page)
        from aiter.ops.triton.utils.types import e4m3_dtype

        assert k_nvfp4 and not v_store and fp8
        cache_dtype = e4m3_dtype
    q_out = torch.empty(
        (rows, q.shape[1]),
        dtype=cache_dtype if fp8 else q.dtype,
        device=q.device,
    )
    k_out = torch.empty((rows, k.shape[1]), dtype=k.dtype, device=k.device)
    if num_tokens == 0:
        return q_out, k_out
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

    min_tokens = _k4v4pt_min_tokens(q_heads) if v_fp4pt else _k4v8_min_tokens(q_heads)
    if k_nvfp4 and 0 < min_tokens <= num_tokens:
        page_block = k_cache.shape[2]
        cfg = _k4v8_config(num_tokens, q_heads)
        qh_per, bm_q, bm_v = cfg["QH_PER"], cfg["BLOCK_M_Q"], cfg["BLOCK_M_V"]
        grid = (
            triton.cdiv(num_tokens, 16)
            + (q_heads // qh_per) * triton.cdiv(num_tokens, bm_q)
            # fp8 V CTAs (none for A8K4V4PT, v_store False: its fp4pt V CTAs are below)
            + (k_heads * triton.cdiv(num_tokens, bm_v) if v_store else 0)
            # A8K4V4PT fused V: one CTA per V head and 16-token window
            + (k_heads * triton.cdiv(num_tokens, 16) if v_fp4pt else 0),
        )
        fused_qk_norm_mrope_block16_kv_fp4_gfx1250[grid](
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
            k_cache,
            v_cache,
            num_tokens,
            slot_mapping.shape[0],
            cos.shape[0],
            q.stride(0),
            k.stride(0),
            v.stride(0),
            positions.stride(0),
            cos.stride(0),
            1.0 / kv_scale,
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
            **({"V_FP4PT": True, "V_TDM": _K4V4PT_V_TDM} if v_fp4pt else {}),
            num_warps=cfg["num_warps"],
        )
        return q_out, k_out
    # ROW_NUM_WARPS: the row kernel's waves a program where the config's num_warps sizes the TDM kernel
    num_warps = (
        config["num_warps"] if tdm else config.get("ROW_NUM_WARPS", config["num_warps"])
    )
    if cache and slot_mapping.shape[0] == 0:
        # the row kernel reads slot_mapping[min(token, num_slots - 1)]: one pad slot, no token past it
        slot_mapping = torch.full(
            (1,), -1, dtype=slot_mapping.dtype, device=slot_mapping.device
        )
    args = (q, k, q_weight, k_weight, positions, cos, sin, q_out, k_out)
    strides = (q.stride(0), k.stride(0), positions.stride(0), cos.stride(0))
    cache_args = (v, slot_mapping, v.stride(0), slot_mapping.shape[0], k_cache, v_cache)
    fp8_kwargs = {}
    if tdm:
        # A program does block_m tokens, so that the K tile has one row a wave.
        assert q_heads & (q_heads - 1) == 0 and k_heads & (k_heads - 1) == 0
        assert q.stride(0) % 8 == 0 and k.stride(0) % 8 == 0 and v.stride(0) % 8 == 0
        kernel, num = fused_qk_norm_mrope_tdm_gfx1250, num_tokens
        block_m = max(1, num_warps // k_heads)
    else:
        # ROWS_PER_WAVE rows a wave (the layouts repeat over the block): the rows' loads are in flight
        # together. The rows of token t are its Q heads, then its K heads.
        kernel, num = fused_qk_norm_mrope_gfx1250, num_rows
        block_m = num_warps * config.get("ROWS_PER_WAVE", 1)
        cache_args += (1.0 / kv_scale if fp8 else 1.0,)
        fp8_kwargs = {
            "FP8": fp8,
            "FP8_MAX": torch.finfo(cache_dtype).max if fp8 else 0.0,
            "K_NVFP4": k_nvfp4,
            "SCALE_PRESHUFFLE": min(128, k_cache.shape[2]) if k_nvfp4 else 128,
        }
        if not v_store:
            # only A8K4V4PT, whose fp4pt V the V_FP4PT path writes
            assert k_nvfp4 and v_fp4pt
            fp8_kwargs["V_STORE"] = False
        if v_fp4pt:
            fp8_kwargs["V_FP4PT"] = True
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
    num_tokens: int | None = None,
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

        num_tokens: when given, only the first num_tokens rows are computed and written (q_out / k_out rows
            past it are left unwritten): a graph bucket's pad rows that no sequence owns.

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
    _K4V8_PAGE_MIN_TOKENS tokens on by fused_qk_norm_mrope_block16_kv_fp4_gfx1250 (the same bytes).
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
        num_tokens=num_tokens,
    )


def fused_qk_norm_mrope_cache_k4v4pt(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_scale: float,
    v: torch.Tensor,
    v_cache: torch.Tensor,
):
    """A8K4V4PT cache write: Q out in fp8, the NVFP4 K into k_cache (byte-identical to the A8K4V8 call) and the fp4pt
    V into v_cache (bit-exact with fp4pt_quant_v_cache). k_cache uint8 [blocks, Hk, block, 256 // 2 + 256 // 16];
    v [T, Hk * 256]; v_cache the fp4pt V cache, [blocks, Hk, 128, 256 // 2 + 4] float4_e2m1fn_x2 or uint8."""
    assert k_cache.dtype == torch.uint8 and k_cache.dim() == 4 and k_cache.is_contiguous()
    num_blocks, k_heads, block, k_row = k_cache.shape
    assert k_row == HEAD_DIM.value // 2 + HEAD_DIM.value // 16, k_cache.shape
    assert block >= 64 and (block <= 128 or block % 128 == 0), block
    assert kv_scale is not None and kv_scale > 0 and slot_mapping.dim() == 1
    assert k.stride(1) == 1
    vc = v_cache.view(torch.uint8)
    assert vc.shape == (num_blocks, k_heads, block, HEAD_DIM.value // 2 + 4), vc.shape
    assert vc.is_contiguous() and block == 128
    assert v.shape == k.shape and v.stride(1) == 1 and v.stride(0) % 8 == 0
    return _launch(
        q, k, v, q_weight, k_weight, eps, positions, cos, sin, mrope_section,
        k_cache, vc, slot_mapping, True, kv_scale, v_store=False, v_fp4pt=True,
    )
