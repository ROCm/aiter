import inspect

import torch
import triton
from packaging.version import Version

from aiter.ops.triton._triton_kernels.attention.fp8_mqa_logits import (
    _fp8_mqa_logits_kernel,
    _fp8_paged_mqa_logits_rows_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)

TRITON_VERSION = Version(triton.__version__)
TRITON_GE_36 = TRITON_VERSION >= Version("3.6.0")
TRITON_GE_38 = TRITON_VERSION >= Version("3.8.0")

arch = arch_info.get_arch()
_gluon_fp8_mqa_logits_kernel = None
if TRITON_GE_36:
    try:
        if arch == "gfx950":
            from aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits import (
                _gluon_fp8_mqa_logits_kernel,
            )
        elif arch in ("gfx1250", "gfx1250-strict"):
            from aiter.ops.triton._gluon_kernels.gfx1250.attention.fp8_mqa_logits import (
                _gluon_fp8_mqa_logits_kernel,
                _gluon_fp8_mqa_logits_rowsplit_kernel,
                _gluon_fp8_paged_mqa_logits_kernel,
            )
    except Exception:  # noqa: BLE001
        _gluon_fp8_mqa_logits_kernel = None


# Hacks to see if we can use some newer features
# TODO: remove when the next Triton release happens so we can rely on version
# Latest official release do not have these features
def _async_copy_accepts_distributed_layout() -> bool:
    try:
        from triton.experimental.gluon.language.amd.cdna4 import async_copy

        src = inspect.getsource(async_copy.global_load_to_shared)
    except (OSError, TypeError, ImportError, AttributeError):
        return False
    return "DistributedLayout" in src


def _permute_accepts_constexpr_tuple() -> bool:
    """
    True iff Triton's _unwrap_iterable unwraps an inner constexpr.

    On versions before PR #9751 (commit 0688e7736a), passing a constexpr-wrapped
    tuple as the sole arg to permute/trans/reshape leaves the constexpr wrapped,
    causing `len(constexpr)` to fail in semantic.permute. After #9751, it gets
    unwrapped to a raw tuple of ints.
    """
    try:
        from triton.language.core import _unwrap_iterable, constexpr
    except ImportError:
        return False
    probe = constexpr((0, 1, 2))
    result = _unwrap_iterable((probe,))
    return not isinstance(result, constexpr)


ASYNC_COPY_SUPPORTS_DISTRIBUTED = _async_copy_accepts_distributed_layout()
FOLDED_REDUCTED_SUPPORT = _permute_accepts_constexpr_tuple()

# gfx942 (MI300X) LDS size per CU.
_GFX942_CU_LDS_BYTES = 64 * 1024

# paged max_ctx hint: per-row kernel up to this B * next_n * max_ctx, by next_n
_PAGED_ROWS_MAX_WORK = {1: 131072, 2: 65536, 3: 65536, 4: 65536}
_PAGED_ROWS_MAX_WORK_DEFAULT = 196608


def _gfx950_kv_splits(seq_len, seq_len_kv, block_m, num_warps, waves_per_eu):
    """How many workgroups to put on one query row block's KV walk."""
    MIN_SPLIT_KV = 16384
    GFX950_SIMDS = 256 * 4
    SPLIT_ROUNDS = 8
    target_wgs = SPLIT_ROUNDS * waves_per_eu * GFX950_SIMDS // num_warps
    num_blocks = triton.cdiv(seq_len, block_m)
    if num_blocks >= target_wgs:
        return 1
    return min(
        target_wgs // num_blocks,
        max(1, triton.cdiv(seq_len_kv, MIN_SPLIT_KV)),
    )


def _gfx942_tile_fits_lds(
    block_kv: int, head_size: int, num_stages: int, occupancy: int
) -> bool:
    # Only the double-buffered KV tile lives in LDS (Q and the fp32 scores
    # accumulator stay in registers in Triton 3.6+). Account for `occupancy`
    # co-resident workgroups and keep a 0.9 safety factor for compiler
    # overhead.
    # If a future Triton spills Q or scores to LDS, re-add a `q + kv + scores <= 64 KB` upper-bound term here to avoid re-triggering the JIT abort.
    lds_bytes = occupancy * num_stages * block_kv * head_size
    return lds_bytes <= 0.9 * _GFX942_CU_LDS_BYTES


def _gfx1250_launch_config(seq_len_kv: int, num_heads: int, head_size: int) -> dict:
    cfg_dir = resolve_config_dir("attention", "FP8_MQA_LOGITS", backend="gluon")
    table = load_config_json(f"{cfg_dir}/DEFAULT.json")
    shape = table.get(f"H{num_heads}_D{head_size}", table["any"])
    return select_leq_config(shape, seq_len_kv, prefix="KV_LEQ_")


def fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
    out: torch.Tensor | None = None,
    out_offsets: torch.Tensor | None = None,
    key_vec: int | None = None,
    k_early: bool | None = None,
    max_window: int | None = None,
    big_tiles: bool | None = None,
):
    """
    This function computes the logits to be used by a topk function for sparse attention.

    Q:           [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:          [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:   [seq_len_kv], dtype float32
    weights:     [seq_len, NUM_HEADS], dtype float32
    cu_starts:   [seq_len], dtype int32, start indices
    cu_ends:     [seq_len], dtype int32, end indices
    clean_logits: bool. If True, positions outside [cu_starts[i], cu_ends[i]) in row i
                  are explicitly written as -inf. If False those positions are
                  unspecified.
    out:         optional output on Q.device: [seq_len, seq_len_kv] float32, unit
                  column stride, row stride of at least seq_len_kv. With
                  out_offsets: a contiguous 1-D float32 buffer. A mismatch raises
                  ValueError.
    out_offsets: optional contiguous [seq_len] int32 or int64 on Q.device. Row i
                  writes its scores for keys [cu_starts[i], cu_ends[i]) from
                  element out_offsets[i] of out, so out must hold
                  out_offsets[i] + min(cu_ends[i], seq_len_kv) - max(cu_starts[i], 0)
                  elements. The wrapper does not check this (it needs a host
                  sync). The rows must not overlap. Other elements of out are
                  not written; clean_logits has no effect.
    max_window:  optional host upper bound on cu_ends[i] - cu_starts[i] (e.g. vLLM's
                  max_seqlen_k); picks the kernel and KV split only, so a wrong bound
                  costs speed, not correctness.
    key_vec, k_early, big_tiles: gfx1250 32 x 128 tuning (None: the config):
                  keys per vector store (1, 2 or 4), next K sub-tile load before the
                  WMMAs (matrix-core head sum only), longer KV tiles for many rows x
                  long KV. The logits do not depend on them.

    Returns:
    logits:      out, or an allocated [seq_len, seq_len_kv] float32 tensor. Only
                  with clean_logits and not in packed mode, values outside each
                  row's range are -inf.
    """

    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    # TODO: Currently assuming num_heads and head_size is power of 2.
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    # An allocated output pads each row to 256 columns.
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    packed = out_offsets is not None
    logits = out
    if packed:
        if out is None or out.dtype != torch.float32 or out.dim() != 1:
            raise ValueError("out_offsets requires a 1-D float32 out buffer")
        if out.stride(0) != 1 or out.device != Q.device:
            raise ValueError("packed out must be contiguous and on Q.device")
        if (
            out_offsets.dtype not in (torch.int32, torch.int64)
            or tuple(out_offsets.shape) != (seq_len,)
            or not out_offsets.is_contiguous()
            or out_offsets.device != Q.device
        ):
            raise ValueError(
                "out_offsets must be contiguous [seq_len] int32 or int64 on Q.device"
            )
    elif out is not None:
        if out.dtype != torch.float32 or out.dim() != 2:
            raise ValueError(
                f"out must be a 2-D float32 tensor, got {out.dtype} "
                f"with {out.dim()} dims"
            )
        if tuple(out.shape) != (seq_len, seq_len_kv):
            raise ValueError(
                f"out must be [{seq_len}, {seq_len_kv}], got {tuple(out.shape)}"
            )
        if out.stride(1) != 1 or (seq_len > 1 and out.stride(0) < seq_len_kv):
            raise ValueError(
                "out needs a unit column stride and a row stride of at least "
                f"seq_len_kv={seq_len_kv}, got strides {out.stride()}"
            )
        if out.device != Q.device:
            raise ValueError(f"out is on {out.device}, Q on {Q.device}")
        if clean_logits:
            out.fill_(-float("inf"))
    elif clean_logits:  # -inf outside each row's causal range
        logits = torch.full(
            (seq_len, seq_len_kv_aligned),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv_aligned),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]

    use_gluon = TRITON_GE_36 and _gluon_fp8_mqa_logits_kernel is not None
    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = (0, 1) if packed else logits.stride()
    if seq_len == 0:
        return logits
    if not use_gluon:
        # On gfx942 (MI300X), drop to (64, 1) when our LDS estimate predicts
        # the default (128, 2) tile would not fit two co-resident workgroups
        # on a CU; keep the default tile otherwise.
        if arch == "gfx942" and not _gfx942_tile_fits_lds(
            block_kv=128, head_size=head_size, num_stages=2, occupancy=2
        ):
            block_kv = 64
            num_stages = 1
        else:
            block_kv = 128
            num_stages = 2

        # heuristic for MFMA instruction shape
        matrix_instr_nonkdim = 32
        if seq_len <= 1024:
            matrix_instr_nonkdim = 16

        _fnuz = torch.float8_e4m3fnuz
        # The FN->FNUZ recast + scale compensation is only correct on gfx942,
        # whose fp8 MFMA interprets operands as FNUZ. Other fp8 archs read the
        # operands' native dtype, so converting there would corrupt them.
        convert_q_fn = arch == "gfx942" and Q.dtype != _fnuz
        convert_kv_fn = arch == "gfx942" and KV.dtype != _fnuz
        scale_mul = 1.0
        if convert_q_fn:
            scale_mul *= 2.0
            Q = (Q.to(torch.float32) * 0.5).to(_fnuz)
        if convert_kv_fn:
            scale_mul *= 2.0
            KV = (KV.to(torch.float32) * 0.5).to(_fnuz)
        if scale_mul != 1.0:
            kv_scales = kv_scales.to(torch.float32) * scale_mul

        _fp8_mqa_logits_kernel[(seq_len,)](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
            weights_ptr=weights,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            logits_ptr=logits,
            out_offsets_ptr=out_offsets,
            PACKED=packed,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            stride_q_s=stride_q_s,
            stride_q_h=stride_q_h,
            stride_q_d=stride_q_d,
            stride_kv_s=stride_kv_s,
            stride_kv_d=stride_kv_d,
            stride_w_s=stride_w_s,
            stride_w_h=stride_w_h,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=block_kv,
            num_warps=4,
            num_stages=num_stages,
            waves_per_eu=2,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            enable_fp_fusion=True,
        )
    else:
        # The buffer path keeps the row strides 32-bit and re-bases the pointer
        # per row and per KV tile, so what must fit in int32 is the largest
        # element offset the kernel forms, not the tensor's byte size. The
        # fallback path widens those strides to int64 instead.
        INT32_MAX = 2**31 - 1
        max_kv_offset = (seq_len_kv - 1) * stride_kv_s + (head_size - 1) * stride_kv_d
        max_logits_offset = (
            logits.numel() - 1
            if packed
            else (seq_len - 1) * stride_logits_s + (seq_len_kv - 1) * stride_logits_k
        )
        use_buffer_load = max_kv_offset <= INT32_MAX
        use_buffer_store = max_logits_offset <= INT32_MAX

        num_buffers = 2
        USE_FOLDED_REDUCTION = FOLDED_REDUCTED_SUPPORT and num_heads > 16
        if arch == "gfx950":
            MIN_BLOCK_M2_WGS = 1024

            # Buffer store/load issues are resolved via changing pointer arithmetic
            # so offsets are localized on a shifted pointer
            use_buffer_load = True
            use_buffer_store = True
            num_buffers = 2
            loop_variant = 0
            # Temporary workaround to handle register spill
            waves_per_eu = 2 if TRITON_GE_38 else 3
            num_warps = 2
            block_kv = 64
            # BLOCK_M=2 halves the grid, so it only pays once there are enough
            # rows to spare or the split puts the workgroups back.
            num_kv_splits = _gfx950_kv_splits(
                seq_len, seq_len_kv, 2, num_warps, waves_per_eu
            )
            if (
                num_heads <= 32
                and seq_len >= 2
                and (
                    seq_len > 4096
                    or triton.cdiv(seq_len, 2) * num_kv_splits >= MIN_BLOCK_M2_WGS
                )
            ):
                block_m = 2
            else:
                block_m = 1
                num_kv_splits = _gfx950_kv_splits(
                    seq_len, seq_len_kv, block_m, num_warps, waves_per_eu
                )
            # Single warp to save barrier cycles
            if block_m == 1 and seq_len > 4096:
                num_warps = 1
                block_kv = 32
                num_kv_splits = _gfx950_kv_splits(
                    seq_len, seq_len_kv, block_m, num_warps, waves_per_eu
                )
            # 32x32x64 over 16x16x128: its output layout leaves only one head
            # bit in lanes, so the head sum needs one cross-lane step
            mfma_nonk_dim = 32 if (head_size <= 64 or num_heads >= 32) else 16
            num_chains = (2 if block_m == 2 else 1) if USE_FOLDED_REDUCTION else 0
            # Fold one head chunk at a time to lower reg. pressure
            if (
                num_chains >= 1
                and num_heads > mfma_nonk_dim
                and block_m == 1
                and mfma_nonk_dim == 32
            ):
                m_chunk = mfma_nonk_dim
            else:
                m_chunk = 0
            # Relax the store masking without clean logits; packed rows need exact masks
            relaxed_store = 0 if clean_logits or packed else 1
            other = {
                "USE_PADDED_SHARED_LAYOUT": ASYNC_COPY_SUPPORTS_DISTRIBUTED,
                "BLOCK_M": block_m,
                "MFMA_NONK_DIM": mfma_nonk_dim,
                "M_CHUNK": m_chunk,
                # two KV tiles per loop body for the scheduler to interleave
                "UNROLL": 2,
                "RELAXED_STORE": relaxed_store,
                "HAS_KV_SPLIT": 1 if num_kv_splits > 1 else 0,
                "num_kv_splits": num_kv_splits,
            }
            grid = ((seq_len + block_m - 1) // block_m, num_kv_splits)
        else:
            cfg = _gfx1250_launch_config(seq_len_kv, num_heads, head_size)
            rs = cfg.get("rowsplit")
            # few rows x short KV or windows: the per-row kernel's chain is shorter
            short = rs and (
                (
                    seq_len <= rs.get("PR_MAX_ROWS", 0)
                    and seq_len * seq_len_kv <= rs.get("PR_MAX_WORK", 0)
                )
                or (
                    max_window is not None
                    and seq_len <= rs.get("HINT_MAX_ROWS", 0)
                    and max_window <= rs.get("HINT_MAX_WINDOW", 0)
                )
            )
            if rs and stride_q_d == 1 and stride_kv_d == 1 and not short:
                # 32 x 128 row-split kernel; a logit does not depend on the launch
                rcfg = rs["small"] if seq_len < rs["SMALL_ROWS"] else rs
                # many rows x long KV: longer tiles, fewer barriers per key
                big = rs.get("big", {}) if big_tiles is None or big_tiles else {}
                if (
                    seq_len >= big.get("FROM_ROWS", 1 << 62)
                    and seq_len_kv >= big["FROM_KV"]
                ):
                    rcfg = {**rs, **big}
                rows_per_block = rcfg["BM"] * rcfg["NUM_WARPS"]
                n_blocks = triton.cdiv(seq_len, rows_per_block)
                # a hint well below the KV length: size the splits for one window
                span = seq_len_kv
                if max_window is not None and 2 * (max_window + rows_per_block) <= span:
                    span = max(1, max_window + rows_per_block)
                kv_tiles = triton.cdiv(span, rcfg["BLOCK_KV"])
                num_kv_splits = max(
                    1,
                    min(
                        triton.cdiv(rs["TARGET_PROGRAMS"], n_blocks),
                        triton.cdiv(kv_tiles, rcfg["MIN_TILES_PER_SPLIT"]),
                    ),
                )
                # few row blocks: equal splits down to one tile, up to MIN_PROGRAMS
                boost = min(triton.cdiv(rs.get("MIN_PROGRAMS", 0), n_blocks), kv_tiles)
                if boost > num_kv_splits:
                    num_kv_splits = triton.cdiv(kv_tiles, triton.cdiv(kv_tiles, boost))
                kv_scales_flat = kv_scales.reshape(-1).contiguous()
                # buffer stores are re-based per row block
                rs_buffer_store = (
                    not packed
                    and rows_per_block * stride_logits_s + seq_len_kv <= INT32_MAX
                )
                if key_vec is None:
                    # 4-key stores need 16-byte aligned rows; 1 at many rows per key
                    aligned = (
                        not packed
                        and stride_logits_s % 16 == 0
                        and logits.data_ptr() % 16 == 0
                    )
                    key_vec = rcfg.get("KEY_VEC", 1)
                    key_vec = key_vec if aligned else min(key_vec, 2)
                    max_rpk = rcfg.get("KEY_VEC_MAX_RPK", float("inf"))
                    key_vec = 1 if seq_len / seq_len_kv >= max_rpk else key_vec
                if key_vec not in (1, 2, 4):
                    raise ValueError(f"key_vec must be 1, 2 or 4, got {key_vec}")
                while rcfg["BLOCK_KV"] % (rcfg["SUB_N"] * key_vec):
                    key_vec //= 2  # a key group must fit in a tile
                if k_early is None:
                    k_early = rcfg.get("K_EARLY", 0)
                _gluon_fp8_mqa_logits_rowsplit_kernel[(n_blocks, num_kv_splits)](
                    Q,
                    KV,
                    kv_scales_flat,
                    weights,
                    cu_starts,
                    cu_ends,
                    logits,
                    out_offsets,
                    seq_len=seq_len,
                    seq_len_kv=seq_len_kv,
                    stride_q_s=stride_q_s,
                    stride_q_h=stride_q_h,
                    stride_kv_s=stride_kv_s,
                    stride_w_s=stride_w_s,
                    stride_w_h=stride_w_h,
                    stride_logits_s=stride_logits_s,
                    num_kv_splits=num_kv_splits,
                    BM=rcfg["BM"],
                    NUM_WARPS=rcfg["NUM_WARPS"],
                    BLOCK_KV=rcfg["BLOCK_KV"],
                    SUB_N=rcfg["SUB_N"],
                    NUM_BUFFERS=rcfg["NUM_BUFFERS"],
                    HEADSUM_MMA=bool(rcfg["HEADSUM_MMA"]),
                    USE_BUFFER_STORE=rs_buffer_store,
                    EXACT_ROWS=clean_logits or packed,
                    PACKED=packed,
                    KEY_VEC=key_vec,
                    K_EARLY=bool(k_early) and bool(rcfg["HEADSUM_MMA"]),
                    num_warps=rcfg["NUM_WARPS"],
                    waves_per_eu=rcfg["waves_per_eu"],
                )
                return logits
            loop_variant = cfg["LOOP_VARIANT"]
            waves_per_eu = cfg["waves_per_eu"]
            num_chains = cfg["NUM_CHAINS"] if USE_FOLDED_REDUCTION else 0
            num_warps = cfg["NUM_WARPS"]
            block_kv = cfg["BLOCK_KV"]
            # This kernel has no BLOCK_M: it walks one query row per program.
            block_m = 1
            other = {"LOOP_VARIANT": loop_variant}
            grid = ((seq_len + block_m - 1) // block_m,)

        _gluon_fp8_mqa_logits_kernel[grid](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
            weights_ptr=weights,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            logits_ptr=logits,
            out_offsets_ptr=out_offsets,
            PACKED=packed,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            stride_q_s=stride_q_s,
            stride_q_h=stride_q_h,
            stride_q_d=stride_q_d,
            stride_kv_s=stride_kv_s,
            stride_kv_d=stride_kv_d,
            stride_w_s=stride_w_s,
            stride_w_h=stride_w_h,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=block_kv,
            NUM_WARPS=num_warps,
            NUM_BUFFERS=num_buffers,
            NUM_CHAINS=num_chains,
            USE_BUFFER_LOAD=use_buffer_load,
            USE_BUFFER_STORE=use_buffer_store,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            **other,
            enable_fp_fusion=True,
        )

    return logits


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    out: torch.Tensor | None = None,
    max_ctx: int | None = None,
    early_q: bool | None = None,
) -> torch.Tensor:
    """
    gfx1250 decode / MTP-verify indexer logits over a paged, block-flat fp8 K cache.

    q:            [B, next_n, 32, 128] float8_e4m3fn, contiguous per request
    kv_cache:     [num_blocks, block_size, 1, 132] uint8; per page block_size x 128 fp8
                  values, then block_size fp32 scales (vLLM's NORMAL indexer cache)
    weights:      [B * next_n, 32] float32, contiguous
    context_lens: [B] int32 (row n of request b sees ctx - next_n + n + 1 keys), or
                  [B, next_n] / [B * next_n] int32, the keys each row sees
    block_tables: [B, max_blocks] int32
    out:          optional [B * next_n, >= max_model_len] float32, unit column stride
    max_ctx:      optional host upper bound on the context lengths (not while capturing
                  a CUDA graph); sizes the KV split, and small B * next_n * max_ctx runs
                  a per-row kernel (other head-sum order, not bit-identical). A wrong
                  bound costs speed, not correctness.
    early_q:      load Q and the weights before the context lengths (None: on).

    Returns out (or a new [B * next_n, max_model_len] float32 tensor) holding
    relu(q . k) . weights * scale for the keys each row sees; other entries untouched.
    """
    batch, next_n, num_heads, head_size = q.shape
    num_blocks, block_size = kv_cache.shape[:2]
    if (
        arch not in ("gfx1250", "gfx1250-strict")
        or _gluon_fp8_mqa_logits_kernel is None
    ):
        raise NotImplementedError("fp8_paged_mqa_logits: gfx1250 Gluon kernel only")
    if (num_heads, head_size) != (32, 128) or next_n > 8 or block_size % 32:
        raise NotImplementedError(
            "fp8_paged_mqa_logits: 32 heads x 128, next_n <= 8, block_size % 32 == 0"
        )
    if out is None:
        out = torch.empty(
            batch * next_n, max_model_len, dtype=torch.float32, device=q.device
        )
    cache = kv_cache.reshape(num_blocks, -1)
    ctx = context_lens.reshape(-1)
    ctx_per_row = ctx.numel() == batch * next_n and next_n > 1
    span = max_model_len if max_ctx is None else max(1, min(max_ctx, max_model_len))
    rows = batch * next_n
    max_work = _PAGED_ROWS_MAX_WORK.get(next_n, _PAGED_ROWS_MAX_WORK_DEFAULT)
    if max_ctx is not None and rows * span <= max_work:
        splits = max(1, min(triton.cdiv(span, block_size), triton.cdiv(1024, rows)))
        _fp8_paged_mqa_logits_rows_kernel[(rows, splits)](
            q,
            cache.view(torch.float8_e4m3fn),
            cache.view(torch.float32),
            weights,
            ctx,
            block_tables,
            out,
            q.stride(0),
            cache.stride(0),
            block_tables.stride(0),
            out.stride(0),
            block_tables.shape[1],
            num_blocks,
            max_model_len,
            NEXT_N=next_n,
            CTX_PER_ROW=ctx_per_row,
            PAGE=block_size,
            N_SPLITS=splits,
            NUM_STAGES=2,
            num_warps=4,
        )
        return out
    # one row per warp (the shortest chain); 2 for next_n > 4 in a larger batch
    bm, nw = (1, triton.next_power_of_2(next_n))
    if next_n > 4 and batch > 4:
        bm, nw = 2, 4
    target = {1: 2048, 2: 2048, 4: 1024}.get(nw, 512) if bm == 1 else 512
    splits = max(1, min(triton.cdiv(target, batch), triton.cdiv(span, block_size)))
    _gluon_fp8_paged_mqa_logits_kernel[(batch, splits)](
        q,
        cache.view(torch.float8_e4m3fn),
        cache.view(torch.float32),
        weights,
        ctx,
        block_tables,
        out,
        stride_q_b=q.stride(0),
        stride_blk=cache.stride(0),
        stride_bt=block_tables.stride(0),
        stride_logits=out.stride(0),
        max_blocks=block_tables.shape[1],
        num_blocks=num_blocks,
        max_model_len=max_model_len,
        num_kv_splits=splits,
        NEXT_N=next_n,
        CTX_PER_ROW=ctx_per_row,
        PAGE=block_size,
        BM=bm,
        NUM_WARPS=nw,
        SUB_N=32,
        NUM_BUFFERS=3,
        EARLY_Q=early_q is None or bool(early_q),
        num_warps=nw,
    )
    return out
