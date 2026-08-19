import json
import os

import torch

from aiter.ops.triton._triton_kernels.attention.fp8_mqa_logits import (
    _fp8_mqa_logits_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
import inspect
from typing import Optional

from packaging.version import Version
import triton

TRITON_VERSION = Version(triton.__version__)
TRITON_GE_36 = TRITON_VERSION >= Version("3.6.0")

arch = arch_info.get_arch()
_gluon_fp8_mqa_logits_kernel = None
if TRITON_GE_36:
    try:
        if arch == "gfx950":
            from aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits import (
                _gluon_fp8_mqa_logits_kernel,
            )
        elif arch == "gfx1250":
            from aiter.ops.triton._gluon_kernels.gfx1250.attention.fp8_mqa_logits import (
                _gluon_fp8_mqa_logits_kernel,
            )
    except Exception:
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


# gfx942 (MI300X) non-gluon indexer tile selection.
#
# The tile is *not* tuned at runtime: it is read from a checked-in, arch
# specific config file (configs/{arch}-FP8_MQA_LOGITS.json), so a given
# (arch, head_size, seq_len bucket) always compiles the same kernel. The
# config files are produced offline by
# op_tests/op_benchmarks/triton/tune_fp8_mqa_logits.py.
#
# A static LDS check still guards the config: if the configured tile cannot
# fit a single workgroup's KV tile in 64KB (e.g. an unusually large
# head_size, or an older Triton that spills Q/scores to LDS), we shrink
# deterministically down to (BLOCK_KV=64, num_stages=1) -- the config #3257
# validated as crash-safe for the DSv4 indexer on MI300X. This is pure
# arithmetic on the launch parameters, so it is still fully predictable.
_MQA_SEQ_BUCKETS = (1024, 4096, 16384, 65536)
_MQA_DEFAULT_CONFIG = {"BLOCK_KV": 128, "num_stages": 2}


def _mqa_seq_bucket_key(seq_len: int) -> str:
    for b in _MQA_SEQ_BUCKETS:
        if seq_len <= b:
            return f"SEQ_LEN_LEQ_{b}"
    return f"SEQ_LEN_GT_{_MQA_SEQ_BUCKETS[-1]}"


def _get_config(head_size: int, seq_len: int) -> dict:
    """
    Tile config for the non-gluon fp8_mqa_logits kernel.

    Looked up in configs/{arch}-FP8_MQA_LOGITS.json as
    ``["HEAD_SIZE={head_size}"]["SEQ_LEN_LEQ_{bucket}"]``, falling back to the
    per-head_size ``"default"``, then the file-level ``"default"``, then the
    built-in (BLOCK_KV=128, num_stages=2). Archs without a config file use the
    built-in default, i.e. behave exactly as before this file existed.
    """
    if not hasattr(_get_config, "_config_dict"):
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{arch}-FP8_MQA_LOGITS.json"
        cfg = {}
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                cfg = json.load(file)
        _get_config._config_dict = cfg

    file_cfg = _get_config._config_dict
    config = dict(_MQA_DEFAULT_CONFIG)
    config.update(file_cfg.get("default", {}))
    head_cfg = file_cfg.get(f"HEAD_SIZE={head_size}")
    if head_cfg is not None:
        config.update(head_cfg.get("default", {}))
        config.update(head_cfg.get(_mqa_seq_bucket_key(seq_len), {}))
    return config


def _gfx942_lds_safe_tile(
    block_kv: int, num_stages: int, head_size: int
) -> tuple[int, int]:
    # Deterministically shrink a configured tile until its KV tile fits LDS for
    # one workgroup. (64, 1) is the crash-safe floor from #3257.
    while not _gfx942_tile_fits_lds(block_kv, head_size, num_stages, occupancy=1):
        if num_stages > 1:
            num_stages -= 1
        elif block_kv > 64:
            block_kv //= 2
        else:
            return 64, 1
    return block_kv, num_stages


def fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
    config: Optional[dict] = None,
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
                  are explicitly written as -inf. If False, the kernel skips writing
                  those positions and leaves whatever was in the output buffer there
                  (the caller is responsible for pre-filling with -inf or ignoring them).
    config:      Optional dict of kernel tuning parameters (BLOCK_KV, num_stages).
                  When None (default), the config is read from the checked-in
                  arch-specific config file. No tuning happens at runtime.

    Returns:
    logits:      [seq_len, seq_len_kv], dtype float32 (must be initialized to -inf, because of causal masking)
    """

    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    # TODO: Currently assuming num_heads and head_size is power of 2.
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    # Initialize with -inf because of causal masking
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    if clean_logits:
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
    stride_logits_s, stride_logits_k = logits.stride()
    if not use_gluon:
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

        def _launch(block_kv, num_stages):
            _fp8_mqa_logits_kernel[(seq_len,)](
                Q_ptr=Q,
                KV_ptr=KV,
                kv_scales_ptr=kv_scales,
                weights_ptr=weights,
                cu_start_ptr=cu_starts,
                cu_end_ptr=cu_ends,
                logits_ptr=logits,
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
            )

        # Tile comes from the checked-in arch-specific config (no runtime
        # tuning); on gfx942 a static LDS check can only shrink it.
        if config is None:
            config = _get_config(head_size, seq_len)
        block_kv = config.get("BLOCK_KV", _MQA_DEFAULT_CONFIG["BLOCK_KV"])
        num_stages = config.get("num_stages", _MQA_DEFAULT_CONFIG["num_stages"])
        if arch == "gfx942":
            block_kv, num_stages = _gfx942_lds_safe_tile(
                block_kv, num_stages, head_size
            )
        _launch(block_kv, num_stages)
    else:
        num_buffers = 2
        USE_FOLDED_REDUCTION = FOLDED_REDUCTED_SUPPORT and num_heads > 16
        if arch == "gfx950":
            num_buffers = 2
            loop_variant = 0
            waves_per_eu = 3
            num_chains = 4 if USE_FOLDED_REDUCTION else 0
            num_warps = 1
            block_kv = 32
            other = {"USE_PADDED_SHARED_LAYOUT": ASYNC_COPY_SUPPORTS_DISTRIBUTED}
        else:
            loop_variant = 1
            waves_per_eu = 1
            num_chains = 8 if USE_FOLDED_REDUCTION else 0
            num_warps = 4
            block_kv = 128
            other = {"LOOP_VARIANT": loop_variant}

        # Buffer ops use a 32-bit byte offset (2 GiB resource descriptor cap).
        # Fall back to plain global load/store when a tensor exceeds that.
        BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
        use_buffer_load = KV.numel() * KV.element_size() < BUFFER_LIMIT_BYTES
        use_buffer_store = logits.numel() * logits.element_size() < BUFFER_LIMIT_BYTES
        _gluon_fp8_mqa_logits_kernel[(seq_len,)](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
            weights_ptr=weights,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            logits_ptr=logits,
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
        )

    return logits
