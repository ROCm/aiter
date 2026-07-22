# SPDX-License-Identifier: MIT

"""High-level FlyDSL decode TopK-per-row API."""

from __future__ import annotations
import functools
import math
import os
import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode_tiered import (
    BLOCK_THREADS as _TIERED_BLOCK_THREADS,
    LOAD_VEC as _TIERED_LOAD_VEC,
    SCAN_STAGES as _TIERED_SCAN_STAGES,
    create_topk_per_row_decode_tiered_kernel,
    needs_workspace_zero,
    topk_workspace_slots,
)

from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP


##################################

# Independent of K: K only affects the final O(K) index scatter, negligible vs
# the O(L) scan.
_TIERED_MID_MAX = 65536

# The short-vs-multi-block crossover is independent of K for the same reason,
# but does depend on the rows: short_max = min(cap, base + num_rows*slope).
# These params were found empirically on MI300X and MI355X
_SHORT_MAX_PARAMS = {
    # arch:   (base, slope, cap)
    "gfx942": (16384, 1536, 40960),
    "gfx950": (18432, 1536, 40960),
}


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


@functools.cache
def _multi_processor_count(arch: str | None = None) -> int:
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return int(props.multi_processor_count)
    except Exception:
        fallback_min_cu = {"gfx942": 228, "gfx950": 256}
        return fallback_min_cu.get(arch or get_rocm_arch(), 64)


@functools.cache
def _environ_kernel_config() -> dict:
    cfg = dict(
        scan_stages=_env_int("FLYDSL_TOPK_SCAN_STAGES"),
        tiered_short_max=_env_int("FLYDSL_TOPK_TIERED_SHORT_MAX"),
        tiered_mid_cap=_env_int("FLYDSL_TOPK_TIERED_MID_CAP"),
        tiered_mid_max=_env_int("FLYDSL_TOPK_TIERED_MID_MAX"),
        tiered_long_cap=_env_int("FLYDSL_TOPK_TIERED_LONG_CAP"),
        bits_per_pass=_env_int("FLYDSL_TOPK_TIERED_BPP"),
        # 0/1 override for the non-finite mask (default on); set 0 to disable.
        mask_non_finite=_env_int("FLYDSL_TOPK_TIERED_MASK_NONFINITE"),
        # Force a single tier for every row (auto/short/mid/long)
        tier_mode=os.environ.get("FLYDSL_TOPK_TIERED_OVERRIDE"),
    )
    return {k: v for k, v in cfg.items() if v is not None}


def _default_kernel_config(
    num_rows: int,
    max_model_len: int,
) -> dict:
    arch = get_rocm_arch()

    # Grid width per row: enough workgroups to cover the row at LOAD_VEC elements
    # per thread, clamped to [2, 32] (32 = the wg cap the mid/long tiers can use;
    # BLOCK_THREADS is fixed at 1024, so max_blocks is 32). blocks_per_row is
    # rounded to the next pow 2 to reduce the number of compilations.
    items_per_block = _TIERED_LOAD_VEC * _TIERED_BLOCK_THREADS
    raw_blocks_per_row = max(2, math.ceil(max_model_len / items_per_block))
    blocks_per_row = min(32, _next_pow2(raw_blocks_per_row))

    # bits_per_pass: 11 (2048-bin LDS histogram) whenever the arch can afford it;
    # the short tier requires 11. gfx942/gfx950 both qualify (CU count >= 128).
    cu_count = _multi_processor_count(arch)
    bits_per_pass = (
        11 if cu_count >= 128 or SMEM_CAPACITY_MAP.get(arch, 0) >= 128 * 1024 else 10
    )

    # Max cooperating workgroups per row for the mid/long tiers (the real wg count
    # is min(blocks_per_row, cap)). Scales down with batch size: a single long row
    # wants the full wg32 set, while multi-row batches already fill the device so
    # fewer workgroups/row cut barrier and histogram-merge cost.
    if num_rows <= 1:
        tiered_mid_cap_default = 32
    else:  # num_rows > 1
        tiered_mid_cap_default = 8

    if num_rows <= 1:
        tiered_long_cap_default = 32
    elif num_rows < 8:
        tiered_long_cap_default = 16
    else:  #  num_rows >= 8
        tiered_long_cap_default = 8

    # Batch-aware short vs multi-block crossover (arch-specific base/slope/cap). The
    # multi-block barrier floor grows under CU contention as more rows launch, while
    # the single-workgroup path is flat in batch. Bucket num_rows to the next pow 2
    # for the crossover, so nearby batch sizes share one compiled kernel.
    base, slope, cap = _SHORT_MAX_PARAMS.get(arch, _SHORT_MAX_PARAMS["gfx942"])
    short_max_rows = _next_pow2(num_rows)
    tiered_short_max = min(cap, base + short_max_rows * slope)

    # Dead-block trim (gfx950 only). The launch grid is (blocks_per_row, num_rows) but
    # the real workers per row = min(blocks_per_row, tier_cap); the excess workgroups
    # return immediately yet still occupy co-resident slots ("dead" blocks). Trimming
    # blocks_per_row down to that cap leaves active_parts (a min) unchanged, so results
    # are identical and only wasted scheduling is cut. Applied only when the full padded
    # grid (32*num_rows) already fits one co-resident wave (cu_count*occ, occ=2 ->
    # num_rows <= 16); beyond that the extra blocks help hide latency, so those are left
    # to the batch co-resident cap. blocks_per_row stays >= 2 here (the bpr==1 grid=1
    # fold requires the kernel's single-workgroup launch path). Env
    # FLYDSL_TOPK_TIERED_TRIM (0/1) overrides; gfx942 is untouched (default 0).
    trim_on = _env_int("FLYDSL_TOPK_TIERED_TRIM", 1 if arch == "gfx950" else 0)
    if trim_on and max_model_len > tiered_short_max and num_rows * 32 <= cu_count * 2:
        if max_model_len <= _TIERED_MID_MAX:
            max_active_parts = tiered_mid_cap_default
        else:
            max_active_parts = max(tiered_mid_cap_default, tiered_long_cap_default)
        blocks_per_row = max(2, min(blocks_per_row, max_active_parts))

    # Mid-batch coordination cap (gfx950 only). Once the batch alone fills the device
    # (num_rows > 16), the co-resident budget over-provisions blocks_per_row; the measured
    # cooperation optimum in the multi-block regime is well below it. Cap blocks_per_row by
    # a small L-keyed step. Only reduces blocks_per_row (a min), so results stay valid.
    # (The rows>63 short-mid cap of 1 is omitted; it needs the kernel's single-workgroup
    # launch path.) Env FLYDSL_TOPK_TIERED_MIDBATCH_CAP overrides the matched cap (0 off).
    if arch == "gfx950" and max_model_len > tiered_short_max:
        mb_cap = None
        for mb_min_rows, mb_L_max, mb_cap_val in (
            (16, 131072, 4),
            (20, None, 6),
            (16, None, 8),
        ):
            if num_rows > mb_min_rows and (mb_L_max is None or max_model_len <= mb_L_max):
                mb_cap = mb_cap_val
                break
        mb_env = _env_int("FLYDSL_TOPK_TIERED_MIDBATCH_CAP")
        if mb_env is not None and mb_cap is not None:
            mb_cap = mb_env
        if mb_cap:
            blocks_per_row = max(2, min(blocks_per_row, mb_cap))

    # Row-proportional parts (gfx950 only). The launch grid width is sized for the
    # padded buffer, so short rows over-provision cooperating workgroups; the kernel
    # caps participating parts by each row's actual coverage need (a min, so results
    # are unchanged). Env FLYDSL_TOPK_TIERED_RPP (0/1) overrides; gfx942 frozen (0).
    rpp_on = _env_int("FLYDSL_TOPK_TIERED_RPP", 1 if arch == "gfx950" else 0)

    return dict(
        blocks_per_row=blocks_per_row,
        bits_per_pass=bits_per_pass,
        scan_stages=_TIERED_SCAN_STAGES,
        tiered_short_max=tiered_short_max,
        tiered_mid_cap=tiered_mid_cap_default,
        tiered_mid_max=_TIERED_MID_MAX,
        tiered_long_cap=tiered_long_cap_default,
        mask_non_finite=True,
        tier_mode="auto",
        row_proportional_parts=bool(rpp_on),
    )


def _kernel_config(num_rows: int, max_model_len: int) -> dict:
    default_config = _default_kernel_config(num_rows, max_model_len)
    environ_config = _environ_kernel_config()

    kernel_config = {
        **default_config,
        **environ_config,
    }

    bits_per_pass = kernel_config["bits_per_pass"]
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")

    tier_mode = kernel_config["tier_mode"]
    if tier_mode not in ("auto", "short", "mid", "long"):
        raise ValueError(
            "FLYDSL_TOPK_TIERED_OVERRIDE must be one of auto/short/mid/long, "
            f"got {tier_mode!r}"
        )

    kernel_config = _apply_deadlock_guard(kernel_config, num_rows, max_model_len)
    return kernel_config


def _apply_deadlock_guard(
    kernel_config: dict,
    num_rows: int,
    max_model_len: int,
) -> dict:
    """Clamp the tiered config so the mid/long-tier inter-workgroup barrier cannot
    deadlock. The possibility of deadlock requires both a wide batch (num_rows > ~80-90)
    and long rows (L > ~16-40K). Note: The HIP kernel also redirects to a
    single-workgroup tier for large batch-sizes, but doesn't explicitly call out the
    deadlock risk (see the should_use_mulblocks function).

    The tiered kernels spin on a barrier over a non-cooperative launch, which gives
    no guarantee that a row's participating workgroups are all resident at the same
    time. A workgroup spinning at the barrier holds its slot until every participant
    of its row arrives. A deadlock can potentially happen once the barrier-blocked
    workgroups exceeds the resident capacity. In this case, we cap the active
    workgroups or force the barrier-free short tier (single workgroup/row).

    Example for MI300X:
    The deadlock guard is active around num_rows >= 87 with max_model_len > 32768
    CUs * Occupancy = 304 * 2 = 608
    8 cooperating workgroups, 87*7 >= 608
    """
    if num_rows <= 0:
        return kernel_config

    mode = kernel_config["tier_mode"]
    if mode == "short":
        return kernel_config  # single workgroup/row -> barrier-free

    mid_cap = kernel_config["tiered_mid_cap"]
    long_cap = kernel_config["tiered_long_cap"]
    blocks_per_row = kernel_config["blocks_per_row"]

    # Worst-case cooperating workgroups any single row can put on the barrier.
    # Forced mid/long use that tier's cap for every row; auto only reaches a
    # multi-block tier for rows longer than short_max.
    if mode == "mid":
        max_active_workgroups_per_row = min(blocks_per_row, mid_cap)
    elif mode == "long":
        max_active_workgroups_per_row = min(blocks_per_row, long_cap)
    else:  # auto
        if max_model_len <= kernel_config["tiered_short_max"]:
            return kernel_config  # all rows short-tier -> barrier-free
        if max_model_len <= kernel_config["tiered_mid_max"]:
            max_active_workgroups_per_row = min(blocks_per_row, mid_cap)
        else:
            max_active_workgroups_per_row = min(blocks_per_row, max(mid_cap, long_cap))

    if max_active_workgroups_per_row <= 1:
        return kernel_config  # single-workgroup -> barrier-free

    # Co-resident envelope N = num_CU x occupancy. Occupancy is 2 on all CDNA:
    # the 1024-thread block is wave-limited (32 waves/CU / 16), with VGPR/LDS
    # headroom (measured gfx942: VGPR=40, LDS=8.7KB). Re-check if scan_stages or
    # the histogram grows enough to push VGPR>64 / LDS>32KB (would drop occ to 1).
    max_coresident_workgroups = _multi_processor_count() * 2
    is_deadlock_free = (
        num_rows * (max_active_workgroups_per_row - 1) < max_coresident_workgroups
    )
    if is_deadlock_free:
        return kernel_config

    # Largest cap A satisfying num_rows * (A - 1) < N.
    max_safe_active_workgroups = (max_coresident_workgroups - 1) // num_rows + 1
    if max_safe_active_workgroups >= 2:
        kernel_config["tiered_mid_cap"] = min(mid_cap, max_safe_active_workgroups)
        kernel_config["tiered_long_cap"] = min(long_cap, max_safe_active_workgroups)
    else:
        # max_safe_active_workgroups < 2 -> force short tier
        kernel_config["tier_mode"] = "short"
    return kernel_config


def flydsl_top_k_per_row_decode_workspace_size(
    num_rows: int,
    max_model_len: int,
) -> int:
    """
    Number of int32 elements the decode TopK workspace needs for this shape.
    max_model_len = int(logits.shape[1])
    """
    if num_rows <= 0:
        return 0

    kernel_config = _kernel_config(num_rows, max_model_len)
    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    return workspace_slots


@functools.lru_cache(maxsize=16384)
def _build_launcher(
    top_k: int,
    num_rows: int,
    max_model_len: int,
):
    """Build (and lru-cache) the launcher + workspace metadata for this shape.

    Returns the flyc.jit launcher object, does not compile. The first
    _run_compiled() call triggers flyc.compile.

    Cached per unique (top_k, num_rows, max_model_len).
    """
    kernel_config = _kernel_config(num_rows, max_model_len)

    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    workspace_zero = needs_workspace_zero(
        max_model_len,
        top_k,
        kernel_config["tiered_short_max"],
        tier_mode=kernel_config["tier_mode"],
    )
    launcher = create_topk_per_row_decode_tiered_kernel(
        top_k=top_k,
        **kernel_config,
    )
    return launcher, workspace_slots, workspace_zero


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")


def _required_seq_rows(next_n: int, num_rows: int) -> int:
    if num_rows <= 0:
        return 0
    return math.ceil(num_rows / next_n)


def _validate_inputs(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    ordered: bool = False,
    workspace: torch.Tensor | None = None,
):
    _check_cuda_tensor("logits", logits)
    _check_cuda_tensor("seqLens", seqLens)
    _check_cuda_tensor("indices", indices)

    if logits.dtype is not torch.float32:
        raise TypeError(f"logits must be torch.float32, got {logits.dtype}")
    if seqLens.dtype is not torch.int32:
        raise TypeError(f"seqLens must be torch.int32, got {seqLens.dtype}")
    if indices.dtype is not torch.int32:
        raise TypeError(f"indices must be torch.int32, got {indices.dtype}")
    if logits.device != seqLens.device or logits.device != indices.device:
        raise ValueError("logits, seqLens, and indices must be on the same device")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape={tuple(indices.shape)}")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if numRows < 0:
        raise ValueError(f"numRows must be non-negative, got {numRows}")
    if numRows > logits.shape[0]:
        raise ValueError(f"numRows={numRows} exceeds logits rows={logits.shape[0]}")
    if numRows > indices.shape[0]:
        raise ValueError(f"numRows={numRows} exceeds indices rows={indices.shape[0]}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if ordered:
        raise ValueError(
            "FlyDSL decode TopK returns unordered set output only; "
            "ordered=True is not supported"
        )
    if indices.shape[1] < k:
        raise ValueError(f"indices second dimension must be at least k={k}")
    if indices.stride(1) != 1:
        raise ValueError("indices must have contiguous per-row storage")
    if stride1 != 1:
        raise NotImplementedError(
            f"FlyDSL decode TopK currently supports stride1 == 1 only, got {stride1}"
        )
    if stride0 != logits.stride(0) or stride1 != logits.stride(1):
        raise ValueError(
            "stride0/stride1 must match logits.stride(); received "
            f"({stride0}, {stride1}) for logits.stride()={logits.stride()}"
        )

    required_seq_rows = _required_seq_rows(next_n, numRows)
    if required_seq_rows > seqLens.numel():
        raise ValueError(
            f"numRows={numRows} with next_n={next_n} requires at least "
            f"{required_seq_rows} seqLens entries, got {seqLens.numel()}"
        )

    if workspace is not None:
        _check_cuda_tensor("workspace", workspace)
        if workspace.dtype is not torch.int32:
            raise TypeError(f"workspace must be torch.int32, got {workspace.dtype}")
        if workspace.device != logits.device:
            raise ValueError("workspace must be on the same device as logits")


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
    stream: torch.cuda.Stream | None = None,
    ordered: bool = False,
    workspace: torch.Tensor | None = None,
) -> None:
    if numRows == 0:
        return

    _validate_inputs(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k,
        ordered,
        workspace,
    )

    launcher, workspace_slots, workspace_zero = _build_launcher(
        k,
        numRows,
        logits.shape[1],
    )

    if workspace is None:
        workspace = torch.empty(
            workspace_slots,
            dtype=torch.int32,
            device=logits.device,
        )
    elif workspace.numel() < workspace_slots:
        raise ValueError(
            f"workspace too small: need >= {workspace_slots} int32 "
            f"elements, got {workspace.numel()} (use "
            f"flydsl_top_k_per_row_decode_workspace_size)"
        )

    if stream is None:
        stream = torch.cuda.current_stream(logits.device)

    if workspace_zero:
        with torch.cuda.stream(stream):
            workspace.zero_()

    with torch.cuda.device(logits.device.index):
        _run_compiled(
            launcher,
            logits,
            int(next_n),
            seqLens,
            indices,
            workspace,
            int(numRows),
            int(stride0),
            int(stride1),
            stream,
        )


def flydsl_top_k_per_row_decode_unordered(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Benchmark-friendly wrapper for the unordered set-output path."""

    flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k=k,
        stream=stream,
        ordered=False,
    )
