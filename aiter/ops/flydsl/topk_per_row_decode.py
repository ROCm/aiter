# SPDX-License-Identifier: MIT

"""High-level FlyDSL decode TopK-per-row API."""

from __future__ import annotations

import functools
import math
import os

import torch
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode_tiered import BLOCK_THREADS as _TIERED_BLOCK_THREADS
from .kernels.topk_per_row_decode_tiered import LOAD_VEC as _TIERED_LOAD_VEC
from .kernels.topk_per_row_decode_tiered import SCAN_STAGES as _TIERED_SCAN_STAGES
from .kernels.topk_per_row_decode_tiered import (
    create_topk_per_row_decode_tiered_kernel,
    needs_workspace_zero,
    topk_workspace_slots,
)

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

# Co-resident scheduling envelope = CU_count * occupancy. Occupancy is 2 on CDNA
# (1024-thread wave-limited block). Beyond _COCAP_OCC2_MAX_ROWS rows the 512-wg
# envelope spills into a 2nd wave, so the batch cap switches to occ=1 (one true wave).
_CDNA_OCCUPANCY = 2
_COCAP_OCC2_MAX_ROWS = 32


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


_FALLBACK_MIN_CU = {"gfx942": 228, "gfx950": 256}


def _arch_floor(arch: str | None) -> int:
    """Smallest CU count the arch is known to ship with.

    Used when no device can answer for the arch being asked about. Guessing low is
    the safe direction: the count sizes a co-residency guard, and an over-count is
    the one that lets a grid through that cannot be co-resident.
    """
    return _FALLBACK_MIN_CU.get((arch or get_rocm_arch()).lower(), 64)


def _multi_processor_count(arch: str | None = None) -> int:
    """CU count of the device that will run the kernel, or the arch floor.

    Cached per (device, arch), not per arch: the count sizes the co-residency guard
    in _deadlock_safe_config, and one gfx942 name spans 80 to 304 CU. Caching under
    the arch alone would let a multi-device process, or an AOT build on a large host,
    reuse an over-count -- which loosens the guard and is what deadlocks. Device
    properties are believed only when the live device is the arch asked for.
    """
    try:
        device = torch.cuda.current_device()
    except Exception:  # noqa: BLE001 - no device visible during AOT; use the floor
        device = None
    return _cu_count_for(device, arch)


def decode_cu_count(device: torch.device | int | None, arch: str | None = None) -> int:
    """CU count of `device`, which is not always the current device.

    Pass None for the current-device behaviour, which is what AOT needs.
    """
    if isinstance(device, torch.device):
        device = device.index
    if device is None:
        return _multi_processor_count(arch)
    return _cu_count_for(device, arch or get_rocm_arch())


@functools.cache
def _cu_count_for(device: int | None, arch: str | None) -> int:
    if device is None:
        return _arch_floor(arch)
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:  # noqa: BLE001 - properties unreadable; use the floor
        return _arch_floor(arch)
    if arch is None:
        return int(props.multi_processor_count)
    # gcnArchName carries feature suffixes: "gfx942:sramecc+:xnack-".
    live_arch = getattr(props, "gcnArchName", "").split(":")[0].lower()
    if live_arch != arch.lower():
        return _arch_floor(arch)
    return int(props.multi_processor_count)


@functools.cache
def _environ_kernel_config() -> dict:
    cfg = {
        "scan_stages": _env_int("FLYDSL_TOPK_SCAN_STAGES"),
        "tiered_short_max": _env_int("FLYDSL_TOPK_TIERED_SHORT_MAX"),
        "tiered_mid_cap": _env_int("FLYDSL_TOPK_TIERED_MID_CAP"),
        "tiered_mid_max": _env_int("FLYDSL_TOPK_TIERED_MID_MAX"),
        "tiered_long_cap": _env_int("FLYDSL_TOPK_TIERED_LONG_CAP"),
        "bits_per_pass": _env_int("FLYDSL_TOPK_TIERED_BPP"),
        # 0/1 override for the non-finite mask (default off, matching torch.topk
        # and HIP); set 1 to mask +inf/NaN out of the selection instead.
        "mask_non_finite": _env_int("FLYDSL_TOPK_TIERED_MASK_NONFINITE"),
        # Force a single tier for every row (auto/short/mid/long)
        "tier_mode": os.environ.get("FLYDSL_TOPK_TIERED_OVERRIDE"),
    }
    return {k: v for k, v in cfg.items() if v is not None}


def _resolved_kernel_config(
    num_rows: int,
    max_model_len: int,
    arch: str | None = None,
    overrides: dict | None = None,
    cu_count: int | None = None,
) -> dict:
    """Derive the tiered config, folding each override in at the point the field is
    defined rather than over the finished dict. Several fields (blocks_per_row above
    all) are derived *from* other fields, so an override applied afterwards would
    leave the config internally inconsistent -- e.g. a forced bits_per_pass of 10
    against a blocks_per_row already collapsed to 1 on the strength of the default
    11, which the kernel then rejects.
    """
    arch = arch or get_rocm_arch()
    overrides = overrides or {}

    # Grid width per row: enough workgroups to cover the row at LOAD_VEC elements
    # per thread, clamped to [2, 32] (32 = the wg cap the mid/long tiers can use;
    # BLOCK_THREADS is fixed at 1024, so max_blocks is 32). blocks_per_row is
    # rounded to the next pow 2 to reduce the number of compilations.
    items_per_block = _TIERED_LOAD_VEC * _TIERED_BLOCK_THREADS
    raw_blocks_per_row = max(2, math.ceil(max_model_len / items_per_block))
    blocks_per_row = min(32, _next_pow2(raw_blocks_per_row))

    # bits_per_pass: 11 (2048-bin LDS histogram) whenever the arch can afford it;
    # the short tier requires 11. gfx942/gfx950 both qualify (CU count >= 128).
    if cu_count is None:
        cu_count = _multi_processor_count(arch)
    bits_per_pass = (
        11 if cu_count >= 128 or SMEM_CAPACITY_MAP.get(arch, 0) >= 128 * 1024 else 10
    )
    bits_per_pass = overrides.get("bits_per_pass", bits_per_pass)
    tier_mode = overrides.get("tier_mode", "auto")

    # The kernel compiles its single-workgroup launch path (blocks_per_row == 1) only
    # when the short tier exists, which needs an 11-bit histogram and a tier mode that
    # can reach that tier. Every grid=1 fold below is gated on this, so a forced
    # bits_per_pass or tier_mode cannot leave behind a width the kernel refuses.
    single_wg_ok = bits_per_pass == 11 and tier_mode in ("auto", "short")

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

    tiered_mid_cap_default = overrides.get("tiered_mid_cap", tiered_mid_cap_default)
    tiered_long_cap_default = overrides.get("tiered_long_cap", tiered_long_cap_default)

    # Batch-aware short vs multi-block crossover (arch-specific base/slope/cap). The
    # multi-block barrier floor grows under CU contention as more rows launch, while
    # the single-workgroup path is flat in batch. Bucket num_rows to the next pow 2
    # for the crossover, so nearby batch sizes share one compiled kernel.
    base, slope, cap = _SHORT_MAX_PARAMS.get(arch, _SHORT_MAX_PARAMS["gfx942"])
    short_max_rows = _next_pow2(num_rows)
    tiered_short_max = min(cap, base + short_max_rows * slope)
    tiered_short_max = overrides.get("tiered_short_max", tiered_short_max)

    # Local copy so the grid=1 fold below can raise it to keep mid_max >= short_max
    # (kernel validation requires it when force_single_wg lifts short_max to L).
    tiered_mid_max = overrides.get("tiered_mid_max", _TIERED_MID_MAX)

    # Dead-block trim (gfx950 only; FLYDSL_TOPK_TIERED_TRIM, gfx942 default 0). A grid
    # wider than the row's tier_cap only adds workgroups that return immediately while
    # holding co-resident slots. Trimming to that cap is a min on active_parts, so
    # results are identical. Skipped once the padded grid outgrows one co-resident wave,
    # where the extra blocks hide latency instead.
    trim_on = _env_int("FLYDSL_TOPK_TIERED_TRIM", 1 if arch == "gfx950" else 0)
    if trim_on:
        if max_model_len <= tiered_short_max and single_wg_ok:
            # Every row short-tier: all but one block is dead and the barrier-free tier
            # has nothing to hide. Collapse to grid=1 via the kernel's bpr==1 path.
            blocks_per_row = 1
        elif (
            max_model_len > tiered_short_max
            and num_rows * 32 <= cu_count * _CDNA_OCCUPANCY
        ):
            if max_model_len <= tiered_mid_max:
                max_active_parts = tiered_mid_cap_default
            else:
                max_active_parts = max(tiered_mid_cap_default, tiered_long_cap_default)
            blocks_per_row = max(2, min(blocks_per_row, max_active_parts))

    # Batch co-resident grid-width cap (gfx950 only; FLYDSL_TOPK_TIERED_BATCH_CAP,
    # gfx942 default 0). Keeps blocks_per_row*num_rows inside one co-resident wave
    # (CU*occ) so the persistent barrier does not serialize rows into separate waves.
    force_single_wg = False
    batch_cap_on = _env_int(
        "FLYDSL_TOPK_TIERED_BATCH_CAP", 1 if arch == "gfx950" else 0
    )
    if batch_cap_on and num_rows > 1:
        # occ=2 co-resides for modest grids; past _COCAP_OCC2_MAX_ROWS the envelope
        # spills into a second wave, so occ=1 is faster. FLYDSL_TOPK_TIERED_OCC forces.
        occ = _env_int("FLYDSL_TOPK_TIERED_OCC")
        if not occ:
            occ = _CDNA_OCCUPANCY if num_rows <= _COCAP_OCC2_MAX_ROWS else 1
        envelope = cu_count * occ
        budget = envelope // num_rows
        if budget >= 2:
            blocks_per_row = min(blocks_per_row, budget)
        elif single_wg_ok:
            # Even a width-2 grid cannot fit the batch in one wave, and the padded grid
            # would launch dead blocks that hold slots and serialize the real workers.
            # Collapse to grid=(1, num_rows), every row barrier-free (kernel bpr==1).
            blocks_per_row = 1
            force_single_wg = True

    if force_single_wg:
        # active_parts=1 everywhere now. Say so explicitly so needs_workspace_zero()
        # stays False and the mid-batch cap below is skipped. The kernel requires
        # mid_max >= short_max, so both move together.
        tiered_short_max = max(tiered_short_max, max_model_len)
        tiered_mid_max = max(tiered_mid_max, tiered_short_max)

    # Mid-batch coordination cap (gfx950 only; FLYDSL_TOPK_TIERED_MIDBATCH_CAP). Once
    # the batch alone fills the device the co-resident budget over-provisions
    # blocks_per_row, so cap it by a small L-keyed step. Only a min, so results stay
    # valid. The rows>63 rule reaches cap=1 without force_single_wg -- short_max stays
    # below L, so the row runs the mid tier with one cooperating block, barrier-free.
    if arch == "gfx950" and max_model_len > tiered_short_max:
        mb_cap = None
        for mb_min_rows, mb_L_max, mb_cap_val in (
            (63, 65536, 1),
            (16, 131072, 4),
            (20, None, 6),
            (16, None, 8),
        ):
            if num_rows > mb_min_rows and (
                mb_L_max is None or max_model_len <= mb_L_max
            ):
                mb_cap = mb_cap_val
                break
        mb_env = _env_int("FLYDSL_TOPK_TIERED_MIDBATCH_CAP")
        if mb_env is not None and mb_cap is not None:
            mb_cap = mb_env
        if mb_cap:
            blocks_per_row = max(1 if single_wg_ok else 2, min(blocks_per_row, mb_cap))

    # Row-proportional parts (gfx950 only; FLYDSL_TOPK_TIERED_RPP, gfx942 default 0).
    # The kernel caps participating parts by each row's own coverage need -- a min, so
    # results are unchanged.
    rpp_on = _env_int("FLYDSL_TOPK_TIERED_RPP", 1 if arch == "gfx950" else 0)

    # Early-stop (gfx950 only; FLYDSL_TOPK_TIERED_ES, gfx942 default 0). Skips the last
    # radix pass when the boundary bucket is taken whole. Single-row only.
    es_on = _env_int("FLYDSL_TOPK_TIERED_ES", 1 if arch == "gfx950" else 0)

    # mask_non_finite stays off (FLYDSL_TOPK_TIERED_MASK_NONFINITE=1 restores it): HIP
    # and torch.topk both rank +inf/NaN by raw twiddled bits, putting them at the top.
    # Masking to -inf here would make the answer depend on which kernel the gate picked,
    # and would bury an upstream numerical failure instead of surfacing it.
    return {
        "blocks_per_row": blocks_per_row,
        "bits_per_pass": bits_per_pass,
        "scan_stages": overrides.get("scan_stages", _TIERED_SCAN_STAGES),
        "tiered_short_max": tiered_short_max,
        "tiered_mid_cap": tiered_mid_cap_default,
        "tiered_mid_max": tiered_mid_max,
        "tiered_long_cap": tiered_long_cap_default,
        "mask_non_finite": bool(overrides.get("mask_non_finite", False)),
        "tier_mode": tier_mode,
        "row_proportional_parts": bool(rpp_on),
        "early_stop": bool(es_on) and num_rows <= 1,
    }


def _kernel_config(
    num_rows: int,
    max_model_len: int,
    arch: str | None = None,
    cu_count: int | None = None,
) -> dict:
    kernel_config = _resolved_kernel_config(
        num_rows, max_model_len, arch, _environ_kernel_config(), cu_count
    )

    bits_per_pass = kernel_config["bits_per_pass"]
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")

    tier_mode = kernel_config["tier_mode"]
    if tier_mode not in ("auto", "short", "mid", "long"):
        raise ValueError(
            "FLYDSL_TOPK_TIERED_OVERRIDE must be one of auto/short/mid/long, "
            f"got {tier_mode!r}"
        )

    kernel_config = _apply_deadlock_guard(
        kernel_config, num_rows, max_model_len, arch, cu_count
    )
    return kernel_config


def _apply_deadlock_guard(
    kernel_config: dict,
    num_rows: int,
    max_model_len: int,
    arch: str | None = None,
    cu_count: int | None = None,
) -> dict:
    """Clamp the tiered config so the mid/long-tier inter-workgroup barrier cannot
    deadlock.

    The barrier spins over a non-cooperative launch, so a row's participating
    workgroups are not guaranteed to be resident together, and one that arrives early
    holds its slot until the rest of its row shows up. Once the workgroups blocked
    that way outnumber the co-resident capacity nothing can drain, so the guard caps
    active workgroups per row or forces the barrier-free short tier. Reaching that
    state needs both a wide batch and long rows.
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
    if cu_count is None:
        cu_count = _multi_processor_count(arch or get_rocm_arch())
    max_coresident_workgroups = cu_count * 2
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
        # max_safe_active_workgroups < 2 -> force short tier, which the kernel
        # compiles only with the 2048-bin histogram. The sizing helpers read
        # bits_per_pass back out of this dict, so raising it here keeps them level.
        kernel_config["tier_mode"] = "short"
        kernel_config["bits_per_pass"] = 11
    return kernel_config


def flydsl_top_k_per_row_decode_workspace_size(
    num_rows: int,
    max_model_len: int,
    cu_count: int | None = None,
) -> int:
    """
    Number of int32 elements the decode TopK workspace needs for this shape.
    max_model_len = int(logits.shape[1])

    cu_count must match the device the kernel will run on, or the buffer is
    sized for the wrong bits_per_pass.
    """
    if num_rows <= 0:
        return 0

    kernel_config = _kernel_config(num_rows, max_model_len, None, cu_count)
    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    return workspace_slots


@functools.lru_cache(maxsize=1)
def _current_arch() -> str:
    """Arch of the device this process runs on.

    Cached because _build_launcher needs it on every decode step and it cannot
    change within a process. AOT compiles for other archs and passes them
    explicitly instead of going through here.
    """
    return get_rocm_arch()


@functools.lru_cache(maxsize=16384)
def _build_launcher(
    top_k: int,
    num_rows: int,
    max_model_len: int,
    arch: str,
    cu_count: int | None = None,
    ordered: bool = False,
):
    """Build (and lru-cache) the launcher + workspace metadata for this shape.

    Returns the flyc.jit launcher object, does not compile. The first
    _run_compiled() call triggers flyc.compile.

    Cached per unique (top_k, num_rows, max_model_len, arch, cu_count). The arch
    belongs in the key: it decides several config fields (short_max, and the
    gfx950-only row_proportional_parts / early_stop), and the JitFunction freezes
    its compile target on first use. Without it, an AOT process compiling several
    archs hands the first arch's launcher to the rest and their kernels never
    reach the cache. cu_count keys the same way one step finer, because one arch
    spans SKUs whose bits_per_pass differs. ``ordered`` selects a different kernel.
    """
    kernel_config = _kernel_config(num_rows, max_model_len, arch, cu_count)

    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    workspace_zero = needs_workspace_zero(
        max_model_len,
        top_k,
        kernel_config["tiered_short_max"],
        tier_mode=kernel_config["tier_mode"],
        bits_per_pass=kernel_config["bits_per_pass"],
    )
    launcher = create_topk_per_row_decode_tiered_kernel(
        top_k=top_k,
        ordered=ordered,
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
    if indices.shape[1] < k:
        raise ValueError(f"indices second dimension must be at least k={k}")
    if indices.stride() != (k, 1):
        raise ValueError(
            "indices rows must be packed k apart: the kernel writes row r at "
            f"element offset r * k, so stride() must be ({k}, 1), got "
            f"{tuple(indices.stride())}"
        )
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

    if not seqLens.is_contiguous():
        raise ValueError(
            "seqLens must be packed: the kernel reads entry i at element offset "
            f"i, so a strided view silently reads its neighbours; got "
            f"stride={tuple(seqLens.stride())}"
        )

    if workspace is not None:
        _check_cuda_tensor("workspace", workspace)
        if workspace.dtype is not torch.int32:
            raise TypeError(f"workspace must be torch.int32, got {workspace.dtype}")
        if workspace.device != logits.device:
            raise ValueError("workspace must be on the same device as logits")
        if not workspace.is_contiguous():
            raise ValueError(
                "workspace must be packed: zero_() follows the tensor's view "
                "while the kernel addresses it linearly from the base pointer, "
                f"so the two disagree; got stride={tuple(workspace.stride())}"
            )


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
    """Write each row's top-k column indices into ``indices``.

    ``ordered=False``: unordered set. ``ordered=True``: ascending, smallest-index
    tie-break on the kth value.
    """
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

    arch = _current_arch()
    launcher, workspace_slots, workspace_zero = _build_launcher(
        k,
        numRows,
        logits.shape[1],
        arch,
        decode_cu_count(logits.device, arch),
        ordered,
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

    # A caller that leaves stream unset gets the stream torch.cuda.stream() would
    # switch to anyway, so entering that context manager below buys nothing while
    # still paying a current_stream() and two set_stream() calls per launch.
    stream_is_current = stream is None
    if stream_is_current:
        stream = torch.cuda.current_stream(logits.device)

    if workspace_zero:
        if stream_is_current:
            workspace.zero_()
        else:
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
