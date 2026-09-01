# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Minimal repro: transpose J global→LDS staging for jdbba dDense backward.

Context (PR #5025, Commit 1 / A′):
  Replace the manual Jagged g2s ownership map in ``grad_dense_partials_kernel`` with
  FlyDSL ``make_tiled_copy`` + ``partition_S`` / ``partition_D`` + ``fx.copy``.
  The manual loop is correct; every tiled-copy variant we tried yields ~99% dDense error.

What this file isolates:
  * Global J is row-major ``(m, k)`` (shape ``(64, 128)`` for one m-tile).
  * LDS ``sJ`` is logical ``(k, m)`` with ``Swizzle<3,3,3>`` + ordered layout ``(1, 0)``
    — same as production ``grad_dense_partials_kernel``.
  * Manual map: ``lin = tid + 256*i`` → read ``(m, k)``, store ``sJ(k, m)``.
  * Tiled map: ``make_layout_tv((16,16), (1,32))`` + ``make_tile(128, 64)`` +
    transpose buffer view ``(128,64):(1,K)`` + two-step ``fx.copy`` (buffer→frag→sJ).

Both paths read back ``sJ`` with the *same* logical ``(k, m)`` loop into a plain
global ``OUT (128, 64)`` tensor for host comparison.

Approaches tried in the full kernel (all failed correctness or compile):
  * 16×16 subtiles with wrong TV val width (4 vals/thread vs 1).
  * Full-tile ``make_tiled_copy`` / ``make_tiled_copy_tv`` + fragment or direct copy.
  * ``make_tiled_copy_A`` (mirror s2r) with 16- and 32-row chunks.
  * ``UniversalCopy`` on raw global memref (legalization error; needs buffer tensor).
  * ``logical_divide`` + ``_load_scalar`` on ``J_buf`` (slice profile mismatch).

Run (inside ``flydsl_venv``, CDNA gfx942/gfx950):
    source flydsl_venv/bin/activate
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=0 \\
        python op_tests/repros/flydsl_j_g2s_transpose_repro.py
    # or:
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=0 \\
        pytest op_tests/repros/flydsl_j_g2s_transpose_repro.py -v

Expected:
    manual vs ref:  max_err = 0          PASS
    tiled  vs ref:  large finite err or NaN in sJ/OUT  FAIL (documents the issue)
    Use rel_err (vs ref RMS) when max_abs is NaN — see ``run_repro()`` diagnostics.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.derived import make_layout_tv
    from flydsl.expr.typing import T

    from aiter.ops.flydsl.kernels import buffer_ops

    _HAS_FLYDSL = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    _HAS_FLYDSL = False

torch.set_default_device("cuda")

SUPPORTED_GFX = ("gfx942", "gfx950")

# Production dDense J staging shape (D = K = 128 tile).
K = 128
DDENSE_BK = 128
DDENSE_BM = 64
DDENSE_THREADS = 256
_J_LDS_LOADS = (DDENSE_BM * DDENSE_BK) // DDENSE_THREADS  # 32
_SJ_SMEM_BYTES = DDENSE_BK * DDENSE_BM * 2  # bf16


def _make_bounded_buffer_tensor(tensor, num_records_bytes):
    """Bounded buffer descriptor (OOB load == 0), vendored from jdbba bwd helpers."""
    from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace

    from aiter.ops.flydsl.kernels.buffer_ops import _get_buffer_flags

    elem_ty = tensor.element_type
    ptr = fx.get_iter(tensor)
    layout = fx.get_layout(tensor)
    buf_ptr_ty = fx.PointerType.get(
        elem_ty=elem_ty.ir_type,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=ptr.alignment,
    )
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [
            ptr,
            fx.Int16(0).ir_value(),
            num_records_bytes.ir_value(),
            fx.Int32(_get_buffer_flags()).ir_value(),
        ],
    )
    return fx.make_view(buf_ptr, layout)


def _readback_sJ_to_out(sJ, OUT, tid):
    """Logical (k, m) readback — identical in manual and tiled kernels."""
    out_layout = fx.make_layout((DDENSE_BK, DDENSE_BM), (DDENSE_BM, 1))
    out_view = fx.make_view(fx.get_iter(OUT), out_layout)
    for i in fx.range_constexpr(_J_LDS_LOADS):
        lin = tid + fx.Int32(i * DDENSE_THREADS)
        m_local = lin // fx.Int32(DDENSE_BK)
        k_local = lin % fx.Int32(DDENSE_BK)
        val = fx.memref_load(sJ, (k_local, m_local))
        fx.memref_store(val, out_view, (k_local, m_local))


@functools.lru_cache(maxsize=1)
def _build_launchers():
    """Compile manual + tiled staging kernels once."""

    @flyc.kernel
    def stage_manual_kernel(JAGGED: fx.Tensor, OUT: fx.Tensor, M_b: fx.Int32):
        tid = fx.thread_idx.x
        j_rsrc = buffer_ops.create_buffer_resource(
            JAGGED,
            max_size=False,
            num_records_bytes=fx.Int64(M_b) * fx.Int64(K * 2),
            base_byte_offset=fx.Int64(0),
        )
        composed_J = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 3, 3)),
            fx.make_ordered_layout((DDENSE_BK, DDENSE_BM), (1, 0)),
        )
        sJ = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_J)

        mt = fx.Int32(0)
        k_off = fx.Int32(0)
        for i in fx.range_constexpr(_J_LDS_LOADS):
            lin = tid + fx.Int32(i * DDENSE_THREADS)
            m_local = lin // fx.Int32(DDENSE_BK)
            k_local = lin % fx.Int32(DDENSE_BK)
            joff = (mt * fx.Int32(DDENSE_BM) + m_local) * fx.Int32(K) + (
                k_off + k_local
            )
            jval = buffer_ops.buffer_load(j_rsrc, joff, vec_width=1, dtype=T.bf16)
            fx.memref_store(jval, sJ, (k_local, m_local))

        fx.gpu.barrier()
        _readback_sJ_to_out(sJ, OUT, tid)

    @flyc.kernel
    def stage_tiled_kernel(
        JAGGED: fx.Tensor,
        OUT: fx.Tensor,
        tiled_copy_g2s_J: fx.TiledCopy,
        M_b: fx.Int32,
    ):
        tid = fx.thread_idx.x
        J_buf = _make_bounded_buffer_tensor(
            fx.make_view(fx.get_iter(JAGGED), fx.get_layout(JAGGED)),
            fx.Int64(M_b) * fx.Int64(K * 2),
        )
        composed_J = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 3, 3)),
            fx.make_ordered_layout((DDENSE_BK, DDENSE_BM), (1, 0)),
        )
        sJ = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_J)

        buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        thr_copy_g2s_J = tiled_copy_g2s_J.get_slice(tid)
        gJ_tile = fx.make_view(
            fx.get_iter(J_buf),
            fx.make_layout((DDENSE_BK, DDENSE_BM), (1, K)),
        )
        thr_gJ = thr_copy_g2s_J.partition_S(gJ_tile)
        thr_sJ = thr_copy_g2s_J.partition_D(sJ)
        copy_frag_J = fx.make_fragment_like(thr_sJ[None, None, 0])
        fx.copy(
            buffer_copy_128b,
            thr_gJ[None, None, 0],
            copy_frag_J,
        )
        fx.copy(
            uni_copy_128b,
            copy_frag_J,
            thr_sJ[None, None, 0],
        )

        fx.gpu.barrier()
        _readback_sJ_to_out(sJ, OUT, tid)

    @flyc.jit
    def launch_manual(
        JAGGED: fx.Tensor, OUT: fx.Tensor, M_b: fx.Int32, stream=fx.Stream(None)
    ):
        stage_manual_kernel(JAGGED, OUT, M_b).launch(
            grid=(1, 1, 1),
            block=(DDENSE_THREADS, 1, 1),
            smem=_SJ_SMEM_BYTES,
            stream=stream,
        )

    @flyc.jit
    def launch_tiled(
        JAGGED: fx.Tensor, OUT: fx.Tensor, M_b: fx.Int32, stream=fx.Stream(None)
    ):
        _, j_g2s_tv = make_layout_tv(
            fx.make_layout((16, 16), (16, 1)),
            fx.make_layout((1, _J_LDS_LOADS), (1, 1)),
        )
        tiled_copy_g2s_J = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            j_g2s_tv,
            fx.make_tile(DDENSE_BK, DDENSE_BM),
        )
        stage_tiled_kernel(JAGGED, OUT, tiled_copy_g2s_J, M_b).launch(
            grid=(1, 1, 1),
            block=(DDENSE_THREADS, 1, 1),
            smem=_SJ_SMEM_BYTES,
            stream=stream,
        )

    return launch_manual, launch_tiled


def _tensor_stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Error metrics robust to NaNs in the tiled (broken) path."""
    diff = (got.float() - ref.float()).abs()
    nan_got = int(torch.isnan(got).sum().item())
    nan_diff = int(torch.isnan(diff).sum().item())
    finite_diff = diff[torch.isfinite(diff)]
    max_abs = finite_diff.max().item() if finite_diff.numel() else float("nan")
    ref_rms = ref.float().pow(2).mean().sqrt().item()
    rel = max_abs / ref_rms if ref_rms > 0 and math.isfinite(max_abs) else float("nan")
    return {
        "max_abs": max_abs,
        "rel": rel,
        "nan_got": float(nan_got),
        "nan_diff": float(nan_diff),
    }


def run_repro(*, m_rows: int = DDENSE_BM, seed: int = 0) -> dict[str, dict[str, float]]:
    """Run both paths; return error stats vs torch reference."""
    launch_manual, launch_tiled = _build_launchers()

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    jagged = torch.randn(m_rows, K, dtype=torch.bfloat16, device="cuda", generator=gen)
    ref = jagged.T.contiguous()  # logical (k, m)

    # Zero-init so uncopied OUT lanes show as wrong data, not uninitialized NaN.
    out_manual = torch.zeros(DDENSE_BK, DDENSE_BM, dtype=torch.bfloat16, device="cuda")
    out_tiled = torch.zeros(DDENSE_BK, DDENSE_BM, dtype=torch.bfloat16, device="cuda")

    m_b = fx.Int32(m_rows)
    launch_manual(jagged, out_manual, m_b)
    launch_tiled(jagged, out_tiled, m_b)
    torch.cuda.synchronize()

    return {
        "manual_vs_ref": _tensor_stats(out_manual, ref),
        "tiled_vs_ref": _tensor_stats(out_tiled, ref),
        "manual_vs_tiled": _tensor_stats(out_manual, out_tiled),
    }


def _tiled_is_broken(stats: dict[str, float]) -> bool:
    """Tiled path is wrong if NaNs appear or error is large vs reference."""
    if stats["nan_got"] > 0 or stats["nan_diff"] > 0:
        return True
    if math.isfinite(stats["max_abs"]) and stats["max_abs"] > 0.5:
        return True
    if math.isfinite(stats["rel"]) and stats["rel"] > 0.5:
        return True
    return False


def _skip_unless_flydsl_cdna():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _HAS_FLYDSL:
        pytest.skip("flydsl not installed")
    if not is_flydsl_available():
        pytest.skip("flydsl not available in this environment")
    gfx = (get_gfx() or "").lower()
    if not any(gfx.startswith(g) for g in SUPPORTED_GFX):
        pytest.skip(f"CDNA required (gfx942/gfx950), got {gfx!r}")


@pytest.mark.repro
def test_j_g2s_manual_matches_reference():
    _skip_unless_flydsl_cdna()
    errs = run_repro()
    assert errs["manual_vs_ref"]["max_abs"] == 0.0, errs
    assert errs["manual_vs_ref"]["nan_got"] == 0.0, errs


@pytest.mark.repro
def test_j_g2s_tiled_diverges_from_reference():
    """Documents the FlyDSL tiled-copy failure (not an assertion we want green)."""
    _skip_unless_flydsl_cdna()
    errs = run_repro()
    assert errs["manual_vs_ref"]["max_abs"] == 0.0, errs
    assert _tiled_is_broken(errs["tiled_vs_ref"]), errs
    assert _tiled_is_broken(errs["manual_vs_tiled"]), errs


def _print_summary(errs: dict[str, dict[str, float]]) -> None:
    def status_manual(s: dict[str, float]) -> str:
        return "PASS" if s["max_abs"] == 0.0 and s["nan_got"] == 0 else "FAIL"

    def status_tiled(s: dict[str, float]) -> str:
        return "FAIL (expected)" if _tiled_is_broken(s) else "unexpected PASS"

    def fmt(s: dict[str, float]) -> str:
        parts = [f"max_abs={s['max_abs']:.6g}", f"rel={s['rel']:.4g}"]
        if s["nan_got"]:
            parts.append(f"nan_got={int(s['nan_got'])}")
        return "  ".join(parts)

    print("flydsl_j_g2s_transpose_repro")
    for label, key, st in (
        ("manual vs ref", "manual_vs_ref", status_manual),
        ("tiled  vs ref", "tiled_vs_ref", status_tiled),
        ("manual vs tiled", "manual_vs_tiled", status_tiled),
    ):
        s = errs[key]
        print(f"  {label:16s}  {fmt(s)}  {st(s)}")


if __name__ == "__main__":
    if not torch.cuda.is_available() or not _HAS_FLYDSL:
        print("SKIP: need CUDA + flydsl")
        sys.exit(0)
    gfx = (get_gfx() or "").lower()
    if not any(gfx.startswith(g) for g in SUPPORTED_GFX):
        print(f"SKIP: CDNA required, got {gfx!r}")
        sys.exit(0)
    _print_summary(run_repro())
