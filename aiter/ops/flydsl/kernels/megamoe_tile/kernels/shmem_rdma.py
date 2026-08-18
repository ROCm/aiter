# SPDX-License-Identifier: MIT
"""Compile-time MORI-SHMEM put+signal bridge.

The first milestone links this kernel but does not execute it.  Raw addresses
must refer to matching symmetric allocations when the real transport is enabled.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu


def mori_flydsl_available() -> bool:
    try:
        from mori.ir import flydsl as _unused  # noqa: F401

        return True
    except Exception:
        return False


def build_mori_put_signal_module():
    try:
        from mori.ir import flydsl as mori_shmem
    except Exception as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError("mori.ir.flydsl is required for RDMA compilation") from exc

    @flyc.kernel(name="megamoe_tile_mori_put_signal", known_block_size=[64, 1, 1])
    def mori_put_signal_kernel(
        symmetric_dst_addr: fx.Int64,
        symmetric_src_addr: fx.Int64,
        nbytes: fx.Int64,
        symmetric_signal_addr: fx.Int64,
        generation: fx.Int64,
        remote_pe: fx.Int32,
        qp_id: fx.Int32,
    ):
        # Exactly one wave participates; all lanes must call the warp-scope API.
        _ = gpu.thread_id("x")
        mori_shmem.putmem_nbi_signal_warp(
            symmetric_dst_addr,
            symmetric_src_addr,
            nbytes,
            symmetric_signal_addr,
            generation,
            fx.Int32(mori_shmem.SIGNAL_SET),
            remote_pe,
            qp_id,
        )

    @flyc.jit
    def launch_mori_put_signal(
        symmetric_dst_addr: fx.Int64,
        symmetric_src_addr: fx.Int64,
        nbytes: fx.Int64,
        symmetric_signal_addr: fx.Int64,
        generation: fx.Int64,
        remote_pe: fx.Int32,
        qp_id: fx.Int32,
        stream: fx.Stream,
    ):
        launch = mori_put_signal_kernel(
            symmetric_dst_addr,
            symmetric_src_addr,
            nbytes,
            symmetric_signal_addr,
            generation,
            remote_pe,
            qp_id,
        )
        launch.launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch_mori_put_signal


def build_mori_quiet_module():
    """Drain one ``(PE, QP)`` at ring wrap or epoch retirement."""

    try:
        from mori.ir import flydsl as mori_shmem
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mori.ir.flydsl is required for RDMA compilation") from exc

    @flyc.kernel(name="megamoe_tile_mori_quiet", known_block_size=[1, 1, 1])
    def mori_quiet_kernel(remote_pe: fx.Int32, qp_id: fx.Int32):
        mori_shmem.quiet_thread_pe_qp(remote_pe, qp_id)

    @flyc.jit
    def launch_mori_quiet(remote_pe: fx.Int32, qp_id: fx.Int32, stream: fx.Stream):
        launch = mori_quiet_kernel(remote_pe, qp_id)
        launch.launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    return launch_mori_quiet


def build_mori_eos_module():
    """Publish a non-zero 8-byte EOS WQE; zero-byte put+signal is invalid."""

    try:
        from mori.ir import flydsl as mori_shmem
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mori.ir.flydsl is required for RDMA compilation") from exc

    @flyc.kernel(name="megamoe_tile_mori_eos", known_block_size=[1, 1, 1])
    def mori_eos_kernel(
        symmetric_eos_addr: fx.Int64,
        epoch_and_count: fx.Int64,
        remote_pe: fx.Int32,
        qp_id: fx.Int32,
    ):
        mori_shmem.uint64_p(symmetric_eos_addr, epoch_and_count, remote_pe, qp_id)

    @flyc.jit
    def launch_mori_eos(
        symmetric_eos_addr: fx.Int64,
        epoch_and_count: fx.Int64,
        remote_pe: fx.Int32,
        qp_id: fx.Int32,
        stream: fx.Stream,
    ):
        launch = mori_eos_kernel(symmetric_eos_addr, epoch_and_count, remote_pe, qp_id)
        launch.launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    return launch_mori_eos
