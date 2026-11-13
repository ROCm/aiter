# SPDX-License-Identifier: MIT
"""Device addresses for C1's separate local and rail windows.

The table's stable rows are runtime values. The region index is selected from
the buffer plan while compiling a kernel; parity remains a device value so
graph replay follows the current generation. Persistent regions have stride
zero. The caller keeps the owning CcoBufferSet alive until all uses finish.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco


def _field(table, region, field):
    # Bindings are immutable and initialized before any consumer launches;
    # their reads need no inter-GPU readiness polling or acquire operation.
    pointer_type = fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8)
    pointer = fx.inttoptr(pointer_type, table + fx.Int64(region * 32 + field * 8))
    return fx.Int64(fx.ptr_load(pointer))


@flyc.jit
def region_local_address(table, region, parity):
    return _field(table, region, 0) + parity * _field(table, region, 3)


@flyc.jit
def region_window_address(table, region, parity):
    handle = _field(table, region, 1)
    offset = _field(table, region, 2) + parity * _field(table, region, 3)
    return handle, offset


@flyc.jit
def region_lsa_address(table, region, parity, peer):
    handle, offset = region_window_address(table, region, parity)
    return fx.Int64(cco.Window(handle).lsa_ptr(peer, offset))


@flyc.jit
def region_local_dev_comm(table, region_count):
    """The optional trailing row owns SDMA on the local window communicator."""
    return _field(table, region_count, 0)
