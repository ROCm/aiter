# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch


def test_mori_self_put_signal_quiet_and_eos():
    """Compile and execute the real MORI FlyDSL ABI on a one-PE world."""

    from mori import shmem as ms

    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_mori_eos_module,
        build_mori_put_signal_module,
        build_mori_quiet_module,
    )

    torch.cuda.set_device(0)
    uid = ms.shmem_get_unique_id()
    ms.shmem_init_attr(ms.MORI_SHMEM_INIT_WITH_UNIQUEID, 0, 1, uid)
    src = dst = signal = None
    try:
        src = ms.mori_shmem_create_tensor((4096,), torch.uint8)
        dst = ms.mori_shmem_create_tensor((4096,), torch.uint8)
        signal = ms.mori_shmem_create_tensor((1,), torch.int64)
        src.copy_(torch.arange(256, dtype=torch.uint8, device="cuda").repeat(16))
        dst.zero_()
        signal.zero_()

        put = build_mori_put_signal_module()
        quiet = build_mori_quiet_module()
        eos = build_mori_eos_module()
        stream = torch.cuda.current_stream()
        put(
            dst.data_ptr(),
            src.data_ptr(),
            src.numel(),
            signal.data_ptr(),
            9,
            0,
            0,
            stream=stream,
        )
        quiet(0, 0, stream=stream)
        torch.cuda.synchronize()
        assert torch.equal(src, dst)
        assert signal.item() == 9

        # EOS is an actual 8-byte WQE, not a zero-byte put+signal.
        eos(signal.data_ptr(), 11, 0, 0, stream=stream)
        quiet(0, 0, stream=stream)
        torch.cuda.synchronize()
        assert signal.item() == 11
    finally:
        if signal is not None:
            ms.mori_shmem_free_tensor(signal)
        if dst is not None:
            ms.mori_shmem_free_tensor(dst)
        if src is not None:
            ms.mori_shmem_free_tensor(src)
        ms.shmem_finalize()
