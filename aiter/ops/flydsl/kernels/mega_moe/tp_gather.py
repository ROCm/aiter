# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""In-kernel P2P activation all-gather for tensor-parallel MoE.

Replaces ``dist.all_gather_into_tensor`` with plain stores into peer memory from
inside a kernel. On one node the 8 GPUs are XGMI-connected and peer memory is
directly addressable, so the transfer costs no separate launch and no collective
rendezvous -- the ~17us fixed cost per collective measured in phase 1 goes away.

Result is bit-identical to the NCCL all-gather: destination row is
``tp_rank * m_local + local_row``, i.e. the same rank-major layout.
"""

import torch

import mori.shmem as ms

_SUPPORTED_TP = (4, 8)
_RESET_COUNTERS = 1  # push_done


def _shmem_looks_initialised():
    """Best-effort check that Mori SHMEM is up on the current device.

    Reads Mori's own Python-level bookkeeping: ``_ensure_shmem_module()`` is the
    common entry point of all three init paths and records the HIP device id in
    ``_shmem_module_loaded_gpus``; ``shmem_finalize()`` removes it.

    This exists because calling ``ms.shmem_npes()`` on an uninitialised Mori
    SEGFAULTs the process -- it dereferences an unset GpuStates in C++, so there
    is nothing for Python to catch. Measured on this box: exit code 139, no
    output. A best-effort guard that turns that into a readable error is worth
    depending on two private names for.

    Returns None if Mori's internals no longer look like this, in which case the
    caller should skip the check rather than fail: this is a foot-gun guard, not
    a correctness mechanism.
    """
    try:
        import mori.shmem.api as _api

        loaded = getattr(_api, "_shmem_module_loaded_gpus", None)
        current = getattr(_api, "_current_hip_device", None)
        if loaded is None or current is None:
            return None
        return current() in loaded
    except Exception:  # noqa: BLE001 - never let the guard itself break construction
        return None


def _p2p_table(t, rank, npes, device):
    """i64[npes] table of intra-node P2P pointers to ``t`` on every peer."""
    table = torch.zeros(npes, dtype=torch.int64, device=device)
    for pe in range(npes):
        table[pe] = ms.shmem_ptr_p2p(t.data_ptr(), rank, pe)
    return table


class TPActivationGather:
    """Owns the symmetric receive buffers and runs the push kernel.

    Preconditions, checked in ``__init__``:
      * ``mori.shmem.shmem_torch_process_group_init`` has already been called
        by the caller, over a communicator whose size equals ``tp_size``
      * every rank constructs this with identical arguments, in the same order
        (``mori_shmem_create_tensor`` is collective)

    Buffers are sized by ``max_tok_per_rank`` once at construction; Mori's
    symmetric allocation cannot be resized per call.
    """

    def __init__(
        self,
        *,
        model_dim,
        tp_size,
        tp_rank,
        max_tok_per_rank,
        device=None,
        num_waves=4,
        producer_blocks=32,
        double_buffer=True,
    ):
        if int(tp_size) not in _SUPPORTED_TP:
            raise ValueError(
                f"tp_size={tp_size} unsupported; expected one of {_SUPPORTED_TP}"
            )
        if not (0 <= int(tp_rank) < int(tp_size)):
            raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
        if int(max_tok_per_rank) <= 0:
            raise ValueError("max_tok_per_rank must be positive")
        if int(model_dim) % 512 != 0:
            # copy_row moves dwordx4 chunks and the scale row is model_dim/32
            # bytes; both need 16-byte granularity, which model_dim % 512 == 0
            # guarantees for the activation row (model_dim bytes) and the scale
            # row (model_dim/32 bytes) simultaneously.
            raise ValueError(f"model_dim={model_dim} must be a multiple of 512")
        if int(producer_blocks) % int(tp_size) != 0:
            raise ValueError(
                f"producer_blocks={producer_blocks} must be divisible by tp_size={tp_size}"
            )

        initialised = _shmem_looks_initialised()
        if initialised is False:
            raise RuntimeError(
                "Mori SHMEM is not initialised; call "
                "mori.shmem.shmem_torch_process_group_init(<pg name>) before "
                "constructing TPActivationGather. (Constructing anyway would "
                "SEGFAULT inside Mori rather than raise.)"
            )
        # initialised is None -> Mori's internals changed shape; skip the guard
        # rather than block a valid caller.
        if initialised:
            shmem_npes = int(ms.shmem_npes())
            if shmem_npes != int(tp_size):
                raise ValueError(
                    f"Mori SHMEM world size {shmem_npes} != tp_size {tp_size}; the "
                    "shmem communicator must be the TP group"
                )

        self.model_dim = int(model_dim)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.mtpr = int(max_tok_per_rank)
        self.num_waves = int(num_waves)
        self.producer_blocks = int(producer_blocks)
        self.slots = 2 if double_buffer else 1
        dev = device or torch.device("cuda", torch.cuda.current_device())
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev

        self.scale_dim = self.model_dim // 32
        # +1 row: a zeroed PAD row the fused GEMM1 will clamp out-of-range token
        # ids to. Unused by this class, allocated now so the layout is stable.
        self.rows = self.tp_size * self.mtpr + 1
        self.row_bytes = self.model_dim
        self.scale_bytes = self.scale_dim

        # Slab byte counts are passed to the kernel as i32, and buffer resources
        # carry a 32-bit offset anyway (mega_moe_stage1.py:29).
        if self.rows * self.row_bytes >= (1 << 31):
            raise ValueError(
                f"symmetric slab {self.rows * self.row_bytes} bytes exceeds the i32 "
                f"offset ABI; lower max_tok_per_rank={self.mtpr}"
            )

        self.rx_x = ms.mori_shmem_create_tensor(
            (self.slots, self.rows, self.row_bytes), torch.uint8
        )
        self.rx_scale = ms.mori_shmem_create_tensor(
            (self.slots, self.rows, self.scale_bytes), torch.uint8
        )
        self.payload_ready = ms.mori_shmem_create_tensor((2,), torch.int32)
        self.launch_ready = ms.mori_shmem_create_tensor((self.tp_size,), torch.int32)
        for t in (self.rx_x, self.rx_scale, self.payload_ready, self.launch_ready):
            t.zero_()
        ms.shmem_barrier_all()

        self.p2p_rx_x = _p2p_table(self.rx_x, self.tp_rank, self.tp_size, dev)
        self.p2p_rx_scale = _p2p_table(self.rx_scale, self.tp_rank, self.tp_size, dev)
        self.p2p_payload_ready = _p2p_table(
            self.payload_ready, self.tp_rank, self.tp_size, dev
        )
        self.p2p_launch_ready = _p2p_table(
            self.launch_ready, self.tp_rank, self.tp_size, dev
        )

        # Local (non-symmetric) per-launch state.
        self.epoch_parity = torch.zeros(1, dtype=torch.int32, device=dev)
        self.epoch_expected = torch.zeros(2, dtype=torch.int32, device=dev)
        self.epoch_gate = torch.zeros(10, dtype=torch.int32, device=dev)
        self.entry_count = torch.zeros(10, dtype=torch.int64, device=dev)
        # One 64-byte cache line per counter, like MegaMoE's work_head.
        self.reset_counters = torch.zeros(
            _RESET_COUNTERS * 16, dtype=torch.int32, device=dev
        )

    def slab_bytes(self, kind):
        """Bytes per parity slab, for the kernel's parity offset arithmetic."""
        if kind == "x":
            return self.rows * self.row_bytes
        if kind == "scale":
            return self.rows * self.scale_bytes
        raise ValueError(kind)

    def _validate(self, x_q, x_scale):
        m_local = int(x_q.shape[0])
        if m_local <= 0:
            raise ValueError("m_local must be positive")
        if m_local > self.mtpr:
            raise ValueError(f"m_local={m_local} exceeds max_tok_per_rank={self.mtpr}")
        if x_q.dtype != torch.float8_e4m3fn or not x_q.is_contiguous():
            raise ValueError("x_q must be contiguous float8_e4m3fn")
        if tuple(x_q.shape) != (m_local, self.model_dim):
            raise ValueError(
                f"x_q must be [{m_local}, {self.model_dim}], got {tuple(x_q.shape)}"
            )
        if x_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(
                f"x_scale must be uint8 or float8_e8m0fnu, got {x_scale.dtype}"
            )
        if not x_scale.is_contiguous():
            raise ValueError("x_scale must be contiguous")
        if tuple(x_scale.shape) != (m_local, self.scale_dim):
            raise ValueError(
                f"x_scale must be [{m_local}, {self.scale_dim}], got {tuple(x_scale.shape)}"
            )
        for name, t in (("x_q", x_q), ("x_scale", x_scale)):
            if t.device != self.device:
                raise ValueError(f"{name} is on {t.device}, expected {self.device}")
        return m_local

    def current_parity(self):
        """Parity the NEXT gather() will write. The kernel flips before pushing."""
        return int(self.epoch_parity[0].item()) ^ 1 if self.slots == 2 else 0

    def views(self, m_local, parity):
        """Views of the gathered result for a given round. Valid until the next gather."""
        n = self.tp_size * int(m_local)
        x = self.rx_x[parity, :n].view(torch.float8_e4m3fn)
        scale = self.rx_scale[parity, :n]
        return x, scale

    def gather(self, x_q, x_scale):
        raise NotImplementedError("push kernel lands in task 3")
