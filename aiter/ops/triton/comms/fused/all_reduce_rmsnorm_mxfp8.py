# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx950.comms.fused.all_reduce_rmsnorm_mxfp8 import (
    _all_reduce_rmsnorm_mxfp8_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "ALL_REDUCE_RMSNORM_MXFP8"
_SUPPORTED_WORLDS = (2, 4, 8)
_MAX_HIDDEN = 8192  # 2 vectors of 8 bf16 values for each of 512 threads
_ERROR_BYTES = 256  # the error record [flag, rank, colour, site], then padding


def _get_config(hidden: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, hidden)


def _push_units(rows, world, blocks):
    """The most push units of a CTA, for 1 to rows tokens."""
    most = 1
    for g in range(1, triton.cdiv(rows, world) + 1):
        units = (world - 1) * g * 2
        most = max(most, triton.cdiv(units, min(blocks - g, units)))
    return most


class AllReduceRmsnormMxfp8:
    """
    The IPC buffers and launch colours of ``fused_all_reduce_rmsnorm_mxfp8`` for one
    group of 2, 4 or 8 ranks on gfx950.

    Each rank has one uncached IPC buffer. It holds an error record, then the
    reduce-scatter inbox and the all-gather inbox. Each inbox has 2 halves (for odd
    and even launches) of world x ceil(max_tokens / world) rows of hidden bf16
    values. The constructor, ``reset`` and ``close`` are collective: all ranks of
    the group must call them.

    Args:
        group: The CPU (gloo) process group of the ranks.
        max_tokens (int): The largest number of tokens of a call.
        hidden (int): The number of bf16 values in a row.
        timeout_s (float): The time limit of each wait in the kernel.
    """

    def __init__(self, group, max_tokens, hidden, timeout_s=60.0):
        from aiter.dist.device_communicators.custom_all_reduce import IPCBufferPool

        assert get_arch() == "gfx950", "needs gfx950"
        self.world = dist.get_world_size(group)
        assert (
            self.world in _SUPPORTED_WORLDS
        ), f"the group size must be one of {_SUPPORTED_WORLDS}"
        assert (
            hidden % 64 == 0 and hidden <= _MAX_HIDDEN
        ), f"hidden must be a multiple of 64, up to {_MAX_HIDDEN}"
        self.timeout_us = max(1, int(timeout_s * 1e6))
        assert self.timeout_us < 2**31, "timeout_s is too large"
        self.group = group
        self.rank = dist.get_rank(group)
        self.hidden, self.max_tokens = hidden, max_tokens
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.config = _get_config(hidden)
        cus = torch.cuda.get_device_properties(self.device).multi_processor_count
        # All CTAs must run at the same time: one CTA a CU.
        self.blocks = min(self.config["BLOCKS"], cus)
        self.gmax = triton.cdiv(max_tokens, self.world)
        assert (
            0 < max_tokens and self.world * self.gmax <= self.blocks
        ), f"max_tokens must be 1 to {self.blocks // self.world * self.world}"
        self.push_units = _push_units(max_tokens, self.world, self.blocks)
        inbox = 2 * self.world * self.gmax * hidden * 2
        self.offsets = {"rs": _ERROR_BYTES, "ag": _ERROR_BYTES + inbox}
        self.nbytes = triton.cdiv(_ERROR_BYTES + 2 * inbox, 2 << 20) * (2 << 20)
        self._pool = IPCBufferPool(self.device, group)
        self._pool.create("inbox", self.nbytes, uncached=True)
        addrs = self._pool.open_peer_buffers("inbox")
        # The buffers of ranks rank, rank + 1, ..., rank + world - 1.
        self.peers = tuple(
            addrs[(self.rank + d) % self.world] for d in range(self.world)
        )
        self.colors = torch.empty(self.blocks, dtype=torch.int32, device=self.device)
        self._clear()
        dist.barrier(group=self.group)

    def _clear(self):
        """Fills the inboxes with 0xFFFF (empty) and clears the error record and the
        colours."""
        from aiter.dist.device_communicators.custom_all_reduce import (
            _loaded_hip_runtime,
        )

        rt = _loaded_hip_runtime()
        torch.cuda.synchronize(self.device)
        own = ctypes.c_void_p(self.peers[0])
        err = rt.hipMemset(own, 0xFF, ctypes.c_size_t(self.nbytes))
        err = err or rt.hipMemset(own, 0, ctypes.c_size_t(_ERROR_BYTES))
        assert err == 0, f"hipMemset failed: {err}"
        self.colors.zero_()
        torch.cuda.synchronize(self.device)

    def error(self):
        """
        Returns the error record of this rank: [0, 0, 0, 0], or [1, rank, colour,
        site] after a wait timed out (site 1: reduce-scatter, site 2: all-gather).
        Synchronizes the device.
        """
        from aiter.dist.device_communicators.custom_all_reduce import (
            _loaded_hip_runtime,
        )

        record = (ctypes.c_int32 * 4)()
        torch.cuda.synchronize(self.device)
        err = _loaded_hip_runtime().hipMemcpy(
            record, ctypes.c_void_p(self.peers[0]), ctypes.c_size_t(16), 2
        )
        assert err == 0, f"hipMemcpy failed: {err}"
        return list(record)

    def reset(self):
        """Empties the inboxes and clears the error record and the colours, for
        example after a wait timed out. All ranks must call it with no launch in
        flight."""
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._clear()
        dist.barrier(group=self.group)

    def close(self):
        """Unmaps the peer buffers and frees the buffer of this rank. All ranks must
        call it."""
        if self._pool is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._pool.close_peer_buffers()
        dist.barrier(group=self.group)  # no peer uses the buffer of this rank now
        self._pool.close()
        self._pool = None


def fused_all_reduce_rmsnorm_mxfp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm: AllReduceRmsnormMxfp8,
    gemma: bool = False,
    out: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the MXFP8 quant of RMSNorm(all-reduce(x) + residual) and the new
    residual, over the ranks of ``comm``, with one kernel launch. All ranks get the
    same bits.

    The all-reduce adds the ranks in rank order in fp32 and rounds once to bf16, as
    the AITER custom all-reduce does. Then, in fp32: s = h + residual, residual_out =
    bf16(s), y = bf16(s * rstd * w'), with w' = 1 + weight (gemma) or weight. Each
    group of 32 values gets the E8M0 scale clamp(ceil(log2(amax / 448)) + 127, 0,
    254), and y / 2^(scale - 127) is rounded to e4m3. rstd = 1 / sqrt(mean(s^2) +
    eps): the sum of squares has a fixed order, see the kernel.

    All ranks must make the same calls in the same order. HIP graphs are supported.

    Args:
        x (torch.Tensor): Contiguous [tokens, hidden] bf16, the partial sum of this
            rank.
        residual (torch.Tensor): Contiguous [tokens, hidden] bf16, the same on all
            ranks.
        weight (torch.Tensor): Contiguous [hidden] bf16 norm weight.
        eps (float): The epsilon of the norm.
        comm (AllReduceRmsnormMxfp8): The buffers of the group.
        gemma (bool): Use 1 + weight (Gemma norm) in place of weight.
        out (tuple, optional): (x_q, x_scale, residual_out) to write into.

    Returns:
        tuple: x_q [tokens, hidden] float8_e4m3fn, x_scale [tokens, hidden / 32]
        uint8 (E8M0) and residual_out [tokens, hidden] bf16, all rows on each rank.

    Limits: gfx950, 2, 4 or 8 ranks, bf16, hidden a multiple of 64 up to 8192,
    tokens 1 to comm.max_tokens. All CTAs of the launch must run at the same time.
    If a wait times out, the outputs are wrong and ``comm.error()`` is not 0. Then
    call ``comm.reset()`` on all ranks.
    """
    tokens, hidden = x.shape
    assert hidden == comm.hidden and 0 < tokens <= comm.max_tokens
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert residual.shape == x.shape and weight.shape == (hidden,)
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    if out is None:
        x_q = torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device=x.device)
        x_s = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=x.device)
        res_out = torch.empty_like(x)
    else:
        x_q, x_s, res_out = out
        assert x_q.shape == x.shape and x_q.dtype == torch.float8_e4m3fn
        assert x_s.shape == (tokens, hidden // 32) and x_s.dtype == torch.uint8
        assert res_out.shape == x.shape and res_out.dtype == torch.bfloat16
        assert x_q.is_contiguous() and x_s.is_contiguous() and res_out.is_contiguous()

    _LOGGER.info(
        "ALL_REDUCE_RMSNORM_MXFP8 [gluon]: x=%s world=%d gemma=%s",
        tuple(x.shape),
        comm.world,
        gemma,
    )

    _all_reduce_rmsnorm_mxfp8_kernel[(comm.blocks,)](
        x.view(torch.int32),
        residual.view(torch.int32),
        weight.view(torch.int32),
        x_q.view(torch.int32),
        x_s,
        res_out.view(torch.int32),
        comm.colors,
        comm.peers,
        tokens,
        comm.rank,
        comm.timeout_us,
        eps,
        HIDDEN=hidden,
        WORLD=comm.world,
        GEMMA=gemma,
        GMAX=comm.gmax,
        RS0=comm.offsets["rs"],
        AG0=comm.offsets["ag"],
        BLOCKS=comm.blocks,
        PMAX=comm.push_units,
        num_warps=comm.config["num_warps"],
        enable_fp_fusion=False,  # the sum of squares adds rounded products
    )
    return x_q, x_s, res_out
