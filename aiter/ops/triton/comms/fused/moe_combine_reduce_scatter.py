# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx950.comms.fused.moe_combine_reduce_scatter import (
    _moe_combine_reduce_scatter_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "MOE_COMBINE_REDUCE_SCATTER"
_WORLD = 4
# The tested (hidden, topk) pairs. Larger rows or a larger top-k spill registers.
_SUPPORTED_SHAPES = ((6144, 4), (3072, 4), (5120, 2))
_ERROR_BYTES = 256  # the error record [flag, rank, colour, site], then padding


def _get_config(hidden: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, hidden, prefix="N_LEQ_")


class MoeCombineReduceScatter:
    """
    The IPC buffers and launch colours of ``fused_moe_combine_reduce_scatter`` for
    one group of 4 ranks on gfx950.

    Each rank has one uncached IPC buffer. It holds an error record, the row flags,
    the done words and an inbox of 3 x ceil(max_tokens / 4) rows of hidden bf16
    values. The constructor, ``reset`` and ``close`` are collective: all ranks of
    the group must call them.

    Args:
        group: The CPU (gloo) process group of the 4 ranks.
        max_tokens (int): The largest number of tokens of a call.
        hidden (int): The number of bf16 values in a row.
        topk (int): The number of experts of a token.
        timeout_s (float): The time limit of each wait in the kernel.
    """

    def __init__(self, group, max_tokens, hidden, topk, timeout_s=60.0):
        from aiter.dist.device_communicators.custom_all_reduce import IPCBufferPool

        assert get_arch() == "gfx950", "needs gfx950"
        assert dist.get_world_size(group) == _WORLD, "needs a group of 4 ranks"
        assert (
            hidden,
            topk,
        ) in _SUPPORTED_SHAPES, f"(hidden, topk) must be in {_SUPPORTED_SHAPES}"
        assert max_tokens > 0
        self.timeout_us = max(1, int(timeout_s * 1e6))
        assert self.timeout_us < 2**31, "timeout_s is too large"
        self.group = group
        self.rank = dist.get_rank(group)
        self.hidden, self.topk, self.max_tokens = hidden, topk, max_tokens
        self.config = _get_config(hidden)
        blocks = self.config["BLOCKS"]
        self.rows_max = triton.cdiv(max_tokens, _WORLD)
        flags = _ERROR_BYTES
        done = flags + (_WORLD - 1) * self.rows_max * 4
        inbox = triton.cdiv(done + _WORLD * blocks * 4, 1 << 16) << 16
        self.offsets = {"flags": flags, "done": done, "inbox": inbox}
        size = inbox + (_WORLD - 1) * self.rows_max * hidden * 2
        self.nbytes = triton.cdiv(size, 2 << 20) * (2 << 20)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._pool = IPCBufferPool(self.device, group)
        self._pool.create("inbox", self.nbytes, uncached=True)  # filled with 0
        addrs = self._pool.open_peer_buffers("inbox")
        # The buffers of ranks rank, rank + 1, rank + 2 and rank + 3.
        self.peers = tuple(addrs[(self.rank + d) % _WORLD] for d in range(_WORLD))
        self.colors = torch.zeros(blocks, dtype=torch.int32, device=self.device)

    def error(self):
        """
        Returns the error record of this rank: [0, 0, 0, 0], or [1, rank, colour,
        site] after a wait timed out (site 1: done words, site 2: row flags).
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
        """Clears the error record, the flags, the done words and the colours, for
        example after a wait timed out. All ranks must call it with no launch in
        flight."""
        from aiter.dist.device_communicators.custom_all_reduce import (
            _loaded_hip_runtime,
        )

        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        err = _loaded_hip_runtime().hipMemset(
            ctypes.c_void_p(self.peers[0]), 0, ctypes.c_size_t(self.offsets["inbox"])
        )
        assert err == 0, f"hipMemset failed: {err}"
        self.colors.zero_()
        torch.cuda.synchronize(self.device)
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


def fused_moe_combine_reduce_scatter(
    inter: torch.Tensor,
    reverse_sorted: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_tokens: int,
    comm: MoeCombineReduceScatter,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns the rows of this rank of reduce-scatter(combine(inter)) over the 4 ranks
    of ``comm``, with one kernel launch. The combined [num_tokens, hidden] tensor is
    not written to memory.

    combine is ``mxfp4_moe_scatter_reduce``. For each token t and k in order:
    p = reverse_sorted[t * topk + k] and acc = fma(inter[p], sorted_weights[p], acc)
    in fp32, from acc = 0. A p out of [0, max_sorted) is skipped. acc is rounded
    once to bf16. The reduce-scatter adds the 4 ranks in rank order in fp32 and
    rounds once to bf16. Rank r gets tokens r * R to r * R + R - 1, with
    R = ceil(num_tokens / 4). Rows past num_tokens are 0.

    All ranks must make the same calls in the same order. HIP graphs are supported.

    Args:
        inter (torch.Tensor): Contiguous [max_sorted, hidden] bf16, one unweighted
            row for each sorted position, for example
            ``scatter_intermediate["inter"]`` of the FlyDSL A4W4 stage 2.
        reverse_sorted (torch.Tensor): Contiguous int32, the sorted position of
            each (token, k), num_tokens * topk values or more.
        sorted_weights (torch.Tensor): Contiguous fp32, the weight of each sorted
            position.
        num_tokens (int): The number of tokens, the same on all ranks. 1 to
            comm.max_tokens.
        comm (MoeCombineReduceScatter): The buffers of the group.
        out (torch.Tensor, optional): Contiguous [R, hidden] bf16 output.

    Returns:
        torch.Tensor: The [R, hidden] bf16 rows of this rank.

    Limits: gfx950, 4 ranks, bf16 and (hidden, topk) in (6144, 4), (3072, 4) or
    (5120, 2). If a wait times out, the output is wrong and ``comm.error()`` is not
    0. Then call ``comm.reset()`` on all ranks.
    """
    hidden, topk = comm.hidden, comm.topk
    assert inter.dtype == torch.bfloat16 and inter.is_contiguous()
    assert inter.dim() == 2 and inter.shape[1] == hidden
    assert reverse_sorted.dtype == torch.int32 and reverse_sorted.is_contiguous()
    assert sorted_weights.dtype == torch.float32 and sorted_weights.is_contiguous()
    assert 0 < num_tokens <= comm.max_tokens
    assert reverse_sorted.numel() >= num_tokens * topk
    rows = triton.cdiv(num_tokens, _WORLD)
    if out is None:
        out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=inter.device)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert out.shape == (rows, hidden)
    max_sorted = min(inter.shape[0], sorted_weights.numel())

    _LOGGER.info(
        "MOE_COMBINE_REDUCE_SCATTER [gluon]: inter=%s num_tokens=%d topk=%d",
        tuple(inter.shape),
        num_tokens,
        topk,
    )

    config = comm.config
    _moe_combine_reduce_scatter_kernel[(config["BLOCKS"],)](
        inter.view(torch.int32),
        reverse_sorted,
        sorted_weights,
        out.view(torch.int32),
        comm.colors,
        comm.peers,
        comm.timeout_us,
        num_tokens,
        rows,
        max_sorted,
        comm.rank,
        HIDDEN=hidden,
        TOPK=topk,
        BLOCKS=config["BLOCKS"],
        RMAX=comm.rows_max,
        FLAG0=comm.offsets["flags"],
        DONE0=comm.offsets["done"],
        INBOX0=comm.offsets["inbox"],
        LAG=config["LAG"],
        num_warps=config["num_warps"],
    )
    return out
