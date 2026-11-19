# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.comms.fused.moe_combine_reduce_scatter import (
    _moe_combine_reduce_scatter_kernel,
)
from aiter.ops.triton.comms.ipc_buffers import ERROR_BYTES, IpcCommBuffers
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "MOE_COMBINE_REDUCE_SCATTER"
# The tested (hidden, topk) pairs. Larger rows or a larger top-k spill registers.
_SUPPORTED_SHAPES = ((6144, 4), (3072, 4), (5120, 2))


def _get_config(world: int, hidden: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, axes=("WORLD", "N"), WORLD=world, N=hidden)


class MoeCombineReduceScatter(IpcCommBuffers):
    """
    The IPC buffers and launch colours of ``fused_moe_combine_reduce_scatter`` for
    one group of 2, 4 or 8 ranks on gfx950.

    Each rank has one uncached IPC buffer. It holds an error record, the row flags,
    the done words and an inbox of (world - 1) x ceil(max_tokens / world) rows of
    hidden bf16 values. The constructor, ``reset`` and ``close`` are collective:
    all ranks of the group must call them. Error sites: 1 (done words), 2 (row
    flags).

    Args:
        group: The CPU (gloo) process group of the ranks.
        max_tokens (int): The largest number of tokens of a call.
        hidden (int): The number of bf16 values in a row.
        topk (int): The number of experts of a token.
        timeout_s (float): The time limit of each wait in the kernel.
        config (dict, optional): Values that replace the values of the tuned config
            (BLOCKS, PARTS, LAG, num_warps), for example in a benchmark sweep. An
            unknown key is an error. The config is fixed for the comm: all its
            launches must use the same CTAs.
    """

    def __init__(self, group, max_tokens, hidden, topk, timeout_s=60.0, config=None):
        super().__init__(group, timeout_s)
        assert (
            hidden,
            topk,
        ) in _SUPPORTED_SHAPES, f"(hidden, topk) must be in {_SUPPORTED_SHAPES}"
        assert max_tokens > 0
        self.hidden, self.topk, self.max_tokens = hidden, topk, max_tokens
        self.config = _get_config(self.world, hidden)
        if config is not None:
            unknown = sorted(set(config) - set(self.config))
            assert not unknown, f"unknown config keys: {unknown}"
            self.config.update(config)
        parts, warps = self.config["PARTS"], self.config["num_warps"]
        assert hidden // 2 % parts == 0, "PARTS must divide hidden / 2"
        assert self.config["LAG"] >= 2, "LAG must be 2 or more"
        assert warps > 0 and warps & (warps - 1) == 0, "num_warps must be a power of 2"
        # All CTAs must run at the same time: one CTA a CU, PARTS CTAs a row.
        self.blocks = min(self.config["BLOCKS"], self.num_cus) // parts * parts
        assert self.blocks > 0, "fewer CUs than PARTS"
        blocks, world = self.blocks, self.world
        # A thread polls at most 4 done words.
        assert (world - 1) * blocks <= 4 * 64 * warps, "too few warps for BLOCKS"
        self.rows_max = triton.cdiv(max_tokens, world)
        flags = ERROR_BYTES
        done = flags + (world - 1) * self.rows_max * parts * 4
        inbox = triton.cdiv(done + world * blocks * 4, 1 << 16) << 16
        self.offsets = {"flags": flags, "done": done, "inbox": inbox}
        size = inbox + (world - 1) * self.rows_max * hidden * 2
        # A clear zeroes the error record, the flags and the done words.
        self._open(size, blocks, fill=0, fill_bytes=inbox)


def fused_moe_combine_reduce_scatter(
    inter: torch.Tensor,
    reverse_sorted: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_tokens: int,
    comm: MoeCombineReduceScatter,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns the rows of this rank of reduce-scatter(combine(inter)) over the ranks
    of ``comm``, with one kernel launch. The combined [num_tokens, hidden] tensor is
    not written to memory.

    combine is ``mxfp4_moe_scatter_reduce``. For each token t and k in order:
    p = reverse_sorted[t * topk + k] and acc = fma(inter[p], sorted_weights[p], acc)
    in fp32, from acc = 0. A p out of [0, max_sorted) is skipped. acc is rounded
    once to bf16. The reduce-scatter adds the ranks in rank order in fp32 and
    rounds once to bf16. Rank r gets tokens r * R to r * R + R - 1, with
    R = ceil(num_tokens / world). Rows past num_tokens are 0.

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

    Limits: gfx950, 2, 4 or 8 ranks, bf16 and (hidden, topk) in (6144, 4),
    (3072, 4) or (5120, 2). If a wait times out, the output is wrong and ``comm.error()`` is not
    0. Then call ``comm.reset()`` on all ranks.
    """
    hidden, topk = comm.hidden, comm.topk
    assert inter.dtype == torch.bfloat16 and inter.is_contiguous()
    assert inter.dim() == 2 and inter.shape[1] == hidden
    assert reverse_sorted.dtype == torch.int32 and reverse_sorted.is_contiguous()
    assert sorted_weights.dtype == torch.float32 and sorted_weights.is_contiguous()
    # The kernel loads a weight for each (token, k), also for a skipped position.
    assert sorted_weights.numel() > 0, "sorted_weights is empty"
    assert 0 < num_tokens <= comm.max_tokens
    assert reverse_sorted.numel() >= num_tokens * topk
    rows = triton.cdiv(num_tokens, comm.world)
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
    _moe_combine_reduce_scatter_kernel[(comm.blocks,)](
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
        WORLD=comm.world,
        PARTS=config["PARTS"],
        BLOCKS=comm.blocks,
        RMAX=comm.rows_max,
        FLAG0=comm.offsets["flags"],
        DONE0=comm.offsets["done"],
        INBOX0=comm.offsets["inbox"],
        LAG=config["LAG"],
        num_warps=config["num_warps"],
    )
    return out
