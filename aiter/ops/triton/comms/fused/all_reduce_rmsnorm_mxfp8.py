# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.comms.fused.all_reduce_rmsnorm_mxfp8 import (
    _all_reduce_rmsnorm_mxfp8_kernel,
)
from aiter.ops.triton.comms.ipc_buffers import ERROR_BYTES, IpcCommBuffers
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "ALL_REDUCE_RMSNORM_MXFP8"
_MAX_HIDDEN = 8192  # 2 vectors of 8 bf16 values for each of 512 threads


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


class AllReduceRmsnormMxfp8(IpcCommBuffers):
    """
    The IPC buffers of ``fused_all_reduce_rmsnorm_mxfp8`` for one group of 2, 4 or 8
    ranks on gfx950. After the error record, each buffer holds a reduce-scatter inbox
    and an all-gather inbox of 2 x world x ceil(max_tokens / world) rows. The
    constructor, ``reset`` and ``close`` are collective. Error sites: 1
    (reduce-scatter), 2 (all-gather).

    Args:
        group: The CPU (gloo) process group of the ranks.
        max_tokens (int): The largest number of tokens of a call.
        hidden (int): The number of bf16 values in a row.
        timeout_s (float): The time limit of each wait in the kernel.
        config (dict, optional): Values that replace the values of the tuned config
            (BLOCKS, num_warps), for example in a benchmark sweep. An unknown key is
            an error. The config is fixed for the comm: all its launches must use
            the same CTAs.
    """

    def __init__(self, group, max_tokens, hidden, timeout_s=60.0, config=None):
        super().__init__(group, timeout_s)
        assert (
            hidden % 64 == 0 and hidden <= _MAX_HIDDEN
        ), f"hidden must be a multiple of 64, up to {_MAX_HIDDEN}"
        self.hidden, self.max_tokens = hidden, max_tokens
        self.config = _get_config(hidden)
        if config is not None:
            unknown = sorted(set(config) - set(self.config))
            assert not unknown, f"unknown config keys: {unknown}"
            self.config.update(config)
        assert hidden // 16 <= 64 * self.config["num_warps"], "too few warps for hidden"
        # All CTAs must run at the same time: one CTA a CU.
        self.blocks = min(self.config["BLOCKS"], self.num_cus)
        self.gmax = triton.cdiv(max_tokens, self.world)
        assert (
            0 < max_tokens and self.world * self.gmax <= self.blocks
        ), f"max_tokens must be 1 to {self.blocks // self.world * self.world}"
        self.push_units = _push_units(max_tokens, self.world, self.blocks)
        inbox = 2 * self.world * self.gmax * hidden * 2
        self.offsets = {"rs": ERROR_BYTES, "ag": ERROR_BYTES + inbox}
        self._open(ERROR_BYTES + 2 * inbox, self.blocks)


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
    residual over the ranks of ``comm``, in one launch. All ranks get the same bits.

    The all-reduce adds the ranks in rank order in fp32 and rounds once to bf16. Then
    s = h + residual, residual_out = bf16(s) and y = bf16(s * rstd * w'), with w' = 1
    + weight (gemma) or weight. The sum of squares has a fixed order. The E8M0 scale
    of a group of 32 is ``mx_e8m0_scale(amax, 8)``, as in the other AITER MXFP8 ops.

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

    Limits: gfx950, 2, 4 or 8 ranks, hidden a multiple of 64 up to 8192, tokens 1 to
    comm.max_tokens. If a wait times out, the outputs are wrong and ``comm.error()``
    is not 0. Then call ``comm.reset()`` on all ranks.
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
