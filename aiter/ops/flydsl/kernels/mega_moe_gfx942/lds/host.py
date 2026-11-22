# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LdsMegaMoE: the gfx942 EP8 MegaMoE on FP8 128x128 block-scale experts (wq "fp8") or a8w4's fd2 MXFP4
experts (wq "fp4", decoded to FP8 on the way into LDS).

Per call, on the current stream (x is first quantized to FP8 per 1x128 by aiter's per_1x128 quant):
  ep_prep    count routes per global expert, exchange the count rows, derive every route's sorted row
  ep_plan    the send list and this rank's sorted blocks
  ep_mega    one persistent kernel: P2P dispatch, GateUp and Down interleaved, every route added with BF16 atomics
             into its source rank's out
  ep_finish  all ranks' adds done -> out is complete

The kernels are pyhip's EPMoEGfx942 (branch luwei/942-fp8-megamoe, written for MiMo-V2.6-Flash prefill); comm.py
and mega.py list the changes: the call epoch in device memory (CUDA graphs), ids out of [0, experts) not routed,
swiglu_limit, and a topk that need not be a power of two.

The receive buffers hold the worst case (every route of every rank to this rank), so no route is ever dropped.
Construction is collective: every EP rank builds the same instances in the same order after MORI shmem is
initialized, and calls forward the same number of times.
"""

from contextlib import nullcontext

import torch

from ..host import peer_table, symmetric
from .comm import (
    ARRIVED,
    E_LOCAL,
    NPES,
    PLAN_INTS,
    ROW_BITS,
    SCHED_EXPECTED,
    ep_finish,
    ep_plan,
    ep_prep,
)
from .gateup import OOB_OFFSET, SORT_BLOCK_M
from .mega import ep_mega

_MODEL_DIM, _INTER_DIM, _MAX_TOPK = 4096, 2048, 8
# Producer workgroups of the mega kernel and the units Down lags GateUp by (EPMoEGfx942's defaults).
_PRODUCERS, _LAG = 16, 2


def eligible(*, world_size, model_dim, inter_dim, experts, topk, act="silu"):
    """Whether the mega kernel runs the layer: gfx942, EP8 x 32 local experts, H 4096, I 2048, top-k <= 8, silu."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return (
        get_gfx_runtime() == "gfx942"
        and (world_size, model_dim, inter_dim, experts)
        == (NPES, _MODEL_DIM, _INTER_DIM, NPES * E_LOCAL)
        and 0 < topk <= _MAX_TOPK
        and act == "silu"
    )


def _on(stream):
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


class LdsMegaMoE:
    """One rank's EP8 MoE layer (32 local experts) on the mega kernel."""

    def __init__(
        self,
        *,
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        max_tok_per_rank,
        swiglu_limit=0.0,
        wq="fp8",
        w1=None,
        w1_scale=None,
        w2=None,
        w2_scale=None,
    ):
        if not eligible(
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
        ):
            raise ValueError(
                f"LdsMegaMoE does not run world_size={world_size}, model_dim={model_dim}, "
                f"inter_dim={inter_dim}, experts={experts}, topk={topk} on this device"
            )
        if not swiglu_limit >= 0:
            raise ValueError(
                f"swiglu_limit={swiglu_limit}: expected >= 0 (0: no clamp)"
            )
        if wq not in ("fp8", "fp4"):
            raise ValueError(f"wq={wq!r}: expected 'fp8' or 'fp4'")
        import mori.shmem as ms

        import aiter

        self.rank, self.world, self.topk, self.experts = rank, world_size, topk, experts
        self.hidden, self.inter = model_dim, inter_dim
        self.swiglu_limit = float(swiglu_limit)
        self.wq = wq
        self.combine = "bf16"
        self.token_shift = max(int(max_tok_per_rank - 1).bit_length(), 1)
        self.tcap = 1 << self.token_shift
        # Every route of every rank to this rank, plus one partial block per local expert.
        rows = world_size * self.tcap * topk + E_LOCAL * SORT_BLOCK_M
        self.s_cap = -(-rows // SORT_BLOCK_M) * SORT_BLOCK_M
        # send_info's row field, and the buffer descriptors: a masked lane's OOB_OFFSET must stay out of range.
        if self.s_cap > 1 << ROW_BITS or self.s_cap * model_dim >= OOB_OFFSET:
            raise ValueError(
                f"max_tok_per_rank={max_tok_per_rank}: the worst-case receive buffer of {self.s_cap} rows x "
                f"{model_dim} B exceeds the kernels' 2 GiB / 2^{ROW_BITS}-row bound"
            )
        self.slices = _PRODUCERS * 8
        if self.tcap * topk > self.slices * 1024:
            raise ValueError(
                f"max_tok_per_rank={max_tok_per_rank}: too many routes per ep_plan slice"
            )
        self.dev = torch.device("cuda", torch.cuda.current_device())
        H, I, S, dev = model_dim, inter_dim, self.s_cap, self.dev
        self.recv_a = symmetric((S, H), torch.uint8)
        self.recv_scale = symmetric((S, H // 128), torch.float32)
        self.down_ids = symmetric((S,), torch.int32)
        self.recv_w = symmetric((S,), torch.float32)
        self.ws = symmetric((ARRIVED + S // SORT_BLOCK_M,), torch.int32)
        self.out = symmetric((self.tcap, H), torch.bfloat16)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        names = ["recv_a", "recv_scale", "down_ids", "recv_w", "ws", "out"]
        self.tables = {
            name: peer_table(getattr(self, name), rank, world_size) for name in names
        }

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.int32, device=dev)

        routes = self.tcap * topk
        self.route_idx = zeros(routes)
        self.send_base = zeros(NPES * E_LOCAL)
        self.gu_ids = zeros(S)
        self.expert_ids = zeros(S // SORT_BLOCK_M)
        self.num_valid = zeros(1)
        self.status = zeros(8)
        self.send_route = zeros(routes)
        self.send_info = zeros(routes)
        self.plan = zeros(PLAN_INTS)
        self.sched = zeros(SCHED_EXPECTED + 3 * (S // SORT_BLOCK_M))
        # The call epoch (ep_prep and ep_finish use epoch[0] + 1; ep_finish stores it).
        self.epoch = zeros(1)
        # The mega kernel's GateUp A scales, transposed block by block from recv_scale; GateUp's FP8 output.
        self.recv_scale_t = torch.empty((H // 128, S), dtype=torch.float32, device=dev)
        self.a2 = torch.empty((S, I), dtype=torch.uint8, device=dev)
        self.a2_scale_t = torch.empty((I // 128, S), dtype=torch.float32, device=dev)
        self._quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
        self.w1 = self.w1_scale = self.w2 = self.w2_scale = None
        if w1 is not None:
            self.bind_weights(w1, w1_scale, w2, w2_scale)

    def bind_weights(self, w1, w1_scale, w2, w2_scale):
        """This rank's experts: w1 [32, 2I, H] / w2 [32, H, I] FP8 (float8_e4m3fnuz) in aiter's
        shuffle_weight(layout=(16, 16)) form, gate rows before up rows; w1_scale [32, 2I/128, H/128] and
        w2_scale [32, H/128, I/128] FP32 dequant scales of the 128x128 blocks.
        wq "fp4": a8w4's prepare_weights output instead, w1 [32, 2I, H/2] / w2 [32, H, I/2] bytes and
        w1_scale [32 * 2I * (H/32 + 4)] / w2_scale [32 * H * (I/32 + 4)] bytes."""
        H, I = self.hidden, self.inter
        expect = {
            "w1": (w1, (E_LOCAL, 2 * I, H), torch.float8_e4m3fnuz),
            "w1_scale": (w1_scale, (E_LOCAL, 2 * I // 128, H // 128), torch.float32),
            "w2": (w2, (E_LOCAL, H, I), torch.float8_e4m3fnuz),
            "w2_scale": (w2_scale, (E_LOCAL, H // 128, I // 128), torch.float32),
        }
        if self.wq == "fp4":
            expect = {
                "w1": (w1, (E_LOCAL, 2 * I, H // 2), None),
                "w1_scale": (w1_scale, (E_LOCAL * 2 * I * (H // 32 + 4),), None),
                "w2": (w2, (E_LOCAL, H, I // 2), None),
                "w2_scale": (w2_scale, (E_LOCAL * H * (I // 32 + 4),), None),
            }
        for name, (t, shape, dtype) in expect.items():
            if (
                tuple(t.shape) != shape
                or (t.dtype != dtype if dtype is not None else t.element_size() != 1)
                or not t.is_contiguous()
                or t.device != self.dev
            ):
                raise ValueError(
                    f"{name} {tuple(t.shape)} {t.dtype} on {t.device}: expected contiguous {shape} "
                    f"{dtype or 'bytes'} on {self.dev}"
                )
        self.w1, self.w1_scale, self.w2, self.w2_scale = w1, w1_scale, w2, w2_scale

    def forward(self, x, topk_weights, topk_ids, *, stream=None):
        """x bf16 [T, H], topk_weights [T, topk], topk_ids [T, topk] (ids out of [0, experts) are no route).
        Returns bf16 [T, H], a view of an internal buffer that the next call overwrites.
        """
        T = x.shape[0]
        if x.dim() != 2 or x.shape[1] != self.hidden or x.dtype != torch.bfloat16:
            raise ValueError(
                f"x {tuple(x.shape)} {x.dtype}: expected [T, {self.hidden}] bfloat16"
            )
        self._check_routes(T, topk_weights, topk_ids)
        with _on(stream):
            if T:
                x_q, x_scale = self._quant(
                    x.contiguous(), quant_dtype=torch.float8_e4m3fnuz
                )
            else:
                x_q = torch.empty(
                    (0, self.hidden), dtype=torch.float8_e4m3fnuz, device=x.device
                )
                x_scale = torch.empty(
                    (0, self.hidden // 128), dtype=torch.float32, device=x.device
                )
            return self._run(x_q, x_scale, topk_weights, topk_ids)

    __call__ = forward

    def forward_prequant(self, x_q, x_scale, topk_weights, topk_ids, *, stream=None):
        """forward on x quantized as forward does (aiter's per_1x128 FP8 quant): x_q
        float8_e4m3fnuz [T, H], x_scale f32 [T, H / 128]."""
        T = x_q.shape[0]
        if (
            x_q.dim() != 2
            or x_q.shape[1] != self.hidden
            or x_q.dtype != torch.float8_e4m3fnuz
            or tuple(x_scale.shape) != (T, self.hidden // 128)
            or x_scale.dtype != torch.float32
        ):
            raise ValueError(
                f"x_q {tuple(x_q.shape)} {x_q.dtype}, x_scale {tuple(x_scale.shape)} "
                f"{x_scale.dtype}: expected [T, {self.hidden}] float8_e4m3fnuz and "
                f"[T, {self.hidden // 128}] float32"
            )
        self._check_routes(T, topk_weights, topk_ids)
        with _on(stream):
            return self._run(
                x_q.contiguous(), x_scale.contiguous(), topk_weights, topk_ids
            )

    def _check_routes(self, T, topk_weights, topk_ids):
        if tuple(topk_ids.shape) != (T, self.topk) or tuple(topk_weights.shape) != (
            T,
            self.topk,
        ):
            raise ValueError(
                f"topk_ids {tuple(topk_ids.shape)}, topk_weights {tuple(topk_weights.shape)}: expected "
                f"[{T}, {self.topk}]"
            )
        if T > self.tcap:
            raise ValueError(f"{T} rows: at most {self.tcap} (max_tok_per_rank)")
        if self.w1 is None:
            raise RuntimeError("bind_weights first")

    def _run(self, x_q, x_scale, topk_weights, topk_ids):
        T = x_q.shape[0]
        ids = topk_ids.to(torch.int32).contiguous()
        w = topk_weights.to(torch.float32).contiguous()
        t = self.tables
        out = self.out[:T]
        # Peers add into out only after this rank's ep_prep signalled (stream order).
        out.zero_()
        routes = T * self.topk
        ep_prep(
            ids,
            routes,
            self.rank,
            self.epoch,
            self.route_idx,
            self.ws,
            t["ws"],
            self.send_base,
            self.num_valid,
            self.s_cap,
            self.status,
            self.sched,
            self.plan,
        )
        ep_plan(
            self.slices,
            ids,
            self.route_idx,
            routes,
            self.rank,
            self.s_cap,
            self.plan,
            self.ws,
            self.send_route,
            self.send_info,
            self.gu_ids,
            self.down_ids,
            self.expert_ids,
            self.sched,
        )
        ep_mega(self, x_q, x_scale, w, _PRODUCERS, _LAG)
        ep_finish(self.rank, self.epoch, self.ws, t["ws"])
        return out

    def rows_needed(self):
        """Rows the last call sent to this rank (synchronizes)."""
        return int(self.status[1].item())
