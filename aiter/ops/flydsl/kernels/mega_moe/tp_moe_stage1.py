# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE Stage1.

Each TP rank owns ALL experts and one 1/tp shard of the intermediate
dimension. The caller passes its own DP token shard; this operator
quantizes locally, pushes the rows into every peer over P2P from inside
the GEMM1 kernel (DP group == TP group), runs grouping, GEMM1, SwiGLU and
per-1x32 FP8 output quantization, and returns the six tensors the
ordinary FlyDSL v2 FMoE GEMM2 consumes.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

from .quant import per_1x32_mx_quant

_SUPPORTED_TP = (4, 8)
_DEFAULT_STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


@dataclass(frozen=True)
class TPMoEStage1Output:
    """Everything ordinary FlyDSL v2 FMoE GEMM2 needs, plus host metadata."""

    inter_sorted_quant: torch.Tensor
    inter_sorted_shuffled_scale: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor

    m_logical: int
    max_sorted: int
    num_experts: int
    model_dim: int
    inter_dim: int
    topk: int
    sort_block_m: int


class TPMoEStage1:
    """Stateful TP4/TP8 MoE Stage1 operator.

    Preconditions (documented, not checked at runtime):
      * every rank in the group calls with the same ``m_local``
      * ``w1`` / ``w1_scale`` are already preshuffled for the a16w4
        gate/up-interleaved layout
      * the group used here is both the DP group and the TP group
    """

    def __init__(
        self,
        *,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        group=None,
        tp_size: int | None = None,
        tp_rank: int | None = None,
        device: torch.device | None = None,
        sort_block_m: int = 32,
        swiglu_limit: float = 0.0,
        stage1_kernel_name: str = _DEFAULT_STAGE1_KERNEL,
        max_tok_per_rank: int | None = None,
        pull: bool = True,
    ):
        self.group = group
        if (tp_size is None) != (tp_rank is None):
            raise ValueError(
                "tp_size and tp_rank must be supplied together, or both omitted"
            )
        if tp_size is None or tp_rank is None:
            if not dist.is_initialized():
                raise ValueError(
                    "TPMoEStage1 needs an initialized process group, or both tp_size and tp_rank"
                )
            tp_size = dist.get_world_size(group)
            tp_rank = dist.get_rank(group)
        if int(tp_size) not in _SUPPORTED_TP:
            raise ValueError(
                f"tp_size={tp_size} unsupported; expected one of {_SUPPORTED_TP}"
            )

        params = get_flydsl_kernel_params(stage1_kernel_name)
        if params is None:
            raise ValueError(f"unknown stage1 kernel name: {stage1_kernel_name}")
        if int(sort_block_m) != int(params["tile_m"]):
            raise ValueError(
                f"sort_block_m={sort_block_m} must equal the stage1 kernel tile_m="
                f"{params['tile_m']} ({stage1_kernel_name})"
            )
        if inter_dim % int(params["tile_n"]) != 0:
            raise ValueError(
                f"inter_dim={inter_dim} must be divisible by tile_n={params['tile_n']}"
            )
        # A *_fp4 stage1 kernel passes both checks above but returns an fp4x2
        # payload of shape [rows, inter_dim//2]; _pack's view(float8_e4m3fn) then
        # reports an inter_dim twice the real row width and GEMM2 over-reads.
        if str(params.get("out_dtype")) != "fp8":
            raise ValueError(
                f"{stage1_kernel_name} emits out_dtype={params.get('out_dtype')!r}; "
                "TPMoEStage1 requires an fp8 stage1 kernel (the v2 GEMM2 A operand)"
            )
        if (str(params.get("a_dtype")), str(params.get("b_dtype"))) != ("fp8", "fp4"):
            raise ValueError(
                f"{stage1_kernel_name} is not an afp8/wfp4 stage1 kernel: "
                f"a_dtype={params.get('a_dtype')!r} b_dtype={params.get('b_dtype')!r}"
            )
        if float(swiglu_limit) < 0:
            raise ValueError("swiglu_limit must be non-negative")

        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        if not (0 <= self.tp_rank < self.tp_size):
            raise ValueError(
                f"tp_rank={self.tp_rank} out of range for tp_size={self.tp_size}"
            )
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.topk = int(topk)
        self.sort_block_m = int(sort_block_m)
        self.swiglu_limit = float(swiglu_limit)
        self.stage1_kernel_name = stage1_kernel_name
        self.stage1_params = params
        dev = device or torch.device("cuda", torch.cuda.current_device())
        if dev.type == "cuda" and dev.index is None:
            # Normalise "cuda" -> "cuda:N"; _validate_call compares devices by
            # equality and torch.device("cuda") != torch.device("cuda", 0).
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        # The GEMM1 kernel derives E and inter_dim from w1 itself and ignores the
        # constructor values, so a mis-sized shard (e.g. a TP4 slice handed to a
        # TP8-configured op) would silently produce wrong numbers.
        expected_w1 = (self.experts, 2 * self.inter_dim, self.model_dim // 2)
        if tuple(w1.shape) != expected_w1:
            raise ValueError(
                f"w1 must be a preshuffled MXFP4 shard of shape {expected_w1} "
                f"(shuffle_weight_a16w4 preserves shape), got {tuple(w1.shape)}"
            )
        # shuffle_scale_a16w4 reshapes the scale, so only its element count is
        # a stable invariant.
        expected_scale_numel = (
            self.experts * 2 * self.inter_dim * (self.model_dim // 32)
        )
        if w1_scale.numel() != expected_scale_numel:
            raise ValueError(
                f"w1_scale must have {expected_scale_numel} elements, "
                f"got {w1_scale.numel()}"
            )
        self.w1 = w1
        self.w1_scale = w1_scale

        # The fused path's symmetric receive buffers are a collective allocation:
        # every rank must make it with identical arguments, at the same point in
        # the program. Building it here rather than on the first forward
        # call turns a mismatched max_tok_per_rank into a construction error
        # instead of a hang deep inside Mori.
        self.max_tok_per_rank = (
            None if max_tok_per_rank is None else int(max_tok_per_rank)
        )
        self._fused = None
        self.pull = bool(pull)
        if self.max_tok_per_rank is not None:
            from .tp_fused_stage1 import TPFusedStage1Runner
            from .tp_gather import TPActivationGather

            gather = TPActivationGather(
                model_dim=self.model_dim,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                max_tok_per_rank=self.max_tok_per_rank,
                device=self.device,
                enable_pull=self.pull,
            )
            self._fused = TPFusedStage1Runner(
                gather=gather,
                pull=self.pull,
                w=self.w1,
                w_scale=self.w1_scale,
                model_dim=self.model_dim,
                inter_dim=self.inter_dim,
                experts=self.experts,
                sort_block_m=self.sort_block_m,
                swiglu_limit=self.swiglu_limit,
            )

    def m_logical_for(self, m_local: int) -> int:
        return self.tp_size * int(m_local)

    def max_sorted_for(self, m_local: int) -> int:
        """Mirror of moe_sorting's max_num_tokens_padded.

        NOTE: this is NOT sort_block_m-aligned. The stage1 payload — and hence
        ``TPMoEStage1Output.max_sorted`` — is the next multiple of sort_block_m
        above this value.
        """
        return (
            self.m_logical_for(m_local) * self.topk
            + self.experts * self.sort_block_m
            - self.topk
        )

    def _all_gather_one(self, t):
        t = t.contiguous()
        if self.tp_size == 1:
            return t
        out = torch.empty(
            (t.shape[0] * self.tp_size,) + tuple(t.shape[1:]),
            dtype=t.dtype,
            device=t.device,
        )
        dist.all_gather_into_tensor(out, t, group=self.group)
        return out

    def _validate_call(self, x, route_weights, topk_ids, x_dtype):
        if x.dtype != x_dtype or not x.is_contiguous():
            raise ValueError(f"x must be contiguous {x_dtype}")
        if route_weights.dtype != torch.float32 or not route_weights.is_contiguous():
            raise ValueError("route_weights must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        for name, t in (
            ("x", x),
            ("route_weights", route_weights),
            ("topk_ids", topk_ids),
        ):
            if t.device != self.device:
                raise ValueError(f"{name} is on {t.device}, expected {self.device}")
        m_local = int(x.shape[0])
        if m_local <= 0:
            raise ValueError("m_local must be positive")
        if x.shape[1] != self.model_dim:
            raise ValueError(
                f"x must be [{m_local}, {self.model_dim}], got {tuple(x.shape)}"
            )
        if route_weights.shape != (m_local, self.topk):
            raise ValueError(
                f"route_weights must be [{m_local}, {self.topk}], got {tuple(route_weights.shape)}"
            )
        if topk_ids.shape != (m_local, self.topk):
            raise ValueError(
                f"topk_ids must be [{m_local}, {self.topk}], got {tuple(topk_ids.shape)}"
            )
        return m_local

    def _sort(self, topk_ids_g, weights_g):
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids_g,
            weights_g,
            self.experts,
            self.model_dim,
            torch.bfloat16,
            block_size=self.sort_block_m,
            # accumulate=False -> moe_buf comes back as a (0,0) placeholder and the
            # sorting kernel skips its zero pass (aiter/fused_moe.py:326-334). We
            # discard moe_buf, so paying for a [M_global, model_dim] memset every
            # call would be pure waste.
            accumulate=False,
        )
        return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids

    def _pack(
        self,
        payload,
        scale,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        m_global,
    ):
        return TPMoEStage1Output(
            inter_sorted_quant=payload.view(torch.float8_e4m3fn),
            inter_sorted_shuffled_scale=scale,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            m_logical=m_global,
            max_sorted=int(payload.shape[0]),
            num_experts=self.experts,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            topk=self.topk,
            sort_block_m=self.sort_block_m,
        )

    def _run_fused(
        self, x_q, x_scale, sorted_ids, sorted_expert_ids, num_valid_ids, m_local
    ):
        if self._fused is None:
            raise RuntimeError(
                "forward needs the symmetric receive buffers; construct "
                "TPMoEStage1 with max_tok_per_rank=<max m_local> (a collective "
                "allocation, so every rank must pass the same value)"
            )
        sbm = self.sort_block_m
        max_sorted = -(-int(sorted_ids.shape[0]) // sbm) * sbm
        return self._fused.run(
            x_q=x_q,
            x_scale=x_scale,
            sorted_token_ids=sorted_ids,
            expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            max_sorted=max_sorted,
        )

    def forward(self, x_bf16, route_weights, topk_ids) -> "TPMoEStage1Output":
        """BF16 entry: local quant, one metadata collective, sort, then one kernel.

        The activation rows never travel over a collective: they are pushed into
        every peer's symmetric receive buffer over P2P from inside the GEMM1
        kernel, so the only thing NCCL still carries is the packed routing
        metadata.
        """
        m_local = self._validate_call(x_bf16, route_weights, topk_ids, torch.bfloat16)
        m_global = self.m_logical_for(m_local)
        x_q, x_scale = self.quantize(x_bf16, m_local=m_local)
        # topk_ids and route_weights have the same shape; ship them as one int32
        # buffer so the metadata costs one collective instead of two.
        meta = torch.empty(
            (m_local, 2 * self.topk), dtype=torch.int32, device=self.device
        )
        meta[:, : self.topk] = topk_ids
        meta[:, self.topk :] = route_weights.view(torch.int32)
        meta_g = self._all_gather_one(meta)
        ids_g = meta_g[:, : self.topk].contiguous()
        wts_g = meta_g[:, self.topk :].contiguous().view(torch.float32)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._sort(
            ids_g, wts_g
        )
        payload, scale = self._run_fused(
            x_q, x_scale, sorted_ids, sorted_expert_ids, num_valid_ids, m_local
        )
        return self._pack(
            payload,
            scale,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            m_global,
        )

    def quantize(self, x_bf16, m_local=None):
        """Local per-1x32 BF16 -> FP8 E4M3 + E8M0. Same routine MegaMoEV2 uses.

        Under pull the peers read this rank's rows straight out of a Mori
        symmetric buffer, so the quantize kernel writes there directly. Staging
        the rows with a copy afterwards would be correct but would add two
        launches to the critical path, which is precisely the cost fusing exists
        to remove.
        """
        if self.pull and self._fused is not None and m_local is not None:
            g = self._fused.gather
            dst_x, dst_scale = g.tx_views(m_local, g.current_parity())
            return per_1x32_mx_quant(
                x_bf16, quant_mode="fp8", out=dst_x, scale_out=dst_scale
            )
        return per_1x32_mx_quant(x_bf16, quant_mode="fp8")

    __call__ = forward
    forward_bf16 = forward
