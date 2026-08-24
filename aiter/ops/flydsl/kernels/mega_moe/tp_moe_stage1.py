# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE Stage1.

Each TP rank owns ALL experts and one 1/tp shard of the intermediate
dimension. The caller passes its own DP token shard; this operator
all-gathers across the TP group (DP group == TP group), runs grouping,
GEMM1, SwiGLU and per-1x32 FP8 output quantization, and returns the
six tensors the ordinary FlyDSL v2 FMoE GEMM2 consumes.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, get_flydsl_kernel_params
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort
from aiter.utility.fp4_utils import moe_mxfp4_sort

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
        transport: str = "allgather_bf16",
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
        if transport != "allgather_bf16":
            raise NotImplementedError(
                f"transport={transport!r} is not implemented yet; phase 1 only "
                "supports 'allgather_bf16'"
            )

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
        self.transport = transport
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        # flydsl_moe_stage1 derives E and inter_dim from w1 itself and ignores the
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

    def _all_gather_inputs(self, x, route_weights, topk_ids):
        """Gather the three per-rank inputs in rank-major order.

        Returns (x_g, weights_g, ids_g) laid out so that
        ``global_token = src_rank * m_local + local_token``.
        """
        return (
            self._all_gather_one(x),
            self._all_gather_one(route_weights),
            self._all_gather_one(topk_ids),
        )

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

    def _run_gemm1(
        self, a_fp8, a_scale_sorted, sorted_ids, sorted_expert_ids, num_valid_ids
    ):
        p = self.stage1_params
        payload, scale = flydsl_moe_stage1(
            a_fp8,
            self.w1,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out=None,
            topk=self.topk,
            tile_m=int(p["tile_m"]),
            tile_n=int(p["tile_n"]),
            tile_k=int(p["tile_k"]),
            a_dtype=str(p["a_dtype"]),
            b_dtype=str(p["b_dtype"]),
            out_dtype=str(p["out_dtype"]),
            act="silu",
            w1_scale=self.w1_scale,
            a1_scale=a_scale_sorted,
            sorted_weights=None,
            waves_per_eu=int(p.get("waves_per_eu", 3)),
            b_nt=int(p.get("b_nt", 0)),
            gate_mode=str(p.get("gate_mode", "separated")),
            xcd_swizzle=int(p.get("xcd_swizzle", 0)),
            k_wave=int(p.get("k_wave", 1)),
            swiglu_limit=(self.swiglu_limit or None),
            v2_output_layout=True,
        )
        return payload, scale

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

    def forward(self, x_bf16, route_weights, topk_ids) -> "TPMoEStage1Output":
        """BF16 entry. Gathers bf16, then quantizes after sorting."""
        m_local = self._validate_call(x_bf16, route_weights, topk_ids, torch.bfloat16)
        m_global = self.m_logical_for(m_local)

        x_g, wts_g, ids_g = self._all_gather_inputs(x_bf16, route_weights, topk_ids)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._sort(
            ids_g, wts_g
        )
        a_fp8, a_scale_sorted = fused_dynamic_mxfp8_quant_moe_sort(
            x_g,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=m_global,
            topk=self.topk,
            block_size=self.sort_block_m,
            sorted_weights=None,
        )
        payload, scale = self._run_gemm1(
            a_fp8, a_scale_sorted, sorted_ids, sorted_expert_ids, num_valid_ids
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

    def quantize(self, x_bf16):
        """Local per-1x32 BF16 -> FP8 E4M3 + E8M0. Same routine MegaMoEV2 uses."""
        return per_1x32_mx_quant(x_bf16, quant_mode="fp8")

    def forward_prequant(
        self, x_fp8, x_scale, route_weights, topk_ids
    ) -> TPMoEStage1Output:
        """Prequantized entry.

        Gathers FP8 payload + E8M0 scale instead of BF16, i.e. quantize-then-gather.
        Per row this moves ``model_dim + model_dim/32`` bytes instead of
        ``model_dim * 2``.
        """
        m_local = self._validate_call(
            x_fp8, route_weights, topk_ids, torch.float8_e4m3fn
        )
        if not x_scale.is_contiguous():
            raise ValueError("x_scale must be contiguous")
        if x_scale.device != self.device:
            raise ValueError(f"x_scale is on {x_scale.device}, expected {self.device}")
        # moe_mxfp4_sort does a bare .view(torch.uint8); anything wider than one
        # byte would be silently reinterpreted instead of converted.
        if x_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(
                f"x_scale must be uint8 or float8_e8m0fnu E8M0 scales, "
                f"got {x_scale.dtype}"
            )
        if x_scale.shape != (m_local, self.model_dim // 32):
            raise ValueError(
                f"x_scale must be [{m_local}, {self.model_dim // 32}], "
                f"got {tuple(x_scale.shape)}"
            )
        m_global = self.m_logical_for(m_local)

        x_g, wts_g, ids_g = self._all_gather_inputs(x_fp8, route_weights, topk_ids)
        scale_g = self._all_gather_one(x_scale)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._sort(
            ids_g, wts_g
        )
        a_scale_sorted = moe_mxfp4_sort(
            scale_g.view(m_global, 1, -1),
            sorted_ids,
            num_valid_ids,
            m_global,
            self.sort_block_m,
        )
        payload, scale = self._run_gemm1(
            x_g, a_scale_sorted, sorted_ids, sorted_expert_ids, num_valid_ids
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

    __call__ = forward
    forward_bf16 = forward
