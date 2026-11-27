# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.comms.fused.all_reduce_rmsnorm_moe_route import (
    _all_reduce_rmsnorm_moe_route_kernel,
)
from aiter.ops.triton.comms.ipc_buffers import ERROR_BYTES, IpcCommBuffers
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "ALL_REDUCE_RMSNORM_MOE_ROUTE"
_MAX_HIDDEN = 8192
_MAX_TOKENS = 512
_SCORING = {"sigmoid": 0, "softmax": 1}


def _get_config() -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    return dict(load_config_json(f"{config_dir}/DEFAULT.json")["any"])


def _plan_layout(c):
    """Buffer layout and router plan of a comm-like object c (world, blocks, num_warps, config,
    max_tokens, hidden, num_experts, topk, router_fp32). Router tasks are 16 experts x K part:
    the fewest parts that keep a task's weights within the register budget, spread over the
    CTAs after the reduce CTAs."""
    c.gmax = triton.cdiv(c.max_tokens, c.world)
    c.s1max = min(c.max_tokens, c.config["ONESHOT_MAX_ROWS"])
    c.rmax = max(c.s1max, c.gmax)
    nks = c.hidden // 16
    spw_max = c.config["ROUTER_STEPS_FP32" if c.router_fp32 else "ROUTER_STEPS_BF16"]
    kp = 1
    while triton.cdiv(triton.cdiv(nks, kp), c.num_warps) > spw_max:
        kp += 1
    c.kp = kp
    ntask = c.num_experts // 16 * kp
    c.nrc = min(ntask, c.blocks - c.rmax)
    assert c.nrc > 0, "too many tokens for the CTAs"
    c.tpc = triton.cdiv(ntask, c.nrc)
    hb = c.hidden * 2
    c.offsets = {"os": ERROR_BYTES}
    c.offsets["rs"] = c.offsets["os"] + 2 * c.world * c.s1max * hb
    c.offsets["ag"] = c.offsets["rs"] + 2 * c.world * c.gmax * hb
    c.offsets["pk"] = c.offsets["ag"] + 2 * c.world * c.gmax * hb
    c.nbytes = c.offsets["pk"] + 2 * c.max_tokens * 2 * c.topk * 8  # _open rounds it up
    # scratch (int64 words): ready words, partial logits [KP][rmax][E], one-shot picks
    c.scx = 0
    c.scl = triton.cdiv(c.rmax, 64) * 64
    c.scp = c.scl + c.kp * c.rmax * c.num_experts
    c.scratch_words = c.scp + c.s1max * 2 * c.topk


def _lds_bytes(c, tokens):
    """The LDS bytes of the kernel variant of a call with this many tokens, with a correction
    bias. The compiler reuses LDS between the router, route and sort stages, so the largest
    stage sets the size. Keep it in step with the kernel's allocations."""
    t = 64 * c.num_warps
    e = c.num_experts
    ep = triton.next_power_of_2(e + 1)
    rtmax = triton.next_power_of_2(triton.cdiv(c.routed_rows(tokens), 16))
    router = rtmax * c.num_warps * 16 * 16 * 4
    # partial logits, then logits; selection values and bias only above 256 experts
    kep = triton.next_power_of_2(triton.cdiv(c.kp * e, t) * t)
    route = 4 * (kep + (3 if e > 256 else 1) * ep)
    rows = max(32, triton.next_power_of_2(tokens))
    bmn = triton.next_power_of_2(e * triton.cdiv(rows, 32))
    pkn = triton.next_power_of_2(max(rows * c.topk, t))
    pre = bmn * 8 + pkn * 8 <= 98304  # the kernel's PRE
    # bitmaps, word prefix counts, counts, offsets, picks; scan scratch a warp
    sort = 4 * (bmn * (2 if pre else 1) + 2 * ep + 2 * pkn + min(c.num_warps, ep // 64))
    return max(router, route, sort)


class AllReduceRmsnormMoeRoute(IpcCommBuffers):
    """
    The IPC buffers, scratch and launch colours of ``fused_all_reduce_rmsnorm_moe_route``
    for one group of 2, 4 or 8 ranks on gfx950 and one model shape. The buffer of a rank
    holds the error record, the inboxes and the top-k picks. The constructor, ``reset`` and
    ``close`` are collective. Call ``reset`` after a wait timed out and at least once every
    2^31 launches. Error sites: 1 (reduce), 2 (all-gather), 3 (router), 4 (route), 5 (sort).
    The constructor rejects a shape that needs more than the 160 KiB of LDS of a CU.

    Args:
        group: The CPU (gloo) process group of the ranks.
        max_tokens (int): The most tokens of a call, at most 512. Each row of the routed
            power-of-2 bucket of a call needs one CTA, and one more CTA must be free. For
            example, with 2 ranks and 256 CTAs, max_tokens is at most 256.
        hidden (int): The number of bf16 values in a row.
        num_experts (int): The number of routed experts.
        topk (int): The experts of a token.
        router_fp32 (bool): The router weights are fp32 (else bf16).
        timeout_s (float): The time limit of each wait in the kernel.
        config (dict, optional): Values that replace the values of the JSON config
            (BLOCKS, num_warps, ONESHOT_MAX_ROWS, ONESHOT_MAX_BYTES, ROUTER_STEPS_FP32,
            ROUTER_STEPS_BF16), for example in a benchmark sweep. An unknown key is an
            error.
    """

    def __init__(
        self,
        group,
        max_tokens,
        hidden,
        num_experts,
        topk,
        router_fp32=True,
        timeout_s=60.0,
        config=None,
    ):
        super().__init__(group, timeout_s)
        assert (
            hidden % 64 == 0 and hidden <= _MAX_HIDDEN
        ), f"hidden must be a multiple of 64, up to {_MAX_HIDDEN}"
        assert (
            num_experts % 16 == 0 and num_experts <= 2048
        ), "experts: a multiple of 16"
        assert 1 <= topk <= min(64, num_experts), "topk must be 1 to 64"
        assert 0 < max_tokens <= _MAX_TOKENS, f"max_tokens must be 1 to {_MAX_TOKENS}"
        self.hidden, self.max_tokens = hidden, max_tokens
        self.num_experts, self.topk, self.router_fp32 = num_experts, topk, router_fp32
        self.config = _get_config()
        if config is not None:
            unknown = sorted(set(config) - set(self.config))
            assert not unknown, f"unknown config keys: {unknown}"
            self.config.update(config)
        # All CTAs must run at the same time: one CTA a CU.
        self.blocks = min(self.config["BLOCKS"], self.num_cus)
        self.num_warps = self.config["num_warps"]
        _plan_layout(self)
        routed = max(self.routed_rows(t) for t in self.warmup_tokens())
        assert routed < self.blocks, (
            f"max_tokens={max_tokens}: a call can need {routed} reduce CTAs, but "
            f"there are {self.blocks} CTAs and the pushes need one more"
        )
        lds = max(_lds_bytes(self, t) for t in self.warmup_tokens())
        lds_cap = arch_info._LDS_CAP_BYTES["gfx950"]
        assert lds <= lds_cap, (
            f"a call can need {lds} bytes of LDS, but a gfx950 CU has {lds_cap}: "
            "use fewer experts, a smaller topk or a smaller max_tokens"
        )
        self.scratch = torch.zeros(
            self.scratch_words, dtype=torch.int64, device=self.device
        )
        self._open(self.nbytes, self.blocks)

    def _clear(self):
        """Also clears the scratch."""
        self.scratch.zero_()
        super()._clear()

    def one_shot(self, tokens):
        """The protocol for a call: one-shot push (one hop, every rank reduces and routes every
        row) while the bytes a rank sends stay within the config's limit, else reduce-scatter +
        all-gather."""
        sent = (self.world - 1) * tokens * self.hidden * 2
        return tokens <= self.s1max and sent <= self.config["ONESHOT_MAX_BYTES"]

    def routed_rows(self, tokens):
        """The reduce CTAs of a call: one for each row of its power-of-2 routed bucket."""
        rows_b = triton.next_power_of_2(tokens)
        return rows_b if self.one_shot(tokens) else triton.cdiv(rows_b, self.world)

    def warmup_tokens(self):
        """One token count for each kernel variant a call can compile (power-of-2 row buckets
        and the protocol): launching each once compiles them all, for example before HIP graph
        capture. Launches are collective, so all ranks must use the same list."""
        seen, out = set(), []
        for t in range(1, self.max_tokens + 1):
            key = (triton.next_power_of_2(t), self.one_shot(t))
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def max_sorted(self, tokens, block_size):
        """The length of sorted_ids, as aiter.moe_sorting allocates it."""
        n = tokens * self.topk + self.num_experts * block_size - self.topk
        return triton.cdiv(n, block_size) * block_size


def _constexprs(
    comm,
    tokens,
    one_shot,
    scoring,
    has_bias,
    num_expert_group,
    topk_group,
    renormalize,
    gemma,
    has_res,
    quant,
    block_size,
    zero_moe_buf,
):
    """The compile-time arguments of a call. The row count itself is a runtime argument; the
    loops and LDS sizes come from its power-of-2 bucket."""
    rows_b = triton.next_power_of_2(tokens)
    routed_b = comm.routed_rows(tokens)
    v = triton.cdiv(comm.hidden // 8, 64 * comm.num_warps)
    units = (comm.world - 1) * routed_b * v
    # push units a CTA: the CTAs between the reduce and the router CTAs when they can take
    # them all at up to 8 each, else every CTA after the reduce CTAs
    free = comm.blocks - comm.nrc - routed_b
    if free > 0 and triton.cdiv(units, free) <= 8:
        pmax = triton.next_power_of_2(triton.cdiv(units, free))
    else:
        pmax = triton.next_power_of_2(triton.cdiv(units, comm.blocks - routed_b))
    return {
        "HIDDEN": comm.hidden,
        "WORLD": comm.world,
        "E": comm.num_experts,
        "TOPK": comm.topk,
        "SCORE": _SCORING[scoring],
        "HAS_BIAS": has_bias,
        "NGROUP": num_expert_group,
        "TOPKG": topk_group,
        "RENORM": renormalize,
        "GEMMA": gemma,
        "HAS_RES": has_res,
        "QUANT": quant,
        "WFP32": comm.router_fp32,
        "SBLK": block_size,
        "ZERO": zero_moe_buf,
        "BLOCKS": comm.blocks,
        "PROTO": 0 if one_shot else 1,
        "RTMAX": triton.next_power_of_2(triton.cdiv(routed_b, 16)),
        "SORT_ROWS": max(32, rows_b),
        "PMAX": pmax,
        "KP": comm.kp,
        "NRC": comm.nrc,
        "TPC": comm.tpc,
        "S1MAX": comm.s1max,
        "GMAX": comm.gmax,
        "SMAX": comm.max_tokens,
        "RMAXL": comm.rmax,
        "OS0": comm.offsets["os"],
        "RS0": comm.offsets["rs"],
        "AG0": comm.offsets["ag"],
        "PK0": comm.offsets["pk"],
        "SCX": comm.scx,
        "SCL": comm.scl,
        "SCP": comm.scp,
    }


def fused_all_reduce_rmsnorm_moe_route(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    router_weight: torch.Tensor,
    comm: AllReduceRmsnormMoeRoute,
    correction_bias: torch.Tensor | None = None,
    scoring: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
    gemma: bool = False,
    quant: bool = True,
    block_size: int = 16,
    zero_moe_buf: bool = True,
    out: dict | None = None,
) -> dict:
    """
    Returns the MoE inputs of a tensor-parallel decode step in one launch: the all-reduce
    of x over the ranks of ``comm``, the residual add, the RMSNorm and its MXFP8 copy, the
    router logits, the top-k experts and the moe_sorting outputs. All ranks get the same
    bits.

    The all-reduce adds the ranks in rank order in fp32 and rounds once to bf16. Then s = h
    + residual, residual_out = bf16(s) and normed = bf16(s * rstd * w'), with w' = 1 +
    weight (gemma) or weight. The E8M0 scale of a group of 32 is ``mx_e8m0_scale(amax, 8)``.
    The logits have fp32 accumulation. The selection value is sigmoid(logit) + bias, or the
    logit (softmax). With num_expert_group > 1, only the topk_group best groups are
    candidates (a group's score is the sum of its 2 best values with a bias, else its best
    value). Ties go to the lower expert. The weights are the sigmoids (or the softmax over
    all experts), divided by their sum when renormalize, times routed_scaling_factor.

    All ranks must make the same calls in the same order. HIP graphs are supported.

    Args:
        x (torch.Tensor): Contiguous [tokens, hidden] bf16, the partial sum of this rank.
        residual (torch.Tensor, optional): Contiguous [tokens, hidden] bf16, the same on all
            ranks. None: no residual add.
        norm_weight (torch.Tensor): Contiguous [hidden] bf16.
        eps (float): The epsilon of the norm.
        router_weight (torch.Tensor): Contiguous [experts, hidden], fp32 or bf16 (as comm).
        comm (AllReduceRmsnormMoeRoute): The buffers of the group.
        correction_bias (torch.Tensor, optional): Contiguous [experts] fp32, added for the
            selection only. Only with sigmoid scoring.
        scoring (str): "sigmoid" or "softmax".
        renormalize (bool): Divide the weights by their sum.
        routed_scaling_factor (float): Factor of the weights.
        num_expert_group (int): Expert groups (1: no group limit). With a correction bias,
            each group needs 2 experts or more.
        topk_group (int): Groups a token can use. They must have topk experts or more.
        gemma (bool): Use 1 + weight (Gemma norm) in place of weight.
        quant (bool): Also write the MXFP8 copy of the normed rows.
        block_size (int): The moe_sorting block.
        zero_moe_buf (bool): Zero moe_buf ([tokens, hidden] bf16).
        out (dict, optional): Output tensors to write into, by the names below.

    Returns:
        dict: normed and residual_out [tokens, hidden] bf16, x_q [tokens, hidden]
        float8_e4m3fn and x_scale [tokens, hidden / 32] uint8 (when quant), topk_ids
        [tokens, topk] int32, topk_weights [tokens, topk] fp32, and sorted_ids,
        sorted_weights, sorted_expert_ids, num_valid_ids and moe_buf as aiter.moe_sorting
        returns them. Each rank gets all rows.

    Limits: gfx950, 2, 4 or 8 ranks, hidden a multiple of 64 up to 8192, experts a multiple
    of 16 up to 2048, topk up to 64, tokens 1 to comm.max_tokens. All CTAs of the launch
    must run at the same time. If a wait times out, the outputs are wrong and
    ``comm.error()`` is not 0. Then call ``comm.reset()`` on all ranks.
    """
    tokens, hidden = x.shape
    e, k = comm.num_experts, comm.topk
    assert hidden == comm.hidden and 0 < tokens <= comm.max_tokens
    assert x.dtype == norm_weight.dtype == torch.bfloat16
    assert (
        x.is_contiguous()
        and norm_weight.is_contiguous()
        and norm_weight.shape == (hidden,)
    )
    if residual is not None:
        assert residual.dtype == torch.bfloat16 and residual.shape == x.shape
        assert residual.is_contiguous()
    wdt = torch.float32 if comm.router_fp32 else torch.bfloat16
    assert router_weight.dtype == wdt and router_weight.shape == (e, hidden)
    assert router_weight.is_contiguous()
    if correction_bias is not None:
        assert correction_bias.dtype == torch.float32 and correction_bias.shape == (e,)
        assert correction_bias.is_contiguous(), "correction_bias must be contiguous"
    assert scoring in _SCORING, f"scoring must be one of {tuple(_SCORING)}"
    assert correction_bias is None or scoring == "sigmoid", (
        "correction_bias needs sigmoid scoring: with softmax, vLLM ranks the experts "
        "by softmax(logit) + bias, and the kernel does not"
    )
    assert num_expert_group >= 1 and e % num_expert_group == 0
    assert 1 <= topk_group <= num_expert_group <= 32
    group_size = e // num_expert_group
    assert topk_group * group_size >= k, (
        f"the {topk_group} kept groups have {topk_group * group_size} experts, "
        f"fewer than topk={k}"
    )
    assert num_expert_group == 1 or correction_bias is None or group_size >= 2, (
        "with a correction bias, the score of a group is the sum of its 2 best "
        "experts: each group needs 2 experts or more"
    )
    assert block_size in (16, 32, 64, 128)
    dev = x.device
    o = dict(out or {})

    def get(name, shape, dtype):
        if name not in o:
            o[name] = torch.empty(shape, dtype=dtype, device=dev)
        t = o[name]
        assert t.dtype == dtype and tuple(t.shape) == tuple(shape), name
        assert t.is_contiguous(), name
        return t

    mp = comm.max_sorted(tokens, block_size)
    normed = get("normed", (tokens, hidden), torch.bfloat16)
    res_out = get("residual_out", (tokens, hidden), torch.bfloat16)
    if quant:
        x_q = get("x_q", (tokens, hidden), torch.float8_e4m3fn)
        x_s = get("x_scale", (tokens, hidden // 32), torch.uint8)
    else:
        x_q, x_s = normed, normed  # not written
    ids = get("topk_ids", (tokens, k), torch.int32)
    wts = get("topk_weights", (tokens, k), torch.float32)
    sids = get("sorted_ids", (mp,), torch.int32)
    sw = get("sorted_weights", (mp,), torch.float32)
    seids = get("sorted_expert_ids", (mp // block_size,), torch.int32)
    nv = get("num_valid_ids", (2,), torch.int32)
    buf = get("moe_buf", (tokens, hidden), torch.bfloat16)

    one_shot = comm.one_shot(tokens)
    _LOGGER.info(
        "ALL_REDUCE_RMSNORM_MOE_ROUTE [gluon]: x=%s world=%d experts=%d topk=%d one_shot=%s",
        tuple(x.shape),
        comm.world,
        e,
        k,
        one_shot,
    )
    _all_reduce_rmsnorm_moe_route_kernel[(comm.blocks,)](
        x.view(torch.int32),
        (residual if residual is not None else x).view(torch.int32),
        norm_weight.view(torch.int32),
        router_weight,
        correction_bias if correction_bias is not None else wts,
        res_out.view(torch.int32),
        normed.view(torch.int32),
        x_q.view(torch.int32),
        x_s.view(torch.uint8),
        ids,
        wts,
        sids,
        sw,
        seids,
        nv,
        buf.view(torch.int32),
        comm.scratch,
        comm.colors,
        comm.peers,
        tokens,
        comm.rank,
        comm.timeout_us,
        eps,
        routed_scaling_factor,
        **_constexprs(
            comm,
            tokens,
            one_shot,
            scoring,
            correction_bias is not None,
            num_expert_group,
            topk_group,
            renormalize,
            gemma,
            residual is not None,
            quant,
            block_size,
            zero_moe_buf,
        ),
        num_warps=comm.num_warps,
        enable_fp_fusion=False,  # the sum of squares adds rounded products
    )
    result = {
        "normed": normed,
        "residual_out": res_out,
        "topk_ids": ids,
        "topk_weights": wts,
        "sorted_ids": sids,
        "sorted_weights": sw,
        "sorted_expert_ids": seids,
        "num_valid_ids": nv,
        "moe_buf": buf,
    }
    if quant:
        result["x_q"], result["x_scale"] = x_q, x_s
    return result
