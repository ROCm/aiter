# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer as one kernel (MegaMoE TP), gfx950.

Experts are replicated over the TP group and ``inter_dim`` is sharded; one launch
runs the collectives around GEMM1 + activation + GEMM2 of this rank's inter slice.
``comm_mode``: ``"ag_rs"`` (sequence-parallel in/out) or ``"ar"`` (replicated in,
all-reduced out). Weights are MXFP4 (``shuffle_weight(16, 16)`` + ``e8m0_shuffle``).

Contract:

* ``forward`` is a collective: every rank calls it the same number of times, in the
  same order, with the same local token count (and, ar, the same input and routing).
  Layers of one shape may share an instance (``set_weights``): their forwards form one
  collective sequence.
* ``prepare(local_tokens)`` before CUDA graph capture: it compiles every launch config,
  runs each once and checks every rank's watchdog (reset + retry on a timeout).
* The kernel is persistent (one CTA per CU, CTAs and ranks spin on each other): it
  needs the whole GPU. Do not run it concurrently with other kernels (other streams,
  other processes, CU-masked / partitioned GPUs).
* A wait that times out (2 s) sets a sticky bit and the launch's output is invalid.
  ``error_flag()`` is that bit as a device tensor (copy it asynchronously, e.g. every N
  steps); ``check_errors()`` synchronizes and raises; ``reset()`` (collective) restarts
  the cross-rank state, which a timed-out launch leaves inconsistent.
* The returned tensors (``out=None``, ar outputs, the fused tail's rows) are views of
  internal buffers, valid until the next forward of this instance; pass ``out=`` to keep
  the output.
* Launch epochs are int32, compared wrap-safe; a flag slot unused for 2**31 launches can
  read stale: ``reset()`` at least that often for very long-lived instances.
* The symmetric memory uses hipIpc handles; in containers on the host network ROCm 7.1
  needs ``HSA_ENABLE_IPC_MODE_LEGACY=1``.

Large batches (ag_rs, Swiglu, fp4, from ``bigm_min_tokens`` global tokens, default 8192, 0
disables) run the big-M kernel (bigm_layer: chunked, SDMA, per-batch tuned configs from
configs/mega_moe_tp_bigm_tuned.csv) behind the same ``forward``; it needs MORI built with
SDMA (importable, or its install directory in ``AITER_MORI_PATH``). ``max_local_tokens`` must
then cover the largest batch (e.g. 8192 for 32768 tokens on tp4); the small-batch engine
only holds up to ``bigm_min_tokens / world_size - 1`` of them.

Schedules (planned in the kernel from the routing): up to 256 global tokens the dynamic
one, from ``lb_min`` (default 256) global tokens up the large-batch (LB) one; its row tile
is 16 * ``lb_mt`` rows (``"T1:N1,...,N"``: up to Ti tokens Ni), 16 * ``lb_mt_small`` up to
``lb_small_max`` tokens; ``lb_npp`` GEMM1 column blocks per A pass; ``lb_q`` output column
chunks per GEMM2 unit. Unset fields use the per-batch tuned configs
(configs/mega_moe_tp_tuned.csv), else the defaults (lb_min 256, lb_mt 5, lb_npp 2, lb_q 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter import ActivationType

from .kernels.mega_moe_tp.mega_moe_tp import (
    COMM_MODES,
    LaunchCfg,
    MegaMoeTPEngine,
    MegaMoeTPLDSError,
)

__all__ = [
    "COMM_MODES",
    "MegaMoeTP",
    "MegaMoeTPConfig",
    "MegaMoeTPLDSError",
    "mega_moe_tp_supported",
]

_ACTS = (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)


def mega_moe_tp_supported(gfx: str | None = None) -> bool:
    """Whether this device has the kernel (gfx950)."""
    if gfx is None:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    return gfx == "gfx950"


@dataclass(frozen=True)
class MegaMoeTPConfig:
    rank: int
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_local_tokens: int
    activation: ActivationType = ActivationType.Silu
    beta: float | None = None
    linear_beta: float | None = None
    swiglu_limit: float | None = None
    comm_mode: str = "ag_rs"
    comm_dtype: str = "fp8"
    ar_gather: str = "auto"
    schedule: str = "dynamic"
    act_dtype: str = "fp4"
    lb_min: int | None = None
    lb_mt: int | str | None = None
    lb_mt_small: int | None = None
    lb_small_max: int | None = None
    lb_npp: int | None = None
    lb_q: int | None = None
    # the fused tail's norm: GemmaRMSNorm (scale 1 + w) or RMSNorm (w), and its eps
    tail_eps: float = 1e-6
    tail_gemma: bool = True
    # global tokens from which forward runs the big-M kernel (None: 8192; 0: never)
    bigm_min_tokens: int | None = None


class MegaMoeTP:
    """One fused TP MoE layer (see the module docstring)."""

    def __init__(
        self,
        cfg: MegaMoeTPConfig,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        group=None,
        device: torch.device | None = None,
    ):
        if cfg.activation not in _ACTS:
            raise ValueError(f"MegaMoeTP: unsupported activation {cfg.activation}")
        if not mega_moe_tp_supported():
            raise ValueError("MegaMoeTP: needs gfx950")
        act = cfg.activation.name.lower()
        if cfg.comm_mode not in COMM_MODES:
            raise ValueError(f"MegaMoeTP: unknown comm_mode {cfg.comm_mode!r}")
        situ = act == "situv2"
        self.cfg = cfg
        bmin = 8192 if cfg.bigm_min_tokens is None else cfg.bigm_min_tokens
        big_ok = (
            cfg.comm_mode == "ag_rs"
            and cfg.activation == ActivationType.Swiglu
            and cfg.act_dtype == "fp4"
        )
        # local tokens from which forward runs the big-M kernel (0: never)
        self.bigm_m = -(-bmin // cfg.world_size) if bmin and big_ok else 0
        self._bigm = None
        self._bigm_args = dict(
            w1=w1, w1_scale=w1_scale, w2=w2, w2_scale=w2_scale, group=group, device=device
        )
        small_max = cfg.max_local_tokens
        if self.bigm_m and cfg.max_local_tokens >= self.bigm_m:
            small_max = self.bigm_m - 1
        self.engine = MegaMoeTPEngine(
            rank=cfg.rank,
            world_size=cfg.world_size,
            model_dim=cfg.model_dim,
            inter_dim=cfg.inter_dim,
            experts=cfg.experts,
            topk=cfg.topk,
            max_local_tokens=small_max,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            activation=act,
            situ_beta=cfg.beta if situ and cfg.beta is not None else 1.0,
            situ_linear_beta=(
                cfg.linear_beta if situ and cfg.linear_beta is not None else 1.0
            ),
            swiglu_limit=cfg.swiglu_limit,
            comm_mode=cfg.comm_mode,
            comm_dtype=cfg.comm_dtype,
            ar_gather=cfg.ar_gather,
            schedule=cfg.schedule,
            act_dtype=cfg.act_dtype,
            lb={
                "min": cfg.lb_min,
                "mt": cfg.lb_mt,
                "mt_small": cfg.lb_mt_small,
                "small_max": cfg.lb_small_max,
                "npp": cfg.lb_npp,
                "q": cfg.lb_q,
            },
            tail_eps=cfg.tail_eps,
            tail_gemma=cfg.tail_gemma,
            group=group,
            device=device,
        )

    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor | None = None,
        tail=None,
        bf16: bool = False,
    ):
        """ag_rs: x [m, H] (own tokens); ar: x [M, H] replicated. tail (ag_rs, see
        tail_ok): (res_in, res_out, norm_w) -> also res_out = y + res_in and every rank's
        FP8 rows (+ fp32 scales; bf16: + the bf16 rows) of the norm of res_out: returns
        (y, q, scale[, rows])."""
        if self._big(x_local.shape[0]):
            local = bool(self.engine.tn_local)
            res = self.bigm().moe(x_local, topk_weights, topk_ids, out=out, tail=tail, local=local)
            if tail is not None and bf16:
                res = (*res, self._bigm_rows(tail[1], tail[2], local))
            return res
        return self.engine(x_local, topk_weights, topk_ids, out, tail=tail, bf16=bf16)

    def _bigm_rows(self, res_out, norm_w, local):
        """the big-M tail's bf16 rows: (Gemma)RMSNorm of res_out, this rank's (local) or every
        rank's rows (the kernel writes the FP8 rows only). Every rank's rows are exchanged
        through the big-M kernel's hipIpc arena (device copies over xGMI), not through the
        caller's group: that is a gloo CPU group in ATOM, where the ~400 MB all-gather of a
        32K batch took ~0.3 s."""
        r = res_out.to(torch.float32, copy=True)  # (scaled in place below)
        w = norm_w.float() + 1.0 if self.engine.tn_gemma else norm_w.float()
        inv = torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + float(self.engine.tn_eps))
        rows = r.mul_(inv).mul_(w).to(torch.bfloat16)
        if local:
            return rows
        return self.bigm().gather_rows(rows)

    def _big(self, local_tokens: int) -> bool:
        return bool(self.bigm_m) and local_tokens >= self.bigm_m

    def bigm(self):
        """The big-M kernel's host object (collective on first use: its SDMA window)."""
        if self._bigm is None:
            from .kernels.mega_moe_tp.bigm_layer import PAD, MegaMoeTPBigM

            cfg, a = self.cfg, self._bigm_args
            self._bigm = MegaMoeTPBigM(
                hidden=cfg.model_dim,
                inter=cfg.inter_dim,
                experts=cfg.experts - 1,  # bigm counts the last expert apart (shared)
                topk=cfg.topk - 1,
                # (the big-M kernel runs on local rows padded to a multiple of PAD)
                max_tokens=-(-cfg.max_local_tokens // PAD) * PAD * cfg.world_size,
                chunks=None,
                eps=self.engine.tn_eps,
                gemma=self.engine.tn_gemma,
                scale=1.0,
                shared_w=0.0,
                swiglu_limit=self.engine.swiglu_limit,  # (None -> the engine's default)
                w1=a["w1"],
                w1_scale=a["w1_scale"],
                w2=a["w2"],
                w2_scale=a["w2_scale"],
                group=a["group"],
                device=a["device"],
                io=True,
                # comm_dtype "bf16" (the accurate mode): bf16 route rows (MXFP8 otherwise)
                opts={"AITER_BIGM_R8": "0"} if cfg.comm_dtype == "bf16" else None,
            )
        b = self._bigm
        # the tail norm's eps / Gemma flag (settable on the engine after construction)
        eps, gemma = float(self.engine.tn_eps), bool(self.engine.tn_gemma)
        if b.cfg[:2] != (eps, gemma):
            b.cfg = (eps, gemma) + tuple(b.cfg[2:])
        return b

    __call__ = forward

    def set_weights(
        self,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Run the next forwards on another layer's weights of the same shape."""
        self.engine.set_weights(w1, w1_scale, w2, w2_scale)
        self._bigm_args.update(w1=w1, w1_scale=w1_scale, w2=w2, w2_scale=w2_scale)
        if self._bigm is not None:
            self._bigm.set_weights(w1, w1_scale, w2, w2_scale)

    def prepare(
        self,
        local_tokens,
        tail: bool = False,
        tail_bf16: bool = False,
        warmup: bool = True,
    ) -> None:
        """Collective: compile, arm and (warmup) run once and check the launch configs
        of these local token counts (tail / tail_bf16: their fused-tail variants); counts on
        the big-M path set up its SDMA window and (warmup) compile and run its kernels once
        (a first launch inside CUDA graph capture could not)."""
        ts = [local_tokens] if isinstance(local_tokens, int) else list(local_tokens)
        small = [t for t in ts if not self._big(t)]
        if small:
            self.engine.prepare(small, tail, tail_bf16, warmup)
        big = [t for t in ts if self._big(t)]
        if not big and self._big(self.cfg.max_local_tokens):
            # the big-M path is reachable: set it up (MORI communicator, SDMA queues, arena)
            # and compile it here, collectively at init, not lazily inside the first big
            # forward (where its 85-90 s compile and the setup interleave with the model)
            big = [self.cfg.max_local_tokens]
        if big:
            self.bigm()
        for m in big if warmup else ():
            cfg, dev = self.cfg, self.bigm().device
            x = torch.zeros((m, cfg.model_dim), dtype=torch.bfloat16, device=dev)
            # balanced routing: token i's slot j to expert (i (topk - 1) + j) % (experts - 1),
            # the shared expert (experts - 1) last
            r = torch.arange(m, device=dev)[:, None] * (cfg.topk - 1) + torch.arange(cfg.topk - 1, device=dev)
            ids = torch.cat([r % (cfg.experts - 1), torch.full((m, 1), cfg.experts - 1, device=dev)], 1)
            ids = ids.to(torch.int32).contiguous()
            tw = torch.full((m, cfg.topk), 1.0 / cfg.topk, dtype=torch.float32, device=dev)
            self.forward(x, tw, ids)
            if tail or tail_bf16:
                res = torch.zeros_like(x)
                self.forward(x, tw, ids, tail=(res, res, x[0]), bf16=tail_bf16)
            torch.cuda.synchronize()

    @property
    def max_local_tokens(self) -> int:
        return self.cfg.max_local_tokens if self.bigm_m else self.engine.mmax

    def tail_ok(self, local_tokens: int) -> bool:
        """forward(tail=...) runs for this local token count."""
        if self._big(local_tokens):
            return True
        return self.engine.tail_ok(local_tokens)

    def launch_config(self, local_tokens: int) -> LaunchCfg:
        return self.engine.config(local_tokens)

    def error_flag(self) -> torch.Tensor:
        """Sticky watchdog bits (nonzero: a launch timed out) as a device tensor."""
        return self.engine.error_flag()

    def poll_errors(self) -> int:
        """Nonzero if a wait inside the kernel gave up (a peer never arrived)."""
        e = self.engine.poll_errors()
        if not e and self._bigm is not None:
            e = self._bigm.poll_errors()
        return e

    def check_errors(self) -> None:
        """Raise RuntimeError (and clear) if a wait gave up since the last check."""
        if self._bigm is not None and self._bigm.poll_errors():
            self._bigm.clear_errors()
            raise RuntimeError("MegaMoeTP: a big-M wait on a peer gave up (results invalid)")
        self.engine.check_errors()

    def clear_errors(self) -> None:
        self.engine.clear_errors()
        if self._bigm is not None:
            self._bigm.clear_errors()

    def reset(self) -> None:
        """Collective: restart the cross-rank state (after a timeout / desync)."""
        self.engine.reset()
