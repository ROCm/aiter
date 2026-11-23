# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MegaMoEGfx942: the EP8 a8w4 / a8w8 MegaMoE layer for gfx942 (MI300X), shaped like
MegaMoEV2, with its three paths:

- main (``forward`` / ``forward_prequant`` while large_M is off; ``main.py``, ``comm.py``):
  five kernels per call. ``comm.py`` quantizes x, exchanges the route counts and plans
  every route's destination row; ``main.py`` is one persistent kernel that writes each
  route's row into its destination, runs gemm1 / gemm2 on 128-row blocks as the rows
  arrive (W streamed into VGPRs) and adds every route into its source rank's output (BF16
  atomics); ``comm.py`` then waits until all ranks' adds are done. Its tasks only wait for
  work that is already running, so it needs no co-residency.
- small_M (``forward_small_m``; ``small_m.py``): one kernel per call for decode-sized
  batches. Every local expert owns fixed slots, so a route takes its row with one atomic
  (no count exchange, no plan), and the combine sums each token's rows in slot order. It
  spins on its peers inside the kernel, so it needs the whole GPU: nothing may run on
  another stream at the same time. a8w8 has no small kernel: its small_M runs the main
  path at the small-input capacity.
- large_M (``forward`` / ``forward_prequant`` once enabled; ``large_m/``): the LDS mega
  kernel, GEMM tiles of 256 rows with A and W staged through LDS (a8w4's MXFP4 decoded to
  FP8 by the W loaders) that skip the MFMAs of a block's padding rows; faster than the
  main path from about 1000 tokens per rank. No co-residency needed.

The main path is built only while large_M is off: unlike MegaMoEV2, an instance holds
either the main path or large_M, and ``set_large_m_enabled`` cannot switch to the other
one. ``x_quant="per_token"`` runs every call on the main path.

Every rank builds the same instances in the same order (after MORI shmem is
initialized) and makes the same calls in the same order, as with MegaMoEV2. Each rank
receives at most every route of every rank (8 x max_tok_per_rank x topk rows plus one
partial block per expert), so no route is dropped. The main path's buffers at
max_tok_per_rank 2048 / 4096 / 8192: 0.42 / 0.82 / 1.63 GiB of the MORI symmetric heap
(``MORI_SHMEM_HEAP_SIZE``) plus 0.21 / 0.41 / 0.81 GiB of device memory (top-6); the
largest is 8192 at top-6 and 4096 at top-8 (the receive buffer stays below 2 GiB).
large_M takes about the same, small_M at 128 tokens per rank 0.18 GiB of the heap.

Weights: the standard aiter a16w4 preshuffle (``shuffle_weight_a16w4(w, 16, False)``,
``shuffle_scale_a16w4``), converted once per layer by ``prepare_weights`` into the fd2
form; the per-channel exponents ride at the tail of each scale tensor, so a layer is
still ``(w1, w1_scale, w2, w2_scale)``. ``quant="a8w8"`` takes fp8 experts with f32
128x128 block scales instead, in aiter's fp8 MoE preshuffle
``shuffle_weight(w, (16, 16))``, as they are (``prepare_weights(..., quant="a8w8")``
checks them and zeroes the W of zero-scale blocks in place).
"""

from contextlib import nullcontext

import torch

from ..mega_moe import envs
from . import weights as _weights

_EP, _MODEL_DIM, _INTER_DIM, _EXPERTS, _NUM_CU = 8, 4096, 2048, 256, 304
# DeepSeek-V4-Flash routes each token to 6 experts, MiMo-V2.6-Flash to 8.
_TOPKS = (6, 8)
# a8w4: mxfp4 experts (fd2 W); a8w8: fp8 experts with 128x128 block scales (W in the
# (16, 16) preshuffle)
_WQ = {"a8w4": "fp4", "a8w8": "fp8"}


def eligible(
    *, world_size, model_dim, inter_dim, experts, topk, quant="a8w4", act="silu"
):
    """Whether this build runs the layer: gfx942 with 304 CUs, EP8 and the MoE shape of
    DeepSeek-V4-Flash (H 4096, I 2048, 256 experts, top-6) or MiMo-V2.6-Flash (the same
    with top-8), a8w4 or a8w8."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    return (
        get_gfx_runtime() == "gfx942"
        and get_cu_num() == _NUM_CU
        and (world_size, model_dim, inter_dim, experts)
        == (_EP, _MODEL_DIM, _INTER_DIM, _EXPERTS)
        and topk in _TOPKS
        and quant in _WQ
        and act == "silu"
    )


def _on(stream):
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


class MegaMoEGfx942:
    """One rank's EP8 a8w4 / a8w8 MoE layer. Construction is collective: every rank
    builds the same instances in the same order after MORI shmem is initialized.

    ``small_m_mtprs``, ``small_m_enabled`` and ``large_m_enabled`` are MegaMoEV2's
    arguments (None: the ``AITER_MEGA_SMALL_M`` / ``AITER_MEGA_LARGE_M`` flags).
    small_M takes up to max(``small_m_mtprs``) tokens per rank. With large_M enabled,
    ``forward`` runs it and the main path is not built."""

    def __init__(
        self,
        *,
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        w1,
        w1_scale,
        w2,
        w2_scale,
        max_tok_per_rank,
        quant="a8w4",
        act="silu",
        swiglu_limit=0.0,
        small_m_mtprs=(128,),
        small_m_enabled=None,
        large_m_enabled=None,
        x_quant="per_1x128",
    ):
        if not eligible(
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            act=act,
        ):
            raise ValueError(
                "MegaMoEGfx942 runs gfx942 (304 CUs), EP8, H 4096, I 2048, 256 experts, "
                f"top-6 or top-8, a8w4 or a8w8 silu; got world_size={world_size} "
                f"model_dim={model_dim} inter_dim={inter_dim} experts={experts} "
                f"topk={topk} quant={quant} act={act}"
            )
        if max_tok_per_rank <= 0:
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be positive")
        if x_quant not in ("per_1x128", "per_token"):
            raise ValueError(f"x_quant={x_quant!r}: 'per_1x128' or 'per_token'")
        if swiglu_limit < 0:
            raise ValueError(f"swiglu_limit={swiglu_limit} must be non-negative")
        from .protocol import qx_of

        self.rank, self.world_size = int(rank), int(world_size)
        self.model_dim, self.inter_dim = int(model_dim), int(inter_dim)
        self.experts, self.topk = int(experts), int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.quant = quant
        self.dev = torch.device("cuda", torch.cuda.current_device())
        limit = float(swiglu_limit)
        # As MegaMoEV2: a limit of 0 means no clamp.
        self._act = "silu_clamped" if limit > 0 else "silu"
        # The kernels' activation scalars, aiter act.situ_params order (beta, 1 / beta,
        # linear_beta, 1 / linear_beta, swiglu_limit): silu_clamped reads the limit.
        self._situ = (1.0, 1.0, 1.0, 1.0, limit if limit > 0 else float("inf"))
        self._x_quant, self._qx = x_quant, qx_of(x_quant)
        self._no_route = torch.full((), -1, dtype=torch.int32, device=self.dev)
        self._weights = {}
        self._W = self._weight_dict(w1, w1_scale, w2, w2_scale)
        if quant != "a8w4" and x_quant != "per_1x128":
            raise ValueError("a8w8 quantizes x per 1x128 (x_quant='per_1x128')")
        if large_m_enabled is None:
            large_m_enabled = envs.AITER_MEGA_LARGE_M
        if small_m_enabled is None:
            small_m_enabled = envs.AITER_MEGA_SMALL_M
        # large_M quantizes x per 1x128 only.
        self._large = None
        if large_m_enabled and x_quant == "per_1x128":
            from .large_m.host import LargeMRuntime

            self._large = LargeMRuntime(
                rank=rank,
                world_size=world_size,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                max_tok_per_rank=self.mtpr,
                swiglu_limit=limit,
                wq=_WQ[quant],
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
            )
        self._large_on = self._large is not None
        self._main = None if self._large_on else self._main_runtime(self.mtpr)
        self._small, self._small_mtpr = None, 0
        if small_m_enabled:
            self._small_mtpr = max(int(m) for m in small_m_mtprs)
            if quant == "a8w4":
                from .small_m import SmallMRuntime

                self._small = SmallMRuntime(
                    rank=self.rank,
                    world_size=self.world_size,
                    model_dim=self.model_dim,
                    inter_dim=self.inter_dim,
                    topk=self.topk,
                    tmax=self.world_size * self._small_mtpr,
                    act=self._act,
                    x_quant=x_quant,
                )
            else:
                # The small kernel streams fd2 MXFP4 W only.
                self._small = self._main_runtime(self._small_mtpr)

    def _main_runtime(self, mtpr):
        from .main import MainRuntime

        return MainRuntime(
            rank=self.rank,
            world_size=self.world_size,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            topk=self.topk,
            mtpr=mtpr,
            wq=_WQ[self.quant],
            act=self._act,
            qx=self._qx,
        )

    # ---- weights ---------------------------------------------------------------

    @staticmethod
    def prepare_weights(
        w1, w1_scale, w2, w2_scale, *, w1_gate_up_interleaved=False, quant="a8w4"
    ):
        """One layer's local experts -> ``(w1, w1_scale, w2, w2_scale)`` as the kernels
        read them. a8w4: the standard aiter a16w4 preshuffle -> fd2; a8w8: fp8 in aiter's
        (16, 16) preshuffle with f32 128x128 block scales, read as they are (the W of
        zero-scale blocks zeroed in place). Load time, once per layer."""
        if w1_gate_up_interleaved:
            raise NotImplementedError(
                "MegaMoEGfx942 takes w1 shuffled with gate_up=False for now"
            )
        if quant == "a8w8":
            return _weights.prepare_weights_a8w8(w1, w1_scale, w2, w2_scale)
        if quant != "a8w4":
            raise ValueError(f"quant={quant!r}: 'a8w4' or 'a8w8'")
        return _weights.prepare_weights(w1, w1_scale, w2, w2_scale)

    def _weight_dict(self, w1, w1_scale, w2, w2_scale):
        key = (w1.data_ptr(), w1_scale.data_ptr(), w2.data_ptr(), w2_scale.data_ptr())
        W = self._weights.get(key)
        if W is None:
            self._check_weights(w1, w1_scale, w2, w2_scale)
            fp8 = self.quant == "a8w8"
            W = {
                "w1": w1,
                "w1_scale": w1_scale,
                # a8w8 has no per-channel exponents: the kernel does not read them
                "w1_exp": w1_scale if fp8 else _weights.tail_exp(w1, w1_scale),
                "w2": w2,
                "w2_scale": w2_scale,
                "w2_exp": w2_scale if fp8 else _weights.tail_exp(w2, w2_scale),
            }
            self._weights[key] = W
        return W

    def _check_weights(self, w1, w1_scale, w2, w2_scale):
        """The kernels read the weights through raw pointers: ``prepare_weights``'
        output for this rank's experts, contiguous on the layer's device."""
        E, H, I = self.experts // self.world_size, self.model_dim, self.inter_dim
        if self.quant == "a8w8":
            want = (
                ("w1", w1, (E, 2 * I, H), torch.float8_e4m3fnuz),
                ("w1_scale", w1_scale, (E, 2 * I // 128, H // 128), torch.float32),
                ("w2", w2, (E, H, I), torch.float8_e4m3fnuz),
                ("w2_scale", w2_scale, (E, H // 128, I // 128), torch.float32),
            )
            for name, t, shape, dtype in want:
                if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
                    raise ValueError(
                        f"{name} {t.dtype} {tuple(t.shape)}: expected contiguous "
                        f"{dtype} {shape} (prepare_weights(..., quant='a8w8') output)"
                    )
                if t.device != self.dev:
                    raise ValueError(
                        f"{name} is on {t.device}, the layer on {self.dev}"
                    )
            return
        want = (
            ("w1", w1, (E, 2 * I, H // 2)),
            ("w1_scale", w1_scale, (E * 2 * I * (H // 32 + 4),)),
            ("w2", w2, (E, H, I // 2)),
            ("w2_scale", w2_scale, (E * H * (I // 32 + 4),)),
        )
        for name, t, shape in want:
            ok = t.dim() == len(shape) and tuple(t.shape) == shape
            if len(shape) == 1:
                ok = t.numel() == shape[0]
            if not ok or t.element_size() != 1 or not t.is_contiguous():
                raise ValueError(
                    f"{name} {t.dtype} {tuple(t.shape)}: expected contiguous bytes "
                    f"{shape} (prepare_weights output)"
                )
            if t.device != self.dev:
                raise ValueError(f"{name} is on {t.device}, the layer on {self.dev}")

    def bind_weights(self, w1, w1_scale, w2, w2_scale):
        self._W = self._weight_dict(w1, w1_scale, w2, w2_scale)
        if self._large is not None:
            self._large.bind_weights(w1, w1_scale, w2, w2_scale)

    def bind_layer(self, layer_slot, w1, w1_scale, w2, w2_scale):
        self.bind_weights(w1, w1_scale, w2, w2_scale)

    def select_layer_state(self, layer_slot):
        """No per-layer protocol state on gfx942."""

    def preload_aot_bundles(self):
        """No AOT bundles yet: the kernels compile on the first call."""

    # ---- small / large paths -----------------------------------------------------

    @property
    def small_m_mtpr(self):
        """Largest per-rank token count ``forward_small_m`` accepts: 0 when the path is
        off."""
        return self._small_mtpr

    @property
    def large_m_available(self):
        return self._large is not None

    def set_large_m_enabled(self, enabled):
        """MegaMoEV2's switch between large_M and the main path; every rank must make the
        same choice. An instance holds only one of the two, so this only checks that the
        path asked for is the one built."""
        if enabled and self._large is None:
            raise RuntimeError(
                "the large-input path is not built (large_m_enabled or "
                "AITER_MEGA_LARGE_M, x_quant='per_1x128')"
            )
        if not enabled and self._main is None:
            raise RuntimeError(
                "the main path is not built: this instance runs large_M "
                "(large_m_enabled=False builds the main path instead)"
            )
        self._large_on = bool(enabled)

    # ---- calls -------------------------------------------------------------------

    def _routes(self, x, topk_ids, wts, main=None):
        """The routes as int32 ids (out-of-range ids -1) and f32 weights: padded to the
        main kernel's top-k in ``main``'s buffers, else as they are."""
        # The kernels take raw pointers: every input must be on this layer's device.
        T = x.shape[0]
        for name, t in (("x", x), ("topk_ids", topk_ids), ("wts", wts)):
            if t.device != self.dev:
                raise ValueError(f"{name} is on {t.device}, the layer on {self.dev}")
        if x.dim() != 2 or any(
            tuple(t.shape) != (T, self.topk) for t in (topk_ids, wts)
        ):
            raise ValueError(
                f"x {tuple(x.shape)}, topk_ids {tuple(topk_ids.shape)}, wts "
                f"{tuple(wts.shape)}: expected [T, H], [T, {self.topk}], [T, {self.topk}]"
            )
        # Out-of-range ids are no routes, as in MegaMoEV2 (sglang marks padding rows with
        # id == experts); the kernels skip negative ids only.
        topk_ids = topk_ids.to(torch.int32)
        if main is None:
            ids = torch.where(topk_ids < self.experts, topk_ids, self._no_route)
            return ids.contiguous(), wts.to(torch.float32).contiguous()
        # Padded to the main kernel's top-k in place: the extra slots of these buffers
        # stay -1 / 0 (no route) from its constructor on.
        if T > main.tcap:
            raise ValueError(f"{T} rows: at most {main.tcap} (max_tok_per_rank)")
        ids, w = main.ids_pad[:T], main.wts_pad[:T]
        torch.where(
            topk_ids < self.experts, topk_ids, self._no_route, out=ids[:, : self.topk]
        )
        w[:, : self.topk].copy_(wts)
        return ids, w

    def forward(
        self,
        x_bf16,
        wts,
        topk_ids,
        *,
        stream=None,
        slice_output=True,
        mask_invalid_slots=None,
    ):
        """x bf16 [T, H], wts f32 [T, topk], topk_ids int32 [T, topk]. Returns bf16
        [T, H], a view of an internal buffer that the next call overwrites. Invalid ids
        are always skipped, so ``mask_invalid_slots`` has no effect; the output is always
        sliced to T rows."""
        with _on(stream):
            if self._large_on:
                ids, w = self._routes(x_bf16, topk_ids, wts)
                return self._large.forward(x_bf16, w, ids)
            ids, w = self._routes(x_bf16, topk_ids, wts, self._main)
            return self._main.forward(x_bf16, ids, w, self._W, self._situ)

    __call__ = forward

    def forward_prequant(
        self, x_q, x_scale, wts, topk_ids, *, stream=None, slice_output=True
    ):
        """``forward`` on x already quantized as this instance's ``x_quant``: x_q
        float8_e4m3fnuz [T, H] and x_scale f32 [T, H / 128] ("per_1x128") or [T]
        ("per_token")."""
        T, H = x_q.shape
        shape = (T, H // 128) if self._qx else (T,)
        if x_q.dtype != torch.float8_e4m3fnuz or x_scale.dtype != torch.float32:
            raise TypeError(
                f"x_q {x_q.dtype} / x_scale {x_scale.dtype}: expected "
                "float8_e4m3fnuz / float32"
            )
        if tuple(x_scale.shape) != shape or x_scale.device != self.dev:
            raise ValueError(
                f"x_scale {tuple(x_scale.shape)} on {x_scale.device}: expected "
                f"{shape} on {self.dev}"
            )
        with _on(stream):
            if self._large_on:
                ids, w = self._routes(x_q, topk_ids, wts)
                return self._large.forward_prequant(x_q, x_scale, w, ids)
            ids, w = self._routes(x_q, topk_ids, wts, self._main)
            return self._main.forward(x_q, ids, w, self._W, self._situ, x_scale)

    def forward_small_m(
        self, x_bf16, wts, topk_ids, *, max_tokens, stream=None, slice_output=True
    ):
        """MegaMoEV2's small-input call: the small kernel (a8w4), the main path at the
        small-input capacity (a8w8). ``max_tokens``: the most rows of this call on any
        EP rank, the same on every rank."""
        if self._small is None:
            raise RuntimeError(
                "the small-input path is off (small_m_enabled, AITER_MEGA_SMALL_M)"
            )
        need = int(max_tokens)
        if need < int(x_bf16.shape[0]) or need > self._small_mtpr:
            raise ValueError(
                f"max_tokens={need}: this rank has {int(x_bf16.shape[0])} rows, the "
                f"small-input path takes at most {self._small_mtpr}"
            )
        # The main kernel takes the routes padded to its top-k, the small one as they are.
        pad = self._small if self.quant == "a8w8" else None
        with _on(stream):
            ids, w = self._routes(x_bf16, topk_ids, wts, pad)
            return self._small.forward(x_bf16, ids, w, self._W, self._situ)
