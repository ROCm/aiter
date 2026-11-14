# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel EP MegaMoE: launch configs, launches, and the
pure-torch reference oracle used to validate the kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.utility import fp4_utils

from ..mega_moe.quant import per_1x32_mx_quant
from ..mega_moe_tp.mega_moe_tp import SymmetricArena
from ..tensor_shim import _preload_compiled, _run_compiled
from .mega_moe_ep_config import (
    EPOCH_SLOT,
    FLAG_INTS,
    MAX_CTAS,
    MAX_RANKS,
    N_XCD,
    SMALL_SHORT_INTER,
    TILE_K,
    a_banks,
    ctrl_ints,
    ep_shape_supported,
    inter_rows,
    mega_moe_ep_consts,
)
from .mega_moe_ep_kernel import compile_mega_moe_ep

__all__ = ["LaunchCfg", "MegaMoeEPEngine", "ep_reference"]

LDS_LIMIT = 160 * 1024
# a4w4 row tile: mt=8 (TILE_M=128: each W tile serves twice the rows) from
# MT8_MIN_TOKENS tokens, else mt=4.  mt=8 needs the per-echunk inter rebase
# (kernel_config forces it): without it the K-loop spills.  a8w4 mt=8 needs > 160 KB LDS
# (only a 2-bank A ring fits), so a8 stays at mt=4.
MT8_MIN_TOKENS = 768
# Small-batch kernel variant (LaunchCfg.small) for engines of max_tok_per_rank <=
# SMALL_MAX_TOKENS: picked from mtpr, so every rank runs the same variant.  a8w4 row
# tile mt=2 up to SMALL_MT2_MAX tokens; mt only shapes the rank-local echunks.
SMALL_MAX_TOKENS = 512
SMALL_MT2_MAX = 159
# a4w4 likewise up to SMALL_MT2_MAX_A4.
SMALL_MT2_MAX_A4 = 159
# small-batch kernel streams its weights (kernel_config STREAM_WEIGHTS) from
# SMALL_STREAM_MIN tokens; at inter_dim <= SMALL_SHORT_INTER only up to
# SMALL_STREAM_MAX_SHORT_INTER.
SMALL_STREAM_MIN = 2
SMALL_STREAM_MAX_SHORT_INTER = 416


def _dequant_expert(weight, scale, rows, cols):
    """Dequantize one MXFP4 expert weight (packed FP4 values * per-1x32 E8M0 scale)."""
    values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather_variable(tensor, token_counts=None, group=None):
    """All-gather rows across the ranks of ``group`` with per-rank variable counts
    (pad, gather, trim)."""
    world = dist.get_world_size(group=group)
    if token_counts is None:
        local = torch.tensor(tensor.shape[0], dtype=torch.int64, device=tensor.device)
        counts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(counts, local, group=group)
        token_counts = [int(c.item()) for c in counts]
    max_count = max(token_counts)
    if tensor.shape[0] < max_count:
        padding = torch.empty(
            (max_count - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding))
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat([v[:c] for v, c in zip(gathered, token_counts)]), token_counts


@torch.no_grad()
def ep_reference(
    x,
    route_weights,
    ids,
    ref_weights,
    *,
    rank,
    world_size,
    model_dim,
    inter_dim,
    experts,
    swiglu_limit,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    group=None,
):
    """Pure-torch EP MoE oracle: dispatch (all-gather +
    route to the expert owner) -> per-local-expert GEMM1 + act + GEMM2 scaled by the
    top-k weight -> combine (all-reduce the partials, keep this rank's rows).

    ``ref_weights`` are the *unshuffled* quantized weights
    ``(w1_q, w1_scale, w2_q, w2_scale)`` (see
    ``test_mega_moe_ep.local_weights``). Collective over the EP ``group`` (``rank`` /
    ``world_size`` are its rank / size) when torch.distributed is initialized and
    ``world_size > 1``; otherwise a single-process local MoE (all experts local).
    """
    distributed = world_size > 1 and dist.is_available() and dist.is_initialized()
    if distributed:
        x_all, token_counts = _all_gather_variable(x, group=group)
        weights_all, _ = _all_gather_variable(route_weights, token_counts, group)
        ids_all, _ = _all_gather_variable(ids, token_counts, group)
    else:
        x_all, weights_all, ids_all = x, route_weights, ids
        token_counts = [x.shape[0]]
    partial = torch.zeros(
        (x_all.shape[0], model_dim), dtype=torch.float32, device=x.device
    )
    w1_q, w1_scale, w2_q, w2_scale = ref_weights
    local_experts = experts // world_size
    expert_start = rank * local_experts
    active = torch.unique(ids_all)
    active = active[(active >= expert_start) & (active < expert_start + local_experts)]
    for expert in active.tolist():
        positions = torch.nonzero(ids_all == expert, as_tuple=False)
        rows, slots = positions[:, 0], positions[:, 1]
        lid = expert - expert_start
        w1 = _dequant_expert(w1_q[lid], w1_scale[lid], 2 * inter_dim, model_dim)
        w2 = _dequant_expert(w2_q[lid], w2_scale[lid], model_dim, inter_dim)
        inp = x_all[rows].float()
        gate = inp @ w1[:inter_dim].T
        up = inp @ w1[inter_dim:].T
        if act == "situv2":
            gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
            hidden = gate * up
        else:
            if swiglu_limit > 0:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(-swiglu_limit, swiglu_limit)
            hidden = F.silu(gate) * up
        out = (hidden @ w2.T) * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
    if distributed:
        dist.all_reduce(partial, group=group)
        start = sum(token_counts[:rank])
        return partial[start : start + x.shape[0]]
    return partial


@dataclass(frozen=True)
class LaunchCfg:
    """One EP kernel variant: row tile 16*mt; ``a8`` MXFP8 activations.  The payload is
    always received source-indexed (row = rank*mtpr + token, bounded by n_ranks*mtpr).
    """

    mt: int
    a8: bool = False
    small: bool = False  # small-batch kernel variant (mtpr <= SMALL_MAX_TOKENS)
    # small-batch only: non-temporal weight loads (m >= SMALL_STREAM_MIN)
    stream_weights: bool = False


class MegaMoeEPEngine:
    """Host side of the fused EP MoE (see the module docstring): owns the symmetric
    arena and the plan tables, binds the weights, and launches one grid=#CU, block=512
    kernel per ``forward``.  ``reference_forward`` / ``ep_reference`` are the pure-torch
    oracle."""

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
        w1,
        w1_scale,
        w2,
        w2_scale,
        quant="a4w4",
        act="silu",
        situ_beta=1.0,
        situ_linear_beta=1.0,
        swiglu_limit=0.0,
        group=None,
        device=None,
        weights_preshuffled=False,
    ):
        if quant not in ("a4w4", "a8w4"):
            raise ValueError("MegaMoeEP quant must be 'a4w4' or 'a8w4'")
        if not ep_shape_supported(model_dim, inter_dim, experts, world_size):
            raise ValueError(
                f"fused EP MegaMoE does not tile h{model_dim} i{inter_dim} "
                f"e{experts} np{world_size}"
            )
        mtpr = int(max_tok_per_rank)
        if mtpr <= 0 or mtpr & (mtpr - 1):
            raise ValueError(f"max_tok_per_rank={mtpr} must be a power of two")
        if not 0 < topk <= 16:
            raise ValueError(f"MegaMoeEP topk must be in [1, 16], got {topk}")
        if act not in ("silu", "situv2"):
            raise ValueError(f"MegaMoeEP act must be 'silu' or 'situv2', got {act!r}")
        if not float(swiglu_limit) >= 0:
            raise ValueError(f"swiglu_limit must be non-negative, got {swiglu_limit}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.H = int(model_dim)
        self.I = int(inter_dim)
        self.E = int(experts)
        self.epr = int(experts // world_size)
        self.K = int(topk)
        self.mtpr = mtpr
        self.stream_max = (
            SMALL_STREAM_MAX_SHORT_INTER if self.I <= SMALL_SHORT_INTER else mtpr
        )
        self.a8 = quant == "a8w4"
        self.act = act
        self.situ = (float(situ_beta), float(situ_linear_beta))
        self.swiglu_limit = float(swiglu_limit)
        self.group = group
        if device is None:
            device = torch.device("cuda", self.rank)
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, self.rank)
        self.device = device
        small = mtpr <= SMALL_MAX_TOKENS
        token_slots = self.world_size * mtpr
        a_row = self.H if self.a8 else self.H // 2
        # kernel_config's 32-bit offset bounds, checked before any compile
        if not (
            token_slots * a_row < 2**31
            and token_slots < 2**24
            and mtpr * self.K * (self.H + self.H // 8) < 2**31
            and self.epr * token_slots * 4 < 2**31
        ):
            raise ValueError(
                f"max_tok_per_rank={mtpr} x world_size={world_size} overflows the "
                "kernel's 32-bit buffer offsets"
            )
        # the GEMM A ring stages a_banks // 2 K-steps ahead
        if self.I // TILE_K < a_banks(small) // 2:
            raise ValueError(
                f"inter_dim={self.I} needs >= {a_banks(small) // 2} K-steps of "
                f"{TILE_K} at max_tok_per_rank={mtpr}"
            )
        lds = mega_moe_ep_consts(self.H, self.I, 4, self.epr, self.a8, small)
        if lds["LDS_BYTES"] > LDS_LIMIT:
            raise ValueError(
                f"h{self.H} i{self.I} epr{self.epr} needs {lds['LDS_BYTES']} B LDS at "
                f"mt=4 > {LDS_LIMIT}"
            )
        self._bind_weights(w1, w1_scale, w2, w2_scale, weights_preshuffled)
        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        if self.n_cta > MAX_CTAS:
            raise ValueError(f"{self.n_cta} CUs > MAX_CTAS={MAX_CTAS} (flag sizing)")
        self.gfx = getattr(props, "gcnArchName", "").split(":")[0]
        if self.gfx != "gfx950":
            raise ValueError(f"fused EP MegaMoE needs gfx950, not {self.gfx}")
        self._cfgs: dict = {}
        self._launchers: dict = {}
        self._armed = False
        # hipIpc symmetric arena: dispatch deposits each routed token's payload here on
        # the expert-owner rank (indexed layout, world_size*mtpr rows); combine deposits
        # the weighted GEMM2 partials back on the source rank; the epoch flags live in it
        # too. Reserved in the same order on every rank (collective commit maps peers).
        arena = SymmetricArena(group=group, device=self.device)
        if (arena.rank, arena.world_size) != (self.rank, self.world_size):
            raise ValueError(
                f"rank={self.rank} world_size={self.world_size} != the group's "
                f"{arena.rank} / {arena.world_size}"
            )
        self._payload = arena.reserve("payload", (token_slots, a_row), torch.uint8)
        self._pscale = arena.reserve("pscale", (token_slots, self.H // 32), torch.uint8)
        self._recv = arena.reserve("recv_expert", (token_slots * self.K,), torch.int32)
        # combine partials sent fp8 per-8-blockwise (half the a2a traffic): each slot
        # row is H fp8-e4m3 bytes then H/8 per-8 E8M0 scale bytes (co-located, one
        # buffer). small-batch: two banks by epoch parity (no barrier orders a launch's
        # part stores after the previous launch's combine on the source).
        nbank = 2 if self.mtpr <= SMALL_MAX_TOKENS else 1
        self._part = arena.reserve(
            "part", (nbank * self.mtpr * self.K, self.H + self.H // 8), torch.uint8
        )
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        arena.commit()
        self.arena = arena
        # small-batch: scrubbed words (SCRUBBED), never a launch's
        self._recv.local.fill_(-1)
        self.ctrl = torch.zeros(
            ctrl_ints(token_slots, self.K, self.epr),
            dtype=torch.int32,
            device=self.device,
        )
        # Epoch starts at 1: the arena ready-flags are zero-initialized and the waits are
        # serial-number ``flag >= epoch`` (wrap-safe ``before``), so epoch 0 would make
        # every dispatch / combine wait pass instantly (0 >= 0) and all cross-wave /
        # cross-rank ordering would collapse into a race.  It lives in ctrl (the kernel
        # reads and bumps it) so CUDA-graph replays each get a new one.
        self.ctrl[EPOCH_SLOT] = 1
        # GEMM scratch (local, NOT peer-visible): GEMM1 writes the E8M0-quantized
        # intermediate (inter / inter_scale), GEMM2 reads it.
        a_pack = 1 if self.a8 else 2
        # inter / inter_scale / chunk_payload_row / chunk_recv_slot are indexed by
        # ECHUNK id (0..nch) * TILE_M + row, not bid: sized for the worst-case routing
        # (inter_rows), so any skew fits.
        rows_cap = inter_rows(token_slots, self.K, self.epr)
        self._inter = torch.empty(
            rows_cap * (self.I // a_pack), dtype=torch.uint8, device=self.device
        )
        self._inter_scale = torch.empty(
            rows_cap * (self.I // 32), dtype=torch.uint8, device=self.device
        )
        # Per-echunk GEMM1 A-gather source keys written to global by the gather:
        # chunk_payload_row = payload row (recv_slot // topk) for the ATileLoader
        # indexed gather; chunk_recv_slot = the full recv_slot (the GEMM2 combine
        # scatter recovers route = idx % topk and the source rank/token from it -- read
        # by other CTAs' tiles, so it cannot stay in the per-CTA LDS_GATHER). Indexed
        # echunk*TILE_M + r.
        self._chunk_payload_row = torch.empty(
            rows_cap, dtype=torch.int32, device=self.device
        )
        self._chunk_recv_slot = torch.empty(
            rows_cap, dtype=torch.int32, device=self.device
        )
        # Balanced-gather tables: the planner's pre-pass assigns each received
        # (token,route) pair a GLOBAL per-local-expert position (0..R_e-1 rank via
        # _expert_count atomics) and buckets its recv slot at _expert_recv_slots[le*cap
        # + pos] (cap = world*mtpr: a token routes to an expert at most once). The
        # planner chunks an expert's rows by position//TILE_M, so every chunk but the
        # expert's last is a full TILE_M-row tile, and the gather reads it by position.
        # _expert_count is zeroed each forward; the small-batch kernel uses one EPR
        # block of it per XCD.
        self._expert_recv_slots = torch.empty(
            self.epr * self.world_size * self.mtpr,
            dtype=torch.int32,
            device=self.device,
        )
        self._expert_count = torch.zeros(
            self.epr * N_XCD, dtype=torch.int32, device=self.device
        )

    def _bind_weights(self, w1, w1_scale, w2, w2_scale, preshuffled=False):
        # Both GEMMs read their weights and E8M0 scales in the
        # shuffle_weight_a16w4 / shuffle_scale_a16w4 B-layout (W1 gate|up interleaved,
        # W2 down-proj), so natural weights (the tests / oracle) are shuffled host-side.
        want = (
            (w1, self.epr * 2 * self.I * self.H // 2),
            (w1_scale, self.epr * 2 * self.I * self.H // 32),
            (w2, self.epr * self.H * self.I // 2),
            (w2_scale, self.epr * self.H * self.I // 32),
        )
        for t, n in want:
            if t.device != self.device or t.numel() * t.element_size() != n:
                raise ValueError(
                    f"MegaMoeEP: weight {tuple(t.shape)} {t.dtype} on {t.device} is not "
                    f"a {n}-byte tensor on {self.device}"
                )
        if preshuffled:
            # Already in those layouts: bound by pointer, not copied; the
            # caller keeps them alive (and may update them in place, e.g. EPLB).
            for t, _ in want:
                if not t.is_contiguous():
                    raise ValueError(
                        f"MegaMoeEP: preshuffled weight {tuple(t.shape)} {t.dtype} is "
                        "not contiguous"
                    )
            self.w1, self.w1_scale, self.w2, self.w2_scale = w1, w1_scale, w2, w2_scale
            return
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        w1 = w1 if w1.is_contiguous() else w1.contiguous()
        self.w1 = shuffle_weight_a16w4(w1, 16, True).contiguous()
        self.w1_scale = shuffle_scale_a16w4(
            w1_scale.reshape(-1, self.H // 32).contiguous(), self.epr, True
        ).contiguous()
        # W2 + its scale: the down-proj (gate_up=False) B-layout
        w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2 = shuffle_weight_a16w4(w2, 16, False).contiguous()
        self.w2_scale = shuffle_scale_a16w4(
            w2_scale.reshape(-1, self.I // 32).contiguous(), self.epr, False
        ).contiguous()
        # Live as long as the instance: a graph captured with them keeps their
        # pointers after set_weights binds other weights.
        self._construction_weights = (self.w1, self.w1_scale, self.w2, self.w2_scale)

    def set_weights(self, w1, w1_scale, w2, w2_scale, preshuffled=False):
        # Launches and captured graphs keep the bound pointers, so a shuffled copy
        # made here would be freed while a graph still reads it.
        if not preshuffled:
            raise ValueError(
                "MegaMoeEP.set_weights binds by pointer: pass the weights in the "
                "kernel layout (shuffle_weight_a16w4 / shuffle_scale_a16w4) with "
                "preshuffled=True and keep them alive"
            )
        self._bind_weights(w1, w1_scale, w2, w2_scale, preshuffled=True)

    def _fit_mt(self, m, small):
        # The pipelined GEMM1 packs rows in pairs (MfmaScaleGU / AScaleLoader use
        # m_repeat//2 scale groups), so the row tile mt (=m_repeat) must be EVEN.
        # The inline-A GEMM stages A with all NT=512 threads (16 B/thread); a smaller A
        # tile (a4 mt=2: 32 x 128 B) has the surplus threads repeat a chunk.
        if small and m <= (SMALL_MT2_MAX if self.a8 else SMALL_MT2_MAX_A4):
            mt = 2
        else:
            mt = 8 if not self.a8 and m >= MT8_MIN_TOKENS else 4
        if mt % 2:
            raise ValueError(f"mt must be even, got mt={mt}")
        while mt > 4:
            if (
                mega_moe_ep_consts(
                    self.H,
                    self.I,
                    mt,
                    self.epr,
                    self.a8,
                    small,
                )["LDS_BYTES"]
                <= LDS_LIMIT
            ):
                break
            mt -= 4
        return mt

    def config(self, m):
        cfg = self._cfgs.get(m)
        if cfg is None:
            small = self.mtpr <= SMALL_MAX_TOKENS
            cfg = LaunchCfg(
                mt=self._fit_mt(m, small),
                a8=self.a8,
                small=small,
                stream_weights=small and SMALL_STREAM_MIN <= m <= self.stream_max,
            )
            self._cfgs[m] = cfg
        return cfg

    def _variants(self):
        """Every LaunchCfg a token count in [0, mtpr] selects (mt / stream_weights only
        change at the SMALL_STREAM_MIN / stream_max / SMALL_MT2_MAX / MT8_MIN_TOKENS
        bounds)."""
        th = SMALL_MT2_MAX if self.a8 else SMALL_MT2_MAX_A4
        wx = self.stream_max
        ms = {
            0,
            1,
            SMALL_STREAM_MIN,
            wx,
            wx + 1,
            th,
            th + 1,
            MT8_MIN_TOKENS - 1,
            MT8_MIN_TOKENS,
            self.mtpr,
        }
        return sorted(
            {self.config(m) for m in ms if 0 <= m <= self.mtpr}, key=lambda c: c.mt
        )

    def prepare(self):
        """Collective: compile every launch variant on every rank, then barrier.  The
        ranks of one launch may pick different variants (per-rank m), and a peer that
        JITs inside the launch keeps the others spinning in their cross-rank waits;
        forward() runs this on its first call (must not be under CUDA graph capture)."""
        if self._armed:
            return
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoeEP: first forward inside CUDA graph capture; call "
                    "prepare() (or run forward eagerly) first"
                )
            stream = torch.cuda.current_stream(self.device)
            for cfg in self._variants():
                _preload_compiled(
                    self._launcher(cfg),
                    *self._launch_args(0, None, None, None, None, None),
                    stream,
                )
            torch.cuda.synchronize(self.device)
        self.arena.barrier()
        self._armed = True

    def _launcher(self, cfg):
        fn = self._launchers.get(cfg)
        if fn is None:
            fn = compile_mega_moe_ep(
                H=self.H,
                I=self.I,
                TOPK=self.K,
                MT=cfg.mt,
                TOKEN_SLOTS=self.mtpr * self.world_size,
                E=self.E,
                EPR=self.epr,
                N_RANKS=self.world_size,
                act=self.act,
                situ_beta=self.situ[0],
                situ_linear_beta=self.situ[1],
                swiglu_limit=self.swiglu_limit,
                a8=cfg.a8,
                SMALL_BATCH=cfg.small,
                STREAM_WEIGHTS=cfg.stream_weights,
            )
            self._launchers[cfg] = fn
        return fn

    def reference_forward(self, x_bf16, wts, topk_ids, ref_weights):
        """Run the pure-torch oracle for these inputs. ``ref_weights`` are the
        unshuffled quantized weights ``(w1_q, w1_scale, w2_q, w2_scale)``."""
        return ep_reference(
            x_bf16,
            wts,
            topk_ids,
            ref_weights,
            rank=self.rank,
            world_size=self.world_size,
            model_dim=self.H,
            inter_dim=self.I,
            experts=self.E,
            swiglu_limit=self.swiglu_limit,
            act=self.act,
            situ_beta=self.situ[0],
            situ_linear_beta=self.situ[1],
            group=self.group,
        )

    def _launch_args(self, m, x, x_scale, topk_ids, topk_weights, out):
        """The kernel's positional args (minus stream): weights, local input, output,
        global ctrl, every rank's arena base, arena region offsets, scalars, grid (#CU).
        """
        arena_bases = [int(b) for b in self.arena.base_ptrs]
        arena_bases += [0] * (MAX_RANKS - self.world_size)

        def ptr(t):
            return 0 if t is None else t.data_ptr()

        return (
            self.w1.data_ptr(),
            self.w1_scale.data_ptr(),
            self.w2.data_ptr(),
            self.w2_scale.data_ptr(),
            ptr(x),
            ptr(x_scale),
            ptr(topk_ids),
            ptr(topk_weights),
            ptr(out),
            self.ctrl.data_ptr(),
            *arena_bases,
            self._payload.offset,
            self._pscale.offset,
            self._recv.offset,
            self._part.offset,
            self._flag.offset,
            self._inter.data_ptr(),
            self._inter_scale.data_ptr(),
            self._chunk_payload_row.data_ptr(),
            self._chunk_recv_slot.data_ptr(),
            self._expert_recv_slots.data_ptr(),
            self._expert_count.data_ptr(),
            self.rank,
            self.world_size,
            self.epr,
            m,
            self.mtpr,
            self.n_cta,
        )

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
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        # The kernel reads these by raw pointer.
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(x_bf16.shape) != (run_tokens, self.H) or any(
            tuple(t.shape) != (run_tokens, self.K) for t in (wts, topk_ids)
        ):
            raise ValueError(
                f"expected x_bf16 [{run_tokens}, {self.H}] and wts / topk_ids "
                f"[{run_tokens}, {self.K}], got {tuple(x_bf16.shape)}, "
                f"{tuple(wts.shape)}, {tuple(topk_ids.shape)}"
            )
        if any(t.device != self.device for t in (x_bf16, wts, topk_ids)):
            raise ValueError(f"x_bf16, wts and topk_ids must be on {self.device}")
        if stream is not None and stream.device != self.device:
            raise ValueError(f"stream is on {stream.device}, not {self.device}")
        del mask_invalid_slots  # -1 / >= experts top-k slots are always skipped in-kernel
        self.prepare()
        with torch.cuda.device(self.device):
            return self._forward(x_bf16, wts, topk_ids, stream, slice_output)

    __call__ = forward

    def _forward(self, x_bf16, wts, topk_ids, stream, slice_output):
        run_tokens = int(x_bf16.shape[0])
        cfg = self.config(run_tokens)
        fn = self._launcher(cfg)
        cur = torch.cuda.current_stream(self.device)
        if stream is None:
            stream = cur
        elif stream != cur:
            # the inputs come from the current stream; the caller waits on ``stream``
            # before reading the output
            stream.wait_stream(cur)
            if not torch.cuda.is_current_stream_capturing():
                for t in (x_bf16, wts, topk_ids):
                    t.record_stream(stream)
        # Dynamic MX quant of the activations (per_1x32_mx_quant); the scales are
        # natural [m, H/32], which dispatch copies verbatim into pscale.  small-batch:
        # small_send quantizes the bf16 rows itself (one launch less).  On the kernel
        # stream, so the launch is ordered after it.
        with torch.cuda.stream(stream):
            if cfg.small:
                x_q, x_scale = x_bf16, None
            elif run_tokens:
                x_q, x_scale = per_1x32_mx_quant(
                    x_bf16, quant_mode="fp8" if self.a8 else "fp4"
                )
            else:  # HIP rejects the quant's zero grid; the rank still launches (peers wait on it)
                x_q = torch.empty(
                    (0, self.H if self.a8 else self.H // 2),
                    dtype=torch.uint8,
                    device=self.device,
                )
                x_scale = torch.empty(
                    (0, self.H // 32), dtype=torch.uint8, device=self.device
                )
        # combine_final writes every row of [0, run_tokens); only the unsliced tail needs zeros
        with torch.cuda.stream(stream):
            out = torch.empty(
                (self.mtpr, self.H), dtype=torch.bfloat16, device=self.device
            )
            if not slice_output:
                out[run_tokens:].zero_()
        # recv_expert init (SENTINEL = epr, masked by the planner) on the kernel stream,
        # so it is ordered before the launch; the kernel's barrier_init orders it
        # cross-rank. No host sync / dist.barrier -> forward() is a pure stream
        # pipeline.
        with torch.cuda.stream(stream):
            # small-batch: senders write whole routing blocks; small_bucket re-zeroes
            # its counters
            if not cfg.small:
                self._recv.local.fill_(self.epr)
                # per-expert position counters for the balanced-gather pre-pass
                self._expert_count.zero_()
        _run_compiled(
            fn, *self._launch_args(run_tokens, x_q, x_scale, topk_ids, wts, out), stream
        )
        return out[:run_tokens] if slice_output else out

    # --- sticky error word / lifecycle ---
    def poll_errors(self):
        return int(self.ctrl[0].item())

    def error_flag(self):
        return self.ctrl[0:1]

    def clear_errors(self):
        self.ctrl[0] = 0

    def reset(self):
        ep = self.ctrl[EPOCH_SLOT].clone()  # the arena flags hold past epochs
        self.ctrl.zero_()
        self.ctrl[EPOCH_SLOT] = ep
        self._expert_count.zero_()
