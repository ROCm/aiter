# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end MoonEP EP with the tuned stages wired in.

Subclass rather than an edit: ``MoonEPBF16ReferenceEP`` and every kernel it
calls stay exactly as they are, so the two can be constructed side by side in
one process and A/B'd on the same inputs.  Each stage is an independent flag,
because they were validated independently and one of them (push combine) costs
real memory.

=====================  ==========================================  ===========
flag                   swaps in                                    validated by
=====================  ==========================================  ===========
``gpu_planner``        ``build_plan_gpu`` (fused single kernel)     bit-exact
                       for the 802 ms CPU reference planner         vs reference
``fast_epilogue``      ``moonep_dispatch_epilogue_fast``            byte-exact
``push_combine``       prologue + push + reduce, replacing the      torch ground
                       remote-read combine                          truth, 3.9e-3
``fast_prefetch``      ``moonep_weight_prefetch_fast``              byte-exact
=====================  ==========================================  ===========

Why push combine rather than a deeper combine pipeline
------------------------------------------------------
``moonep_link_probe`` measured this machine: remote reads are pinned at
230-240 GB/s across in-flight depth 1..32, 256..2048 blocks and every cache
policy, while remote writes reach 448 GB/s.  The stock combine already runs at
232 GB/s, i.e. 97% of the best read the probe ever saw, so no amount of
pipelining helps it.  Turning the reads into writes does.

Cost: ``push_combine`` needs a symmetric ``[S*K, H]`` staging buffer -- 940 MB
at S=8192 H=7168 K=8.  Only ``primaries`` rows of it are ever written (~43k of
65k at uniform routing), so a prefix sum over ``is_primary`` would compact it to
~620 MB; that is a plan-side change and is not done here.

Numerics: folding duplicates in the prologue rounds to bf16 once more than
summing all K in fp32 inside combine, so the push path is close to but not
bit-identical with the reference.  Compare with a tolerance.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
import torch.distributed as dist
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

from aiter.ops.flydsl.kernels.moonep_combine_fast import (
    make_moonep_combine_fast_jit,
)
from aiter.ops.flydsl.kernels.moonep_combine_prologue import (
    make_moonep_combine_prologue_jit,
)
from aiter.ops.flydsl.kernels.moonep_combine_push import (
    make_moonep_publish_src_slots_jit,
    make_moonep_push_rows_jit,
    make_moonep_reduce_local_jit,
)
from aiter.ops.flydsl.kernels.moonep_dispatch_epilogue_fast import (
    make_moonep_dispatch_epilogue_fast_jit,
)
from aiter.ops.flydsl.kernels.moonep_weight_prefetch_fast import (
    make_moonep_weight_prefetch_fast_jit,
)
from aiter.ops.flydsl.moonep import (
    MoonEPReferencePlan,
    build_plan_gpu,
)
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP


class _Launcher:
    """compile-then-call wrapper whose first invocation actually executes.

    The ops' own lazy pattern (``if compiled is None: compile(...)`` with the
    call only in the ``else`` branch) compiles without running; relying on it
    silently produced an all-zero combine reference in an earlier harness.
    """

    def __init__(self, jit, ptr_args, stream):
        self._raw = tuple(ptr_args) + (stream,)
        self._compiled = flyc.compile(
            jit, *(fx.Int64(p) for p in ptr_args), stream
        )

    def __call__(self):
        self._compiled(*self._raw)


class MoonEPBF16FastEP(MoonEPBF16ReferenceEP):
    """Reference EP with the measured-faster stage implementations enabled."""

    def __init__(
        self,
        config,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        gpu_planner: bool = True,
        fast_epilogue: bool = True,
        push_combine: bool = True,
        fast_prefetch: bool = True,
        combine_block_num: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(config, hidden_dim, intermediate_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.use_gpu_planner = gpu_planner
        self.use_push_combine = push_combine
        self._fast_closed = False
        op = self.dispatch_op
        cfg = self.config
        device = op.device
        stream = torch.cuda.current_stream(device)
        nvs = cfg.num_dispatch_rows
        self._n_entries = cfg.num_tokens * cfg.top_k

        if fast_epilogue:
            # Same launch ABI, so swapping the JIT and dropping the cached
            # compile is enough; the op recompiles on its next epilogue.
            op._epilogue_jit = make_moonep_dispatch_epilogue_fast_jit(
                hidden_dim=hidden_dim,
                num_dispatch_rows=nvs,
                num_groups=cfg.num_experts + int(cfg.prefetch_slots),
                block_num=op.block_num,
                warp_num_per_block=op.warp_num_per_block,
            )
            op._epilogue_compiled = None

        if fast_prefetch:
            for wop in (self.gate_op, self.up_op, self.down_op):
                wop._jit = make_moonep_weight_prefetch_fast_jit(
                    experts_per_rank=wop.experts_per_rank,
                    prefetch_slots=wop.prefetch_slots,
                    weight_numel=wop.weight_numel,
                    block_num=wop.block_num,
                    block_threads=wop.block_threads,
                )
                wop._compiled = None

        if not push_combine:
            return

        # Symmetric buffers for the push path.  Allocated in the same order on
        # every PE, as MORI SHMEM requires.
        self._src_slot = mori_shmem_create_tensor((nvs,), torch.int32)
        self._staging = mori_shmem_create_tensor(
            (self._n_entries, hidden_dim), torch.bfloat16
        )
        self._src_slot.fill_(-1)
        torch.cuda.synchronize(device)
        ms.shmem_barrier_all()
        self._peer_src_slot_ptrs = torch.tensor(
            [
                ms.shmem_ptr_p2p(self._src_slot.data_ptr(), cfg.rank, p)
                for p in range(cfg.world_size)
            ],
            dtype=torch.int64,
            device=device,
        )
        self._peer_staging_ptrs = torch.tensor(
            [
                ms.shmem_ptr_p2p(self._staging.data_ptr(), cfg.rank, p)
                for p in range(cfg.world_size)
            ],
            dtype=torch.int64,
            device=device,
        )
        self._dup_count = torch.zeros(nvs, dtype=torch.int32, device=device)
        self._dup_list = torch.zeros(
            nvs * (cfg.top_k - 1), dtype=torch.int32, device=device
        )

        self._prologue = _Launcher(
            make_moonep_combine_prologue_jit(
                hidden_dim=hidden_dim,
                num_dispatch_rows=nvs,
                top_k=cfg.top_k,
                block_num=min(combine_block_num, 1024),
            ),
            (
                # combine reads expert_output (moonep_dispatch_op.py:104 maps
                # peer_expert_output_ptrs there), so that is what the prologue
                # must fold duplicates into.
                op.expert_output.data_ptr(),
                op.recv_duplicate_src.data_ptr(),
                self._dup_count.data_ptr(),
                self._dup_list.data_ptr(),
            ),
            stream,
        )
        self._push = _Launcher(
            make_moonep_push_rows_jit(
                hidden_dim=hidden_dim,
                num_dispatch_rows=nvs,
                num_tokens=cfg.num_tokens,
                top_k=cfg.top_k,
                block_num=combine_block_num,
            ),
            (
                op.expert_output.data_ptr(),
                self._src_slot.data_ptr(),
                self._peer_staging_ptrs.data_ptr(),
            ),
            stream,
        )
        self._publish_jit = make_moonep_publish_src_slots_jit(
            num_tokens=cfg.num_tokens,
            top_k=cfg.top_k,
            num_dispatch_rows=nvs,
            rank=cfg.rank,
            block_num=min(combine_block_num, 256),
        )
        self._reduce_jit = make_moonep_reduce_local_jit(
            num_tokens=cfg.num_tokens,
            hidden_dim=hidden_dim,
            top_k=cfg.top_k,
            num_dispatch_rows=nvs,
        )
        # Both take plan.dst, which is only known per call, so they are bound
        # lazily and cached against the plan's buffer address.
        self._bound_dst_ptr = None
        self._publish = None
        self._reduce = None

    # ------------------------------------------------------------------
    def _bind_plan(self, plan: MoonEPReferencePlan) -> None:
        """(Re)build the two plan-dependent launchers when ``dst`` moves."""

        dst_ptr = plan.dst.data_ptr()
        if self._bound_dst_ptr == dst_ptr:
            return
        op = self.dispatch_op
        stream = torch.cuda.current_stream(op.device)
        self._publish = _Launcher(
            self._publish_jit,
            (dst_ptr, self._peer_src_slot_ptrs.data_ptr()),
            stream,
        )
        self._reduce = _Launcher(
            self._reduce_jit,
            (
                dst_ptr,
                self._staging.data_ptr(),
                op.peer_weight_ptrs.data_ptr(),
                op.combine_output.data_ptr(),
                op.gathered_route_weights.data_ptr(),
            ),
            stream,
        )
        self._bound_dst_ptr = dst_ptr

    def publish_plan(self, plan: MoonEPReferencePlan) -> None:
        """Publish this plan's reverse map into every peer's ``src_slot``.

        Must run whenever the plan changes.  ``src_slot`` is cleared first
        because only primary entries write, so a slot left over from a previous
        plan would push a row that no longer exists; the barrier in between
        stops a fast rank from publishing into a peer that has not cleared yet.
        """

        op = self.dispatch_op
        stream = torch.cuda.current_stream(op.device)
        self._bind_plan(plan)
        self._src_slot.fill_(-1)
        ms.shmem_barrier_on_stream(stream)
        self._publish()
        ms.shmem_barrier_on_stream(stream)

    # ------------------------------------------------------------------
    def dispatch(
        self,
        hidden_sh: torch.Tensor,
        route_weights_sk: torch.Tensor | None = None,
        topk_experts_sk: torch.Tensor | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        plan: MoonEPReferencePlan | None = None,
        *,
        zero_copy: bool = False,
    ):
        """Same contract as the reference, with the GPU planner substituted."""

        if plan is None and self.use_gpu_planner:
            if topk_experts_sk is None or tokens_per_expert is None:
                raise ValueError("routing inputs are required when plan is absent")
            if tokens_per_expert.ndim == 1:
                gathered = [
                    torch.empty_like(tokens_per_expert)
                    for _ in range(self.config.world_size)
                ]
                dist.all_gather(gathered, tokens_per_expert)
                tokens_per_expert = torch.stack(gathered)
            # The planner owns its output buffers and reuses them on the next
            # call for the same config, which is fine here because the plan is
            # consumed before the next build.
            plan = build_plan_gpu(self.config, topk_experts_sk, tokens_per_expert)
        out = super().dispatch(
            hidden_sh,
            route_weights_sk,
            topk_experts_sk,
            tokens_per_expert,
            plan,
            zero_copy=zero_copy,
        )
        if self.use_push_combine:
            self.publish_plan(out[3])
        return out

    # ------------------------------------------------------------------
    def combine(
        self,
        plan: MoonEPReferencePlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None = None,
        *,
        zero_copy: bool = False,
    ):
        """Push-based combine: remote writes home, then a local reduction."""

        if not self.use_push_combine:
            return super().combine(
                plan, hidden_nvsh, route_weights_nvs, zero_copy=zero_copy
            )

        op = self.dispatch_op
        if hidden_nvsh.data_ptr() != op.expert_output.data_ptr():
            raise ValueError(
                "push combine reads expert_output in place; pass it directly "
                "rather than a copy"
            )
        self._bind_plan(plan)
        stream = torch.cuda.current_stream(op.device)
        # Fold duplicates locally so only primaries cross the link, exactly as
        # upstream's combine_prologue does (MoonEP/moonep/combine.py:320).
        self._prologue()
        self._push()
        ms.shmem_barrier_on_stream(stream)
        self._reduce()
        return (
            op.combine_output.clone(),
            op.gathered_route_weights.clone()
            if route_weights_nvs is not None
            else None,
            None,
        )

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        """Reference forward, but combine consumes ``expert_output`` in place.

        The reference copies ``expert_output`` into ``recv_hidden`` before
        combining, which is dead weight now that ``peer_expert_output_ptrs``
        maps to ``expert_output``: 117 MB of local traffic per step that nothing
        reads.  Push combine reads ``expert_output`` directly.
        """

        return super().forward(*args, **kwargs)

    def close(self) -> None:
        if not self._fast_closed and self.use_push_combine:
            torch.cuda.synchronize(self.dispatch_op.device)
            ms.shmem_barrier_all()
            mori_shmem_free_tensor(self._staging)
            mori_shmem_free_tensor(self._src_slot)
            self._fast_closed = True
        super().close()


__all__ = ["MoonEPBF16FastEP"]
