# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus fp8 MLA decode on flat-row plans (gfx950).

q/o rows are ``token * H + head``; the host planner cuts each request into 128-row chunks and its
KV into pieces balanced over the XCDs and their workgroups, stage 1 runs the pieces, and a merge
combines the split chunks' bf16 partials. Built for MTP decode, where a request's tokens and heads
pack into whole workgroups at any head count (96 heads under 8-way round-robin CP, 12 or 16 without).

Usage, once per decode step (the plan is built on the CPU, outside any CUDA graph)::

    dec = OpusMlaFlatDecode(max_batch=128)
    dec.plan(kv_indptr_host, q_len, num_heads, cp_world_size)
    lse = dec(q, kv, out, qo_indptr, kv_indptr, kv_indices, sm_scale, q_scale, kv_scale,
              g_kv_indptr, cp_world_size, cp_rank)

The plan and partial buffers keep fixed addresses, so a graph captured around ``__call__`` replays
with every later plan of up to ``max_batch`` requests.

``q_len`` is either one token count for the whole batch or a per-request array, so a step of
adaptive verification carries each request's own query count instead of a uniform rectangle. The
kernel already reads the causal span from ``qo_indptr`` and takes its rows from the work item, so
only the planner distinguishes the two.

Requirements: fp8 Q / KV with per-tensor descales, page_size 1, one KV head, a KV cache below
4 GiB, causal decode, and at most ``128 * max_chunks`` rows (``q_len * num_heads``) for the widest
request. A plan that needs more than ``item_cap`` work items (1024 by default; 128 heads x 8
tokens x batch 128 under CP needs 1216) makes ``plan`` raise a ``ValueError`` naming both numbers.

When to use it. Speedup over the faster of the two ``mla_decode_fwd`` paths (default asm or
``AITER_MLA_USE_OPUS=1``), on MI355X at 66k tokens of context per request and batch 1-64:

- round-robin CP, requests of 96 rows or more (``q_len * num_heads``): 0.9-1.2x at batch 1-8
  and 1.0-1.4x from batch 16 when the request is a single chunk; with 2+ chunks 0.9-1.6x at
  batch 1-4 (2.0-2.8x at 768 rows) and 1.0-2.2x from batch 8;
- no CP, 12 or 16 heads, 2+ tokens per request: 0.7-1.0x at batch 1-2 and 0.9-1.2x from batch
  4 (on par at 16 heads with 3-4 tokens);
- no CP, one token per request: 0.6-0.9x at batch 1-4 and 0.9-1.1x from batch 8;
- round-robin CP, requests under 96 rows (a few tokens at 12-64 heads): 0.55-1.15x up to
  batch 16, 0.75-1.15x at batch 32 and 0.9-1.2x at batch 64.

So keep ``mla_decode_fwd`` without CP at batch 1-2, at 16 heads with 3-4 tokens and with one
token per request, and under round-robin CP for requests under 96 rows below batch 32 (64-72
rows) to batch 64 (12-16 rows).

Toolchain: the kernel keeps O in all 256 AccVGPRs and issues its PV MFMAs as inline asm, so the
module is built with ``--amdgpu-mfma-vgpr-form`` to keep the compiler's own MFMA results out of
the AccVGPRs. Validated with ROCm 7.2.4 and ROCm 10.1.
"""

import numpy as np
import torch

from .mla import opus_mla_decode_flat_fwd, opus_mla_decode_flat_plan

SLOT_ROWS = 128  # rows per chunk and per partial slot
NUM_XCD = 8


class OpusMlaFlatDecode:
    """Persistent plan, partial buffers and planner state of the flat-row decode on one device.

    The plan spreads the work over every workgroup. With round-robin CP (``cp_world_size > 1``)
    the merge loads 4 partials per round trip; without CP the planner lets only an XCD's last
    workgroup absorb a remainder, and the merge loads 16 partials per round trip, for the few rows
    and many splits per chunk of a small batch at 12 or 16 heads.
    """

    def __init__(
        self,
        max_batch: int,
        max_chunks: int = 8,
        item_cap: int = 1024,
        num_workers: int | None = None,
        device: torch.device | None = None,
    ):
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "create OpusMlaFlatDecode outside graph capture"
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        if num_workers is None:
            num_workers = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
        if num_workers % NUM_XCD:
            raise ValueError(
                f"num_workers {num_workers} is not a multiple of {NUM_XCD}"
            )
        self.max_batch = max_batch
        self.max_chunks = max_chunks
        self.num_workers = num_workers
        self.item_cap = item_cap
        self.max_units = max_batch * max_chunks
        n_wip = num_workers + 1
        n_wis = item_cap * 8
        self.plan_ints = n_wip + n_wis + self.max_units * 4
        i32 = {"dtype": torch.int32, "device": self.device}
        self.plan_buf = torch.zeros(self.plan_ints, **i32)
        self.work_indptr = self.plan_buf[:n_wip]
        self.work_info = self.plan_buf[n_wip : n_wip + n_wis]
        self.unit_info = self.plan_buf[n_wip + n_wis :]
        self.o_accum = torch.empty(
            item_cap * SLOT_ROWS, 512, dtype=torch.bfloat16, device=self.device
        )
        self.lse_accum = torch.empty(
            item_cap * SLOT_ROWS, dtype=torch.float32, device=self.device
        )
        # Two pinned staging buffers: one is rewritten while the other's copy may be in flight.
        # The planner's buffers are host memory whatever the caller's default device is.
        self._host = [
            torch.zeros(self.plan_ints, dtype=torch.int32, device="cpu").pin_memory()
            for _ in range(2)
        ]
        self._events = [torch.cuda.Event(), torch.cuda.Event()]
        self._busy = [False, False]
        self._stats = torch.zeros(2, dtype=torch.int32, device="cpu")
        self._step = 0
        self.last = {}

    def _staging(self) -> torch.Tensor:
        slot = self._step & 1
        if self._busy[slot]:
            self._events[slot].synchronize()
        return self._host[slot]

    def _publish(self, host: torch.Tensor) -> None:
        slot = self._step & 1
        self.plan_buf.copy_(host, non_blocking=True)
        self._events[slot].record()
        self._busy[slot] = True
        self._step += 1

    def plan(
        self,
        kv_indptr: np.ndarray | torch.Tensor,
        q_len: int | np.ndarray | torch.Tensor,
        num_heads: int,
        cp_world_size: int = 1,
        bucketed: bool = True,
    ) -> dict:
        """Builds this step's plan from the host rank-local ``kv_indptr`` [B+1] and queues its copy
        to the GPU on the current stream. Returns {bs, items, units}.

        ``q_len`` is either one token count for the whole batch or a per-request [B] array (a [B+1]
        indptr is also accepted), which is what adaptive verification produces: a decode step then
        carries per-request query counts instead of a uniform rectangle. ``bucketed`` groups each
        XCD's requests by chunk depth so no worker idles above a shallow request; it has no effect
        when every request is the same width.
        """
        kv = torch.as_tensor(np.asarray(kv_indptr), dtype=torch.int32, device="cpu")
        if not kv.is_contiguous():
            kv = kv.contiguous()
        bs = kv.numel() - 1
        if bs > self.max_batch:
            raise ValueError(f"batch {bs} > max_batch {self.max_batch}")

        qo, scalar_q = None, 0
        if isinstance(q_len, (int, np.integer)):
            scalar_q = int(q_len)
            max_q = scalar_q
        else:
            lens = np.asarray(q_len, dtype=np.int64).ravel()
            if lens.size == bs + 1 and lens[0] == 0 and np.all(np.diff(lens) >= 0):
                lens = np.diff(lens)  # already an indptr
            if lens.size != bs:
                raise ValueError(f"q_len has {lens.size} entries, batch is {bs}")
            if np.any(lens < 0):
                raise ValueError(f"q_len is negative: {lens.tolist()}")
            if np.all(lens == lens[:1]):
                scalar_q = int(lens[0])  # uniform: the scalar path plans it identically
                max_q = scalar_q
            else:
                indptr = np.zeros(bs + 1, dtype=np.int32)
                np.cumsum(lens, out=indptr[1:], dtype=np.int32)
                qo = torch.from_numpy(indptr)
                max_q = int(lens.max(initial=0))
        if max_q * num_heads > SLOT_ROWS * self.max_chunks:
            raise ValueError(
                f"{max_q} tokens x {num_heads} heads > {SLOT_ROWS * self.max_chunks} rows"
            )

        host = self._staging()
        try:
            opus_mla_decode_flat_plan(
                kv,
                qo,
                scalar_q,
                num_heads,
                self.num_workers,
                cp_world_size == 1,
                bucketed,
                NUM_XCD,
                self.item_cap,
                self.max_units,
                host,
                self._stats,
            )
        except RuntimeError as err:
            # Publish an empty plan, so a later __call__ cannot run the previous step's plan
            # on this step's tensors.
            host.zero_()
            self._publish(host)
            self.last = {"bs": bs, "items": 0, "units": 0}
            if "item_cap" in str(err):
                raise ValueError(
                    f"batch {bs} x {max_q} tokens x {num_heads} heads (cp_world_size "
                    f"{cp_world_size}) needs {int(self._stats[0])} work items, item_cap is "
                    f"{self.item_cap}: build OpusMlaFlatDecode with a larger item_cap, or run "
                    "this step through mla_decode_fwd"
                ) from err
            raise
        self._publish(host)
        items, units = self._stats.tolist()
        self.last = {"bs": bs, "items": items, "units": units}
        return self.last

    def clear(self) -> None:
        """Publishes a plan with no work, for a capture warm-up before any step plan exists: stage
        1 and the merge then launch and exit."""
        host = self._staging()
        host.zero_()
        self._publish(host)

    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
        q_scale: torch.Tensor,
        kv_scale: torch.Tensor,
        g_kv_indptr: torch.Tensor | None = None,
        cp_world_size: int = 1,
        cp_rank: int = 0,
        lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs the last published plan: writes ``out`` [total_q, H, 512] bf16 and returns the LSE
        [total_q, H] fp32 (into ``lse`` when given)."""
        if lse is None:
            lse = torch.empty(
                q.shape[0], q.shape[1], dtype=torch.float32, device=q.device
            )
        opus_mla_decode_flat_fwd(
            q,
            kv.view(-1, kv.shape[-1]),
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.work_indptr,
            self.work_info,
            self.unit_info,
            self.o_accum,
            self.lse_accum,
            out,
            lse,
            q_scale,
            kv_scale,
            sm_scale,
            True,
            g_kv_indptr,
            cp_world_size,
            cp_rank,
            cp_world_size == 1,
        )
        return lse
