# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

"""Plan/quant and standalone or S1F pull dispatch resources."""

import logging
import os

import flydsl.expr as fx
import torch

from ...moe_apreshuffled_quant_scatter import validate_apre_k
from ...tensor_shim import _preload_compiled, _run_compiled, ptr_arg
from ..compact_plan import compact_plan_blocks, compile_tdm_compact_plan
from ..dispatch_pull import compile_pull_dispatch
from ..s1f_ctl import _QUANT_FUSE_BLOCKS, read_s1f_mode
from .base import BatchLayout, ExpertGrouping, _DispatchBatch


def enabled(config):
    """Opt into pull; S1F_PULL selects the fused producer/consumer path."""
    return os.environ.get("AITER_MEGA_DISPATCH_PULL", "0") == "1"


def check(config):
    """Reject unsupported pull wire/layout combinations before allocation."""
    if not enabled(config):
        return
    packed = os.environ.get("AITER_MEGA_PULL_PLAN_PACK", "0") == "1"
    source_shuffle = os.environ.get("AITER_MEGA_PULL_SRC_SHUFFLE", "0") == "1"
    if source_shuffle and not packed:
        raise ValueError("source-rank shuffle requires packed pull")
    if packed and (
        os.environ.get("AITER_MEGA_S1F_PULL", "0") != "1"
        or os.environ.get("AITER_MOE_S1F_READY_FIRST", "0") == "1"
        or os.environ.get("AITER_MOE_S1F_PRODUCER_GROUP", "1") != "1"
    ):
        raise ValueError("packed pull requires external-ready S1F group1")
    if config.dispatch_wire != "fp4":
        raise ValueError("AITER_MEGA_DISPATCH_PULL requires FP4 dispatch")
    if os.environ.get("AITER_MEGA_S1F_PULL", "0") == "1":
        group = int(os.environ.get("AITER_MOE_S1F_PRODUCER_GROUP", "1"))
        if group not in (1, 4):
            raise ValueError("AITER_MOE_S1F_PRODUCER_GROUP must be 1 or 4")
        if group == 4 and int(os.environ.get("AITER_MEGA_S1F_PULL_PRODUCERS", "64")) % 4:
            raise ValueError("four-TG producers require producers % 4 == 0")
        if not config.stage2_fused:
            raise ValueError("pull S1F requires fused-scatter combine")
        if not read_s1f_mode():
            raise ValueError("AITER_MEGA_S1F_PULL requires AITER_MEGA_S1F=1")
        if os.environ.get("AITER_MEGA_PULL_TDM_GATHER", "0") != "1":
            raise ValueError("pull S1F requires AITER_MEGA_PULL_TDM_GATHER=1")
    validate_apre_k(config.hidden_dim)
    if config.geometry.inter_dim % 128:
        raise ValueError("pull preshuffle requires intermediate % 128 == 0")
    if _QUANT_FUSE_BLOCKS <= 0:
        raise ValueError("pull requires AITER_MEGA_QUANT_FUSE > 0")


def regions(config, geometry):
    """Extra symmetric source storage; destination rowmap is shared with push."""
    if not enabled(config):
        return {}
    from ..s1f_stage1 import FLAG_M, FLAG_STRIDE

    extra = {
        "pull_wire": config.max_tokens_per_rank * geometry.wire_row,
        "pull_hist": geometry.hist_stride * 4,
        "pull_done": 4,
        "pull_ready": 4,
    }
    if os.environ.get("AITER_MEGA_PULL_PLAN_PACK", "0") == "1":
        frame_bytes = 16 * (config.hidden_dim // 2 + config.hidden_dim // 32)
        extra["pull_packed"] = config.world_size * ((geometry.cap + 15) // 16) * frame_bytes
        extra["pull_pack_done"] = (64 + 128 * 32) * 4
    if os.environ.get("AITER_MEGA_S1F_PULL", "0") == "1":
        if os.environ.get("AITER_MOE_S1F_READY_FIRST", "0") == "1":
            extra["pull_ready"] = 12
        extra["pull_tile_flag"] = (geometry.cap + FLAG_M - 1) // FLAG_M * FLAG_STRIDE
    return extra


class PullDispatch:
    """Reuse the existing planner and supply its result to receiver-side pull."""

    def __init__(self, dispatch):
        self.dispatch = dispatch
        config, geo = dispatch.config, dispatch.compact
        self.config = config
        self.tdm_gather = os.environ.get("AITER_MEGA_PULL_TDM_GATHER", "0") == "1"
        self.fused = os.environ.get("AITER_MEGA_S1F_PULL", "0") == "1"
        self.producer_group = int(os.environ.get("AITER_MOE_S1F_PRODUCER_GROUP", "1"))
        self.tile_ready = os.environ.get("AITER_MOE_S1F_TILE_READY", "1") == "1"
        self.ready_first = self.fused and os.environ.get(
            "AITER_MOE_S1F_READY_FIRST", "0"
        ) == "1"
        self.cluster_local = os.environ.get(
            "AITER_MOE_S1F_CLUSTER_LOCAL", "0"
        ) == "1"
        if self.cluster_local and not self.ready_first:
            raise ValueError("cluster-local S1F requires READY_FIRST=1")
        self.ready_queue = os.environ.get("AITER_MOE_S1F_READY_QUEUE", "0") == "1"
        if self.producer_group == 4 and (
            not self.fused or self.cluster_local or self.ready_queue
        ):
            raise ValueError("four-TG producers require the original pull S1F consumer")
        if self.cluster_local and self.ready_queue:
            raise ValueError("cluster-local S1F cannot use the global ready queue")
        if self.ready_queue and not self.ready_first:
            raise ValueError("ready queue requires pull S1F with READY_FIRST=1")
        self.src_rank_shuffle = os.environ.get("AITER_MEGA_PULL_SRC_SHUFFLE", "0") == "1"
        self.packed = os.environ.get("AITER_MEGA_PULL_PLAN_PACK", "0") == "1"
        if self.packed and (not self.fused or self.ready_first or self.producer_group != 1):
            raise ValueError("experimental packed pull requires external-ready S1F group1")
        if self.src_rank_shuffle and not self.packed:
            raise ValueError("source shuffle requires packed pull")
        self.arena = dispatch.res.arena
        if self.packed:
            self.arena.zero("pull_pack_done")
        device = dispatch.res.total_recv.device
        routes = geo.max_routes
        segments = config.world_size * config.experts_per_rank
        self.tok_map = torch.empty(routes, dtype=torch.int32, device=device)
        self.block_hist = torch.empty(16 * segments, dtype=torch.int32, device=device)
        self.send_base = torch.empty(
            segments, dtype=torch.int32, device=device,
        )
        self.masked_m = torch.empty(
            config.experts_per_rank, dtype=torch.int32, device=device
        )
        self.psum = torch.empty_like(self.masked_m)
        self.barrier = torch.zeros(4, dtype=torch.int32, device=device)
        self.payload = torch.empty(
            (geo.cap, config.hidden_dim // 2), dtype=torch.uint8, device=device
        )
        self.scales = torch.empty(
            (geo.cap, config.hidden_dim // 32), dtype=torch.uint8, device=device
        )
        self.wire_ptr = self.arena.local_ptr("pull_wire")
        self._launches = {}
        self._fused_launches = {}
        self.affinity_heads = None
        if self.ready_first:
            from .. import ready_queue

            words = ready_queue.AFFINITY_WORDS
            if self.ready_queue:
                words += ready_queue.storage_words((geo.cap + 63) // 64)
            self.affinity_heads = torch.empty(words, dtype=torch.int32, device=device)
        self.tile_info = None
        self.ctl = None
        if self.fused:
            from ..s1f_stage1 import FLAG_M, s1f_ctl_words

            tile_cap = (geo.cap + FLAG_M - 1) // FLAG_M
            self.tile_storage = torch.empty(
                tile_cap * (2 if self.ready_first else 1),
                dtype=torch.int32, device=device,
            )
            self.tile_info = self.tile_storage[:tile_cap]
            self.tile_local_chunks = self.tile_storage[tile_cap:]
            self.ctl = torch.zeros(
                s1f_ctl_words(npes=config.world_size, tile_cap=tile_cap),
                dtype=torch.int32,
                device=device,
            )

    def _plan_args(self, ids, hidden, weights, tokens, stream):
        return (
            self.arena.handle,
            ids,
            self.tok_map.data_ptr(),
            self.block_hist.data_ptr(),
            self.send_base.data_ptr(),
            self.masked_m.data_ptr(),
            self.psum.data_ptr(),
            self.barrier.data_ptr(),
            self.config.rank,
            tokens,
            hidden,
            self.wire_ptr,
            weights,
            self.tile_storage.data_ptr() if self.ready_first else 0,
            self.affinity_heads.data_ptr() if self.ready_first else 0,
            stream,
        )

    def _pull_args(self, rows, stream):
        return (
            self.arena.handle,
            self.psum,
            self.barrier,
            self.payload.view(torch.int32).reshape(-1),
            self.scales.view(torch.int32).reshape(-1),
            rows,
            stream,
        )

    def warmup(self, step, token_count):
        """Compile this alignment before graph capture, without GPU launches."""
        key = (
            step.align_m,
            compact_plan_blocks(token_count * self.config.topk),
        )
        if key in self._launches:
            return self._launches[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the pull dispatch token bucket before capture")
        config, geo, arena = self.config, self.dispatch.compact, self.arena
        plan = compile_tdm_compact_plan(
            rank=config.rank,
            npes=config.world_size,
            experts_per_rank=config.experts_per_rank,
            topk=config.topk,
            tile_m=step.align_m,
            compact_cap=geo.cap,
            off_hist=arena.offset("pull_hist"),
            off_done=arena.offset("pull_done"),
            hist_stride=geo.hist_stride,
            max_routes=geo.max_routes,
            hist_pingpong=False,
            plan_blocks=key[1],
            quant_blocks=_QUANT_FUSE_BLOCKS,
            quant_model_dim=config.hidden_dim,
            quant_row_stride=geo.wire_row,
            quant_payload_bytes=config.dispatch_token_nbytes,
            pull=True,
            write_tok_map=self.packed,
            pull_pack=self.packed,
            pull_src_shuffle=self.src_rank_shuffle,
            off_packed=arena.offset("pull_packed") if self.packed else 0,
            off_pack_done=arena.offset("pull_pack_done") if self.packed else 0,
            off_pull_map=arena.offset("ep_rowmap"),
            pull_ready_first=self.ready_first,
            pull_ready_queue=self.ready_queue,
            off_pull_ready=arena.offset("pull_ready"),
        )
        pull = compile_pull_dispatch(
            rank=config.rank,
            world=config.world_size,
            experts=config.experts_per_rank,
            max_routes=geo.max_routes,
            model_dim=config.hidden_dim,
            wire_row=geo.wire_row,
            off_wire=arena.offset("pull_wire"),
            off_ready=arena.offset("pull_ready"),
            off_rowmap=arena.offset("ep_rowmap"),
            align_m=step.align_m,
            topk=config.topk,
            tdm_gather=self.tdm_gather,
        )
        stream = fx.Stream(torch.cuda.current_stream())
        _preload_compiled(plan, *self._plan_args(0, 0, 0, token_count, stream))
        _preload_compiled(pull, *self._pull_args(step.rows, stream))
        self._launches[key] = (plan, pull)
        return plan, pull

    def run(self, hidden, weights, ids):
        """Run plan/quant and pull in caller stream order, matching push S1F."""
        dispatch, step = self.dispatch, self.dispatch._step
        tokens = int(hidden.shape[0])
        plan, pull = self.warmup(step, tokens)
        # Main runs plans in stream order. Any prefetched ordinary plan has
        # completed before this plan reuses ep_rowmap; it cannot supply the
        # quantized source wire needed by pull.
        dispatch._plan_pending = False
        _run_compiled(
            plan,
            *self._plan_args(
                ids.data_ptr(),
                hidden.data_ptr(),
                weights.data_ptr(),
                tokens,
                fx.Stream(torch.cuda.current_stream()),
            ),
        )
        if not self.fused:
            _run_compiled(
                pull,
                *self._pull_args(step.rows, fx.Stream(torch.cuda.current_stream())),
            )
        return _DispatchBatch(
            x=self.payload[: step.rows],
            topk_weights=dispatch._dummy_wts,
            topk_ids=dispatch._dummy_ids,
            scales=self.scales[: step.rows],
            num_local_tokens=None,
            routing=dispatch._routing(tokens),
            layout=BatchLayout.EXPERT_GROUPED,
            grouping=ExpertGrouping(
                masked_m=self.masked_m,
                psum=self.psum,
                ep_rowmap=dispatch._ep_rowmap,
                wire_row_stride=0,
                recv_bound=step.recv_bound,
                align_m=step.align_m,
                rows=step.rows,
                a_preshuffled=True,
            ),
        )

    def gemm1(self, out, a, w, a_scales, w_scales, m_tile_map, **kw):
        """Run fused producers/consumers, with optional in-kernel readiness."""
        from aiter.jit.utils.chip_info import get_cu_num
        from ..s1f_pull import (
            compile_s1f_pull, pull_affinity_geometry, resolve_pull_cluster,
        )
        from ..s1f_producer_group import validate_producer_group

        if kw.get("bias") is not None or not kw.get("stage1_quant_out"):
            raise ValueError("pull S1F requires bias-free fused activation/quant")
        if not kw.get("a_preshuffled") or not kw.get("row_major_ascale"):
            raise ValueError("pull S1F requires preshuffled A and row-major scales")
        config, step = self.config, self.dispatch._step
        producers = int(os.environ.get("AITER_MEGA_S1F_PULL_PRODUCERS", "64"))
        from aiter.ops.flydsl.grouped_gemm_mxfp4 import (
            _select_bool_env,
            _select_cluster_m,
            _select_cluster_n,
            _select_num_waves_per_tensor_tdm,
            _select_tdm_b_th,
        )

        if int(kw.get("k_waves", 1)) != 1:
            raise ValueError("pull S1F does not support split-K wave groups")
        if int(kw.get("tdm_as_in_prologue", 0)):
            raise ValueError("pull S1F must gate A-scale loads on tile readiness")
        workers = int(os.environ.get(
            "AITER_FLYDSL_PERSISTENT_WORKERS1",
            os.environ.get("AITER_FLYDSL_PERSISTENT_WORKERS",
                           kw.get("persistent_workers", 0)),
        ))
        # CSV zero requests a non-persistent ordinary GEMM. S1F needs resident
        # producer/consumer roles, so zero retains its full-CU worker count.
        if workers == 0:
            workers = int(get_cu_num())
        validate_producer_group(self.producer_group, workers, producers, int(kw["tile_m"]))
        if self.ready_first and not 0 < workers <= min(256, int(get_cu_num())):
            raise ValueError(
                "local-ready pull requires resident workers within the "
                "256-head plan capacity"
            )
        cluster_m, cluster_n, cluster_reason = 1, 1, "cluster mode disabled"
        affinity_size, affinity_producers = 1, 0
        if self.producer_group == 4:
            if not workers <= min(256, int(get_cu_num())):
                raise ValueError("four-TG producers require resident workers <= 256")
            cluster_reason = (
                "producer-only 2x2 clusters; independent 1x1 GEMM consumers; "
                f"{producers // 4} affinity queues with cross-queue stealing"
            )
        elif self.cluster_local or _select_bool_env("AITER_MOE_S1F_AFFINITY", 0):
            affinity_size, affinity_producers = pull_affinity_geometry(
                workers, producers
            )
            cluster_reason = (
                f"mixed-role affinity: {affinity_producers} producers / "
                f"{affinity_size - affinity_producers} consumers per cluster; "
                "local queue first, cross-cluster stealing enabled"
            )
        elif _select_bool_env("AITER_MEGA_S1F_PULL_CLUSTER", 0):
            tile_n = int(kw["tile_n"])
            n_tiles = (2 * config.geometry.inter_dim + tile_n - 1) // tile_n
            requested_n = max(1, int(os.environ.get(
                "AITER_FLYDSL_MXFP4_CLUSTER_N", kw.get("cluster_n", 1)
            )))
            selected_n = _select_cluster_n(n_tiles, requested_n)
            selected_m = _select_cluster_m(kw.get("cluster_m", 1), selected_n, 1)
            cluster_m, cluster_n, cluster_reason = resolve_pull_cluster(
                selected_m, requested_n, workers, producers, n_tiles,
                int(kw["tile_m"]), step.align_m,
            )
        if self.cluster_local and step.align_m != int(kw["tile_m"]):
            raise ValueError("cluster-local plan alignment must equal GEMM1 tile_m")
        if self.ready_queue and (cluster_m != 1 or cluster_n != 1):
            raise ValueError("ready queue requires independent GEMM consumers")
        if self.ready_queue and step.align_m != int(kw["tile_m"]):
            raise ValueError("ready queue requires plan alignment == GEMM1 tile_m")
        if self.ready_queue:
            cluster_reason = (
                "producer affinity retained; consumers share a completed-A queue"
            )
        if self.cluster_local:
            cluster_reason = "cluster-local rounds with hardware signal/wait"
        self.cluster_selection = (cluster_m, cluster_n, cluster_reason)
        self.affinity_selection = (affinity_size, affinity_producers)
        gemm = {
            "tile_m": int(kw["tile_m"]),
            "tile_n": int(kw["tile_n"]),
            "tile_k": int(kw["tile_k"]),
            "m_warp": int(kw["m_warp"]),
            "n_warp": int(kw["n_warp"]),
            "num_buffers": int(kw["num_buffers"]),
            "next_stage_prefetch": _select_bool_env(
                "AITER_TDM_NEXT_STAGE_PREFETCH", kw.get("next_stage_prefetch", 0)
            ),
            "waves_per_tensor_tdm": _select_num_waves_per_tensor_tdm(
                int(kw.get("waves_per_tensor_tdm", -1))
            ),
            "tdm_b_th": _select_tdm_b_th(kw.get("tdm_b_th", 0)),
            "lds_soa_load_interleave": int(kw.get("lds_soa_load_interleave", 0)),
            "rolled_drain": int(kw.get("rolled_drain", 0)),
            "cluster_m": cluster_m,
            "cluster_n": cluster_n,
            "affinity_size": affinity_size,
            "affinity_producers": affinity_producers,
            "ready_first": self.ready_first,
            "ready_queue": self.ready_queue,
            "cluster_local": self.cluster_local,
            "producer_group": self.producer_group,
            "packed_input": self.packed,
            "src_rank_shuffle": self.src_rank_shuffle,
            "off_packed": self.arena.offset("pull_packed") if self.packed else 0,
            "tile_ready": (
                self.tile_ready and int(kw["tile_m"]) == 256
                and not self.ready_queue and not self.cluster_local
                and self.producer_group == 1
            ),
        }
        key = (
            step.align_m,
            int(kw["stage1_act"]),
            int(kw["quant_wmma_rep"]),
            producers,
            workers,
            tuple(sorted(gemm.items())),
        )
        if key not in self._fused_launches:
            logging.getLogger("aiter").info(
                "pull S1F GEMM1 effective config %s; "
                "CSV requested cluster=%sx%s, "
                "workers=%s producers=%s; %s",
                gemm, kw.get("cluster_m", -1), kw.get("cluster_n", -1),
                workers, producers,
                cluster_reason or "complete role-homogeneous clusters",
            )
            compiled = compile_s1f_pull(
                rank=config.rank,
                world=config.world_size,
                experts=config.experts_per_rank,
                max_routes=self.dispatch.compact.max_routes,
                compact_cap=self.dispatch.compact.cap,
                model_dim=config.hidden_dim,
                inter_dim=config.geometry.inter_dim,
                wire_row=self.dispatch.compact.wire_row,
                off_wire=self.arena.offset("pull_wire"),
                off_ready=self.arena.offset("pull_ready"),
                off_rowmap=self.arena.offset("ep_rowmap"),
                off_tile_flag=self.arena.offset("pull_tile_flag"),
                align_m=step.align_m,
                topk=config.topk,
                workers=workers,
                producers=producers,
                stage1_act=key[1],
                quant_wmma_rep=key[2],
                **gemm,
            )
            mailbox = None
            if self.ready_queue:
                mailbox = self.affinity_heads
            elif affinity_size > 1 or self.producer_group == 4:
                # One 128-byte-separated head per cluster, reset by compact-plan
                # in local-first mode or by the original ready kernel otherwise.
                mailbox = self.affinity_heads if self.ready_first else torch.empty(
                    (producers // 4 if self.producer_group == 4 else workers // affinity_size) * 32,
                    dtype=torch.int32, device=self.ctl.device,
                )
            elif cluster_n > 1:
                from ..s1f_stage1 import s1f_cluster_mailbox_words

                mailbox = torch.zeros(
                    s1f_cluster_mailbox_words(
                        compact_cap=self.dispatch.compact.cap,
                        tile_m=gemm["tile_m"],
                        n_tiles=2 * config.geometry.inter_dim // gemm["tile_n"],
                        cluster_n=cluster_n, workers=workers, producers=producers,
                    ), dtype=torch.int32, device=self.ctl.device,
                )
            self._fused_launches[key] = (*compiled, mailbox)
        ready, fused, mailbox = self._fused_launches[key]
        stream = fx.Stream(kw.get("stream") or torch.cuda.current_stream())
        rows = int(kw["contiguous_m"])
        if not self.ready_first:
            _run_compiled(
                ready,
                self.arena.handle,
                self.barrier,
                m_tile_map,
                self.tile_storage,
                rows,
                mailbox.data_ptr() if mailbox is not None else 0,
                stream,
            )
        _run_compiled(
            fused,
            out,
            ptr_arg(a),
            ptr_arg(w),
            a_scales.view(torch.int32),
            w_scales.view(torch.int32),
            kw["quant_scale"].view(torch.uint8),
            ptr_arg(m_tile_map),
            self.arena.handle,
            self.wire_ptr,
            0,
            0,
            0,
            mailbox.data_ptr() if mailbox is not None else 0,
            self.tile_info.data_ptr(),
            self.ctl.data_ptr(),
            rows,
            0,
            float(kw.get("swiglu_limit", 7.0)),
            stream,
        )
