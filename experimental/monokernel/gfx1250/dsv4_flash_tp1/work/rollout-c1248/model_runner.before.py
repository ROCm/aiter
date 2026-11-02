# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import gc
import inspect
import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, ClassVar, NamedTuple

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, init_dist_env
from aiter.dist.parallel_state import (
    get_dp_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
)
from aiter.dist.utils import get_distributed_init_method
from torch.profiler import record_function

from atom.config import Config, CUDAGraphMode, set_current_atom_config
from atom.distributed.pcp_utils import (
    PcpBalGroup,
    get_pcp_world_size,
    pcp_allgather_rerange,
    pcp_pad_len,
    pcp_round_robin_split,
)
from atom.distributed.pp_comm import (
    async_send_intermediate_tensors,
    commit_pp_send_work,
    recv_intermediate_tensors,
)
from atom.distributed.simulated_tp import apply_simulated_tp, reject_simulated_tp
from atom.kv_transfer.disaggregation import (
    KVConnectorFactory,
    KVConnectorOutput,
    kv_config_has_producer,
)
from atom.metrics.gpu import GPUForwardMetrics, record_gpu_forward
from atom.model_engine.dynamic_chunking_worker import DynamicChunkingWorker
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.run_labels import build_run_label
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    SequenceType,
    new_block_table,
)
from atom.model_engine.state_runtime import StateRuntime
from atom.model_loader.loader import load_model
from atom.model_ops.attentions.pool_layout.entry_arena import plan_regions
from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    InsufficientPoolBudget,
    Pool,
    PoolPlan,
    SubPoolSpec,
    plan_pools,
)
from atom.model_ops.decode_input_ids import (
    NEW_SEQUENCE,
    fill_deferred_decode_ids,
)
from atom.model_ops.eplb import (
    initialize_eplb_runtime,
    with_eplb_forward_monitor,
)
from atom.model_ops.rejection_sampler import RejectionSampler
from atom.model_ops.sampler import SAMPLER_EPS, Sampler
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.factory import build_drafter
from atom.utils import (
    CpuGpuBuffer,
    envs,
    get_hf_text_config,
    init_exit_handler,
    resolve_obj_by_qualname,
    worker_process_name,
)
from atom.utils.cuda_graph import BatchDescriptor
from atom.utils.forward_context import (
    Context,
    DPMetadata,
    ForwardMode,
    get_forward_context,
    get_kvconnector,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)
from atom.utils.gc_utils import freeze_gc_heap
from atom.utils.h2d import h2d_producer
from atom.utils.selector import attn_family, get_attn_backend, has_mla_indexer
from atom.utils.tbo import (
    UBatchSlice,
    UBatchWrapper,
    local_tbo_precompute,
    maybe_create_ubatch_slices,
)

logger = logging.getLogger("atom")

support_model_arch_dict = {
    "Qwen3ForCausalLM": "atom.models.qwen3.Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe.Qwen3MoeForCausalLM",
    "LlamaForCausalLM": "atom.models.llama.LlamaForCausalLM",
    "MixtralForCausalLM": "atom.models.mixtral.MixtralForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV41ForCausalLM": "atom.models.deepseek_v41.runtime.DeepseekV41RuntimeModel",
    "DeepseekV4ForCausalLM": "atom.models.deepseek_v4.DeepseekV4ForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss.GptOssForCausalLM",
    "GlmMoeDsaForCausalLM": "atom.models.deepseek_v2.GlmMoeDsaForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe.Glm4MoeForCausalLM",
    "Qwen3NextForCausalLM": "atom.models.qwen3_next.Qwen3NextForCausalLM",
    "Qwen4ExpForConditionalGeneration": (
        "atom.models.qwen4_exp.Qwen4ExpForConditionalGeneration"
    ),
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel",
    "Qwen3_5MoeForCausalLM": "atom.models.qwen3_5.Qwen3_5MoeForCausalLM",
    "KimiK25ForConditionalGeneration": "atom.models.kimi_k25.KimiK25ForCausalLM",
    "KimiK3ForConditionalGeneration": (
        "atom.models.kimi_k3.KimiK3ForConditionalGeneration"
    ),
    "MiniMaxM2ForCausalLM": "atom.models.minimax_m2.MiniMaxM2ForCausalLM",
    "MiMoV2ForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "MiMoV2FlashForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "Mistral3ForConditionalGeneration": "atom.models.mistral3.Mistral3TextOnly",
    "MistralForCausalLM": "atom.models.mistral3.Mistral3ForCausalLM",
    "MiniMaxM3SparseForCausalLM": "atom.models.minimax_m3.MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration": "atom.models.minimax_m3.MiniMaxM3SparseForConditionalGeneration",
    "Glm5NextForConditionalGeneration": (
        "atom.models.glm5_next.Glm5NextForConditionalGeneration"
    ),
}
# Mono decode by wrapping the loaded model: the installer runs after the drafter
# armed its hooks on the unwrapped layers, and may add per-step buffers to the
# metadata builder before they are bound. Architectures not listed here route
# mono decode inside their own model class instead.
mono_decode_installers = {
    "DeepseekV41ForCausalLM": (
        "atom.models.deepseek_v41.mono.dispatch.install_mono_decode"
    ),
}


def max_schedulable_decode_bs(
    max_num_seqs: int, max_num_batched_tokens: int, full_q_len: int
) -> int:
    """Largest decode batch the scheduler can admit.

    `Scheduler.schedule_decode` stops on either of two bounds: `max_num_seqs`
    sequences, or `max_num_batched_tokens` tokens. It charges every decode
    sequence the full speculative width `full_q_len` (== ``mtp_k + 1``) up
    front, whatever query length the step later replays, so dividing by
    `full_q_len` is exact rather than conservative — and it bounds every
    smaller q bucket too.

    Lives here so CUDAGraph capture can refuse to build a bucket the scheduler
    would never hand it. Must stay in step with `schedule_decode`'s
    `tokens_per_decode_seq`.
    """
    return min(max_num_seqs, max_num_batched_tokens // full_q_len)


# Re-exported under the old private name: this module is where the predicate
# used to live and where callers (and tests) still import it from.
_kv_config_has_producer = kv_config_has_producer


class TokenLocations(NamedTuple):
    """How each request in this decode batch gets its anchor token.

    `deferred_curr[k]` is a position in the CURRENT batch and
    `deferred_prev[k]` the row that request occupied in the previous forward,
    so its sampled id can be read from `prev_token_ids` without a D2H sync.
    `new_curr` holds the positions whose id the scheduler already put on the
    host. Together they cover the batch exactly once.

    A tuple of three same-typed arrays is easy to unpack in the wrong order;
    naming them makes that a typo the reader can see.
    """

    deferred_curr: np.ndarray
    deferred_prev: np.ndarray
    new_curr: np.ndarray


class tokenIDProcessor:

    def __init__(
        self,
        runner: "ModelRunner",
        max_num_batched_tokens: int,
        use_spec: bool = False,
        num_spec_tokens: int = 0,
    ):
        """Asynchronously copy the sampled_token_ids tensor to the host."""
        kv_cfg = getattr(runner.config, "kv_transfer_config", {}) or {}
        self.is_pipeline_parallel = (
            getattr(runner.config, "pipeline_parallel_size", 1) > 1
        )
        # The scheduler never speculates for a P/D producer, though the runner
        # keeps its drafter loaded: every decode it runs is q=1.
        self.is_pd_producer = _kv_config_has_producer(kv_cfg)
        # P/D hands off prompt-end state plus the first sampled token. Deferred
        # output makes the producer decode once more to surface that token, so
        # the consumer applies T0 to a state that already has it. Only a
        # recurrent state notices -- a paged write lands at T0's own position,
        # so repeating it is idempotent.
        hands_off_recurrent_state = (
            self.is_pd_producer and runner.attn_family.has_recurrent_state
        )
        self.is_deferred_out = (
            not self.is_pipeline_parallel and not hands_off_recurrent_state
        )

        self.runner = runner
        device = runner.device
        self.input_ids = CpuGpuBuffer(
            max_num_batched_tokens + 1,
            dtype=torch.int32,
            device=device,
            publication_group="input_ids",
        )
        # One per request, not per token: where each request's anchor comes
        # from. Sized by tokens -- a batch can never hold more requests. The
        # matching prefix sum is `forward_vars["cu_seqlens_q"]`.
        self.decode_src = CpuGpuBuffer(
            max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
            publication_group="input_ids",
        )
        self.use_spec = use_spec
        self.num_spec_tokens = num_spec_tokens

        self.async_copy_stream = torch.cuda.Stream(runner.device)
        self.default_num_rejected_tokens = torch.zeros(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.clean()

    def send_to_cpu_async(
        self,
        gpu_tensor: torch.Tensor,
        cpu_tensor_handle,
        data_ready: torch.cuda.Event,
        copy_done: torch.cuda.Event | None = None,
        gpu_logprobs: torch.Tensor | None = None,
    ):
        copy_done = copy_done or torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            cpu_logprobs = (
                gpu_logprobs.to("cpu", non_blocking=True)
                if gpu_logprobs is not None
                else None
            )
            copy_done.record(self.async_copy_stream)
        cpu_tensor_handle.append((cpu_tensor, copy_done))
        self.logprobs_cpu.append(cpu_logprobs)

    def recv_async_output(self, cpu_tensor_handle) -> torch.Tensor:
        if not cpu_tensor_handle:
            return torch.empty(0, dtype=torch.int32, device="cpu")
        cpu_tensor, event = cpu_tensor_handle.pop(0)
        event.synchronize()
        return cpu_tensor

    def recv_logprobs(self) -> list[float] | None:
        """Pop and return the earliest logprobs from the async copy queue.
        Must be called after recv_async_output (which synchronizes the event).
        """
        if not self.logprobs_cpu:
            return None
        logprob_tensor = self.logprobs_cpu.pop(0)
        if logprob_tensor is not None:
            return logprob_tensor.tolist()
        return None

    def send_to_cpu_async_draft(self, gpu_tensor: torch.Tensor):
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.async_copy_stream)
        # No reverse wait here. `gpu_tensor` is a draft pass's captured output,
        # so its address is fixed and the NEXT replay rewrites it -- but that
        # replay sits behind a whole target forward on the default stream, while
        # this copy is a few KB. Ordering the default stream against it would
        # stall it for the copy every step, which is the overlap the side stream
        # exists to buy. If a future change ever puts a replay closer than one
        # forward away, the wait belongs immediately before THAT replay, not
        # here.
        self.draft_token_ids_cpu.append((cpu_tensor, event))

    def recv_async_output_draft(self) -> np.ndarray:
        if not self.draft_token_ids_cpu:
            return np.array([], dtype=np.int32)
        token_ids, event = self.draft_token_ids_cpu.pop(0)
        event.synchronize()
        return token_ids.numpy()

    def send_mtp_status_to_cpu_async(
        self,
        num_rejected: torch.Tensor,
        num_bonus: torch.Tensor,
        data_ready: torch.cuda.Event,
    ):
        # rejected num and bonus num are slightly different info for mtp
        # take mtp=1 for example:
        #   first decode after prefill have 0 rej, 0 bonus
        #   prev acc decode have 0 rej, 1 bonus
        #   prev rej decode have 1 rej, 0 bonus
        # It is clear that only rejected number is not sufficient for all status tracking, bonus number is also needed.
        # Single Event for both copies (vs. per-tensor send_to_cpu_async) so the
        # consumer pops one queue entry and synchronizes once instead of twice.
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_num_rejected = num_rejected.to("cpu", non_blocking=True)
            cpu_num_bonus = num_bonus.to("cpu", non_blocking=True)
            copy_done.record(self.async_copy_stream)
        self.pending_mtp_status_copies.append(
            (cpu_num_rejected, cpu_num_bonus, copy_done)
        )

    def recv_mtp_status_async(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.pending_mtp_status_copies:
            return None, None
        cpu_num_rejected, cpu_num_bonus, copy_done = self.pending_mtp_status_copies.pop(
            0
        )
        copy_done.synchronize()
        return cpu_num_rejected.numpy(), cpu_num_bonus.numpy()

    def clean(self):
        self.token_ids_cpu: list[torch.Tensor] = []
        self.logprobs_cpu: list[torch.Tensor | None] = []

        self.prev_batch: ScheduledBatch | None = None
        self.prev_token_ids: torch.Tensor | None = None

        self.pre_num_decode_token_per_seq = 1
        self.draft_token_ids: torch.Tensor | None = None
        self.draft_token_ids_cpu: list[torch.Tensor] = []
        # Queue of (cpu_num_rejected, cpu_num_bonus, copy_done_event) — async
        # D2H copies fired by send_mtp_status_to_cpu_async, drained by
        # recv_mtp_status_async after the event syncs.
        self.pending_mtp_status_copies: list[
            tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
        ] = []
        self.num_rejected: np.ndarray | None = None
        self.num_bonus: np.ndarray | None = None

    @staticmethod
    def _batch_process_token_ids(token_ids: list) -> list[tuple[int, ...]]:
        """Batch process token_ids: vectorized -1 truncation using numpy."""
        arr = np.array(token_ids, dtype=np.int64)
        mask = arr == -1
        if not mask.any():
            # No -1 sentinel in any row, convert each row to tuple directly
            return [tuple(row) for row in arr.tolist()]
        # Per-row: find first -1, truncate
        # Use argmax on mask; rows without -1 get 0, disambiguate with ~mask.any(axis=1)
        has_sentinel = mask.any(axis=1)
        first_neg = mask.argmax(axis=1)
        result = []
        rows = arr.tolist()
        for i, row in enumerate(rows):
            if has_sentinel[i]:
                result.append(tuple(row[: first_neg[i]]))
            else:
                result.append(tuple(row))
        return result

    def prepare_sampled_ids(
        self,
        batch: ScheduledBatch,
        sampled_token_ids: torch.Tensor,
        sync_event: torch.cuda.Event,
        sampled_logprobs: torch.Tensor | None = None,
    ) -> tuple[dict[int, tuple[int, ...]], dict[int, float] | None]:
        if not self.is_deferred_out:
            token_ids = sampled_token_ids.tolist()
            req_ids = batch.req_ids
            if token_ids and isinstance(token_ids[0], list):
                processed = self._batch_process_token_ids(token_ids)
            else:
                processed = [(tid,) for tid in token_ids]
            ret = dict(zip(req_ids, processed))
            ret[-1] = 0  # is_deferred_out flag
            logprobs_map = None
            if sampled_logprobs is not None:
                logprobs = sampled_logprobs.tolist()
                logprobs_map = {
                    seq_id: logprob for seq_id, logprob in zip(req_ids, logprobs)
                }
            return ret, logprobs_map

        token_ids = self.recv_async_output(self.token_ids_cpu)
        logprobs = self.recv_logprobs()
        self.send_to_cpu_async(
            sampled_token_ids,
            self.token_ids_cpu,
            sync_event,
            gpu_logprobs=sampled_logprobs,
        )
        token_id_dict = {}
        logprobs_map = None
        self.prev_req_ids = None
        if self.prev_batch is not None:
            self.prev_req_ids = self.prev_batch.req_ids
            token_ids_list = (
                token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
            )
            if token_ids_list and isinstance(token_ids_list[0], list):
                processed = self._batch_process_token_ids(token_ids_list)
            else:
                processed = [(tid,) for tid in token_ids_list]
            token_id_dict = dict(zip(self.prev_req_ids, processed))
            if logprobs is not None:
                logprobs_map = {
                    seq_id: logprob
                    for seq_id, logprob in zip(self.prev_req_ids, logprobs)
                }
        else:
            # first time, no previous tokens
            token_ids = {}
            logprobs_map = None

        self.prev_batch = batch
        self.prev_token_ids = sampled_token_ids
        token_id_dict[-1] = 1
        return token_id_dict, logprobs_map

    def get_token_locations(self, batch: ScheduledBatch) -> TokenLocations:
        prev_req_ids = self.prev_batch.req_ids
        cur_req_ids = batch.req_ids
        num_prev = len(prev_req_ids)
        num_cur = len(cur_req_ids)

        # A fabricated batch carries nothing over -- and the DP-sync dummy
        # reuses one id, so without this the next dummy matches it and reads a
        # request that never existed out of `prev_token_ids`/`draft_token_ids`.
        # Real ids are non-negative and never matched it either way.
        if self.prev_batch.is_dummy_run:
            none = np.empty(0, dtype=np.intp)
            return TokenLocations(none, none, np.arange(num_cur, dtype=np.intp))

        prev_id_to_idx = dict(zip(prev_req_ids, range(num_prev)))

        deferred_curr = np.empty(num_cur, dtype=np.intp)
        deferred_prev = np.empty(num_cur, dtype=np.intp)
        new_curr = np.empty(num_cur, dtype=np.intp)
        n_deferred = 0
        n_new = 0

        for cur_idx in range(num_cur):
            prev_idx = prev_id_to_idx.get(cur_req_ids[cur_idx])
            if prev_idx is not None:
                deferred_curr[n_deferred] = cur_idx
                deferred_prev[n_deferred] = prev_idx
                n_deferred += 1
            else:
                new_curr[n_new] = cur_idx
                n_new += 1

        deferred_curr = deferred_curr[:n_deferred]
        deferred_prev = deferred_prev[:n_deferred]
        new_curr = new_curr[:n_new]

        # Every request must be classified exactly once: `deferred_curr` and
        # `new_curr` are what the caller addresses the batch through, so a gap
        # or an overlap would leave a request reading whatever was staged for
        # it -- silently, and only in whatever batch shape produced the gap.
        assert (
            n_deferred + n_new == num_cur
        ), f"{n_deferred} deferred + {n_new} new != {num_cur} requests"
        return TokenLocations(deferred_curr, deferred_prev, new_curr)

    def _publish_input_ids(self, count, group):
        if group is None:
            return self.input_ids.copy_to_gpu(count)
        group.counts[group.indices["input_ids"]] = count
        group.counts[group.indices["decode_src"]] = None
        group.publish(group.counts)
        return self.input_ids.gpu[:count]

    @h2d_producer("input_ids", runner="runner")
    def prepare_input_ids(
        self,
        batch: ScheduledBatch,
        max_seqlen_q: int,
        *,
        publication_group=None,
    ) -> torch.Tensor:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids.
        """
        scheduled_tokens = batch.scheduled_tokens  # tokens per req
        total_tokens_prefill = batch.total_tokens_num_prefill
        total_tokens_decode = batch.total_tokens_num_decode
        total_reqs_prefill = batch.total_seqs_num_prefill
        # The MTP status queue is filled in postprocess but drained here, so a
        # step whose postprocess is skipped must not drain it: `forward()` bails
        # before postprocess when the batch produces no output (every prefill in
        # it is a middle chunk), and the status it popped belongs to the batch
        # whose deferred tokens the NEXT output-producing step will surface.
        # Draining it here would hand that step `num_rejected=None`.
        if batch.produces_output():
            self.prev_rejected_num, self.prev_bonus_num = self.recv_mtp_status_async()

        # TODO: remove this when we support mixed prefill and decode in one batch
        if total_reqs_prefill > 0:
            # Decode does not publish an empty prefill prefix: an explicit
            # zero copy is still a publication in this forward's ledger.
            self.input_ids.np[:total_tokens_prefill] = scheduled_tokens[
                :total_tokens_prefill
            ]
            return self._publish_input_ids(total_tokens_prefill, publication_group)

        if not self.is_deferred_out:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            # No spec path supports pipeline parallel yet; it wants the deferred
            # branch's per-request staging. Gate on PP itself rather than on
            # `use_spec`: a P/D producer also reaches this branch, and the flag
            # is still set there (the runner keeps the drafter loaded to write
            # the draft's context KV at prefill) even though the scheduler
            # turned speculation off for it.
            if self.use_spec and self.is_pipeline_parallel:
                raise NotImplementedError("pipeline parallel + speculative decode")
            # PP verifies nothing locally and a P/D producer never speculates
            # (its DP dummy included), so no row here carries drafts. One that
            # did would need the deferred branch's per-request staging: under
            # DP a block drafter's `max_seqlen_q` is the group's maximum, not
            # this batch's row width.
            assert batch.num_spec_step == 0, (
                "non-deferred decode got a speculative batch "
                f"(num_spec_step={batch.num_spec_step})"
            )

            self.input_ids.np[:total_tokens_decode] = token_ids
            return self._publish_input_ids(total_tokens_decode, publication_group)

        # PD consumer first decode: no prior prefill step initialized
        # prev_batch, so use scheduled_tokens directly for this step.
        if self.prev_batch is None:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            self.input_ids.np[:total_tokens_decode] = token_ids
            return self._publish_input_ids(total_tokens_decode, publication_group)

        """for decode: input ids are from prev_sampled_token_ids"""
        locs = self.get_token_locations(batch)
        deferred_curr_indices = locs.deferred_curr
        deferred_prev_indices = locs.deferred_prev
        new_curr_indices = locs.new_curr
        num_deferred_seqs = len(deferred_curr_indices)
        num_new_seqs = len(new_curr_indices)

        # The GRAPH's row stride, for sizing the padded tail only. NOT a
        # per-request length: dp>1 raises it to the DP-wide maximum.
        tokens_per_seq = max_seqlen_q if self.use_spec else 1

        # Receive and map bonus_list to current batch order
        self.num_rejected = batch.num_rejected
        self.num_bonus = batch.num_bonus
        if num_deferred_seqs > 0 and self.prev_rejected_num is not None:
            # Remap prev step's rejected/bonus counts onto the current batch order
            # (prev_idx → curr_idx) for the deferred (carried-over) sequences.
            self.num_rejected[deferred_curr_indices] = self.prev_rejected_num[
                deferred_prev_indices
            ]
            self.num_bonus[deferred_curr_indices] = self.prev_bonus_num[
                deferred_prev_indices
            ]

        # ---- One path for every decode step -------------------------------
        # `num_scheduled_tokens` is the one statement of the per-request
        # lengths -- the ragged shrink rewrites it in place, so ragged and
        # rectangular are the same array. Its prefix sum is `cu_seqlens_q`,
        # already published; a second copy is what let the two disagree under DP.
        bs, lens, cu_np = self.runner.attn_metadata_builder.decode_spans(batch)

        # Stage the scheduler's ids over the whole region. For a request the
        # scheduler just admitted this is already its real anchor (and drafts);
        # for a carried-over one it is a placeholder the kernel overwrites.
        self.input_ids.np[:total_tokens_decode] = scheduled_tokens[
            total_tokens_prefill : total_tokens_prefill + total_tokens_decode
        ]
        # A newly admitted request has no row in `prev_token_ids`, so
        # `fill_deferred_decode_ids` skips it (`src < 0`) and its draft columns
        # must be staged here -- `scheduled_tokens` gave it committed history,
        # correct only at column 0. A batch without drafts (`num_spec_step ==
        # 0`: a P/D producer's decodes and DP dummies) has no draft table.
        #
        # Looped, not scattered: measured, the loop wins below ~24 admissions
        # per step (1.2us at 2) and a ragged scatter is a flat ~8us; above it
        # the scatter wins (12us vs 88us at bs=256). Swap if admissions stop
        # being few.
        if self.use_spec and batch.num_spec_step > 0 and num_new_seqs > 0:
            spec = batch.scheduled_spec_decode_tokens
            for i in new_curr_indices:
                n_draft = int(lens[i]) - 1
                if n_draft > 0:
                    s = int(cu_np[i]) + 1
                    self.input_ids.np[s : s + n_draft] = spec[i, :n_draft]
        src_np = self.decode_src.np[:bs]
        src_np.fill(NEW_SEQUENCE)
        src_np[deferred_curr_indices] = deferred_prev_indices
        group = (
            self.runner.h2d_groups["input_ids"]
            if publication_group is None
            else publication_group
        )
        counts = group.counts
        counts[group.indices["input_ids"]] = total_tokens_decode
        counts[group.indices["decode_src"]] = bs
        group.publish(counts)
        # How wide the forward reads: a replayed decode graph takes a fixed
        # `running_bs * tokens_per_seq` whatever the batch scheduled, and `bs`
        # sits between two captured buckets on most steps -- a 65-request batch
        # replays the 128 graph. The kernel zeroes the difference.
        fill_to = total_tokens_decode
        if not self.runner.enforce_eager:
            gbs = next(
                (g for g in reversed(self.runner.capture_sizes) if g >= bs), None
            )
            if gbs is not None:
                fill_to = max(fill_to, int(gbs) * tokens_per_seq)
        fill_deferred_decode_ids(
            self.input_ids.gpu,
            self.runner.forward_vars["cu_seqlens_q"].gpu[: bs + 1],
            self.decode_src.gpu[:bs],
            self.prev_token_ids,
            self.draft_token_ids if self.pre_num_decode_token_per_seq > 1 else None,
            max_tokens_per_seq=int(lens.max()) if bs else 1,
            width=fill_to,
        )

        # Slice by the width this path actually staged. Prefill returned above,
        # so the decode total is the whole batch; a worker-side speculative q
        # shrink rewrites it, and reading any other total here would leave the
        # attention metadata wider than input_ids.
        input_ids = self.input_ids.gpu[:total_tokens_decode]
        return input_ids

    def prepare_draft_ids(
        self, batch: ScheduledBatch, draft_token_ids: torch.Tensor
    ) -> np.ndarray:
        if not self.is_deferred_out:
            # propose() builds this on the drafter's device; the scheduler wants
            # host rows.
            ret = draft_token_ids.cpu().numpy()
        else:
            self.draft_token_ids = draft_token_ids
            self.pre_num_decode_token_per_seq = self.num_spec_tokens + 1
            token_ids = self.recv_async_output_draft()
            self.send_to_cpu_async_draft(draft_token_ids)
            ret = (
                token_ids
                if self.prev_req_ids is not None
                else np.array([], dtype=np.int32)
            )
        return ret


def _profile_runner_stage(method):
    """Annotate runner stages only while its torch profiler is active."""
    name = method.__name__
    label = f"ATOM::{name}"

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "profiler", None) is None:
            return method(self, *args, **kwargs)
        stage_label = label
        if name == "forward":
            batch = args[0] if args else kwargs["batch"]
            stage_label += (
                f" tokens={batch.total_tokens_num}"
                f" prefill={batch.total_seqs_num_prefill}"
                f" decode={batch.total_seqs_num_decode}"
            )
        with record_function(stage_label):
            return method(self, *args, **kwargs)

    return wrapped


class ModelRunner:

    def __init__(self, rank: int, config: Config):
        self.config = config
        from atom.model_engine.multimodal_runtime import VisionEmbeddingCache

        self.vision_embeddings = VisionEmbeddingCache()
        self.mark_trace = getattr(config, "mark_trace", False)
        from atom.utils.graph_marker import set_graph_marker_enabled

        set_graph_marker_enabled(self.mark_trace)
        set_current_atom_config(config)
        hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size
        self.kv_cache_dtype = config.kv_cache_dtype
        self.enforce_eager = config.enforce_eager
        # world_size: the logical TP width, i.e. how many shards each weight is
        # cut into -- what the KV-head math below divides by.
        # tp_world_size: how many of those shards have a process.
        # They differ only under simulated TP.
        self.world_size = config.tensor_parallel_size
        self.tp_world_size = config.tp_world_size
        self.rank = rank
        self.label = f"Model Runner{rank}/{self.tp_world_size}"
        self._dynamic_chunking = DynamicChunkingWorker(self)
        self.hf_text_config = get_hf_text_config(hf_config)
        if self.hf_text_config.model_type in ["llama"] and self.config.torch_dtype in [
            torch.bfloat16,
            torch.float16,
        ]:
            os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
        # Which attention this model's config asks for. The runner holds the
        # answer, not the question: what makes a model MLA or a hybrid is the
        # selector's to know, and it is asked the same way of a draft.
        self.attn_family = attn_family(self.hf_text_config)

        rope_parameters = getattr(self.hf_text_config, "rope_parameters", None) or {}
        self.use_mrope = "mrope_section" in rope_parameters
        self.has_mla_indexer = has_mla_indexer(self.hf_text_config)
        # Initialize profiler for this rank (before _setup_device_and_distributed
        # so that dp config fields are still at their original values)
        self.profiler = None
        self.profiler_dir = None
        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        if dp_rank_local > 0 or config.parallel_config.data_parallel_size > 1:
            self.rank_name = f"dp{dp_rank_local}_tp{rank}"
        else:
            self.rank_name = f"rank_{rank}"
        if config.torch_profiler_dir is not None:
            rank_name = self.rank_name
            if config.pipeline_parallel_size > 1:
                rank_name = (
                    f"pp{config.parallel_config.pipeline_parallel_rank}_{rank_name}"
                )
            self.profiler_dir = os.path.join(config.torch_profiler_dir, rank_name)
            os.makedirs(self.profiler_dir, exist_ok=True)

        self.attn_backend = get_attn_backend(self.attn_family)
        self._setup_device_and_distributed(rank, config)

        self.capture_sizes = [0]  # for eager fallback
        # The same ladder as an ASCENDING int32 array, which is what
        # `ForwardMode.decide` searches. Separate from the list because the list
        # is re-sorted in both directions during capture; rebound (never
        # mutated) once capture narrows it, so a search cannot read a transient
        # order.
        self.capture_sizes_np = np.asarray(self.capture_sizes, dtype=np.int32)
        # PIECEWISE cudagraph state, populated by capture_cudagraph. Empty when
        # capture never ran (enforce_eager), so the ragged-bucket paths no-op.
        self._piecewise_captured_tokens: set[int] = set()
        self._piecewise_sorted_tokens: list[int] = []

        init_exit_handler(self)
        default_dtype = self.config.torch_dtype
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(self.device)
        use_spec = bool(self.config.speculative_config) and get_pp_group().is_last_rank
        self.num_spec_tokens = (
            self.config.speculative_config.num_speculative_tokens if use_spec else 0
        )

        self._pp_pending_send: list = []
        self.tokenID_processor = tokenIDProcessor(
            self,
            self.config.max_num_batched_tokens,
            use_spec,
            self.num_spec_tokens,
        )
        self.sampler = Sampler()
        self.arange_np = np.arange(
            max(
                self.config.max_num_seqs + 1,
                self.config.max_model_len,
                self.config.max_num_batched_tokens,
            ),
            dtype=np.int64,
        )

        model_class = resolve_obj_by_qualname(support_model_arch_dict[hf_config.architectures[0]])  # type: ignore
        # The model construction depends on quant_config,
        # so we must complete the remapping for layers before constructing the model.
        config.quant_config.remap_layer_name(
            config.hf_config,
            packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
            quant_exclude_name_mapping=getattr(
                model_class, "quant_exclude_name_mapping", {}
            ),
        )

        self._build_and_load_model(model_class)

        # Optional debug instrumentation; no-op when env vars unset.
        # See atom/utils/debug_helper/.
        from atom.utils.debug_helper import (
            install_block_forward_hooks,
            maybe_dump_weights_and_exit,
        )

        _n_fwd_hooks = install_block_forward_hooks(self.model)
        if _n_fwd_hooks > 0:
            logger.info(f"[ATOM_FWD_DUMP] {_n_fwd_hooks} Block forward hooks installed")
        maybe_dump_weights_and_exit(self.model)

        if self.config.speculative_config and get_pp_group().is_last_rank:
            from atom.utils.backends import set_model_tag

            torch.set_default_device(self.device)
            with set_model_tag("drafter"):
                self.drafter = build_drafter(self.config, self.device, self)
            self.rejection_sampler = RejectionSampler(
                synthetic_acceptance_rates=(
                    self.config.speculative_config.synthetic_acceptance_rates
                )
            )
            torch.set_default_device(None)
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.model)
            # NOTE: aux-hidden-state capture is armed AFTER the optional TBO wrap
            # below, so the drafter's capture hook lands on the object whose
            # forward returns the final (concatenated) output. See arm_aux_capture.

        torch.set_default_device(self.device)
        self.async_execute_stream = torch.cuda.Stream(self.device)
        self.allocate_forward_vars()
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            model_runner=self
        )
        # Sub-pool sizing needs a memory profile, so it cannot run until after
        # warmup. Install the empty plan now: `warmup_model` below drives the
        # builder through paths that ask for their entry counts, and those must
        # read 0 ("no pool yet") rather than trip over a missing attribute.
        self.pool_plan = PoolPlan.empty()
        self.state_runtime = StateRuntime()
        # Sanity-check: any builder that allocates a per-request cache must
        # have its model_type listed in `InputOutputProcessor`'s
        # `per_req_cache_model_types` set; otherwise sequences will be
        # constructed with `has_per_req_cache=False`, the BlockManager will
        # never assign them a slot, and the builder will silently read
        # tensor[-1] on first decode. Catch the misconfiguration up front
        # rather than producing wrong outputs at inference time.
        if self._state_pool_names():
            from atom.model_engine.llm_engine import InputOutputProcessor as _IOProc

            mt = self.config.hf_config.model_type
            known = _IOProc._per_req_cache_model_types()
            assert mt in known, (
                f"Attention builder {type(self.attn_metadata_builder).__name__} "
                f"declares a per-request state pool but model_type={mt!r} is not in "
                f"InputOutputProcessor.per_req_cache_model_types ({sorted(known)}). "
                "Add it to the set or sequences will not be assigned slots "
                "(silent corruption)."
            )
        if config.enable_tbo:
            dp_gather_scatter = (
                config.enable_dp_attention and not config.enable_expert_parallel
            )
            self.model = UBatchWrapper(
                self.model,
                attn_metadata_builder=self.attn_metadata_builder,
                dp_gather_scatter=dp_gather_scatter,
            )
            logger.info("TBO enabled: model wrapped with UBatchWrapper")
        if getattr(self, "drafter", None) is not None:
            self.drafter.arm_aux_capture(self.model)
        installer = mono_decode_installers.get(hf_config.architectures[0])
        if installer is not None:
            self.model = resolve_obj_by_qualname(installer)(
                self.model,
                config,
                getattr(self, "drafter", None),
                self.attn_metadata_builder,
            )
        self._init_forward_vars_ring()
        self._init_h2d_publication()
        self.forward_done_event = torch.cuda.Event()
        initialize_eplb_runtime(self)
        self._maybe_warmup()

        # Restore the implicit CPU default. An explicit "cpu" leaves a
        # DeviceContext intercepting every torch call during serving, including
        # tensor views and Triton's pointer/stride specialization queries.
        torch.set_default_device(None)
        torch.set_default_dtype(default_dtype)

        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")
            if hasattr(self, "drafter"):
                self.drafter.model = torch.compile(
                    self.drafter.model, fullgraph=True, backend="eager"
                )

        # Install after initialization warmup, which is not request traffic.
        # Graph capture runs later via RPC and bypasses run_model, so its
        # timing decorator does not run during capture.
        self.gpu_forward_metrics = None
        if envs.ATOM_ENABLE_METRICS_DEVICE_TIMER:
            self.gpu_forward_metrics = GPUForwardMetrics(
                lambda: torch.cuda.Event(enable_timing=True),
                dp_rank=config.parallel_config.data_parallel_rank,
                pp_rank=config.parallel_config.pipeline_parallel_rank,
                tp_rank=self.rank,
                engine_role=(
                    ("decode" if config.disagg_is_decode else "prefill")
                    if config.enable_rapidserve
                    else "default"
                ),
            )

    def poll_forward_metrics(self):
        if self.gpu_forward_metrics is not None:
            self.gpu_forward_metrics.poll()

    def _build_and_load_model(self, model_class):
        """Construct the model and load its weights from disk.

        Override point: subclasses (e.g. the rapidserve decode process) may
        construct on the meta device and import weights via IPC instead.
        """
        config = self.config
        self.model = model_class(config)
        fused_shared_expert_load_fn = None
        if hasattr(self.model, "load_fused_expert_weights"):
            fused_shared_expert_load_fn = self.model.load_fused_expert_weights
        torch.set_default_device(None)
        load_start = time.perf_counter()
        load_model(
            self.model,
            config.model,
            config.hf_config,
            config.load_dummy,
            load_fused_expert_weights_fn=fused_shared_expert_load_fn,
        )
        load_elapsed = time.perf_counter() - load_start
        logger.info(
            f"[{self.rank_name}] Model load done: {config.model} "
            f"(weights loaded in {load_elapsed:.2f}s)"
        )

    def _maybe_warmup(self):
        """Run model warmup. Override point: the rapidserve decode process
        skips warmup since it imports weights/kvcache from prefill later."""
        self.warmup_model()
        logger.info(f"Model warmup done: {self.config.model}")

    def _kv_budget_extra_reserve(self, total_bytes: int) -> int:
        """Extra GPU bytes to hold back from the KV cache budget beyond the
        base overhead. Base runner reserves nothing; override point for
        setups that share the GPU with another process."""
        return 0

    def _setup_device_and_distributed(self, rank: int, config: Config):
        # Calculate local device rank considering DP, PP and PCP.
        # On a single node the physical GPU index equals the global distributed
        # rank in the DPxPPxPCPxTP layout: each EngineCore (one per (dp,pp)
        # stage) owns a contiguous tp*pcp GPU slice. `rank` is this worker's
        # local index (0..tp*pcp-1) within its stage.
        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        pp_rank = config.parallel_config.pipeline_parallel_rank
        pp_size = config.pipeline_parallel_size
        if pp_size > 1:
            # Reject before any collective can wait for nonexistent ranks.
            reject_simulated_tp(config, "pipeline parallel")
        # tp_world_size: how many GPUs this stage actually occupies.
        stage_span = config.tp_world_size * config.prefill_context_parallel_size
        engine_index = dp_rank_local * pp_size + pp_rank
        local_device_rank = engine_index * stage_span + rank
        num_gpus = torch.cuda.device_count()
        if local_device_rank >= num_gpus:
            raise ValueError(
                f"Calculated local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}). "
            )

        self.device = torch.device(f"cuda:{local_device_rank}")
        logger.info(
            f"ModelRunner rank={rank}, dp_rank_local={dp_rank_local}, "
            f"pp_rank={pp_rank}, local_device_rank={local_device_rank}, "
            f"device={self.device}"
        )

        torch.cuda.set_device(self.device)
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.port)
        distributed_init_method = get_distributed_init_method(
            config.parallel_config.data_parallel_master_ip,
            config.parallel_config.data_parallel_base_port,
        )
        dp_size = config.parallel_config.data_parallel_size
        world_size = dp_size * pp_size * stage_span
        dp_rank = config.parallel_config.data_parallel_rank
        global_rank = (dp_rank * pp_size + pp_rank) * stage_span + rank
        if (
            config.parallel_config._managed_distributed_store
            and not torch.distributed.is_initialized()
        ):
            # Preserve AITER's environment setup when preinitializing its group.
            os.environ.setdefault(
                "HIP_VISIBLE_DEVICES", ",".join(map(str, range(world_size)))
            )
            store = torch.distributed.TCPStore(
                config.parallel_config.data_parallel_master_ip,
                config.parallel_config.data_parallel_base_port,
                is_master=False,
            )
            torch.distributed.init_process_group(
                backend="nccl", store=store, rank=global_rank, world_size=world_size
            )
        # AITER reuses the default group and creates the model-parallel groups.
        if config.pipeline_parallel_size > 1:
            from atom.distributed.pp_comm import init_pp_aware_dist_env

            # No local_rank here, unlike the non-PP branch below. Safe only
            # because PP is single-node today: CoreManager rejects multi-node
            # DP when pp_size > 1, and asserts PP+DP out entirely, so
            # global_rank is already the physical device index. Revisit if
            # either restriction is lifted.
            init_pp_aware_dist_env(
                tensor_model_parallel_size=config.tensor_parallel_size,
                pipeline_model_parallel_size=pp_size,
                global_rank=global_rank,
                world_size=world_size,
                distributed_init_method=distributed_init_method,
                backend="nccl",
                data_parallel_size=dp_size,
                prefill_context_model_parallel_size=config.prefill_context_parallel_size,
            )
        else:
            # The group spans the devices that exist; apply_simulated_tp then
            # makes it *report* the logical width so layers shard that many ways.
            init_dist_env(
                config.tp_world_size,
                rankID=rank,
                backend="nccl",
                distributed_init_method=distributed_init_method,
                # This node's physical device index. Without it aiter derives a
                # local rank from the DP-scaled global rank, which overruns the
                # device list on every node after the first.
                local_rank=local_device_rank,
                data_parallel_size=config.parallel_config.data_parallel_size,
                data_parallel_rank=config.parallel_config.data_parallel_rank,
                prefill_context_model_parallel_size=config.prefill_context_parallel_size,
                decode_context_parallel_size=getattr(
                    config, "decode_context_parallel_size", 1
                ),
            )
            apply_simulated_tp(config)

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=True, with_numpy=numpy
        )

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def release_multimodal_requests(self, request_ids):
        self.vision_embeddings.release(request_ids)

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        self.vision_embeddings.clear()
        # 0. Join any offload connector's copy threads. Its ThreadPoolExecutors
        #    are non-daemon, so leaving them running wedges interpreter shutdown
        #    or races an in-flight copy against atexit. Must run BEFORE the KV
        #    pool it copies out of is dropped and before the dist env goes away.
        #    Guarded: only offload workers define close() (moriio etc. do not).
        connector = get_kvconnector()
        close = getattr(connector, "close", None) if connector is not None else None
        if callable(close):
            close()
        builder = getattr(self, "attn_metadata_builder", None)
        if builder is not None:
            builder.close()
        # 1. Destroy distributed env (NCCL + CustomAllreduce + process groups)
        #    Must happen while ops module is still alive for CustomAllreduce cleanup.
        destroy_dist_env()
        # 2. Release CUDA graphs
        if not self.enforce_eager:
            self.graphs = self.graph_pool = None  # type: ignore
        if isinstance(self.model, UBatchWrapper):
            self.model.tbo_graphs.clear()
        # 3. Release GPU tensors. `kv_cache` is the whole paged pool -- the
        # scales and any indexer cache are regions of it, not attributes.
        for attr in (
            "kv_cache",
            "mamba_k_cache",
            "mamba_v_cache",
            "kpool_tail_cache",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "drafter"):
            del self.drafter
        torch.cuda.empty_cache()
        return True

    def start_profiler(self, trace_name: str | None = None):
        """
        Start profiling for this rank.

        Set ATOM_PROFILER_RECORD_SHAPES, ATOM_PROFILER_WITH_STACK, or
        ATOM_PROFILER_PROFILE_MEMORY to "1"/"0" to enable/disable the matching
        profiler option. Any left unset falls back to ATOM_PROFILER_MORE, which
        enables all three when "1" (default: all disabled).
        """
        if self.profiler_dir is not None and self.profiler is None:
            record_shapes = envs.ATOM_PROFILER_RECORD_SHAPES
            with_stack = envs.ATOM_PROFILER_WITH_STACK
            profile_memory = envs.ATOM_PROFILER_PROFILE_MEMORY
            model_name = os.path.basename(self.config.model.rstrip("/"))
            safe_model_name = "".join(
                c if c.isalnum() or c in ("_", "-", ".") else "_" for c in model_name
            )
            worker_name = safe_model_name or "trace"
            if isinstance(trace_name, str) and trace_name:
                worker_name = "".join(
                    c if c.isalnum() or c in ("_", "-", ".") else "_"
                    for c in trace_name
                )
            if worker_name == "capture_graph" and safe_model_name:
                worker_name = f"{worker_name}_{safe_model_name}"
            output_prefix = os.path.join(self.profiler_dir, worker_name)

            def _on_trace_ready(prof):
                import gzip as _gzip

                # Use a short human-readable timestamp in file name.
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                ms = int((time.time() % 1) * 1000)
                output_path = f"{output_prefix}_ts_{ts}_{ms:03d}.pt.trace.json.gz"
                tmp_json_path = output_path[:-3]
                try:
                    t0 = time.monotonic()
                    prof.export_chrome_trace(tmp_json_path)
                    # Chunked gzip: read 64 MB at a time to avoid loading
                    # the entire JSON (~30 GB) into memory at once.
                    with (
                        open(tmp_json_path, "rb") as src,
                        _gzip.open(output_path, "wb") as dst,
                    ):
                        while chunk := src.read(64 * 1024 * 1024):
                            dst.write(chunk)
                    os.remove(tmp_json_path)
                    sz = os.path.getsize(output_path)
                    logger.info(
                        "Rank %d: trace exported to %s (%.1f MB, %.1fs)",
                        self.rank,
                        output_path,
                        sz / 1e6,
                        time.monotonic() - t0,
                    )
                except Exception:
                    logger.exception(
                        "Rank %d: failed to export trace to %s",
                        self.rank,
                        output_path,
                    )
                    for p in (tmp_json_path, output_path):
                        if os.path.exists(p):
                            os.remove(p)

            self.profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=record_shapes,
                with_stack=with_stack,
                profile_memory=profile_memory,
                on_trace_ready=_on_trace_ready,
            )
            self.profiler.__enter__()
            logger.info(
                "Rank %d: profiler started "
                "(record_shapes=%s, with_stack=%s, profile_memory=%s, dir=%s)",
                self.rank,
                record_shapes,
                with_stack,
                profile_memory,
                self.profiler_dir,
            )
        return True

    def stop_profiler(self):
        """Stop profiling for this rank.

        Returns a dict with ``trace_dir`` and ``elapsed`` so the caller
        can report where the trace was written.
        """
        if self.profiler is None:
            return {"trace_dir": self.profiler_dir, "elapsed": 0.0}
        t0 = time.monotonic()
        logger.info("Rank %d: stopping profiler...", self.rank)
        try:
            self.profiler.__exit__(None, None, None)
        except Exception:
            logger.exception("Rank %d: profiler stop failed", self.rank)
        finally:
            self.profiler = None
        elapsed = round(time.monotonic() - t0, 1)
        logger.info(
            "Rank %d: profiler stop completed in %.1fs",
            self.rank,
            elapsed,
        )
        return {"trace_dir": self.profiler_dir, "elapsed": elapsed}

    def debug(self, *args: Any):
        if self.rank == 0:
            logger.info(*args)

    def dummy_execution(self):
        """Execute dummy decode batch for DP synchronization.

        Two mechanisms lean on the fabricated id being -1: this pass flushes
        the previous real step's tokens (`prepare_sampled_ids` reports the
        batch before), and its own land on the key the deferred flag then
        overwrites. What -1 must NOT do is look like a request --
        `get_token_locations`.
        """
        # The scheduler's spec width, not the drafter's. On a P/D producer the
        # real decodes are q=1, and under DP an idle rank's dummy is unified
        # with them: a q=mtp_k+1 dummy would hand the uniform decode
        # collectives a different `running_tokens` than its peers'.
        mtp_k = (
            self.drafter.mtp_k
            if hasattr(self, "drafter") and not self.tokenID_processor.is_pd_producer
            else 0
        )
        mtp_factor = mtp_k + 1
        num_tokens_original = mtp_factor

        seq = Sequence(
            [0] * num_tokens_original,
            block_size=self.block_size,
            id=-1,
        )
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = new_block_table([0])

        spec_tokens = {seq.id: np.zeros(mtp_k, dtype=np.int32)} if mtp_k > 0 else None
        dummy_batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([num_tokens_original], dtype=np.int32),
            total_tokens_num=num_tokens_original,
            total_tokens_num_decode=num_tokens_original,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            is_dummy_run=True,
            num_spec_step=mtp_k,
            scheduled_spec_decode_tokens=spec_tokens,
        )

        self.forward(dummy_batch)
        logger.debug(
            f"{self.label}: dummy batch executed with {dummy_batch.total_tokens_num} tokens"
        )
        return True

    def warmup_max_tokens(self) -> int:
        """Tokens one warmup forward carries: the activation peak KV sizing allows."""
        max_tokens = self.config.max_num_batched_tokens
        if not self.config.enable_dp_attention:
            max_tokens //= get_dp_group().world_size
        pcp_size = self.config.prefill_context_parallel_size
        if pcp_size > 1:
            max_tokens = max(1, max_tokens // pcp_size)
        return max_tokens

    def warmup_model(self):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_model_len = self.config.max_model_len
        dp_size = get_dp_group().world_size
        warmup_max_tokens = self.warmup_max_tokens()

        num_seqs = min(warmup_max_tokens // max_model_len, self.config.max_num_seqs)

        # torch.compile's mark_dynamic can't make a size-1 batch dim dynamic, so
        # a DSpark block drafter (rows == num_seqs) must first-compile at B >= 2
        # (EAGLE gets that free -- its first draft step is a many-row prefill).
        # Other cases only need the usual >= 1 floor.
        drafter = getattr(self, "drafter", None)
        min_seqs = 2 if getattr(drafter, "is_block_drafter", False) else 1
        num_seqs = max(num_seqs, min_seqs)

        # Split the token budget across the seqs so >1 sequences never exceed it
        # (peak memory unchanged); a lone seq keeps up to max_model_len.
        seq_len = max(1, min(max_model_len, warmup_max_tokens // num_seqs))

        if warmup_max_tokens < max_model_len:
            logger.warning(
                f"{self.label}: dp_size={dp_size}, dp_attn={self.config.enable_dp_attention}, "
                f"warmup_max_tokens={warmup_max_tokens} < max_model_len={max_model_len}. "
                f"Using {num_seqs} seq(s) with length {seq_len} for warmup."
            )

        seqs = [
            Sequence(
                [0] * seq_len,
                block_size=self.block_size,
            )
            for _ in range(num_seqs)
        ]
        seqs = {seq.id: seq for seq in seqs}

        num_scheduled_tokens = np.array([seq_len] * num_seqs, dtype=np.int32)
        total_tokens_num = int(num_scheduled_tokens.sum())

        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num,
            total_tokens_num_prefill=total_tokens_num,
            total_seqs_num=num_seqs,
            total_seqs_num_prefill=num_seqs,
            is_dummy_run=True,
        )
        self.forward(dummy_batch)
        self.tokenID_processor.clean()
        torch.cuda.empty_cache()
        logger.info(
            f"{self.label}: warmup_model {time.time() - start_time:.2f} seconds with {num_seqs} reqs {total_tokens_num} tokens"
        )

    def profile_dynamic_chunking(self):
        """Fit chunk overhead from startup dummy forwards (engine RPC)."""
        return self._dynamic_chunking.profile()

    def take_dynamic_chunking_fit(self):
        """Return newly calibrated chunk latency coefficients (engine RPC)."""
        return self._dynamic_chunking.take_fit()

    def allocate_forward_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.torch_dtype
        self.max_bs = self.config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        f32_kwargs = {"dtype": torch.float, "device": self.device}

        # TODO: remove it in forward_context
        self.forward_vars = {
            "input_ids": self.tokenID_processor.input_ids,
            "decode_src": self.tokenID_processor.decode_src,
            "positions": CpuGpuBuffer(
                self.max_num_batched_tokens, publication_group="positions", **i64_kwargs
            ),
            "temperatures": CpuGpuBuffer(
                self.max_bs, publication_group="sampling", **f32_kwargs
            ),
            "top_ks": CpuGpuBuffer(
                self.max_bs, publication_group="sampling", **i32_kwargs
            ),
            "top_ps": CpuGpuBuffer(
                self.max_bs, publication_group="sampling", **f32_kwargs
            ),
            # Keep enough space for MTP decode (max_q_len > 1).
            # `extra_output_dims` lets a model insert dims between N and dim
            # (e.g. DeepSeek-V4 returns the un-reduced mHC residual
            # [N, hc_mult, dim] from forward, with hc_head + LM head deferred
            # to compute_logits). Default `()` keeps the standard 2D layout.
            "outputs": torch.empty(
                self.max_num_batched_tokens,
                *getattr(self.model, "extra_output_dims", ()),
                hidden_size,
                dtype=hidden_type,
            ),
        }
        if self.use_mrope:
            self.forward_vars["mrope_positions"] = CpuGpuBuffer(
                3,
                self.max_num_batched_tokens,
                publication_group="mrope",
                publication_unit="elements",
                **i64_kwargs,
            )
        if hasattr(self, "drafter"):
            self.forward_vars["mtp_k"] = self.drafter.mtp_k
            self.forward_vars.update(self.drafter.metadata_buffers)
            self.forward_vars["num_accepted_tokens"] = CpuGpuBuffer(
                self.max_bs, **i32_kwargs
            )
            # Per in-flight slot via forward_vars; PP ring clones it.
            self.forward_vars["draft_next_tokens"] = CpuGpuBuffer(
                self.max_bs, publication_group="draft_anchors", **i32_kwargs
            )

    def _init_forward_vars_ring(self):
        """Build a ring of independent ``forward_vars`` copies, one per possible
        in-flight pipeline microbatch.

        The head launches up to ``pp_size`` forwards back-to-back without a GPU
        sync, so a single reused ``forward_vars`` set would let microbatch N+1's
        ``prepare_inputs`` overwrite staging buffers microbatch N's kernels are
        still reading. Each in-flight slot gets its own buffer set; reuse of a
        slot is gated by a per-slot CUDA event (see ``_advance_forward_vars`` /
        ``_record_forward_vars_event``), bounding the CPU's GPU lead to the ring
        size even when the head runs middle-chunk or DP-sync dummy batches
        without a GPU sync.

        When ``pp_size == 1`` the ring is the single original dict and advance is
        a no-op, so behavior is unchanged.
        """
        pp_size = self.config.pipeline_parallel_size
        self._fv_idx = 0
        self._stage_h2d_done = None
        if pp_size <= 1:
            self._fv_ring = [self.forward_vars]
            self._fv_slot_events = None
            # Nothing to rotate to, so bound the lead in time instead. See
            # `_gate_staging_reuse`.
            self._stage_h2d_done = torch.cuda.Event()
            logger.info("forward_vars ring: 1 slot (staging reuse gated on its H2D)")
            return

        assert self.enforce_eager, (
            "pipeline_parallel_size > 1 requires eager execution "
            "(--enforce-eager): the forward_vars ring swaps metadata "
            "buffers per microbatch, which is incompatible with CUDAGraph replay."
        )

        def _clone_slot(src: dict) -> dict:
            # CpuGpuBuffers are the per-forward staging buffers, and only their
            # host half can be rewritten while an earlier microbatch's kernels
            # are still reading. Everything else is immutable, unused on the
            # eager PP path, or device-only, where the stream orders the writing
            # kernel after those readers.
            return {
                k: (v.clone() if isinstance(v, CpuGpuBuffer) else v)
                for k, v in src.items()
            }

        self._fv_ring = [self.forward_vars] + [
            _clone_slot(self.forward_vars) for _ in range(pp_size - 1)
        ]
        # One event per slot, marking completion of the last forward that used
        # it. Fresh (never-recorded) events synchronize immediately, so the
        # first pass over the ring is unthrottled.
        self._fv_slot_events = [torch.cuda.Event() for _ in range(pp_size)]
        logger.info(f"forward_vars ring: {pp_size} slots (pipeline parallel)")

    def _init_h2d_publication(self):
        from atom.utils.h2d import PublicationOwner, PublicationRegistry

        registry = PublicationRegistry()
        self.publication_registry = registry
        transport = envs.ATOM_H2D_BACKEND
        if transport not in ("direct", "packed"):
            raise ValueError("ATOM_H2D_BACKEND must be direct or packed")
        self._h2d_owners = []
        self._h2d_groups = []
        for index, variables in enumerate(self._fv_ring):
            event = (
                self._stage_h2d_done
                if self._fv_slot_events is None
                else self._fv_slot_events[index]
            )
            owner = PublicationOwner(self.device, event, registry=registry)
            members = {}
            for name, buffer in variables.items():
                if isinstance(buffer, CpuGpuBuffer) and buffer.publication_group:
                    binding = owner.bind(buffer, name, unit=buffer.publication_unit)
                    members.setdefault(buffer.publication_group, []).append(binding)
            groups = {name: owner.group(name, items) for name, items in members.items()}
            if transport == "packed":
                # One upload immediately before token assembly's first GPU
                # consumer. Sampling has no earlier device consumer.
                if all(name in groups for name in ("sampling", "early", "input_ids")):
                    token_members = ("sampling", "early", "input_ids")
                    if "spec_decode" in groups:
                        token_members += ("spec_decode",)
                    groups["token_inputs"] = owner.group(
                        "token_inputs",
                        [b for name in token_members for b in groups[name].members],
                    )
                if "prefill" in groups and "positions" in groups:
                    groups["prefill_inputs"] = owner.group(
                        "prefill_inputs",
                        groups["prefill"].members + groups["positions"].members,
                    )
            # Some buffers also publish together at a later consumer boundary
            # (e.g. MHA decode shares block tables with the prefill group).
            for name, names in getattr(
                getattr(self, "attn_metadata_builder", None), "h2d_group_members", {}
            ).items():
                groups[name] = owner.group(
                    name, [variables[item]._publication for item in names]
                )
            if transport == "packed":
                for group in owner.use_packed_transport():
                    logger.info(
                        "H2D group %s: %s%s",
                        group.name,
                        group.transport,
                        f" ({group.fallback_reason})" if group.fallback_reason else "",
                    )
            # Cover constructor uploads, ring clones and transport pointer
            # tables before the first preparation, even on another stream.
            event.record()
            self._h2d_owners.append(owner)
            self._h2d_groups.append(groups)
        self.h2d_owner = self._h2d_owners[self._fv_idx]
        self.h2d_groups = self._h2d_groups[self._fv_idx]

    def _advance_forward_vars(self):
        """Rotate to the next in-flight slot before any buffer is written.

        Dummy forwards use the same staging buffers as real forwards, so they
        participate in the ring too. No-op when the ring has a single slot.
        """
        if len(self._fv_ring) == 1:
            return
        self._fv_idx = (self._fv_idx + 1) % len(self._fv_ring)
        # Select the slot now; _gate_staging_reuse waits on its existing event
        # before any producer writes. Avoid waiting twice on the same event.
        self.forward_vars = self._fv_ring[self._fv_idx]
        self.h2d_owner = self._h2d_owners[self._fv_idx]
        self.h2d_groups = self._h2d_groups[self._fv_idx]
        # Token assembly holds aliases outside forward_vars; select both
        # payload and source-index mirrors from this slot before writing.
        self.tokenID_processor.input_ids = self.forward_vars["input_ids"]
        self.tokenID_processor.decode_src = self.forward_vars["decode_src"]

    @_profile_runner_stage
    def _gate_staging_reuse(self):
        """Block until the previous forward's staging H2Ds have executed.

        `_stage` / `CpuGpuBuffer.copy_to_gpu` copy `non_blocking=True` out of
        ONE pinned buffer per name, so the next forward's `buf.np[:n] = arr`
        races the previous forward's DMA. Sampling forwards close that window
        by accident, since `postprocess` synchronizes to read the sampled ids,
        but a chunked prefill's middle chunk returns before `postprocess` and
        closes nothing. Measured: the host reached 4042 packets ahead and the
        GPU read a `batch_id_per_q_token` from a later batch (id 3 in a `bs=2`
        batch), tripping the bounds assert in
        `cu_committed_gpu[batch_id_per_q_token]` -- which wedges the queue with
        no fault line and no traceback.

        One buffer admits one forward of lead, so the gate is depth-1. Decode
        already syncs every step, so it never blocks in steady state; it
        throttles only runs of middle chunks, where the unbounded lead was
        buying nothing. A never-recorded event passes, so the first forward is
        not held.

        DP-sync dummy forwards use these same buffers and can also return while
        their copies are in flight. They must therefore enter this gate and
        record the event just like real forwards.

        The publication owner uses the existing single-slot or PP slot event,
        then opens one ledger epoch before any producer writes host metadata.
        """
        self.h2d_owner.begin()

    @_profile_runner_stage
    def _mark_staging_h2d_enqueued(self):
        """Close the window the gate above waits on.

        Covers this preparation phase's source reads, including registered
        direct copies and grouped publishers. Late V4 TBO preparation resumes
        this epoch and records the completion again after its uploads.
        Independent subsystem uploads retain their own completion protocols.
        """
        self.h2d_owner.finish()

    def _record_forward_vars_event(self):
        """Mark the current slot's forward as done on the GPU stream. Paired
        with the owner wait in ``_gate_staging_reuse``. Called at the end of
        every forward, including DP-sync dummies. No-op when the ring has a
        single slot."""
        if len(self._fv_ring) == 1:
            return
        try:
            self._fv_slot_events[self._fv_idx].record()
        except BaseException:
            self.h2d_owner.fail()
            raise

    def _get_num_kv_heads(self):
        """Return the per-rank number of KV heads."""
        hf_config = self.config.hf_config
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            return hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            return 1

    def _mrope_positions_view(self, num_tokens: int) -> torch.Tensor:
        return self.forward_vars["mrope_positions"].gpu.as_strided(
            (3, num_tokens), (num_tokens, 1)
        )

    def draft_shares_kv_pool(self) -> bool:
        """Whether a draft's attention layers land in the target's pools.

        A draft whose flavor asked for a pool of its own has a
        `draft_kv_builder` and is priced and bound through that instead. One
        spelling, because the layer count and the walk that fills those layers
        have to answer this the same way.
        """
        return bool(
            self.config.speculative_config
            and hasattr(self, "drafter")
            and not hasattr(self, "draft_kv_builder")
        )

    def _sub_pool_specs(self) -> list[SubPoolSpec]:
        """Cache-class declarations from every builder attached to this runner.

        The target builder always, plus an optional `draft_kv_builder` when a
        spec-decode draft owns its own KV. Each builder knows its own tensor
        layout (MLA 576-dim packed, GDN-hybrid full-attn-only, MiMo-V2
        per-layer-type, standard MHA split-K/V); the runner only sums bytes.
        Specs sharing a name merge in `plan_pools`, which is how the draft KV
        joins the target's block ids instead of forming a second pool.
        """
        specs = list(self.attn_metadata_builder.sub_pool_specs())
        if hasattr(self, "draft_kv_builder"):
            specs += self.draft_kv_builder.sub_pool_specs()
        return specs

    def _state_pool_names(self) -> list[str]:
        """The per-request STATE classes the attached builders declare.

        Named rather than counted because both callers want to say which: one
        refuses a model whose `model_type` is missing from the per-req-cache
        set, the other refuses P/D for it.
        """
        return [s.name for s in self._sub_pool_specs() if s.pool is Pool.STATE]

    def _estimate_cudagraph_overhead(self):
        """Estimate GPU memory consumed by CUDA graph capture.

        CUDA graphs allocate a shared memory pool for intermediate activations.
        The pool size is roughly the peak activation memory during a single
        forward pass. We estimate this from the gap between warmup peak and
        current (steady-state) allocation.

        Returns 0 when enforce_eager is set (no CUDA graphs).
        """
        if self.config.enforce_eager:
            return 0
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        activation_bytes = max(peak - current, 0)

        # PIECEWISE pool ~ per_token * Σ(captured num_tokens). per_token from model
        # geometry (hidden*dtype*layers*k), not a magic constant. Under-reserve is
        # safe: capture re-checks live free mem per bucket and skips oversized.
        if self._piecewise_cg_active():
            cap_sizes = self.config.compilation_config.cudagraph_capture_sizes or [
                self.config.max_num_seqs
            ]
            # Captured num_tokens shapes are bs * q. The capture loop uses
            # full_q_len = mtp_k+1 for ANY spec-decode drafter (plain MTP or
            # DSpark), and q=1 for non-spec. Mirror that here or the estimate
            # under-counts Σtok by a factor of q (plain MTP q=4 -> 4x under ->
            # pool est 8.5GB vs actual 33GB -> OOM). DSpark additionally captures
            # multiple q-buckets, so fold its whole bucket set in.
            if hasattr(self, "drafter"):
                full_q = self.drafter.mtp_k + 1
                q_buckets = self._dspark_capture_q_buckets(full_q)
                if (
                    self.drafter.uses_confidence_schedule
                    and os.environ.get("ATOM_PIECEWISE_FINE_TOKENS", "0") == "1"
                ):
                    q_buckets = sorted(set(q_buckets) | set(range(1, full_q + 1)))
            else:
                q_buckets = [1]
            per_token_bytes = self._piecewise_per_token_bytes()
            dp_size = self.config.parallel_config.data_parallel_size
            # Cap the reserved buckets at a fraction of the KV budget so a huge
            # capture list can't starve KV. Use the utilization budget (not raw
            # total) as the reference — it tracks the configured memory envelope.
            budget = self.config.gpu_memory_utilization * torch.cuda.mem_get_info()[1]
            target_reserve = 0.15 * budget
            all_shapes = sorted({bs * q for bs in cap_sizes for q in q_buckets})
            # Mirror the capture-loop token-budget skip: a bucket over
            # `max_num_batched_tokens` is not schedulable and is not captured,
            # so reserving for it would only shrink KV.
            all_shapes = [
                s for s in all_shapes if s <= self.config.max_num_batched_tokens
            ]
            # Mirror the capture-loop DP+spec num_tokens cap (see capture_cudagraph)
            # so the reservation only counts buckets we actually capture.
            if dp_size > 1 and hasattr(self, "drafter"):
                _dp_cap = int(os.environ.get("ATOM_PIECEWISE_DP_MAX_TOKENS", "512"))
                all_shapes = [s for s in all_shapes if s <= _dp_cap]
            captured = []
            acc = 0
            for num_tokens in all_shapes:
                if captured and per_token_bytes * (acc + num_tokens) > target_reserve:
                    break
                captured.append(num_tokens)
                acc += num_tokens
            overhead = int(per_token_bytes * acc)
            logger.info(
                "PIECEWISE cudagraph mem estimate: n_shapes=%d/%d Σtok=%d "
                "per_token=%.3fMB -> overhead=%.2fGB",
                len(captured),
                len(all_shapes),
                acc,
                per_token_bytes / (1 << 20),
                overhead / (1 << 30),
            )
            return overhead
        overhead = activation_bytes * 0.2
        # DSpark RAGGED captures one graph set PER q-bucket, so scale by the
        # number of captured buckets (the pool grows ~linearly with bucket
        # count, each bucket ~one graph set). This stays a safe upper bound:
        # measured per-bucket pool (~1.4GB) << 0.2*act.
        if hasattr(self, "drafter") and self.drafter.uses_confidence_schedule:
            # Match the capture loop's bucket source so we count the graphs
            # actually captured; the pool grows ~linearly with bucket count.
            buckets = self._dspark_capture_q_buckets(self.drafter.mtp_k + 1)
            n_buckets = len(buckets)
            overhead = activation_bytes * 0.2 * n_buckets
            logger.info(
                "DSpark cudagraph mem estimate: buckets=%s n=%d act=%.2fGB "
                "-> overhead=%.2fGB",
                buckets,
                n_buckets,
                activation_bytes / (1 << 30),
                overhead / (1 << 30),
            )
        return int(overhead)

    def freeze_gc_heap(self) -> int:
        """RPC target: freeze this worker's startup heap. Pauses here reached
        979 ms, the largest of any process.

        The count is returned because `busy_loop` replies only `if out is not
        None` -- an RPC target returning None hangs its `wait_out=True` caller.
        """
        return freeze_gc_heap(worker_process_name(self.config, self.rank))

    def get_num_blocks(self) -> dict[str, object]:
        torch.set_default_device(self.device)
        config = self.config
        hf_config = config.hf_config
        if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
            hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # weights + peak activation tensors (PyTorch allocator high-water).
        peak_torch = max(peak, current)
        # RCCL/NCCL buffers etc. held outside the allocator: device-used minus
        # torch-reserved. Ignoring it over-allocates KV and OOMs at runtime.
        non_torch = max((total - free) - torch.cuda.memory_reserved(), 0)

        cudagraph_overhead = self._estimate_cudagraph_overhead()
        safety_margin = int(total * 0.02)

        budget = int(total * config.gpu_memory_utilization)
        non_kv_overhead = peak_torch + non_torch + cudagraph_overhead + safety_margin
        available_for_kv_budget = budget - non_kv_overhead

        # Physical clamp: never exceed what's actually free on the GPU.
        # Subclasses may reserve extra headroom (override point).
        extra_reserve = self._kv_budget_extra_reserve(total)
        available_for_kv_budget -= extra_reserve
        # GPU buffers the KV connector allocates after the KV cache is sized.
        connector_bytes = KVConnectorFactory.kv_budget_reserve_bytes(config)
        if connector_bytes:
            logger.info(
                "Reserving %.2fGB for KV connector buffers", connector_bytes / (1 << 30)
            )
            available_for_kv_budget -= connector_bytes
        # This prevents OOM when other processes share the GPU. Free HBM must
        # cover the connector buffers too.
        free_for_kv = free - connector_bytes
        if connector_bytes and free_for_kv <= 0:
            # Otherwise the KV cache gets nothing and the connector allocates
            # from memory that is not there.
            raise RuntimeError(
                f"Only {free / (1 << 30):.2f}GB is free on the GPU, less than "
                f"the {connector_bytes / (1 << 30):.2f}GB the KV connector "
                "buffers need after the KV cache is sized; free GPU memory or "
                "turn those buffers off (ATOM_PD_MLA_STAGING=0 / "
                "ATOM_PD_MLA_LANDING=0)."
            )
        available_for_kv = min(available_for_kv_budget, free_for_kv)

        torch.set_default_device(None)

        specs = self._sub_pool_specs()

        # Sub-pool sizing is pure arithmetic over the byte budget — see
        # atom/model_ops/attentions/pool_layout/sub_pool_spec.py. STATE classes (GDN
        # recurrent state, the V4 compressor ring, the V4 sliding window) take
        # their floor first because a request cannot run without them; the
        # PAGE class absorbs the rest. Which classes exist, and what they are
        # called, is the backend's business — the runner sizes them and
        # publishes the counts, then every consumer looks up the class it
        # declared itself.
        try:
            plan = plan_pools(specs, available_for_kv, config.max_num_seqs)
        except InsufficientPoolBudget as exc:
            # Minimum gpu_memory_utilization that makes the budget just cover the
            # per-request pools. Rounded UP to the next 0.01 so the printed value
            # is actually sufficient, not the exact threshold.
            min_util = (
                non_kv_overhead + extra_reserve + connector_bytes + exc.reserved_bytes
            ) / total
            min_util_hint = math.ceil(min_util * 100) / 100
            base_msg = (
                f"Per-request cache tensor "
                f"({exc.reserved_bytes / (1 << 30):.2f}GB for "
                f"{exc.entries} slots) exceeds available KV budget "
                f"({available_for_kv / (1 << 30):.2f}GB) at "
                f"--gpu-memory-utilization {config.gpu_memory_utilization:.2f}."
            )
            if (
                available_for_kv_budget > free_for_kv
                or free_for_kv <= exc.reserved_bytes
            ):
                # The physical free-memory clamp is the binding limit, now or
                # once the budget grows — raising --gpu-memory-utilization won't help.
                connector_note = (
                    f"{connector_bytes / (1 << 30):.2f}GB of it reserved for KV "
                    f"connector buffers; "
                    if connector_bytes
                    else ""
                )
                fix_msg = (
                    f" Only {free / (1 << 30):.2f}GB is physically free on the GPU "
                    f"({connector_note}other processes may be holding memory); "
                    f"raising --gpu-memory-utilization will NOT help. Free GPU memory "
                    f"or reduce --max-num-seqs (currently {config.max_num_seqs})."
                )
            elif min_util_hint <= 1.0:
                fix_msg = (
                    f" Set --gpu-memory-utilization >= {min_util_hint:.2f} "
                    f"(this only zeroes out the deficit; use a higher value for "
                    f"actual KV capacity) or reduce --max-num-seqs "
                    f"(currently {config.max_num_seqs})."
                )
            else:
                fix_msg = (
                    f" Even --gpu-memory-utilization 1.0 is insufficient "
                    f"(would need {min_util:.2f}); reduce --max-num-seqs "
                    f"(currently {config.max_num_seqs}) or free GPU memory."
                )
            raise RuntimeError(base_msg + fix_msg) from exc

        # PP stages compute different block counts; block ids must be valid on
        # every stage's KV tensor, so reduce to the global minimum. Fold the
        # result back into the plan before publishing anything: the plan is the
        # single source for every entry count, so it must never disagree with
        # the number the pool is actually built at.
        num_kvcache_blocks = plan.paged_entries
        if config.pipeline_parallel_size > 1 and torch.distributed.is_initialized():
            t = torch.tensor(
                [num_kvcache_blocks], dtype=torch.int64, device=self.device
            )
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            num_kvcache_blocks = int(t.item())
            plan = plan.with_paged_entries(num_kvcache_blocks)

        block_bytes = plan.entry_bytes[plan.paged_class]
        # The whole plan travels to the engine process; BlockManager, the
        # sliding-window pool and the attention builder each index it by the
        # class name they declared. Nothing here needs to know those names.
        self.pool_plan = plan
        config.pool_entries = dict(plan.entries)
        config.pool_entries_per_req = dict(plan.entries_per_req)
        # Keep runtime state metadata out of Config.
        transfer = self.attn_metadata_builder.state_transfer()
        uses_paged_state = transfer.copies
        if uses_paged_state and config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "PAGE-backed state checkpoints do not yet support pipeline "
                "parallelism: every stage must first agree on one atomic "
                "checkpoint/unit ownership transaction"
            )
        if uses_paged_state and config.enable_rapidserve:
            raise RuntimeError(
                "PAGE-backed state checkpoints do not yet support RapidServe "
                "prefill/decode disaggregation"
            )
        checkpoint_spec = None
        if uses_paged_state:
            if plan.paged_class is None:
                raise RuntimeError(
                    "PAGE-backed state checkpoints require a PAGE sub-pool"
                )
            slot_bytes = int(plan.entry_bytes[STATE_SLOT_CLASS])
            # None means the backend has not narrowed its image: carry it all.
            narrowed = self.attn_metadata_builder.checkpoint_image_bytes()
            # The charged entry can be larger than the bytes a transfer
            # publishes: MHA alignment padding is not a region, and a draft
            # pool merged into the same class is ordinary KV after the
            # checkpoint regions. The backend names the published unit.
            page_unit_bytes = int(plan.entry_bytes[plan.paged_class])
            published_unit = getattr(
                self.attn_metadata_builder, "checkpoint_page_unit_bytes", None
            )
            if callable(published_unit):
                published = published_unit()
                if published is not None:
                    published = int(published)
                    if published <= 0 or published > page_unit_bytes:
                        raise RuntimeError(
                            f"checkpoint PAGE unit is {published} B but the "
                            f"paged pool entry is {page_unit_bytes} B"
                        )
                    page_unit_bytes = published
            checkpoint_spec = PagedStateCheckpointSpec(
                page_unit_bytes=page_unit_bytes,
                slot_bytes=slot_bytes,
                image_bytes=slot_bytes if narrowed is None else int(narrowed),
                layout_id=transfer.paged_layout_id,
            )
            logger.info(
                "PAGE-backed state checkpoints enabled: unit_bytes=%d, "
                "slot_bytes=%d, image_bytes=%d (%.1f%% of a slot), "
                "units_per_checkpoint=%d, layout=%s",
                checkpoint_spec.page_unit_bytes,
                checkpoint_spec.slot_bytes,
                checkpoint_spec.image_bytes,
                100.0 * checkpoint_spec.image_bytes / checkpoint_spec.slot_bytes,
                checkpoint_spec.units_per_checkpoint,
                checkpoint_spec.layout_id,
            )
        state_runtime = StateRuntime(
            transfer=transfer,
            checkpoint_spec=checkpoint_spec,
        )
        self.state_runtime = state_runtime
        for name in sorted(plan.entries):
            logger.info(
                f"sub-pool {name}: entries={plan.entries[name]}, "
                f"entry_bytes={plan.entry_bytes[name]}, "
                f"reserved={plan.reserved_bytes[name] / (1 << 30):.2f}GB"
            )

        logger.info(
            f"Memory budget: total_gpu={total / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB, "
            f"utilization={config.gpu_memory_utilization}, "
            f"budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"non_torch={non_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"available_for_kv={available_for_kv / (1 << 30):.2f}GB, "
            f"block_bytes={block_bytes}, "
            f"num_kvcache_blocks={num_kvcache_blocks}"
        )
        # Concurrent-capacity table: at each context-length percentage of
        # max_model_len, how many requests can simultaneously hold their
        # KV in the pool. Per-req block usage = ceil(ctx_len/block_size).
        # Active Slots are reserved; PAGE checkpoints borrow from the paged pool.
        max_model_len = config.max_model_len
        cap = config.max_num_seqs
        dcp_w = max(1, getattr(config, "decode_context_parallel_size", 1) or 1)
        pct_lines = []
        for pct in (10, 30, 50, 70, 90, 100):
            ctx = max(1, max_model_len * pct // 100)
            local_ctx = math.ceil(ctx / dcp_w)
            blocks_per_req = math.ceil(local_ctx / self.block_size)
            block_bound = (
                num_kvcache_blocks // blocks_per_req if blocks_per_req > 0 else 0
            )
            max_conc = min(cap, block_bound) if cap > 0 else block_bound
            bound_label = (
                "slots" if cap > 0 and max_conc == cap < block_bound else "blocks"
            )
            local_note = f" (local {local_ctx:>7})" if dcp_w > 1 else ""
            pct_lines.append(
                f"  {pct:>3}% ({ctx:>7} tok){local_note}: {blocks_per_req:>6} blk/req "
                f"→ max_concurrent={max_conc:<5} (bound by {bound_label})"
            )
        logger.info(
            f"Concurrent capacity vs context length "
            f"(max_model_len={max_model_len}, block_size={self.block_size}, "
            f"max_slots={cap}, pool_blocks={num_kvcache_blocks}"
            + (f", dcp={dcp_w} (blk/req is per-rank)" if dcp_w > 1 else "")
            + "):\n"
            + "\n".join(pct_lines)
        )

        assert num_kvcache_blocks > 0, (
            f"Not enough memory for KV cache with block size({self.block_size}). "
            f"At least 1 block ({block_bytes / (1 << 20):.2f}MB) is required, "
            f"but available_for_kv={available_for_kv / (1 << 20):.2f}MB "
            f"(budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"non_torch={non_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB)"
        )
        # get_num_blocks runs in the RUNNER subprocess, so nothing it writes
        # to `config` is visible to the engine process that builds
        # BlockManager. Ship the whole per-class entry table across instead of
        # a hand-picked field per architecture; consumers over there look up
        # the class they declared.
        return {
            "num_kvcache_blocks": num_kvcache_blocks,
            "pool_entries": dict(plan.entries),
            "pool_entries_per_req": dict(plan.entries_per_req),
            "state_runtime": state_runtime.to_wire(),
        }

    def _carve_paged_pool(self, blocks: int, buf=None):
        """One allocation for every paged pool, and each builder's region of it.

        The target's KV, its scales, whatever indexer cache rides along and a
        draft's sibling pool were up to five allocations under five names. One
        buffer here, for the reason the declarations were collapsed: a block's
        bytes are one fact, and every name it is stored under is a place the
        next reader can disagree with it. `kv_cache` is what is left, which is
        what the IPC export ships and the sleep path frees.

        Returns `(buffer, regions, builders)`, positionally aligned. `buf` is
        the imported pool on the P/D decode side; that side carves with the
        same walk rather than being told the offsets, so the two cannot drift.
        A builder that allocates its own answers zero and gets an empty region.
        """
        builders = [self.attn_metadata_builder]
        if hasattr(self, "draft_kv_builder"):
            builders.append(self.draft_kv_builder)
        # Before pricing, not between pricing and filling: the price below
        # comes from the walk.
        for builder in builders:
            builder.invalidate_pool_rows()
        sizes = [b.paged_pool_bytes(blocks) for b in builders]
        # A builder either allocates its own pool or takes exactly what its own
        # PAGE spec was charged -- nothing between. It catches the two hooks
        # disagreeing (separate overrides; a field added to one), not a stale
        # walk, which both readings share. A startup check and not a unit test
        # because what it catches surfaces while serving, never as a bad number
        # here.
        for builder, size in zip(builders, sizes):
            if not size:
                continue
            charged = blocks * sum(
                s.entry_bytes for s in builder.sub_pool_specs() if s.pool is Pool.PAGE
            )
            if size != charged:
                raise ValueError(
                    f"{type(builder).__name__} wants {size} B of the paged pool "
                    f"at {blocks} blocks but was charged {charged} B"
                )
        offsets, total = plan_regions(sizes)
        if buf is None:
            buf = torch.zeros(total, dtype=torch.uint8, device=self.device)
        elif buf.numel() < total:
            raise ValueError(
                f"the imported paged pool holds {buf.numel()} B but this side's "
                f"declarations want {total} B at {blocks} blocks"
            )
        regions = [buf[o : o + n] for o, n in zip(offsets, sizes)]
        return buf, regions, builders

    def _back_paged_pools(self, blocks: int, buf=None):
        """Back every builder's paged pool and publish what comes back.

        The one place, because there are two callers: this side allocates the
        buffer and the decode side of a P/D pair imports it. What the hook
        returns -- the aligned indexer dimension, the compact layer maps,
        GLM-5.3's k-pool tail -- has to reach its readers as a runner attribute
        on both, and a path that only backed the pool leaves the other half
        dereferencing attributes nobody set.
        """
        pool, regions, builders = self._carve_paged_pool(blocks, buf=buf)
        for builder, region in zip(builders, regions):
            for name, value in builder.allocate_kv_cache_tensors(
                blocks=blocks, buf=region
            ).items():
                setattr(self, name, value)
            logger.info(
                "%s caches %s",
                type(builder).__name__,
                ", ".join(
                    f"{n} {kind} rows" for kind, n in builder.row_counts().items()
                )
                or "nothing",
            )
        return pool

    def allocate_kv_cache(self, num_kvcache_blocks):
        pre_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        config = self.config
        config.num_kvcache_blocks = num_kvcache_blocks
        hf_config = config.hf_config
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            num_kv_heads = 1
        # Promote to self so attention builders' build_kv_cache_tensor()
        # hooks can access it without re-deriving from hf_config.
        self.num_kv_heads = num_kv_heads
        self.aligned_index_dim = None  # set below for DeepSeek-V3.2

        # Primary KV cache allocation: one buffer, one region per builder, each
        # owning the layout inside its own. A draft that cannot share the
        # target's pool is simply the next region -- no second allocation, and
        # nothing that asks whether its attention matches the target's.
        #
        # What comes back is only what a region cannot carry: scalars like
        # `aligned_index_dim` and the compact layer maps, setattr'd here so
        # their readers find them where they always have.
        pool = self._back_paged_pools(num_kvcache_blocks)
        if pool.numel():
            self.kv_cache = pool

        # Per-request cache allocation (model-agnostic, delegated to the
        # attention metadata builder). For GDN this returns
        # `{"mamba_k_cache": ..., "mamba_v_cache": ...}`; for stateless
        # attentions it returns an empty dict (no-op). Values are setattr'd
        # on `self` so model layers can access them as `model_runner.<name>`.
        per_req_state = self.attn_metadata_builder.allocate_per_req_cache(
            self.pool_plan.entries
        )
        for name, value in per_req_state.items():
            setattr(self, name, value)
        # The pools are reachable through `self` only now, which is the
        # earliest the builder can touch its own addresses — and the last
        # moment before a request could.
        self.attn_metadata_builder.warmup_per_req_cache()

        # Build KVCacheConfig
        # lirong TODO: This is a simple solution to build KVCacheConfig,
        # models with only one type of attention, but not support multi-type of attention models.
        # We need to support it by kv_cache_group in the future.

        # Prepare list of models to bind KV cache
        models_to_bind = [("target", self.model)]
        if self.config.speculative_config and hasattr(self, "drafter"):
            models_to_bind.append(("draft", self.drafter.model))

        # Keyed by each module's own global layer_num, which is what it looks
        # up at forward time (`kv_cache_data[f"layer_{self.layer_num}"]`). The
        # walk's ordinal used to be passed down and stood in for this, for a
        # pool row and for an index-map key; under PP it is none of the three.
        kv_cache_data = {}
        for model_name, model in models_to_bind:
            logger.info("Binding KV cache for the %s model", model_name)
            for module in model.modules():
                # A draft that owns an independent KV pool binds through its
                # sibling builder first; for unrecognized modules it returns
                # None and we fall through to the target builder.
                kv_cache_tensor = None
                if model_name == "draft" and hasattr(self, "draft_kv_builder"):
                    kv_cache_tensor = self.draft_kv_builder.build_kv_cache_tensor(
                        module
                    )
                # Per-attention-type binding is owned by the attention
                # metadata builder; ModelRunner only walks modules and
                # collects the resulting KVCacheTensor entries. The builder
                # returns None for modules it does not recognize (so a
                # sibling module like nn.LayerNorm is silently skipped).
                if kv_cache_tensor is None:
                    kv_cache_tensor = self.attn_metadata_builder.build_kv_cache_tensor(
                        module
                    )
                if kv_cache_tensor is not None:
                    kv_cache_data[f"layer_{module.layer_num}"] = kv_cache_tensor

        transfer_tensors = self.attn_metadata_builder.get_kv_transfer_tensors()
        if transfer_tensors is not None:
            # The tier is built inside `register_kv_caches` and needs
            # `state_entry_views` to name the bytes it packs. This is the only
            # place the builder and the connector are both in scope.
            transfer_tensors.state_backend = self.attn_metadata_builder
        if hasattr(self, "draft_kv_builder") and transfer_tensors is not None:
            transfer_tensors.merge_pages(
                self.draft_kv_builder.get_kv_transfer_tensors()
            )
        if transfer_tensors is not None:
            # After the draft's regions are in, and here because this is the
            # only place holding both the complete region list and the
            # scheduler's block count. The builders cannot answer this: each
            # counts in its own page, and the transfer protocol addresses
            # ``req.block_ids``, which are the scheduler's.
            transfer_tensors.set_block_count(num_kvcache_blocks)
        # The same count reaches the codecs, which likewise index by scheduler
        # block: MLA's cache is allocated in page-size-1 rows, so its own count
        # is block_ratio times larger and would make a codec treat one token as
        # a complete scheduler block.
        set_kv_cache_data(
            kv_cache_data,
            config,
            transfer_tensors,
            num_blocks=num_kvcache_blocks,
        )

        # Cross-validate: compare estimated vs actual KV cache allocation.
        # `actual_kv_bytes` includes BOTH the unified pool tensors (counted by
        # `block_bytes × num_blocks`) AND the per-request cache tensors (state
        # buffers + SWA window prefix embedded in unified_kv). The budget
        # math in `get_num_blocks()` reserves both separately, so the cross-
        # check must mirror that — otherwise it spuriously fires for any
        # backend that declares a per-request state pool (V4, GDN).
        post_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        actual_kv_bytes = post_alloc - pre_alloc
        # Each sub-pool contributes `entry_bytes × entries`. The counts come
        # straight from the sizing plan — which already absorbed the pipeline-
        # parallel reconciliation — so this mirrors the budget by construction
        # rather than re-deriving it.
        expected_kv_bytes = self.pool_plan.total_reserved_bytes
        if expected_kv_bytes > 0:
            diff_pct = abs(actual_kv_bytes - expected_kv_bytes) / expected_kv_bytes
            # 3% threshold: budget formula matches allocation exactly, but the
            # measured `post_alloc - pre_alloc` includes allocator alignment
            # (round to 256 B / 16 MiB segments) and whatever transient the
            # builders touch while initializing their pools, accounting for
            # ~2% noise on multi-GiB pools. Lower thresholds spuriously fire.
            if diff_pct > 0.03:
                logger.warning(
                    f"KV cache allocation mismatch: "
                    f"expected={expected_kv_bytes / (1 << 30):.3f}GB, "
                    f"actual={actual_kv_bytes / (1 << 30):.3f}GB, "
                    f"diff={diff_pct:.1%}"
                )

        # Skip on single-rank: a world_size==1 barrier is a no-op but still
        # forces lazy NCCL communicator creation (CUDA-allocs its buffers),
        # which can OOM/fail on single-card runs. The process group stays
        # initialized so get_tp_group() and friends keep working.
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        return True

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        dp_size = self.config.parallel_config.data_parallel_size
        dp_rank = self.config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1:
            # Early exit.
            return 0, None
        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank
        )
        max_tokens_across_dp = int(torch.max(num_tokens_across_dp))

        return max_tokens_across_dp - num_tokens, num_tokens_across_dp

    def _maybe_create_tbo_slices(
        self,
        batch,
        is_prefill,
        scheduled_bs,
        scheduled_tokens,
        num_scheduled_tokens,
        tbo_collective_active: bool,
    ):
        """Create TBO ubatch slices when the collective DP decision is True.

        With the packed-reduce path the eligibility (local + cross-DP AND)
        is decided in ``ForwardMode.decide``; here we just realise the split.
        """
        if not tbo_collective_active:
            return None

        tbo_num_reqs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs
        # tbo_collective_active is the OR-reduced cross-DP decision: this rank
        # is committed to splitting even if it's below ATOM_TBO_PREFILL_MIN_TOKENS
        # (a peer cleared the bar). force=True bypasses the local min-token gate
        # so we don't desync from peers and hang.
        ubatch_slices = maybe_create_ubatch_slices(
            num_reqs=tbo_num_reqs,
            num_tokens=scheduled_tokens,
            is_prefill=is_prefill,
            num_scheduled_tokens=num_scheduled_tokens if is_prefill else None,
            force=True,
        )
        if ubatch_slices is not None:
            logger.debug(
                f"[TBO] splitting {'prefill' if is_prefill else 'decode'} batch: "
                f"num_reqs={tbo_num_reqs}, ubatches={len(ubatch_slices)}"
            )
        return ubatch_slices

    def _local_tbo_eligibility(
        self, batch: ScheduledBatch
    ) -> tuple[bool, bool, int, int]:
        """This rank's TBO answer, and the PCP flags that ride with it.

        `(meets_min_tokens, can_split, ub0_tokens, ub1_tokens)`. The first two
        are reduced across DP inside the step's one packed all_gather -- OR and
        AND respectively -- so this is the local half of that question, not a
        decision. Sets the PCP+TBO routing flags as a side effect because they
        come out of the same sizing.
        """
        is_prefill = batch.total_tokens_num_prefill > 0
        if not self.config.enable_tbo:
            self._pcp_tbo_balanced_active = False
            self._pcp_bal_groups = None
            return (False, False, 0, 0)

        local_tbo = local_tbo_precompute(
            self.config, batch, is_prefill, batch.num_scheduled_tokens
        )

        # PCP+TBO prefill: split requests into two GROUPS at a request boundary
        # (never split a sequence's tokens), so each ubatch = "non-TBO PCP on a
        # request subset". Requires num_reqs >= 2 (request-boundary split needs
        # two non-empty groups); bs=1 falls back to non-TBO.
        pcp_size = self.config.prefill_context_parallel_size
        # Read by build_ubatch / run_model / prepare_prefill to route the
        # per-group path; reset every step so a stale value cannot route one.
        self._pcp_tbo_balanced_active = False
        self._pcp_bal_groups = None
        if is_prefill and pcp_size > 1 and not batch.is_dummy_run:
            # PCP is always dp=1, so the single-rank path returns this verbatim
            # as `tbo_collective_active`; ub0/ub1 are only read under dp > 1.
            local_tokens = batch.total_tokens_num_prefill // pcp_size
            eligible = batch.total_seqs_num_prefill >= 2 and local_tokens >= 2
            ub0 = local_tokens // 2
            local_tbo = (eligible, eligible, ub0, local_tokens - ub0)
            self._pcp_tbo_balanced_active = eligible
        return local_tbo

    def _dspark_apply_q_bucket(self, batch: ScheduledBatch) -> int | None:
        """Shrink this decode step's verify length to one CUDA-graph bucket q,
        and return it -- or None when nothing shrank and the step keeps the
        configured `num_spec_step + 1`.

        q = quantize_up(max ell_i + 1) over the batch (ell_i = last step's
        per-req schedule). All seqs then forward q tokens (anchor + q-1 drafts)
        instead of mtp_k+1, and replay picks the (bs, q) graph; the dropped
        draft suffix is re-drafted next step -> lossless.

        Mutates only the worker's batch copy (counts + scheduled_spec_decode_
        tokens truncated to q-1); KV stays reserved at mtp_k+1. No-op unless
        DSpark confidence scheduling is on and this is a pure-decode batch.
        The bucket math lives in `apply_q_bucket` so CPU CI can run it without
        importing this module (which loads AITER).
        """
        from atom.spec_decode.dspark_scheduler import apply_q_bucket

        drafter = getattr(self, "drafter", None)
        dspark = getattr(getattr(self, "config", None), "dspark", None)
        return apply_q_bucket(
            batch,
            drafter,
            dspark,
            apply_ragged=lambda scheduled_bs, full_q, by_req: self._dspark_apply_ragged(
                batch, scheduled_bs, full_q, by_req
            ),
        )

    def _dspark_apply_ragged(self, batch, scheduled_bs, full_q, by_req):
        """DSpark per-request RAGGED verify (paper §5.2 avoid-padding).

        Sets num_scheduled_tokens[i] = max(ell_i, num_bonus_i) + 1, clamped to
        [1, full_q] -- both terms request i's own, so two requests that accepted
        2 and 5 forward 3 and 6 tokens, not 6 and 6. Downstream V4 attn is
        marker-driven (cu_seqlens etc.), so ragged lengths flow through
        unchanged; the dropped draft suffix is re-drafted next step (lossless).
        KV stays reserved at mtp_k+1.
        """
        old_nst = batch.num_scheduled_tokens

        tp = getattr(self, "tokenID_processor", None)
        prev_b = getattr(tp, "prev_batch", None) if tp is not None else None
        cur_req = list(batch.req_ids[:scheduled_bs])
        prev_req = list(prev_b.req_ids) if prev_b is not None else None
        # Shrinking needs the previous batch to be exactly this decode set in
        # the same order (no new/prefill seqs, no reorder), because the per-
        # request lengths below are read off that batch's `num_bonus`. Any
        # deviation → boundary step.
        if prev_req is None or prev_req != cur_req:
            return  # boundary / reorder step: skip ragged, stay rectangular

        # Positional, which is what the `prev_req == cur_req` guard buys: entry
        # i is request i's own count from the step that produced it, before
        # `prepare_input_ids` remaps the deferred rows.
        nb = batch.num_bonus[:scheduled_bs]

        from atom.spec_decode.dspark_scheduler import (
            quantize_to_bucket,
            ragged_verify_lens,
            resolve_q_buckets,
        )

        # Bounded per request by nb[i]+1 (its anchor must stay inside its OWN
        # segment) and old_nst[i] (a stale ell must not grow a seq past what the
        # scheduler scheduled). None -> some request's bounds cross.
        new_len = ragged_verify_lens(
            (by_req.get(rid) for rid in batch.req_ids[:scheduled_bs]),
            full_q,
            nb,
            old_nst[:scheduled_bs],
        )
        if new_len is None:
            return  # stay rectangular

        if not (new_len < old_nst[:scheduled_bs]).any():
            return  # nothing to shrink this step -> Phase-1 layout

        # q_eff, and the replay-shape feasibility check, BEFORE anything on the
        # batch is rewritten. Shrinking the flat token layout is only safe if the
        # replay can follow it down; when it cannot, the rebuild is what makes
        # the step unsafe, so the decision has to come first.
        #   * max_seqlen_q (scalar) q_eff : the PER-SEQ length bound
        #     (>= max(new_len), quantized up to a captured bucket). Per-seq
        #     structures (compressor grid, rectangular indexer) size by it, so no
        #     seq can overflow them. It is NOT the total compute size -- that is
        #     the flat num_tokens bucket (running_tokens), sized to the
        #     real sum, so a long-tail seq no longer inflates the batch row count.
        buckets = resolve_q_buckets(self.config.dspark.ragged_graph_sizes, full_q)
        if self.enforce_eager:
            # Eager: no graph → capacity == exact Σ (no bucket). Scalar = batch max
            # real len (positions/attn bound); layout is pure flat Σ.
            q_eff = int(new_len.max()) if scheduled_bs > 0 else full_q
        else:
            # Graph: q_eff = smallest bucket >= the real MAX per-seq len, so no
            # seq ever exceeds q_eff. Per-seq structures (compressor grid,
            # rectangular indexer) size by q_eff and can't overflow -- no separate
            # full_q cap needed. The TOTAL compute size is `running_tokens`,
            # settled apart from this by `ForwardMode`, so q_eff no longer needs
            # to track the sum/avg.
            q_eff = (
                quantize_to_bucket(int(new_len.max()), buckets)
                if scheduled_bs > 0
                else full_q
            )
            # Under FULL cudagraphs nothing flat is recorded, so a packed run has
            # no width to land on and shrinking buys nothing -- the replay is
            # rectangular either way. WHICH width it lands on is not asked here:
            # `ForwardMode.decide` settles that on the DP-agreed total, and a
            # second search on this rank's own count is how the two came to
            # disagree. A miss there forwards the run eagerly, which is what the
            # branch above already does under `enforce_eager`.
            if not self._piecewise_cg_active():
                return  # stay rectangular

        # Rebuild scheduled_tokens (flat) to the ragged per-seq layout: keep the
        # first new_len[i] of each seq's old segment (token[0]=anchor, rest=draft
        # placeholders already populated by the scheduler from seq.token_ids).
        sched = np.asarray(batch.scheduled_tokens)
        old_cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
        np.cumsum(old_nst[:scheduled_bs], out=old_cu[1:])
        new_cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
        np.cumsum(new_len, out=new_cu[1:])
        total_new = int(new_cu[-1])
        new_sched = np.empty(total_new, dtype=sched.dtype)
        for i in range(scheduled_bs):
            s_old = int(old_cu[i])
            s_new = int(new_cu[i])
            new_sched[s_new : s_new + new_len[i]] = sched[s_old : s_old + new_len[i]]
        batch.scheduled_tokens = new_sched

        nst = old_nst.copy()
        nst[:scheduled_bs] = new_len
        batch.num_scheduled_tokens = nst
        prefill_tok = int(batch.total_tokens_num_prefill)
        batch.total_tokens_num_decode = total_new
        batch.total_tokens_num = prefill_tok + total_new
        # No marker is published for "this step shrank": a shrunk step is
        # `num_scheduled_tokens` with smaller entries, and every reader reads
        # those. `scheduled_spec_decode_tokens` keeps its full width --
        # `prepare_input_ids` slices it per request (`spec[i, :len_i - 1]`).
        return int(q_eff)

    def _verifies_drafts(self, batch: ScheduledBatch, is_prefill: bool) -> bool:
        """Whether this forward scores drafts with the rejection sampler.

        A P/D producer keeps the drafter only to write its context KV; the
        scheduler never pads drafts for it, so its decodes (the step that
        surfaces a deferred T0) are q=1 and have nothing to verify.
        """
        return (
            not is_prefill
            and hasattr(self, "drafter")
            and not self.tokenID_processor.is_pd_producer
            and not batch.is_dummy_run
        )

    def prepare_inputs(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        spec_decode_indices: tuple[np.ndarray, int] | None = None,
    ):
        # Always supplied, settled in `prepare_model` (which is where the reason
        # lives). The q-bucket shrink ran there too, so `batch` is already
        # reduced here.
        is_prefill = batch.total_tokens_num_prefill > 0
        scheduled_bs = batch.total_seqs_num
        scheduled_tokens = batch.total_tokens_num
        num_scheduled_tokens = batch.num_scheduled_tokens
        sync = forward_mode.sync
        num_tokens_across_dp = None if sync is None else sync.num_tokens_across_dp
        tbo_collective_active = forward_mode.tbo_collective_active
        ub_max_tokens_across_dp = None if sync is None else sync.ub_max_tokens_across_dp
        ub_tokens_across_dp = None if sync is None else sync.ub_tokens_across_dp
        running_tokens_are_unified = forward_mode.running_tokens_are_unified

        if not tbo_collective_active:
            self._pcp_tbo_balanced_active = False

        # The step's two units, read from where they were settled: `running_bs`
        # sizes everything per-sequence, `running_tokens` everything per-row.
        running_bs = forward_mode.running_bs
        running_tokens = forward_mode.running_tokens
        spec_decode_metadata = None
        if self._verifies_drafts(batch, is_prefill):
            _, lens, cu = self.attn_metadata_builder.decode_spans(batch)
            # Inside the staging window: without the token group it publishes
            # the indices itself. Packed indices already share the token upload.
            spec_decode_metadata = self.drafter.calc_spec_decode_metadata(
                lens, cu[1:], input_ids, prepared_indices=spec_decode_indices
            )
        attn_metadata, positions = self.attn_metadata_builder.build(
            batch=batch,
            running_bs=running_bs,
            running_tokens=running_tokens,
            max_seqlen_q=forward_mode.max_seqlen_q,
        )
        self.attn_metadata_builder.prepare_model_inputs(input_ids, attn_metadata)
        context = Context(
            positions=positions,
            is_prefill=is_prefill,
            is_dummy_run=batch.is_dummy_run,
            scheduled_bs=forward_mode.scheduled_bs,
            scheduled_tokens=scheduled_tokens,
            running_bs=running_bs,
            running_tokens=running_tokens,
            running_tokens_are_unified=running_tokens_are_unified,
            forward_mode=forward_mode,
        )

        pcp_size = self.config.prefill_context_parallel_size
        _pcp_tbo_balanced = (
            is_prefill
            and pcp_size > 1
            and tbo_collective_active
            and not batch.is_dummy_run
            and getattr(self, "_pcp_tbo_balanced_active", False)
        )
        if _pcp_tbo_balanced:
            # Request-boundary split for PCP+TBO prefill (see
            # _build_pcp_balanced_slices). forward_vars stay GLOBAL here.
            ubatch_slices, self._pcp_bal_groups = self._build_pcp_balanced_slices(
                batch, num_scheduled_tokens, pcp_size
            )
        else:
            ubatch_slices = self._maybe_create_tbo_slices(
                batch,
                is_prefill,
                scheduled_bs if not is_prefill else 0,
                scheduled_tokens,
                num_scheduled_tokens,
                tbo_collective_active,
            )

        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=self.config,
            context=context,
            num_tokens=scheduled_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            spec_decode_metadata=spec_decode_metadata,
            ubatch_slices=ubatch_slices,
            ub_max_tokens_across_dp=ub_max_tokens_across_dp,
            ub_tokens_across_dp=ub_tokens_across_dp,
        )

    @h2d_producer("sampling")
    def prepare_sample(
        self, batch: ScheduledBatch, *, publication_group=None
    ) -> tuple[
        torch.Tensor, int | torch.Tensor | None, float | torch.Tensor | None, bool, bool
    ]:
        bs = batch.total_seqs_num

        # Check on CPU whether all requests are greedy (temperature=0)
        all_greedy = (batch.temperatures == 0).all()

        # Check on CPU whether any fan-out sibling needs per-row random noise.
        # Missing attribute (e.g. dummy runs, older callers) -> False.
        noise = getattr(batch, "needs_independent_noise", None)
        needs_independent_noise = noise is not None and bool(noise.any())

        temp_buffer = self.forward_vars["temperatures"]
        # Clamp temperatures on CPU to avoid division by zero in sampler
        np.maximum(batch.temperatures, SAMPLER_EPS, out=temp_buffer.np[:bs])

        # Check on CPU whether filtering is needed to avoid GPU sync in sampler.
        # If no filtering needed, return None to skip GPU copy entirely.
        needs_topk = (batch.top_ks != -1).any()
        needs_topp = (batch.top_ps < 1.0).any()

        # Uniform filters are already known on the CPU. Keep them as scalars
        # for AITER's fast dispatch instead of uploading and reading them back.
        top_ks = top_ps = None
        top_k_count = top_p_count = None
        if needs_topk:
            if bs == 1 or (batch.top_ks == batch.top_ks[0]).all():
                top_ks = int(batch.top_ks[0])
            else:
                top_k_buffer = self.forward_vars["top_ks"]
                top_k_buffer.np[:bs] = batch.top_ks
                top_ks = top_k_buffer.gpu[:bs]
                top_k_count = bs

        if needs_topp:
            if bs == 1 or (batch.top_ps == batch.top_ps[0]).all():
                top_ps = float(np.float32(batch.top_ps[0]))
            else:
                top_p_buffer = self.forward_vars["top_ps"]
                top_p_buffer.np[:bs] = batch.top_ps
                top_ps = top_p_buffer.gpu[:bs]
                top_p_count = bs

        group = (
            self.h2d_groups["sampling"]
            if publication_group is None
            else publication_group
        )
        counts = group.counts
        counts[group.indices["temperatures"]] = bs
        counts[group.indices["top_ks"]] = top_k_count
        counts[group.indices["top_ps"]] = top_p_count
        if publication_group is None:
            group.publish(counts)
        temperatures = temp_buffer.gpu[:bs]

        return temperatures, top_ks, top_ps, all_greedy, needs_independent_noise

    @_profile_runner_stage
    def prepare_model(self, batch: ScheduledBatch):
        shrunk_q = self._dspark_apply_q_bucket(batch)
        # The step's shape, settled once. Here rather than in prepare_inputs
        # because DSpark under DP needs the reduced query length BEFORE
        # prepare_input_ids sizes the buffer, and because one call site is the
        # only way the step is guaranteed a single cross-DP collective.
        dp_size = self.config.parallel_config.data_parallel_size
        forward_mode = ForwardMode.decide(
            batch=batch,
            dp_size=dp_size,
            dp_group=get_dp_group().cpu_group if dp_size > 1 else None,
            enforce_eager=self.enforce_eager,
            step_needs_eager=self.attn_metadata_builder.step_needs_eager(batch),
            capture_sizes=self.capture_sizes_np,
            captured_tokens=(
                self._piecewise_sorted_tokens if self._piecewise_cg_active() else None
            ),
            is_block_drafter=(
                hasattr(self, "drafter") and self.drafter.is_block_drafter
            ),
            tbo_on=self.config.enable_tbo,
            local_tbo=self._local_tbo_eligibility(batch),
            max_seqlen_q=(batch.num_spec_step + 1 if shrunk_q is None else shrunk_q),
            graph_shapes=self._producer_graph_shapes(),
            dp_stage_report=(
                self.attn_metadata_builder.dp_stage_report(batch)
                if dp_size > 1
                else None
            ),
        )
        # Stash the DP-wide prefill OR for the EPLB prefill gate; reused free by
        # on_forward_pass_end when the DP group == the migration (EP) group.
        self._eplb_any_rank_has_prefill = (
            None
            if forward_mode.sync is None
            else forward_mode.sync.any_rank_has_prefill
        )
        total_tokens_num = batch.total_tokens_num
        assert total_tokens_num > 0

        token_group = self.h2d_groups.get("token_inputs")
        spec_decode_indices = None
        temperatures, top_ks, top_ps, all_greedy, needs_independent_noise = (
            self.prepare_sample(batch, publication_group=token_group)
        )
        if token_group is None:
            self.attn_metadata_builder.publish_cu_seqlens_q(batch, forward_mode)
            input_ids = self.tokenID_processor.prepare_input_ids(
                batch, forward_mode.max_seqlen_q
            )
        else:
            cu_count = self.attn_metadata_builder.prepare_cu_seqlens_q(
                batch, forward_mode
            )
            token_group.counts[token_group.indices["cu_seqlens_q"]] = cu_count
            spec_group = self.h2d_groups.get("spec_decode")
            if spec_group is not None:
                if self._verifies_drafts(batch, batch.total_tokens_num_prefill > 0):
                    _, lens, cu = self.attn_metadata_builder.decode_spans(batch)
                    spec_decode_indices = self.drafter.prepare_spec_decode_indices(
                        lens, cu[1:], token_group
                    )
                else:
                    # A preceding decode may have filled these counts. Forwards
                    # that verify nothing must not publish its stale indices.
                    for binding in spec_group.members:
                        token_group.counts[token_group.indices[binding.name]] = None
            # No GPU consumer runs between these host producers and this
            # upload. Token assembly publishes before launching its kernel.
            input_ids = self.tokenID_processor.prepare_input_ids(
                batch, forward_mode.max_seqlen_q, publication_group=token_group
            )
        self.prepare_inputs(
            batch,
            input_ids,
            forward_mode=forward_mode,
            spec_decode_indices=spec_decode_indices,
        )

        # Stage scheduler anchor overrides while this forward's staging window
        # is still open. This pinned mirror is reused, so copying it later from
        # postprocess would fall outside the preparation completion event.
        if hasattr(self, "drafter"):
            forward_context = get_forward_context()
            if batch.next_token_ids is not None:
                forward_context.context.draft_anchor_overrides = (
                    self.drafter.anchors_to_gpu(batch.next_token_ids)
                )
        return (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        )

    @staticmethod
    def _detailed_label_suffix(batch: ScheduledBatch | None) -> str:
        """Detailed attention aggregates for the trace label, or ``""``.

        These fields are only populated by
        `Scheduler.compute_detailed_aggregates` when profiling is active
        and ``ATOM_ENABLE_DETAILED_ANNOTATION`` is set, so on the normal
        (unprofiled) path this returns an empty string without any extra work.
        Appending here keeps the annotation on the ``prefill[]``/``decode[]``
        ``record_function`` (a GPU-recognized layer) instead of nesting an
        extra span above ``run_model``.
        """
        if batch is None or batch.detailed_sqsq is None:
            return ""
        return (
            f" sqsq={batch.detailed_sqsq}"
            f" sqsk={batch.detailed_sqsk}"
            f" sk={batch.detailed_sk}"
        )

    def _build_pcp_balanced_slices(
        self,
        batch: ScheduledBatch,
        num_scheduled_tokens: np.ndarray,
        pcp_size: int,
    ) -> "tuple[list[UBatchSlice], list[PcpBalGroup]]":
        """Build request-boundary-split ubatch slices for PCP+TBO prefill.

        Split REQUESTS into two groups at a request boundary near the token
        midpoint. Each group is an independent "non-TBO PCP mini-batch": padded
        to a pcp multiple and round-robin striped as a whole, so every sequence
        stays intact in one group (root-fixes token-split R1/R2). forward_vars
        stay GLOBAL here; build_ubatch_prefill_metadata slices the FULL
        (un-reindexed) metadata per group and calls _apply_pcp_reindex on it.

        Returns (ubatch_slices, groups): token_slice is in the LOCAL concat
        space [g0_local | g1_local] that run_model produces (see
        _apply_pcp_balanced_stripe); groups are the PcpBalGroup descriptors
        consumed by run_model (per-group stripe) and
        build_ubatch_prefill_metadata (slice + reindex).
        """
        num_prefill_reqs = batch.total_seqs_num_prefill
        per_req = np.asarray(num_scheduled_tokens[:num_prefill_reqs], dtype=np.int64)
        total_tok = int(per_req.sum())
        cum = np.cumsum(per_req)  # cum[j] = sum of reqs [0..j]
        target = total_tok // 2
        # request boundary whose cumulative token count is closest to target
        split_idx = int(np.searchsorted(cum, target, side="left")) + 1
        split_idx = max(1, min(split_idx, num_prefill_reqs - 1))
        # global token count of group0 (reqs [0:split_idx])
        b0 = int(cum[split_idx - 1])
        h0 = pcp_pad_len(b0, pcp_size)
        h1 = pcp_pad_len(total_tok - b0, pcp_size)
        l0 = h0 // pcp_size
        l1 = h1 // pcp_size
        ubatch_slices = [
            UBatchSlice(
                request_slice=slice(0, split_idx),
                token_slice=slice(0, l0),
            ),
            UBatchSlice(
                request_slice=slice(split_idx, num_prefill_reqs),
                token_slice=slice(l0, l0 + l1),
            ),
        ]
        groups = [
            PcpBalGroup(0, split_idx, 0, b0, h0),
            PcpBalGroup(split_idx, num_prefill_reqs, b0, total_tok, h1),
        ]
        return ubatch_slices, groups

    def _apply_pcp_balanced_stripe(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        groups: "list[PcpBalGroup]",
        pcp_size: int,
        forward_context,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """PCP+TBO prefill per-group round-robin stripe, before UBatchWrapper.

        Each request group is padded to a pcp multiple and round-robin striped
        as a WHOLE (so sequences stay intact per group), then the two groups'
        1/pcp shards are concatenated into [g0_local | g1_local]. token_slice
        (built in prepare_inputs) indexes into this concat. Returns the striped
        (input_ids, positions).
        """
        g_ids, g_pos = [], []
        for grp in groups:
            seg_ids = input_ids[grp.tok_start : grp.tok_end]
            seg_pos = positions[grp.tok_start : grp.tok_end]
            pad = grp.pad_total - (grp.tok_end - grp.tok_start)
            if pad > 0:
                seg_ids = torch.cat([seg_ids, seg_ids.new_zeros(pad)])
                seg_pos = torch.cat([seg_pos, seg_pos.new_zeros(pad)])
            g_ids.append(pcp_round_robin_split(seg_ids, pcp_size))
            g_pos.append(pcp_round_robin_split(seg_pos, pcp_size))
        input_ids = torch.cat(g_ids)
        positions = torch.cat(g_pos)
        # context.positions = local per-group concat so _make_ubatch_context
        # slices each ubatch's forward positions correctly.
        forward_context.context.positions = positions
        # Hash MoE: local per-group-concat ids. Each ForCausalLM.forward
        # allgathers its ubatch's slice (g_i local, H_i/pcp) across pcp ranks →
        # H_i ids, matching moe_pcp_merge_forward's per-ubatch hidden allgather.
        if envs.ATOM_PCP_MOE_MERGE:
            forward_context.context.input_ids = input_ids
        return input_ids, positions

    def _restore_pcp_balanced_output(
        self,
        mo: torch.Tensor,
        groups: "list[PcpBalGroup]",
        pcp_size: int,
    ) -> torch.Tensor:
        """Restore PCP+TBO request-boundary-split output.

        UBatchWrapper concatenated the two groups' 1/pcp output shards
        [g0_local | g1_local]. Each group was striped independently, so restore
        per group: pcp_allgather_rerange its shard back to the group's global
        order, crop the per-group pad, then concat to the full global sequence.
        """
        outs = []
        off = 0
        for grp in groups:
            local_len = grp.pad_total // pcp_size  # group's 1/pcp token count
            seg = pcp_allgather_rerange(mo[off : off + local_len], pcp_size)
            outs.append(seg[: grp.tok_end - grp.tok_start])  # crop per-group pad
            off += local_len
        return torch.cat(outs)

    def _setup_pp_shared_indexer(self):
        """Cache per-rank predicates for GLM-5.2 DSA IndexShare PP-boundary
        top-k transfer. Computed once.

        A "shared" attention layer reuses the prior "full" layer's sparse top-k
        via the per-rank scratch buffer ``_sparse_kv_indices_gpu``. When a PP
        boundary splits a shared group, the receiving rank's leading shared
        layers need the sending rank's top-k, so it is carried across the
        boundary. No-op for dense models, sparse models with no shared layers,
        pp=1, or when every rank starts on a "full" layer.
        """
        if getattr(self, "_pp_share_indexer_ready", False):
            return
        self._pp_share_indexer_ready = True
        self._pp_send_needs_sparse = False
        self._pp_recv_needs_sparse = False
        self._pp_index_topk = 0
        if not self.has_mla_indexer:
            return
        pp = get_pp_group()
        if pp.world_size <= 1:
            return
        # Unwrap to the module exposing the PP layer range (make_layers sets
        # start_layer/end_layer on the inner model; UBatchWrapper/CausalLM wrap it).
        inner = self.model
        while not hasattr(inner, "start_layer") and hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "start_layer"):
            return

        # Replicate the model's per-layer shared/full classification
        # (_should_skip_index_topk in deepseek_v2.py).
        hf = self.config.hf_config
        num_layers = int(hf.num_hidden_layers)
        indexer_types = getattr(hf, "indexer_types", None)
        index_topk_pattern = getattr(hf, "index_topk_pattern", None)
        index_topk_freq = int(getattr(hf, "index_topk_freq", 1))
        index_skip_topk_offset = int(getattr(hf, "index_skip_topk_offset", 1))

        def _is_shared(layer_idx):
            if not 0 <= layer_idx < num_layers:
                return False
            if indexer_types is not None:
                return indexer_types[layer_idx] == "shared"
            if index_topk_pattern is not None:
                return index_topk_pattern[layer_idx] == "S"
            if index_topk_freq <= 1:
                return False
            return max(layer_idx - index_skip_topk_offset, 0) % index_topk_freq != 0

        # This rank consumes the prior rank's top-k iff its first layer is shared.
        self._pp_recv_needs_sparse = (not pp.is_first_rank) and _is_shared(
            inner.start_layer
        )
        # The next rank consumes this rank's top-k iff ITS first layer
        # (== this rank's end_layer) is shared.
        self._pp_send_needs_sparse = (not pp.is_last_rank) and _is_shared(
            inner.end_layer
        )
        # Transfer the physical producer row width, not the logical top-k.
        # GLM-5.3 appends up to index_kpool-1 tail tokens and rounds each row
        # to 128 columns; slicing at index_topk would start every later row at
        # the wrong offset.
        self._pp_index_topk = int(
            getattr(
                self.attn_metadata_builder,
                "index_topk_out",
                self.config.hf_config.index_topk,
            )
        )
        if self._pp_recv_needs_sparse or self._pp_send_needs_sparse:
            logger.info(
                "[%s] PP shared-indexer transfer: recv=%s send=%s "
                "(layers [%d,%d), index_topk=%d)",
                self.rank_name,
                self._pp_recv_needs_sparse,
                self._pp_send_needs_sparse,
                inner.start_layer,
                inner.end_layer,
                self._pp_index_topk,
            )

    def _padded_decode_inputs(self, forward_mode: ForwardMode):
        """Expose the full decode run, independently of graph replay.

        Metadata and uniform DP collectives already use running_tokens. Keep
        model activations at that height too, with legal ids/positions in the
        unused tail. Mixed prefill/decode uses varlen collectives and must not
        call this helper.
        """
        assert not forward_mode.is_prefill
        assert forward_mode.running_tokens_are_unified
        scheduled = forward_mode.scheduled_tokens
        running = forward_mode.running_tokens
        ids = self.forward_vars["input_ids"].gpu[:running]
        positions = (
            self._mrope_positions_view(running)
            if self.use_mrope
            else self.forward_vars["positions"].gpu[:running]
        )
        assert ids.shape[0] == running and positions.shape[-1] == running
        if running > scheduled:
            ids[scheduled:].zero_()
            positions[..., scheduled:].zero_()
        return ids, positions

    @_profile_runner_stage
    @record_gpu_forward
    def run_model(
        self,
        input_ids: torch.Tensor,
        batch: ScheduledBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_context = get_forward_context()
        context = forward_context.context
        bs = context.scheduled_bs
        is_prefill = context.is_prefill
        positions = context.positions

        # Dispatch is owned by ForwardMode.decide() (called in prepare_model).
        # Every run_model caller MUST go through prepare_inputs first, so
        # forward_mode is always set here.
        forward_mode = context.forward_mode
        assert forward_mode is not None, (
            "context.forward_mode is None; run_model invoked without going "
            "through prepare_model. Add ForwardMode.decide() at the new "
            "entry point instead of re-deriving the 4-OR dispatch here."
        )

        # Single canonical shape check; contract owned by ForwardMode, which
        # short-circuits only on prefill. The padded step is the one it is for.
        forward_mode.assert_shape_contract(input_ids, forward_context.attn_metadata)

        # Profiler label. Kind (prefix) distinguishes real/dummy and
        # eager/cudagraph; `tbo=1` marks a step that ran TBO ubatches. See
        # `build_run_label`.
        label = build_run_label(
            is_prefill=is_prefill,
            use_cudagraph=forward_mode.use_cudagraph,
            is_dummy=context.is_dummy_run,
            tbo_on=forward_context.ubatch_slices is not None,
            scheduled_bs=bs,
            running_bs=context.running_bs,
            batch=batch,
            detailed_suffix=self._detailed_label_suffix(batch),
        )

        # PCP+TBO prefill: per-group round-robin stripe before UBatchWrapper (see
        # _apply_pcp_balanced_stripe). _pcp_tbo_balanced also gates the per-group
        # output restore further below.
        _pcp_size = self.config.prefill_context_parallel_size
        _pcp_bal_groups = getattr(self, "_pcp_bal_groups", None)
        _pcp_tbo_balanced = (
            _pcp_size > 1
            and isinstance(self.model, UBatchWrapper)
            and forward_context.ubatch_slices is not None
            and is_prefill
            and not forward_context.context.is_dummy_run
            and _pcp_bal_groups is not None
        )
        if _pcp_tbo_balanced:
            input_ids, positions = self._apply_pcp_balanced_stripe(
                input_ids, positions, _pcp_bal_groups, _pcp_size, forward_context
            )

        if not forward_mode.use_cudagraph:
            # prefill, or decode forced eager (enforce_eager / DP peer
            # prefill / bs above the largest captured graph).
            with record_function(label):
                if not is_prefill and forward_mode.running_tokens_are_unified:
                    input_ids, positions = self._padded_decode_inputs(forward_mode)
                # The multimodal runtime owns request leases and span scatter.
                inputs_embeds = None
                if (
                    is_prefill
                    and hasattr(self.model, "get_vision_embeddings")
                    and batch is not None
                    and hasattr(batch, "multimodal_data")
                    and batch.multimodal_data
                ):
                    if all(
                        "embedding_spans" in data
                        for data in batch.multimodal_data.values()
                    ):
                        from atom.model_engine.multimodal_runtime import (
                            embed_multimodal_batch,
                        )

                        inputs_embeds = embed_multimodal_batch(
                            self.model,
                            self.vision_embeddings,
                            input_ids,
                            batch,
                            self.device,
                            self.config.torch_dtype,
                        )
                    else:
                        mm_data_values = list(batch.multimodal_data.values())
                        pixel_values = torch.cat(
                            [mm_data["pixel_values"] for mm_data in mm_data_values],
                            dim=0,
                        ).to(device=self.device, dtype=self.config.torch_dtype)
                        grid_thw = torch.cat(
                            [mm_data["image_grid_thw"] for mm_data in mm_data_values],
                            dim=0,
                        ).to(device=self.device)
                        vision_embeds = self.model.get_vision_embeddings(
                            pixel_values, grid_thw
                        )
                        text_embeds = self.model.embed_input_ids(input_ids)
                        inputs_embeds = self.model.merge_multimodal_embeddings(
                            input_ids, text_embeds, vision_embeds
                        )

                pp_group = get_pp_group()
                pp_enabled = pp_group.world_size > 1
                if pp_enabled:
                    self._setup_pp_shared_indexer()

                intermediate_tensors = None
                if pp_enabled and not pp_group.is_first_rank:
                    intermediate_tensors = recv_intermediate_tensors()
                    # GLM-5.2 IndexShare: load prior rank's top-k for leading
                    # shared layers. Pop so compiled model sees only hidden_states.
                    recv_sparse = intermediate_tensors.tensors.pop(
                        "sparse_kv_indices", None
                    )
                    if recv_sparse is not None and self._pp_recv_needs_sparse:
                        tgt = self.attn_metadata_builder._sparse_kv_indices_gpu
                        tgt[: recv_sparse.numel()].copy_(recv_sparse)

                # Time only the chunk computation between PP receive and send.
                chunk_sample = self._dynamic_chunking.start_sample(batch)
                if pp_enabled:
                    model_output = self.model(
                        input_ids,
                        positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                    )
                elif inputs_embeds is None:
                    model_output = self.model(input_ids, positions)
                else:
                    model_output = self.model(
                        input_ids, positions, inputs_embeds=inputs_embeds
                    )
                self._dynamic_chunking.finish_sample(chunk_sample)
                if pp_enabled and not pp_group.is_last_rank:
                    # GLM-5.2 IndexShare: carry top-k for next rank's shared layers.
                    if self._pp_send_needs_sparse:
                        # Use hidden_states rows (correct under PCP shard).
                        num_tokens = model_output.tensors["hidden_states"].shape[0]
                        n = num_tokens * self._pp_index_topk
                        model_output.tensors["sparse_kv_indices"] = (
                            self.attn_metadata_builder._sparse_kv_indices_gpu[:n]
                        )
                    if self._pp_pending_send:
                        commit_pp_send_work(self._pp_pending_send)
                    self._pp_pending_send = async_send_intermediate_tensors(
                        model_output
                    )
                    hidden_states = None
                    logits = None
                elif self._is_pure_middle_chunk(batch):
                    if _pcp_tbo_balanced:
                        model_output = self._restore_pcp_balanced_output(
                            model_output, _pcp_bal_groups, _pcp_size
                        )
                    # Middle chunk: no logits, but drafter needs hidden states.
                    hidden_states = model_output
                    logits = None
                else:
                    if _pcp_tbo_balanced:
                        model_output = self._restore_pcp_balanced_output(
                            model_output, _pcp_bal_groups, _pcp_size
                        )
                    # PP carries the full run between stages. Only the last
                    # stage returns scheduled rows to sampling/draft consumers.
                    hidden_states = (
                        model_output[: context.scheduled_tokens]
                        if not is_prefill
                        else model_output
                    )
                    logits = self.model.compute_logits(hidden_states)
        else:
            # decode[bs=128 tok=128 d=128] / decode[... p=2 d=126 spec=3] /
            # dummy_decode[...] — see build_run_label.
            with record_function(label):
                running_bs = context.running_bs
                running_tokens = forward_mode.running_tokens
                scheduled_tokens = context.scheduled_tokens

                if self._piecewise_cg_active():
                    _ids, _pos = self._padded_decode_inputs(forward_mode)
                    forward_context.cudagraph_runtime_mode = (
                        CUDAGraphMode.PIECEWISE
                        if forward_mode.piecewise_captured
                        else CUDAGraphMode.NONE
                    )
                    forward_context.batch_descriptor = BatchDescriptor(
                        num_tokens=running_tokens
                    )
                    model_output = self.model(_ids, _pos)
                    forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                    forward_context.batch_descriptor = None
                    # model_output is always a plain Tensor; drafter aux capture
                    # (if any) already wrote its own buffers inside the forward.
                    # Spec keeps the padded layout (postprocess/draft re-gather to
                    # bs via next_token_locs); non-spec cuts to the scheduled rows
                    # so pad rows never leak into sampled_token_ids ->
                    # prev_token_ids -> next-step shape mismatch.
                    hidden_states = model_output[
                        : (
                            running_tokens
                            if hasattr(self, "drafter")
                            else scheduled_tokens
                        )
                    ]
                    logits = self.model.compute_logits(hidden_states)
                    return logits, hidden_states

                graph_key = (running_bs, forward_context.attn_metadata.max_seqlen_q)
                self.graphs[graph_key].replay()
                hidden_states = self.forward_vars["outputs"][:scheduled_tokens]
                # Drafter aux buffers (if any) refresh on replay: their in-place
                # copy ops were captured into the graph.
                if self.logits_in_graph:
                    logits = self.graph_logits[graph_key][:scheduled_tokens]
                else:
                    logits = self.model.compute_logits(hidden_states)

        return logits, hidden_states

    def flush_pp_send(self) -> bool:
        """Flush pending PP isend. Returns True for call_func wait_out."""
        if self._pp_pending_send:
            commit_pp_send_work(self._pp_pending_send)
        return True

    @_profile_runner_stage
    def postprocess(
        self,
        batch: ScheduledBatch,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
        all_greedy: bool,
        # following for draft
        hidden_states: torch.Tensor,
        needs_independent_noise: bool = False,
    ) -> ScheduledBatchOutput:
        spec_decode_metadata = get_forward_context().spec_decode_metadata
        bs = batch.total_seqs_num
        if spec_decode_metadata is None:
            # The LM head emitted one row per sequence the step FORWARDED,
            # which prefill pads to `running_bs` for the draft pass that
            # follows it. Cut to the scheduled batch here and nowhere else:
            # this is the boundary where a padded forward becomes a
            # per-request result, and everything below counts requests -- the
            # sampler's per-row parameters and the logprob gather both. Cut
            # after either and the pad rows divide `[running_bs, V]` by
            # `[scheduled_bs, 1]`.
            logits = logits[:bs]
            sampled_tokens = self.sampler(
                logits,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                needs_independent_noise=needs_independent_noise,
            )
            num_reject_tokens = self.tokenID_processor.default_num_rejected_tokens[:bs]
            next_token_locs = num_reject_tokens
            # No drafts scored -> no verdict; the drafter anchors on each
            # segment's last row.
            anchors = None
        else:
            assert logits is not None
            bonus_logits_indices = spec_decode_metadata.bonus_logits_indices
            target_logits_indices = spec_decode_metadata.target_logits_indices

            bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
            target_logits = torch.index_select(logits, 0, target_logits_indices)
            target_token_ids = None
            if not all_greedy:
                target_token_ids = self.sampler.sample_verification_tokens(
                    target_logits,
                    spec_decode_metadata.cu_num_draft_tokens,
                    temperatures,
                    top_ks,
                    top_ps,
                )
                if target_token_ids.numel() and get_tp_group().world_size > 1:
                    target_token_ids = get_tp_group().broadcast(target_token_ids, src=0)
            bonus_token_ids = self.sampler(
                logits=bonus_logits,
                temperatures=temperatures,
                top_ks=top_ks,
                top_ps=top_ps,
                all_greedy=all_greedy,
                needs_independent_noise=needs_independent_noise or not all_greedy,
            )
            sampled_tokens, verdict = self.rejection_sampler.forward(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
                target_token_ids=target_token_ids,
            )
            # PCP ranks decode redundantly and are consistent only while their
            # kernels agree bit-for-bit -- they don't (hidden differs by ~1 bf16
            # ULP, flipping ~24% of the near-tie verify argmaxes). Accept counts
            # then differ per rank and the emitted streams fork. Sync the
            # decision instead: the ids and the verdict.
            if get_pcp_world_size() > 1 and hasattr(self, "drafter"):
                _g = get_pcp_group()
                sampled_tokens = _g.broadcast(sampled_tokens.contiguous(), src=0)
                verdict = _g.broadcast(verdict, src=0)
            num_bonus_tokens, num_reject_tokens, anchors = verdict
            next_token_locs = num_bonus_tokens

        # Drafter input must agree across TP ranks.
        if get_tp_group().world_size > 1 and (
            self.tokenID_processor.is_deferred_out or hasattr(self, "drafter")
        ):
            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)

        # Compute logprobs if any sequence requested them
        need_logprobs = any(batch.return_logprobs)
        sampled_logprobs = None
        if need_logprobs:
            logits_fp32 = logits.float()
            log_probs = torch.log_softmax(logits_fp32, dim=-1)
            sampled_logprobs = log_probs.gather(
                -1, sampled_tokens.to(torch.long).unsqueeze(-1)
            ).squeeze(-1)
            if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
                sampled_logprobs = get_tp_group().broadcast(sampled_logprobs, src=0)

        self.forward_done_event.record()
        # Capture before prepare_sampled_ids(), which advances self.prev_batch to current batch.
        prev_batch = self.tokenID_processor.prev_batch
        token_id_dict, logprobs_map = self.tokenID_processor.prepare_sampled_ids(
            batch, sampled_tokens, self.forward_done_event, sampled_logprobs
        )
        # Extract req_ids and token_ids from dict (key -1 is the is_deferred_out flag)
        req_ids_out = [k for k in token_id_dict if k != -1]
        token_ids_out = [token_id_dict[k] for k in req_ids_out]

        draft_token_ids: np.ndarray | None = None
        if self.tokenID_processor.is_deferred_out:
            if hasattr(self, "drafter"):
                prev_rejected_num = self.tokenID_processor.prev_rejected_num
                prev_bonus_num = self.tokenID_processor.prev_bonus_num
                self.tokenID_processor.send_mtp_status_to_cpu_async(
                    num_reject_tokens, next_token_locs, self.forward_done_event
                )  # Async copy to CPU
                next_token_ids = torch.gather(
                    sampled_tokens.view(bs, -1), 1, next_token_locs.view(-1, 1)
                ).view(bs)
                self.tokenID_processor.prev_token_ids = next_token_ids
                draft_token_ids = self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    next_token_ids,
                    num_reject_tokens,
                    anchors,
                )

            elif prev_batch is not None:
                prev_rejected_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
                prev_bonus_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
            else:
                # First forward pass: no deferred output yet, req_ids_out is empty
                prev_rejected_num = np.zeros(0, dtype=np.int32)
                prev_bonus_num = np.zeros(0, dtype=np.int32)
        else:
            prev_rejected_num = np.zeros(batch.total_seqs_num, dtype=np.int32)
            prev_bonus_num = np.zeros(batch.total_seqs_num, dtype=np.int32)
            # PP stages and P/D producers still run the drafter.
            if hasattr(self, "drafter"):
                # Mid-prompt sequences get their anchor corrected inside
                # propose_draft_token_ids, from `batch.next_token_ids`.
                next_token_ids = torch.gather(
                    sampled_tokens.view(bs, -1), 1, next_token_locs.view(-1, 1)
                ).view(bs)
                draft_token_ids = self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    next_token_ids,
                    num_reject_tokens,
                    anchors,
                )

        # DSpark Phase 2: carry this step's per-request ell back to the scheduler
        # as a {req_id: ell} dict (req_id-keyed avoids any output/draft batch
        # ordering ambiguity). The worker already fired this map in propose() via
        # verify_scheduler.record_ell(batch.req_ids).
        dspark_ell = None
        drafter = getattr(self, "drafter", None)
        verify_scheduler = getattr(drafter, "verify_scheduler", None)
        if verify_scheduler is not None:
            dspark_ell = verify_scheduler.ell_nonblocking()

        return ScheduledBatchOutput(
            req_ids=req_ids_out,
            token_ids=token_ids_out,
            draft_token_ids=draft_token_ids,
            is_deferred_out=self.tokenID_processor.is_deferred_out,
            num_rejected=prev_rejected_num,
            num_bonus=prev_bonus_num,
            logprobs=logprobs_map,
            dspark_ell=dspark_ell,
        )

    def _record_kv_cache_ready(self, batch: ScheduledBatch) -> None:
        """Publish a GPU event for final prefill chunks to transfer connectors."""
        if batch.total_seqs_num_prefill <= 0:
            return
        if batch.is_final_chunk is None:
            req_ids = batch.req_ids
        else:
            req_ids = [
                req_id
                for req_id, is_final in zip(
                    batch.req_ids, batch.is_final_chunk, strict=True
                )
                if is_final
            ]
        if not req_ids:
            return
        connector = get_kvconnector()
        callback = getattr(connector, "record_kv_cache_ready", None)
        if callable(callback):
            callback(req_ids)

    @_profile_runner_stage
    @torch.inference_mode()
    @with_eplb_forward_monitor
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        # Make this forward's staging buffers safe to overwrite before
        # prepare_inputs writes them: rotate to a free slot if there is a ring,
        # otherwise wait out the previous forward's copies.
        # Dummy forwards use and asynchronously upload the same staging
        # buffers. Excluding them here leaves no event between a dummy and the
        # following real forward, allowing that real forward's CPU writes to
        # race the dummy's still-pending H2D copies.
        self._advance_forward_vars()
        self._gate_staging_reuse()
        (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        ) = self.prepare_model(batch)
        self._mark_staging_h2d_enqueued()
        logits, hidden_states = self.run_model(input_ids, batch)

        pp_group = get_pp_group()
        pp_non_last = pp_group.world_size > 1 and not pp_group.is_last_rank

        drafter = getattr(self, "drafter", None)
        # An output-less batch still runs propose() for its DP collectives.
        will_align_draft = (
            self._dp_draft_lockstep_active()
            and self._is_pure_middle_chunk(batch)
            and not batch.is_dummy_run
        )
        # Runs after EVERY target forward -- `postprocess` (hence propose) is
        # skipped for a middle chunk. Not on the aligning step: that pass is one
        # the peers never mirror.
        run_compute_draft_kv = (
            drafter is not None
            and not pp_non_last
            and not batch.is_dummy_run
            and not (will_align_draft and drafter.draft_kv_duplicates_propose)
        )
        if run_compute_draft_kv:
            drafter.compute_draft_kv(
                get_forward_context().context.positions,
                hidden_states,
                batch.next_token_ids,
            )
        if pp_non_last or self._is_pure_middle_chunk(batch):
            # This return skips `postprocess`, hence propose() and the DP
            # collectives it carries. Run it for those and drop the ids.
            if will_align_draft:
                self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    torch.zeros(
                        batch.total_seqs_num, dtype=torch.int32, device=self.device
                    ),
                    torch.zeros(
                        batch.total_seqs_num, dtype=torch.int32, device=self.device
                    ),
                    None,  # nothing verified -> segment's last row
                    align_only=True,
                )
            reset_forward_context()
            # Mark this slot's GPU work (attention consumed its metadata) done.
            self._record_forward_vars_event()
            self._record_kv_cache_ready(batch)
            return ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            )

        fwd_output = self.postprocess(
            batch,
            logits,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            hidden_states,
            needs_independent_noise=needs_independent_noise,
        )

        reset_forward_context()
        self._record_forward_vars_event()
        self._record_kv_cache_ready(batch)
        return fwd_output

    @staticmethod
    def _is_pure_middle_chunk(batch) -> bool:
        return batch is not None and not batch.produces_output()

    def _dp_draft_lockstep_active(self) -> bool:
        """Are this rank's draft passes bound to what the DP peers run?

        Only under DP attention -- `_publish_draft_shape` returns early at
        `data_parallel_size <= 1`, where an output-less batch legitimately
        drafts nothing. PP is excluded, and asked about directly:
        `is_deferred_out` is also off on a P/D producer, a rank that keeps its
        drafter loaded and still owes its peers the collectives propose()
        carries.
        """
        return (
            hasattr(self, "drafter")
            and self.config.parallel_config.data_parallel_size > 1
            and not self.tokenID_processor.is_pipeline_parallel
        )

    def _producer_graph_shapes(self):
        """The recorded `(bs, q)` keys, for the one runner allowed to miss them.

        A P/D producer keeps its drafter, so capture records q=mtp_k+1, but
        every decode it runs is q=1 -- on every rank, its DP dummy included --
        so the lookup comes out the same on each and the group goes eager
        together. Anywhere else a missing key is a capture gap, and it stays a
        `KeyError` at replay.
        """
        if (
            not self.tokenID_processor.is_pd_producer
            or not hasattr(self, "drafter")
            or self.enforce_eager
            or self._piecewise_cg_active()
            or not hasattr(self, "graphs")
        ):
            return None
        return self.graphs.keys()

    @torch.inference_mode()
    def process_kvconnector_output(self, connector_meta_output):
        """Dispatch KV connector metadata to initiate async KV loading."""
        if connector_meta_output is not None:
            connector = get_kvconnector()
            if connector is not None:
                connector.start_load_kv(connector_meta_output)

    @torch.inference_mode()
    def async_proc_aggregation(self) -> KVConnectorOutput:
        """Collect finished send/recv status from the KV connector."""
        connector = get_kvconnector()
        if connector is None:
            return KVConnectorOutput()

        finished = connector.get_finished()
        # New connectors may return the full KVConnectorOutput so they can
        # report richer states. LMCache offload uses failed_recving to wake a
        # request for local recompute, and finished_saving to release blocks
        # whose free was deferred while a background save read their KV.
        if isinstance(finished, KVConnectorOutput):
            return finished

        # Legacy P/D connectors still return the old
        # (done_sending, done_recving) tuple. Normalize it so EngineCore and
        # Scheduler only need to consume KVConnectorOutput.
        done_sending, done_recving = finished

        return KVConnectorOutput(
            finished_sending=done_sending, finished_recving=done_recving
        )

    def propose_draft_token_ids(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        # num_reject_tokens: KV rows to release, against the `mtp_k` RESERVATION.
        # anchors: each request's flat row of its last emitted token, from the
        # rejection verdict; None when nothing was verified.
        num_reject_tokens: torch.Tensor,
        anchors: torch.Tensor | None,
        align_only: bool = False,
    ):
        """`align_only` runs the draft purely for its DP collectives.

        Its caller is the all-middle-chunk batch, where every seq is mid-prompt:
        the `batch.next_token_ids` override below replaces `next_token_ids`
        wholesale, so the zeros it passes are never read. The ids are dropped --
        the scheduler is not expecting a draft for a seq that produced no token.
        """
        forward_context = get_forward_context()

        # A sequence still mid-prompt samples nothing usable, so its anchor is
        # the scheduler's successor token instead. Per SEQUENCE, not per batch:
        # a middle chunk can sit beside one on its final chunk.
        nxt = batch.next_token_ids
        if nxt is not None:
            assert len(nxt) == next_token_ids.shape[0], (
                f"{len(nxt)} scheduler anchors != {next_token_ids.shape[0]} "
                "sampled -- they are matched positionally"
            )
            # -1 marks "sampling supplies it", so keep the sampled value there.
            override = forward_context.context.draft_anchor_overrides
            assert override is not None
            next_token_ids = torch.where(
                override >= 0, override.to(next_token_ids.dtype), next_token_ids
            )

        positions = forward_context.context.positions

        assert isinstance(self.drafter, Drafter)

        last_token_indices = (
            self.drafter.prepare_inputs(batch.total_seqs_num)
            if anchors is None
            else anchors
        )
        self.attn_metadata_builder.commit_speculative_state(
            forward_context.attn_metadata, last_token_indices
        )

        draft_token = self.drafter.propose(
            target_token_ids=input_ids,
            target_positions=positions,
            target_hidden_states=hidden_states,
            num_reject_tokens=num_reject_tokens,
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
        )
        # PCP runs the drafter on every rank and the ids come out different.
        # Align them before verification consumes them, so all ranks accept the
        # same count.
        if draft_token is not None and get_pcp_world_size() > 1:
            draft_token = get_pcp_group().broadcast(draft_token.contiguous(), src=0)
        if align_only:
            return None
        # DSpark Phase 2: stash this step's scheduler-chosen ell keyed by req_id,
        # so next step's calc_spec_decode_metadata can re-map it onto the (possibly
        # reordered) batch. Keying by req_id (not batch position) is required:
        # continuous batching reorders requests between steps.
        verify_scheduler = getattr(self.drafter, "verify_scheduler", None)
        if verify_scheduler is not None:
            verify_scheduler.record_ell(batch.req_ids[: batch.total_seqs_num])
        return self.tokenID_processor.prepare_draft_ids(batch, draft_token)

    def start_capture_profiler(self):
        """Set up the per-bs CUDA graph capture profiler (profiles in place).

        Profiles the capture phase as graphs are captured and writes one trace
        per (batch size, q-bucket), per rank
        (``bs_<bs>_q_<max_q_len>_rank<rank>.json.gz``). Enabled on every rank
        when a torch profiler dir is set and mark-trace is on.
        """
        self._capture_profile_enabled = (
            self.profiler_dir is not None and self.mark_trace
        )
        if self._capture_profile_enabled:
            record_shapes = envs.ATOM_PROFILER_RECORD_SHAPES
            with_stack = envs.ATOM_PROFILER_WITH_STACK
            profile_memory = envs.ATOM_PROFILER_PROFILE_MEMORY
            self._capture_trace_tag = None
            self.capture_traces_dir = os.path.join(self.profiler_dir, "capture_traces")
            os.makedirs(self.capture_traces_dir, exist_ok=True)
            logger.info(
                "%s: Starting CUDA graph capture profiler "
                "(record_shapes=%s, with_stack=%s, profile_memory=%s)...",
                self.label,
                record_shapes,
                with_stack,
                profile_memory,
            )

            def on_trace_ready(prof):
                # The window is named from the tag the capture loop stashes
                # before each prof.step(), not from a step counter: batch sizes
                # are skipped (_piecewise_skip_capture) and repeated (once per
                # q-bucket), so any index into self.capture_sizes drifts out of sync
                # with what was actually captured.
                #
                # A cleared tag means this is the trailing window that opens
                # after the last step() and closes at __exit__ — it holds only
                # post-loop bookkeeping, so there is nothing worth writing.
                tag = self._capture_trace_tag
                if tag is None:
                    return
                trace_file = os.path.join(
                    self.capture_traces_dir, f"{tag}_rank{self.rank}.json.gz"
                )
                prof.export_chrome_trace(trace_file)
                logger.info(f"Saved capture trace for {tag} to {trace_file}")

            self.capture_profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CUDA,
                    torch_profiler.ProfilerActivity.CPU,
                ],
                # wait=0: recording from __enter__, and every step() closes one
                # window and immediately opens the next, so each iteration of the
                # capture loop lands in its own file with nothing dropped between
                # them (wait>0 would silently skip alternate batch sizes).
                schedule=torch_profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
                record_shapes=record_shapes,
                with_stack=with_stack,
                profile_memory=profile_memory,
                on_trace_ready=on_trace_ready,
            )
        else:
            self.capture_profiler = nullcontext()

    @torch.inference_mode()
    def _piecewise_cg_active(self) -> bool:
        """True when the compiled model's dense pieces self-capture PIECEWISE
        cudagraphs (attention eager between them). In that mode the runner does
        NOT build the manual FULL whole-forward graphs — decode calls the model
        directly and the per-piece CUDAGraphWrapper handles capture/replay."""
        if self.enforce_eager:
            return False
        # Driven by --cudagraph-mode (default FULL -> manual capture, unchanged).
        # PIECEWISE / FULL_AND_PIECEWISE -> per-piece cudagraph path.
        mode = getattr(self.config.compilation_config, "cudagraph_mode", None)
        return mode is not None and mode.requires_piecewise_compilation()

    def _force_aiter_unreg_capture_for_piecewise(self):
        """PIECEWISE cudagraph + aiter custom all_gather/reduce_scatter: force the
        copy-in ('unreg') capture path instead of the direct-read ('registered')
        one.

        The registered path lets the collective kernel directly read each peer's
        ORIGINAL input pointer (cross-registered at register_graph_buffers). That
        is only safe under a single whole-forward FULL cudagraph, whose global
        read/overwrite ordering holds across all ranks. PIECEWISE splits the
        forward into many small graphs with eager sections between them, losing
        that ordering: a fast rank can overwrite its pool-recycled input via a
        later piece while a slow peer is still reading it -> stale cross-rank
        reads -> progressive hidden corruption -> repeated-token garbage
        (DP+PIECEWISE accuracy bug). The unreg path snapshots the input into a
        pre-registered pool before the collective, so it is order-independent.
        """
        seen = set()
        for getter in ("get_tp_group", "get_dp_group", "get_ep_group"):
            try:
                from aiter.dist import parallel_state as _ps

                group = getattr(_ps, getter)()

            # initialised is the ordinary case here, and which exception says
            # so is the aiter build's business, not this probe's.
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s is not available: %s", getter, exc)
                continue
            dc = getattr(group, "device_communicator", None)
            ca = getattr(dc, "ca_comm", None) if dc is not None else None
            if ca is None or id(ca) in seen:
                continue
            seen.add(id(ca))
            if getattr(ca, "enable_register_for_capturing", False):
                ca.enable_register_for_capturing = False
                logger.info(
                    "PIECEWISE: forced aiter ca_comm (%s) to unreg copy-in "
                    "capture path for cudagraph-safe DP collectives.",
                    getter,
                )

    def _dspark_capture_q_buckets(self, full_q: int) -> list[int]:
        """DSpark query-length buckets to capture graphs for (paper Phase 2).

        Confidence scheduling replays a SMALLER max_q_len than full_q, so we
        capture one rectangular graph set per bucket. RAGGED and the older
        q-bucket path use independent size sets. Defaults to ``[full_q]`` (the
        Phase-1 single-graph behavior) when confidence scheduling is off.
        """
        if not (hasattr(self, "drafter") and self.drafter.uses_confidence_schedule):
            return [full_q]
        from atom.spec_decode.dspark_scheduler import resolve_q_buckets

        dspark = self.config.dspark
        sizes = dspark.ragged_graph_sizes if dspark.ragged else dspark.q_buckets
        return resolve_q_buckets(sizes, full_q)

    def _piecewise_per_token_bytes(self) -> float:
        """Estimated GPU bytes a captured PIECEWISE graph retains per token.

        Derived from model geometry (hidden * dtype * layers * live-tensors/layer)
        so it holds for ANY model, not a magic per-token constant. Under DP the
        MoE all_gathers hidden to ~dp_size x local tokens, so each piece retains
        far more per local token than TP (measured DSV4: TP 2.32MB/tok vs DP
        7.7MB/tok, ~3.3x at dp=8). Attention doesn't amplify, so scale by a
        sub-linear dp**0.6 (8**0.6=3.48, just above the measured 3.3).
        """
        hf = self.config.hf_config
        dtype_bytes = torch.finfo(self.config.torch_dtype).bits // 8
        _LIVE_TENSORS_PER_LAYER = 2.8
        per_token = (
            int(hf.hidden_size)
            * dtype_bytes
            * int(hf.num_hidden_layers)
            * _LIVE_TENSORS_PER_LAYER
        )
        dp_size = self.config.parallel_config.data_parallel_size
        if dp_size > 1:
            per_token *= float(dp_size) ** 0.6
        return per_token

    def _piecewise_skip_capture(self, num_tokens: int) -> bool:
        """Whether to skip capturing a PIECEWISE bucket of ``num_tokens`` tokens.

        Two guards, both DP-safe (the decision must be identical on every rank,
        else capture loops desync and the next get_dp_padding all_reduce couples
        mismatched num_tokens -> "batch_id_per_q_token len < T"):

        1. DP+spec hard cap: big bs*q buckets never run under DP but bloat the
           pool and don't overlap comm, so cap at ATOM_PIECEWISE_DP_MAX_TOKENS.
        2. Memory guard: skip a bucket whose estimated capture footprint won't
           fit in free GPU memory (adapts to GPU size / config, no hardcoded
           cap). DP amplifies the retained per-token footprint (MoE all_gather
           ~dp_size x tokens); scale the slope by dp**0.6 to match
           _estimate_cudagraph_overhead. Free is min-reduced across DP so all
           ranks skip the same set.
        """
        dp_size = self.config.parallel_config.data_parallel_size
        if dp_size > 1 and hasattr(self, "drafter"):
            dp_cap = int(os.environ.get("ATOM_PIECEWISE_DP_MAX_TOKENS", "512"))
            if num_tokens > dp_cap:
                if self.rank == 0:
                    logger.info(
                        "PIECEWISE DP-cap skip num_tokens=%d "
                        "(> %d = ATOM_PIECEWISE_DP_MAX_TOKENS)",
                        num_tokens,
                        dp_cap,
                    )
                return True

        # Memory guard slope: capture footprint grows ~linearly with hidden size.
        # Empirically ~600 bytes/token per hidden-dim (0.004GB/token measured at
        # hidden=7168 -> 0.004*2**30/7168 = 599B, rounded).
        _GUARD_BYTES_PER_TOKEN_PER_DIM = 600
        slope = _GUARD_BYTES_PER_TOKEN_PER_DIM * self.config.hf_config.hidden_size
        if dp_size > 1:
            slope *= float(dp_size) ** 0.6
        free = torch.cuda.mem_get_info()[0]
        if dp_size > 1:
            import torch.distributed as dist
            from aiter.dist.parallel_state import get_dp_group

            free_t = torch.tensor([free], device="cpu", dtype=torch.int64)
            dist.all_reduce(
                free_t, op=dist.ReduceOp.MIN, group=get_dp_group().cpu_group
            )
            free = int(free_t.item())
        need = slope * num_tokens * 1.25 + (4 << 30)
        if (free >> 30) < (int(need) >> 30):
            if self.rank == 0:
                logger.info(
                    "PIECEWISE skip num_tokens=%d: free=%.1fGB < need=%.1fGB",
                    num_tokens,
                    free / 1e9,
                    need / 1e9,
                )
            return True
        return False

    def _capture_attn_ffn_graphs(
        self, bs, max_q_len, rectangle_tokens, build_capture, input_ids
    ):
        """AF_PIECEWISE: capture the attn_ffn graphs for the smaller ragged buckets
        (num_tokens_pad = b*max_q_len < this bs's rectangle) a real ragged step at
        this bs may replay. Runs one PIECEWISE forward per new bucket on a ragged
        synthetic batch: dense pieces REPLAY (already captured, deduped by
        num_tokens); the attn_ffn op captures its fresh (bs, q_eff, num_tokens_pad)
        key. The rectangle bucket was already captured by the caller.
        """
        positions = self.forward_vars["positions"].gpu
        for b in self.capture_sizes:
            num_tokens_pad = b * max_q_len
            if num_tokens_pad >= rectangle_tokens or num_tokens_pad < bs:
                # >= rectangle: the rectangle case (already captured) or larger.
                # < bs: fewer than 1 token/seq — unreachable at real decode.
                continue
            if self._piecewise_skip_capture(num_tokens_pad):
                continue
            # Ragged synthetic metadata: bs seqs whose lengths sum to num_tokens_pad.
            attn_metadata, context = build_capture(
                bs=bs, max_q_len=max_q_len, num_tokens_pad=num_tokens_pad
            )
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens_pad)
            num_tokens_dp = num_tokens_pad + num_pad
            if num_tokens_across_dp is not None:
                num_tokens_across_dp = torch.full_like(
                    num_tokens_across_dp, num_tokens_dp
                )
            model_positions = (
                self._mrope_positions_view(num_tokens_dp)
                if self.use_mrope
                else positions[:num_tokens_dp]
            )
            set_forward_context(
                attn_metadata=attn_metadata,
                atom_config=self.config,
                context=context,
                num_tokens=num_tokens_dp,
                num_tokens_across_dp=num_tokens_across_dp,
                ubatch_slices=None,
                in_hipgraph=True,
            )
            # Warmup, then the PIECEWISE forward: dense pieces replay (deduped by
            # num_tokens); attn_ffn op captures its (bs, q_eff, num_tokens_pad) graph.
            self.model(input_ids[:num_tokens_dp], model_positions)
            fc = get_forward_context()
            fc.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            fc.batch_descriptor = BatchDescriptor(num_tokens=num_tokens_dp)
            self.model(input_ids[:num_tokens_dp], model_positions)
            fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
            fc.batch_descriptor = None
            self._piecewise_captured_tokens.add(num_tokens_dp)

    @torch.inference_mode()
    def capture_cudagraph(self):
        # M3 indexer-only CP puts an all-to-all in the captured decode path, and
        # NCCL sets up peer connections on a group's FIRST collective -- doing
        # that inside a capture hangs. warmup_model() does not cover it: its
        # dummy batch is prefill-only, and prefill stays on the TP path with no
        # all-to-all. No-op unless the flag is on.
        from atom.distributed.indexer_cp import warmup_exchange

        warmup_exchange(self.device)

        _piecewise = self._piecewise_cg_active()
        # AF_PIECEWISE: also capture the attn core (ragged combos below)
        cudagraph_mode = getattr(self.config.compilation_config, "cudagraph_mode", None)
        attn_ffn_piecewise = (
            cudagraph_mode is not None and cudagraph_mode.is_attn_ffn_piecewise()
        )
        if _piecewise:
            logger.info(
                "PIECEWISE cudagraph: capturing per-piece graphs (attention "
                "eager); manual FULL whole-forward capture disabled."
            )
            self._force_aiter_unreg_capture_for_piecewise()
        start_time = time.time()
        # Config owns the declared ladder; the runner only narrows it to what
        # this deployment can actually schedule (see the bound below).
        self.capture_sizes = self.config.capture_sizes
        self.capture_sizes.sort(reverse=True)

        # Drop any capture size the scheduler could never produce. `schedule_decode`
        # bounds a decode batch two ways: at most `max_num_seqs` sequences, and at
        # most `max_num_batched_tokens` tokens, charging `mtp_k + 1` tokens per
        # sequence whatever query length the step ends up replaying. So the
        # reachable batch size is the min of the two, and under speculation the
        # token budget is what binds first — mtp_k=3 turns 256 sequences into 1024
        # tokens.
        #
        # Filtering on the token budget is not just about avoiding a graph that is
        # never replayed. The per-token forward buffers (`positions`, `input_ids`,
        # `outputs`) are sized `max_num_batched_tokens`, so capture at a bs past
        # this bound writes out of bounds — it used to surface as a bare
        # `could not broadcast input array from shape (1024,) into shape (512,)`
        # out of `capture_cudagraph`, which reads like a shape bug rather than a
        # config one. Charging `mtp_k + 1` (never a smaller q bucket) also keeps
        # every (bs, max_q_len) pair the loop below visits within the bound, so
        # the runtime `self.graphs[(bs, max_q_len)]` lookup cannot miss.
        #
        # Warn rather than raise so a misconfig (default cuda_graph_sizes=[512]
        # vs e.g. max_num_seqs=16) stays recoverable.
        full_q_len = self.drafter.mtp_k + 1 if hasattr(self, "drafter") else 1
        max_seq_bs = self.config.max_num_seqs
        max_tok_bs = self.config.max_num_batched_tokens // full_q_len
        max_bs = max_schedulable_decode_bs(
            max_seq_bs, self.config.max_num_batched_tokens, full_q_len
        )
        oversized = [s for s in self.capture_sizes if s > max_bs]
        if oversized:
            self.capture_sizes = [s for s in self.capture_sizes if s <= max_bs]
            logger.warning(
                "cudagraph capture sizes %s exceed the schedulable batch size "
                "min(max_num_seqs=%d, max_num_batched_tokens=%d // (mtp_k+1)=%d "
                "= %d) = %d; dropping. Remaining: %s",
                oversized,
                max_seq_bs,
                self.config.max_num_batched_tokens,
                full_q_len,
                max_tok_bs,
                max_bs,
                self.capture_sizes,
            )
        assert self.capture_sizes, (
            f"no cudagraph capture sizes left: the scheduler can only reach "
            f"bs <= min(max_num_seqs={max_seq_bs}, "
            f"max_num_batched_tokens={self.config.max_num_batched_tokens} // "
            f"(mtp_k+1)={full_q_len} = {max_tok_bs}) = {max_bs}. Pass "
            f"--cudagraph-capture-sizes, raise --max-num-seqs, or raise "
            f"--max-num-batched-tokens."
        )

        # PIECEWISE: the set of num_tokens shapes whose dense pieces we captured
        # (reset here; initialized empty in __init__). run_model dispatches by
        # num_tokens; a shape NOT in here would force a runtime (uncoordinated)
        # capture that hangs on collectives, so run_model falls back to eager for
        # uncaptured shapes.
        self._piecewise_captured_tokens = set()

        if "kv_indptr" in self.forward_vars:
            self.forward_vars["kv_indptr"].gpu.zero_()
        # Present exactly when the model has an indexer -- the builder makes it
        # under the same answer -- so the buffer's own existence is the test.
        if "sparse_kv_indptr" in self.forward_vars:
            self.forward_vars["sparse_kv_indptr"].gpu.zero_()

        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.graph_logits: dict[tuple[int, int], torch.Tensor] = {}
        self.graph_pool = None
        is_tbo = self.config.enable_tbo and isinstance(self.model, UBatchWrapper)
        # TBO graphs don't capture compute_logits, so disable logits_in_graph.
        self.logits_in_graph = self.world_size == 1 and not is_tbo

        # start capture profiler
        self.start_capture_profiler()

        @contextmanager
        def pause_gc():
            # No GC during capture: a finalizer's hipModuleUnload aborts it (HIP 900).
            gc.collect()
            gc.disable()
            try:
                yield
            finally:
                gc.enable()
                gc.collect()

        _rsv_before_capture = torch.cuda.memory_reserved()
        _alloc_before_capture = torch.cuda.memory_allocated()

        input_ids = self.forward_vars["input_ids"].gpu
        positions = self.forward_vars["positions"].gpu
        outputs = self.forward_vars["outputs"]

        # Capture one graph per (bs, query-length bucket). Buckets default to
        # [full_q_len] (single-graph, classic per-bs capture); DSpark confidence
        # scheduling expands to the smaller q-buckets a decode step may replay.
        q_buckets = self._dspark_capture_q_buckets(full_q_len)
        if q_buckets != [full_q_len]:
            logger.info("DSpark CUDA-graph query buckets: %s", q_buckets)
        elif hasattr(self, "drafter") and self.drafter.uses_confidence_schedule:
            # resolve_q_buckets always folds full_q in, so a spec naming only
            # full_q (or nothing, or nothing valid) collapses to [full_q] and
            # every step replays at full length. The step still pays the
            # confidence schedule + ragged rebuild, so say so rather than look
            # like it is shrinking anything.
            dspark = self.config.dspark
            spec = dspark.ragged_graph_sizes if dspark.ragged else dspark.q_buckets
            logger.warning(
                "DSpark %s=%r resolves to [%d] (== full verify length), so no "
                "query-length shrink is possible and every decode step replays "
                "at full length. Pass sizes BELOW %d to get any benefit.",
                "ragged_graph_sizes" if dspark.ragged else "q_buckets",
                spec,
                full_q_len,
                full_q_len,
            )

        self.h2d_owner.begin()
        self.attn_metadata_builder.blank_cache_write_targets()
        self.h2d_owner.finish()

        # Whether this backend's capture builder supports a dynamic (per-bucket)
        build_capture = self.attn_metadata_builder.build_for_cudagraph_capture
        _build_params = inspect.signature(build_capture).parameters
        supports_dynamic_q_len = "max_q_len" in _build_params
        # Whether it supports a ragged num_tokens_pad (zero-copy-q attn-core graphs).
        supports_ragged_capture = "num_tokens_pad" in _build_params

        raw_build_capture = build_capture

        @wraps(raw_build_capture)
        def build_capture(*args, **kwargs):
            # Preparation is outside actual graph capture, including calls
            # made by the drafter and ragged bucket builder. Each invocation
            # owns a synthetic transaction; normal forwards never use it.
            bs = kwargs.get("bs", args[0] if args else None)
            q_len = kwargs.get("max_q_len", full_q_len)
            self.h2d_owner.begin()
            try:
                if not self.attn_metadata_builder.capture_owns_cu_seqlens_q:
                    cu = self.forward_vars["cu_seqlens_q"]
                    cu.np[: bs + 1] = np.arange(
                        0, (bs + 1) * q_len, q_len, dtype=np.int32
                    )
                    cu.copy_to_gpu(bs + 1)
                tokens = bs * q_len
                self.forward_vars["positions"].np[:tokens] = (
                    np.arange(tokens, dtype=np.int64) % q_len
                )
                result = raw_build_capture(*args, **kwargs)
                self.h2d_owner.finish()
                return result
            except BaseException:
                self.h2d_owner.fail()
                raise

        with pause_gc(), graph_capture() as capture_ctx, self.capture_profiler as prof:
            for max_q_len in q_buckets:
                capture_range = (
                    tqdm.tqdm(self.capture_sizes)
                    if self.rank == 0
                    else self.capture_sizes
                )
                for bs in capture_range:
                    if self.rank == 0:
                        capture_range.set_description(f"Capturing {bs=}, {max_q_len=}")

                    num_tokens = bs * max_q_len
                    if _piecewise and self._piecewise_skip_capture(num_tokens):
                        continue
                    # Names the capture trace this iteration will export. Set
                    # after the skip above so a skipped bs never claims a file;
                    # its handful of Python statements just fold into the next
                    # iteration's window.
                    self._capture_trace_tag = f"bs_{bs}_q_{max_q_len}"
                    if supports_dynamic_q_len:
                        attn_metadata, context = build_capture(
                            bs=bs, max_q_len=max_q_len
                        )
                    else:
                        attn_metadata, context = build_capture(bs=bs)
                    if self.use_mrope:
                        mrope_positions = self._mrope_positions_view(num_tokens)
                        mrope_positions.copy_(
                            positions[:num_tokens].unsqueeze(0).expand(3, -1)
                        )
                        context.positions = mrope_positions
                    num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
                    num_tokens += num_pad
                    # get_dp_padding built num_tokens_across_dp from the PRE-pad
                    # count, but we just padded num_tokens. Capture is symmetric
                    # (every DP rank captures the same bs), so the padded count is
                    # uniform across ranks. Rebuild the tensor at the padded size so
                    # DPMetadata.make's `across_dp[rank] == num_tokens` holds.
                    if num_tokens_across_dp is not None:
                        num_tokens_across_dp = torch.full_like(
                            num_tokens_across_dp, num_tokens
                        )
                    # Create ubatch slices for TBO capture (need > 2 requests)
                    ubatch_slices = None
                    ub_tokens_across_dp = None
                    if is_tbo and self.config.enable_tbo_decode and bs > 2:
                        ubatch_slices = maybe_create_ubatch_slices(
                            num_reqs=bs,
                            num_tokens=num_tokens,
                        )
                        # The rebuild above's symmetry, one level down: every
                        # rank splits this bucket the same way, so a ubatch's
                        # per-rank counts are its own repeated. Stated, because
                        # a capture context declares its shape where a real step
                        # reduces one, and no consumer can tell an absent
                        # reduction from a uniform answer.
                        if ubatch_slices and num_tokens_across_dp is not None:
                            dp = len(num_tokens_across_dp)
                            ub_tokens_across_dp = tuple(
                                (s.token_slice.stop - s.token_slice.start,) * dp
                                for s in ubatch_slices
                            )

                    set_forward_context(
                        attn_metadata=attn_metadata,
                        atom_config=self.config,
                        context=context,
                        num_tokens=num_tokens,
                        num_tokens_across_dp=num_tokens_across_dp,
                        ubatch_slices=ubatch_slices,
                        ub_tokens_across_dp=ub_tokens_across_dp,
                        in_hipgraph=True,
                    )

                    # Warmup
                    model_positions = (
                        self._mrope_positions_view(num_tokens)
                        if self.use_mrope
                        else positions[:num_tokens]
                    )
                    model_output = self.model(input_ids[:num_tokens], model_positions)
                    outputs[:num_tokens] = model_output
                    if self.logits_in_graph:
                        self.model.compute_logits(outputs[:num_tokens])

                    if _piecewise:
                        # PIECEWISE: no manual whole-forward graph; the compiled
                        # per-piece wrappers self-capture. Replay once to register.
                        fc = get_forward_context()
                        fc.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                        fc.batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
                        self.model(input_ids[:num_tokens], model_positions)
                        fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
                        fc.batch_descriptor = None
                        self._piecewise_captured_tokens.add(num_tokens)
                        # also capture the attn_ffn graphs this bs can replay ragged
                        if attn_ffn_piecewise and supports_ragged_capture:
                            self._capture_attn_ffn_graphs(
                                bs=bs,
                                max_q_len=max_q_len,
                                rectangle_tokens=num_tokens,
                                build_capture=build_capture,
                                input_ids=input_ids,
                            )
                        if prof is not None:
                            # Drain before closing the window so this bs's
                            # kernels land in this bs's file. Profiling-only —
                            # the unprofiled PIECEWISE path stays sync-free.
                            torch.cuda.synchronize()
                            prof.step()
                            self._capture_trace_tag = None
                        continue

                    # Capture
                    with (
                        record_function(f"capture_graph_bs_{bs}")
                        if self.mark_trace
                        else nullcontext()
                    ):
                        if ubatch_slices is not None:
                            # TBO capture: threads + multi-stream captured in graph.
                            # Drafter aux (if any) is written inside the graph by
                            # capture_tbo_graph replaying the model's forward hooks.
                            graph, _ = self.model.capture_tbo_graph(
                                input_ids[:num_tokens],
                                positions[:num_tokens],
                                self.graph_pool,
                                capture_ctx.stream,
                                output_buffer=outputs[:num_tokens],
                            )
                        else:
                            # Standard single-stream capture. The drafter's aux
                            # capture hook runs inside the forward and writes its
                            # own buffers (captured in-graph); model_output is a
                            # plain Tensor.
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(
                                graph, self.graph_pool, stream=capture_ctx.stream
                            ):
                                model_output = self.model(
                                    input_ids[:num_tokens], model_positions
                                )
                                outputs[:num_tokens] = model_output
                                if self.logits_in_graph:
                                    graph_logits = self.model.compute_logits(
                                        outputs[:num_tokens]
                                    )
                    if self.graph_pool is None:
                        self.graph_pool = graph.pool()
                    self.graphs[(bs, max_q_len)] = graph
                    if self.logits_in_graph and ubatch_slices is None:
                        self.graph_logits[(bs, max_q_len)] = graph_logits
                    torch.cuda.synchronize()
                    # After the sync: the warmup forward's kernels must have
                    # completed before the window closes, or they spill into the
                    # next bs's file.
                    if prof is not None:
                        prof.step()
                        self._capture_trace_tag = None
            # Inside the `with`: graph_capture() arms the custom all-reduce for
            # capture and pause_gc() keeps the collector from aborting one. The
            # drafter gets the capture builder already bound to this backend's
            # signature -- which of them takes a `max_q_len` is the runner's to
            # know, and it is the same probe the loop above ran.
            if hasattr(self, "drafter"):
                self.drafter.warmup_draft_graphs(
                    (
                        partial(build_capture, max_q_len=full_q_len)
                        if supports_dynamic_q_len
                        else build_capture
                    ),
                    capture_ctx.stream,
                )
        self.capture_sizes.sort()
        self.capture_sizes_np = np.asarray(self.capture_sizes, dtype=np.int32)

        # PIECEWISE: sorted 1D num_tokens buckets for run_model's round_up_1d(Σ)
        # dispatch (bisect_left over this to pick the tightest captured shape).
        self._piecewise_sorted_tokens = sorted(self._piecewise_captured_tokens)
        if _piecewise and self.rank == 0:
            logger.info(
                "PIECEWISE captured %d num_tokens buckets: %s",
                len(self._piecewise_sorted_tokens),
                self._piecewise_sorted_tokens,
            )

        # DSpark Phase 2: calibrate the SPS(B) throughput profile from the just-
        # captured target graphs (each is a B = bs*max_q_len token forward, i.e.
        # exactly one verification step at batch B). Cheap, GPU-only, one-shot.
        self._maybe_calibrate_dspark_sps(full_q_len)

        # How much GPU memory the CUDA graph capture consumed (pool = reserved
        # delta; the allocated delta is what the graphs pin live).
        _pool_bytes = max(torch.cuda.memory_reserved() - _rsv_before_capture, 0)
        _alloc_bytes = max(torch.cuda.memory_allocated() - _alloc_before_capture, 0)
        if self.rank == 0:
            logger.info(
                "CUDA graph capture memory: %d graphs | pool(reserved)=%.2fGB "
                "allocated=%.2fGB",
                len(self.graphs) + len(self._piecewise_captured_tokens),
                _pool_bytes / (1 << 30),
                _alloc_bytes / (1 << 30),
            )

        # Post-init memory validation
        free_after, total_after = torch.cuda.mem_get_info()
        actual_usage = total_after - free_after
        target_usage = int(total_after * self.config.gpu_memory_utilization)
        usage_ratio = actual_usage / total_after
        logger.info(
            f"Post-init memory: "
            f"actual={actual_usage / (1 << 30):.2f}GB ({usage_ratio:.1%}), "
            f"target={target_usage / (1 << 30):.2f}GB "
            f"({self.config.gpu_memory_utilization:.0%}), "
            f"reserved={torch.cuda.memory_reserved() / (1 << 30):.2f}GB, "
            f"allocated={torch.cuda.memory_allocated() / (1 << 30):.2f}GB"
        )
        if usage_ratio > self.config.gpu_memory_utilization + 0.02:
            logger.warning(
                f"Actual GPU memory usage ({usage_ratio:.1%}) exceeds target "
                f"({self.config.gpu_memory_utilization:.0%}) by "
                f"{(usage_ratio - self.config.gpu_memory_utilization):.1%}. "
                f"Consider reducing gpu_memory_utilization."
            )

        return time.time() - start_time, self.capture_sizes, _pool_bytes

    @torch.inference_mode()
    def _maybe_calibrate_dspark_sps(self, max_q_len: int, n_iters: int = 20) -> None:
        """Profile SPS(B) by timing the captured target graphs, then hand a dense
        cost table to the DSpark drafter (paper §3.2.2, scheduler input).

        Each captured graph ``self.graphs[(bs, max_q_len)]`` is a forward over
        ``B = bs * max_q_len`` tokens — exactly one verification step at batch B.
        We replay each a few times, take the median step time, and densify the
        (B, steps/sec) samples into ``sps_table[B]``. No-op unless a DSpark
        drafter with confidence scheduling enabled is present.
        """
        drafter = getattr(self, "drafter", None)
        if drafter is None or not drafter.is_block_drafter:
            return
        verify_scheduler = getattr(drafter, "verify_scheduler", None)
        if verify_scheduler is None:
            return
        if verify_scheduler.calibration_profile is not None:
            return  # Explicit offline measurements must not be overwritten.
        if not getattr(self, "graphs", None):
            return
        if self.config.dspark.disable_sps_calib:
            logger.info("DSpark SPS calibration disabled; using synthetic stub.")
            return

        from atom.spec_decode.dspark_scheduler import build_sps_table

        # DSpark RAGGED graph: replay-based SPS calibration is UNSAFE here. Each
        # `graph.replay()` runs the FULL decode graph (incl. SWA/KV writes) with
        # synthetic data at real cache slots [0:bs], polluting the KV cache real
        # requests then read. The scheduler only needs a monotone SPS(B) shape,
        # so use a synthetic table instead (matches the proven DISABLE_SPS_CALIB
        # path). Timed ragged calibration is a follow-up (needs a scratch KV pool
        # + buffer save/restore around the replays).
        if self.config.dspark.ragged:
            logger.info(
                "DSpark SPS calibration skipped under RAGGED graph "
                "(replay would pollute KV cache); using synthetic stub."
            )
            return

        token_points: list[int] = []
        sps_points: list[float] = []
        for bs in self.capture_sizes:
            graph = self.graphs.get((bs, max_q_len))
            if graph is None:
                continue
            B = bs * max_q_len
            # Every row of the recording is real here: time the whole of it
            # (the DP pad-row mask reads the real token count from here).
            cu = self.forward_vars["cu_seqlens_q"]
            cu.np[: bs + 1] = np.arange(0, B + 1, max_q_len, dtype=np.int32)
            cu.copy_to_gpu(bs + 1)
            # Warm replay, then timed replays (median for robustness to jitter).
            graph.replay()
            torch.cuda.synchronize()
            times_ms: list[float] = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_iters):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times_ms.append(start.elapsed_time(end))
            times_ms.sort()
            median_ms = times_ms[len(times_ms) // 2]
            if median_ms <= 0:
                continue
            token_points.append(B)
            sps_points.append(1000.0 / median_ms)  # steps per second

        if not token_points:
            logger.warning("DSpark SPS calibration found no timeable graphs.")
            return

        max_b = self.config.max_num_seqs * max_q_len
        sps_table = build_sps_table(token_points, sps_points, max_b).to(self.device)
        verify_scheduler.sps_table = sps_table
        logger.info(
            "DSpark SPS calibrated over %d points (B=%d..%d), table size %d.",
            len(token_points),
            token_points[0],
            token_points[-1],
            sps_table.numel(),
        )


class RapidServeModelRunner(ModelRunner):
    """ModelRunner for intra-GPU prefill/decode disaggregation.

    The same class runs in both the prefill and decode processes; behavior that
    differs between them keys off config.disagg_is_decode.
    """

    def __init__(self, rank, config):
        if not config.disagg_is_decode:
            self.forward = self.prefill_forward
        super().__init__(rank, config)
        import hashlib

        # Session ID derived from the disagg IPC address so IPC handoff temp
        # files from different engine runs never collide.
        self._disagg_session_id = hashlib.md5(
            config.disagg_kvcache_ipc_addr.encode()
        ).hexdigest()[:12]

    @staticmethod
    @contextmanager
    def _init_weight_params_on_meta():
        """Construct a model with all `nn.Parameter`s on the meta device (no
        persistent GPU weight allocation), leaving the default device unchanged
        so init code that explicitly targets CUDA (e.g. aiter RoPE) and buffers
        work normally. Each parameter is briefly created on the real device then
        replaced with a meta tensor, so the transient peak is one parameter, not
        the whole model. Used by decode in disagg, which fills params from
        prefill via IPC.
        """
        orig_register = torch.nn.Module.register_parameter

        def register_parameter(self, name, param):
            orig_register(self, name, param)
            p = self._parameters.get(name)
            if p is not None:
                self._parameters[name] = torch.nn.Parameter(
                    p.detach().to("meta"), requires_grad=p.requires_grad
                )

        torch.nn.Module.register_parameter = register_parameter
        try:
            yield
        finally:
            torch.nn.Module.register_parameter = orig_register

    # ------------------------------------------------------------------
    # Base ModelRunner override points
    # ------------------------------------------------------------------

    def _build_and_load_model(self, model_class):
        config = self.config
        if not config.disagg_is_decode:
            super()._build_and_load_model(model_class)
            return
        # Decode imports prefill's weights via CUDA IPC and owns no weight
        # memory. Build on the meta device so construction allocates zero GPU
        # bytes (avoids the transient 2x-weights peak that OOMs at TP=4);
        # import_model_weight_ipc_handles() materializes params from prefill,
        # and RoPE caches are recomputed locally.
        with self._init_weight_params_on_meta():
            self.model = model_class(config)
        torch.set_default_device(None)

    def _maybe_warmup(self):
        # Decode owns no GPU memory yet (weights/kvcache imported from prefill
        # later), so warmup would run against meta-device tensors — skip it.
        if self.config.disagg_is_decode:
            return
        super()._maybe_warmup()

    def _kv_budget_extra_reserve(self, total_bytes: int) -> int:
        # Two processes share the GPU in disagg mode; reserve extra headroom so
        # NCCL/system allocs (e.g. the 512MB NCCL barrier buffer) don't OOM.
        safety_margin = int(total_bytes * 0.02)
        return 4 * safety_margin

    def get_num_blocks(self) -> dict[str, object]:
        # Decode in disagg mode owns no GPU memory — kvcache is imported from
        # prefill.
        if self.config.disagg_is_decode:
            transfer = self.attn_metadata_builder.state_transfer()
            if transfer.copies:
                raise RuntimeError(
                    "PAGE-backed state checkpoints do not yet support RapidServe "
                    "prefill/decode disaggregation"
                )
            return {
                "num_kvcache_blocks": 0,
                "state_runtime": StateRuntime(transfer=transfer).to_wire(),
            }
        return super().get_num_blocks()

    def allocate_kv_cache(self, num_kvcache_blocks):
        # Decode in disagg mode: kvcache is imported from prefill, not allocated.
        if self.config.disagg_is_decode:
            logger.info("decode skipping kv cache allocation")
            return True
        return super().allocate_kv_cache(num_kvcache_blocks)

    @_profile_runner_stage
    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        # Decode runs the model forward on a dynamically selected (optionally
        # CU-masked) stream so it doesn't contend with prefill on the shared
        # GPU; postprocess (sampling + async CPU copy) stays on the default
        # stream. The stream pool is created by DecodeEngineCore via
        # create_decode_stream_pool().
        stream = self._decode_streams[batch.cu_stream_fraction]
        self._done_event.record()
        stream.wait_event(self._done_event)
        with torch.cuda.stream(stream):
            self._advance_forward_vars()
            self._gate_staging_reuse()
            (
                input_ids,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                needs_independent_noise,
            ) = self.prepare_model(batch)
            self._mark_staging_h2d_enqueued()
            logits, hidden_states = self.run_model(input_ids, batch)
        self._model_fwd_event.record(stream)
        torch.cuda.current_stream().wait_event(self._model_fwd_event)

        # postprocess (sampling + async CPU copy) always runs on default stream.
        fwd_output = self.postprocess(
            batch,
            logits,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            hidden_states,
            needs_independent_noise=needs_independent_noise,
        )

        self._record_forward_vars_event()
        reset_forward_context()
        return fwd_output

    # ------------------------------------------------------------------
    # Disagg IPC helpers (TP-aware)
    #
    # With TP>1, AsyncIOProcManager broadcasts each RPC to all ranks but only
    # rank 0's return value reaches the engine (the other ranks' output sockets
    # are wired to None).  For IPC handle exchange we therefore use a per-rank
    # temp-file rendezvous: every rank pickles its local handles to
    #   /tmp/atom_disagg_<tag>_rank<N>.pkl
    # Rank 0 waits until all N files exist, then returns the list of paths to
    # the engine.  On the import side every rank reads its own file (index=self.rank).
    # ------------------------------------------------------------------

    def _disagg_rank_file_path(self, tag: str, rank: int) -> str:
        import tempfile

        return os.path.join(
            tempfile.gettempdir(),
            f"atom_disagg_{self._disagg_session_id}_{tag}_rank{rank}.pkl",
        )

    def _disagg_write_rank_file(self, tag: str, payload) -> str:
        """Pickle *payload* to a session-unique per-rank temp file.

        Uses a session ID derived from the disagg IPC address so that files
        from different engine runs never collide.  No deletion: all ranks
        write concurrently when they receive the broadcast RPC, so rank 0
        must never delete sibling files.
        """
        import pickle

        path = self._disagg_rank_file_path(tag, self.rank)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    def _disagg_collect_rank_files(self, tag: str) -> list[str] | None:
        """Rank 0: poll until all world_size rank files exist, return paths.
        Other ranks: return None immediately (rank 0 is the sole publisher).
        """
        import time

        if self.rank != 0:
            return None
        paths = [self._disagg_rank_file_path(tag, r) for r in range(self.tp_world_size)]
        deadline = time.monotonic() + 120  # 2 min timeout
        while time.monotonic() < deadline:
            if all(os.path.exists(p) for p in paths):
                return paths
            time.sleep(0.05)
        raise TimeoutError(
            f"Timed out waiting for disagg rank files for tag={tag!r}: "
            + str([p for p in paths if not os.path.exists(p)])
        )

    def export_model_weight_ipc_handles(self) -> list[str] | None:
        """Export all model parameters as CUDA IPC handles (prefill process only).

        TP-aware: each rank writes its own handles to a temp file.  Rank 0 waits
        for all ranks' files and returns the list of paths; other ranks return None.
        Decode calls import_model_weight_ipc_handles(paths) to replace its own
        weight tensors with zero-copy views into prefill's allocation.
        """
        from atom.model_engine.ipc_utils import export_model_weight_handles

        logger.info(f"ModelRunner rank {self.rank}: export_model_weight_ipc_handles")
        handles = export_model_weight_handles(self.model)
        self._disagg_write_rank_file("weights", handles)
        paths = self._disagg_collect_rank_files("weights")
        if paths is not None:
            logger.info(
                f"ModelRunner rank 0: all {self.tp_world_size} weight files ready"
            )
        return paths  # non-None only for rank 0

    def import_model_weight_ipc_handles(self, paths: list[str]) -> bool:
        """Replace model parameters with views into prefill's GPU allocation.

        TP-aware: each rank reads its own handles file (index=self.rank) and
        deletes it after import.  Returns True as sentinel for wait_out=True.
        """
        import gc
        import pickle

        from atom.model_engine.ipc_utils import import_model_weights

        path = paths[self.rank]
        logger.info(f"ModelRunner rank {self.rank}: reading weight handles from {path}")
        with open(path, "rb") as f:
            handles = pickle.load(f)
        os.remove(path)
        import_model_weights(self.model, handles)
        gc.collect()
        torch.cuda.empty_cache()
        # Surface any tensor prefill didn't export (would crash later in forward).
        leftover = [n for n, p in self.model.named_parameters() if p.is_meta] + [
            n
            for n, b in self.model.named_buffers()
            if isinstance(b, torch.Tensor) and b.is_meta
        ]
        if leftover:
            logger.warning(
                f"ModelRunner rank {self.rank}: {len(leftover)} tensors still on "
                f"meta after IPC import (not materialized): {leftover[:10]}"
            )
        logger.info(
            f"ModelRunner rank {self.rank}: weight IPC import complete — own weights freed"
        )
        return True

    def export_kv_cache_ipc_handle(self) -> list[str] | None:
        """Export self.kv_cache — the whole paged pool — as a CUDA IPC handle.

        TP-aware: each rank writes its handles to a temp file.  Rank 0 waits for
        all ranks and returns the list of paths; other ranks return None.
        """
        from atom.model_engine.ipc_utils import export_kv_cache_handle

        logger.info(f"ModelRunner rank {self.rank}: export_kv_cache_ipc_handle")
        handles = export_kv_cache_handle(self.kv_cache)
        self._disagg_write_rank_file("kvcache", handles)
        paths = self._disagg_collect_rank_files("kvcache")
        if paths is not None:
            logger.info(
                f"ModelRunner rank 0: all {self.tp_world_size} kvcache files ready"
            )
        return paths  # non-None only for rank 0

    def import_kv_cache_ipc_handle(
        self, paths: list[str], num_kvcache_blocks: int
    ) -> bool:
        """Import kvcache from prefill's GPU allocation into this (decode) process.

        TP-aware: each rank reads its own handles file (index=self.rank) and
        deletes it after import.  Also imports kv_scale when present (fp8).
        Returns True as sentinel for wait_out=True.
        """
        import pickle

        from atom.model_engine.ipc_utils import import_kv_cache

        # The count travels with the handle and reaches the builder through
        # `_bind_kv_cache_to_modules` -> `allocate_kv_cache_tensors` below; this
        # side never ran sizing, so there is nothing else it could come from.
        path = paths[self.rank]
        logger.info(
            f"ModelRunner rank {self.rank}: reading kvcache handles from {path}"
        )
        with open(path, "rb") as f:
            meta = pickle.load(f)
        os.remove(path)
        logger.info(f"ModelRunner rank {self.rank}: hipIpcOpenMemHandle for kvcache...")
        self.kv_cache = import_kv_cache(meta)
        logger.info(
            f"ModelRunner rank {self.rank}: kvcache IPC import done, binding..."
        )
        self._bind_kv_cache_to_modules(num_kvcache_blocks)
        logger.info(f"ModelRunner rank {self.rank}: import_kv_cache_ipc_handle done")
        return True

    def _bind_kv_cache_to_modules(self, num_kvcache_blocks: int):
        """Bind an IPC-imported KV pool to every attention module.

        The decode side of a P/D pair gets the pool as a handle, so it never
        reaches `allocate_kv_cache`'s bind loop. What it must not do is bind it
        a *second way*: this used to spell the views out again, with a 4-D V
        where every builder declared 5-D. Nothing could catch that — the two
        agree on every byte and differ only in what a reader branches on. So it
        runs the same loop over the same hook, down to backing the pools.

        The paged pool only. A per-request STATE class is sized by `pool_plan`,
        which is empty here -- the block count arrives with the handle and
        nothing else does -- so `allocate_per_req_cache` cannot run and the
        attributes it publishes do not exist. Refused up front rather than met
        as an `AttributeError` on whichever layer binds first.
        """
        stateful = self._state_pool_names()
        if stateful:
            raise NotImplementedError(
                f"per-request state {stateful} has no entry count on the decode "
                "side of a P/D pair: only the paged pool arrives with the "
                "handle, and this side never ran sizing. Run this model "
                "without disaggregation."
            )
        self._back_paged_pools(num_kvcache_blocks, buf=self.kv_cache)

        models_to_bind = [("target", self.model)]
        if self.config.speculative_config and hasattr(self, "drafter"):
            models_to_bind.append(("draft", self.drafter.model))

        kv_cache_data = {}
        for model_name, model in models_to_bind:
            for module in model.modules():
                # Same dispatch as the allocating path's loop: a draft with a
                # pool of its own binds through its own builder, and its pool
                # is now a region of the same imported buffer.
                bound = None
                if model_name == "draft" and hasattr(self, "draft_kv_builder"):
                    bound = self.draft_kv_builder.build_kv_cache_tensor(module)
                if bound is None:
                    bound = self.attn_metadata_builder.build_kv_cache_tensor(module)
                if bound is not None:
                    # The same key the allocating path uses. Keyed by the walk's
                    # ordinal here until now, which agreed with it only while
                    # the first bound module was layer 0.
                    kv_cache_data[f"layer_{module.layer_num}"] = bound

        from atom.utils.forward_context import set_kv_cache_data

        set_kv_cache_data(kv_cache_data)

    # ------------------------------------------------------------------
    # CU-masked stream pools + prefill forward
    # ------------------------------------------------------------------

    @staticmethod
    def _stream_with_cu_mask(mask_bits: list[int]) -> torch.cuda.ExternalStream:
        """Create a HIP stream restricted to the CUs described by mask_bits.

        mask_bits is a list of uint32 words; bit i of word w represents
        CU (w*32 + i).  Uses hipExtStreamCreateWithCUMask (ROCm only).
        """
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipExtStreamCreateWithCUMask.restype = ctypes.c_int
        hip.hipExtStreamCreateWithCUMask.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
        ]
        raw_stream = ctypes.c_void_p()
        mask_arr = (ctypes.c_uint * len(mask_bits))(*mask_bits)
        ret = hip.hipExtStreamCreateWithCUMask(
            ctypes.byref(raw_stream), len(mask_bits), mask_arr
        )
        assert ret == 0, f"HIP err {ret} creating masked stream"
        return torch.cuda.ExternalStream(raw_stream.value)

    @staticmethod
    def _cu_mask_for_fraction(fraction: float, upper: bool) -> list[int]:
        """Return CU mask bits for the given fraction.

        For upper=False (prefill): CUs [0, split).
        For upper=True  (decode):  CUs [split, total).
        split = round(total * fraction).
        """
        total = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        split = max(1, min(total - 1, round(total * fraction)))
        start = split if upper else 0
        end = total if upper else split
        num_words = (total + 31) // 32
        words = [0] * num_words
        for cu in range(start, end):
            words[cu // 32] |= 1 << (cu % 32)
        side = "decode" if upper else "prefill"
        logger.info(
            f"CU mask ({side}): CUs [{start},{end}) "
            f"(fraction={fraction}, total={total})"
        )
        return words

    # CU fractions for which we pre-create masked streams.
    _CU_POOL_FRACTIONS: ClassVar[list[float]] = [0.5]

    def create_prefill_stream_pool(self) -> bool:
        """Create a pool of CUDA streams for disaggregated prefill.

        Called once by PrefillEngineCore._init_disagg() after IPC import.
        In constrained mode, pre-creates one CU-masked stream per fraction
        in _CU_POOL_FRACTIONS plus a full-CU fallback (None key);
        prefill_forward() selects the stream dynamically each iteration via
        _optimal_cu_fraction().  In unconstrained mode, only the plain
        full-CU stream (None key) is created.
        """
        self._prefill_streams = {}
        if getattr(self.config, "disagg_constrained", False):
            for f in self._CU_POOL_FRACTIONS:
                mask = self._cu_mask_for_fraction(f, upper=False)
                self._prefill_streams[f] = self._stream_with_cu_mask(mask)
        # Full-CU fallback (no mask) — always present, sole entry in unconstrained mode
        self._prefill_streams[None] = torch.cuda.Stream()
        logger.info(
            f"Prefill stream pool created: fractions={list(self._prefill_streams.keys())}"
        )
        return True

    def create_decode_stream_pool(self) -> bool:
        """Create a pool of CUDA streams for disaggregated decode.

        Called once by DecodeEngineCore._init_disagg().  In constrained mode,
        complementary to the prefill pool: for fraction F, decode gets CUs
        [F*total, total).  forward() selects the stream dynamically each
        iteration.  In unconstrained mode, only the plain full-CU stream
        (None key) is created.
        """
        self._decode_streams = {}
        if getattr(self.config, "disagg_constrained", False):
            for f in self._CU_POOL_FRACTIONS:
                mask = self._cu_mask_for_fraction(f, upper=True)
                self._decode_streams[f] = self._stream_with_cu_mask(mask)
        # Full-CU fallback (no mask) — always present, sole entry in unconstrained mode
        self._decode_streams[None] = torch.cuda.Stream()
        self._model_fwd_event = torch.cuda.Event()
        self._done_event = torch.cuda.Event()
        logger.info(
            f"Decode stream pool created: fractions={list(self._decode_streams.keys())}"
        )
        return True

    @torch.inference_mode()
    def prefill_forward(self, batch: ScheduledBatch) -> list[int]:
        """Run a prefill forward pass on a dynamically selected CU-masked stream.

        Writes KV for all prompt tokens, samples the first generated token for
        each sequence, then synchronizes before returning so decode's default
        stream sees all KV writes.  Returns a list of sampled token IDs (one
        per sequence, in batch order) — these are included in PrefillDone so
        the decode process can append them before the first decode step,
        matching the num_tokens state that non-disagg postprocess would produce.
        """
        prefill_streams = getattr(self, "_prefill_streams", None)
        if prefill_streams is not None:
            stream = self._prefill_streams[batch.cu_stream_fraction]
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self._advance_forward_vars()
            self._gate_staging_reuse()
            (
                input_ids,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                _needs_independent_noise,
            ) = self.prepare_model(batch)
            self._mark_staging_h2d_enqueued()
            logits, _ = self.run_model(input_ids, batch)
            # Sample the first generated token from each sequence's last logit
            sampled = self.sampler(logits, temperatures, top_ks, top_ps, all_greedy)
            sampled_cpu = sampled.view(-1).tolist()
            self._record_forward_vars_event()
        # Synchronize so decode's default stream sees all KV writes.
        stream.synchronize()
        self._record_kv_cache_ready(batch)
        reset_forward_context()
        return sampled_cpu
