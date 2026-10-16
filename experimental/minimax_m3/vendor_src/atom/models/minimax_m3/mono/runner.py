# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One decode step of 1..MAX_TOKENS tokens on the mono kernels (a speculative
verify's q tokens of a request count as q, see ``token_rows``).

Each dense layer 0..2 is three launches: K1 built dense (``dense_pre``: q, the
K / V insert), its original attention kernel, then ``dense_post`` (o_proj
through the MLP, both all-reduces in-kernel -> (ar, h_mid)). Every sparse MoE
layer is one launch of K4 with the layer's K1 fused in front of its stages::

    K1 part        (ar, res) -> h, q; K / V / index cache insert; indexer scores
    K4 part        top-k, attention .. FFN, both all-reduces in-kernel
                   -> (ar, h_mid)

A layer the original path runs with ``skip_index_topk`` launches the build
without the indexer (no index q / k, scores or top-k): its attention takes the
sparse table the step's last selecting layer left in scratch.

with ``(ar, res)`` the (reduced partials, residual) pair the original path hands
its fused all-reduce + RMSNorm, so a layer computes the same values either way.
A launch reads the previous layer's (ar, h_mid) while it writes its own, so both
alternate between two buffers by layer parity.
"""

import functools
import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

import atom.model_ops.topK as topk_meta
from atom.distributed.indexer_cp import indexer_cp_enabled
from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    N_ROUTED,
    ONE_INDEX_HEAD,
    TOP_K,
    TOPK_BLOCKS,
    TP,
    IndexHeads,
)
from atom.models.minimax_m3.mono.kernels.dense_post import (
    DENSE_POST_ABI,
    build_dense_post_kernel,
)
from atom.models.minimax_m3.mono.kernels.post_attn import (
    K4_ABI,
    TL_POINTS,
    build_post_attn_kernel,
)
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    K1_ABI,
    K1_ARGS,
    build_pre_attn_kernel,
)
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    SCRATCH_BYTES as K1_SCRATCH_BYTES,
)
from atom.models.minimax_m3.mono.layout import SCRATCH, diag_region_names, sym_layout
from atom.models.minimax_m3.mono.layout import SCRATCH_BYTES as K4_SCRATCH_BYTES
from atom.models.minimax_m3.mono.weights import DenseLayer, SparseMoeLayer
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.debug import (
    DIAG_BYTES,
    given_up_waits,
    raise_if_given_up,
    region_namer,
)
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.timeline import LayerTimeline
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

N_DENSE = 3
# a width's kernels by kind (``_build_layer_kernels``): the sparse ones' is K4's
_KERNEL_ABI = {
    "dense_pre0": K1_ABI,
    "dense_pre": K1_ABI,
    "dense_post": DENSE_POST_ABI,
}


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _sparse_metadata(fwd):
    md = fwd.attn_metadata
    return getattr(md, "sparse_attention_metadata", None) or md


def token_rows(fwd, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(block table, seq_lens) with one row per token of the step: a request of q
    query tokens (speculative verify) becomes q rows, its token j seeing
    ``seq_len - (q - 1 - j)`` keys, so the kernels treat every token alike."""
    decode = _sparse_metadata(fwd).decode
    q = decode.max_query_len
    if q == 1:
        return decode.block_table, decode.seq_lens
    reqs = n // q
    back = torch.arange(
        q - 1, -1, -1, dtype=decode.seq_lens.dtype, device=decode.seq_lens.device
    )
    return (
        decode.block_table[:reqs].repeat_interleave(q, dim=0),
        (decode.seq_lens[:reqs, None] - back).reshape(-1),
    )


class MonoDecodeRunner:
    """Owns the per-layer weight views, scratch, peer buffers and compiled kernels.

    Construction is collective over the TP group: every rank binds and checks its
    shard, then all agree (``consensus.bind_agreed``) before the peer handshake, so a rank that
    refuses turns mono off on every rank instead of leaving the others waiting."""

    def __init__(self, causal_lm) -> None:
        group = get_tp_group().cpu_group
        bind_agreed(lambda: self._bind(causal_lm), group)
        self._allocate(group)
        # per decode batch size (every graph captures its own) a sparse layer
        # kernel of each kind the layers use (``self.k4[n][lw.index_topk]``) and
        # the dense layers' (``self.k4[n]["dense_pre0" | "dense_pre" |
        # "dense_post"]``), built and compiled on its first step (``prepare``)
        self.k4 = WidthBuilds(
            self._build_layer_kernels, self._layer_kernels, group, "MiniMax-M3 mono"
        )

    def _build_layer_kernels(self, n: int) -> dict:
        kernels = {
            index_topk: self._build_k4(n, index_topk=index_topk)
            for index_topk in {lw.index_topk for lw in self.sparse}
        }
        kernels["dense_pre0"] = self._build_dense_pre(n, plain_norm=True)
        kernels["dense_pre"] = self._build_dense_pre(n)
        kernels["dense_post"] = self._build_dense_post(n)
        return kernels

    @staticmethod
    def _layer_kernels(kernels: dict) -> list:
        """A width's layer kernels, each with its ABI."""
        return [
            (launch, _KERNEL_ABI.get(kind, K4_ABI)) for kind, launch in kernels.items()
        ]

    def _bind(self, causal_lm) -> None:
        """This rank's checks and weight views; raises ``MonoUnsupported``."""
        model = causal_lm.model
        config = causal_lm.config
        npes = get_tensor_model_parallel_world_size()
        if npes != TP:
            raise MonoUnsupported(f"TP {npes}")
        layers = model.layers
        if len(layers) <= N_DENSE or any(
            getattr(layers[i].self_attn, "is_indexed_sparse_attention", False)
            or layers[i].is_moe_layer
            for i in range(N_DENSE)
        ):
            raise MonoUnsupported("layers 0..2 must be dense full-attention layers")
        self.model = model
        self.npes = npes
        self.rank = get_tensor_model_parallel_rank()
        # indexer context parallelism: the fused projection holds every index q
        # head, the rank's own being its TP rank
        heads = IndexHeads(TP, self.rank) if indexer_cp_enabled() else ONE_INDEX_HEAD
        self.index_heads = heads.count
        # bounded mailbox waits, their records raised after the step (finish_step)
        self.debug = envs.ATOM_MONO_DEBUG
        self.sparse = [
            SparseMoeLayer.from_layer(layers[i], i, heads)
            for i in range(N_DENSE, len(layers))
        ]
        # a reusing layer reads the selection an earlier layer of the step left
        # (the original path raises on a cache miss there)
        if not self.sparse[0].index_topk:
            raise MonoUnsupported("the first sparse layer reuses a selection")
        self.dense = [DenseLayer.from_layer(layers[i], i) for i in range(N_DENSE)]
        for d in self.dense:
            # the decode dispatch reads what the original rope_cache step sets on
            # the impl; mono skips that step (a decode can precede every prefill:
            # the graph capture), so it sets it by the same rule
            d.attn_impl.use_triton_attn = (
                envs.ATOM_FORCE_ATTN_TRITON
                or d.attn_impl.sliding_window != -1
                or d.attn_impl.head_dim != 128
            )
        swiglu = {(d.swiglu_alpha, d.swiglu_beta, d.swiglu_limit) for d in self.dense}
        if len(swiglu) != 1:
            raise MonoUnsupported(f"dense layers' swiglu parameters {swiglu}")
        impl = self.sparse[0].attn_impl
        experts = layers[N_DENSE].block_sparse_moe.experts
        eps = config.rms_norm_eps
        route_scale = float(layers[N_DENSE].block_sparse_moe.routed_scaling_factor)
        # the fused shared expert's routing weight is whatever the topK metadata holds
        # read through the module: the metadata is assigned after this file is imported
        total_w, total_ids = topk_meta.aiter_topK_meta_data
        if int(total_ids[0, TOP_K].item()) != N_ROUTED:
            raise MonoUnsupported("fused shared expert id")
        shared_weight = float(total_w[0, TOP_K].item())
        self.sm_scale = impl.scale
        if abs(self.sm_scale - HEAD_DIM**-0.5) > 1e-12:
            raise MonoUnsupported(f"softmax scale {self.sm_scale}")
        if impl.topk != TOPK_BLOCKS:
            raise MonoUnsupported(f"indexer top-{impl.topk} blocks")

        self.dev = torch.device("cuda", torch.cuda.current_device())
        cus = torch.cuda.get_device_properties(self.dev).multi_processor_count
        if cus != BLOCKS:
            raise MonoUnsupported(
                f"{cus} CUs: the kernels need all {BLOCKS} CTAs co-resident"
            )
        tl_prefix = envs.ATOM_MONO_TIMELINE
        self.timeline = (
            LayerTimeline(tl_prefix, len(self.sparse), TL_POINTS, self.rank, self.dev)
            if tl_prefix
            else None
        )
        # the s-token layer kernel is ``self._build_k4(s)`` (``WidthBuilds``)
        self._build_k4 = functools.partial(
            build_post_attn_kernel,
            npes,
            self.sm_scale,
            eps,
            route_scale,
            shared_weight,
            experts.swiglu_limit,
            impl.init_blocks,
            impl.local_blocks,
            fuse_k1=True,
            timeline=self.timeline is not None,
            heads=heads,
            debug=self.debug,
        )
        # a dense layer's kernels around its original attention
        self._build_dense_pre = functools.partial(
            build_pre_attn_kernel,
            eps,
            self.sm_scale,
            impl.init_blocks,
            impl.local_blocks,
            dense=True,
        )
        self._build_dense_post = functools.partial(
            build_dense_post_kernel,
            npes,
            eps,
            *swiglu.pop(),
            debug=self.debug,
            index_heads=heads.count,
        )

    def _allocate(self, group) -> None:
        """Scratch, the peer handshake and the activation buffers (every rank)."""
        dev = self.dev
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch4 = torch.zeros(K4_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            sym_layout(self.npes)["_bytes"],
            group,
            self.rank,
            self.npes,
            dev,
        )
        # every mailbox pair, zeroed at each step's start (forward)
        self.mailboxes = StepMailboxes(
            self.peers, self.scratch1, self.scratch4, debug=self.debug
        )
        bf16 = torch.bfloat16
        # row k = token k of the step; sparse layer i reads ars[i % 2] (the
        # previous layer's output) and writes ars[(i + 1) % 2], likewise h_mids
        self.ars = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h_mids = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        self.q = torch.empty(
            MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf16, device=dev
        )
        self.iq = torch.empty(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=dev)
        # layer 0's residual input: its K1 takes (embedding, 0), acc = embedding
        self.zero_res = torch.zeros(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        # each layer's K1 pointers that do not change per step (K1_ARGS order)
        self.k1_args = []
        for i, lw in enumerate(self.sparse):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in), "w_qkv": _ptr(lw.w_qkv),
                "s_qkv": _ptr(lw.s_qkv), "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik), "cos_sin": _ptr(lw.cos_sin),
                "index_cache": _ptr(lw.attn_impl.index_cache), "iq_out": _ptr(self.iq),
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )

    def prepare(self, n: int) -> bool:
        """Whether every rank holds the n-token kernel (``WidthBuilds``)."""
        return self.k4.prepare(n)

    def close(self) -> None:
        """Free the peer memory now (once no rank can launch another step)."""
        self._finalizer()

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """The final hidden states; with Eagle3 aux layers set, also the residual
        stream entering each of them, as the original model returns."""
        aux_layers = self.model.aux_hidden_state_layers
        aux: list[torch.Tensor] = []
        self.mailboxes.begin_step()
        fwd = get_forward_context()
        res = self.run_dense_mono(input_ids, positions, fwd, aux_layers, aux)
        rows = token_rows(fwd, res.shape[0])
        for i, lw in enumerate(self.sparse):
            res = self.run_sparse_layer(i, lw, fwd, positions, res, rows)
            if lw.layer_id in aux_layers:
                aux.append(self.h[: res.shape[0]].clone())
        hidden = self.finish_step(res)
        if self.timeline is not None:
            self.timeline.step_done(res.shape[0])
        return (hidden, aux) if aux else hidden

    def run_dense_mono(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        fwd,
        aux_layers: tuple[int, ...],
        aux: list[torch.Tensor],
    ) -> torch.Tensor:
        """The embedding, then each dense layer as dense_pre -> its original
        attention -> dense_post. Dense layer d writes ars[d % 2] / h_mids[d % 2]:
        the last one leaves the first sparse layer's (ar, res) in ars[0] and the
        returned residual (h_mids[0]). Appends the aux hidden state of the dense
        ones among ``aux_layers``."""
        n = input_ids.numel()
        kernels = self.k4[n]
        stream = torch.cuda.current_stream()
        md = fwd.attn_metadata
        ar, res = self.model.get_input_embeddings(input_ids), self.zero_res[:n]
        for d, lw in enumerate(self.dense):
            kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
            pre = kernels["dense_pre0" if d == 0 else "dense_pre"]
            pre(
                *K1_ABI.pack(
                    {
                        "ar": _ptr(ar), "res": _ptr(res), "h_out": _ptr(self.h),
                        "g_in": _ptr(lw.g_in), "w_qkv": _ptr(lw.w_qkv),
                        "s_qkv": _ptr(lw.s_qkv), "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                        "g_iq": 0, "g_ik": 0, "cos_sin": _ptr(lw.cos_sin),
                        "positions": _ptr(positions), "slot_mapping": _ptr(md.slot_mapping),
                        "k_cache": _ptr(kv.k_cache), "v_cache": _ptr(kv.v_cache),
                        "k_scale": _ptr(kv.k_scale), "v_scale": _ptr(kv.v_scale),
                        "index_cache": 0, "q_out": _ptr(self.q), "iq_out": 0,
                        "block_table": 0, "seq_lens": 0, "bt_width": 0, "q_len": 1,
                        "iscore": 0, "scratch": _ptr(self.scratch1),
                        "layer": lw.layer_id, "tl": 0,
                    }
                ),
                stream=stream,
            )  # fmt: skip
            if lw.layer_id in aux_layers:
                aux.append(self.h[:n].clone())
            attn = self.attend_dense(lw, n, fwd, kv)
            kernels["dense_post"](
                *DENSE_POST_ABI.pack(
                    {
                        "attn": _ptr(attn), "h_in": _ptr(self.h), "w_o": _ptr(lw.w_o),
                        "s_o": _ptr(lw.s_o), "g_post": _ptr(lw.g_post),
                        "w_gu": _ptr(lw.w_gu), "s_gu": _ptr(lw.s_gu),
                        "w_dn": _ptr(lw.w_dn), "s_dn": _ptr(lw.s_dn),
                        "h_mid": _ptr(self.h_mids[d % 2]),
                        "ar_out": _ptr(self.ars[d % 2]), "scratch": _ptr(self.scratch4),
                        **self.peers.kernel_args(), "layer": lw.layer_id, "tl": 0,
                    }
                ),
                stream=stream,
            )  # fmt: skip
            ar, res = self.ars[d % 2][:n], self.h_mids[d % 2][:n]
        return res

    def attend_dense(self, lw: DenseLayer, n: int, fwd, kv) -> torch.Tensor:
        """Dense layer ``lw``'s original decode attention over dense_pre's q and
        the cache it inserted (the dispatch the original forward makes, without
        its rope / cache step) -> [n, O_K] bf16. A dummy run skips it as the
        original does."""
        q = self.q[:n].view(n, LOCAL_Q_HEADS, HEAD_DIM)
        if fwd.context.is_dummy_run:
            return q.view(n, -1)
        kvq = q[:, :1]  # decode reads q and the caches only
        impl = lw.attn_impl
        attend = impl.dispatch_backend(fwd, q, kvq, kvq)
        o = attend(q, kvq, kvq, kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale, fwd)
        return o.view(n, -1)

    def run_dense(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux_layers: tuple[int, ...] = (),
        aux: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Embedding and the dense layers on the original modules; leaves the first
        sparse layer's (ar, res) in ``self.ars[0][:S]`` and the returned residual.
        Appends the aux hidden state of the dense ones among ``aux_layers``."""
        n = input_ids.numel()
        hidden = self.model.get_input_embeddings(input_ids)
        residual = None
        for i in range(N_DENSE):
            if i in aux_layers:
                hidden, residual, aux_hidden = self.model.layers[i](
                    positions, hidden, residual, capture_aux=True
                )
                aux.append(aux_hidden)
            else:
                hidden, residual = self.model.layers[i](positions, hidden, residual)
        # the all-reduce layer N_DENSE's fused AR + RMSNorm would do, without the norm
        self.ars[0][:n].copy_(tensor_model_parallel_all_reduce(hidden).view(n, HIDDEN))
        return residual.view(n, HIDDEN)

    def run_sparse_layer(
        self,
        i: int,
        lw: SparseMoeLayer,
        fwd,
        positions: torch.Tensor,
        res: torch.Tensor,
        rows: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Sparse layer i: (ars[i % 2][:S], res) -> (ars[(i + 1) % 2][:S], returned
        residual). ``rows``: ``token_rows``."""
        n = res.shape[0]
        sparse_md = _sparse_metadata(fwd)
        block_table, seq_lens = rows
        stream = torch.cuda.current_stream()
        impl = lw.attn_impl
        kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
        k16, v16, ks16, vs16 = impl._to_page16_shuffle(
            kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale
        )
        args = {
            "h_in": _ptr(self.h), "q": _ptr(self.q), "block_table": _ptr(block_table),
            "seq_lens": _ptr(seq_lens), "k_cache": _ptr(k16), "v_cache": _ptr(v16),
            "k_scale": _ptr(ks16), "v_scale": _ptr(vs16), "w_o": _ptr(lw.w_o),
            "s_o": _ptr(lw.s_o), "g_post": _ptr(lw.g_post), "w_gate": _ptr(lw.gate),
            "bias": _ptr(lw.bias), "w13": _ptr(lw.w13), "s13": _ptr(lw.s13),
            "w2": _ptr(lw.w2), "s2": _ptr(lw.s2), "h_mid": _ptr(self.h_mids[(i + 1) % 2]),
            "ar_out": _ptr(self.ars[(i + 1) % 2]), "scratch": _ptr(self.scratch4),
            **self.peers.kernel_args(), "layer": lw.layer_id,
            "bt_width": block_table.shape[1],
            "q_len": sparse_md.decode.max_query_len,
            "tl": self.timeline.ptr(i) if self.timeline else 0,
            "k1_args": _ptr(self.k1_args[i]), "positions": _ptr(positions),
            "slot_mapping": _ptr(sparse_md.slot_mapping), "res": _ptr(res),
            "batch_ids": _ptr(fwd.attn_metadata.batch_id_per_q_token),
        }  # fmt: skip
        self.k4[n][lw.index_topk](*K4_ABI.pack(args), stream=stream)
        return self.h_mids[(i + 1) % 2][:n]

    def finish_step(self, res: torch.Tensor) -> torch.Tensor:
        """The final norm (the mailboxes are cleared at the next step's start:
        ``StepMailboxes.begin_step``)."""
        ar = self.ars[len(self.sparse) % 2]
        hidden, _ = self.model.norm(ar[: res.shape[0]], res)
        if self.debug and not torch.cuda.is_current_stream_capturing():
            self.raise_on_given_up_waits(res.shape[0])
        return hidden

    def raise_on_given_up_waits(self, n: int) -> None:
        """A debug build's step: raise with every mailbox wait that gave up (its
        region, pair index, tags, CTA) and every rank the fence gave up on."""
        off = SCRATCH["diag"]
        waits = given_up_waits(
            self.scratch4[off : off + DIAG_BYTES],
            region_namer(diag_region_names(n, self.index_heads)),
        )
        raise_if_given_up("MiniMax-M3 mono", self.rank, n, waits, self.mailboxes)
