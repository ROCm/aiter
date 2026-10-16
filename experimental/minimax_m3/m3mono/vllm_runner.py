# SPDX-License-Identifier: MIT
"""MiniMax-M3 decode on the mono kernels inside vLLM's own model.

vLLM runs the embedding and the dense layers 0..2 itself; this runner then runs
sparse layers 3..59 as one K4 launch each (the layer's K1 fused in front), both TP
all-reduces in-kernel, and hands back ``(hidden, residual)`` exactly as vLLM's layer
59 would: hidden = the reduced FFN output, residual = the running sum before the
next norm. vLLM's final ``norm(hidden, residual)`` and lm_head follow unchanged.

Kernels: ATOM's FlyDSL M3 mono (vendored, ``kernels/``) or the Gluon port
(``VLLM_M3_MONO_BACKEND=gluon``), both on vLLM's cache contract (``config.VLLM_CACHE``).

Construction is collective over the TP group (every rank binds, all agree, then
the peer handshake). Weights: vLLM keeps attention / indexer projections BF16; the
kernels read ptpc-FP8 (row scale amax / 448, (16, 16) preshuffled) as ATOM's model
holds them, so the runner keeps FP8 copies (o_proj's row scale over the full K).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import torch

from vllm.models.minimax_m3.amd.mono.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    N_ROUTED,
    ONE_INDEX_HEAD,
    PAGE_STRIDE,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
    TP,
    qkv_rows,
)
from vllm.models.minimax_m3.amd.mono.fw.plan.execution import BLOCKS
from vllm.models.minimax_m3.amd.mono.fw.runtime.consensus import MonoUnsupported, bind_agreed
from vllm.models.minimax_m3.amd.mono.fw.runtime.lifecycle import owned_peer_buffer
from vllm.models.minimax_m3.amd.mono.fw.runtime.mailboxes import StepMailboxes
from vllm.models.minimax_m3.amd.mono.fw.runtime.widths import WidthBuilds
from vllm.models.minimax_m3.amd.mono.kernels.post_attn import K4_ABI, build_post_attn_kernel
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import K1_ARGS
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import SCRATCH_BYTES as K1_SCRATCH_BYTES
from vllm.models.minimax_m3.amd.mono.layout import SCRATCH_BYTES as K4_SCRATCH_BYTES
from vllm.models.minimax_m3.amd.mono.layout import sym_layout

logger = logging.getLogger(__name__)

N_DENSE = 3
FP8_MAX = 448.0


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _ptpc_fp8(w: torch.Tensor, tp_group=None) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] -> (FP8 [N, K] preshuffled (16, 16), fp32 row scales [N]): the
    per-token (per output row) quant ATOM's online ptpc_fp8 applies. ``tp_group``:
    a row-parallel weight (K sharded) whose scale spans the full row."""
    from aiter.ops.shuffle import shuffle_weight

    amax = w.float().abs().amax(dim=1)
    if tp_group is not None:
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (w.float() / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), scale.contiguous()


@dataclass
class SparseLayer:
    """One sparse MoE layer's tensors as the kernels read them."""

    layer_id: int
    attn: object  # vLLM MiniMaxM3SparseAttention (its caches and layer name)
    index_cache: object  # MiniMaxM3IndexerCache
    index_topk: bool
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


def _find_index_cache(attn):
    from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerCache

    for m in attn.modules():
        if isinstance(m, MiniMaxM3IndexerCache):
            return m
    raise MonoUnsupported("no indexer cache module")


def _bf16(t: torch.Tensor, n: int, what: str) -> torch.Tensor:
    _need(t.numel() == n, f"{what}: {tuple(t.shape)}")
    return t.detach().to(torch.bfloat16).contiguous().view(n)


def bind_layer(layer, layer_id: int, tp_group, cos_sin: torch.Tensor) -> SparseLayer:
    attn = layer.self_attn
    moe = layer.block_sparse_moe
    _need(hasattr(attn, "skip_index_topk") and hasattr(attn, "indexer"), f"layer {layer_id}: not sparse")
    rows = qkv_rows(ONE_INDEX_HEAD.count)
    wq = attn.qkv_proj.weight
    _need(wq.dtype == torch.bfloat16 and tuple(wq.shape) == (rows, HIDDEN), f"qkv {wq.dtype} {tuple(wq.shape)}")
    w_qkv, s_qkv = _ptpc_fp8(wq)
    wo = attn.o_proj.weight
    _need(wo.dtype == torch.bfloat16 and tuple(wo.shape) == (HIDDEN, LOCAL_Q_HEADS * HEAD_DIM), f"o_proj {tuple(wo.shape)}")
    w_o, s_o = _ptpc_fp8(wo, tp_group)
    # vLLM's FusedMoEFactory returns a MoERunner; the module holding the weights
    # is one of its children
    ex = next((m for m in moe.experts.modules() if hasattr(m, "w13_weight")), None)
    _need(ex is not None, "no module with w13_weight under experts")
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        _need(hasattr(ex, name), f"experts.{name}")
    _need(ex.w13_weight.shape[0] == N_ROUTED + 1, f"experts {ex.w13_weight.shape[0]}: fused shared expert")
    gate = moe.gate.weight
    _need(tuple(gate.shape) == (N_ROUTED, HIDDEN), f"gate {tuple(gate.shape)}")
    bias = moe.e_score_correction_bias
    _need(bias is not None and bias.numel() == N_ROUTED, "routing bias")
    return SparseLayer(
        layer_id=layer_id, attn=attn, index_cache=_find_index_cache(attn),
        index_topk=not attn.skip_index_topk,
        g_in=_bf16(layer.input_layernorm.weight, HIDDEN, "input norm"),
        w_qkv=w_qkv, s_qkv=s_qkv,
        g_q=_bf16(attn.q_norm.weight, HEAD_DIM, "q_norm"), g_k=_bf16(attn.k_norm.weight, HEAD_DIM, "k_norm"),
        g_iq=_bf16(attn.index_q_norm.weight, HEAD_DIM, "index_q_norm"),
        g_ik=_bf16(attn.index_k_norm.weight, HEAD_DIM, "index_k_norm"),
        cos_sin=cos_sin, w_o=w_o, s_o=s_o,
        g_post=_bf16(layer.post_attention_layernorm.weight, HIDDEN, "post norm"),
        # ATOM's mono reads the bf16 view of the gate (vLLM computes fp32 logits)
        gate=gate.detach().to(torch.bfloat16).contiguous(),
        bias=bias.detach().float().contiguous(),
        w13=ex.w13_weight.data, s13=ex.w13_weight_scale.data,
        w2=ex.w2_weight.data, s2=ex.w2_weight_scale.data,
    )  # fmt: skip


class M3MonoRunner:
    """Owns the weight views, scratch, peer buffer and the per-width kernels."""

    tl_buf = None  # a timeline build's stamps: int64 [layer][CTA][point] (bench)

    def __init__(self, model, vllm_config) -> None:
        from vllm.distributed import get_tp_group

        self.group = get_tp_group()
        cpu_group = self.group.cpu_group
        bind_agreed(lambda: self._bind(model, vllm_config), cpu_group)
        self._allocate(cpu_group)
        self.k4 = WidthBuilds(self._build_width, self._width_kernels, cpu_group, "MiniMax-M3 mono (vLLM)")

    # ------------------------------------------------------------------ setup
    def _bind(self, model, vllm_config) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        npes = get_tensor_model_parallel_world_size()
        _need(npes == TP, f"TP {npes}")
        self.npes, self.rank = npes, get_tensor_model_parallel_rank()
        self.model = model
        cfg = model.config
        layers = model.layers
        _need(len(layers) > N_DENSE and model.start_layer == 0, "layers")
        cache = vllm_config.cache_config
        _need(cache.block_size == 128, f"block size {cache.block_size}")
        _need(str(cache.cache_dtype).startswith("fp8"), f"kv cache {cache.cache_dtype}")
        self.dev = torch.device("cuda", torch.cuda.current_device())
        cus = torch.cuda.get_device_properties(self.dev).multi_processor_count
        _need(cus == BLOCKS, f"{cus} CUs: the kernels need all {BLOCKS} CTAs co-resident")
        rot = layers[N_DENSE].self_attn.rotary_emb
        cs = rot.cos_sin_cache
        _need(cs.dim() == 2 and cs.shape[1] == ROTARY_DIM, f"cos/sin {tuple(cs.shape)}")
        self.cos_sin = cs.detach().to(device=self.dev, dtype=torch.bfloat16).contiguous()
        tp_group = self.group.device_group
        self.sparse = [bind_layer(layers[i], i, tp_group, self.cos_sin) for i in range(N_DENSE, len(layers))]
        _need(self.sparse[0].index_topk, "the first sparse layer reuses a selection")
        attn = self.sparse[0].attn
        _need(abs(attn.scaling - HEAD_DIM**-0.5) < 1e-12, f"softmax scale {attn.scaling}")
        from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

        meta = rocm_aiter_moe.aiter_topK_meta_data
        _need(meta is not None, "no fused shared expert routing table")
        total_w, total_ids = meta
        _need(int(total_ids[0, TOP_K].item()) == N_ROUTED, "fused shared expert id")
        shared_weight = float(total_w[0, TOP_K].item())
        moe = layers[N_DENSE].block_sparse_moe
        route_scale = float(getattr(moe, "routed_scaling_factor", getattr(cfg, "routed_scaling_factor", 2.0)))
        swiglu_limit = float(getattr(cfg, "swiglu_limit", 7.0))
        _need(abs(float(getattr(cfg, "swiglu_alpha", 1.702)) - 1.702) < 1e-6, "swiglu alpha")
        sparse_cfg = cfg.sparse_attention_config
        _need(sparse_cfg["sparse_topk_blocks"] == TOPK_BLOCKS, "indexer top-k blocks")
        from vllm.models.minimax_m3.amd.mono.compat import envs

        self.debug = bool(envs.ATOM_MONO_DEBUG)
        self._build_k4 = functools.partial(
            build_post_attn_kernel, npes, HEAD_DIM**-0.5, float(cfg.rms_norm_eps), route_scale, shared_weight,
            swiglu_limit, 0, 1, fuse_k1=True, heads=ONE_INDEX_HEAD, debug=self.debug,
        )  # fmt: skip
        logger.info("MiniMax-M3 mono: %d sparse layers bound (route scale %s, shared weight %s)",
                    len(self.sparse), route_scale, shared_weight)

    def _build_width(self, n: int) -> dict:
        return {it: self._build_k4(n, index_topk=it) for it in {lw.index_topk for lw in self.sparse}}

    @staticmethod
    def _width_kernels(kernels: dict) -> list:
        return [(launch, K4_ABI) for launch in kernels.values()]

    def _allocate(self, cpu_group) -> None:
        dev = self.dev
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch4 = torch.zeros(K4_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.peers, self._finalizer = owned_peer_buffer(
            self, sym_layout(self.npes)["_bytes"], cpu_group, self.rank, self.npes, dev
        )
        self.mailboxes = StepMailboxes(self.peers, self.scratch1, self.scratch4, debug=self.debug)
        bf16 = torch.bfloat16
        self.ars = [torch.zeros(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)]
        self.h_mids = [torch.zeros(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)]
        self.h = torch.zeros(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        self.q = torch.zeros(MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf16, device=dev)
        self.iq = torch.zeros(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=dev)
        self.batch_ids = torch.full((MAX_TOKENS,), -1, dtype=torch.int32, device=dev)
        self.row_ids = torch.arange(MAX_TOKENS, dtype=torch.int32, device=dev)
        self.k1_args = []
        for i, lw in enumerate(self.sparse):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in), "w_qkv": _ptr(lw.w_qkv),
                "s_qkv": _ptr(lw.s_qkv), "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik), "cos_sin": _ptr(lw.cos_sin),
                "index_cache": 0, "iq_out": _ptr(self.iq), "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(ptrs)
        self._k1_dev = None
        self._seen = set()

    def _k1_arrays(self) -> list[torch.Tensor]:
        """Per layer the K1 pointer array; the index cache pointer is bound once the
        KV caches are (first step)."""
        if self._k1_dev is None:
            arrs = []
            for lw, ptrs in zip(self.sparse, self.k1_args):
                p = dict(ptrs, index_cache=_ptr(lw.index_cache.kv_cache))
                arrs.append(torch.tensor([p[a] for a in K1_ARGS], dtype=torch.int64, device=self.dev))
            self._k1_dev = arrs
        return self._k1_dev

    def prepare(self, n: int) -> bool:
        return self.k4.prepare(n)

    def close(self) -> None:
        self._finalizer()

    # ------------------------------------------------------------------ step
    def sparse_metadata(self):
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return None
        return md.get(self.sparse[0].attn.layer_name)

    def begin(self, positions: torch.Tensor, n: int) -> None:
        """A step's metadata view, row liveness and mailbox reset (before layer 0)."""
        md = self.sparse_metadata()
        dec = md.decode
        q_len = int(dec.decode_query_len)
        block_table, seq_lens = dec.block_table, dec.seq_lens
        if q_len > 1:  # a speculative verify: one row per token (ATOM's token_rows)
            reqs = n // q_len
            back = torch.arange(q_len - 1, -1, -1, dtype=seq_lens.dtype, device=seq_lens.device)
            block_table = block_table[:reqs].repeat_interleave(q_len, dim=0)
            seq_lens = (seq_lens[:reqs, None] - back).reshape(-1)
            req = self.row_ids[:n]
        else:
            req = self.row_ids[:n]
        # row r's request, -1 on a graph pad row (its request has no context)
        live = seq_lens[:n].index_select(0, req.long()) > 0
        self.batch_ids[:n].copy_(torch.where(live, req // q_len, torch.full_like(req, -1)))
        key = (n, tuple(block_table.shape), tuple(block_table.stride()), tuple(seq_lens.shape), q_len,
               tuple(md.slot_mapping.shape), str(md.slot_mapping.dtype))
        if key not in self._seen:
            self._seen.add(key)
            if self.rank == 0:
                logger.info("M3 mono step layout: n=%d block_table %s stride %s seq_lens %s q_len %d slots %s %s "
                            "capturing=%s", *key[:1], key[1], key[2], key[3], q_len, key[5], key[6],
                            torch.cuda.is_current_stream_capturing())
        _need(block_table.stride(-1) == 1 and block_table.dtype == torch.int32, "block table layout")
        self._step = (md, q_len, block_table, seq_lens, positions)
        self.mailboxes.begin_step()

    def layer(self, i: int, res: torch.Tensor) -> torch.Tensor:
        """Sparse layer i: (ars[i % 2][:n], res) -> (ars[(i + 1) % 2][:n], returned residual)."""
        md, q_len, block_table, seq_lens, positions = self._step
        n = res.shape[0]
        lw = self.sparse[i]
        k1 = self._k1_arrays()
        k16, v16 = lw.attn.get_aiter_sparse_pa_kv_cache()
        args = {
            "h_in": _ptr(self.h), "q": _ptr(self.q), "block_table": _ptr(block_table),
            "seq_lens": _ptr(seq_lens), "k_cache": _ptr(k16), "v_cache": _ptr(v16),
            "k_scale": 0, "v_scale": 0, "w_o": _ptr(lw.w_o),
            "s_o": _ptr(lw.s_o), "g_post": _ptr(lw.g_post), "w_gate": _ptr(lw.gate),
            "bias": _ptr(lw.bias), "w13": _ptr(lw.w13), "s13": _ptr(lw.s13),
            "w2": _ptr(lw.w2), "s2": _ptr(lw.s2), "h_mid": _ptr(self.h_mids[(i + 1) % 2]),
            "ar_out": _ptr(self.ars[(i + 1) % 2]), "scratch": _ptr(self.scratch4),
            **self.peers.kernel_args(), "layer": lw.layer_id,
            "bt_width": block_table.stride(0), "q_len": q_len, "tl": 0 if self.tl_buf is None else _ptr(self.tl_buf[i]),
            "k1_args": _ptr(k1[i]), "positions": _ptr(positions),
            "slot_mapping": _ptr(md.slot_mapping), "res": _ptr(res),
            "batch_ids": _ptr(self.batch_ids),
        }  # fmt: skip
        self.k4[n][lw.index_topk](*K4_ABI.pack(args), stream=torch.cuda.current_stream())
        return self.h_mids[(i + 1) % 2][:n]

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor):
        """Layer 2's (hidden, residual) -> layer 59's."""
        n = hidden.shape[0]
        self.begin(positions, n)
        self.ars[0][:n].copy_(hidden)
        res = residual
        for i in range(len(self.sparse)):
            res = self.layer(i, res)
        if self.debug and not torch.cuda.is_current_stream_capturing():
            self.raise_on_given_up_waits(n)
        return self.ars[len(self.sparse) % 2][:n], res

    def raise_on_given_up_waits(self, n: int) -> None:
        """A debug build's step: raise with every mailbox wait that gave up."""
        from vllm.models.minimax_m3.amd.mono.fw.runtime.debug import DIAG_BYTES, given_up_waits, raise_if_given_up, region_namer
        from vllm.models.minimax_m3.amd.mono.layout import SCRATCH, diag_region_names

        off = SCRATCH["diag"]
        waits = given_up_waits(self.scratch4[off : off + DIAG_BYTES], region_namer(diag_region_names(n, ONE_INDEX_HEAD.count)))
        raise_if_given_up("MiniMax-M3 mono", self.rank, n, waits, self.mailboxes)

    def check_caches(self) -> None:
        """vLLM's sparse KV layout must be the interleaved one the kernels assume."""
        for lw in self.sparse:
            k16, v16 = lw.attn.get_aiter_sparse_pa_kv_cache()
            stride = getattr(lw.attn, "_aiter_sparse_pa_block_page_stride", None)
            _need(stride == PAGE_STRIDE, f"layer {lw.layer_id}: page stride {stride}")
            _need(v16.data_ptr() - k16.data_ptr() == 8 * 2048, f"layer {lw.layer_id}: V offset")
            ic = lw.index_cache.kv_cache
            _need(ic.dtype == torch.bfloat16 and ic.is_contiguous() and ic.shape[-1] == HEAD_DIM
                  and ic.shape[-2] == 128, f"layer {lw.layer_id}: index cache {ic.dtype} {tuple(ic.shape)}")
