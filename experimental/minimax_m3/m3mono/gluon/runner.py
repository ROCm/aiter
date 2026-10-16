# SPDX-License-Identifier: MIT
"""Host side of the Gluon MiniMax-M3 step kernel: one launch a decode step for sparse
layers 3..59, on the same weights and vLLM cache contract as the FlyDSL runner
(``vllm_runner.SparseLayer``: ptpc-FP8 qkv / o copies, the bf16 gate, aiter-shuffled
MXFP4 experts, page-16 K / V views, the bf16 index cache).

Every launch argument is a fixed device pointer and the epoch lives on the device, so a
step is graph-capturable. Each width is compiled (and checked to fit one CTA a CU) on
every rank, then the ranks align, before its first launch.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from vllm.models.minimax_m3.amd.mono.config import HEAD_DIM
from vllm.models.minimax_m3.amd.mono.fw.runtime.consensus import MonoUnsupported, tp_agree
from vllm.models.minimax_m3.amd.mono.vllm_runner import M3MonoRunner, _need

from . import layout as LY

logger = logging.getLogger(__name__)

# one launch for every layer of a step (LOOP) or a launch a layer
LOOP = os.environ.get("VLLM_M3_GLUON_LOOP", "0") == "1"
FIRST_ERR = os.environ.get("VLLM_M3_GLUON_FIRST_ERR", "0") == "1"
# stage mask (kernel.layer_body bits; debugging)
STAGES = int(os.environ.get("VLLM_M3_GLUON_STAGES", "511"))

# a layer's pointer-table entries (kernel.NPTR), in this order
LAYER_PTRS = (
    "w_qkv", "s_qkv", "g_in", "g_q", "g_k", "g_iq", "g_ik", "w_o", "s_o", "g_post", "gate", "bias", "w13",
    "s13", "w2", "s2", "k_cache", "v_cache", "index_cache", "index_topk",
)


class _Widths:
    """``prepare(n)``: the width-n kernel compiled on every rank (the FlyDSL runner's
    ``k4`` interface, as the hook calls it)."""

    def __init__(self, runner):
        self.r = runner
        self.ready = set()

    def prepare(self, n: int) -> bool:
        if n in self.ready:
            return True
        if not 1 <= n <= LY.MAX_S:
            return False
        ok = True
        try:
            self.r._compile(n)
        except Exception:
            logger.exception("Gluon M3: building the %d-row kernel failed", n)
            ok = False
        if not tp_agree(ok, self.r.cpu_group):
            return False
        self.ready.add(n)
        return True


class GluonCore:
    """State and launch shared by the vLLM runner and the standalone bench runner.
    Needs ``self.sparse``, ``self.rank``, ``self.dev``, ``self.cos_sin``, ``self.cpu_group``."""

    tl_buf = None

    def _gluon_setup(self, eps: float, route_scale: float, shared_weight: float, swiglu_limit: float,
                     timeline: bool = False) -> None:
        from vllm.models.minimax_m3.amd.mono.fw.runtime.lifecycle import owned_peer_buffer

        dev = self.dev
        self.eps, self.route_scale, self.shared_weight, self.swiglu_limit = eps, route_scale, shared_weight, swiglu_limit
        self.sm_scale = HEAD_DIM ** -0.5
        self.scratch = torch.zeros(LY.StepLayout().nbytes(), dtype=torch.uint8, device=dev)
        self.peers, self._finalizer = owned_peer_buffer(self, 2 * LY.peer_half(), self.cpu_group, self.rank, LY.TP,
                                                        dev)
        self.peers.bytes.zero_()
        self.epoch = torch.zeros(LY.EPOCH_WORDS, dtype=torch.int32, device=dev)
        # [BLOCKS] codes, then (FIRST_ERR debug) the layer being launched
        self.errors = torch.zeros(2 * LY.BLOCKS, dtype=torch.int32, device=dev)
        n = len(self.sparse)
        self.trace = torch.zeros(n * LY.BLOCKS * 32, dtype=torch.int64, device=dev)
        if timeline:
            self.tl_buf = self.trace.view(n, LY.BLOCKS, 32)
        bf16 = torch.bfloat16
        self.ar_out = torch.zeros(16, LY.HIDDEN, dtype=bf16, device=dev)
        self.hm_out = torch.zeros(16, LY.HIDDEN, dtype=bf16, device=dev)
        self._ptab = None
        self.k4 = _Widths(self)

    def _ptab_tensor(self) -> torch.Tensor:
        """Per layer the pointer table (the caches are bound by the first step)."""
        if self._ptab is None:
            rows = []
            for lw in self.sparse:
                k16, v16 = lw.attn.get_aiter_sparse_pa_kv_cache()
                p = dict(
                    w_qkv=lw.w_qkv, s_qkv=lw.s_qkv, g_in=lw.g_in, g_q=lw.g_q, g_k=lw.g_k, g_iq=lw.g_iq, g_ik=lw.g_ik,
                    w_o=lw.w_o, s_o=lw.s_o, g_post=lw.g_post, gate=lw.gate, bias=lw.bias, w13=lw.w13, s13=lw.s13,
                    w2=lw.w2, s2=lw.s2, k_cache=k16, v_cache=v16, index_cache=lw.index_cache.kv_cache,
                )
                rows.append([p[k].data_ptr() for k in LAYER_PTRS[:-1]] + [int(lw.index_topk)])
            self._ptab = torch.tensor(rows, dtype=torch.int64, device=self.dev)
        return self._ptab

    def _consts(self, n: int) -> dict:
        pr = LY.peer_regions()
        return dict(S=n, HALF=LY.StepLayout().half(), PHALF=pr["_half"], FFN_OFF=pr["ffn"], LOOP=LOOP, STAGES=STAGES,
                    num_warps=8, enable_fp_fusion=False)

    def _args(self, positions, slots, block_table, seq_lens, hidden, residual, li0=0):
        return (self._ptab_tensor(), len(self.sparse), self.cos_sin, positions, slots, block_table,
                block_table.stride(0), seq_lens, hidden, residual, self.ar_out, self.hm_out, self.scratch,
                self.peers.local, self.peers.addresses, self.epoch, self.errors, self.trace, self.rank,
                self.sm_scale, self.eps, self.route_scale, self.shared_weight, self.swiglu_limit, li0)

    def _compile(self, n: int) -> None:
        from .kernel import m3_step_kernel

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"Gluon M3: width {n} first reached inside a graph capture")
        dev = self.dev
        z64 = torch.zeros(n, dtype=torch.int64, device=dev)
        bt = torch.zeros(n, 1, dtype=torch.int32, device=dev)
        sl = torch.ones(n, dtype=torch.int32, device=dev)
        h = torch.zeros(n, LY.HIDDEN, dtype=torch.bfloat16, device=dev)
        cc = m3_step_kernel.warmup(*self._args(z64, z64, bt, sl, h, h), grid=(LY.BLOCKS,), **self._consts(n))
        cc._init_handles()
        regs, spills, lds = cc.n_regs, cc.n_spills, cc.metadata.shared
        logger.info("Gluon M3 width %d: %d VGPRs, %d spills, %d B LDS", n, regs, spills, lds)
        assert spills == 0 and lds <= 160 * 1024 and regs <= 256, (n, regs, spills, lds)
        torch.cuda.synchronize()
        if self.cpu_group is not None:
            dist.barrier(group=self.cpu_group)

    def launch(self, positions, slots, block_table, seq_lens, hidden, residual) -> None:
        from .kernel import m3_step_kernel

        n = hidden.shape[0]
        consts = self._consts(n)
        if LOOP:
            m3_step_kernel[(LY.BLOCKS,)](*self._args(positions, slots, block_table, seq_lens, hidden, residual),
                                         **consts)
        else:
            for li in range(len(self.sparse)):
                if FIRST_ERR:
                    self.errors[LY.BLOCKS].fill_(li)
                m3_step_kernel[(LY.BLOCKS,)](*self._args(positions, slots, block_table, seq_lens, hidden,
                                                         residual, li), **consts)

    def raise_on_errors(self) -> None:
        errs = self.errors[:LY.BLOCKS]
        bad = (errs != 0).nonzero().flatten().tolist()
        if bad:
            codes = errs[bad].tolist()
            hist = {c: codes.count(c) for c in sorted(set(codes))}
            raise RuntimeError(f"Gluon M3 rank {self.rank}: watchdog fired on CTAs {bad[:8]}..{bad[-4:]} "
                               f"({len(bad)} CTAs; code: count {hist})")


class GluonM3Runner(GluonCore, M3MonoRunner):
    """vLLM's MiniMax-M3 model on the Gluon step kernel (``VLLM_M3_MONO_BACKEND=gluon``)."""

    def __init__(self, model, vllm_config) -> None:
        from vllm.distributed import get_tp_group

        from vllm.models.minimax_m3.amd.mono.fw.runtime.consensus import bind_agreed

        self.group = get_tp_group()
        self.cpu_group = self.group.cpu_group
        bind_agreed(lambda: self._bind(model, vllm_config), self.cpu_group)
        cfg = model.config
        moe = model.layers[3].block_sparse_moe
        from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

        total_w, _ = rocm_aiter_moe.aiter_topK_meta_data
        self._gluon_setup(float(cfg.rms_norm_eps),
                          float(getattr(moe, "routed_scaling_factor", getattr(cfg, "routed_scaling_factor", 2.0))),
                          float(total_w[0, 4].item()), float(getattr(cfg, "swiglu_limit", 7.0)))

    def begin(self, positions, n):  # the layer-by-layer check mode is FlyDSL's
        raise MonoUnsupported("the Gluon backend runs whole steps only")

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor):
        n = hidden.shape[0]
        md = self.sparse_metadata()
        dec = md.decode
        _need(int(dec.decode_query_len) == 1, "Gluon M3: speculative verify rows")
        bt = dec.block_table
        _need(bt.stride(-1) == 1 and bt.dtype == torch.int32, "block table layout")
        self.launch(positions, md.slot_mapping, bt, dec.seq_lens, hidden.contiguous(), residual.contiguous())
        return self.ar_out[:n], self.hm_out[:n]


class GluonStandalone(GluonCore):
    """The step kernel on synthetic layers and a fixed step's metadata (microbenchmark)."""

    # kernel.py stamp slots (VLLM_M3_GLUON_TRACE=1)
    TL_NAMES = {0: "layer start", 1: "qkv end", 2: "split end", 11: "o merged", 3: "o end", 4: "router end",
                12: "shared ug end", 6: "routed", 7: "xn staged", 8: "routed ug end", 9: "mids staged",
                10: "down mfma end", 5: "moe end", 13: "split q ready", 14: "split own key in", 15: "split qk+softmax",
                16: "split pv", 17: "router polled a,h", 18: "router logits", 19: "router scores put",
                20: "qkv input polled", 21: "qkv x quantized", 22: "mids wait start", 23: "mids landed", 24: "tail k normed", 25: "tail cache stored", 26: "tail k inserted", 27: "tail V masked", 28: "tail v landed", 29: "tail v stored", 30: "tail v_own"}

    def __init__(self, layers, rank, cpu_group, dev, debug=False, timeline=False, eps=1e-6, route_scale=2.0,
                 shared_weight=1.0, swiglu_limit=7.0):
        self.sparse, self.rank, self.dev, self.cpu_group = layers, rank, dev, cpu_group
        self.cos_sin = layers[0].cos_sin
        self.debug = debug
        self.md = None
        self._gluon_setup(eps, route_scale, shared_weight, swiglu_limit, timeline)

    def prepare(self, n: int) -> bool:
        return self.k4.prepare(n)

    def forward(self, positions, hidden, residual):
        n = hidden.shape[0]
        md = self.md
        self.launch(positions, md.slot_mapping, md.decode.block_table, md.decode.seq_lens, hidden.contiguous(),
                    residual.contiguous())
        return self.ar_out[:n], self.hm_out[:n]

    def raise_on_given_up_waits(self, n):
        self.raise_on_errors()

    def close(self):
        self._finalizer()
