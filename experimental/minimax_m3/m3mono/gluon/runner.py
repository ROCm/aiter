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

# VLLM_M3_GLUON_SRC=<dir>: the kernel package (layout / prims / kernel.py) taken from that directory
# instead of this one (parallel experiments on private copies of the source)
_SRC = os.environ.get("VLLM_M3_GLUON_SRC")
if _SRC:
    import importlib.util
    import sys

    _spec = importlib.util.spec_from_file_location("m3gluon_src", os.path.join(_SRC, "__init__.py"),
                                                   submodule_search_locations=[_SRC])
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["m3gluon_src"] = _pkg
    _spec.loader.exec_module(_pkg)
    import m3gluon_src.layout as LY

    def _kernel():
        from m3gluon_src.kernel import m3_step_kernel
        return m3_step_kernel
else:
    from . import layout as LY

    def _kernel():
        from .kernel import m3_step_kernel
        return m3_step_kernel

logger = logging.getLogger(__name__)

# one launch for every layer of a step (LOOP) or a launch a layer
LOOP = os.environ.get("VLLM_M3_GLUON_LOOP", "0") == "1"
NL = int(os.environ.get("VLLM_M3_GLUON_NL", "1"))  # layers a launch (statically unrolled; LOOP: all)
FIRST_ERR = os.environ.get("VLLM_M3_GLUON_FIRST_ERR", "0") == "1"
# stage mask (kernel.layer_body bits; debugging)
STAGES = int(os.environ.get("VLLM_M3_GLUON_STAGES", "511"))
# hybrid: AITER's fused MoE between launches (kernel.HYB)
HYB = os.environ.get("VLLM_M3_GLUON_HYBRID", "0") == "1"
MAX_AUX = 3  # EAGLE3 aux hidden-state layers the kernel can emit
# ahead-of-time binaries (aot.py) instead of the JIT: an image whose Triton cannot build the kernel
AOT_DIR = os.environ.get("VLLM_M3_GLUON_AOT_DIR", "")
# split hybrid: per layer launch A (FFN all-reduce .. qkv), vLLM's own attention (q/k norm, RoPE,
# KV / index insert, indexer, sparse attention: the image's code), launch B (o input .. top-k),
# AITER's MoE
SPLIT = HYB and os.environ.get("VLLM_M3_GLUON_SPLIT", "0") == "1"
SPLIT_DBG = os.environ.get("M3_SPLIT_DBG", "0") == "1"  # (diagnostics) layer 3 vs vLLM's modules, eager steps

# a layer's pointer-table entries (kernel.NPTR), in this order
LAYER_PTRS = (
    "w_qkv", "s_qkv", "g_in", "g_q", "g_k", "g_iq", "g_ik", "w_o", "s_o", "g_post", "gate", "bias", "w13",
    "s13", "w2", "s2", "k_cache", "v_cache", "index_cache", "index_topk",
)


class _Given(torch.nn.Module):
    """Stands in for qkv_proj: returns the given tensor (launch A's qkv)."""

    def __init__(self, t: torch.Tensor):
        super().__init__()
        self.t = t

    def forward(self, x):
        return self.t, None


class _Passthrough(torch.nn.Module):
    """Stands in for o_proj: the attention output itself (launch B takes it from here)."""

    def forward(self, x):
        return x, None


class _Widths:
    """``prepare(n)``: the width-n kernel compiled on every rank (the FlyDSL runner's
    ``k4`` interface, as the hook calls it)."""

    def __init__(self, runner):
        self.r = runner
        self.ready = set()

    def prepare(self, n: int) -> bool:
        if n in self.ready:
            return True
        if not 1 <= n <= (LY.SUPPORTED_S if HYB else LY.FUSED_MAX_S):
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
        # every mailbox word starts as (1, tag 0): stale for any tag and never 0, which the
        # lean wait reads as an unused (masked) element
        self.scratch.view(torch.int64).fill_(int(os.environ.get("M3_SCR_FILL", "1")))
        self.peers.bytes.view(torch.int64).fill_(int(os.environ.get("M3_PEER_FILL", "1")))
        for par in range(2):  # the dense hand-offs' slots start empty (DSENT)
            for off, nb in LY.dense_spans():
                self.peers.bytes[par * LY.peer_half() + off:][:nb].fill_(255)
        self.epoch = torch.zeros(LY.EPOCH_WORDS, dtype=torch.int32, device=dev)
        # [BLOCKS] codes, then (FIRST_ERR debug) the layer being launched
        self.errors = torch.zeros(2 * LY.BLOCKS, dtype=torch.int32, device=dev)
        n = len(self.sparse)
        self.trace = torch.zeros(n * LY.BLOCKS * 66, dtype=torch.int64, device=dev)
        if timeline:
            self.tl_buf = self.trace[: n * LY.BLOCKS * 64].view(n, LY.BLOCKS, 64)
            # [layer][CTA]: (HW_ID, XCC_ID) of its last launch
            self.hw_buf = self.trace[n * LY.BLOCKS * 64:].view(n, LY.BLOCKS, 2)
        bf16 = torch.bfloat16
        # [slot][rows][hidden]: slot 0 the step's output, slots 1 .. the EAGLE3 aux layers'
        self.ar_out = torch.zeros(1 + MAX_AUX, LY.MAX_S, LY.HIDDEN, dtype=bf16, device=dev)
        self.hm_out = torch.zeros(1 + MAX_AUX, LY.MAX_S, LY.HIDDEN, dtype=bf16, device=dev)
        self.ql = getattr(self, "ql", 1)  # rows a request (speculative verify: 1 + draft tokens)
        self.aux_sparse = getattr(self, "aux_sparse", ())  # sparse layer index of each aux slot 1 ..
        # hybrid: AITER's MoE input (xn, top-k ids / weights) and output (this rank's partial)
        self.moe_x = torch.zeros(LY.MAX_S, LY.HIDDEN, dtype=bf16, device=dev)
        self.moe_ids = torch.zeros(LY.MAX_S, LY.TOP_K + 1, dtype=torch.int32, device=dev)
        self.moe_w = torch.zeros(LY.MAX_S, LY.TOP_K + 1, dtype=torch.float32, device=dev)
        self.moe_out = torch.zeros(LY.MAX_S, LY.HIDDEN, dtype=bf16, device=dev)
        # split hybrid: launch A's qkv in vLLM's layout (the attention's input)
        self.qkv_buf = torch.zeros(LY.MAX_S, LY.QKV_ROWS, dtype=bf16, device=dev)
        self._aiter_w = None
        self._fenced = set()  # widths whose first step ran fenced
        self._ptab = None
        self._ptab_key = None
        self.k4 = _Widths(self)
        if os.environ.get("M3_PRINT_PTRS"):
            print(f"rank {self.rank} ptrs scratch {self.scratch.data_ptr():#x}+{self.scratch.numel():#x} "
                  f"peers {self.peers.bytes.data_ptr():#x}+{self.peers.bytes.numel():#x} epoch {self.epoch.data_ptr():#x} "
                  f"ar_out {self.ar_out.data_ptr():#x} hm_out {self.hm_out.data_ptr():#x}", flush=True)

    def _cache_key(self) -> int:
        return self.sparse[0].attn.get_aiter_sparse_pa_kv_cache()[0].data_ptr()

    def _ptab_tensor(self) -> torch.Tensor:
        """Per layer the pointer table. Widths compile before vLLM allocates the KV cache, so
        each eager step re-checks the cache pointers; the table is rewritten in place (graphs
        captured later keep its address)."""
        key = self._cache_key() if self._ptab is None or not torch.cuda.is_current_stream_capturing() else self._ptab_key
        if self._ptab is None or key != self._ptab_key:
            rows = []
            for si, lw in enumerate(self.sparse):
                k16, v16 = lw.attn.get_aiter_sparse_pa_kv_cache()
                p = dict(
                    w_qkv=lw.w_qkv, s_qkv=lw.s_qkv, g_in=lw.g_in, g_q=lw.g_q, g_k=lw.g_k, g_iq=lw.g_iq, g_ik=lw.g_ik,
                    w_o=lw.w_o, s_o=lw.s_o, g_post=lw.g_post, gate=lw.gate, bias=lw.bias, w13=lw.w13, s13=lw.s13,
                    w2=lw.w2, s2=lw.s2, k_cache=k16, v_cache=v16, index_cache=lw.index_cache.kv_cache,
                )
                aux = self.aux_sparse.index(si) + 1 if si in self.aux_sparse else 0
                rows.append([p[k].data_ptr() for k in LAYER_PTRS[:-1]] + [int(lw.index_topk) | (aux << 8)])
            t = torch.tensor(rows, dtype=torch.int64, device=self.dev)
            if self._ptab is None:
                self._ptab = t
            else:
                self._ptab.copy_(t)
            self._ptab_key = key
        return self._ptab

    def _consts(self, n: int) -> dict:
        pr = LY.peer_regions()
        return dict(S=n, HALF=LY.StepLayout().half(), PHALF=pr["_half"], FFN_OFF=pr["ffn"], LOOP=LOOP, STAGES=STAGES,
                    QL=self.ql, NL=NL, num_warps=8, enable_fp_fusion=False)

    def _args(self, positions, slots, block_table, seq_lens, hidden, residual, li0=0, attn_in=None):
        if getattr(LY, "KARG_V3", False):
            # the prologue's operands and the layer's qkv weights first (preloaded kernarg SGPRs)
            lw = self.sparse[min(li0, len(self.sparse) - 1)]
            return (self._ptab_tensor(), len(self.sparse), li0, self.epoch, self.scratch, lw.w_qkv, lw.s_qkv,
                    self.cos_sin, positions, slots, block_table, block_table.stride(0), seq_lens, hidden, residual,
                    self.ar_out, self.hm_out, self.peers.local, self.peers.addresses, self.errors, self.trace,
                    self.rank, self.sm_scale, self.eps, self.route_scale, self.shared_weight, self.swiglu_limit,
                    self.moe_x, self.moe_ids, self.moe_w, self.moe_out, self.qkv_buf,
                    self.moe_out if attn_in is None else attn_in)
        if getattr(LY, "KARG_V2", False):
            # the prologue's operands first: they sit in the preloaded kernarg SGPRs
            return (self._ptab_tensor(), len(self.sparse), li0, self.epoch, self.scratch, self.errors, self.trace,
                    self.rank, self.cos_sin, positions, slots, block_table, block_table.stride(0), seq_lens, hidden,
                    residual, self.ar_out, self.hm_out, self.peers.local, self.peers.addresses, self.sm_scale,
                    self.eps, self.route_scale, self.shared_weight, self.swiglu_limit, self.moe_x, self.moe_ids,
                    self.moe_w, self.moe_out, self.qkv_buf, self.moe_out if attn_in is None else attn_in)
        return (self._ptab_tensor(), len(self.sparse), self.cos_sin, positions, slots, block_table,
                block_table.stride(0), seq_lens, hidden, residual, self.ar_out, self.hm_out, self.scratch,
                self.peers.local, self.peers.addresses, self.epoch, self.errors, self.trace, self.rank,
                self.sm_scale, self.eps, self.route_scale, self.shared_weight, self.swiglu_limit, li0, self.moe_x,
                self.moe_ids, self.moe_w, self.moe_out, self.qkv_buf, self.moe_out if attn_in is None else attn_in)

    def _compile(self, n: int) -> None:
        m3_step_kernel = _kernel()

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"Gluon M3: width {n} first reached inside a graph capture")
        dev = self.dev
        z64 = torch.zeros(n, dtype=torch.int64, device=dev)
        bt = torch.zeros(n, 1, dtype=torch.int32, device=dev)
        sl = torch.ones(n, dtype=torch.int32, device=dev)
        h = torch.zeros(n, LY.HIDDEN, dtype=torch.bfloat16, device=dev)
        if AOT_DIR:
            from .aot import AotKernel

            if not hasattr(self, "_aot"):
                self._aot = {}
            self._aot[n] = AotKernel(AOT_DIR, n, self.ql, self._consts(n))
            cc = None
        elif SPLIT:
            assert not AOT_DIR, "the split hybrid JIT-compiles (no AOT binaries)"
            for ph in (1, 2):
                cc = m3_step_kernel.warmup(*self._args(z64, z64, bt, sl, h, h), grid=(LY.BLOCKS,),
                                           **self._consts(n), PHASE=ph)
                cc._init_handles()
                logger.info("Gluon M3 width %d phase %d: %d VGPRs, %d spills, %d B LDS", n, ph, cc.n_regs,
                            cc.n_spills, cc.metadata.shared)
        else:
            cc = m3_step_kernel.warmup(*self._args(z64, z64, bt, sl, h, h), grid=(LY.BLOCKS,), **self._consts(n))
            cc._init_handles()
        if HYB:
            # AITER's MoE kernels for this width compiled and loaded now: a lazy code-object
            # load later (under rocprofiler, at least) waits on device state while peers'
            # persistent kernels spin on this rank -- a cross-rank deadlock
            self.moe_ids[:n].copy_(torch.arange(n * (LY.TOP_K + 1), device=dev).view(n, -1) % (LY.N_ROUTED + 1))
            self.moe_w[:n].fill_(0.25)
            self._moe(0, n)
            torch.cuda.synchronize()
        if cc is None:
            logger.info("Gluon M3 width %d: AOT binary %s (Triton %s), %d B LDS", n, AOT_DIR,
                        self._aot[n].meta.get("triton"), self._aot[n].shared)
        else:
            regs, spills, lds = cc.n_regs, cc.n_spills, cc.metadata.shared
            logger.info("Gluon M3 width %d: %d VGPRs, %d spills, %d B LDS", n, regs, spills, lds)
            assert spills == 0 and lds <= 160 * 1024 and regs <= 256, (n, regs, spills, lds)
        torch.cuda.synchronize()
        if self.cpu_group is not None:
            dist.barrier(group=self.cpu_group)

    def _aiter_weights(self):
        """Per layer (w13, w2, s13, s2) as vLLM hands them to aiter.fused_moe (fp4x2 views
        flagged preshuffled: the same bytes the kernel reads)."""
        if self._aiter_w is None:
            def fp4(t):
                v = t if t.dtype == torch.float4_e2m1fn_x2 else t.view(torch.float4_e2m1fn_x2)
                v.is_shuffled = True
                return v

            self._aiter_w = [(fp4(lw.w13), fp4(lw.w2), lw.s13.view(torch.uint8), lw.s2.view(torch.uint8))
                             for lw in self.sparse]
        return self._aiter_w

    def _moe(self, li: int, n: int) -> None:
        """Layer li's MoE: AITER's fused_moe (sorting, up / gate + SwiGLU, down), the
        baseline's call."""
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe

        tpl = getattr(self, "_moe_tpl", None)
        if tpl is not None:
            self._moe_replay(tpl, li, n)
            return
        w13, w2, s13, s2 = self._aiter_weights()[li]
        if os.environ.get("M3_HYB_QX") == "1":  # diagnostics: xn rounded as the fused MoE's MXFP8
            x = self.moe_x[:n].float().view(n, -1, 32)
            amax = x.abs().amax(-1, keepdim=True)
            sc = torch.exp2(torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=2.0 ** -126))))
            self.moe_x[:n] = ((x / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc).view(n, -1).bfloat16()
        if os.environ.get("M3_HYB_DUMP") and li == 0 and self.rank == 0 and not torch.cuda.is_current_stream_capturing():
            torch.save({"x": self.moe_x[:n].cpu(), "ids": self.moe_ids[:n].cpu(), "w": self.moe_w[:n].cpu()},
                       os.environ["M3_HYB_DUMP"])
        fused_moe(self.moe_x[:n], w13, w2, self.moe_w[:n], self.moe_ids[:n], None, ActivationType.Swiglu,
                  QuantType.per_1x32, False, s13, s2, None, None, dtype=torch.bfloat16, hidden_pad=0,
                  intermediate_pad=0, bias1=None, bias2=None, moe_sorting_dispatch_policy=0,
                  swiglu_limit=self.swiglu_limit, output=self.moe_out[:n])

    def _fence(self) -> None:
        """Device idle on every rank (CPU barrier over the TP group)."""
        torch.cuda.synchronize()
        if self.cpu_group is not None:
            dist.barrier(group=self.cpu_group)

    def _dbg_sync(self, what: str) -> None:
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"rank {self.rank}: fault at/before {what}: {e}", flush=True)
            raise
        errs = self.errors[:LY.BLOCKS]
        if bool((errs != 0).any()):
            print(f"rank {self.rank}: watchdog after {what}: {errs.nonzero().flatten()[:8].tolist()}", flush=True)

    def launch(self, positions, slots, block_table, seq_lens, hidden, residual) -> None:
        m3_step_kernel = _kernel()

        n = hidden.shape[0]
        consts = self._consts(n)
        if HYB:
            nl = len(self.sparse)
            # a width's first (eager) step fenced on every rank: whatever else loads lazily
            # around it (vLLM's own kernels) never overlaps a persistent kernel in flight
            fence = n not in self._fenced and not torch.cuda.is_current_stream_capturing()
            if fence:
                self._fence()
                if os.environ.get("M3_HYB_SYNC") == "1":
                    def rng(t):
                        return f"{tuple(t.shape)} {t.dtype} [{int(t.min())}, {int(t.max())}]" if t.numel() else "empty"
                    print(f"rank {self.rank} width {n}: positions {rng(positions)} slots {rng(slots)} "
                          f"block_table {rng(block_table)} stride {block_table.stride()} seq_lens {rng(seq_lens)} "
                          f"{seq_lens[:16].tolist()} pos {positions[:8].tolist()}", flush=True)
            dbg =os.environ.get("M3_HYB_SYNC") == "1" and not torch.cuda.is_current_stream_capturing()
            if SPLIT:
                for li in range(nl + 1):
                    a = self._args(positions, slots, block_table, seq_lens, hidden, residual, li)
                    m3_step_kernel[(LY.BLOCKS,)](*a, **consts, PHASE=1)
                    if dbg:
                        self._dbg_sync(f"m3_step A {li}")
                    if li == nl:
                        break
                    att = self._attention(li, positions, n)
                    if dbg:
                        self._dbg_sync(f"vLLM attention {li}")
                    a = self._args(positions, slots, block_table, seq_lens, hidden, residual, li, att)
                    m3_step_kernel[(LY.BLOCKS,)](*a, **consts, PHASE=2)
                    if li == 0 and SPLIT_DBG and not torch.cuda.is_current_stream_capturing():
                        self._split_debug(hidden, residual, att, n)
                    if dbg:
                        self._dbg_sync(f"m3_step B {li}")
                    self._moe(li, n)
                    if dbg:
                        self._dbg_sync(f"aiter moe {li}")
                if fence:
                    self._fence()
                    self._fenced.add(n)
                return
            aot = self._aot[n] if AOT_DIR else None
            for li in range(nl + 1):
                if aot is not None:
                    aot(LY.BLOCKS, self._args(positions, slots, block_table, seq_lens, hidden, residual, li))
                else:
                    m3_step_kernel[(LY.BLOCKS,)](*self._args(positions, slots, block_table, seq_lens, hidden,
                                                             residual, li), **consts)
                if dbg:
                    self._dbg_sync(f"m3_step launch {li}")
                if li < nl:
                    self._moe(li, n)
                    if dbg:
                        self._dbg_sync(f"aiter moe {li}")
            if fence:
                self._fence()
                self._fenced.add(n)
            return
        if LOOP:
            m3_step_kernel[(LY.BLOCKS,)](*self._args(positions, slots, block_table, seq_lens, hidden, residual),
                                         **consts)
        else:
            for li in range(0, len(self.sparse), NL):
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
        spec = getattr(vllm_config, "speculative_config", None)
        self.ql = 1 + int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec is not None else 1
        # EAGLE3: aux index i is layer i - 1's output; the sparse ones (layer >= 3) come from
        # the kernel (slot j + 1 for the j-th), the dense ones from the hook's loop
        aux = sorted(getattr(model, "aux_hidden_state_layers", ()) or ())
        self.aux_sparse = tuple(i - 1 - 3 for i in aux if i - 1 >= 3)
        _need(len(self.aux_sparse) <= MAX_AUX, f"{len(self.aux_sparse)} aux layers")
        logger.info("Gluon M3: %d rows a request, aux sparse layers %s, hybrid %s", self.ql, self.aux_sparse, HYB)
        self._gluon_setup(float(cfg.rms_norm_eps),
                          float(getattr(moe, "routed_scaling_factor", getattr(cfg, "routed_scaling_factor", 2.0))),
                          float(total_w[0, 4].item()), float(getattr(cfg, "swiglu_limit", 7.0)))
        if HYB and os.environ.get("VLLM_M3_GLUON_MOE_CAPTURE", "1") == "1":
            self._capture_moe(model)

    # ---- the MoE call between launches, captured from this vLLM's own MoE layer -------------------
    # (VLLM_M3_GLUON_MOE_CAPTURE=1, default): sparse layer 0's MoE runs once on a dummy input with
    # aiter.fused_moe spied; every later call repeats that call (activation, quant type, gate mode,
    # limit, biases, weight views: whatever this vLLM / AITER pass) with the kernel's xn and top-k,
    # each layer's tensors found under the same parameter names
    def _capture_moe(self, model) -> None:
        import aiter.fused_moe as afm

        experts0 = model.layers[3].block_sparse_moe.experts
        named = dict(experts0.named_parameters(recurse=True))
        named.update(dict(experts0.named_buffers(recurse=True)))
        rec = {}
        orig = afm.fused_moe

        def spy(*a, **k):
            if not rec:
                rec["a"], rec["k"] = a, k
            return orig(*a, **k)

        afm.fused_moe = spy
        try:
            x = torch.randn(self.ql, LY.HIDDEN, dtype=torch.bfloat16, device=self.dev) * 0.1
            with torch.no_grad():
                model.layers[3].block_sparse_moe(x)
            torch.cuda.synchronize()
        finally:
            afm.fused_moe = orig
        _need(bool(rec), "Gluon M3: the MoE layer made no aiter.fused_moe call")
        import inspect
        params = list(inspect.signature(orig).parameters)

        def alias(t):
            for name, p in named.items():
                if p.untyped_storage().data_ptr() == t.untyped_storage().data_ptr():
                    return name
            return None

        slots = []  # (key: int position or kwarg name, kind, payload)
        items = list(enumerate(rec["a"])) + list(rec["k"].items())
        for key, v in items:
            pname = params[key] if isinstance(key, int) else key
            if pname == "hidden_states":
                slots.append((key, "x", None))
            elif pname == "topk_weight":
                _need(v.dtype == torch.float32 and v.shape[-1] == LY.TOP_K + 1, f"topk_weight {v.dtype} {tuple(v.shape)}")
                slots.append((key, "w", None))
            elif pname == "topk_ids":
                _need(v.dtype == torch.int32 and v.shape[-1] == LY.TOP_K + 1, f"topk_ids {v.dtype} {tuple(v.shape)}")
                slots.append((key, "ids", None))
            elif isinstance(v, torch.Tensor):
                name = alias(v)
                _need(name is not None, f"Gluon M3: fused_moe argument {pname} is not an experts parameter")
                slots.append((key, "param", (name, v.dtype, tuple(v.shape), tuple(v.stride()), v.storage_offset(),
                                             getattr(v, "is_shuffled", None))))
            else:
                slots.append((key, "value", v))
        self._moe_out_kw = "output" in params
        desc = {(params[k] if isinstance(k, int) else k): (p[0] if kind == "param" else kind if kind != "value" else p)
                for k, kind, p in slots}
        logger.info("Gluon M3: captured the MoE call: %s", desc)
        layers = model.layers[3:3 + len(self.sparse)]
        per_layer = []
        for layer in layers:
            ex = layer.block_sparse_moe.experts
            nm = dict(ex.named_parameters(recurse=True))
            nm.update(dict(ex.named_buffers(recurse=True)))
            ts = {}
            for key, kind, p in slots:
                if kind != "param":
                    continue
                name, dt, shape, stride, off, shuf = p
                t = nm[name].detach()
                if t.dtype != dt:
                    t = t.view(dt)
                t = t.as_strided(shape, stride, off)
                if shuf is not None:
                    t.is_shuffled = shuf
                ts[key] = t
            per_layer.append(ts)
        self._moe_tpl = (slots, per_layer)

    def _moe_replay(self, tpl, li: int, n: int) -> None:
        import aiter.fused_moe as afm

        slots, per_layer = tpl
        args, kw = [], {}
        for key, kind, p in slots:
            v = (self.moe_x[:n] if kind == "x" else self.moe_w[:n] if kind == "w" else self.moe_ids[:n]
                 if kind == "ids" else per_layer[li][key] if kind == "param" else p)
            if isinstance(key, int):
                args.append(v)
            else:
                kw[key] = v
        if self._moe_out_kw:
            afm.fused_moe(*args, **kw, output=self.moe_out[:n])
        else:
            self.moe_out[:n].copy_(afm.fused_moe(*args, **kw))

    def _attention(self, li: int, positions: torch.Tensor, n: int) -> torch.Tensor:
        """Sparse layer li's attention exactly as vLLM's MiniMaxM3SparseAttention.forward runs it,
        between its qkv_proj (here: launch A's qkv) and its o_proj (here: launch B): the module's
        own forward with those two projections stood in for."""
        attn = self.model.layers[3 + li].self_attn
        mods = attn._modules
        qkv_proj, o_proj = mods["qkv_proj"], mods["o_proj"]
        mods["qkv_proj"] = _Given(self.qkv_buf[:n])
        mods["o_proj"] = _Passthrough()
        try:
            out = attn.forward(positions, self.qkv_buf[:n])
        finally:
            mods["qkv_proj"], mods["o_proj"] = qkv_proj, o_proj
        if out.shape[-1] != LY.O_K:  # (vLLM's memory-profiling run returns zeros [n, hidden])
            out = out.new_zeros(n, LY.O_K)
        return out.contiguous()

    def _split_debug(self, hidden, residual, att, n) -> None:
        """Layer 3 of an eager split step against vLLM's own modules on the same inputs: qkv,
        and (on our attention output) o_proj + residual + post-attention norm."""
        torch.cuda.synchronize()
        layer = self.model.layers[3]

        def nr(a, b):
            a, b = a.float(), b.float()
            return float((a - b).norm() / b.norm().clamp_min(1e-30))

        with torch.no_grad():
            xn, h = layer.input_layernorm(hidden.clone(), residual.clone())
            qkv_ref, _ = layer.self_attn.qkv_proj(xn)
            from vllm.models.minimax_m3.amd import model as m3model

            o_ref, _ = layer.self_attn.o_proj(att)  # (a rank's partial: the norm below all-reduces)
            xn2, h2 = m3model.fused_allreduce_gemma_rms_norm(o_ref, h.clone(), layer.post_attention_layernorm)
        half = LY.StepLayout().half()
        hm = self.scratch[0 * half + LY.region("hm"):].view(torch.int32)[: n * LY.HIDDEN // 2]
        hm = hm.view(torch.bfloat16).view(n, LY.HIDDEN)
        q = self.qkv_buf[:n]
        msg = (f"split dbg n={n}: qkv {nr(q, qkv_ref):.3e} (q {nr(q[:, :2048], qkv_ref[:, :2048]):.3e} "
               f"k {nr(q[:, 2048:2176], qkv_ref[:, 2048:2176]):.3e} v {nr(q[:, 2176:2304], qkv_ref[:, 2176:2304]):.3e} "
               f"iq {nr(q[:, 2304:2432], qkv_ref[:, 2304:2432]):.3e} ik {nr(q[:, 2432:], qkv_ref[:, 2432:]):.3e}) "
               f"att rms {att.float().pow(2).mean().sqrt().item():.3e}; h_mid {nr(hm, h2):.3e}; moe x {nr(self.moe_x[:n], xn2):.3e}")
        if self.rank == 0:
            logger.info(msg)

    def check_caches(self) -> None:
        if not SPLIT:  # the split hybrid never reads the KV / index caches itself
            super().check_caches()

    def begin(self, positions, n):  # the layer-by-layer check mode is FlyDSL's
        raise MonoUnsupported("the Gluon backend runs whole steps only")

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor):
        n = hidden.shape[0]
        md = self.sparse_metadata()
        dec = md.decode
        _need(int(dec.decode_query_len) == self.ql, f"Gluon M3: {dec.decode_query_len} rows a request")
        bt = dec.block_table
        _need(bt.stride(-1) == 1 and bt.dtype == torch.int32, "block table layout")
        self.launch(positions, md.slot_mapping, bt, dec.seq_lens, hidden.contiguous(), residual.contiguous())
        return self.ar_out[0, :n], self.hm_out[0, :n]

    def aux_states(self, n: int) -> list:
        """EAGLE3's aux hidden states of the sparse aux layers (hidden + residual), in order."""
        return [self.ar_out[j, :n] + self.hm_out[j, :n] for j in range(1, 1 + len(self.aux_sparse))]

    def max_tokens(self) -> int:
        return LY.SUPPORTED_S if HYB else LY.FUSED_MAX_S


class GluonStandalone(GluonCore):
    """The step kernel on synthetic layers and a fixed step's metadata (microbenchmark)."""

    # kernel.py stamp slots (VLLM_M3_GLUON_TRACE=1)
    TL_NAMES = {0: "layer start", 1: "qkv end", 2: "split end", 11: "o merged", 3: "o end", 4: "router end",
                12: "shared ug end", 6: "routed", 7: "xn staged", 8: "routed ug end", 9: "mids staged",
                10: "down mfma end", 5: "moe end", 13: "split q ready", 14: "split own key in", 15: "split qk+softmax",
                16: "split pv", 17: "router polled a,h", 18: "router logits", 19: "router scores put",
                20: "qkv input polled", 21: "qkv x quantized", 22: "mids wait start", 23: "mids landed", 24: "o reduced", 25: "o puts issued", 26: "tail k inserted", 27: "tail V masked", 28: "tail v landed", 29: "o AR landed", 30: "route top-4 done", 31: "o peeks issued", 32: "o m,l landed", 33: "o parts landed", 34: "own k landed", 35: "own v landed", 36: "rank: floor", 37: "rank: compacted", 38: "ffn puts issued", 39: "ffn partials landed", 40: "route wait start", 41: "route scores landed", 42: "router peeks issued", 43: "router a,h part A landed", 44: "index scores done", 45: "selection done", 46: "sel: share scores landed", 47: "sel: share ranked", 48: "sel: candidates landed", 49: "ug half 0 / norm task start", 50: "ug half 1 / norm input landed", 51: "ug half 2 / norm x8s put", 52: "ug half 3 / router xn stored", 53: "ug half 4", 54: "ug half 5", 55: "ug half 6", 56: "ug half 7", 57: "ug pair published", 58: "down experts", 59: "down first loads", 60: "first weights out", 61: "xn stored", 62: "before first weights", 63: "o inputs staged"}

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
        return self.ar_out[0, :n], self.hm_out[0, :n]

    def raise_on_given_up_waits(self, n):
        self.raise_on_errors()

    def close(self):
        self._finalizer()
