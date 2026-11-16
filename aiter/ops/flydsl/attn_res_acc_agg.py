"""SILOTIGER-1277: acc-cache aggregation consumer + 3-way AttnRes all-reduce router.

TP8 Kimi-K3 decode AttnRes fusion. Two decode regimes share ONE acc-cache producer
(``OneShotAllReduceAttnRes.precompute_bank_acc`` -> ``(acc, ms)``, Kevin Mi's
sglang#44988 bank_partial_hip.cuh port):

  * Low concurrency (T <= T_fused, default 8): the flydsl ONE-SHOT fused AR
    (``allreduce_attn_res(..., bank_acc=(acc, ms))``) wins 1.4-1.8x -- 1 kernel,
    acc-cache read folded in registers (no HBM bank touch).
  * High concurrency (T > T_fused): the one-shot AR loses to the production
    ``cross_device_reduce_2stage`` AR above T~10 (fixed-handshake vs per-byte
    crossover, iter015). There the router keeps the production AR but replaces the
    RAW ``attn_res_hip`` agg (reads all nvb bank rows, ~8.3us) with ``acc_agg``
    below: it folds ONLY the new prefix row against the cached (acc, m, sum),
    reading 3*H vs ~10*H -> ~2x cheaper (iter017). acc-cache is AR-agnostic, so the
    SAME producer output drives both branches.

Measured (gfx950, nvb=8, overlapped producer): whole TP8 decode band wins >=1.39x
vs ``cdr_2stage + raw attn_res_hip`` (iter017), crossover T~10.

Correctness: ``acc_agg`` is a bit-level drop-in for the production agg -- gated
18/18 max_ulp<=1.0 prefix_exact vs the REAL ``attn_res_hip`` on the actual
cdr_2stage AR output at TP8 (iter017/019). The fused branch is gated 72/72 by
test_attnres_tp2.py.

This module is ADDITIVE: it imports but never mutates the one-shot AR dispatch.
"""
import torch
from torch.utils.weak import WeakTensorKeyDictionary

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton always present on gfx950
    _HAVE_TRITON = False


# cw (and any fixed per-layer weight) is immutable across decode steps but was
# being re-cast to fp32+contiguous EVERY call -> ~4.6us (graph) / ~5.5us (eager)
# of pure .to()+.contiguous() dispatch floor on the LOAD-BEARING producer and
# acc_agg paths (iter025 measurement). Cache the conversion keyed weakly on the
# source weight tensor: hits every step (the layer weight object is reused),
# auto-evicts on GC, and the version counter guards the rare in-place update.
# The cached tensor is bit-identical to the previously-gated host conversion.
_FP32_CACHE = WeakTensorKeyDictionary()


def _as_fp32_cached(w):
    """Return a cached fp32, contiguous copy of a fixed per-layer weight ``w``.

    No-op pass-through if ``w`` is already fp32+contiguous. Bit-identical to
    ``w.to(torch.float32).contiguous()`` but computed once per (weight, version)
    instead of once per decode step."""
    if w is None:
        return None
    if w.dtype == torch.float32 and w.is_contiguous():
        return w
    entry = _FP32_CACHE.get(w)
    if entry is None or entry[0] != w._version:
        c = w.to(torch.float32).contiguous()
        _FP32_CACHE[w] = (w._version, c)
        return c
    return entry[1]


if _HAVE_TRITON:

    @triton.jit
    def _acc_agg_kernel(
        prefix_ptr, addend_ptr, acc_ptr, ms_ptr, cw_ptr, ow_ptr,
        out_ptr, pout_ptr,
        H: tl.constexpr, SCORE_EPS: tl.constexpr, OUT_EPS: tl.constexpr,
        HAS_OW: tl.constexpr, WRITE_PREFIX: tl.constexpr,
        MS_STRIDE: tl.constexpr, BLK: tl.constexpr,
    ):
        """One program per token. Mirrors OneShotAllReduceAttnRes acc-cache read +
        agg_reference. ``ms`` is read with a configurable row stride so the SAME
        producer tensor (T,3 [m,sum,accsq] from precompute_bank_acc, or T,2 [m,sum])
        drives both router branches -- only columns 0,1 (m_bank, sum_bank) are used
        here; accsq is for the fused out-norm, irrelevant to the standalone agg."""
        t = tl.program_id(0)
        h = tl.arange(0, BLK)
        hm = h < H
        base = t * H
        prefix = tl.load(prefix_ptr + base + h, mask=hm, other=0.0).to(tl.float32)
        addend = tl.load(addend_ptr + base + h, mask=hm, other=0.0).to(tl.float32)
        pv = (prefix + addend).to(tl.bfloat16).to(tl.float32)   # bf16-rounded prefix+addend
        if WRITE_PREFIX:
            tl.store(pout_ptr + base + h, pv.to(tl.bfloat16), mask=hm)
        cw = tl.load(cw_ptr + h, mask=hm, other=0.0).to(tl.float32)
        psum = tl.sum(pv * cw, axis=0)
        pss = tl.sum(pv * pv, axis=0)
        p_sc = psum * tl.rsqrt(pss * (1.0 / H) + SCORE_EPS)

        m_bank = tl.load(ms_ptr + t * MS_STRIDE + 0)
        sum_bank = tl.load(ms_ptr + t * MS_STRIDE + 1)
        m = tl.maximum(m_bank, p_sc)
        resc = tl.exp(m_bank - m)
        p_w = tl.exp(p_sc - m)
        inv = 1.0 / (sum_bank * resc + p_w)
        a = p_w * inv
        b = resc * inv

        acc = tl.load(acc_ptr + base + h, mask=hm, other=0.0).to(tl.float32)
        mix = a * pv + b * acc
        o = mix
        if HAS_OW:  # out RMSNorm is bundled with ow (matches agg_reference: no ow -> no norm)
            mss = tl.sum(mix * mix, axis=0)
            rnorm = tl.rsqrt(mss * (1.0 / H) + OUT_EPS)
            ow = tl.load(ow_ptr + h, mask=hm, other=0.0).to(tl.float32)
            o = mix * rnorm * ow
        tl.store(out_ptr + base + h, o.to(out_ptr.dtype.element_ty), mask=hm)

    @triton.jit
    def _bank_acc_online_kernel(
        bank_ptr, cw_ptr, acc_ptr, ms_ptr,
        H: tl.constexpr, NB: tl.constexpr, NVB: tl.constexpr, EPS: tl.constexpr,
        BLK: tl.constexpr,
    ):
        """Single-read online-softmax producer (iter006). Emits (T,3) ms
        [m, sum, accsq] to match precompute_bank_acc exactly (accsq = sum_h acc^2,
        iter010). Math identical to the torch precompute (bank max, not global),
        reassociated via the online rescale."""
        t = tl.program_id(0)
        h = tl.arange(0, BLK)
        hm = h < H
        cwv = tl.load(cw_ptr + h, mask=hm, other=0.0).to(tl.float32)
        bank_t = bank_ptr + t * (NB * H)
        m = float("-inf")
        s = 0.0
        acc = tl.zeros([BLK], tl.float32)
        for r in tl.static_range(NVB):
            v = tl.load(bank_t + r * H + h, mask=hm, other=0.0).to(tl.float32)
            dot = tl.sum(v * cwv, axis=0)
            sq = tl.sum(v * v, axis=0)
            s_r = dot * tl.rsqrt(sq * (1.0 / H) + EPS)
            m2 = tl.maximum(m, s_r)
            scale = tl.exp(m - m2)
            w = tl.exp(s_r - m2)
            acc = acc * scale + w * v
            s = s * scale + w
            m = m2
        tl.store(acc_ptr + t * H + h, acc, mask=hm)
        accsq = tl.sum(acc * acc, axis=0)
        tl.store(ms_ptr + t * 3 + 0, m)
        tl.store(ms_ptr + t * 3 + 1, s)
        tl.store(ms_ptr + t * 3 + 2, accsq)

    @triton.jit
    def _fused_producer_accagg_kernel(
        bank_ptr, prefix_ptr, addend_ptr, cw_ptr, ow_ptr, out_ptr, pout_ptr,
        H: tl.constexpr, NB: tl.constexpr, NVB: tl.constexpr,
        SCORE_EPS: tl.constexpr, OUT_EPS: tl.constexpr,
        HAS_OW: tl.constexpr, WRITE_PREFIX: tl.constexpr, BLK: tl.constexpr,
    ):
        """iter028: producer (_bank_acc_online_kernel) and consumer (_acc_agg_kernel)
        FUSED into one launch. Phase 1 online-softmax over nvb bank rows leaves
        (m, sum, acc[H]) in registers; phase 2 folds the prefix row straight from
        the register-resident acc -> out. Eliminates the acc[T,H] HBM round-trip
        AND one kernel dispatch (~1.9us, iter018) vs producer_online+acc_agg.
        1.27-1.32x on the producer+agg marginal cost (iter028, gated 120/120 <=1 ulp
        physical envelope). Use ONLY on the NON-overlapped high-T branch: the fused
        kernel runs same-step AFTER the AR, so the producer cannot be precomputed one
        layer ahead -- if the producer IS overlapped, acc_agg alone is cheaper."""
        t = tl.program_id(0)
        h = tl.arange(0, BLK)
        hm = h < H
        cwv = tl.load(cw_ptr + h, mask=hm, other=0.0).to(tl.float32)
        # Phase 1: producer online-softmax over bank rows 0..NVB-1 (acc in registers).
        bank_t = bank_ptr + t * (NB * H)
        m = float("-inf")
        s = 0.0
        acc = tl.zeros([BLK], tl.float32)
        for r in tl.static_range(NVB):
            v = tl.load(bank_t + r * H + h, mask=hm, other=0.0).to(tl.float32)
            dot = tl.sum(v * cwv, axis=0)
            sq = tl.sum(v * v, axis=0)
            s_r = dot * tl.rsqrt(sq * (1.0 / H) + SCORE_EPS)
            m2 = tl.maximum(m, s_r)
            scale = tl.exp(m - m2)
            w = tl.exp(s_r - m2)
            acc = acc * scale + w * v
            s = s * scale + w
            m = m2
        # Phase 2: consumer folds the prefix row (== _acc_agg_kernel from acc/m/s).
        base = t * H
        prefix = tl.load(prefix_ptr + base + h, mask=hm, other=0.0).to(tl.float32)
        addend = tl.load(addend_ptr + base + h, mask=hm, other=0.0).to(tl.float32)
        pv = (prefix + addend).to(tl.bfloat16).to(tl.float32)
        if WRITE_PREFIX:
            tl.store(pout_ptr + base + h, pv.to(tl.bfloat16), mask=hm)
        p_sc = tl.sum(pv * cwv, axis=0) * tl.rsqrt(tl.sum(pv * pv, axis=0) * (1.0 / H) + SCORE_EPS)
        mm = tl.maximum(m, p_sc)
        resc = tl.exp(m - mm)
        p_w = tl.exp(p_sc - mm)
        inv = 1.0 / (s * resc + p_w)
        mix = (p_w * inv) * pv + (resc * inv) * acc
        o = mix
        if HAS_OW:
            rnorm = tl.rsqrt(tl.sum(mix * mix, axis=0) * (1.0 / H) + OUT_EPS)
            ow = tl.load(ow_ptr + h, mask=hm, other=0.0).to(tl.float32)
            o = mix * rnorm * ow
        tl.store(out_ptr + base + h, o.to(out_ptr.dtype.element_ty), mask=hm)


def fused_producer_accagg(prefix, addend, bank, cw, ow, nvb, score_eps, out_eps,
                          *, out=None, prefix_out=None, write_prefix=True, nw=None):
    """Non-overlapped high-T branch: ONE kernel = producer_online + acc_agg (iter028).
    Reads the nvb bank rows + the prefix row, folds in registers -> out (+ prefix_out).
    1.27-1.32x vs precompute_bank_acc_online+acc_agg (saves the acc HBM round-trip and
    one dispatch). Bit-close (<=1 ulp physical envelope) to the two-kernel path, NOT
    bit-identical (fused reduction order differs). Select this ONLY when the producer
    is not overlapped one layer ahead.

    ``nw`` (num_warps): None -> nvb-aware optimum. iter037 measured this kernel vs the
    production ``attn_res_hip`` HIP agg (standalone, graph+event, H=7168): at nvb=8
    nw=8 is best (0.97-0.98x prod), at nvb=4 nw=8 is only 0.90x but **nw=4 reaches
    1.00-1.01x = parity/win** (fewer bank rows -> less per-WG work -> lower warp count
    gives better occupancy). The non-overlapped high-T router branch used the fixed
    nw=8 default and so ran ~3% below production at nvb=4 (iter037 e2e sweep). An
    nvb-aware default (nw=8 @nvb>=8, nw=4 below) clamps the non-overlapped floor to
    production parity at all nvb with no nvb=8 regression. runs/iter037_fpa_nw_nvb{4,8}
    _tp8.json. Pass an explicit int to override."""
    T, Hh = prefix.shape
    NB = bank.shape[1]
    if nw is None:
        nw = 8 if int(nvb) >= 8 else 4
    BLK = triton.next_power_of_2(Hh)
    if out is None:
        out = torch.empty_like(prefix)
    if prefix_out is None:
        prefix_out = torch.empty_like(prefix)
    cwf = _as_fp32_cached(cw)
    owf = ow if ow is not None else prefix
    _fused_producer_accagg_kernel[(T,)](
        bank, prefix, addend, cwf, owf, out, prefix_out,
        H=Hh, NB=NB, NVB=int(nvb), SCORE_EPS=float(score_eps), OUT_EPS=float(out_eps),
        HAS_OW=ow is not None, WRITE_PREFIX=write_prefix, BLK=BLK, num_warps=nw)
    return out, prefix_out


def precompute_bank_acc_online(bank, cw, nvb, score_eps, *, nw=4):
    """Triton single-read producer -> ``(acc (T,H fp32), ms (T,3 fp32))`` matching
    ``OneShotAllReduceAttnRes.precompute_bank_acc``. ~10us standalone on gfx950;
    SGLang overlaps it one layer ahead on the route-sort kernel's spare WGs.
    Drop-in for the torch precompute; use whichever is cheaper in the serving loop."""
    T, NB_, H_ = bank.shape
    BLK = triton.next_power_of_2(H_)
    acc = torch.empty((T, H_), dtype=torch.float32, device=bank.device)
    ms = torch.empty((T, 3), dtype=torch.float32, device=bank.device)
    cwf = _as_fp32_cached(cw)
    _bank_acc_online_kernel[(T,)](bank, cwf, acc, ms, H=H_, NB=NB_, NVB=int(nvb),
                                  EPS=float(score_eps), BLK=BLK, num_warps=nw)
    return acc, ms


def acc_agg(prefix, addend, acc, ms, cw, ow, score_eps, out_eps,
            *, out=None, prefix_out=None, write_prefix=True, nw=8):
    """acc-cache aggregation consumer (iter017, nw=8 tuned). Drop-in for the
    production ``attn_res_hip`` RAW agg: given the AR output ``addend``, the
    residual ``prefix``, and the producer's ``(acc, ms)``, folds ONLY the prefix
    row -> final out (+ optional prefix_out). ``ms`` may be (T,2) or (T,3); only
    columns 0,1 (m_bank, sum_bank) are read. ``ow is None`` -> no out-norm."""
    T, H = prefix.shape
    BLK = triton.next_power_of_2(H)
    if out is None:
        out = torch.empty_like(prefix)
    if prefix_out is None:
        prefix_out = torch.empty_like(prefix)
    ms_stride = int(ms.shape[1])
    cwf = _as_fp32_cached(cw)
    owf = ow if ow is not None else prefix  # dummy ptr when HAS_OW False
    _acc_agg_kernel[(T,)](
        prefix, addend, acc, ms, cwf, owf, out, prefix_out,
        H=H, SCORE_EPS=float(score_eps), OUT_EPS=float(out_eps),
        HAS_OW=ow is not None, WRITE_PREFIX=write_prefix,
        MS_STRIDE=ms_stride, BLK=BLK, num_warps=nw)
    return out, prefix_out


# Default one-shot -> cdr_2stage handoff. Fine crossover sweep (iter021,
# runs/iter021_tfused_fine_nvb{8,4}_tp8.json) pins the break-even at T8/T9 for
# BOTH nvb=8 and nvb=4 -- fused wins T<=8 (nvb8 1.43x, nvb4 1.18x), cdr+acc_agg
# wins T>=9 (nvb8 1.38x, nvb4 1.15x). nvb=4 has a structural cliff at T9 (split
# ladder caps at T<=8). So t_fused=8 is EXACTLY optimal for both nvb -- no
# per-nvb ladder needed. Override per deployment if the mix is known.
#
# ...BUT t_fused=8 is the OVERLAPPED crossover (producer precomputed one layer
# ahead -> ~free; iter021). On the realistic gfx950 path the producer is paid in
# full (streams net-neg iter024, bank rebuilt each pass / no reuse iter029), so the
# low-T fused one-shot branch (which runs the online producer on the critical path)
# crosses EARLIER. iter032 (runs/iter032_nonov_tfused_crossover_nvb{8,4}_tp8.json)
# pins the NON-overlapped crossover: the fused one-shot is the faster of the two
# self-contained options through T6 (nvb>=8) / T7 (nvb<8); from T7/T8 the cdr +
# fused_producer_accagg branch (flat ~0.97-0.99x prod) wins. Keeping t_fused=8 at
# bank_acc=None routes T7-8 to the losing fused path (nvb8 0.946x @T8 = -5.4%;
# nvb4 cliffs 0.78x @T9). So when NO overlapped bank_acc is supplied and the caller
# did not override t_fused, drop the cutoff to this nvb-aware non-overlap value.
DEFAULT_T_FUSED = 8


def _nonoverlap_t_fused(nvb):
    """Non-overlapped (bank_acc=None) fused-branch cutoff. fused one-shot wins
    through T6 (both nvb), cdr+fused_producer_accagg wins from T7.

    iter032 pinned nvb<8 at T7, but that measured the OLD fused_producer_accagg
    (nw fixed at 8). iter037 pt2 made fpa nvb-aware (nw=4 at nvb<8), which sped up
    pathB at nvb=4 so the crossover moved 7->6: iter038 (runs/iter038_nonov_tfused_
    crossover_nvb4_tp8.json) measures T7 nvb=4 pathA 0.981x (LOSS) vs pathB 1.004x,
    while T6 pathA 1.016x still wins. nvb>=8 unchanged at 6 (T7 pathA 0.978x loses).
    Both nvb now crossover at 6."""
    return 6


def attn_res_allreduce_router(
    ar,
    inp,
    prefix,
    bank,
    nvb,
    cw,
    ow,
    score_eps,
    out_eps,
    *,
    out=None,
    prefix_out=None,
    write_bank=False,
    bank_acc=None,
    t_fused=None,
    all_reduce_fn=None,
    fuse_producer_agg=True,
):
    """3-way AttnRes decode router (SILOTIGER-1277).

    ``ar``: a built ``OneShotAllReduceAttnRes`` instance (collectives set up).
    ``bank_acc``: optional precomputed ``(acc, ms)`` -- when the serving loop
    already ran the producer one layer ahead (overlapped, ~free) pass it in;
    otherwise it is computed here via ``ar.precompute_bank_acc`` (critical path).
    ``all_reduce_fn``: the production AR for the high-T branch; defaults to
    ``aiter.dist.communication_op.tensor_model_parallel_all_reduce`` (cdr_2stage).

    ``t_fused``: fused-branch cutoff. Default (``None``) resolves to the OVERLAPPED
    crossover ``DEFAULT_T_FUSED`` (8) when ``bank_acc`` is supplied (producer free),
    or the iter032 NON-overlapped nvb-aware crossover ``_nonoverlap_t_fused(nvb)``
    (6 at nvb>=8, 7 at nvb<8) when ``bank_acc is None`` (producer on the critical
    path, so the fused one-shot crosses earlier). Pass an explicit int to override.

    Returns ``(out, prefix_out)``. For T <= t_fused dispatches the one-shot fused
    AR (acc-cache); for T > t_fused runs ``all_reduce_fn(inp)`` then ``acc_agg``.
    """
    T = int(inp.shape[0])

    # Resolve the default cutoff (None sentinel): overlapped -> 8; non-overlapped
    # (bank_acc=None, producer paid in full) -> earlier nvb-aware crossover
    # (iter032). An explicit caller-supplied t_fused is honored verbatim.
    if t_fused is None:
        t_fused = DEFAULT_T_FUSED if bank_acc is not None else _nonoverlap_t_fused(nvb)

    # High-T branch with NO overlapped producer supplied (``bank_acc is None``):
    # skip the standalone producer entirely and fold producer+agg into ONE post-AR
    # kernel (iter028, 1.27-1.32x on the producer+agg marginal cost). This is the
    # DEFAULT (``fuse_producer_agg=True``) because the only alternative on the
    # non-overlapped path -- the online producer PLUS a separate ``acc_agg`` (two
    # kernels, an acc[T,H] fp32 HBM round-trip, an extra dispatch) -- is STRICTLY
    # DOMINATED: iter030 measured it at 0.75x vs production raw agg (a 33%
    # pessimization) while the fused kernel is ~0.98x (near-parity). The fused path
    # is the best self-contained non-overlapped option; the overlapped path (below,
    # bank_acc provided) reuses the ~free cached acc via a single acc_agg and is the
    # real headline win. Pass ``fuse_producer_agg=False`` only to force the legacy
    # two-kernel path (kept for audit/comparison -- it is never faster).
    if T > t_fused and fuse_producer_agg and bank_acc is None:
        if all_reduce_fn is None:
            from aiter.dist.communication_op import tensor_model_parallel_all_reduce
            all_reduce_fn = tensor_model_parallel_all_reduce
        addend = all_reduce_fn(inp)
        return fused_producer_accagg(
            prefix, addend, bank, cw, ow, nvb, score_eps, out_eps,
            out=out, prefix_out=prefix_out, write_prefix=True,
        )

    if bank_acc is None:
        # Fallback (producer NOT overlapped one layer ahead): use the single-read
        # online Triton producer, NOT the torch ``ar.precompute_bank_acc``. On the
        # non-overlapped critical path the torch producer is 50-84us vs ~9.5-12.5us
        # for the online kernel = 4.8-7.6x slower (iter025, runs/iter025_producer_
        # default_tp8.json), and the two are output-equivalent (acc diverges <=0.05
        # ulp; fp64-straddling, gated 48/48 end-to-end through BOTH router branches,
        # iter023). To force the torch reference instead, pass
        # ``bank_acc=ar.precompute_bank_acc(...)`` explicitly.
        bank_acc = precompute_bank_acc_online(bank, cw, nvb, score_eps)
    acc, ms = bank_acc

    if T <= t_fused:
        return ar.allreduce_attn_res(
            inp, prefix, bank, nvb, cw, ow, score_eps, out_eps,
            out=out, prefix_out=prefix_out, write_bank=write_bank, bank_acc=bank_acc,
        )

    if all_reduce_fn is None:
        from aiter.dist.communication_op import tensor_model_parallel_all_reduce
        all_reduce_fn = tensor_model_parallel_all_reduce
    addend = all_reduce_fn(inp)
    return acc_agg(
        prefix, addend, acc, ms, cw, ow, score_eps, out_eps,
        out=out, prefix_out=prefix_out, write_prefix=True,
    )
