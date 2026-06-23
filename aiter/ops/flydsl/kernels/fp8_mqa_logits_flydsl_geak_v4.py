"""FlyDSL translation of the FP8 MQA-logits Triton kernel (MFMA path).

logits[i, j] = sum_h relu( kv_scales[j] * sum_d Q[i,h,d] * KV[j,d] ) * weights[i,h]
for j in [cu_starts[i], cu_ends[i]); else -inf.

Implemented with mfma_f32_16x16x32_fp8_fp8. One wave (64 lanes) per
(query-row, kv-column-block) block; columns split across NCB blocks in grid.y.

mfma_f32_16x16x32_fp8_fp8 lane layout (CDNA3):
  A[16][32]: lane l -> A[row=l%16, k=(l//16)*8 .. +8]   (8 packed fp8 = i64)
  B[32][16]: lane l -> B[k=(l//16)*8 .. +8, col=l%16]   (8 packed fp8 = i64)
  C[16][16]: f32x4 per lane -> C[row=(l//16)*4+{0..3}, col=l%16]
A = Q[head, dim], B[k,n] = KV[kvcol_n, k] (KV is [kvcol, dim] contiguous).

Note on fp8 dtype: the gfx942 fp8 MFMA interprets operands as e4m3 *fnuz*.
The vLLM caller on gfx942 produces fp8 via current_platform.fp8_dtype() which
is torch.float8_e4m3fnuz, so the bytes line up. Feeding e4m3fn bytes here
would mis-interpret -0.0 (0x80) as NaN and corrupt the output.

Shape handling: the kernel's per-block tiling is fixed (NUM_HEADS=64,
HEAD_DIM=128, columns staged 16 wide), but the launch is parametric:
  * S_K (number of kv columns) is a build-time constant -> grid.y and the
    per-row output stride. Kernels are cached per S_K (rounded up to a
    multiple of COLS_PER_BLOCK = 128).
  * M (number of query rows) is a runtime scalar -> grid.x, so a single
    compiled kernel serves any row count for a given S_K.
"""

import torch
import torch.nn as nn

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl._mlir import ir


def _iv_to_i32(idx_val):
    """Cast an scf.for induction variable (index type) to fx.Int32.

    The MLIR i32 type must be built inside an active MLIR context (i.e. during
    kernel tracing), so it is resolved lazily here rather than at import.
    """
    i32_ty = ir.IntegerType.get_signless(32)
    return fx.Int32(arith.index_cast(i32_ty, fx.arith._to_raw(idx_val)))


# Each wave stages into and reads from its own private LDS slab, so a
# workgroup barrier is not required for correctness (intra-wave LDS
# write->read is ordered by the compiler's lds waitcnt). Kept as a toggle so
# the safe behaviour can be restored if a future layout shares slabs.
import os as _os
_USE_LDS_BARRIER = _os.environ.get("FLYDSL_MQA_LDS_BARRIER", "0") == "1"
# KV-staging strategy:
#   False (DEFAULT, v4): skip LDS entirely and read each lane's column KV
#     directly from global into the B fragment (strided global load, no LDS
#     round-trip / no occupancy LDS cap). The B fragment is held in registers and
#     reused across the ROWS rows, so row-tiling reuse is kept. Borrowed from
#     GEAK rank1's operand-reuse-without-LDS; combined with our causal early-exit
#     + fn->fnuz convert it is 22-39% faster than the LDS path and ~2.8x vs stock
#     (fnuz/fnuz), while staying correct on the live fn/fnuz combo.
#   True (v3): coalesced global->LDS stage, then strided LDS read. Kept as a
#     fallback (set FLYDSL_MQA_LDS_STAGE=1).
USE_LDS_STAGING = _os.environ.get("FLYDSL_MQA_LDS_STAGE", "0") == "1"

_FNUZ = torch.float8_e4m3fnuz
_DTYPE_LOGGED = False

NUM_HEADS = 64
HEAD_DIM = 128

WAVE = 64
M_TILES = NUM_HEADS // 16     # 4
K_STEPS = HEAD_DIM // 32      # 4
N_TILE = 16
WAVES = 4                     # waves per block
TPW = 2                       # ntiles processed per wave (ILP)
BLOCK_THREADS = WAVE * WAVES  # 256
COLS_PER_BLOCK = N_TILE * WAVES * TPW   # 128 kv columns finished per grid.y block
# Query rows handled per block. Each kv column tile is staged into LDS ONCE and
# reused across all ROWS_PER_BLOCK rows' MFMAs (rows processed sequentially so
# accumulators/weights are reused -> VGPR/occupancy stay healthy). This cuts KV
# global traffic and per-tile loop/staging overhead by ~ROWS_PER_BLOCK.
ROWS_PER_BLOCK = 2
NEG_INF = float("-inf")

# Back-compat default shape for the standalone Model below.
S_Q = 128
S_K = 1024


def build_mqa_logits(S_K, ROWS=ROWS_PER_BLOCK, use_lds=None):
    if use_lds is None:
        use_lds = USE_LDS_STAGING
    assert S_K % COLS_PER_BLOCK == 0, (
        f"S_K ({S_K}) must be a multiple of {COLS_PER_BLOCK}"
    )
    COLS_PER_WAVE = 16 * TPW          # kv columns staged per wave
    KV_SLAB_I64 = COLS_PER_WAVE * (HEAD_DIM // 8)  # i64 per wave slab
    NSTAGE = (KV_SLAB_I64 + WAVE - 1) // WAVE   # coalesced copy steps per wave

    @fx.struct
    class SharedStorage:
        # Single per-wave-private KV slab, shared across the ROWS rows in the
        # block. (A double-buffered variant regressed: doubling LDS halved
        # occupancy; this kernel is occupancy/TLP-bound.)
        kv: fx.Array[fx.Int64, WAVES * KV_SLAB_I64, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def mqa_logits_kernel(
        Q: fx.Tensor,        # i64 view: 8 fp8 per i64
        KV: fx.Tensor,       # i64 view
        KVSCALE: fx.Tensor,  # f32 [S_K]
        W: fx.Tensor,        # f32 [M*NUM_HEADS]
        STARTS: fx.Tensor,   # i32 [M]
        ENDS: fx.Tensor,     # i32 [M]
        OUT: fx.Tensor,      # f32 [M*S_K]
        GY: fx.Int32,        # grid.y extent (column-block split factor)
    ):
        # One block per ROWS query rows. The block (4 waves = 256 lanes) walks
        # the union of the rows' active kv-column blocks with an internal runtime
        # loop; each column tile is staged into LDS ONCE and reused across all
        # ROWS rows. Q frags for all ROWS rows stay resident; accumulators are
        # reused per row (rows processed sequentially) to keep VGPR pressure low.
        rt = fx.block_idx.x
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)

        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(KV, max_size=True)
        ks_buf = fx.rocdl.make_buffer_tensor(KVSCALE)
        w_buf = fx.rocdl.make_buffer_tensor(W)
        st_buf = fx.rocdl.make_buffer_tensor(STARTS)
        en_buf = fx.rocdl.make_buffer_tensor(ENDS)
        OUT_buf = fx.rocdl.make_buffer_tensor(OUT)

        c16 = fx.Int32(16)
        lane_row = lane % c16
        lane_kg = lane // c16
        ROW_I64 = fx.Int32(HEAD_DIM // 8)   # 16 i64 per head/kvcol row
        cpb = fx.Int32(COLS_PER_BLOCK)

        # ---- per-row metadata, Q frags, weights, out base (all resident) ----
        starts = [None] * ROWS
        ends = [None] * ROWS
        q_frag = [None] * ROWS
        wvals = [None] * ROWS
        out_base = [None] * ROWS
        for j in range_constexpr(ROWS):
            r = rt * fx.Int32(ROWS) + fx.Int32(j)
            s = fx.memref_load(st_buf, r)
            e = fx.memref_load(en_buf, r)
            starts[j] = (s > fx.Int32(0)).select(s, fx.Int32(0))
            ends[j] = (e < fx.Int32(S_K)).select(e, fx.Int32(S_K))
            out_base[j] = r * fx.Int32(S_K)
            q_row_base = r * fx.Int32(NUM_HEADS) * ROW_I64
            frag = [[None] * K_STEPS for _ in range(M_TILES)]
            for mt in range_constexpr(M_TILES):
                head_i64 = q_row_base + (fx.Int32(mt * 16) + lane_row) * ROW_I64
                for ks in range_constexpr(K_STEPS):
                    frag[mt][ks] = buffer_ops.buffer_load(
                        q_rsrc, head_i64 + fx.Int32(ks * 4) + lane_kg,
                        vec_width=1, dtype=fx.Int64)
            q_frag[j] = frag
            w_row_base = r * fx.Int32(NUM_HEADS)
            wj = [[None] * 4 for _ in range(M_TILES)]
            for mt in range_constexpr(M_TILES):
                for i in range_constexpr(4):
                    head = fx.Int32(mt * 16) + lane_kg * fx.Int32(4) + fx.Int32(i)
                    wj[mt][i] = fx.memref_load(w_buf, w_row_base + head)
            wvals[j] = wj

        # union of the rows' active column-block ranges (j=0 init is idempotent)
        cb_start = starts[0] // cpb
        cb_end = (ends[0] + fx.Int32(COLS_PER_BLOCK - 1)) // cpb
        for j in range_constexpr(ROWS):
            csj = starts[j] // cpb
            cb_start = (csj < cb_start).select(csj, cb_start)
            cej = (ends[j] + fx.Int32(COLS_PER_BLOCK - 1)) // cpb
            cb_end = (cej > cb_end).select(cej, cb_end)
        n_iter = cb_end - cb_start

        # 2-D grid: this block owns the strided subset {gy, gy+GY, gy+2*GY, ...}
        # of the active column-block range, so the column loop is split across
        # grid.y -> the launch has n_row_blocks*GY blocks and fills the GPU even
        # when M (row blocks) is small. GY==1 reproduces the 1-D path exactly.
        gy = fx.block_idx.y
        rem = n_iter - gy
        count = (rem > fx.Int32(0)).select(
            (rem + GY - fx.Int32(1)) // GY, fx.Int32(0))

        if const_expr(use_lds):
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            kv_sh = lds.kv.view(fx.make_layout(WAVES * KV_SLAB_I64, 1))
            wave_slab = wave * fx.Int32(KV_SLAB_I64)

        for cbi, _state in range(0, fx.arith._to_raw(count), 1, init=[fx.Int32(0)]):
            cb = cb_start + gy + _iv_to_i32(cbi) * GY
            nt0 = (cb * fx.Int32(WAVES) + wave) * fx.Int32(TPW)

            if const_expr(use_lds):
                # ---- coalesced global->LDS stage of this column tile (once) ----
                kv_global_base = nt0 * fx.Int32(16) * ROW_I64
                for e in range_constexpr(NSTAGE):
                    idx = lane + fx.Int32(e * WAVE)
                    if const_expr(KV_SLAB_I64 % WAVE != 0):
                        if idx < fx.Int32(KV_SLAB_I64):
                            v = buffer_ops.buffer_load(kv_rsrc, kv_global_base + idx,
                                                       vec_width=1, dtype=fx.Int64)
                            fx.memref_store(v, kv_sh, wave_slab + idx)
                    else:
                        v = buffer_ops.buffer_load(kv_rsrc, kv_global_base + idx,
                                                   vec_width=1, dtype=fx.Int64)
                        fx.memref_store(v, kv_sh, wave_slab + idx)
                if const_expr(_USE_LDS_BARRIER):
                    gpu.barrier()

            # B fragments + columns + per-column kv scale (shared across rows).
            b_frag = [[None] * K_STEPS for _ in range(TPW)]
            cols = [None] * TPW
            kvsc = [None] * TPW
            for tp in range_constexpr(TPW):
                cols[tp] = (nt0 + fx.Int32(tp)) * fx.Int32(16) + lane_row
                kvsc[tp] = fx.memref_load(ks_buf, cols[tp])
                for ks in range_constexpr(K_STEPS):
                    chunk = fx.Int32(ks * 4) + lane_kg
                    if const_expr(use_lds):
                        col_in_slab = fx.Int32(tp * 16) + lane_row
                        slab_idx = wave_slab + col_in_slab * ROW_I64 + chunk
                        b_frag[tp][ks] = fx.memref_load(kv_sh, slab_idx)
                    else:
                        # rank1-style: read this lane's column KV directly from
                        # global (strided, no LDS); reused across rows in regs.
                        g_idx = cols[tp] * ROW_I64 + chunk
                        b_frag[tp][ks] = buffer_ops.buffer_load(
                            kv_rsrc, g_idx, vec_width=1, dtype=fx.Int64)

            # ---- per row: MFMA (reusing staged KV) + reduce + store ----
            for j in range_constexpr(ROWS):
                accs = [[None] * M_TILES for _ in range(TPW)]
                for tp in range_constexpr(TPW):
                    for mt in range_constexpr(M_TILES):
                        accs[tp][mt] = arith.constant_vector(0.0, T.f32x4)
                for ks in range_constexpr(K_STEPS):
                    for tp in range_constexpr(TPW):
                        for mt in range_constexpr(M_TILES):
                            accs[tp][mt] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4,
                                [fx.arith._to_raw(q_frag[j][mt][ks]),
                                 fx.arith._to_raw(b_frag[tp][ks]),
                                 fx.arith._to_raw(accs[tp][mt]), 0, 0, 0])
                for tp in range_constexpr(TPW):
                    col = cols[tp]
                    in_window = (col >= starts[j]) & (col < ends[j])
                    partial = fx.Float32(0.0)
                    for mt in range_constexpr(M_TILES):
                        accv = fx.Vector(accs[tp][mt])
                        for i in range_constexpr(4):
                            sc = accv[i] * kvsc[tp]
                            sc = sc.maximumf(fx.Float32(0.0))
                            sc = sc * wvals[j][mt][i]
                            partial = partial + sc
                    r = partial
                    r = r + r.shuffle_xor(16, WAVE)
                    r = r + r.shuffle_xor(32, WAVE)
                    if lane_kg == fx.Int32(0):
                        if in_window:
                            fx.memref_store(r, OUT_buf, out_base[j] + col)

            # Per-wave-private slab: next iteration's staging waits on this
            # iteration's reads via the compiler's lds waitcnt (no barrier).
            if const_expr(_USE_LDS_BARRIER):
                gpu.barrier()
            _next = yield [_state[0] + fx.Int32(1)]

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KV: fx.Tensor,
        KVSCALE: fx.Tensor,
        W: fx.Tensor,
        STARTS: fx.Tensor,
        ENDS: fx.Tensor,
        OUT: fx.Tensor,
        n_blocks: int,
        gy_blocks: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        l = mqa_logits_kernel(Q, KV, KVSCALE, W, STARTS, ENDS, OUT, gy_blocks)
        l.launch(grid=(n_blocks, gy_blocks, 1), block=(BLOCK_THREADS, 1, 1),
                 stream=stream)

    return launch


# ---------------------------------------------------------------------------
# Shape-cached launcher + drop-in entry point
# ---------------------------------------------------------------------------

_LAUNCHERS = {}

try:
    from vllm.logger import init_logger as _init_logger
    _logger = _init_logger(__name__)
except Exception:  # standalone use without vLLM
    import logging
    _logger = logging.getLogger(__name__)


def _get_launcher(s_k):
    key = (s_k, ROWS_PER_BLOCK, USE_LDS_STAGING)
    fn = _LAUNCHERS.get(key)
    if fn is None:
        _logger.info("[flydsl-indexer] JIT-compiling fp8_mqa_logits kernel for "
                     "S_K=%d ROWS=%d use_lds=%s (cached variants so far: %d)",
                     s_k, ROWS_PER_BLOCK, USE_LDS_STAGING, len(_LAUNCHERS) + 1)
        fn = build_mqa_logits(s_k, ROWS_PER_BLOCK, USE_LDS_STAGING)
        _LAUNCHERS[key] = fn
    return fn


def flydsl_fp8_mqa_logits(q, k_fp8, kv_scales, weights, cu_starts, cu_ends):
    """Drop-in replacement for aiter/vendored fp8_mqa_logits on gfx942.

    Args:
        q:          [M, NUM_HEADS, HEAD_DIM] fp8 (e4m3fnuz)
        k_fp8:      [N, HEAD_DIM] fp8 (e4m3fnuz)
        kv_scales:  [N] or [N, 1] float32
        weights:    [M, NUM_HEADS] float32
        cu_starts:  [M] int32 (inclusive start per row)
        cu_ends:    [M] int32 (exclusive end per row)

    Returns:
        logits:     [M, N] float32; positions outside [start, end) are -inf.
    """
    M, H, D = q.shape
    assert H == NUM_HEADS and D == HEAD_DIM, (
        f"FlyDSL mqa_logits expects H={NUM_HEADS},D={HEAD_DIM}, got H={H},D={D}"
    )
    N = k_fp8.shape[0]
    dev = q.device

    # The gfx942 fp8 MFMA interprets operands as e4m3 *fnuz* (max 240). The
    # DeepSeek-V4 indexer hands q in e4m3 *fn* (OCP, max 448) and k in fnuz
    # (current_platform.fp8_dtype()). A naive value-cast fn->fnuz would saturate
    # any |x|>240. Instead we HALVE before the cast: an fp8 value scaled by 0.5
    # only decrements the exponent, so it is bit-exact in fnuz AND <=224<240, so
    # no saturation and no precision loss. The logit is linear in the q.kv dot
    # (relu is positive-homogeneous), so we undo the 0.5 factor(s) by scaling
    # kv_scales by 1/(a*b). a,b are the per-operand 0.5 factors actually applied.
    global _DTYPE_LOGGED
    if not _DTYPE_LOGGED:
        try:
            _logger.info("[flydsl-indexer] inputs q.dtype=%s |q|max=%.3f "
                         "k.dtype=%s |k|max=%.3f",
                         q.dtype, q.float().abs().max().item(),
                         k_fp8.dtype, k_fp8.float().abs().max().item())
        except Exception:
            pass
        _DTYPE_LOGGED = True
    scale_mul = 1.0
    if q.dtype != _FNUZ:
        q = (q.to(torch.float32) * 0.5).to(_FNUZ)
        scale_mul *= 2.0
    if k_fp8.dtype != _FNUZ:
        k_fp8 = (k_fp8.to(torch.float32) * 0.5).to(_FNUZ)
        scale_mul *= 2.0

    s_k = ((N + COLS_PER_BLOCK - 1) // COLS_PER_BLOCK) * COLS_PER_BLOCK

    # The kernel reads full COLS_PER_WAVE slabs, so KV/scales must be padded
    # up to s_k rows. Padded columns end up -inf (start/end clamp) and are
    # sliced off before returning.
    ks_src = kv_scales.reshape(-1).to(torch.float32)
    if scale_mul != 1.0:
        ks_src = ks_src * scale_mul
    if N == s_k:
        kv_p = k_fp8.contiguous()
        ks_p = ks_src.contiguous()
    else:
        kv_p = torch.zeros((s_k, D), dtype=k_fp8.dtype, device=dev)
        kv_p[:N] = k_fp8
        ks_p = torch.zeros((s_k,), dtype=torch.float32, device=dev)
        ks_p[:N] = ks_src

    # Row-tiling: each block handles ROWS_PER_BLOCK query rows, so pad M up to a
    # multiple of ROWS. Padded rows get start=end=0 (empty window -> no store),
    # and the padded output rows are sliced off. (Pad is <= ROWS-1 rows.)
    rows = ROWS_PER_BLOCK
    m_pad = ((M + rows - 1) // rows) * rows
    n_blocks = m_pad // rows
    # Adaptive column-split (grid.y) so small-M shapes still fill the GPU. At
    # large M, n_blocks alone saturates -> gy_blocks=1 (the 1-D path). Capped so
    # we never split into more groups than there are column blocks.
    _TARGET_BLOCKS = 2048
    _GY_CAP = 32
    ncb = s_k // COLS_PER_BLOCK
    gy_blocks = max(1, min(_GY_CAP, ncb, -(-_TARGET_BLOCKS // n_blocks)))

    q2 = q.contiguous()
    w_src = weights.contiguous().to(torch.float32)
    st = cu_starts.contiguous().view(-1).to(torch.int32)
    en = cu_ends.contiguous().view(-1).to(torch.int32)
    if m_pad != M:
        qpad = torch.zeros((m_pad, H, D), dtype=q2.dtype, device=dev)
        qpad[:M] = q2
        q2 = qpad
        wpad = torch.zeros((m_pad, NUM_HEADS), dtype=torch.float32, device=dev)
        wpad[:M] = w_src
        w_src = wpad
        stp = torch.zeros((m_pad,), dtype=torch.int32, device=dev)
        stp[:M] = st
        st = stp
        enp = torch.zeros((m_pad,), dtype=torch.int32, device=dev)  # end=0 -> empty
        enp[:M] = en
        en = enp

    # Pre-fill -inf: the kernel only writes in-window columns and skips
    # fully-masked column blocks, so masked/padded positions must already be
    # -inf for the downstream top-k (matches the stock kernel's semantics).
    out = torch.full((m_pad, s_k), float("-inf"), dtype=torch.float32, device=dev)

    q_i64 = q2.view(torch.int64).view(-1)
    kv_i64 = kv_p.view(torch.int64).view(-1)
    w = w_src.view(-1)
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)

    launcher = _get_launcher(s_k)
    launcher(q_i64, kv_i64, ks_p, w, st, en, out.view(-1), n_blocks, gy_blocks, stream)
    return out[:M, :N]


class Model(nn.Module):
    """Standalone fixed-shape (S_Q x S_K) module kept for the validation harness."""

    def __init__(self, s_k=S_K):
        super().__init__()
        self._s_k = s_k
        self._fn = build_mqa_logits(s_k)
        self._out = None

    def forward(self, Q, KV, kv_scales, weights, cu_starts, cu_ends):
        return flydsl_fp8_mqa_logits(Q, KV, kv_scales, weights, cu_starts, cu_ends)
