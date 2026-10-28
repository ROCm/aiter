# SPDX-License-Identifier: MIT
"""The Gluon MiniMax-M3 decode step kernel (gfx950, TP4, S <= 4 rows, contexts <= 2048).

One persistent launch of 256 CTAs x 512 threads runs every sparse layer (3..59) of a
decode step; layer i's tasks hand off through tagged pairs (tag = epoch * 64 + i + 1)
in the parity-(i % 2) halves of the scratch and peer buffers. Per layer, task t of a
stage on its CTA, every CTA walking the stages in order:

    qkv    (160 / 144, CTAs 0 ..): 16 rows of q | k | v | index q | index k; the
           layer's input (ar + res: vLLM's tensors at layer 0, the previous layer's
           mailboxes after) -> gemma RMSNorm -> per-token FP8 -> FP8 GEMV; the weights
           are issued before the input wait. CTA k < S also publishes token k's h.
    ik     (CTA 160, selecting layers): index k -> norm / rope -> the bf16 index cache
    split  (8 S, top CTAs): one 256-key partition x 16 heads; q's per-head norm and
           RoPE made here from the GEMV rows (no head hop); K / V issued before q
           lands; the partition holding the token's own key makes K / V itself,
           inserts them in the cache and re-reads that key
    o      (192, CTAs 0 ..): every token's partitions merged -> per-token FP8 ->
           o_proj (32 rows, weights issued at the layer's start) -> bf16 partials
           to every rank -> rank-ordered sum -> a, h_mid
    router (16 S, CTAs 192 ..): gemma RMSNorm(a + h) -> bf16 gate GEMV (weights
           issued early) -> (routing key, sigmoid) x 8 copies; its xn slice as MXFP8
    moe    every CTA routes every token itself (top-4 + the fused shared expert);
           shared up / gate (48, CTAs 192 ..), routed up / gate (192 S, CTAs 0 ..),
           down (192, CTAs 0 ..): route-weighted MXFP4 x MXFP8 -> bf16 partials to
           every rank -> rank-ordered sum -> ar (the next layer's input)

Numerics follow ATOM's FlyDSL M3 mono (ported in ``..kernels``) stage by stage; the
reductions differ in split and order only.
"""

import os

import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

from . import layout as LY
from .prims import (put_dev, PUT_CM, uniforms, TRACE_SLOTS, T1, TRACE, WAIT2, wait_words2, stamp_init, stamp_flush, 
    _peek, addr_of, poll1, put_sys, pair_word, wait_words, BLOCKS, THREADS, as_ptr, bf_hi, bf_lo, bf_pair, bf16_round, clamp_fp8, code_ceil, drain, hw_exp2, hw_rcp,
    hw_rsq, note_err, pin, pollv1, put, rcp_pow2, route_key, scale_index, stamp, tids, uniform, uniform64,
    put_dense, peek_dense, wait_dense, reset_dense, put_dense2, reset_dense2,
)

HIDDEN = gl.constexpr(LY.HIDDEN)
HPAIRS = gl.constexpr(LY.HIDDEN // 2)
HEAD_DIM = gl.constexpr(LY.HEAD_DIM)
HEADS = gl.constexpr(LY.HEADS)
QKV_PAIRS = gl.constexpr(LY.QKV_ROWS // 2)
O_K = gl.constexpr(LY.O_K)
INTER = gl.constexpr(LY.INTER)
N_SPLIT = gl.constexpr(LY.N_SPLIT)
SPLIT_KEYS = gl.constexpr(LY.SPLIT_KEYS)
MAX_S = gl.constexpr(LY.MAX_S)  # the scratch / peer row stride (rows a build serves: S)


@gluon.constexpr_function
def ncols(S):
    """Token columns of a build's GEMV tiles: 4 up to 4 rows, else S rounded up to 2^n."""
    S = getattr(S, "value", S)
    return 4 if S <= 4 else 1 << (S - 1).bit_length()
N_ROUTED = gl.constexpr(LY.N_ROUTED)
TOP_K = gl.constexpr(LY.TOP_K)
SLOTS = gl.constexpr(LY.SLOTS)
SCORE_COPIES = gl.constexpr(LY.SCORE_COPIES)
TAGS = gl.constexpr(LY.TAGS_PER_STEP)
EP_MARK_PAIRS = gl.constexpr(LY.EP_MARKS // 2)
LOG2E = gl.constexpr(1.4426950408889634)
NEG = gl.constexpr(-3.4e38)
FP8_MAX = gl.constexpr(448.0)
NPTR = gl.constexpr(20)  # pointer-table entries a layer (runner.LAYER_PTRS)

# error codes (errors[cta]): which wait's watchdog fired
E_X, E_QKV, E_SPLIT, E_ATTN, E_H, E_A, E_SC, E_XN, E_MID, E_FFN, E_EPOCH, E_OQ, E_X8 = (gl.constexpr(i) for i in range(1, 14))
E_DOWN, E_DLAND, E_DH = (gl.constexpr(i) for i in range(14, 17))  # dense attention all-reduce: owner, landing, h
POLL_ROUNDS = gl.constexpr(1 << 20)  # wait_group's watchdog (rounds of ~1 us)
DN_STREAM = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_STREAM", "0")))  # down rounds in the ug tail
DN_DEPTH = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_DEPTH", "12")))  # down weight rounds prefetched

# ---------------------------------------------------------------- layouts
# GEMV tiles: 8 waves stacked along M (16 rows each), N = 16 P columns
MFMA = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1]))
A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=16))
B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=16))
# attention: 8 waves side by side along N (keys for QK, dims for PV)
MFMA2 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 8]))
A2 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MFMA2, k_width=16))
B2 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MFMA2, k_width=16))
X_SH3 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[2, 1, 0]))
FLAT4 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[3, 2, 1, 0]))
FLAT1 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]))
FLAT2 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]))
FLAT3 = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]))
# GEMV B tiles [rounds, columns, 128 B] (column = token * P + k-slice p): 16 B chunks XOR-swizzled
# by row, so a B_L read (lane = column, 16 B of k) spreads over the banks (flat: 8-way conflicts)
BSW = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=4, order=[2, 1, 0]))
ROWSH = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0]))  # rows indexable
R128 = gl.constexpr(gl.BlockedLayout([1], [64], [8], [0]))  # [128] bytes, replicated
ROW1 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[0]))  # a ROWSH row
BSW2 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=4, order=[1, 0]))
# an FP8 weight tile [128 rows, 8 chunks, 16 B] of aiter's (16, 16) shuffle: lane = chunk %
# 4 * 16 + row % 16 holds 16 B chunks c and c + 4 of a 128-k step -- A_L once reshaped
W8_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 4, 0]],
    lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0], [0, 2, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0], [64, 0, 0]], block_bases=[], shape=[128, 8, 16]))
# an MXFP4 weight tile [128 rows, 4 chunks, 16 B] (one 128-k step): A_L once reshaped
W4_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8]],
    lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0], [0, 2, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0], [64, 0, 0]], block_bases=[], shape=[128, 4, 16]))
# K^T [8 d-chunks, 16 B, 256 keys] of a partition (B2 once reshaped): a lane's 16 d of
# one key are 16 contiguous bytes of a page-16 K page [d / 16][t][16]
KT_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [4, 0, 0], [0, 0, 128]],
    lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]],
    warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]], block_bases=[], shape=[8, 16, 256]))
# V [16 pages, 16 t, 128 d] (B2 once reshaped): a lane's 16 keys of one dim are 16
# contiguous bytes of a page-16 V page [d][t]
VT_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [4, 0, 0], [8, 0, 0]],
    lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]],
    warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]], block_bases=[], shape=[16, 16, 128]))
# TILE_W: the same tiles as i32 words (4 bytes a register): K [8 d-chunks, 4 words, 256 keys],
# V [16 pages, 4 words (4 keys each), 128 d]
KTW_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1, 0], [0, 2, 0], [4, 0, 0], [0, 0, 128]],
    lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]],
    warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]], block_bases=[], shape=[8, 4, 256]))
VTW_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1, 0], [0, 2, 0], [4, 0, 0], [8, 0, 0]],
    lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]],
    warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]], block_bases=[], shape=[16, 4, 128]))
# the words' LDS twin of KSW2 (16 B vectors, the same XOR phase): bytes read back in B2
KSW2W = gl.constexpr(gl.SwizzledSharedLayout(vec=4, per_phase=1, max_phase=8, order=[1, 0]))
H2 = gl.constexpr(gl.BlockedLayout([1, 2], [2, 32], [8, 1], [1, 0]))  # [rows, 64 pairs], 2 a thread
H4 = gl.constexpr(gl.BlockedLayout([1, 1], [1, 64], [8, 1], [1, 0]))  # [4 rows, 64 pairs]: H1's lane order a row
# [8 d-chunks, 4 words, 4 in-step keys]: the keys in registers, (chunk, word) as KTW_L's slice
KD_L = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 2, 0], [4, 0, 0]],
    lane_bases=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 4]))
G4 = gl.constexpr(gl.BlockedLayout([1, 1, 4], [1, 64, 1], [8, 1, 1], [2, 1, 0]))  # [4 rows, 64, 4 groups]
VD4 = gl.constexpr(gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0]))  # [128 d, 4 keys]: keys in registers
# a [16 rows, 16 tokens] epilogue tile, and the [8 pairs, 16 tokens] one
EP_L = gl.constexpr(gl.BlockedLayout([1, 1], [4, 16], [4, 2], [1, 0]))
PR_L = gl.constexpr(gl.BlockedLayout([1, 1], [8, 8], [1, 8], [0, 1]))
# PUSH_W: a task's [8 row pairs, 4 tokens, 2 tiles] partial words once a wave, a word a lane
PW4 = gl.constexpr(gl.BlockedLayout([1, 1, 1, 1], [1, 8, 4, 2], [8, 1, 1, 1], [3, 2, 1, 0]))
# indexer scores: [16 step rows, keys] = index q [16, 128] x index K^T [128, keys], bf16 (the
# Triton scorer's v_mfma_f32_16x16x32_bf16 chained over four 32-k chunks, as FlyDSL's)
MFS = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 8]))
AS_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MFS, k_width=8))
BS_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MFS, k_width=8))
# selection: [512 threads, 64] -- a thread's own entry (row) against a whole LDS chunk (registers)
CL = gl.constexpr(gl.BlockedLayout([1, 64], [64, 1], [8, 1], [1, 0]))
G16 = gl.constexpr(gl.BlockedLayout([1, 1], [4, 16], [8, 1], [1, 0]))  # [32, 16]: group g = thread % 16
# a 64 x 64 tile of candidate pairs (row: mine, column: other): 8 columns a thread, 8 lanes a row
TL64 = gl.constexpr(gl.BlockedLayout([1, 8], [8, 8], [8, 1], [1, 0]))


# scratch regions (byte offsets in a half)
R_QKV = gl.constexpr(LY.region("qkv"))
R_SPO = gl.constexpr(LY.region("spo"))
R_SPM = gl.constexpr(LY.region("spm"))
R_SPL = gl.constexpr(LY.region("spl"))
R_A = gl.constexpr(LY.region("a"))
R_H = gl.constexpr(LY.region("h"))
R_HM = gl.constexpr(LY.region("hm"))
R_SC = gl.constexpr(LY.region("sc"))
R_XN8 = gl.constexpr(LY.region("xn8"))
R_XSC = gl.constexpr(LY.region("xsc"))
R_MID = gl.constexpr(LY.region("mid"))
R_OWN = gl.constexpr(LY.region("own"))
R_MIDW = gl.constexpr(LY.region("midw"))
R_XNP = gl.constexpr(LY.region("xnp"))
R_XSCP = gl.constexpr(LY.region("xscp"))
R_XNF = gl.constexpr(LY.region("xnf"))
R_UGF = gl.constexpr(LY.region("ugf"))
R_MIDWSC = gl.constexpr(LY.region("midwsc"))
R_AR = gl.constexpr(LY.region("ar"))
R_OQ = gl.constexpr(LY.region("oq"))
R_OQS = gl.constexpr(LY.region("oqs"))
R_X8 = gl.constexpr(LY.region("x8"))
R_X8S = gl.constexpr(LY.region("x8s"))
R_QKVP = gl.constexpr(LY.region("qkvp"))
# the indexer (contexts past MAX_SHORT): scores, long-share candidates, index-key flags, and the
# step-global sparse tables (past both scratch halves)
R_ISC = gl.constexpr(LY.region("isc"))
R_CAND = gl.constexpr(LY.region("cand"))
R_IKF = gl.constexpr(LY.region("ikf"))
STAB_OFF = gl.constexpr(LY.StepLayout().stab())
STAB_WORDS = gl.constexpr(LY.STAB_WORDS)
MAX_SHORT = gl.constexpr(LY.MAX_SHORT)
TOPK_BLOCKS = gl.constexpr(LY.TOPK_BLOCKS)
LONG_FROM = gl.constexpr(LY.LONG_FROM_BLOCKS)
MAX_IB = gl.constexpr(LY.MAX_INDEX_BLOCKS)
LONG_S = gl.constexpr(LY.LONG_S)
IQ_P0 = gl.constexpr((LY.HEADS + 2) * 64)  # a token's index q rows (pairs) in the qkv rows
SC_NB = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_NB", "2")))  # index-K blocks a score tile
CAND_CAP = gl.constexpr(256)  # candidates past the group floor counted against each other
WIDE_FROM = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_WIDE_FROM", str(LY.WIDE_FROM))))
AG_RS_FROM = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_AG_RS_FROM", str(LY.AG_RS_FROM))))
AAG_OFF = gl.constexpr(LY.peer_regions()["aag"])
FAG_OFF = gl.constexpr(LY.peer_regions()["fag"])
ATTND_OFF = gl.constexpr(LY.peer_regions()["attnd"])
FFND_OFF = gl.constexpr(LY.peer_regions()["ffnd"])
AAGD_OFF = gl.constexpr(LY.peer_regions()["aagd"])
FAGD_OFF = gl.constexpr(LY.peer_regions()["fagd"])


@gluon.constexpr_function
def norm_base(S):
    """layout.norm_base."""
    S = getattr(S, "value", S)
    return 192 if S > 16 else 161 + max(S, 4)


@gluon.constexpr_function
def merge_base(S):
    """layout.merge_base."""
    S = getattr(S, "value", S)
    return 192 if S <= 4 else 161 + 2 * max(S, 4)


ROWS16 = gl.constexpr(LY.ROWS16)
NQ_REUSE = gl.constexpr(LY.NQ_REUSE)
N_O = gl.constexpr(LY.N_O)
N_DN = gl.constexpr(LY.N_DN)
UG_TASKS = gl.constexpr(LY.UG_TASKS)
ROUTER_BASE = gl.constexpr(LY.ROUTER_BASE)
SHARED_BASE = gl.constexpr(LY.SHARED_BASE)


# ---------------------------------------------------------------- small helpers
@gluon.jit
def opq():
    """An opaque 0 (a side-effecting s_mov), asserted a multiple of 2^16: added to every
    index tensor so that LLVM cannot hoist a stage's index math out of the layer loop
    (it would hold every stage's indices live across all of them: hundreds of spills)."""
    return gl.multiple_of(gl.inline_asm_elementwise("s_mov_b32 $0, 0", "=s", [], dtype=gl.int32, is_pure=False,
                                                    pack=1), 65536)


# the multi-layer launch (VLLM_M3_GLUON_LOOP=1) needs loop-variant index tensors; a launch a
# layer has no loop to hoist out of, and an opaque arange only blocks constant folding
# (every element's index math in VALU, recomputed in every unrolled round)
DBG_NOTAIL = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DBG_NOTAIL", "0")))  # diagnostics: no tail split work
TAIL_CACHE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_TAIL_CACHE", "1")))  # own key via the cache + a flag (FlyDSL)
TILE_W = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_TILE_W", "0")))  # split: K / V tiles as i32 words (TAIL_CACHE)
TAIL_SELF = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_TAIL_SELF", "0")))  # tail split: in-step keys from their GEMV rows
# word path (TAIL_CACHE=0): the QK of the cached keys before the in-step keys' words land (they
# then only replace their columns): the tail partition's wait overlaps its loads and QK
TAIL_LATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_TAIL_LATE", "0")))
# TAIL_CACHE: past the own-key flags only the in-step keys' K columns / V groups are re-read (not
# the partition's whole 64 KB, ~2.4 us of the tail partition)
TAIL_PATCH = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_TAIL_PATCH", "0")))
# 2 < S <= 8, TAIL_CACHE: the own key's K on its own-key CTA, its V on the token's norm CTA (idle from
# its norm to its o task), each with a flag (pairs k * 128, k * 128 + 64; the tail split waits on both)
OWN_SPLIT = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_OWN_SPLIT", "1")))
# S <= 16, not hybrid: the norm tasks' FP8 words tagged (no drain before the scale), the qkv tasks
# poll them directly (no scale gate, then a second load): norm -> qkv in one hop
X8_TAG = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_X8_TAG", "0")))


@gluon.constexpr_function
def own_split(S):
    S = getattr(S, "value", S)
    return bool(OWN_SPLIT.value) and bool(TAIL_CACHE.value) and not TAIL_SELF.value and 2 < S <= 8
XN_GATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_XN_GATE", "1")))  # xn wait: gated (FlyDSL fetch_xn)
# a launch a layer: the previous layer's ar / hm come from the previous kernel, complete at
# its end -- plain bf16 words, no tags and no wait (FlyDSL's layer boundary)
PLAIN_IN = gl.constexpr(os.environ.get("VLLM_M3_GLUON_LOOP", "0") != "1" and
                        os.environ.get("VLLM_M3_GLUON_PLAIN_IN", "1") == "1")
ROUTER_ONEWAIT = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTER_ONEWAIT", "0")))  # router: a and h in one wait
ROUTE_GATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTE_GATE", "1")))  # top-k: gated scores wait
ROUTE_GATE_W = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTE_GATE_W", "1")))  # wide top-k: gated scores wait
SC_LINE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_LINE", "1")))  # wide router: a copy's 8 scores as one full-line store
XN_LDS_W = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_XN_LDS_W", "1")))  # wide router: the xn slice from LDS (coalesced)
ROUTER_AAG = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTER_AAG", "1")))  # AG_RS: the router reads the gathered sums itself
# wide router task: xn stores before the logits, flag after the scores without a vmcnt drain (FlyDSL's)
ROUTER_XN_EARLY = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTER_XN_EARLY", "0")))
SC_PUT_DEV = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_PUT_DEV", "0")))  # router scores put at device scope (asm)
DN_EUS_EARLY = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_EUS_EARLY", "0")))  # wide down: unit experts read before the ug loop
DN_NOMASK = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_NOMASK", "0")))  # wide down: offset-clamped instead of masked loads
DN_NOWASTE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_NOWASTE", "3")))  # wide down: missing chunks re-read a loaded line (3), masked (1), waste (0)
DN_LEAN = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_LEAN", "0")))  # wide down: real units only, tables once a unit, fma
UG_FLAG = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_UG_FLAG", "0")))  # wide: a flag a pair, the down gates on them
DN_MIDS_FIRST = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_MIDS_FIRST", "0")))  # wide down: mids peeked before the weight prefetch
SPLIT_BTPF = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SPLIT_BTPF", "0")))  # split: the tail keys' block-table entries loaded early
# TAIL_ACQ: cache rows written in this launch (the own keys' K / V in the tail split, the new index keys in
# the scorers) are re-read after a flag that only wave 0's lanes poll (``land``, no barrier): the other
# waves re-read the rows before the flag landed and attend stale bytes (the dense path: a 4% error on the
# attention output, every token's own key); and a CU whose L1 holds the rows from the early load (the
# speculative rows share the tail page) re-reads them stale. 8 (default): vmcnt(0), CTA barrier, L1
# invalidate + vmcnt(0), CTA barrier; 2: L1 invalidate + vmcnt(0) + barrier (timing dependent); 6: barrier only (dense rows exact, the shared tail
# page not); 1: L2 invalidate + barrier (~1 us a layer); 4 / 5: invalidate without the barrier (WRONG);
# 0: off (WRONG).
TAIL_ACQ = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_TAIL_ACQ", "8")))


@gluon.jit
def l2_acquire(x):
    """The readers' wait after a wave-0 flag poll (TAIL_ACQ modes; ``x``: any [512] T1 tensor)."""
    if TAIL_ACQ == 4:
        y = gl.inline_asm_elementwise("buffer_inv sc0", "=v,0,~{memory}", [x], dtype=x.dtype, is_pure=False, pack=1)
    elif TAIL_ACQ == 8:  # every wave's in-flight loads drained, the CTA barrier, then the L1 invalidate: the
        # early page loads of a wave returning after another wave's invalidate would refill the L1 stale
        y = gl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=v,0,~{memory}", [x], dtype=x.dtype, is_pure=False,
                                      pack=1)
        gl.barrier()
        y = gl.inline_asm_elementwise("buffer_inv sc0\ns_waitcnt vmcnt(0)", "=v,0,~{memory}", [y], dtype=y.dtype,
                                      is_pure=False, pack=1)
        gl.barrier()
    elif TAIL_ACQ == 6:  # CTA barrier only: every wave waits for the polling wave (no cache invalidate)
        y = x
    elif TAIL_ACQ == 5:  # L1 invalidate + this wave's wait, no CTA barrier
        y = gl.inline_asm_elementwise("buffer_inv sc0\ns_waitcnt vmcnt(0)", "=v,0,~{memory}", [x], dtype=x.dtype,
                                      is_pure=False, pack=1)
    elif TAIL_ACQ == 2:
        y = gl.inline_asm_elementwise("buffer_inv sc0\ns_waitcnt vmcnt(0)", "=v,0,~{memory}", [x], dtype=x.dtype,
                                      is_pure=False, pack=1)
    else:
        y = gl.inline_asm_elementwise("buffer_inv sc1\ns_waitcnt vmcnt(0)", "=v,0,~{memory}", [x], dtype=x.dtype,
                                      is_pure=False, pack=1)
    if (TAIL_ACQ != 4) and (TAIL_ACQ != 5) and (TAIL_ACQ != 8):
        gl.barrier()
    return y
MIDS_GATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_MIDS_GATE", "0")))  # down: gated mids wait
XN_FIRST = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_XN_FIRST", "1")))
PUSH_W = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_PUSH_W", "1")))  # o / FFN partials: wave r -> rank r
SH_RFIRST = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SH_RFIRST", "0")))  # shared CTAs: the routing poll first
SH_LOAD_LATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SH_LOAD_LATE", "0")))  # SH_AFTER: weights after routing
SHARED_AFTER = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SHARED_AFTER", "1")))  # 4 rows: shared GEMV after routing
UG_TOKC = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_UG_TOKC", "1")))  # 4 rows: token-contiguous up / gate tasks  # routed CTAs: xn before routing
# S > 16 hybrid: the all-reduces' hand-offs dense (plain bf16 pairs, the value its own flag:
# prims' put_dense / wait_dense) instead of tagged pairs -- half the link bytes. A slot's reset
# (by its reader) is ordered before its next write (two layers on, another rank) only by the
# layer chain's cross-rank hand-offs: bench/ablate.py's replays of one stage have none, and its
# dense "far" region can lose a partial (watchdog) -- ablate with VLLM_M3_GLUON_DENSE_AR=0
DENSE_AR = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DENSE_AR", "1")))
# qkv past 16 rows: a batched MFMA (a batch a wave = its own 128-k slice, as the wide MoE's), B
# gathered from the x8 rows straight into registers (no LDS), the next 16-token group's B in
# flight under the current one's MFMAs. The block-diagonal form ran 8 MFMAs a wave a round
# (7 discarded) and every wave read the whole 16 KB B tile from LDS a round.
QKV_B3 = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_B3", "1")))
# qkv past 16 rows, K split across CTAs: CTA 8 u + c takes row tiles 8 u .. + 8 (a wave each)
# over K slice c (steps 8 r + c) -- the batched path's wave c, so its partials are bit for bit
# the old per-wave ones -- reading 1/8 of every token's x8 row (each CTA read all of it: 46 MB a
# layer at 48 rows, the qkv stage's cost); the 8 slices of a tile meet on CTA 8 u + c (tagged
# f32 partials), summed in slice order (the old wave order)
QKV_KS = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_KS", "1")))
QKV_KS_FROM = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_KS_FROM", "16")))  # rows past this: K-split, else batched
# up to 16 rows the batched qkv / o pay only where the block-diagonal tiles were 16 tokens wide
# (S > 8: 4 resp. 8x redundant); at 1-4 rows they were not (4 tokens x 4 / 8 slices filled the
# MFMA) and the padded 16-token batch costs 1-2 us a layer
SMALL_FROM = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SMALL_FROM", "8")))
QKV_SMALL = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_SMALL", "1")))  # SMALL_FROM < S <= 16: batched qkv
# o GEMM past 16 rows: batched over its 4 K slices (warps [4 slices, 2 row tiles]: the tiles share
# a slice's B), each wave reading only its slice from LDS -- the block-diagonal form ran 4 MFMAs
# a wave a round (3 discarded) and read the whole round's B
O_B4 = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_O_B4", "1")))
QKV_BYTES = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_BYTES", "1")))  # B3's x8 loaded as bytes
QKV_DBG = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_DBG", "0")))  # (diagnostics) qkv parts off
# dense, one-shot: every rank's partial to every rank, each summing them in rank order (the
# owner's sum, bit for bit) -- one hop instead of reduce-scatter + all-gather, at the link bytes
# the tagged two-shot had
DENSE_1S = gl.constexpr(os.environ.get("VLLM_M3_GLUON_DENSE_1S", "0") == "1")  # (48 rows: slower, 96 vs 86 us -- the links)
DENSE_ATTN = gl.constexpr(DENSE_AR.value and os.environ.get("VLLM_M3_GLUON_DENSE_ATTN", "1") == "1")
DENSE_FFN = gl.constexpr(DENSE_AR.value != 0
                         and os.environ.get("VLLM_M3_GLUON_DENSE_FFN", "1") == "1"
                         and os.environ.get("VLLM_M3_GLUON_NORM_P3", "1") == "1"
                         and os.environ.get("VLLM_M3_GLUON_NORM_WAVE", "0") == "0")
O_PIPE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_O_PIPE", "1")))  # S > 16: attention all-reduce pipelined behind the o GEMM
# dense pipelined o: an iteration's owner / landing / next-gate loads issued before its GEMM, one
# wait after it (they were three round trips in a row, ~3.6 us a 16-token group)
O_TST = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_O_TST", "0")))  # (diagnostics) o group 2 stamps
O_LAG = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_O_LAG", "2")))  # O_OVL: land group g - O_LAG (3: no gain)
O_OVL = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_O_OVL", "1")))
O_PF = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_O_PF", "1")))  # S > 16 o tasks: next group's words prefetched
ROUTER_PF = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTER_PF", "1")))  # S > 16: next token's a, h prefetched
XN_SLICE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_XN_SLICE", "1")))  # router: xn slice quantized alone
DN_UNI = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_DN_UNI", "1")))  # down: expert id made uniform
ROUTER_HPRE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTER_HPRE", "1")))  # router: h read before the a poll
ROUTER_GATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTER_GATE", "1")))  # router: gated a wait
NORM_TASKS = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_NORM_TASKS", "1")))  # S > 2: per-token norm tasks
# the norm / merge tasks' FP8 rows as plain words, their scale's pair the flag (FlyDSL's): half
# the bytes of tagged pairs for every GEMV CTA reading every token's row
PLAIN_HO = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_PLAIN_HO", "1")))
MERGE_ON_SPLIT = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_MERGE_ON_SPLIT", "1")))  # S > 4: merges on split CTAs
MERGE_SPREAD = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_MERGE_SPREAD", "2")))  # 32 < S <= 48 (1) / 60 (2): merges rebalanced
MERGE_TASKS = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_MERGE_TASKS", "1")))  # S > 1: per-token merge tasks
# merge: m / l of the 8 partitions x 16 heads loaded once per word in a compact [8, 16] layout (ML's
# slice put every word on 32 lanes: 16 replicated loads a wave); 2 = the sum in ML's slice as before
MERGE_CML = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_MERGE_CML", "1")))
# split: q's rows and RoPE partners landed in one batched wait (not two serial lands)
HEAD_WG = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_HEAD_WG", "1")))
# split: the V tile's past-context zeroing unconditional (the branch over it skips ~2 KB of code)
SPLIT_NOBR = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SPLIT_NOBR", "1")))
# the next layer's weights of this CTA's tasks pulled into this XCD's L2 during the FFN all-reduce
# wait (pid p runs on XCD p % 8 every launch; L2 survives the kernel boundary): 1 = its qkv slice
# (96 KB, one dword a 128 B line from waves 1-7), 2 = + its o_proj slice (64 KB)
PF_NEXT = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_PF_NEXT", "0")))
O_LGATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_O_LGATE", "2")))  # o merge: wave-gated l poll first
# NL layers a launch (VLLM_M3_GLUON_NL, the runner's constexpr NL): the layer bodies unrolled in
# sequence (no loop, so no loop-carried live ranges); the index tensors opaque so that nothing is
# CSE'd across the bodies
NL_ENV = int(os.environ.get("VLLM_M3_GLUON_NL", "1"))
OPQ_AR = gl.constexpr(bool(int(os.environ.get("VLLM_M3_GLUON_OPQ",
                                             "1" if (os.environ.get("VLLM_M3_GLUON_LOOP", "0") == "1" or NL_ENV > 1) else "0"))))
# hybrid: the MoE (sorting, up / gate, down) is AITER's, between launches. Launch li runs
# layer li - 1's FFN all-reduce (of AITER's output) then layer li up to the router, which
# leaves xn (bf16) and the top-k ids / weights for AITER; launch n_layers is the last
# all-reduce alone
HYB = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_HYBRID", "0")))
NOAUX = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_NOAUX", "0")))  # (debug: no EAGLE3 aux reads)


@gluon.jit
def ar(n: gl.constexpr, layout: gl.constexpr):
    """``gl.arange(0, n, layout)``, loop-variant (``opq``) in the multi-layer launch."""
    if OPQ_AR:
        return gl.arange(0, n, layout=layout) + opq()
    else:
        return gl.arange(0, n, layout=layout)


@gluon.jit
def zacc(shape: gl.constexpr, layout: gl.constexpr):
    """Zeros for an MFMA accumulator, opaque (``opq``): with an inline-constant 0 as src C,
    LLVM gives v_mfma_scale a destination partially overlapping its A / scale sources and
    reuses those registers right after it -- the late result write then clobbers them
    (seen as memory faults at random addresses)."""
    return gl.full(shape, 0.0, gl.float32, layout) + opq().to(gl.float32)


LROW_ASM = gl.constexpr(int(__import__("os").environ.get("VLLM_M3_GLUON_LROW_ASM", "1")))


@gluon.constexpr_function
def _row_asm(js):
    n = len(js)
    return "\n".join([f"s_load_dwordx2 ${k}, ${n}, {8 * j}" for k, j in enumerate(js)] + ["s_waitcnt lgkmcnt(0)"])


# EP_SLOAD: the epoch and the layer's 160 B pointer-table row (its 64 B scalar-cache lines; the
# row values themselves unused here) in one block of scalar loads and one wait
_EP_ASM = gl.constexpr("s_load_dword $0, $5, 0\ns_load_dwordx2 $1, $6, 0\ns_load_dwordx2 $2, $6, 64\n"
                       "s_load_dwordx2 $3, $6, 128\ns_load_dwordx2 $4, $6, 152\ns_waitcnt lgkmcnt(0)")


@gluon.constexpr_function
def _row_constraints(js):
    return ",".join(["=&s"] * len(js) + ["s"])  # early clobber: an output must not take the base


@gluon.constexpr_function
def _row_dtypes(js):
    return (gl.int64,) * len(js)


@gluon.jit
def lrow(ptab, li, js: gl.constexpr):
    """Pointer-table entries js of layer li as CTA-uniform i64 scalars: one asm block of
    scalar loads and one wait (an entry at a time, each is a serial vector round trip)."""
    if LROW_ASM:
        base = (ptab + li * NPTR).to(gl.int64, bitcast=True)
        return gl.inline_asm_elementwise(_row_asm(js), _row_constraints(js), [base], dtype=_row_dtypes(js),
                                         is_pure=False, pack=1)
    else:
        out = ()
        for j in gl.static_range(len(js)):
            out = out + (uniform64(gl.load(ptab + li * NPTR + js[j])),)
        return out


@gluon.jit
def lp(v, ty: gl.constexpr):
    """A table entry as a ``*ty`` (every table pointer is a torch allocation's or a 16 B
    aligned view's: told to the alignment analysis, else every buffer load of it goes out
    byte by byte)."""
    return gl.multiple_of(v.to(gl.pointer_type(ty), bitcast=True), 16)


@gluon.jit
def rgn(base, off: gl.constexpr):
    """A pair region of a scratch half (``base``: *u8; ``off``: an R_* offset)."""
    return as_ptr(base, off, gl.int64)


@gluon.jit
def i32p(p):
    return p.to(gl.pointer_type(gl.int32), bitcast=True)


@gluon.jit
def u8p(p):
    return p.to(gl.pointer_type(gl.uint8), bitcast=True)


# the weights' cache policy (FlyDSL: nontemporal, CM_NT -- a weight a step reads once
# streams past the caches it would only evict the step's reused data from); "" = default
# ablation-only STAGES bits (bench/ablate.py): one stage alone (the default 511 runs them all)
AB_QKV = gl.constexpr(1 << 23)  # the qkv GEMV tasks without the norm tasks
AB_OWN = gl.constexpr(1 << 24)  # the own-key tasks (k norm / RoPE / KV insert) without the split
AB_SPLIT = gl.constexpr(1 << 25)  # the split attention without the own-key tasks
AB_MERGE = gl.constexpr(1 << 26)  # the merges without the o tasks
AB_O = gl.constexpr(1 << 27)  # the o tasks (GEMV + attention all-reduce) without the merges
AB_ONOAR = gl.constexpr(1 << 30)  # the o tasks' GEMM alone (no all-reduce: sends, owner sums, landing)
AB_ROUTER = gl.constexpr(1 << 28)  # the router tasks without the top-k
AB_TOPK = gl.constexpr(1 << 29)  # the top-k (route_out) alone
NORM_WAVE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_NORM_WAVE", "0")))  # past 16 rows: a token a wave, 8 a CTA
NORM_P3 = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_NORM_P3", "1")))  # hybrid past 16 rows: one batched wait for the row
NORM_1R = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_NORM_1R", "1")) * int(os.environ.get("VLLM_M3_GLUON_HYBRID", "0")))  # norm task: ss + amax in one reduction
NORM_NB = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_NORM_NB", "1")))  # norm task: scale put without a barrier
# norm task (not hybrid): the row as 16 B a thread ([chunks of 8, 4 pairs]: 6 loads a thread, not
# 18 dwords) and flat CTA reductions (no 3-D fold / convert_layout); same arithmetic
NORM_W = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_NORM_W", "0")))
# the step's epoch as a scalar load in one block with the layer's pointer-table row (its lines
# then sit in the scalar cache for every lrow): one round trip at a CTA's start, not two
EP_SLOAD = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_EP_SLOAD", "1")))
QKV_SCPF = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_SCPF", "0")))  # qkv epilogue scales peeked at the gate
QKV_WDELAY = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_WDELAY", "32")))  # (experiment) s_sleep before qkv weights


@gluon.constexpr_function
def _sleep_asm(n):
    return f"s_sleep {getattr(n, 'value', n)}"


QKV_GGATE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_QKV_GGATE", "0")))  # S > 16 qkv: gate per token group
WCACHE = gl.constexpr(os.environ.get("VLLM_M3_GLUON_WCACHE", ".cs"))  # .cs: sc0 nt on gfx950


@gluon.jit
def wload(base, off, mask=None, other=None):
    """A weight stream's buffer load (WCACHE)."""
    if WCACHE == "":
        v = cdna4.buffer_load(base, off, mask=mask, other=other)
    else:
        v = cdna4.buffer_load(base, off, mask=mask, other=other, cache=WCACHE)
    return v


@gluon.jit
def w8_rows():
    return ar(128, layout=gl.SliceLayout(1, gl.SliceLayout(2, W8_L)))


@gluon.jit
def w8_tile(base, row_term):
    """An FP8 A tile [128, 128] (A_L) of a (16, 16)-shuffled weight: virtual row m's
    16-row group and step folded into ``row_term[m]`` (bytes, + (m % 16) * 16)."""
    c = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, W8_L)))[None, :, None]
    b = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, W8_L)))[None, None, :]
    v = wload(base, row_term[:, None, None] + c * 256 + b)
    return gl.convert_layout(gl.reshape(v, [128, 128]), A_L)


@gluon.jit
def w4_rows():
    return ar(128, layout=gl.SliceLayout(1, gl.SliceLayout(2, W4_L)))


@gluon.jit
def w4_off(HALF_OFF: gl.constexpr, KMAX: gl.constexpr = 4):
    """The lane's byte offsets of an MXFP4 A tile [128, 64 B] of aiter's shuffled layout (one
    128-k step), relative to its scalar base (``w4_tile``): row m's group wv = m / 16 is
    (half wv / 4, k step wv % 4) -> (wv / 4) HALF_OFF + (wv % 4) 1024 + (m % 16) 16 + 256 c
    + byte. Steps wv % 4 >= KMAX re-read step KMAX - 1 (their B rows are zero, FP4 has no
    NaN): no masks. Computed once a stage (``ar`` is opaque: no CSE across rounds)."""
    m = w4_rows()[:, None, None]
    c = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(2, W4_L)))[None, :, None]
    b = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, W4_L)))[None, None, :]
    wv = m >> 4
    ks = wv & 3
    if KMAX < 4:
        ks = gl.minimum(ks, KMAX - 1)
    return (wv >> 2) * HALF_OFF + ks * 1024 + (m & 15) * 16 + c * 256 + b


@gluon.jit
def w4_tile(base, off, pred=None):
    """The A tile at scalar ``base`` (every per-round term folded into it, as FlyDSL's
    buffer offsets), lane offsets ``off`` (``w4_off``); ``pred`` (debug only) zeroes."""
    if pred is None:
        v = wload(base, off)
    else:
        v = wload(base, off, (off >= 0) & pred, 0)
    return gl.convert_layout(gl.reshape(v, [128, 64]), A_L)


@gluon.jit
def ug_offs():
    """ug_round's loop-invariant lane offsets: (A tile, scale byte). aiter's shuffle_scale
    byte of (row e 1536 + 16 g + r16, col (4 r + p) 4 + sg), 192 cols, g = 48 (sm / 64) + cg:
    (e 48 + cg / 2) 6144 + r 512 + cg % 2 (scalar) + (sm / 64) 24 6144 + (p / 2) 256 + (p % 2)
    2 + sg 64 + r16 4 (lane), p = (sm / 16) % 4."""
    AS_L: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
    sm = ar(128, layout=gl.SliceLayout(1, AS_L))[:, None]
    sg = ar(4, layout=gl.SliceLayout(0, AS_L))[None, :]
    sp = (sm >> 4) & 3
    sb = (sm >> 6) * (24 * 6144) + (sp >> 1) * 256 + (sp & 1) * 2 + sg * 64 + (sm & 15) * 4
    return w4_off(INTER * (HIDDEN // 2)), sb


@gluon.jit
def dn_offs():
    """down_round's loop-invariant lane offsets: (A tile half 0, half 1, scale half 0, half 1).
    aiter's shuffle_scale byte of (row e 6144 + 32 t + 16 tt + r16, col c 4 + sg), 24 cols,
    c = 4 half + p: (e 192 + t) 768 + half 512 (scalar) + (p / 2) 256 + (p % 2) 2 + sg 64 +
    r16 4 + tt (lane), p = (sm / 16) % 4, tt = sm / 64; chunks past 6 read chunk 5's (finite)
    scale, their B is zero."""
    AS_L: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
    sm = ar(128, layout=gl.SliceLayout(1, AS_L))[:, None]
    sg = ar(4, layout=gl.SliceLayout(0, AS_L))[None, :]
    rest = sg * 64 + (sm & 15) * 4 + (sm >> 6)
    p0 = (sm >> 4) & 3
    p1 = gl.minimum(p0, 1)
    return (w4_off(16 * (INTER // 2), 4), w4_off(16 * (INTER // 2), 2),
            (p0 >> 1) * 256 + (p0 & 1) * 2 + rest, (p1 >> 1) * 256 + (p1 & 1) * 2 + rest)


@gluon.jit
def a_scales(ws, srow, col, cols: gl.constexpr):
    """A's E8M0 codes [128, 4] (one 128-k step) from an aiter shuffle_scale tensor."""
    b = scale_index(srow, col, cols)
    dw = cdna4.buffer_load(i32p(ws), b >> 2)
    return ((dw >> ((b & 3) * 8)) & 0xFF).to(gl.uint8)


@gluon.jit
def rnd(buf, r):
    """Round r ([columns, bytes]) of a 3-D B tile buffer: ``index`` takes a layout one rank
    below the shape (slicing, a full-rank one), so the same bytes seen with the rank-2
    form of its layout first."""
    if buf.type.layout == BSW:
        v = buf._reinterpret(buf.dtype, buf.shape, BSW2)
    else:
        v = buf._reinterpret(buf.dtype, buf.shape, FLAT2)
    return v.index(r)


@gluon.jit
def unit_scales(shape: gl.constexpr, layout: gl.constexpr):
    return gl.full(shape, 127, gl.uint8, cdna4.get_mfma_scale_layout(layout, shape))


@gluon.jit
def bd_reduce(acc, T: gl.constexpr, P: gl.constexpr, sm):
    """A block-diagonal accumulator [128, P MAX_S] (column = token P + p; wave w = t P + p
    owns rows 16 w .. and only its columns . P + p are valid) -> (lo, hi) [8 row pairs,
    MAX_S tokens] a tile (PR_L): rows 2 j, 2 j + 1 of tile t, the K slices summed in order
    p = 0 .. P - 1. Each wave's valid columns are picked in registers first (adding exact
    zeros), so only [128, MAX_S] goes through LDS (``sm``: LDS [128, MAX_S] f32)."""
    NC: gl.constexpr = acc.shape[1]
    NS: gl.constexpr = NC // P
    c = ar(NC, layout=gl.SliceLayout(0, MFMA))[None, :]
    m = ar(128, layout=gl.SliceLayout(1, MFMA))[:, None]
    mine = gl.where(c % P == (m // 16) % P, acc, 0.0)
    red = gl.sum(gl.reshape(mine, [128, NS, P]), axis=2)
    gl.barrier()
    sm.store(red)
    gl.barrier()
    j = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    n = ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :]
    outs_lo = ()
    outs_hi = ()
    for t in gl.static_range(T):
        lo = gl.zeros([8, NS], gl.float32, PR_L)
        hi = gl.zeros([8, NS], gl.float32, PR_L)
        for p in gl.static_range(P):
            lo = lo + sm.gather((t * P + p) * 16 + 2 * j + n * 0, 0)
            hi = hi + sm.gather((t * P + p) * 16 + 2 * j + 1 + n * 0, 0)
        outs_lo = outs_lo + (lo,)
        outs_hi = outs_hi + (hi,)
    return outs_lo, outs_hi


@gluon.jit
def poll_pairs(ptr, idx, tag, pred, errors, bid, code: gl.constexpr):
    """Tagged pairs ``ptr[idx]`` (bf16 pairs) -> (lo, hi) f32; a watchdog error is noted."""
    v, err = pollv1(ptr, idx, tag, pred)
    note_err(errors, bid, err, code)
    return bf_lo(v), bf_hi(v)


@gluon.jit
def peek(ptr, idx, tag, pred):
    """A tagged-pair load issued (``land`` finishes it): issue every peek of a wait group
    before landing any -- each land is an asm memory barrier, so a peek after one is a
    further serial round trip (~0.5-1 us through device-scope memory)."""
    if WAIT2:
        p = pred.to(gl.int32) != 0
        return cdna4.buffer_load(ptr, idx, mask=p, other=0, cache=".cv"), p.to(gl.int32) + idx * 0
    return _peek(ptr, idx, tag, pred)


@gluon.jit
def land(ptr, idx, tag, pk, errors, bid, code: gl.constexpr, SYS: gl.constexpr = False):
    """A ``peek``'s words, polling the elements whose tag had not landed -> i32 values
    (SYS: a peer-written region)."""
    v, need = pk
    if WAIT2:  # a WAIT2 peek carries the element mask
        need = ((need != 0) & ((v >> 32).to(gl.int32) != tag)).to(gl.int32)
    r, err = poll1(addr_of(ptr, idx), tag, need, SYS)
    note_err(errors, bid, err, code)
    return gl.where(need != 0, r, v.to(gl.int32))


@gluon.jit
def wait_items(items, tag, errors, bid, code: gl.constexpr, PACK: gl.constexpr, SYS: gl.constexpr = False):
    """``(ptr, idx, pred, pk)`` items (pk: a ``peek``) landed -> i32 words (a tuple), in one
    asm block (FlyDSL's batched wait: per lane, no CTA barrier). Every item has the same
    shape and layout, PACK = its elements a thread."""
    its = ()
    for i in gl.static_range(len(items)):
        ptr, idx, pred, pk = items[i]
        its = its + ((ptr, idx, pred, pk[0]),)
    if WAIT2:
        ws, err = wait_words2(its, tag, PACK, SYS)
    else:
        ws, err = wait_words(its, tag, PACK, SYS)
    note_err(errors, bid, err, code)
    return ws


@gluon.jit
def sleep1():
    return gl.inline_asm_elementwise("s_sleep 1", "=s", [], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def any_need(need):
    """1 if any element of the CTA's ``need`` is set (a CTA-wide reduction)."""
    if len(need.shape) == 1:
        r = gl.max(need, axis=0)
    elif len(need.shape) == 2:
        r = gl.max(gl.max(need, axis=1), axis=0)
    else:
        r = gl.max(gl.max(gl.max(need, axis=2), axis=1), axis=0)
    return r


@gluon.jit
def wait_group(items, tag, errors, bid, code: gl.constexpr, PACK: gl.constexpr):
    """Every ``(ptr, idx, peek)`` item landed -> their i32 words (a tuple), in one asm block
    (``wait_items``): the elements a peek already saw landed are not checked again. Same
    shape and layout each, PACK = its elements a thread."""
    its = ()
    for i in gl.static_range(len(items)):
        ptr, idx, pk = items[i]
        its = its + ((ptr, idx, pk[1], pk),)
    return wait_items(its, tag, errors, bid, code, PACK)


@gluon.jit
def layer_in_pairs(li, idx, pred, hid_in, res_in, psc, ptag, errors, bid, PACK: gl.constexpr, arsrc=0):
    """The layer's input at pair index ``idx`` (token * HPAIRS + pair): (ar, res) as
    (lo, hi) f32 each -- vLLM's tensors at the first layer, the previous layer's ar /
    hm mailboxes (``psc``: its scratch half, tag ``ptag``) after. Hybrid: ar is this
    launch's all-reduce (tagged words at ``arsrc``), hm the previous launch's (plain)."""
    if HYB:
        if li == 0:
            aw = cdna4.buffer_load(i32p(hid_in), idx, mask=pred, other=0)
            rw = cdna4.buffer_load(i32p(res_in), idx, mask=pred, other=0)
        else:
            rw = cdna4.buffer_load(i32p(rgn(psc, R_HM)), idx, mask=pred, other=0)
            aw = wait_group(((arsrc, idx, peek(arsrc, idx, ptag, pred)),), ptag, errors, bid, E_X, PACK)[0]
        a_lo = bf_lo(aw)
        a_hi = bf_hi(aw)
        r_lo = bf_lo(rw)
        r_hi = bf_hi(rw)
    elif PLAIN_IN:
        pa = i32p(rgn(psc, R_AR))
        pr = i32p(rgn(psc, R_HM))
        if li == 0:
            pa = i32p(hid_in)
            pr = i32p(res_in)
        aw = cdna4.buffer_load(pa, idx, mask=pred, other=0)
        rw = cdna4.buffer_load(pr, idx, mask=pred, other=0)
        a_lo = bf_lo(aw)
        a_hi = bf_hi(aw)
        r_lo = bf_lo(rw)
        r_hi = bf_hi(rw)
    elif li == 0:
        aw = cdna4.buffer_load(i32p(hid_in), idx, mask=pred, other=0)
        rw = cdna4.buffer_load(i32p(res_in), idx, mask=pred, other=0)
        a_lo = bf_lo(aw)
        a_hi = bf_hi(aw)
        r_lo = bf_lo(rw)
        r_hi = bf_hi(rw)
    else:
        ws = wait_group(((rgn(psc, R_AR), idx, peek(rgn(psc, R_AR), idx, ptag, pred)),
                         (rgn(psc, R_HM), idx, peek(rgn(psc, R_HM), idx, ptag, pred))), ptag, errors, bid, E_X, PACK)
        a_lo = bf_lo(ws[0])
        a_hi = bf_hi(ws[0])
        r_lo = bf_lo(ws[1])
        r_hi = bf_hi(ws[1])
    return a_lo, a_hi, r_lo, r_hi


# ============================================================ qkv GEMV (+ input norm)
# x tiles [S, steps, 64 pairs] (pair (k, st, q) = elements st * 128 + 2 q, + 1), the 48
# 128-k steps of a row as a 32-step part and a 16-step part (shapes are powers of two)
XL3 = gl.constexpr(gl.BlockedLayout([1, 1, 2], [1, 2, 32], [1, 8, 1], [2, 1, 0]))


@gluon.jit
def qkv_in(li, st0: gl.constexpr, NST: gl.constexpr, SP: gl.constexpr, S: gl.constexpr, hid_in, res_in, psc,
           ptag, errors, bid, g_in, k0=0, arsrc=0):
    """Steps st0 .. st0 + NST of every token's input row, loaded and polled at once: (x =
    ar + res lo / hi, gamma pairs, pair index, live mask)."""
    k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL3)))[None, :, None]
    q3 = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL3)))[None, None, :]
    ok = (k3 < S) & (st3 < 48)
    pidx = (k0 + k3) * HPAIRS + st3 * 64 + q3
    a_lo, a_hi, r_lo, r_hi = layer_in_pairs(li, pidx, ok, hid_in, res_in, psc, ptag, errors, bid, SP * (NST // 16),
                                            arsrc)
    gw = cdna4.buffer_load(g_in, st3 * 64 + q3 + k3 * 0, mask=ok, other=0)
    return a_lo + r_lo, a_hi + r_hi, gw, pidx, ok


@gluon.jit
def qkv_in_issue(li, st0: gl.constexpr, NST: gl.constexpr, SP: gl.constexpr, S: gl.constexpr, hid_in, res_in,
                 psc, ptag, g_in):
    """Steps st0 .. st0 + NST of every token's input row, loads issued only: (ar words,
    res words, their not-yet-landed masks, gamma pairs, pair index, live mask). vLLM's
    tensors at the first layer, else one coherent load of the previous layer's ar / hm
    pairs (``qkv_in_finish`` polls whatever had not landed)."""
    k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL3)))[None, :, None]
    q3 = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL3)))[None, None, :]
    ok = (k3 < S) & (st3 < 48)
    pidx = k3 * HPAIRS + st3 * 64 + q3
    # both sources issued, predicated (a branch around the loads would wait for them
    # inside it, before the weights go out)
    first = li == 0
    if PLAIN_IN:
        pa = i32p(rgn(psc, R_AR))
        pr = i32p(rgn(psc, R_HM))
        if first:
            pa = i32p(hid_in)
            pr = i32p(res_in)
        hw = cdna4.buffer_load(pa, pidx, mask=ok, other=0)
        hr = cdna4.buffer_load(pr, pidx, mask=ok, other=0)
        av = hw
        rv = hr
    else:
        hw = cdna4.buffer_load(i32p(hid_in), pidx, mask=ok & first, other=0)
        hr = cdna4.buffer_load(i32p(res_in), pidx, mask=ok & first, other=0)
        av = cdna4.buffer_load(rgn(psc, R_AR), pidx, mask=ok & (li != 0), other=0, cache=".cv")
        rv = cdna4.buffer_load(rgn(psc, R_HM), pidx, mask=ok & (li != 0), other=0, cache=".cv")
    gw = cdna4.buffer_load(g_in, st3 * 64 + q3 + k3 * 0, mask=ok, other=0)
    return (hw, hr), (av, rv), gw, pidx, ok


@gluon.jit
def qkv_in_finish(li, hin, mail, ok, pidx, psc, ptag, errors, bid, PACK: gl.constexpr):
    """x = ar + res (lo, hi): vLLM's tensors at the first layer, else the mailbox pairs
    (any that had not landed polled)."""
    hw, hr = hin
    av, rv = mail
    if PLAIN_IN:
        aw = hw
        rw = hr
    elif li == 0:
        aw = hw
        rw = hr
    else:
        if WAIT2:  # the element mask (the loads were masked by it)
            an = ok.to(gl.int32)
            rn = ok.to(gl.int32)
        else:
            an = (ok & ((av >> 32).to(gl.int32) != ptag)).to(gl.int32)
            rn = (ok & ((rv >> 32).to(gl.int32) != ptag)).to(gl.int32)
        ws = wait_group(((rgn(psc, R_AR), pidx, (av, an)), (rgn(psc, R_HM), pidx, (rv, rn))), ptag, errors, bid, E_X,
                        PACK)
        aw = ws[0]
        rw = ws[1]
    return bf_lo(aw) + bf_lo(rw), bf_hi(aw) + bf_hi(rw)


@gluon.jit
def sum3(x):
    return gl.sum(gl.sum(x, axis=2), axis=1)


@gluon.jit
def max3(x):
    return gl.max(gl.max(x, axis=2), axis=1)


@gluon.jit
def div_mk(x, d, r):
    """``x / d`` correctly rounded given ``r`` = RN(1 / d), one true division per divisor
    (FlyDSL's div_rn, Markstein): 3 VALU a quotient, not the IEEE expansion's ~10."""
    q = x * r
    return gl.fma(gl.fma(-q, d, x), r, q)


@gluon.jit
def qkv_quant(x_lo, x_hi, gw, rstd, x_scale, xs, NST: gl.constexpr, SP: gl.constexpr):
    # xs: LDS [NST / 8 rounds, 8 MAX_S columns (token * 8 + step), 128]
    """A part's normed rows -> FP8 (x / x_scale, correctly rounded) -> its LDS buffer."""
    r3 = rstd[:, None, None]
    n_lo = x_lo * r3 * (bf_lo(gw) + 1.0)
    n_hi = x_hi * r3 * (bf_hi(gw) + 1.0)
    xs3 = x_scale[:, None, None]
    xr3 = gl.div_rn(gl.full(x_scale.shape, 1.0, gl.float32, x_scale.type.layout), x_scale)[:, None, None]
    q_lo = div_mk(n_lo, xs3, xr3).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    q_hi = div_mk(n_hi, xs3, xr3).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    x8 = gl.permute(gl.reshape(gl.join(q_lo, q_hi), [SP, NST // 8, 8, 128]), (1, 0, 2, 3))
    xs.slice(0, SP * 8, dim=1).store(gl.reshape(x8, [NST // 8, SP * 8, 128]))


@gluon.jit
def qkv_fold(pa, pb, MAX: gl.constexpr):
    """[SP, 32 steps] and [SP, 16 steps] per-token partials (steps over 2 lanes x 8 warps: the
    32-step part's halves sit in one thread) -> [SP]: the halves and the parts combined in
    registers first, then one cross-warp reduction (FlyDSL's single set of barriers)."""
    SP: gl.constexpr = pa.shape[0]
    a2 = gl.reshape(pa, [SP, 2, 16])
    if MAX:
        t = gl.max(a2, axis=1)
        t = gl.maximum(t, gl.convert_layout(pb, t.type.layout))
        r = gl.max(t, axis=1)
    else:
        t = gl.sum(a2, axis=1)
        t = t + gl.convert_layout(pb, t.type.layout)
        r = gl.sum(t, axis=1)
    # back in the parts' token layout (every thread holds every token's total: no data moves)
    return gl.convert_layout(r, gl.SliceLayout(1, pb.type.layout))


@gluon.jit
def _sum_max(a0, a1, b0, b1):
    return a0 + b0, gl.maximum(a1, b1)


@gluon.jit
def qkv_fold2(sa, sb, ma, mb):
    """qkv_fold of a sum (sa, sb) and a max (ma, mb) at once: one cross-warp reduction."""
    SP: gl.constexpr = sa.shape[0]
    s = gl.sum(gl.reshape(sa, [SP, 2, 16]), axis=1)
    m = gl.max(gl.reshape(ma, [SP, 2, 16]), axis=1)
    s = s + gl.convert_layout(sb, s.type.layout)
    m = gl.maximum(m, gl.convert_layout(mb, m.type.layout))
    rs, rm = gl.reduce((s, m), 1, _sum_max)
    OL: gl.constexpr = gl.SliceLayout(1, sb.type.layout)
    return gl.convert_layout(rs, OL), gl.convert_layout(rm, OL)


@gluon.jit
def qkv_absg(x_lo, x_hi, gw):
    """|x g'| of a part (g' = gamma + 1), its pair halves' and lanes' max: [SP, steps]."""
    return gl.max(gl.maximum(gl.abs(x_lo * (bf_lo(gw) + 1.0)), gl.abs(x_hi * (bf_hi(gw) + 1.0))), axis=2)


@gluon.jit
def qkv_absn(x_lo, x_hi, gw, rstd):
    """|normed x| of a part, its pair halves' max: [SP, steps] after the lanes' max."""
    r3 = rstd[:, None, None]
    n_lo = x_lo * r3 * (bf_lo(gw) + 1.0)
    n_hi = x_hi * r3 * (bf_hi(gw) + 1.0)
    return gl.max(gl.maximum(gl.abs(n_lo), gl.abs(n_hi)), axis=2)


@gluon.jit
def qkv_amax(x_lo, x_hi, gw, rstd):
    r3 = rstd[:, None, None]
    n_lo = x_lo * r3 * (bf_lo(gw) + 1.0)
    n_hi = x_hi * r3 * (bf_hi(gw) + 1.0)
    return max3(gl.maximum(gl.abs(n_lo), gl.abs(n_hi)))


@gluon.jit
def x8_words(x_lo, x_hi, gw, rstd, x_scale, k, st0: gl.constexpr, sc, tag, TAGW: gl.constexpr = False):
    """Norm task: a part [1, NST, 64 pairs] of token k's normed row -> FP8 (as qkv_quant)
    -> words (4 fp8 a pair: pairs 2 j, 2 j + 1) at step st0 + st, word j of "x8"."""
    NST: gl.constexpr = x_lo.shape[1]
    r3 = rstd[:, None, None]
    n_lo = x_lo * r3 * (bf_lo(gw) + 1.0)
    n_hi = x_hi * r3 * (bf_hi(gw) + 1.0)
    xs3 = x_scale[:, None, None]
    xr3 = gl.div_rn(gl.full(x_scale.shape, 1.0, gl.float32, x_scale.type.layout), x_scale)[:, None, None]
    q_lo = div_mk(n_lo, xs3, xr3).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    q_hi = div_mk(n_hi, xs3, xr3).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    a, c = gl.split(gl.reshape(q_lo | (q_hi << 8), [1, NST, 32, 2]))
    word = a | (c << 16)
    WL: gl.constexpr = word.type.layout
    st = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, WL)))[None, :, None]
    j = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, WL)))[None, None, :]
    if PLAIN_HO and (not TAGW):
        cdna4.buffer_store(word, i32p(rgn(sc, R_X8)), k * (HIDDEN // 4) + st * 32 + j, cache=".wt")
    else:
        put(rgn(sc, R_X8), k * (HIDDEN // 4) + st * 32 + j, word, tag)


@gluon.jit
def p3_idx(k, st0: gl.constexpr):
    """Norm task part (16 steps from st0) of token k in XL3 [1, 16, 64]: (pair index, step, pair)."""
    st3 = st0 + ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL3)))[None, :, None]
    q3 = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL3)))[None, None, :]
    k3 = ar(1, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))[:, None, None]
    return (k + k3) * HPAIRS + st3 * 64 + q3, st3 * 64 + q3 + k3 * 0


@gluon.jit
def stage_norm_p3(k, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, arsrc, tr_, ar_out=0,
                  DN: gl.constexpr = False):
    """Norm task, hybrid past 16 rows: token k's row as three 16-step parts whose loads are all
    issued before one batched wait (two parts loaded a wait each were two serial round trips),
    then stage_norm's math on the parts."""
    stamp(tr_, bid, 49)
    g_in = lp(lrow(ptab, li, (2, 0))[0], gl.int32)
    i0, g0i = p3_idx(k, 0)
    i1, g1i = p3_idx(k, 16)
    i2, g2i = p3_idx(k, 32)
    ok = i0 >= 0
    if li == 0:
        a0 = cdna4.buffer_load(i32p(hid_in), i0)
        a1 = cdna4.buffer_load(i32p(hid_in), i1)
        a2 = cdna4.buffer_load(i32p(hid_in), i2)
        r0 = cdna4.buffer_load(i32p(res_in), i0)
        r1 = cdna4.buffer_load(i32p(res_in), i1)
        r2 = cdna4.buffer_load(i32p(res_in), i2)
    else:
        r0 = cdna4.buffer_load(i32p(rgn(psc, R_HM)), i0)
        r1 = cdna4.buffer_load(i32p(rgn(psc, R_HM)), i1)
        r2 = cdna4.buffer_load(i32p(rgn(psc, R_HM)), i2)
        if DN:
            # the dense FFN all-reduce (fagd, ``arsrc``): this task its one reader -- it resets
            # the slots, and writes an aux layer's ar out (ffn_tail's, tagged)
            if DENSE_1S:
                # one-shot: ``arsrc`` is ffnd, [src rank][token][pair]: the 4 partials of each
                # word summed in rank order (the owner's sum)
                ii = (i0, i1, i2)
                pks = ()
                for j in gl.static_range(3):
                    for r in gl.static_range(4):
                        pks = pks + (peek_dense(arsrc, r * (MAX_S * HPAIRS) + ii[j], ok),)
                its = ()
                for j in gl.static_range(3):
                    for r in gl.static_range(4):
                        its = its + ((arsrc, r * (MAX_S * HPAIRS) + ii[j], pks[4 * j + r]),)
                pw_, err = wait_dense(its, 2)
                note_err(errors, bid, err, E_X)
                for j in gl.static_range(3):
                    for r in gl.static_range(4):
                        reset_dense(arsrc, r * (MAX_S * HPAIRS) + ii[j])
                aw = ()
                for j in gl.static_range(3):
                    s_lo = bf_lo(pw_[4 * j])
                    s_hi = bf_hi(pw_[4 * j])
                    for r in gl.static_range(1, 4):
                        s_lo = s_lo + bf_lo(pw_[4 * j + r])
                        s_hi = s_hi + bf_hi(pw_[4 * j + r])
                    aw = aw + (bf_pair(s_lo, s_hi),)
            else:
                p0 = peek_dense(arsrc, i0, ok)
                p1 = peek_dense(arsrc, i1, ok)
                p2 = peek_dense(arsrc, i2, ok)
                aw, err = wait_dense(((arsrc, i0, p0), (arsrc, i1, p1), (arsrc, i2, p2)), 2)
                note_err(errors, bid, err, E_X)
                reset_dense(arsrc, i0)
                reset_dense(arsrc, i1)
                reset_dense(arsrc, i2)
            aslot = lrow(ptab, li - 1, (19, 0))[0].to(gl.int32) >> 8
            if NOAUX:
                aslot = 0
            if aslot > 0:
                cdna4.buffer_store(aw[0], i32p(ar_out), aslot * (MAX_S * HPAIRS) + i0)
                cdna4.buffer_store(aw[1], i32p(ar_out), aslot * (MAX_S * HPAIRS) + i1)
                cdna4.buffer_store(aw[2], i32p(ar_out), aslot * (MAX_S * HPAIRS) + i2)
        else:
            p0 = peek(arsrc, i0, ptag, ok)
            p1 = peek(arsrc, i1, ptag, ok)
            p2 = peek(arsrc, i2, ptag, ok)
            aw = wait_group(((arsrc, i0, p0), (arsrc, i1, p1), (arsrc, i2, p2)), ptag, errors, bid, E_X, 2)
        a0 = aw[0]
        a1 = aw[1]
        a2 = aw[2]
    ga = cdna4.buffer_load(g_in, g0i)
    gb = cdna4.buffer_load(g_in, g1i)
    gc = cdna4.buffer_load(g_in, g2i)
    x0_lo = bf_lo(a0) + bf_lo(r0)
    x0_hi = bf_hi(a0) + bf_hi(r0)
    x1_lo = bf_lo(a1) + bf_lo(r1)
    x1_hi = bf_hi(a1) + bf_hi(r1)
    x2_lo = bf_lo(a2) + bf_lo(r2)
    x2_hi = bf_hi(a2) + bf_hi(r2)
    stamp(tr_, bid, 50)
    # 16-step parts: steps over 2 lanes x 8 waves, nothing in-thread to fold first
    ssp = (gl.sum(x0_lo * x0_lo + x0_hi * x0_hi, axis=2) + gl.sum(x1_lo * x1_lo + x1_hi * x1_hi, axis=2)
           + gl.sum(x2_lo * x2_lo + x2_hi * x2_hi, axis=2))
    if NORM_1R:
        # one cross-warp reduction for both: amax(|x rstd g'|) = rstd amax(|x g'|) (rounding aside)
        amp = gl.maximum(gl.maximum(qkv_absg(x0_lo, x0_hi, ga), qkv_absg(x1_lo, x1_hi, gb)),
                         qkv_absg(x2_lo, x2_hi, gc))
        ss, am = gl.reduce((ssp, amp), 1, _sum_max)
        rstd = hw_rsq(ss / 6144.0 + eps)
        stamp(tr_, bid, 53)
        amax = am * rstd
    else:
        rstd = hw_rsq(gl.sum(ssp, axis=1) / 6144.0 + eps)
        stamp(tr_, bid, 53)
        amax = gl.max(gl.maximum(gl.maximum(qkv_absn(x0_lo, x0_hi, ga, rstd), qkv_absn(x1_lo, x1_hi, gb, rstd)),
                                 qkv_absn(x2_lo, x2_hi, gc, rstd)), axis=1)
    x_scale = gl.where(amax == 0.0, 1.0, gl.div_rn(amax, gl.full(amax.shape, 448.0, gl.float32, amax.type.layout)))
    stamp(tr_, bid, 54)
    x8_words(x0_lo, x0_hi, ga, rstd, x_scale, k, 0, sc, tag)
    x8_words(x1_lo, x1_hi, gb, rstd, x_scale, k, 16, sc, tag)
    x8_words(x2_lo, x2_hi, gc, rstd, x_scale, k, 32, sc, tag)
    stamp(tr_, bid, 55)
    lt = ar(THREADS, T1)
    if PLAIN_HO:
        lt = drain(lt)  # the plain words stored before their flag
    put(rgn(sc, R_X8S), k + lt * 0, gl.max(x_scale, axis=0) + lt * 0.0, tag, mask=lt == 0)
    stamp(tr_, bid, 51)
    put(rgn(sc, R_H), i0, bf_pair(x0_lo, x0_hi), tag)
    put(rgn(sc, R_H), i1, bf_pair(x1_lo, x1_hi), tag)
    put(rgn(sc, R_H), i2, bf_pair(x2_lo, x2_hi), tag)


@gluon.jit
def stage_norm(k, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, arsrc=0, tr_=None):
    """Norm task (S > 2, FlyDSL's k1 norm tasks): token k's h = bf16(ar + res) -> "h", its
    RMSNorm + per-token FP8 once -> "x8" words, then its scale -> "x8s" (the GEMV tasks
    otherwise each normalize every token)."""
    if NORM_W and PLAIN_IN and (not HYB):
        stage_norm_w(k, li, ptab, hid_in, res_in, sc, psc, tag, bid, eps, tr_)
        return
    stamp(tr_, bid, 49)
    g_in = lp(lrow(ptab, li, (2, 0))[0], gl.int32)
    xa_lo, xa_hi, ga, pa, oka = qkv_in(li, 0, 32, 1, 1, hid_in, res_in, psc, ptag, errors, bid, g_in, k, arsrc)
    xb_lo, xb_hi, gb, pb, okb = qkv_in(li, 32, 16, 1, 1, hid_in, res_in, psc, ptag, errors, bid, g_in, k, arsrc)
    stamp(tr_, bid, 50)
    if NORM_1R:
        # one cross-warp reduction for both: amax(|x rstd g'|) = rstd amax(|x g'|) (rounding aside)
        ss, am = qkv_fold2(gl.sum(xa_lo * xa_lo + xa_hi * xa_hi, axis=2),
                           gl.sum(xb_lo * xb_lo + xb_hi * xb_hi, axis=2),
                           qkv_absg(xa_lo, xa_hi, ga), qkv_absg(xb_lo, xb_hi, gb))
        rstd = hw_rsq(ss / 6144.0 + eps)
        stamp(tr_, bid, 53)
        amax = am * rstd
    else:
        ss = qkv_fold(gl.sum(xa_lo * xa_lo + xa_hi * xa_hi, axis=2), gl.sum(xb_lo * xb_lo + xb_hi * xb_hi, axis=2),
                      False)
        rstd = hw_rsq(ss / 6144.0 + eps)
        stamp(tr_, bid, 53)
        amax = qkv_fold(qkv_absn(xa_lo, xa_hi, ga, rstd), qkv_absn(xb_lo, xb_hi, gb, rstd), True)
    x_scale = gl.where(amax == 0.0, 1.0, gl.div_rn(amax, gl.full(amax.shape, 448.0, gl.float32, amax.type.layout)))
    stamp(tr_, bid, 54)
    # the qkv tasks' words and scale first, h (the router's, a stage later) after
    x8_words(xa_lo, xa_hi, ga, rstd, x_scale, k, 0, sc, tag, X8_TAG)
    x8_words(xb_lo, xb_hi, gb, rstd, x_scale, k, 32, sc, tag, X8_TAG)
    stamp(tr_, bid, 55)
    lt = ar(THREADS, T1)
    if X8_TAG:
        pass  # tagged words: the scale needs no drain ahead of it
    elif PLAIN_HO:
        lt = drain(lt)  # the plain words stored before their flag
    elif not NORM_NB:
        gl.barrier()
    # (NORM_NB: no barrier -- the words are tagged and the qkv tasks wait on each; the scale
    # only gates their spinning)
    put(rgn(sc, R_X8S), k + lt * 0, gl.max(x_scale, axis=0) + lt * 0.0, tag, mask=lt == 0)
    stamp(tr_, bid, 51)
    put(rgn(sc, R_H), pa, bf_pair(xa_lo, xa_hi), tag, mask=oka)
    put(rgn(sc, R_H), pb, bf_pair(xb_lo, xb_hi), tag, mask=okb)


NW_L = gl.constexpr(gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0]))  # [1024 chunks of 8 elements, 4 pairs]


@gluon.jit
def stage_norm_w(k, li, ptab, hid_in, res_in, sc, psc, tag, bid, eps, tr_=None):
    """``stage_norm`` (PLAIN_IN, not hybrid) with the row as 16 B a thread: chunk c (elements
    8 c .. + 8) of 768, two a thread; the sum of squares and the amax as flat CTA reductions.
    Same values: x = ar + res (f32), rstd, n = x rstd g', amax, x_scale, FP8 words, h."""
    stamp(tr_, bid, 49)
    g_in = lp(lrow(ptab, li, (2, 0))[0], gl.int32)
    c = ar(1024, layout=gl.SliceLayout(1, NW_L))[:, None]
    j = ar(4, layout=gl.SliceLayout(0, NW_L))[None, :]
    ok = (c < HIDDEN // 8) & (j < 4)
    pc = c * 4 + j
    pa = i32p(rgn(psc, R_AR))
    pr = i32p(rgn(psc, R_HM))
    if li == 0:
        pa = i32p(hid_in)
        pr = i32p(res_in)
    aw = cdna4.buffer_load(pa, k * HPAIRS + pc, mask=ok, other=0)
    rw = cdna4.buffer_load(pr, k * HPAIRS + pc, mask=ok, other=0)
    gw = cdna4.buffer_load(g_in, pc, mask=ok, other=0)
    stamp(tr_, bid, 50)
    x_lo = bf_lo(aw) + bf_lo(rw)
    x_hi = bf_hi(aw) + bf_hi(rw)
    ss = gl.sum(gl.sum(x_lo * x_lo + x_hi * x_hi, axis=1), axis=0)
    rr = hw_rsq(gl.zeros_like(x_lo) + ss / 6144.0 + eps)
    stamp(tr_, bid, 53)
    n_lo = x_lo * rr * (bf_lo(gw) + 1.0)
    n_hi = x_hi * rr * (bf_hi(gw) + 1.0)
    amax = gl.max(gl.max(gl.maximum(gl.abs(n_lo), gl.abs(n_hi)), axis=1), axis=0)
    am = gl.zeros_like(x_lo) + amax
    xs = gl.where(am == 0.0, 1.0, gl.div_rn(am, gl.full(am.shape, 448.0, gl.float32, NW_L)))
    xr = gl.div_rn(gl.full(am.shape, 1.0, gl.float32, NW_L), xs)
    stamp(tr_, bid, 54)
    q_lo = div_mk(n_lo, xs, xr).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    q_hi = div_mk(n_hi, xs, xr).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    a, b = gl.split(gl.reshape(q_lo | (q_hi << 8), [1024, 2, 2]))
    word = a | (b << 16)  # [1024, 2]: pairs 2 w, 2 w + 1 of the chunk
    WL: gl.constexpr = word.type.layout
    cw = ar(1024, layout=gl.SliceLayout(1, WL))[:, None]
    ww = ar(2, layout=gl.SliceLayout(0, WL))[None, :]
    if X8_TAG:
        put(rgn(sc, R_X8), k * (HIDDEN // 4) + cw * 2 + ww, word, tag, mask=(cw < HIDDEN // 8) & (ww < 2))
        lt = ar(THREADS, T1)
    else:
        cdna4.buffer_store(word, i32p(rgn(sc, R_X8)), k * (HIDDEN // 4) + cw * 2 + ww,
                           mask=(cw < HIDDEN // 8) & (ww < 2), cache=".wt")
        lt = drain(ar(THREADS, T1))  # the plain words stored before their flag
    stamp(tr_, bid, 55)
    a1 = gl.zeros([THREADS], gl.float32, T1) + amax
    xs1 = gl.where(a1 == 0.0, 1.0, gl.div_rn(a1, gl.full(a1.shape, 448.0, gl.float32, T1)))
    put(rgn(sc, R_X8S), k + lt * 0, xs1, tag, mask=lt == 0)
    stamp(tr_, bid, 51)
    put(rgn(sc, R_H), k * HPAIRS + pc, bf_pair(x_lo, x_hi), tag, mask=ok)


XN8 = gl.constexpr(gl.BlockedLayout([1, 1, 2], [1, 2, 32], [8, 1, 1], [2, 1, 0]))  # [8 tokens (a wave each), steps, 64 pairs]


@gluon.jit
def n8_idx(st0: gl.constexpr, NST: gl.constexpr, k0, S: gl.constexpr):
    """Wave norm part indices in XN8: (token [8,1,1], step [1,NST,1], pair [1,1,64], live)."""
    k3 = k0 + ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, XN8)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XN8)))[None, :, None]
    q3 = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, XN8)))[None, None, :]
    return k3, st3, q3, (k3 < S) & (st3 < 48)


@gluon.jit
def norm8_in(li, st0: gl.constexpr, NST: gl.constexpr, k0, S: gl.constexpr, hid_in, res_in, psc, ptag, errors, bid,
             arsrc, sc, tag):
    """Wave norm: tokens k0 .. + 8 (one a wave), steps st0 .. + NST: (h = bf16(ar + res)
    pairs, the f32 sum's squares a token). The row is kept as h and normalized from it (vLLM's
    fused add + RMSNorm: variance of the f32 sum, the stored bf16 residual normalized)."""
    k3, st3, q3, ok = n8_idx(st0, NST, k0, S)
    a_lo, a_hi, r_lo, r_hi = layer_in_pairs(li, k3 * HPAIRS + st3 * 64 + q3, ok, hid_in, res_in, psc, ptag, errors,
                                            bid, NST, arsrc)
    x_lo = a_lo + r_lo
    x_hi = a_hi + r_hi
    hw = bf_pair(x_lo, x_hi)
    # h out at once (the router's): its addresses die here, not live through the task
    put(rgn(sc, R_H), k3 * HPAIRS + st3 * 64 + q3, hw, tag, mask=ok)
    return hw, gl.sum(gl.sum(x_lo * x_lo + x_hi * x_hi, axis=2), axis=1)


@gluon.jit
def n8_gamma(g_in, st0: gl.constexpr, NST: gl.constexpr):
    """gamma + 1 (lo, hi) of a part, [1, NST, 64] (reloaded where used: cache hits, not registers)."""
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XN8)))[None, :, None]
    q3 = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, XN8)))[None, None, :]
    gw = gl.load(g_in + st3 * 64 + q3)
    return bf_lo(gw) + 1.0, bf_hi(gw) + 1.0


@gluon.jit
def norm8_absn(hw, g_in, st0: gl.constexpr, rstd):
    NST: gl.constexpr = hw.shape[1]
    g_lo, g_hi = n8_gamma(g_in, st0, NST)
    r3 = rstd[:, None, None]
    return gl.max(gl.max(gl.maximum(gl.abs(bf_lo(hw) * r3 * g_lo), gl.abs(bf_hi(hw) * r3 * g_hi)), axis=2), axis=1)


@gluon.jit
def norm8_words(hw, g_in, rstd, x_scale, k0, st0: gl.constexpr, S: gl.constexpr, sc, tag):
    """Wave norm: the part's normed rows -> FP8 (n / x_scale, as x8_words) -> "x8" words."""
    NST: gl.constexpr = hw.shape[1]
    g_lo, g_hi = n8_gamma(g_in, st0, NST)
    r3 = rstd[:, None, None]
    n_lo = bf_lo(hw) * r3 * g_lo
    n_hi = bf_hi(hw) * r3 * g_hi
    xs3 = x_scale[:, None, None]
    xr3 = gl.div_rn(gl.full(x_scale.shape, 1.0, gl.float32, x_scale.type.layout), x_scale)[:, None, None]
    q_lo = div_mk(n_lo, xs3, xr3).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    q_hi = div_mk(n_hi, xs3, xr3).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    a, c = gl.split(gl.reshape(q_lo | (q_hi << 8), [8, NST, 32, 2]))
    word = a | (c << 16)
    WL: gl.constexpr = word.type.layout
    k3 = k0 + ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, WL)))[:, None, None]
    st = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, WL)))[None, :, None]
    j = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, WL)))[None, None, :]
    if PLAIN_HO:  # plain words, the scale after a drain (the qkv tasks read them past its gate)
        cdna4.buffer_store(word, i32p(rgn(sc, R_X8)), k3 * (HIDDEN // 4) + st * 32 + j, mask=k3 < S, cache=".wt")
    else:
        put(rgn(sc, R_X8), k3 * (HIDDEN // 4) + st * 32 + j, word, tag, mask=k3 < S)


@gluon.jit
def wave_drain(x):
    """This wave's outstanding stores complete (no CTA barrier): ``x`` carries the order."""
    return gl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=v,0,~{memory}", [x], dtype=x.dtype, is_pure=False,
                                     pack=1)


@gluon.jit
def stage_norm8(kb, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, arsrc, S: gl.constexpr,
                tr_=None):
    """Norm task past 16 rows: tokens 8 kb .. + 8, one a wave -- every reduction within its
    wave (lane shuffles: no LDS, no CTA barrier), each wave publishing on its own."""
    stamp(tr_, bid, 49)
    k0 = 8 * kb
    g_in = lp(lrow(ptab, li, (2, 0))[0], gl.int32)
    # three 16-step parts (a batched wait's addresses and tagged words: 16 a lane each)
    h0, s0 = norm8_in(li, 0, 16, k0, S, hid_in, res_in, psc, ptag, errors, bid, arsrc, sc, tag)
    h1, s1 = norm8_in(li, 16, 16, k0, S, hid_in, res_in, psc, ptag, errors, bid, arsrc, sc, tag)
    h2, s2 = norm8_in(li, 32, 16, k0, S, hid_in, res_in, psc, ptag, errors, bid, arsrc, sc, tag)
    stamp(tr_, bid, 50)
    rstd = hw_rsq((s0 + s1 + s2) / 6144.0 + eps)
    stamp(tr_, bid, 53)
    amax = gl.maximum(gl.maximum(norm8_absn(h0, g_in, 0, rstd), norm8_absn(h1, g_in, 16, rstd)),
                      norm8_absn(h2, g_in, 32, rstd))
    x_scale = gl.where(amax == 0.0, 1.0, gl.div_rn(amax, gl.full(amax.shape, 448.0, gl.float32, amax.type.layout)))
    stamp(tr_, bid, 54)
    norm8_words(h0, g_in, rstd, x_scale, k0, 0, S, sc, tag)
    norm8_words(h1, g_in, rstd, x_scale, k0, 16, S, sc, tag)
    norm8_words(h2, g_in, rstd, x_scale, k0, 32, S, sc, tag)
    stamp(tr_, bid, 55)
    kk = k0 + ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, XN8)))
    if PLAIN_HO:
        kk = wave_drain(kk)  # the wave's plain words stored before its scale (vmcnt is the wave's)
    put(rgn(sc, R_X8S), kk, x_scale, tag, mask=kk < S)
    stamp(tr_, bid, 51)


@gluon.jit
def x8_part(sc, tag, errors, bid, xs8, st0: gl.constexpr, NST: gl.constexpr, SP: gl.constexpr, S: gl.constexpr,
            k0=0, TAGW: gl.constexpr = False):
    """GEMV task: tokens k0 .. k0 + SP's FP8 words of steps st0 .. + NST -> LDS [NST / 8
    rounds, 8 SP columns (token * 8 + step), 128] (qkv_quant's layout)."""
    k3 = k0 + ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XW3)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XW3)))[None, :, None]
    w3 = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, XW3)))[None, None, :]
    iw = k3 * (HIDDEN // 4) + st3 * 32 + w3
    if PLAIN_HO and (not TAGW):  # (the scales' gate passed: the words are in)
        wv = cdna4.buffer_load(i32p(rgn(sc, R_X8)), iw, mask=(k3 < S) & (w3 >= 0), other=0, cache=".cv")
    else:
        pkw = peek(rgn(sc, R_X8), iw, tag, k3 < S)
        wv = wait_group(((rgn(sc, R_X8), iw, pkw),), tag, errors, bid, E_X8, SP * NST // 16)[0]
    b0 = (wv & 255).to(gl.uint8)
    b1 = ((wv >> 8) & 255).to(gl.uint8)
    b2 = ((wv >> 16) & 255).to(gl.uint8)
    b3 = ((wv >> 24) & 255).to(gl.uint8)
    x = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [SP, NST // 8, 8, 128]), (1, 0, 2, 3))
    xs8.slice(0, SP * 8, dim=1).store(gl.reshape(x, [NST // 8, SP * 8, 128]))


@gluon.jit
def w8_tile3(base, row_term):
    """``w8_tile`` as the batched MFMA's A [8 waves, 16 rows, 128 k] (A3)."""
    c = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, W8_L)))[None, :, None]
    b = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, W8_L)))[None, None, :]
    v = wload(base, row_term[:, None, None] + c * 256 + b)
    return gl.convert_layout(gl.reshape(v, [8, 16, 128]), A3)


@gluon.jit
def x8_b3(sc, k0, r: gl.constexpr, S: gl.constexpr):
    """Round r's B of tokens k0 .. + 16: wave w's 128-k slice (step 8 r + w) of each token's x8
    row as int64 words of 128-B lines [128 = (wave, token), 16 words] (G64; B3 once split
    into bytes), loads issued only."""
    wn = ar(128, layout=gl.SliceLayout(1, G64))[:, None]
    q = ar(16, layout=gl.SliceLayout(0, G64))[None, :]
    tok = k0 + wn % 16
    idx = tok * (HIDDEN // 8) + (8 * r + wn // 16) * 16 + q
    if QKV_DBG & 1:  # (diagnostics: no x8 loads)
        return (idx * 0 + 1).to(gl.int64)
    if QKV_DBG & 16:  # (diagnostics: the same bytes as contiguous lines -- wrong data)
        idx = k0 * (HIDDEN // 8) + r * 2048 + wn * 16 + q
    return cdna4.buffer_load(rgn(sc, R_X8), idx, mask=(tok < S) & (q >= 0), other=0, cache=".cv")


@gluon.jit
def x8_b3b(sc, k0, r: gl.constexpr, S: gl.constexpr):
    """``x8_b3`` loaded as bytes [128 = (wave, token), 128] (G8B): B3 by a reshape + permute."""
    wn = ar(128, layout=gl.SliceLayout(1, G8B))[:, None]
    by = ar(128, layout=gl.SliceLayout(0, G8B))[None, :]
    tok = k0 + wn % 16
    idx = tok * HIDDEN + (8 * r + wn // 16) * 128 + by
    return cdna4.buffer_load(u8p(rgn(sc, R_X8)), idx, mask=(tok < S) & (by >= 0), other=0, cache=".cv")


@gluon.jit
def red_waves(acc, red, NS: gl.constexpr):
    """A batched accumulator [8 waves, 16 rows, 16 tokens] (MF3) -> (lo, hi) [8 row pairs, NS]
    (PR_L): rows 2 j, 2 j + 1, the waves' K slices summed in wave order from 0 (bd_reduce's)."""
    gl.barrier()  # the previous group's reads done
    red.store(gl.permute(gl.reshape(acc, [8, 8, 2, 16]), (0, 2, 3, 1)))
    gl.barrier()
    lo = gl.zeros([8, NS], gl.float32, PR_L)
    hi = gl.zeros([8, NS], gl.float32, PR_L)
    for q in gl.static_range(8):
        rw = red._reinterpret(gl.float32, [8, 2, 16, 8], FLAT3).index(q)
        lo = lo + rw._reinterpret(gl.float32, [2, 16, 8], FLAT2).index(0).permute((1, 0)).slice(0, NS, dim=1).load(PR_L)
        hi = hi + rw._reinterpret(gl.float32, [2, 16, 8], FLAT2).index(1).permute((1, 0)).slice(0, NS, dim=1).load(PR_L)
    return lo, hi


@gluon.jit
def qkv_b3(t, w_qkv, ws_lo, ws_hi, rr, sc, tag, errors, bid, S: gl.constexpr, tr_=None, qout=0,
           PHASE: gl.constexpr = 0):
    """GEMV task t past 16 rows, batched (QKV_B3): the weights' rounds in registers, 16 tokens a
    group, group g + 1's B issued before group g's MFMAs."""
    m = w8_rows()
    aq = ()
    for r in gl.static_range(6):
        aq = aq + (w8_tile3(w_qkv, t * (16 * 6144) + (8 * r + m // 16) * 2048 + (m % 16) * 16),)
    m = pin(m)
    lt = ar(THREADS, T1)
    gate_poll(rgn(sc, R_X8S), lt, lt < S, tag, errors, bid, E_X8)
    stamp(tr_, bid, 20)
    one = gl.full([8, 16, 4], 127, gl.uint8, SL3)
    red = gl.allocate_shared_memory(gl.float32, [8, 2, 16, 8], FLAT4)
    G: gl.constexpr = (S + 15) // 16
    xb = ()
    for r in gl.static_range(6):
        if QKV_BYTES:
            xb = xb + (x8_b3b(sc, 0, r, S),)
        else:
            xb = xb + (x8_b3(sc, 0, r, S),)
    for g in gl.static_range(G):
        xn = ()
        if g + 1 < G:
            for r in gl.static_range(6):
                if QKV_BYTES:
                    xn = xn + (x8_b3b(sc, 16 * (g + 1), r, S),)
                else:
                    xn = xn + (x8_b3(sc, 16 * (g + 1), r, S),)
        acc = zacc([8, 16, 16], MF3)
        for r in gl.static_range(6):
            if QKV_BYTES:
                bv = gl.convert_layout(gl.permute(gl.reshape(xb[r], [8, 16, 128]), (0, 2, 1)), B3, assert_trivial=True)
            else:
                bv = gl.convert_layout(gl.permute(gl.reshape(bytes8(xb[r]), [8, 16, 128]), (0, 2, 1)), B3,
                                       assert_trivial=True)
            if QKV_DBG & 4:  # (diagnostics: no MFMA)
                acc = acc + gl.sum(gl.sum(bv.to(gl.float32), axis=2), axis=1)[:, None, None] * 0.0 + \
                    gl.sum(gl.sum(aq[r].to(gl.float32), axis=2), axis=1)[:, None, None] * 0.0
            else:
                acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one, "e4m3",
                                        bv.to(gl.float8e4nv, bitcast=True), one, "e4m3", acc)
        los, his = red_waves(acc, red, 16)
        nn = 16 * g + ar(16, layout=gl.SliceLayout(0, PR_L))[None, :]
        isc = nn + rr * 0
        xsc = wait_group(((rgn(sc, R_X8S), isc, peek(rgn(sc, R_X8S), isc, tag, isc < S)),), tag, errors, bid,
                         E_X8, 1)[0].to(gl.float32, bitcast=True)
        lo = bf16_round(los * xsc * ws_lo)
        hi = bf16_round(his * xsc * ws_hi)
        if PHASE == 1:  # split hybrid: vLLM's qkv layout [token, 2560] bf16 (pairs as words)
            cdna4.buffer_store(bf_pair(lo, hi), i32p(qout), nn * QKV_PAIRS + 8 * t + rr, mask=nn < S)
        elif (QKV_DBG & 8) == 0:
            put(rgn(sc, R_QKV), nn * QKV_PAIRS + 8 * t + rr, bf_pair(lo, hi), tag, mask=nn < S)
        xb = xn


XKS = gl.constexpr(gl.BlockedLayout([1, 1, 16], [1, 8, 8], [2, 2, 2], [2, 1, 0]))  # [rounds, 16 tokens, 128 B]
KSW = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=8, order=[2, 1, 0]))
KSW2 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=8, order=[1, 0]))


@gluon.jit
def ks_issue(sc, c, k0, r0: gl.constexpr, NR: gl.constexpr, S: gl.constexpr):
    """K slice c's x8 bytes of tokens k0 .. + 16, rounds r0 .. + NR (step 8 r + c): [NR, 16, 128]."""
    rd = ar(NR, layout=gl.SliceLayout(1, gl.SliceLayout(2, XKS)))[:, None, None]
    tk = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, XKS)))[None, :, None]
    by = ar(128, layout=gl.SliceLayout(0, gl.SliceLayout(1, XKS)))[None, None, :]
    tok = k0 + tk
    idx = tok * HIDDEN + (8 * (r0 + rd) + c) * 128 + by
    return cdna4.buffer_load(u8p(rgn(sc, R_X8)), idx, mask=(tok < S) & (rd >= 0) & (by >= 0), other=0, cache=".cv")


@gluon.jit
def ks_reduce(tile, g, rr, sc, tag, errors, bid, ws_lo, ws_hi, S: gl.constexpr, qout=0, PHASE: gl.constexpr = 0):
    """Tile ``tile``'s 8 K-slice partials of tokens 16 g .. -> summed in slice order -> qkv."""
    nn = 16 * g + ar(16, layout=gl.SliceLayout(0, PR_L))[None, :]
    live = nn < S
    pks = ()
    for c in gl.static_range(8):
        base = (tile * 8 + c) * (MAX_S * 16) + nn * 16 + 2 * rr
        pks = pks + (peek(rgn(sc, R_QKVP), base, tag, live), peek(rgn(sc, R_QKVP), base + 1, tag, live))
    its = ()
    for c in gl.static_range(8):
        base = (tile * 8 + c) * (MAX_S * 16) + nn * 16 + 2 * rr
        its = its + ((rgn(sc, R_QKVP), base, pks[2 * c]), (rgn(sc, R_QKVP), base + 1, pks[2 * c + 1]))
    isc = nn + rr * 0
    xpk = peek(rgn(sc, R_X8S), isc, tag, live)
    ws = wait_group(its, tag, errors, bid, E_QKV, 1)
    lo = gl.zeros([8, 16], gl.float32, PR_L)
    hi = gl.zeros([8, 16], gl.float32, PR_L)
    for c in gl.static_range(8):
        lo = lo + ws[2 * c].to(gl.float32, bitcast=True)
        hi = hi + ws[2 * c + 1].to(gl.float32, bitcast=True)
    xsc = wait_group(((rgn(sc, R_X8S), isc, xpk),), tag, errors, bid, E_X8, 1)[0].to(gl.float32, bitcast=True)
    lo = bf16_round(lo * xsc * ws_lo)
    hi = bf16_round(hi * xsc * ws_hi)
    if PHASE == 1:
        cdna4.buffer_store(bf_pair(lo, hi), i32p(qout), nn * QKV_PAIRS + 8 * tile + rr, mask=live)
    else:
        put(rgn(sc, R_QKV), nn * QKV_PAIRS + 8 * tile + rr, bf_pair(lo, hi), tag, mask=live)


@gluon.jit
def qkv_ks(b, w_qkv, s_qkv, sc, tag, errors, bid, S: gl.constexpr, tr_=None, qout=0, PHASE: gl.constexpr = 0):
    """GEMV past 16 rows, K split across CTAs (QKV_KS): CTA b = 8 u + c, wave w row tile 8 u + w
    over K slice c; partials to tile 8 u + w's reducer CTA (8 u + w); this CTA reduces tile b."""
    c = b % 8
    u = b // 8
    m = w8_rows()
    aq = ()
    for r in gl.static_range(6):
        aq = aq + (w8_tile3(w_qkv, (8 * u + m // 16) * (16 * 6144) + (8 * r + c) * 2048 + (m % 16) * 16),)
    m = pin(m)
    rr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    ws_lo = gl.load(s_qkv + 16 * b + 2 * rr)
    ws_hi = gl.load(s_qkv + 16 * b + 2 * rr + 1)
    lt = ar(THREADS, T1)
    gate_poll(rgn(sc, R_X8S), lt, lt < S, tag, errors, bid, E_X8)
    stamp(tr_, bid, 20)
    one = gl.full([8, 16, 4], 127, gl.uint8, SL3)
    xs = gl.allocate_shared_memory(gl.uint8, [8, 16, 128], KSW)  # (6 rounds, padded: power-of-2 shapes)
    G: gl.constexpr = (S + 15) // 16
    # the partial's coordinates in MF3: wave = tile in the group, row, token
    pw = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, MF3)))[:, None, None]
    pr = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, MF3)))[None, :, None]
    pn = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, MF3)))[None, None, :]
    xa = ks_issue(sc, c, 0, 0, 4, S)
    xb = ks_issue(sc, c, 0, 4, 2, S)
    for g in gl.static_range(G):
        gl.barrier()  # the previous group's B reads done
        xs.slice(0, 4, dim=0).store(xa)
        xs.slice(4, 2, dim=0).store(xb)
        if g + 1 < G:
            xa = ks_issue(sc, c, 16 * (g + 1), 0, 4, S)
            xb = ks_issue(sc, c, 16 * (g + 1), 4, 2, S)
        gl.barrier()
        acc = zacc([8, 16, 16], MF3)
        for r in gl.static_range(6):
            b2 = xs._reinterpret(gl.uint8, [8, 16, 128], KSW2).index(r).permute((1, 0)).load(gl.SliceLayout(0, B3))
            bv = gl.expand_dims(b2, 0) + gl.zeros([8, 128, 16], gl.uint8, B3)  # (every wave the same B)
            acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one, "e4m3",
                                    bv.to(gl.float8e4nv, bitcast=True), one, "e4m3", acc)
        tok = 16 * g + pn
        put(rgn(sc, R_QKVP), ((8 * u + pw) * 8 + c) * (MAX_S * 16) + tok * 16 + pr, acc, tag, mask=tok < S)
        if g >= 1:
            ks_reduce(b, g - 1, rr, sc, tag, errors, bid, ws_lo, ws_hi, S, qout, PHASE)
    ks_reduce(b, G - 1, rr, sc, tag, errors, bid, ws_lo, ws_hi, S, qout, PHASE)


@gluon.jit
def stage_qkv(t, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, S: gl.constexpr, tr_=None,
              arsrc=0, qout=0, PHASE: gl.constexpr = 0):
    """GEMV task t: rows 16 t .. + 16 of the fused projection for every token."""
    e = lrow(ptab, li, (0, 1, 2))
    w_qkv = lp(e[0], gl.uint8)
    s_qkv = lp(e[1], gl.float32)
    g_in = lp(e[2], gl.int32)
    SP: gl.constexpr = triton.next_power_of_2(S)
    # past 16 rows the B operand goes 16 tokens a group (the weights stay in registers)
    NS: gl.constexpr = 16 if S > 16 else ncols(S)
    # the input's loads first: they return in order, so issued after the weights they
    # would wait for the whole weight stream; the norm / quant then overlap it (S > 2:
    # the input words and the weights in flight together do not fit the registers)
    EARLY: gl.constexpr = (S <= 2) and (not HYB)
    if EARLY:
        hin_a, mail_a, ga, pa, oka = qkv_in_issue(li, 0, 32, SP, S, hid_in, res_in, psc, ptag, g_in)
        hin_b, mail_b, gb, pb, okb = qkv_in_issue(li, 32, 16, SP, S, hid_in, res_in, psc, ptag, g_in)
        pin(tids())
    NEWQ: gl.constexpr = PLAIN_HO and (not QKV_GGATE) and ((S > 16) or (NORM_TASKS and S > max(2, SMALL_FROM) and QKV_SMALL))
    if QKV_KS and NEWQ and (S > QKV_KS_FROM):
        qkv_ks(t, w_qkv, s_qkv, sc, tag, errors, bid, S, tr_, qout, PHASE)
        return
    if QKV_B3 and NEWQ:
        rr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
        qkv_b3(t, w_qkv, gl.load(s_qkv + 16 * t + 2 * rr), gl.load(s_qkv + 16 * t + 2 * rr + 1), rr, sc, tag,
               errors, bid, S, tr_, qout, PHASE)
        return
    # the weights: round r, wave w takes 128-k step 8 r + w of the task's 16 rows
    if QKV_WDELAY > 0:  # (experiment) the norm tasks' row loads ahead of the weight stream
        tid_ = gl.inline_asm_elementwise(_sleep_asm(QKV_WDELAY), "=v,0,~{memory}", [tids()], dtype=gl.int32,
                                         is_pure=False, pack=1)
    m = w8_rows()
    aq = ()
    for r in gl.static_range(6):
        if QKV_DBG & 2:  # (no weight loads)
            aq = aq + (gl.zeros([128, 128], gl.uint8, A_L),)
        else:
            aq = aq + (w8_tile(w_qkv, t * (16 * 6144) + (8 * r + m // 16) * 2048 + (m % 16) * 16),)
    rr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    ws_lo = gl.load(s_qkv + 16 * t + 2 * rr)
    ws_hi = gl.load(s_qkv + 16 * t + 2 * rr + 1)
    m = pin(m)
    xsa = gl.allocate_shared_memory(gl.uint8, [4, 8 * NS, 128], BSW)
    xsb = gl.allocate_shared_memory(gl.uint8, [2, 8 * NS, 128], BSW)
    if S > 16:
        lt = ar(THREADS, T1)
        if not QKV_GGATE:
            gate_poll(rgn(sc, R_X8S), lt, lt < S, tag, errors, bid, E_X8)
        stamp(tr_, bid, 20)
        one_a = unit_scales([128, 4], A_L)
        one_b = unit_scales([8 * NS, 4], B_L)
        red = gl.allocate_shared_memory(gl.float32, [128, NS], FLAT2)
        for g in gl.static_range((S + 15) // 16):
            if QKV_GGATE:  # (each 16-token group's scales: group 0 starts before the last norms)
                gate_poll(rgn(sc, R_X8S), 16 * g + lt, (lt < 16) & (16 * g + lt < S), tag, errors, bid, E_X8)
            else:
                gl.barrier()  # the previous group's B reads done
            if (QKV_DBG & 1) == 0:  # (1: no x8 loads)
                x8_part(sc, tag, errors, bid, xsa, 0, 32, 16, S, 16 * g)
                x8_part(sc, tag, errors, bid, xsb, 32, 16, 16, S, 16 * g)
            gl.barrier()
            acc = zacc([128, 8 * NS], MFMA)
            if (QKV_DBG & 4) == 0:  # (4: no MFMA)
                for r in gl.static_range(6):
                    if r < 4:
                        bv = rnd(xsa, r).permute((1, 0)).load(B_L)
                    else:
                        bv = rnd(xsb, r - 4).permute((1, 0)).load(B_L)
                    acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                            bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
            else:
                acc = acc + gl.sum(gl.sum(aq[5].to(gl.float32), axis=1), axis=0)
            los, his = bd_reduce(acc, 1, 8, red)
            nn = 16 * g + ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :]
            isc = nn + rr * 0
            xsc = wait_group(((rgn(sc, R_X8S), isc, peek(rgn(sc, R_X8S), isc, tag, isc < S)),), tag, errors, bid,
                             E_X8, 1)[0].to(gl.float32, bitcast=True)
            lo = bf16_round(los[0] * xsc * ws_lo)
            hi = bf16_round(his[0] * xsc * ws_hi)
            if PHASE == 1:
                cdna4.buffer_store(bf_pair(lo, hi), i32p(qout), nn * QKV_PAIRS + 8 * t + rr, mask=nn < S)
            elif (QKV_DBG & 8) == 0:  # (8: no output puts)
                put(rgn(sc, R_QKV), nn * QKV_PAIRS + 8 * t + rr, bf_pair(lo, hi), tag, mask=nn < S)
    else:
        if NORM_TASKS and S > 2:
            # the norm tasks' FP8 rows: every token's scale (put after its words), then 6 KB a token
            lt = ar(THREADS, T1)
            if not X8_TAG:
                gate_poll(rgn(sc, R_X8S), lt, lt < S, tag, errors, bid, E_X8)
            stamp(tr_, bid, 20)
            if QKV_SCPF:  # the epilogue's scales issued now (landed past the gate): no round trip after the MFMA
                isc_pf = ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :] + rr * 0
                pk_pf = peek(rgn(sc, R_X8S), isc_pf, tag, isc_pf < S)
            x8_part(sc, tag, errors, bid, xsa, 0, 32, SP, S, 0, X8_TAG)
            x8_part(sc, tag, errors, bid, xsb, 32, 16, SP, S, 0, X8_TAG)
            x_scale = 0.0
        else:
            if EARLY:
                xa_lo, xa_hi = qkv_in_finish(li, hin_a, mail_a, oka, pa, psc, ptag, errors, bid, SP * 4)
                xb_lo, xb_hi = qkv_in_finish(li, hin_b, mail_b, okb, pb, psc, ptag, errors, bid, SP * 2)
            else:
                xa_lo, xa_hi, ga, pa, oka = qkv_in(li, 0, 32, SP, S, hid_in, res_in, psc, ptag, errors, bid, g_in, 0,
                                                   arsrc)
                xb_lo, xb_hi, gb, pb, okb = qkv_in(li, 32, 16, SP, S, hid_in, res_in, psc, ptag, errors, bid, g_in, 0,
                                                   arsrc)
            stamp(tr_, bid, 20)
            ss = qkv_fold(gl.sum(xa_lo * xa_lo + xa_hi * xa_hi, axis=2), gl.sum(xb_lo * xb_lo + xb_hi * xb_hi, axis=2), False)
            rstd = hw_rsq(ss / 6144.0 + eps)
            amax = qkv_fold(qkv_absn(xa_lo, xa_hi, ga, rstd), qkv_absn(xb_lo, xb_hi, gb, rstd), True)
            x_scale = gl.where(amax == 0.0, 1.0, gl.div_rn(amax, gl.full(amax.shape, 448.0, gl.float32, amax.type.layout)))
            # token k's h = bf16(ar + res), from CTA k
            if t < S:
                k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))[:, None, None]
                put(rgn(sc, R_H), pa, bf_pair(xa_lo, xa_hi), tag, mask=(k3 == t) & oka)
                put(rgn(sc, R_H), pb, bf_pair(xb_lo, xb_hi), tag, mask=(k3 == t) & okb)
            # FP8 x -> LDS [steps, 16 tokens, 128] (steps 0..31, 32..47): the B operand, round
            # r = steps 8 r .. + 8
            qkv_quant(xa_lo, xa_hi, ga, rstd, x_scale, xsa, 32, SP)
            qkv_quant(xb_lo, xb_hi, gb, rstd, x_scale, xsb, 16, SP)
        stamp(tr_, bid, 21)
        gl.barrier()
        acc = zacc([128, 8 * NS], MFMA)
        one_a = unit_scales([128, 4], A_L)
        one_b = unit_scales([8 * NS, 4], B_L)
        for r in gl.static_range(6):
            if r < 4:
                bv = rnd(xsa, r).permute((1, 0)).load(B_L)
            else:
                bv = rnd(xsb, r - 4).permute((1, 0)).load(B_L)
            acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                    bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
        red = gl.allocate_shared_memory(gl.float32, [128, NS], FLAT2)
        los, his = bd_reduce(acc, 1, 8, red)
        # the tokens' x scales in the epilogue layout
        nn = ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :]
        kq = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3)))
        xsc = gl.zeros([8, NS], gl.float32, PR_L)
        if NORM_TASKS and S > 2:
            isc = nn + rr * 0
            if QKV_SCPF:
                xsc = wait_group(((rgn(sc, R_X8S), isc, pk_pf),), tag, errors, bid, E_X8, 1)[0].to(gl.float32,
                                                                                                bitcast=True)
            else:
                xsc = wait_group(((rgn(sc, R_X8S), isc, peek(rgn(sc, R_X8S), isc, tag, isc < S)),), tag, errors, bid,
                                 E_X8, 1)[0].to(gl.float32, bitcast=True)
        else:
            for k in gl.static_range(S):
                xsc = gl.where(nn == k, gl.sum(gl.where(kq == k, x_scale, 0.0), axis=0), xsc)
        lo = bf16_round(los[0] * xsc * ws_lo)
        hi = bf16_round(his[0] * xsc * ws_hi)
        if PHASE == 1:
            cdna4.buffer_store(bf_pair(lo, hi), i32p(qout), nn * QKV_PAIRS + 8 * t + rr, mask=nn < S)
        else:
            put(rgn(sc, R_QKV), nn * QKV_PAIRS + 8 * t + rr, bf_pair(lo, hi), tag, mask=nn < S)


# ============================================================ head math (norm + rope)
@gluon.jit
def head_rows(qkv, tok, pair0, npairs: gl.constexpr, layout: gl.constexpr, tag, errors, bid, code: gl.constexpr):
    """``npairs`` / 64 heads of token tok's GEMV rows from pair pair0, each with its RoPE
    partner (dim d +- 32 of the head; d >= 64 none): [heads, 64 pairs] (lo, hi, plo, phi)."""
    NH: gl.constexpr = npairs // 64
    h = ar(NH, layout=gl.SliceLayout(1, layout))[:, None]
    j = ar(64, layout=gl.SliceLayout(0, layout))[None, :]
    pk = head_peek(qkv, tok, pair0, npairs, layout, tag, h >= 0)
    return head_land(qkv, pk, tag, errors, bid, code)


@gluon.jit
def head_peek(qkv, tok, pair0, npairs: gl.constexpr, layout: gl.constexpr, tag, pred):
    """``head_rows``' loads (rows and RoPE partners) issued only; ``head_land`` finishes."""
    NH: gl.constexpr = npairs // 64
    h = ar(NH, layout=gl.SliceLayout(1, layout))[:, None]
    j = ar(64, layout=gl.SliceLayout(0, layout))[None, :]
    base = tok * QKV_PAIRS + pair0 + h * 64
    ok = (h < NH) & pred
    pj = gl.where(j < 16, j + 16, j - 16)
    return (base + j, peek(qkv, base + j, tag, ok), base + pj, peek(qkv, base + pj, tag, ok & (j < 32)))


@gluon.jit
def head_land(qkv, pk, tag, errors, bid, code: gl.constexpr, PACK: gl.constexpr = 0):
    i0, p0, i1, p1 = pk
    if PACK != 0:  # one batched wait (PACK = the layout's elements a thread)
        ws = wait_group(((qkv, i0, p0), (qkv, i1, p1)), tag, errors, bid, code, PACK)
        w = ws[0]
        pw = ws[1]
    else:
        w = land(qkv, i0, tag, p0, errors, bid, code)
        pw = land(qkv, i1, tag, p1, errors, bid, code)
    return bf_lo(w), bf_hi(w), bf_lo(pw), bf_hi(pw)


@gluon.jit
def rope_tabs(gamma, cos_sin, pos, layout: gl.constexpr):
    """``head_norm_rope``'s gamma and cos / sin words for [heads, 64 pairs] in ``layout``:
    loaded before the rows' wait (after it they are one more round trip)."""
    j = ar(64, layout=gl.SliceLayout(0, layout))[None, :]
    gw = gl.load(gamma + j)
    pjj = gl.where(j < 16, j + 16, j - 16)
    pgw = gl.load(gamma + pjj, mask=j < 32, other=0)
    i = j % 16
    cw = gl.load(cos_sin + pos * 32 + i)  # i32 pairs: cos dims 2 i, 2 i + 1
    sw = gl.load(cos_sin + pos * 32 + 16 + i)
    return gw, pgw, cw, sw


@gluon.jit
def head_norm_rope(lo, hi, plo, phi, tabs, eps):
    """Gemma RMSNorm over each head's 128 dims + partial NeoX RoPE (dims < 64): ->
    (lo, hi) f32 of [heads, 64 pairs] (pair j = dims 2 j, 2 j + 1). ``tabs``: ``rope_tabs``."""
    layout: gl.constexpr = lo.type.layout
    j = ar(64, layout=gl.SliceLayout(0, layout))[None, :]
    ss = gl.sum(lo * lo + hi * hi, axis=1)
    rr = hw_rsq(ss / 128.0 + eps)[:, None]
    gw, pgw, cw, sw = tabs
    n_lo = lo * rr * (bf_lo(gw) + 1.0)
    n_hi = hi * rr * (bf_hi(gw) + 1.0)
    p_lo = plo * rr * (bf_lo(pgw) + 1.0)
    p_hi = phi * rr * (bf_hi(pgw) + 1.0)
    # cos / sin of dim d % 32 at this position: cos_sin row [cos 32 | sin 32] bf16
    first = j < 16
    ps_lo = p_lo * bf_lo(sw)
    ps_hi = p_hi * bf_hi(sw)
    r_lo = gl.fma(n_lo, bf_lo(cw), gl.where(first, -ps_lo, ps_lo))
    r_hi = gl.fma(n_hi, bf_hi(cw), gl.where(first, -ps_hi, ps_hi))
    rope = j < 32
    return gl.where(rope, r_lo, n_lo), gl.where(rope, r_hi, n_hi)


HL = gl.constexpr(gl.BlockedLayout([1, 2], [2, 32], [8, 1], [1, 0]))  # [16 heads, 64 pairs]
H1 = gl.constexpr(gl.BlockedLayout([1, 1], [1, 64], [1, 8], [1, 0]))  # [1 head, 64 pairs]


@gluon.jit
def stage_ik(li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps, S: gl.constexpr, k0=0,
             NK: gl.constexpr = 0):
    """Tokens k0 .. k0 + NK (every token: NK = 0)'s index k -> norm / rope -> vLLM's bf16
    index cache at its slot."""
    e = lrow(ptab, li, (6, 18))
    g_ik = lp(e[0], gl.int32)
    icache = lp(e[1], gl.int32)
    NT: gl.constexpr = S if NK == 0 else NK
    for kk in gl.static_range(NT):
        k = k0 + kk
        pos = uniform(gl.load(positions + k).to(gl.int32))
        slot = uniform(gl.load(slots + k).to(gl.int32))
        tabs = rope_tabs(g_ik, cos_sin, pos, H1)
        lo, hi, plo, phi = head_rows(rgn(sc, R_QKV), k, (HEADS + 3) * 64, 64, H1, tag, errors, bid, E_QKV)
        o_lo, o_hi = head_norm_rope(lo, hi, plo, phi, tabs, eps)
        j = ar(64, layout=gl.SliceLayout(0, H1))[None, :]
        cdna4.buffer_store(bf_pair(o_lo, o_hi), icache, slot * 64 + j, mask=slot >= 0, cache=".wt")
    # the keys drained, then a flag a token: a scorer reading a block this launch writes waits
    ft = drain(ar(THREADS, T1))
    put(rgn(sc, R_IKF), k0 + ft, ft * 0 + 1, tag, mask=ft < NT)


# ============================================================ indexer (contexts past MAX_SHORT)
# FlyDSL's (kernels/index_score.py, stages/select.py) on vLLM's caches: a selecting layer
# scores every index-K block a request does not pin (none at the start, the tail block --
# the one holding the token -- at the end) against each of its rows' index q (bf16 MFMA, the
# max over the row's causal keys, times fp32(sm_scale log2 e)); each split task (token k,
# partition p) then picks the TOPK_BLOCKS best by (score, block id), the tail pinned to 1e29,
# full blocks in rank order and the tail last, and attends over slots 2 p, 2 p + 1. Up to
# LONG_FROM blocks every split task ranks them all (exact: an entry's rank is the count of
# entries that beat it); past it task p ranks an eighth and the eight shares' 16 best are
# ranked again. A layer reusing a selection reads the token's sparse table, which the step's
# last selecting layer wrote (a launch a layer: an earlier kernel).
@gluon.jit
def row_seq(k, positions, seq_lens, QL: gl.constexpr):
    """Row k's context (its position + 1 for a speculative verify's rows)."""
    if QL > 1:
        sq = uniform(gl.load(positions + k).to(gl.int32)) + 1
    else:
        sq = gl.maximum(uniform(gl.load(seq_lens + k)), 1)
    return sq


@gluon.jit
def task_rb(u, pre, NR: gl.constexpr):
    """Score task u of the step's list (``pre``: the requests' prefix counts) -> (request, block)."""
    r = u * 0
    for q in gl.static_range(1, NR):
        r = r + (u >= pre[q]).to(gl.int32)
    b = u - pre[0]
    for q in gl.static_range(1, NR):
        b = gl.where(r == q, u - pre[q], b)
    return r, b


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def score_tile(u0, u_hi, pre, NR: gl.constexpr, block_table, bt_stride, icache, CV: gl.constexpr = False,
               pgl=None, u_lo=0):
    """Index K^T [128, SC_NB * 128] (bf16, BS_L) of score tasks u0 .. u0 + SC_NB (masked past
    u_hi; CV: read past the caches). ``pgl``: the CTA's task pages in LDS (task u_lo + i at
    word i), else the block table is read (a dependent round trip ahead of the keys)."""
    NK: gl.constexpr = SC_NB * 128
    dd = ar(128, layout=gl.SliceLayout(1, BS_L))[:, None]
    nb_ = ar(NK, layout=gl.SliceLayout(0, BS_L))[None, :]
    ub = u0 + nb_ // 128
    if SC_NOMASK:
        # unmasked loads of clamped (valid) tasks: a masked load is a branch around every
        # load instruction, after which LLVM only knows a lower bound of the loads in
        # flight and waits for nearly all of them (vmcnt(2)) before the MFMA -- the
        # prefetched tile never overlapped. Dead columns are masked at the max.
        liveb = ub < u_hi
        ub = gl.minimum(ub, u_hi - 1)
    else:
        liveb = ub < u_hi
    if pgl is None:
        rb, bb = task_rb(ub, pre, NR)
        pg = gl.load(block_table + rb * bt_stride + bb, mask=liveb, other=0)
    else:
        nb1 = ar(NK, layout=gl.SliceLayout(0, BS_L))
        ub1 = u0 + nb1 // 128
        if SC_NOMASK:
            ub1 = gl.minimum(ub1, u_hi - 1)
        pg = pgl.gather(gl.minimum(gl.maximum(ub1 - u_lo, 0), SC_NPG - 1), 0)[None, :]
    if SC_NOMASK:
        # a dead column (past u_hi): every lane the same word of its (valid) page -- one
        # line a load instead of a re-read of the last tile (the duplicate 64 KB delayed
        # the scores and the o weights of the CTAs with one tile: 2K +1 us a layer)
        kptr = icache + pg.to(gl.int64) * 16384 + gl.where(liveb, nb_ % 128, 0) * 128 + dd
    else:
        kptr = icache + pg.to(gl.int64) * 16384 + (nb_ % 128) * 128 + dd
    if SC_NOMASK:
        if CV:
            kv = gl.load(kptr, cache_modifier=".cv")
        else:
            kv = gl.load(kptr)
    elif CV:
        kv = gl.load(kptr, mask=liveb, other=0.0, cache_modifier=".cv")
    else:
        kv = gl.load(kptr, mask=liveb, other=0.0)
    return kv


@gluon.jit
def score_body(u0, kv, u_hi, u_lo, pre, NR: gl.constexpr, block_table, bt_stride, icache, pgl, lasts, QL: gl.constexpr,
               S: gl.constexpr, sc, tag, errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2):
    """Score tile u0 .. u0 + SC_NB (``kv``: its keys, landed or in flight): the new-key wait,
    the MFMA, each block's row maxima -> isc. Masked past u_hi (zero keys, no puts)."""
    NK: gl.constexpr = SC_NB * 128
    # a block that may hold an index key this launch inserts (its request's rows end
    # within QL keys of it): stage_ik's flags first, then a read past the caches
    if SC_FUSE:
        newk = u0 * 0
        for jb in gl.static_range(SC_NB):
            ru_, bu_ = task_rb(u0 + jb, pre, NR)
            sl_ = lasts[0]
            for q in gl.static_range(1, NR):
                sl_ = gl.where(ru_ == q, lasts[q], sl_)
            newk = newk | ((u0 + jb < u_hi) & ((bu_ + 1) * 128 > sl_ - QL)).to(gl.int32)
    else:
        ru_, bu_ = task_rb(u0 + jt, pre, NR)
        sl_ = jt * 0 + lasts[0]
        for q in gl.static_range(1, NR):
            sl_ = gl.where(ru_ == q, lasts[q], sl_)
        newk = gl.max(((u0 + jt < u_hi) & ((bu_ + 1) * 128 > sl_ - QL)).to(gl.int32), axis=0)
    if newk != 0:
        fpk = peek(rgn(sc, R_IKF), tl_, tag, tl_ < S)
        land(rgn(sc, R_IKF), tl_, tag, fpk, errors, bid, E_QKV)
        if TAIL_ACQ:  # the new index keys' rows past this CU's L1 (FlyDSL's wait_new_keys acquire)
            l2_acquire(tl_)
        kv = score_tile(u0, u_hi, pre, NR, block_table, bt_stride, icache, True, pgl, u_lo)
    s = cdna4.mfma(qa, kv, zacc([16, NK], MFS))
    un = u0 + nn // 128
    rn, bn = task_rb(un, pre, NR)
    ok = (cc[:, None] // QL == rn[None, :]) & (cc[:, None] < S) & (un[None, :] < u_hi) \
        & (bn[None, :] * 128 + (nn % 128)[None, :] < seqc[:, None])
    sv = gl.where(ok, s, float("-inf"))
    if SC_FUSE:
        # the blocks' maxima in one reduction (one LDS exchange and barrier a tile)
        svs = ()
        for jb in gl.static_range(SC_NB):
            svs = svs + (gl.where((nn // 128 == jb)[None, :], sv, float("-inf")),)
        if SC_NB == 2:
            ms = gl.reduce(svs, 1, _max2)
        else:
            ms = gl.reduce(svs, 1, _max4)
        for jb in gl.static_range(SC_NB):
            ru, bu = task_rb(u0 + jb, pre, NR)
            put(rgn(sc, R_ISC), cc * MAX_IB + bu, (ms[jb] * sm_l2).to(gl.int32, bitcast=True), tag,
                mask=(u0 + jb < u_hi) & (cc < S) & (cc // QL == ru))
    else:
        for jb in gl.static_range(SC_NB):
            m = gl.max(gl.where((nn // 128 == jb)[None, :], sv, float("-inf")), axis=1)
            ru, bu = task_rb(u0 + jb, pre, NR)
            put(rgn(sc, R_ISC), cc * MAX_IB + bu, (m * sm_l2).to(gl.int32, bitcast=True), tag,
                mask=(u0 + jb < u_hi) & (cc < S) & (cc // QL == ru))


SC_W = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_W", "1")))  # score share of a CTA free of qkv tasks
SC_WQ = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_WQ", "1")))  # score share of a CTA with qkv tasks (starts ~8 us later)
SC_PF = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_PF", "1")))  # score tiles in flight ahead of the current one (1 / 2)
SC_PGL = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_PGL", "1")))  # the CTA's task pages in LDS (no block-table round trip a tile)
SC_FUSE = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_FUSE", "1")))  # one cross-warp max a tile; the new-key test in scalars
SC_NPG = gl.constexpr(512)  # score tasks a CTA can hold pages for
SC_NOMASK = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_NOMASK", "1")))  # key loads unmasked (clamped tasks): precise vmcnt
SC_SLOTS = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SC_SLOTS", "2")))  # fixed key-tile slots (2 / 3), no register rotation (0: the rotating loop)


@gluon.jit
def _max2(a0, a1, b0, b1):
    return gl.maximum(a0, b0), gl.maximum(a1, b1)


@gluon.jit
def _max4(a0, a1, a2, a3, b0, b1, b2, b3):
    return gl.maximum(a0, b0), gl.maximum(a1, b1), gl.maximum(a2, b2), gl.maximum(a3, b3)


@gluon.jit
def stage_score(li, ptab, cos_sin, positions, seq_lens, block_table, bt_stride, sc, tag, errors, bid, eps,
                sm_scale, S: gl.constexpr, QL: gl.constexpr, tr_=None):
    """Every CTA: its share of the step's score tasks (a task = one index-K block of a request
    past TOPK_BLOCKS blocks), SC_NB blocks a tile -> isc pair k * MAX_IB + b for each row k of
    the block's request. A CTA without qkv tasks takes SC_W shares (it starts a GEMV earlier);
    its first tile's keys go out before the index q (FlyDSL: keys before prepare()), the next
    tile's before each tile's math. Index q of every row made on every scoring CTA."""
    NR: gl.constexpr = S // QL
    pre = (bid * 0,)
    lasts = ()
    for r in gl.static_range(NR):
        sq = row_seq(r * QL + QL - 1, positions, seq_lens, QL)
        nb = (sq + 127) // 128
        pre = pre + (pre[r] + gl.where(nb > TOPK_BLOCKS, nb - 1, 0),)
        lasts = lasts + (sq,)
    total = pre[NR]
    # shares: CTAs 0 .. ROWS16 - 1 (qkv tasks) one unit each, the rest SC_W
    units = ROWS16 * SC_WQ + (BLOCKS - ROWS16) * SC_W
    per = (total + units - 1) // units
    u_lo = gl.where(bid < ROWS16, bid * SC_WQ * per, ROWS16 * SC_WQ * per + (bid - ROWS16) * SC_W * per)
    u_hi = gl.minimum(u_lo + gl.where(bid < ROWS16, SC_WQ * per, SC_W * per), total)
    if u_lo < u_hi:
        e = lrow(ptab, li, (5, 18))
        g_iq = lp(e[0], gl.int32)
        icache = lp(e[1], gl.bfloat16)
        if SC_PGL:
            # every task's page once (one load, no round trip a tile in the loop); the table's
            # load goes out first, the first tile reads its own pages directly (its keys are in
            # flight while the table is stored: a CTA with one tile pays nothing for the table)
            pgl = gl.allocate_shared_memory(gl.int32, [SC_NPG], FLAT1)
            ti = ar(SC_NPG, layout=T1)
            rbi, bbi = task_rb(u_lo + ti, pre, NR)
            pgv = gl.load(block_table + rbi * bt_stride + bbi, mask=u_lo + ti < u_hi, other=0)
            kv_n = score_tile(u_lo, u_hi, pre, NR, block_table, bt_stride, icache, False, None, u_lo)
            pgl.store(pgv)
            gl.barrier()
        else:
            pgl = None
            kv_n = score_tile(u_lo, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
        if (SC_PF == 2) and (SC_SLOTS == 0):
            kv_nn = score_tile(u_lo + SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
        # every row's index q: gemma RMSNorm + partial RoPE of its GEMV rows, bf16
        tk = ar(16, layout=gl.SliceLayout(1, HL))[:, None]
        j = ar(64, layout=gl.SliceLayout(0, HL))[None, :]
        okr = tk < S
        qkv = rgn(sc, R_QKV)
        pj = gl.where(j < 16, j + 16, j - 16)
        i0 = tk * QKV_PAIRS + IQ_P0 + j
        i1 = tk * QKV_PAIRS + IQ_P0 + pj
        pk0 = peek(qkv, i0, tag, okr)
        pk1 = peek(qkv, i1, tag, okr & (j < 32))
        posr = gl.load(positions + tk, mask=okr, other=0).to(gl.int32)
        gw = gl.load(g_iq + j)
        pgw = gl.load(g_iq + pj, mask=j < 32, other=0)
        cw = gl.load(cos_sin + posr * 32 + j % 16)
        sw = gl.load(cos_sin + posr * 32 + 16 + j % 16)
        w0 = land(qkv, i0, tag, pk0, errors, bid, E_QKV)
        w1 = land(qkv, i1, tag, pk1, errors, bid, E_QKV)
        q_lo, q_hi = head_norm_rope(bf_lo(w0), bf_hi(w0), bf_lo(w1), bf_hi(w1), (gw, pgw, cw, sw), eps)
        q16 = gl.reshape(gl.join(q_lo.to(gl.bfloat16), q_hi.to(gl.bfloat16)), [16, 128])
        qa = gl.convert_layout(q16, AS_L)
        sm_l2 = sm_scale * LOG2E
        NK: gl.constexpr = SC_NB * 128
        cc = ar(16, layout=gl.SliceLayout(1, MFS))
        posc = gl.load(positions + cc, mask=cc < S, other=0).to(gl.int32)
        if QL > 1:
            seqc = posc + 1
        else:
            seqc = gl.maximum(gl.load(seq_lens + cc, mask=cc < S, other=1), 1)
        nn = ar(NK, layout=gl.SliceLayout(0, MFS))
        tl_ = ar(THREADS, T1)
        jt = ar(SC_NB, layout=T1)
        if SC_SLOTS == 0:
            for u0 in range(u_lo, u_hi, SC_NB):
                kv = kv_n
                if SC_PF == 2:
                    kv_n = kv_nn
                    kv_nn = score_tile(u0 + 2 * SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
                else:
                    kv_n = score_tile(u0 + SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
                score_body(u0, kv, u_hi, u_lo, pre, NR, block_table, bt_stride, icache, pgl, lasts, QL, S, sc, tag,
                           errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2)
        elif SC_SLOTS == 2:
            # fixed slots, no rotation: a rotating ``kv = kv_n`` is a register copy of a tile
            # still in flight, which waits for it (the prefetch was never overlapped); tiles
            # past u_hi run masked (zero keys, no puts)
            # the loads stay unconditional (a conditional prefetch leaves LLVM unsure what is in
            # flight: vmcnt(0) before every MFMA again); a CTA with one tile skips the loop (its
            # two dead prefetches were drained at the loop exit ahead of the o stage: 2K +0.9 us)
            kva = kv_n
            if u_hi - u_lo <= SC_NB:
                score_body(u_lo, kva, u_hi, u_lo, pre, NR, block_table, bt_stride, icache, pgl, lasts, QL, S, sc, tag,
                           errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2)
            else:
                for u00 in range(u_lo, u_hi, 2 * SC_NB):
                    kvb = score_tile(u00 + SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
                    score_body(u00, kva, u_hi, u_lo, pre, NR, block_table, bt_stride, icache, pgl, lasts, QL, S, sc,
                               tag, errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2)
                    kva = score_tile(u00 + 2 * SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
                    if u00 + SC_NB < u_hi:
                        score_body(u00 + SC_NB, kvb, u_hi, u_lo, pre, NR, block_table, bt_stride, icache, pgl, lasts,
                                   QL, S, sc, tag, errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2)
        else:
            kva = kv_n
            kvb = score_tile(u_lo + SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
            for u00 in range(u_lo, u_hi, 3 * SC_NB):
                kvc = score_tile(u00 + 2 * SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
                score_body(u00, kva, u_hi, u_lo, pre, NR, block_table, bt_stride, icache, pgl, lasts, QL, S, sc, tag,
                           errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2)
                kva = score_tile(u00 + 3 * SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
                if u00 + SC_NB < u_hi:
                    score_body(u00 + SC_NB, kvb, u_hi, u_lo, pre, NR, block_table, bt_stride, icache, pgl, lasts, QL, S,
                               sc, tag, errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2)
                kvb = score_tile(u00 + 4 * SC_NB, u_hi, pre, NR, block_table, bt_stride, icache, False, pgl, u_lo)
                if u00 + 2 * SC_NB < u_hi:
                    score_body(u00 + 2 * SC_NB, kvc, u_hi, u_lo, pre, NR, block_table, bt_stride, icache, pgl, lasts, QL,
                               S, sc, tag, errors, bid, tl_, jt, qa, nn, cc, seqc, sm_l2)
    stamp(tr_, bid, 44)


@gluon.jit
def index_key(b, word, n_blk):
    """FlyDSL's sort key of block b's raw score word: the tail block pinned to 1e29, then a
    signed int ordered like the float (-0 == 0, NaN below -inf)."""
    scf = gl.where(b >= n_blk - 1, 1e29, word.to(gl.float32, bitcast=True))
    bits = scf.to(gl.int32, bitcast=True)
    least = bits * 0 - 2147483647 - 1
    bits = gl.where(bits == least, 0, bits)
    key = gl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return gl.where((bits & 0x7FFFFFFF) > 0x7F800000, least, key)


@gluon.jit
def key64(key, idx):
    return (key.to(gl.int64) << 32) | idx.to(gl.int64)


@gluon.jit
def count_beaten(mines, lds, m, NCH: gl.constexpr, DYN: gl.constexpr = False):
    """(Fallback: every entry.) For each of a thread's 64-bit sort keys ``mines`` (a tuple of
    [512] T1), how many of the first m entries of LDS ``lds`` ([2 * THREADS] int64) beat it,
    a 16-entry chunk at a time (broadcast reads)."""
    E: gl.constexpr = len(mines)
    CL16: gl.constexpr = gl.BlockedLayout([1, 16], [64, 1], [8, 1], [1, 0])
    jj = ar(16, layout=gl.SliceLayout(0, CL16))[None, :]
    mcs = ()
    cnts = ()
    for e in gl.static_range(E):
        mcs = mcs + (gl.convert_layout(mines[e], gl.SliceLayout(1, CL16))[:, None],)
        cnts = cnts + (gl.zeros([THREADS], gl.int32, gl.SliceLayout(1, CL16)),)
    chunks = lds._reinterpret(gl.int64, [2 * THREADS // 16, 16], FLAT1)
    for c in range(0, (m + 15) // 16):
        oth = chunks.index(c).load(gl.SliceLayout(0, CL16))[None, :]
        ok = c * 16 + jj < m
        nc = ()
        for e in gl.static_range(E):
            nc = nc + (cnts[e] + gl.sum(((oth > mcs[e]) & ok).to(gl.int32), axis=1),)
        cnts = nc
    out = ()
    for e in gl.static_range(E):
        out = out + (gl.convert_layout(cnts[e], T1),)
    return out


@gluon.jit
def rank_cands(lds, lrank, m, NT: gl.constexpr):
    """Candidates 0 .. m (m <= 64 NT) of LDS ``lds`` counted against each other, 64 x 64 pair
    tiles (a thread: its row's key, 8 columns; the row sums within a warp) -> each one's count
    of beating candidates into LDS ``lrank`` (int32, by candidate). Barriers included."""
    gl.barrier()
    for i in gl.static_range(NT):
        if i * 64 < m:
            mine = lds.slice(i * 64, 64).load(gl.SliceLayout(1, TL64))[:, None]
            cnt = gl.zeros([64], gl.int32, gl.SliceLayout(1, TL64))
            for j in gl.static_range(NT):
                if j * 64 < m:
                    oth = lds.slice(j * 64, 64).load(gl.SliceLayout(0, TL64))[None, :]
                    jj = j * 64 + ar(64, layout=gl.SliceLayout(0, TL64))[None, :]
                    cnt = cnt + gl.sum(((oth > mine) & (jj < m)).to(gl.int32), axis=1)
            lrank.slice(i * 64, 64).store(cnt)
    gl.barrier()


@gluon.jit
def rank_entries(keys, ids, lives, lcand, lctr, FLOOR: gl.constexpr = True, tr_=None, bid=0,
                 N_ALL: gl.constexpr = 0):
    """Exact ranks (the count of entries beating each) among a CTA's entries: ``keys`` / ``ids``
    / ``lives`` tuples of [512] T1 (entry t + 512 e of the CTA; dead ones beat nothing).
    FLOOR: only entries at least the group floor (FlyDSL's group_floor: the 16 groups thread %
    16, each group's max, their min -- 16 distinct entries are at least it) can be in the top
    16; those take a slot of LDS ``lcand`` from the counter ``lctr`` (a workgroup atomic,
    FlyDSL's compact) and are ranked among themselves (``rank_cands``); past CAND_CAP of them
    every entry is counted. Else (N_ALL entries, entry t at slot t) all are ranked so. ->
    a tuple of ranks (exact for the top 16; the rest only known to be >= 16)."""
    E: gl.constexpr = len(keys)
    t = ar(THREADS, T1)
    least = t * 0 - 2147483647 - 1
    lrank = lctr  # (the counter's words after it is read: a rank by candidate)
    mines = ()
    for e in gl.static_range(E):
        mines = mines + (gl.where(lives[e], key64(keys[e], ids[e]), key64(least, t * 0)),)
    ranks = ()
    for e in gl.static_range(E):
        ranks = ranks + (t * 0,)
    if FLOOR:
        top = gl.where(lives[0], keys[0], least)
        for e in gl.static_range(1, E):
            top = gl.maximum(top, gl.where(lives[e], keys[e], least))
        gmax = gl.max(gl.convert_layout(gl.reshape(top, [32, 16]), G16), axis=0)
        floor = gl.min(gmax, axis=0)
        stamp(tr_, bid, 36)
        z = t * 0
        gl.barrier()  # lcand's previous readers done
        slots = ()
        cands = ()
        for e in gl.static_range(E):
            ce = lives[e] & (keys[e] >= floor)
            slot = lctr.atomic_scatter_add(z + 1, z, 0, mask=ce)
            lcand.scatter(mines[e], gl.where(ce & (slot < CAND_CAP), slot, 2 * THREADS - 1), 0)
            slots = slots + (slot,)
            cands = cands + (ce,)
        gl.barrier()
        m = gl.max(lctr.load(T1), axis=0)
        stamp(tr_, bid, 37)
        if m <= CAND_CAP:
            rank_cands(lcand, lrank, m, CAND_CAP // 64)
            nr = ()
            for e in gl.static_range(E):
                r = lrank.gather(gl.where(cands[e], gl.minimum(slots[e], CAND_CAP - 1), 0), 0)
                nr = nr + (gl.where(cands[e], r, TOPK_BLOCKS),)
            ranks = nr
        else:  # (rare) every entry counted
            gl.barrier()
            for e in gl.static_range(E):
                lcand.scatter(mines[e], e * THREADS + t, 0)
            gl.barrier()
            ranks = count_beaten(mines, lcand, E * THREADS, 1, True)
    else:
        NA: gl.constexpr = E * THREADS if N_ALL == 0 else N_ALL
        gl.barrier()  # lcand's previous readers done
        for e in gl.static_range(E):
            lcand.scatter(mines[e], e * THREADS + t, 0)
        rank_cands(lcand, lrank, NA, (NA + 63) // 64)
        nr = ()
        for e in gl.static_range(E):
            nr = nr + (lrank.gather(gl.minimum(e * THREADS + t, NA - 1), 0),)
        ranks = nr
    gl.barrier()  # lrank / lcand read
    return ranks


@gluon.jit
def place_slots(keys, ids, pages, ranks, lives, n_blk, seq, p, stab_k, lslot, WRITE: gl.constexpr):
    """Ranked entries -> partition p's two block ids and the sparse context length, as the
    Triton selector emits them: full blocks by rank, the tail block (the token's) last
    (FlyDSL's place). A selected entry writes its block id at its slot of LDS ``lslot`` ([32]
    int32, word 16 zeroed before: set by the tail if selected). WRITE: the token's sparse
    table too (block ids by slot, then the length at word 16)."""
    E: gl.constexpr = len(keys)
    tail = n_blk - 1
    n_sel = gl.minimum(n_blk, TOPK_BLOCKS)
    kt = index_key(tail, tail * 0, n_blk)
    t = ar(THREADS, T1)
    for e in gl.static_range(E):
        sel = lives[e] & (ranks[e] < n_sel)
        is_t = sel & (ids[e] == tail)
        tfirst = (kt > keys[e]) | ((kt == keys[e]) & (tail > ids[e]))
        # (n_sel - 1: the tail's slot when it is selected, as every selection here has it)
        slot = gl.where(is_t, n_sel - 1, ranks[e] - tfirst.to(gl.int32))
        lslot.scatter(gl.where(sel, pages[e], 1), gl.where(sel, slot, gl.where(t == 0, 31, 31)), 0)
        lslot.scatter(gl.where(is_t, 1, 0), gl.where(is_t, 16, 31), 0)
        if WRITE:
            gl.store(stab_k + slot, pages[e], mask=sel)
    gl.barrier()
    sl = lslot.load(gl.BlockedLayout([32], [64], [8], [0]))
    ii = ar(32, layout=gl.BlockedLayout([32], [64], [8], [0]))
    b0 = gl.sum(gl.where(ii == 2 * p, sl, 0), axis=0)
    b1 = gl.sum(gl.where(ii == 2 * p + 1, sl, 0), axis=0)
    tsel = gl.sum(gl.where(ii == 16, sl, 0), axis=0)
    n_full = n_sel - tsel
    n_ctx = gl.where(tsel != 0, n_full * 128 + seq - tail * 128, gl.minimum(n_sel * 128, seq))
    if WRITE:
        gl.store(stab_k + 16 + t * 0, n_ctx + t * 0, mask=t == 0)
    return b0, b1, n_ctx


@gluon.jit
def select_slots(k, p, seq, seq_req, bt, stab_k, sc, tag, errors, bid, tr_=None):
    """Split task (token k, partition p) of a selecting layer, the token's context past
    MAX_SHORT: (block id of slot 2 p, of slot 2 p + 1, sparse context length). Part 0 writes
    the token's sparse table. ``seq_req``: its request's last row's context (decides long)."""
    n_blk = (seq + 127) // 128
    t = ar(THREADS, T1)
    lcand = gl.allocate_shared_memory(gl.int64, [2 * THREADS], FLAT1)
    lctr = gl.allocate_shared_memory(gl.int32, [THREADS], FLAT1)
    lslot = gl.allocate_shared_memory(gl.int32, [32], FLAT1)
    lctr.store(t * 0)
    lslot.store(gl.convert_layout(ar(32, layout=gl.BlockedLayout([1], [64], [8], [0])) * 0,
                                  gl.BlockedLayout([1], [64], [8], [0])))
    isc = rgn(sc, R_ISC)
    if (seq_req + 127) // 128 <= LONG_FROM:
        # every block a thread (b = t), the CTA ranks them all
        b = t
        live = b < n_blk
        need = live & (b < n_blk - 1)
        pk = peek(isc, k * MAX_IB + b, tag, need)
        pg = gl.load(bt + b, mask=live, other=0)
        gl.barrier()
        w = land(isc, k * MAX_IB + b, tag, pk, errors, bid, E_SPLIT)
        stamp(tr_, bid, 46)
        key = index_key(b, w, n_blk)
        (rk,) = rank_entries((key,), (b,), (live,), lcand, lctr, True, tr_, bid)
        stamp(tr_, bid, 47)
        b0, b1, n_ctx = place_slots((key,), (b,), (pg,), (rk,), (live,), n_blk, seq, p, stab_k, lslot, p == 0)
    else:
        # share p: blocks lo .. hi, two a thread; its 16 best to the candidates (key, block)
        span = (n_blk + N_SPLIT - 1) // N_SPLIT
        lo = p * span
        hi = gl.minimum(lo + span, n_blk)
        bs0 = lo + t
        bs1 = lo + THREADS + t
        lv0 = bs0 < hi
        lv1 = bs1 < hi
        n0 = lv0 & (bs0 < n_blk - 1)
        n1 = lv1 & (bs1 < n_blk - 1)
        pk0 = peek(isc, k * MAX_IB + bs0, tag, n0)
        pk1 = peek(isc, k * MAX_IB + bs1, tag, n1)
        gl.barrier()
        w0 = land(isc, k * MAX_IB + bs0, tag, pk0, errors, bid, E_SPLIT)
        w1 = land(isc, k * MAX_IB + bs1, tag, pk1, errors, bid, E_SPLIT)
        stamp(tr_, bid, 46)
        k0 = index_key(bs0, w0, n_blk)
        k1 = index_key(bs1, w1, n_blk)
        r0, r1 = rank_entries((k0, k1), (bs0, bs1), (lv0, lv1), lcand, lctr, True, tr_, bid)
        stamp(tr_, bid, 47)
        cand = rgn(sc, R_CAND)
        cb = (k * N_SPLIT + p) * TOPK_BLOCKS
        put(cand, 2 * (cb + r0), k0, tag, mask=lv0 & (r0 < TOPK_BLOCKS))
        put(cand, 2 * (cb + r0) + 1, bs0, tag, mask=lv0 & (r0 < TOPK_BLOCKS))
        put(cand, 2 * (cb + r1), k1, tag, mask=lv1 & (r1 < TOPK_BLOCKS))
        put(cand, 2 * (cb + r1) + 1, bs1, tag, mask=lv1 & (r1 < TOPK_BLOCKS))
        # the eight shares' candidates (each sent 16 distinct ranks), a thread each, ranked again
        # (all 128 against each other); a candidate's page load goes out under the count
        NC: gl.constexpr = N_SPLIT * TOPK_BLOCKS
        held = t < NC
        ci = k * NC + gl.minimum(t, NC - 1)
        pkk = peek(cand, 2 * ci, tag, held)
        pkb = peek(cand, 2 * ci + 1, tag, held)
        ck = land(cand, 2 * ci, tag, pkk, errors, bid, E_SPLIT)
        cbk = land(cand, 2 * ci + 1, tag, pkb, errors, bid, E_SPLIT)
        stamp(tr_, bid, 48)
        pg = gl.load(bt + gl.maximum(cbk, 0), mask=held, other=0)
        (rc,) = rank_entries((ck,), (cbk,), (held,), lcand, lctr, False, None, bid, NC)
        b0, b1, n_ctx = place_slots((ck,), (cbk,), (pg,), (rc,), (held,), n_blk, seq, p, stab_k, lslot, p == 0)
    return b0, b1, n_ctx


# ============================================================ split attention
@gluon.jit
def stage_own(k, li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps, tr_=None,
              KONLY: gl.constexpr = False):
    """Token k's own key (FlyDSL's k / v head tasks): K = gemma norm / RoPE of the k rows, V
    raw, fp8 -> the cache (vLLM's: unit scale, page-16 slot; for later steps) and to the
    tail split as tagged words (2 fp8 a pair): off its critical path."""
    e = lrow(ptab, li, (16, 17, 4))
    k_cache = lp(e[0], gl.uint8)
    v_cache = lp(e[1], gl.uint8)
    g_k = lp(e[2], gl.int32)
    qkv = rgn(sc, R_QKV)
    kpk = head_peek(qkv, k, HEADS * 64, 64, H1, tag, k >= 0)
    j = ar(64, layout=gl.SliceLayout(0, H1))[None, :]
    vidx = k * QKV_PAIRS + (HEADS + 1) * 64 + j
    vpk = peek(qkv, vidx, tag, j >= 0)
    pos = uniform(gl.load(positions + k).to(gl.int32))
    slot = uniform(gl.load(slots + k).to(gl.int32))
    tabs = rope_tabs(g_k, cos_sin, pos, H1)
    klo, khi, kplo, kphi = head_land(qkv, kpk, tag, errors, bid, E_QKV)
    stamp(tr_, bid, 34)
    ko_lo, ko_hi = head_norm_rope(klo, khi, kplo, kphi, tabs, eps)
    k8lo = clamp_fp8(bf16_round(ko_lo)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    k8hi = clamp_fp8(bf16_round(ko_hi)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    kw = k8lo | (k8hi << 8)
    if not (TAIL_CACHE or TAIL_SELF):
        put(rgn(sc, R_OWN), k * 128 + j, kw, tag)
    if KONLY:
        v8lo = j.to(gl.uint8)
        v8hi = v8lo
    else:
        vw = land(qkv, vidx, tag, vpk, errors, bid, E_QKV)
        v8lo = clamp_fp8(bf_lo(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
        v8hi = clamp_fp8(bf_hi(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    stamp(tr_, bid, 35)
    if not (TAIL_CACHE or TAIL_SELF):
        put(rgn(sc, R_OWN), k * 128 + 64 + j, v8lo.to(gl.int32) | (v8hi.to(gl.int32) << 8), tag)
    # the cache copies (page-16 slot: page p16 / 16, token p16 % 16)
    p16 = slot + (slot // 128) * 128
    pb = p16 // 16
    pt = p16 % 16
    d = 2 * j
    kdst = (pb * 2048 + (d // 16) * 256 + pt * 16 + d % 16) // 2
    cdna4.buffer_store(kw.to(gl.int16), k_cache.to(gl.pointer_type(gl.int16), bitcast=True), kdst,
                       mask=slot >= 0, cache=".wt")
    if not KONLY:
        cdna4.buffer_store(v8lo, v_cache, pb * 2048 + d * 16 + pt, mask=slot >= 0, cache=".wt")
        cdna4.buffer_store(v8hi, v_cache, pb * 2048 + (d + 1) * 16 + pt, mask=slot >= 0, cache=".wt")
    if TAIL_CACHE:
        # FlyDSL's head tasks: the cache rows drained, then one flag (pair k * 128) tells
        # the tail split its partition's last page is whole
        ft = drain(ar(THREADS, T1))
        stamp(tr_, bid, 36)
        put(rgn(sc, R_OWN), k * 128 + ft * 0, ft * 0 + 1, tag, mask=ft == 0)


@gluon.jit
def stage_ownv(k, li, ptab, slots, sc, tag, errors, bid, tr_=None):
    """OWN_SPLIT: token k's own V (raw v rows, fp8) -> the cache, drained, then its flag (pair
    k * 128 + 64): ``stage_own``'s V half, on another CTA."""
    v_cache = lp(lrow(ptab, li, (17, 0))[0], gl.uint8)
    qkv = rgn(sc, R_QKV)
    j = ar(64, layout=gl.SliceLayout(0, H1))[None, :]
    vidx = k * QKV_PAIRS + (HEADS + 1) * 64 + j
    vpk = peek(qkv, vidx, tag, j >= 0)
    slot = uniform(gl.load(slots + k).to(gl.int32))
    vw = land(qkv, vidx, tag, vpk, errors, bid, E_QKV)
    stamp(tr_, bid, 31)
    v8lo = clamp_fp8(bf_lo(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    v8hi = clamp_fp8(bf_hi(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    p16 = slot + (slot // 128) * 128
    pb = p16 // 16
    pt = p16 % 16
    d = 2 * j
    cdna4.buffer_store(v8lo, v_cache, pb * 2048 + d * 16 + pt, mask=slot >= 0, cache=".wt")
    cdna4.buffer_store(v8hi, v_cache, pb * 2048 + (d + 1) * 16 + pt, mask=slot >= 0, cache=".wt")
    ft = drain(ar(THREADS, T1))
    stamp(tr_, bid, 32)
    put(rgn(sc, R_OWN), k * 128 + 64 + ft * 0, ft * 0 + 1, tag, mask=ft == 0)


@gluon.jit
def ktile_b2(ktw):
    """K words [8, 4, 256] (KTW_L) -> B2 bytes [128 d, 256 keys], through LDS (a convert_layout's cost)."""
    sm = gl.allocate_shared_memory(gl.int32, [256, 32], KSW2W)
    sm.store(gl.permute(gl.reshape(ktw, [32, 256]), (1, 0)))
    gl.barrier()
    return sm._reinterpret(gl.uint8, [256, 128], KSW2).permute((1, 0)).load(B2)


@gluon.jit
def vtile_b2(vvw):
    """V words [16, 4, 128] (VTW_L) -> B2 bytes [256 keys, 128 d], through LDS."""
    sm = gl.allocate_shared_memory(gl.int32, [128, 64], KSW2W)
    sm.store(gl.permute(gl.reshape(vvw, [64, 128]), (1, 0)))
    gl.barrier()
    return sm._reinterpret(gl.uint8, [128, 256], KSW2).permute((1, 0)).load(B2)


@gluon.jit
def vmask_w(vp, vw, lim):
    """V words' byte mask: key 16 vp + 4 vw + b (of the partition) kept iff below ``lim``."""
    m = gl.where(vp * 16 + vw * 4 < lim, 255, 0)
    m = m | gl.where(vp * 16 + vw * 4 + 1 < lim, 65280, 0)
    m = m | gl.where(vp * 16 + vw * 4 + 2 < lim, 16711680, 0)
    return m | gl.where(vp * 16 + vw * 4 + 3 < lim, -16777216, 0)


@gluon.jit
def stage_split(task, li, ptab, cos_sin, positions, slots, block_table, bt_stride, seq_lens, sc, tag, errors, bid,
                sm_scale, eps, tr_=None, QL: gl.constexpr = 1, topk=1, scratch=0, S: gl.constexpr = 1):
    """Token k's partition p (keys 256 p .. + 256) x 16 heads -> (o, m, l). QL > 1
    (speculative verify): row k is request k // QL's row jq = k % QL, its context its
    position + 1; the keys rows jq - d of the request insert this launch (d = 0 .. jq)
    enter from the own-key tasks' words, as a token's own key does."""
    k = task // N_SPLIT
    p = task % N_SPLIT
    e = lrow(ptab, li, (16, 17, 3, 4))
    k_cache = lp(e[0], gl.uint8)
    v_cache = lp(e[1], gl.uint8)
    g_q = lp(e[2], gl.int32)
    g_k = lp(e[3], gl.int32)
    if TILE_W:  # the K / V tiles as i32 words
        kcp = k_cache.to(gl.pointer_type(gl.int32), bitcast=True)
        vcp = v_cache.to(gl.pointer_type(gl.int32), bitcast=True)
    else:
        kcp = k_cache
        vcp = v_cache
    pos = uniform(gl.load(positions + k).to(gl.int32))
    if QL > 1:
        seq = pos + 1
        jq = k % QL
    else:
        seq = gl.maximum(uniform(gl.load(seq_lens + k)), 1)
        jq = 0
    key0 = p * SPLIT_KEYS
    spo = rgn(sc, R_SPO)
    qtabs = rope_tabs(g_q, cos_sin, pos, HL)
    tid = tids()
    bt = block_table + (k // QL) * bt_stride
    # past MAX_SHORT (the indexer, S <= LONG_S): partition p = selected slots 2 p, 2 p + 1 of the
    # token's sparse context (sctx keys: full blocks, the tail block last); else the context's
    # keys 256 p .. + 256 in order
    spr = (seq > MAX_SHORT) & (S <= LONG_S)
    if SPLIT_BTPF:
        # the in-step keys' block-table entries (the sparse tail's column lookup below): issued
        # together here, ahead of the selection and the q wait, not 4 serial round trips after
        bl_ = ()
        for d in gl.static_range(QL):
            bl_ = bl_ + (gl.load(bt + gl.maximum(seq - 1 - d, 0) // 128),)
        pgs_ = ()
        for d in gl.static_range(QL):
            pgs_ = pgs_ + (uniform(bl_[d]),)
    if spr:
        stab_k = i32p(as_ptr(scratch, STAB_OFF, gl.int32)) + k * STAB_WORDS
        if topk != 0:
            seq_req = row_seq((k // QL) * QL + QL - 1, positions, seq_lens, QL)
            b0, b1, sctx = select_slots(k, p, seq, seq_req, bt, stab_k, sc, tag, errors, bid, tr_)
            stamp(tr_, bid, 45)
        else:
            b0 = uniform(gl.load(stab_k + 2 * p))
            b1 = uniform(gl.load(stab_k + 2 * p + 1))
            sctx = uniform(gl.load(stab_k + 16))
    else:
        sctx = seq
        b0 = uniform(gl.load(bt + 2 * p))
        b1 = uniform(gl.load(bt + gl.minimum(2 * p + 1, (seq - 1) // 128)))
    if key0 >= sctx:
        # past the context: m = NEG, l = 0 (the merge weighs it 0)
        put(rgn(sc, R_SPM), task * HEADS + tid, gl.full(tid.shape, NEG, gl.float32, tid.type.layout), tag,
            mask=tid < HEADS)
        put(rgn(sc, R_SPL), task * HEADS + tid, gl.full(tid.shape, 0.0, gl.float32, tid.type.layout), tag,
            mask=tid < HEADS)
        put(spo, task * (O_K // 2) + tid, tid * 0, tag)
        put(spo, task * (O_K // 2) + THREADS + tid, tid * 0, tag)
    else:
        # K^T [128 d, 256 keys], V [256 keys, 128 d] of the partition, masked past the context
        kc = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, KT_L)))[:, None, None]
        kb = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, KT_L)))[None, :, None]
        kn = ar(256, layout=gl.SliceLayout(0, gl.SliceLayout(1, KT_L)))[None, None, :]
        kpg = gl.where(kn < 128, b0, b1) * 16 + (kn % 128) // 16
        kok = key0 + kn < sctx
        k_off = kpg * 2048 + kc * 256 + (kn % 16) * 16 + kb
        if TILE_W:
            kc = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, KTW_L)))[:, None, None]
            kb = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(2, KTW_L)))[None, :, None]
            kn = ar(256, layout=gl.SliceLayout(0, gl.SliceLayout(1, KTW_L)))[None, None, :]
            kpg = gl.where(kn < 128, b0, b1) * 16 + (kn % 128) // 16
            kok = key0 + kn < sctx
            k_off = kpg * 512 + kc * 64 + (kn % 16) * 4 + kb
        kt = cdna4.buffer_load(kcp, k_off, mask=kok, other=0)
        vp = ar(16, layout=gl.SliceLayout(1, gl.SliceLayout(2, VT_L)))[:, None, None]
        vt = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, VT_L)))[None, :, None]
        vd = ar(128, layout=gl.SliceLayout(0, gl.SliceLayout(1, VT_L)))[None, None, :]
        vpg = gl.where(vp < 8, b0, b1) * 16 + vp % 8
        v_off = vpg * 2048 + vd * 16 + vt
        if TILE_W:
            vp = ar(16, layout=gl.SliceLayout(1, gl.SliceLayout(2, VTW_L)))[:, None, None]
            vt = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(2, VTW_L)))[None, :, None]
            vd = ar(128, layout=gl.SliceLayout(0, gl.SliceLayout(1, VTW_L)))[None, None, :]
            vpg = gl.where(vp < 8, b0, b1) * 16 + vp % 8
            v_off = vpg * 512 + vd * 4 + vt
        # whole pages (16 B loads), the keys past the context zeroed after: P is 0 there,
        # but a NaN byte would still poison the MFMA
        vv = cdna4.buffer_load(vcp, v_off, mask=key0 + vp * 16 < sctx, other=0)
        pin(tids())  # the K / V loads stay issued before the q wait
        # q: the token's 16 heads from the GEMV rows, normed and roped here
        qkv = rgn(sc, R_QKV)
        # q's rows and (tail) the own-key task's fp8 K / V, all issued before any wait
        tail = ((seq - 1) // SPLIT_KEYS == p) & (DBG_NOTAIL == 0)
        qpk = head_peek(qkv, k, 0, 1024, HL, tag, tail | (seq > 0))
        oj = ar(64, layout=gl.SliceLayout(0, H1))[None, :]
        # the in-step keys d = 0 .. QL - 1 (d = 0: the own key) that fall in this partition:
        # row k - d's words, its key at position seq - 1 - d
        tl = ()
        okp = ()
        ovp = ()
        ocol = ()
        for d in gl.static_range(QL):
            # key d's column in this partition: the context's order, or (sparse) the slot
            # holding its block (2 p: columns 0 .., 2 p + 1: 128 ..), -1 if neither
            pd = seq - 1 - d
            if spr:
                if SPLIT_BTPF:
                    pgd = pgs_[d]
                else:
                    pgd = uniform(gl.load(bt + gl.maximum(pd, 0) // 128))
                cd = gl.where(pgd == b0, pd % 128, gl.where(pgd == b1, 128 + pd % 128, -1))
            else:
                cd = gl.where((pd // SPLIT_KEYS) == p, pd - key0, -1)
            td = (cd >= 0) & (jq >= d) & (DBG_NOTAIL == 0)
            ocol = ocol + (gl.where(td, cd, -1),)
            tl = tl + (td,)
            oidx = (k - d) * 128 + oj
            if TAIL_SELF and TILE_W:
                pass
            elif TAIL_SELF:
                # row k - d's k / v GEMV rows (its own-key task's inputs), issued with q's
                kd_ = gl.maximum(k - d, 0)
                okp = okp + (head_peek(qkv, kd_, HEADS * 64, 64, H1, tag, td),)
                ovp = ovp + (peek(qkv, kd_ * QKV_PAIRS + (HEADS + 1) * 64 + oj, tag, td & (oj >= 0)),)
            else:
                okp = okp + (peek(rgn(sc, R_OWN), oidx, tag, td & (oj >= TAIL_CACHE * 64)),)
                ovp = ovp + (peek(rgn(sc, R_OWN), oidx + 64, tag, td & (oj >= TAIL_CACHE * 64)),)
        anyt = tl[0]
        for d in gl.static_range(1, QL):
            anyt = anyt | tl[d]
        if TAIL_SELF and TILE_W:
            # the in-step keys' rows k - d (d = 0 .. 3): their k, k RoPE partner and v GEMV rows as
            # one [16, 64] block in q's layout (row 4 g + d: g = 0 k, 1 partner, 2 v), landed in
            # q's wait; their RoPE tables, issued with it
            rr_ = ar(16, layout=gl.SliceLayout(1, HL))[:, None]
            jh = ar(64, layout=gl.SliceLayout(0, HL))[None, :]
            dr = rr_ % 4
            gr = rr_ // 4
            tdr = (dr == 0) & tl[0]
            for d in gl.static_range(1, QL):
                tdr = tdr | ((dr == d) & tl[d])
            pjh = gl.where(jh < 16, jh + 16, jh - 16)
            kvidx = (gl.maximum(k - dr, 0) * QKV_PAIRS + HEADS * 64 + gl.where(gr == 1, pjh, jh)
                     + gl.where(gr == 2, 64, 0))
            kvpk = peek(qkv, kvidx, tag, tdr & (gr < 3) & ((gr != 1) | (jh < 32)))
            dd = ar(4, layout=gl.SliceLayout(1, H4))[:, None]
            jj = ar(64, layout=gl.SliceLayout(0, H4))[None, :]
            pjj = gl.where(jj < 16, jj + 16, jj - 16)
            posr = gl.maximum(pos - dd, 0) if QL > 1 else pos + dd * 0
            ktb4 = (gl.load(g_k + jj), gl.load(g_k + pjj, mask=jj < 32, other=0),
                    gl.load(cos_sin + posr * 32 + jj % 16), gl.load(cos_sin + posr * 32 + 16 + jj % 16))
        if TAIL_CACHE and (not TAIL_SELF):
            if anyt:
                # FlyDSL's split: the own-key tasks' flags of every row whose key falls here
                # (lane d: row k - d), one wait, then the partition re-read past every cache
                # (this CTA's L2 holds the pages without those keys)
                ft = ar(THREADS, T1)
                tlv = ft < 0
                for d in gl.static_range(QL):
                    tlv = tlv | ((ft == d) & tl[d])
                    if own_split(S):
                        tlv = tlv | ((ft == QL + d) & tl[d])
                if own_split(S):
                    fidx = (k - gl.where(ft < QL, ft, ft - QL)) * 128 + gl.where(ft < QL, 0, 64)
                else:
                    fidx = (k - ft) * 128
                fpk = peek(rgn(sc, R_OWN), fidx, tag, tlv)
                land(rgn(sc, R_OWN), fidx, tag, fpk, errors, bid, E_QKV)
                stamp(tr_, bid, 37)
                if TAIL_ACQ:
                    l2_acquire(ft)
                if TAIL_PATCH and (not TILE_W):
                    # only the in-step keys are stale in the tiles: their K columns (16 B a lane, one key)
                    # and the 16-key V groups holding them re-read, in the tiles' own layouts, and
                    # selected in -- a few hundred bytes, not the 64 KB partition
                    ink = kn < 0
                    inv = vp < 0
                    for d in gl.static_range(QL):
                        ink = ink | (kn == ocol[d])
                        inv = inv | ((ocol[d] >= 0) & (vp == gl.maximum(ocol[d], 0) // 16))
                    kt2 = cdna4.buffer_load(kcp, k_off, mask=kok & ink, other=0, cache=".cv")
                    vv2 = cdna4.buffer_load(vcp, v_off, mask=(key0 + vp * 16 < sctx) & inv, other=0, cache=".cv")
                    kt = gl.where(ink, kt2, kt)
                    vv = gl.where(inv, vv2, vv)
                else:
                    kt = cdna4.buffer_load(kcp, k_off, mask=kok, other=0, cache=".cv")
                    vv = cdna4.buffer_load(vcp, v_off, mask=key0 + vp * 16 < sctx, other=0, cache=".cv")
        if TAIL_SELF and TILE_W:
            qw = wait_group(((qkv, qpk[0], qpk[1]), (qkv, qpk[2], qpk[3]), (qkv, kvidx, kvpk)), tag, errors, bid,
                            E_QKV, 2)
            lo, hi, plo, phi = bf_lo(qw[0]), bf_hi(qw[0]), bf_lo(qw[1]), bf_hi(qw[1])
        else:
            lo, hi, plo, phi = head_land(qkv, qpk, tag, errors, bid, E_QKV, 2 * HEAD_WG)
        q_lo, q_hi = head_norm_rope(lo, hi, plo, phi, qtabs, eps)
        q8 = gl.reshape(gl.join(bf16_round(q_lo).to(gl.float8e4nv), bf16_round(q_hi).to(gl.float8e4nv)),
                        [16, 128])
        stamp(tr_, bid, 13)
        if TAIL_SELF and TILE_W:
            if anyt:
                # the in-step keys built here (stage_own's math in its lane order: the cache's bytes),
                # all four rows at once, into their columns of the K / V word tiles: no wait on the
                # own-key tasks and no re-read of the partition
                g3 = gl.convert_layout(gl.permute(gl.reshape(qw[2], [4, 4, 64]), (1, 2, 0)), G4)
                ge, go = gl.split(gl.reshape(g3, [4, 64, 2, 2]))  # groups 0, 2 | 1, 3
                kr, vr4 = gl.split(ge)
                pr, _ = gl.split(go)
                kr = gl.convert_layout(kr, H4)
                pr = gl.convert_layout(pr, H4)
                stamp(tr_, bid, 27)
                ko_lo, ko_hi = head_norm_rope(bf_lo(kr), bf_hi(kr), bf_lo(pr), bf_hi(pr), ktb4, eps)
                stamp(tr_, bid, 28)
                k8lo = clamp_fp8(bf16_round(ko_lo)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
                k8hi = clamp_fp8(bf16_round(ko_hi)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
                k16 = gl.convert_layout(k8lo | (k8hi << 8), H2)  # [4 rows, 64]: dims 2 j, 2 j + 1
                wa, wb = gl.split(gl.reshape(k16, [4, 32, 2]))
                kd = gl.convert_layout(gl.permute(gl.reshape(wa | (wb << 16), [4, 8, 4]), (1, 2, 0)), KD_L)
                ke, ko = gl.split(gl.reshape(kd, [8, 4, 2, 2]))  # keys 0, 2 | 1, 3
                k0_, k2_ = gl.split(ke)
                k1_, k3_ = gl.split(ko)
                kds = (k0_, k1_, k2_, k3_)
                stamp(tr_, bid, 34)
                v8lo = clamp_fp8(bf_lo(vr4)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                v8hi = clamp_fp8(bf_hi(vr4)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                vb4 = gl.convert_layout(gl.permute(gl.reshape(gl.join(v8lo, v8hi), [4, 128]).to(gl.int32), (1, 0)),
                                        VD4)
                ve, vo = gl.split(gl.reshape(vb4, [128, 2, 2]))
                v0_, v2_ = gl.split(ve)
                v1_, v3_ = gl.split(vo)
                vds = (v0_, v1_, v2_, v3_)
                stamp(tr_, bid, 35)
                for d in gl.static_range(QL):
                    kdw = gl.convert_layout(kds[d], gl.SliceLayout(2, KTW_L))
                    kt = gl.where(kn == ocol[d], kdw[:, :, None], kt)
                    vdb = gl.convert_layout(vds[d], gl.SliceLayout(0, gl.SliceLayout(1, VTW_L)))[None, None, :]
                    c_ = gl.maximum(ocol[d], 0)
                    sh = 8 * (c_ % 4)
                    hit = (ocol[d] >= 0) & (vp == c_ // 16) & (vt == (c_ % 16) // 4)
                    vv = gl.where(hit, (vv & ~(255 << sh)) | (vdb << sh), vv)
                stamp(tr_, bid, 37)
        qf_lo = bf16_round(q_lo).to(gl.float8e4nv).to(gl.float32)
        qf_hi = bf16_round(q_hi).to(gl.float8e4nv).to(gl.float32)
        if TAIL_LATE and (not TAIL_CACHE) and (not TILE_W) and (not TAIL_SELF):
            # QK of the cached keys first, pinned: the in-step keys' words (the own-key tasks')
            # land under it, their columns replaced after
            s_early = pin(cdna4.mfma_scaled(gl.convert_layout(q8, A2), unit_scales([16, 4], A2), "e4m3",
                                            gl.convert_layout(gl.reshape(kt, [128, 256]), B2).to(gl.float8e4nv,
                                                                                                  bitcast=True),
                                            unit_scales([256, 4], B2), "e4m3", zacc([16, 256], MFMA2)))
        # each in-step key's score q8 . k8 (fp8 products, exact in f32) replaces its column
        sd = ()
        for d in gl.static_range(QL):
            s_d = gl.zeros([16], gl.float32, gl.SliceLayout(1, MFMA2))
            if TAIL_CACHE:
                pass
            elif tl[d] & ((DBG_NOTAIL & 2) == 0):
                if TAIL_SELF:
                    # stage_own's K words of row k - d, built here (no hop through the own-key task)
                    klo, khi, kplo, kphi = head_land(qkv, okp[d], tag, errors, bid, E_QKV)
                    ptb = rope_tabs(g_k, cos_sin, pos - d if QL > 1 else pos, H1)
                    ko_lo, ko_hi = head_norm_rope(klo, khi, kplo, kphi, ptb, eps)
                    kw = (clamp_fp8(bf16_round(ko_lo)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
                          | (clamp_fp8(bf16_round(ko_hi)).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32) << 8))
                else:
                    kw = land(rgn(sc, R_OWN), (k - d) * 128 + oj, tag, okp[d], errors, bid, E_QKV)
                kf_lo = gl.convert_layout(gl.reshape((kw & 255).to(gl.uint8).to(gl.float8e4nv, bitcast=True)
                                                     .to(gl.float32), [64]), gl.SliceLayout(0, HL))
                kf_hi = gl.convert_layout(gl.reshape(((kw >> 8) & 255).to(gl.uint8).to(gl.float8e4nv, bitcast=True)
                                                     .to(gl.float32), [64]), gl.SliceLayout(0, HL))
                s_d = gl.convert_layout(gl.sum(qf_lo * kf_lo[None, :] + qf_hi * kf_hi[None, :], axis=1),
                                        gl.SliceLayout(1, MFMA2))
            sd = sd + (s_d,)
        stamp(tr_, bid, 26)
        # QK: [16 heads, 128] x [128, 256 keys]
        qa = gl.convert_layout(q8, A2)
        if TILE_W:
            kb2 = ktile_b2(kt)
        elif TAIL_CACHE and TAIL_SELF:
            if anyt:
                # the in-step keys' K built here (stage_own's math: the cache's bytes) over their
                # rows of the tile in LDS: no wait on the own-key tasks and no re-read
                tbs = ()
                for d in gl.static_range(QL):
                    tbs = tbs + (rope_tabs(g_k, cos_sin, pos - d if QL > 1 else pos, H1),)
                ksm = gl.allocate_shared_memory(gl.uint8, [256, 128], ROWSH)
                ksm.store(gl.permute(gl.reshape(kt, [128, 256]), (1, 0)))
                kds = ()
                for d in gl.static_range(QL):
                    klo, khi, kplo, kphi = head_land(qkv, okp[d], tag, errors, bid, E_QKV)
                    ko_lo, ko_hi = head_norm_rope(klo, khi, kplo, kphi, tbs[d], eps)
                    k8lo = clamp_fp8(bf16_round(ko_lo)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                    k8hi = clamp_fp8(bf16_round(ko_hi)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                    kds = kds + (gl.convert_layout(gl.reshape(gl.join(k8lo, k8hi), [128]), R128),)
                gl.barrier()
                for d in gl.static_range(QL):
                    if tl[d]:
                        ksm._reinterpret(gl.uint8, [256, 128], ROW1).index(gl.maximum(ocol[d], 0)).store(kds[d])
                gl.barrier()
                kb2 = ksm.permute((1, 0)).load(B2)
                stamp(tr_, bid, 37)
            else:
                kb2 = gl.convert_layout(gl.reshape(kt, [128, 256]), B2)
        else:
            kb2 = gl.convert_layout(gl.reshape(kt, [128, 256]), B2)
        if TAIL_LATE and (not TAIL_CACHE) and (not TILE_W) and (not TAIL_SELF):
            s = s_early
        else:
            s = cdna4.mfma_scaled(qa, unit_scales([16, 4], A2), "e4m3", kb2.to(gl.float8e4nv, bitcast=True),
                                  unit_scales([256, 4], B2), "e4m3", zacc([16, 256], MFMA2))
        kidx = ar(256, layout=gl.SliceLayout(0, MFMA2))[None, :]
        notown = kidx >= 0
        if not TAIL_CACHE:
            for d in gl.static_range(QL):
                s = gl.where(kidx == ocol[d], sd[d][:, None], s)
                notown = notown & (kidx != ocol[d])
        s = gl.where(key0 + kidx < sctx, sm_scale * s, NEG)
        mx = gl.max(s, axis=1)
        pj = hw_exp2((s - mx[:, None]) * LOG2E)
        ls = gl.sum(pj, axis=1)
        stamp(tr_, bid, 15)
        # P in fp8 (448 p); the in-step keys' columns stay out of the MFMA (their V rows are
        # zero there) and join the accumulator from registers: o += p_d v_d (exact products)
        pf = (448.0 * pj).to(gl.float8e4nv).to(gl.float32)
        p8 = gl.where((key0 + kidx < sctx) & notown, pf, 0.0).to(gl.float8e4nv)
        pa = gl.convert_layout(p8, A2)
        if TILE_W:
            if key0 + SPLIT_KEYS > sctx:
                # the keys past the (sparse) context zeroed: P is 0 there, a NaN byte is not
                vv = vv & vmask_w(vp, vt, sctx - key0)
            vb2 = vtile_b2(vv)
        elif spr:
            # sparse: the rows past the sparse context (a NaN byte would poison the MFMA) and
            # (in-step keys from words) those keys' rows, stale in the cache, zeroed
            if SPLIT_NOBR:
                vr = vp * 16 + vt
                vok = key0 + vr < sctx
                if not TAIL_CACHE:
                    for d in gl.static_range(QL):
                        vok = vok & (vr != ocol[d])
                vv = gl.where(vok, vv, 0)
            elif anyt | (key0 + SPLIT_KEYS > sctx):
                vr = vp * 16 + vt
                vok = key0 + vr < sctx
                if not TAIL_CACHE:
                    for d in gl.static_range(QL):
                        vok = vok & (vr != ocol[d])
                vv = gl.where(vok, vv, 0)
        elif TAIL_CACHE:
            pass
        elif anyt & ((DBG_NOTAIL & 4) == 0):
            # the in-step keys' rows (stale in the cache) zeroed: every row past seq - 1 - jq
            vv = gl.where(key0 + vp * 16 + vt < seq - 1 - jq, vv, 0)
            stamp(tr_, bid, 27)
        if TILE_W:
            pass
        elif TAIL_CACHE and (not spr):
            if SPLIT_NOBR:
                vv = gl.where(key0 + vp * 16 + vt < sctx, vv, 0)
            elif key0 + SPLIT_KEYS > sctx:
                # the last page's keys past the context zeroed (P is 0 there, a NaN byte is not)
                vv = gl.where(key0 + vp * 16 + vt < sctx, vv, 0)
        if TILE_W:
            pass
        elif TAIL_CACHE and TAIL_SELF:
            if anyt:
                # the in-step keys' V rows likewise (raw v, fp8), into their rows of the tile
                for d in gl.static_range(QL):
                    vw = land(qkv, gl.maximum(k - d, 0) * QKV_PAIRS + (HEADS + 1) * 64 + oj, tag, ovp[d], errors,
                              bid, E_QKV)
                    v8lo = clamp_fp8(bf_lo(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                    v8hi = clamp_fp8(bf_hi(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                    vd8 = gl.convert_layout(gl.reshape(gl.join(v8lo, v8hi), [128]),
                                            gl.SliceLayout(0, gl.SliceLayout(1, VT_L)))[None, None, :]
                    vv = gl.where(vp * 16 + vt == ocol[d], vd8, vv)
                vb2 = gl.convert_layout(gl.reshape(vv, [256, 128]), B2)
            else:
                vb2 = gl.convert_layout(gl.reshape(vv, [256, 128]), B2)
        else:
            vb2 = gl.convert_layout(gl.reshape(vv, [256, 128]), B2)
        o = cdna4.mfma_scaled(pa, unit_scales([16, 8], A2), "e4m3", vb2.to(gl.float8e4nv, bitcast=True),
                              unit_scales([128, 8], B2), "e4m3", zacc([16, 128], MFMA2))
        if not TAIL_CACHE:
            for d in gl.static_range(QL):
                if tl[d] & ((DBG_NOTAIL & 4) == 0):
                    # key d's V (``stage_own`` of row k - d), after QK / softmax
                    if TAIL_SELF:
                        vw = land(qkv, gl.maximum(k - d, 0) * QKV_PAIRS + (HEADS + 1) * 64 + oj, tag, ovp[d], errors,
                                  bid, E_QKV)
                        v8lo = clamp_fp8(bf_lo(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                        v8hi = clamp_fp8(bf_hi(vw)).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                    else:
                        vw = land(rgn(sc, R_OWN), (k - d) * 128 + oj + 64, tag, ovp[d], errors, bid, E_QKV)
                        v8lo = (vw & 255).to(gl.uint8)
                        v8hi = ((vw >> 8) & 255).to(gl.uint8)
                    stamp(tr_, bid, 28)
                    vf = gl.join(v8lo.to(gl.float8e4nv, bitcast=True).to(gl.float32),
                                 v8hi.to(gl.float8e4nv, bitcast=True).to(gl.float32))
                    v_d = gl.convert_layout(gl.reshape(vf, [128]), gl.SliceLayout(0, MFMA2))
                    # column d of P, from the per-head score and max directly (the same ops: no
                    # cross-warp reduction, which waits for every wave)
                    p_d = (448.0 * hw_exp2((sm_scale * sd[d] - mx) * LOG2E)).to(gl.float8e4nv).to(gl.float32)
                    o = o + p_d[:, None] * v_d[None, :]
                    stamp(tr_, bid, 14)
        inv_l = 1.0 / gl.where(ls > 0.0, ls, 1.0)
        o = (o * (1.0 / 448.0)) * inv_l[:, None]
        stamp(tr_, bid, 16)
        # publish: bf16 pairs [16 heads, 64], m, l
        OL: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0])
        o2 = gl.convert_layout(o, OL)
        o_lo, o_hi = gl.split(gl.reshape(o2, [16, 64, 2]))
        hh = ar(16, layout=gl.SliceLayout(1, o_lo.type.layout))[:, None]
        jj = ar(64, layout=gl.SliceLayout(0, o_lo.type.layout))[None, :]
        put(spo, task * (O_K // 2) + hh * 64 + jj, bf_pair(o_lo, o_hi), tag)
        h1 = ar(16, layout=gl.SliceLayout(1, MFMA2))
        put(rgn(sc, R_SPM), task * HEADS + h1, mx, tag)
        put(rgn(sc, R_SPL), task * HEADS + h1, ls, tag)


# ============================================================ merge + o_proj + attention all-reduce
ML = gl.constexpr(gl.BlockedLayout([1, 1, 2], [1, 2, 32], [1, 8, 1], [2, 1, 0]))  # [8 parts, 16 heads, 64 pairs]


@gluon.jit
def merge_token(k, sc, tag, errors, bid, tr_=None):
    """Token k's partitions merged (weights by l exp(m - max m)) and quantized per token
    (amax / 448) -> (fp8 lo, fp8 hi) [16 heads, 64 pairs], x_scale."""
    pi = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, ML)))[:, None, None]
    hi3 = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, ML)))[None, :, None]
    ji = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, ML)))[None, None, :]
    MH: gl.constexpr = gl.SliceLayout(2, ML)
    if MERGE_CML:
        CL: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [1, 8], [1, 0])  # [8 parts, 16 heads]: 2 warps' lanes
        pm = ar(8, layout=gl.SliceLayout(1, CL))[:, None]
        hm2 = ar(16, layout=gl.SliceLayout(0, CL))[None, :]
    else:
        pm = ar(8, layout=gl.SliceLayout(1, MH))[:, None]
        hm2 = ar(16, layout=gl.SliceLayout(0, MH))[None, :]
    sp0 = k * N_SPLIT
    im = (sp0 + pm) * HEADS + hm2
    io = (sp0 + pi) * (O_K // 2) + hi3 * 64 + ji
    if O_LGATE:
        # FlyDSL's merge wait: two waves poll the partitions' l, the CTA waits at a
        # barrier, then every thread's m, l, o loads land in one round trip (512
        # threads spinning on 64 KB each load the memory system the split reads)
        lt = ar(THREADS, T1)
        if O_LGATE == 2:  # one word a split: its last head's l (the split's last put)
            gate_poll(rgn(sc, R_SPL), (sp0 + lt) * HEADS + HEADS - 1, lt < N_SPLIT, tag, errors, bid, E_SPLIT)
        else:
            lp_ = peek(rgn(sc, R_SPL), sp0 * HEADS + lt, tag, lt < N_SPLIT * HEADS)
            wait_group(((rgn(sc, R_SPL), sp0 * HEADS + lt, lp_),), tag, errors, bid, E_SPLIT, 1)
            gl.barrier()
    pkm = peek(rgn(sc, R_SPM), im, tag, pm < 8)
    pkl = peek(rgn(sc, R_SPL), im, tag, pm < 8)
    pko = peek(rgn(sc, R_SPO), io, tag, pi < 8)
    stamp(tr_, bid, 31)
    wml = wait_group(((rgn(sc, R_SPM), im, pkm), (rgn(sc, R_SPL), im, pkl)), tag, errors, bid, E_SPLIT,
                     1 if MERGE_CML else 8)
    mw = wml[0]
    lw = wml[1]
    if MERGE_CML == 2:
        # the math in ML's slice as before (its sum over the 8 partitions runs in a thread's
        # registers, in order: bit-identical to the original; CML=1's cross-lane sum is not)
        mw = gl.convert_layout(mw, MH)
        lw = gl.convert_layout(lw, MH)
    m_i = mw.to(gl.float32, bitcast=True)
    l_i = lw.to(gl.float32, bitcast=True)
    gm = gl.max(m_i, axis=0)
    sl = l_i * hw_exp2((m_i - gm[None, :]) * LOG2E)
    gs = gl.sum(sl, axis=0)
    gs = gl.where(gs > 0.0, gs, 1.0)
    stamp(tr_, bid, 32)
    wgt = div_mk(sl, gs[None, :], gl.div_rn(gl.full(gs.shape, 1.0, gl.float32, gs.type.layout), gs)[None, :])
    if MERGE_CML == 1:
        wgt = gl.convert_layout(wgt, MH)
    wgt = wgt[:, :, None]
    ow = wait_group(((rgn(sc, R_SPO), io, pko),), tag, errors, bid, E_SPLIT, 16)[0]
    stamp(tr_, bid, 33)
    o_lo = bf_lo(ow)
    o_hi = bf_hi(ow)
    av_lo = bf16_round(gl.sum(wgt * o_lo, axis=0))
    av_hi = bf16_round(gl.sum(wgt * o_hi, axis=0))
    amax = gl.max(gl.max(gl.maximum(gl.abs(av_lo), gl.abs(av_hi)), axis=1), axis=0)
    x_scale = amax * (1.0 / 448.0)
    inv = gl.where(amax == 0.0, 0.0, hw_rcp(x_scale))
    return (av_lo * inv).to(gl.float8e4nv), (av_hi * inv).to(gl.float8e4nv), x_scale


@gluon.jit
def stage_merge(k, sc, tag, errors, bid, tr_=None):
    """Merge task (S > 1): token k's partitions merged once -> its FP8 o input as words (4
    fp8 a pair, head-major) in "oq", then its scale in "oqs" (FlyDSL's stage_merge)."""
    q_lo, q_hi, x_scale = merge_token(k, sc, tag, errors, bid, tr_)
    oq_put(k, q_lo, q_hi, x_scale, sc, tag)


@gluon.jit
def stage_oin(k, attn_in, sc, tag):
    """Split hybrid, launch B: token k's attention output as vLLM computed it ([16 heads x 128]
    bf16, pairs as words) quantized per token (amax / 448) -> "oq" words + "oqs" scale, as a
    merge task would."""
    L2: gl.constexpr = gl.SliceLayout(0, ML)
    h = ar(16, layout=gl.SliceLayout(1, L2))[:, None]
    j = ar(64, layout=gl.SliceLayout(0, L2))[None, :]
    w = cdna4.buffer_load(i32p(attn_in), k * (O_K // 2) + h * 64 + j, cache=".cv")
    av_lo = bf_lo(w)
    av_hi = bf_hi(w)
    amax = gl.max(gl.max(gl.maximum(gl.abs(av_lo), gl.abs(av_hi)), axis=1), axis=0)
    x_scale = amax * (1.0 / 448.0)
    inv = gl.where(amax == 0.0, 0.0, hw_rcp(x_scale))
    oq_put(k, (av_lo * inv).to(gl.float8e4nv), (av_hi * inv).to(gl.float8e4nv), x_scale, sc, tag)


@gluon.jit
def oq_put(k, q_lo, q_hi, x_scale, sc, tag):
    """Token k's FP8 o input -> "oq" words (4 fp8 a pair, head-major), then its scale -> "oqs"."""
    w2 = q_lo.to(gl.uint8, bitcast=True).to(gl.int32) | (q_hi.to(gl.uint8, bitcast=True).to(gl.int32) << 8)
    a, c = gl.split(gl.reshape(w2, [16, 32, 2]))
    word = a | (c << 16)  # [16 heads, 32 words]: pairs 2 j, 2 j + 1
    WL: gl.constexpr = word.type.layout
    h = ar(16, layout=gl.SliceLayout(1, WL))[:, None]
    j = ar(32, layout=gl.SliceLayout(0, WL))[None, :]
    lt = ar(THREADS, T1)
    if PLAIN_HO:
        cdna4.buffer_store(word, i32p(rgn(sc, R_OQ)), k * (O_K // 4) + h * 32 + j, cache=".wt")
        lt = drain(lt)
    else:
        put(rgn(sc, R_OQ), k * (O_K // 4) + h * 32 + j, word, tag)
        gl.barrier()
    put(rgn(sc, R_OQS), k + lt * 0, x_scale + lt * 0.0, tag, mask=lt == 0)


@gluon.jit
def push_peers(w0, w1, peers, rank, t, tag, S: gl.constexpr):
    """Task t's bf16 partial words (tiles tt = 0, 1: [8 row pairs, 4 tokens] each, PR_L) to
    every rank's ``peers[r]`` region (rows (rank, token), pairs 16 t + 8 tt + j), FlyDSL's
    push: wave r stores to rank r, a word a lane. PR_L holds each word on 16 threads: every
    wave storing all of them to all ranks put ~32 system-scope stores a thread on the link
    (ablation: ~2.3 us from the o GEMM's result to its puts issued)."""
    w = gl.convert_layout(gl.join(w0, w1), gl.SliceLayout(0, PW4))  # [8, 4, 2]
    wb = gl.expand_dims(w, 0) + gl.zeros([8, 1, 1, 1], gl.int32, PW4)
    wv = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, PW4))))[:, None, None, None]
    jr = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, PW4))))[None, :, None, None]
    nn = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, PW4))))[None, None, :, None]
    tt = ar(2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, PW4))))[None, None, None, :]
    at = (rank * MAX_S + nn) * HPAIRS + 16 * t + 8 * tt + jr + wv * 0
    ok = (nn < S) & (at >= 0)
    for r in gl.static_range(4):
        put_sys(peers[r], at, wb, tag, mask=ok & (wv == r))


@gluon.jit
def ar_recv(own, t, tag, errors, bid, S: gl.constexpr, code: gl.constexpr, hreg=None):
    """Task t's rows of an all-reduce (``own``: the 4 ranks' partials, rank-major) summed in rank
    order by wave 0 alone, a word a lane (FlyDSL's sum_partials; every wave of PR_L loading
    every word cost ~60 loads a wave): -> (pair index, wave-0 mask, sum lo, sum hi, h words)
    in PW4; ``hreg``: the residual h region, landed in the same wait."""
    wv = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, PW4))))[:, None, None, None]
    jr = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, PW4))))[None, :, None, None]
    nn = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, PW4))))[None, None, :, None]
    tt = ar(2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, PW4))))[None, None, None, :]
    pw = nn * HPAIRS + 16 * t + 8 * tt + jr + wv * 0
    ok = (nn < S) & (wv == 0)
    pks = ()
    for r in gl.static_range(4):
        pks = pks + (peek(own, r * (MAX_S * HPAIRS) + pw, tag, ok),)
    if hreg is not None:
        pks = pks + (peek(hreg, pw, tag, ok),)
    items = ()
    for r in gl.static_range(4):
        items = items + ((own, r * (MAX_S * HPAIRS) + pw, pks[r]),)
    if hreg is not None:
        items = items + ((hreg, pw, pks[4]),)
    ws = wait_group(items, tag, errors, bid, code, 1)
    t_lo = bf_lo(ws[0])
    t_hi = bf_hi(ws[0])
    for r in gl.static_range(1, 4):
        t_lo = t_lo + bf_lo(ws[r])
        t_hi = t_hi + bf_hi(ws[r])
    hw = ws[4] if hreg is not None else ws[0]
    return pw, ok, t_lo, t_hi, hw


@gluon.jit
def o_task_mma(t, sc, tag, errors, bid, aq, ws, xs, nn, jr, rank, attn_peer, S: gl.constexpr, NS: gl.constexpr,
               AGRS: gl.constexpr, tr_=None):
    """o task t up to 16 rows: the tokens' FP8 rows -> LDS, the MFMA, the partial puts."""
    xsc = gl.zeros([8, NS], gl.float32, PR_L)
    if MERGE_TASKS and S > 1:
        # the merge tasks' rows (FlyDSL's stage_merge): each token's scale (put after its
        # words), then 2 KB a token instead of every token's 64 KB of partitions
        SP: gl.constexpr = triton.next_power_of_2(S)
        lt = ar(THREADS, T1)
        gate_poll(rgn(sc, R_OQS), lt, lt < S, tag, errors, bid, E_OQ)
        k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XW3)))[:, None, None]
        h3 = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, XW3)))[None, :, None]
        w3 = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, XW3)))[None, None, :]
        iw = k3 * (O_K // 4) + h3 * 32 + w3
        isc = nn + jr * 0
        if PLAIN_HO:  # (the scales' gate passed: the words are in)
            wv = cdna4.buffer_load(i32p(rgn(sc, R_OQ)), iw, mask=(k3 < S) & (w3 >= 0), other=0, cache=".cv")
            pks = peek(rgn(sc, R_OQS), isc, tag, isc < S)
        else:
            pkw = peek(rgn(sc, R_OQ), iw, tag, k3 < S)
            pks = peek(rgn(sc, R_OQS), isc, tag, isc < S)
            stamp(tr_, bid, 31)
            wv = wait_group(((rgn(sc, R_OQ), iw, pkw),), tag, errors, bid, E_OQ, SP)[0]
        b0 = (wv & 255).to(gl.uint8)
        b1 = ((wv >> 8) & 255).to(gl.uint8)
        b2 = ((wv >> 16) & 255).to(gl.uint8)
        b3 = ((wv >> 24) & 255).to(gl.uint8)
        x = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [SP, 4, 4, 128]), (1, 0, 2, 3))
        xs.slice(0, SP * 4, dim=1).store(gl.reshape(x, [4, SP * 4, 128]))
        xsc = wait_group(((rgn(sc, R_OQS), isc, pks),), tag, errors, bid, E_OQ, 1)[0].to(gl.float32, bitcast=True)
        stamp(tr_, bid, 33)
    else:
        for k in gl.static_range(S):
            q_lo, q_hi, x_scale = merge_token(k, sc, tag, errors, bid, tr_)
            q8 = gl.reshape(gl.join(q_lo, q_hi), [16, 128])
            xs.slice(4 * k, 4, dim=1).store(gl.reshape(q8.to(gl.uint8, bitcast=True), [4, 4, 128]))
            xsc = gl.where(nn == k, x_scale, xsc)
    stamp(tr_, bid, 63)
    gl.barrier()
    if O_B4 and (S > SMALL_FROM):  # (xs holds 16 tokens; the rows past S are garbage, their columns unused)
        red = gl.allocate_shared_memory(gl.float32, [16, 16, 8], FLAT3)
        los, his = o_mma4(aq, xs, red, 16, NS)
    else:
        acc = zacc([128, 4 * NS], MFMA)
        one_a = unit_scales([128, 4], A_L)
        one_b = unit_scales([4 * NS, 4], B_L)
        for r in gl.static_range(4):
            bv = rnd(xs, r).permute((1, 0)).load(B_L)
            acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                    bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
        red = gl.allocate_shared_memory(gl.float32, [128, NS], FLAT2)
        los, his = bd_reduce(acc, 2, 4, red)
    stamp(tr_, bid, 24)
    live = nn < S
    if PUSH_W and (NS == 4) and (not AGRS):
        push_peers(bf_pair(bf16_round(los[0] * xsc * ws[0][0]), bf16_round(his[0] * xsc * ws[0][1])),
                   bf_pair(bf16_round(los[1] * xsc * ws[1][0]), bf16_round(his[1] * xsc * ws[1][1])),
                   attn_peer, rank, t, tag, S)
    else:
        for tt in gl.static_range(2):
            lo = bf16_round(los[tt] * xsc * ws[tt][0])
            hi = bf16_round(his[tt] * xsc * ws[tt][1])
            word = bf_pair(lo, hi)
            at = (rank * MAX_S + nn) * HPAIRS + 16 * t + 8 * tt + jr
            if AGRS:
                put_sys(peer_pick(attn_peer, t % 4), at, word, tag, mask=live)  # to the rows' owner only
            else:
                for r in gl.static_range(4):
                    put_sys(attn_peer[r], at, word, tag, mask=live)
    stamp(tr_, bid, 25)


MF4 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[4, 2, 1]))
A4 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MF4, k_width=16))
B4 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MF4, k_width=16))
# unit scales only: the warp bases' order does not matter for constant values
SL4A = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[], lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
    warp_bases=[[1, 0, 0], [2, 0, 0], [0, 16, 0]], block_bases=[], shape=[4, 32, 4]))
SL4B = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[], lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
    warp_bases=[[1, 0, 0], [2, 0, 0], [0, 0, 0]], block_bases=[], shape=[4, 16, 4]))
BSW3 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=4, order=[2, 1, 0]))


@gluon.jit
def w8_tile4(base, row_term):
    """``w8_tile`` as the 4-slice batched A [4 K slices, 32 rows, 128 k] (A4): virtual row m =
    (slice, tile, row) -- ``row_term`` built in that order."""
    c = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, W8_L)))[None, :, None]
    b = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, W8_L)))[None, None, :]
    v = wload(base, row_term[:, None, None] + c * 256 + b)
    return gl.convert_layout(gl.reshape(v, [4, 32, 128]), A4)


@gluon.jit
def o_mma4(aq4, xs, red, NG: gl.constexpr, NO: gl.constexpr = 0):
    """o task's 32 rows x NG tokens from the round buffers ``xs`` [4 rounds, NG tokens x 4 slices,
    128] (BSW): 4 batched MFMAs a wave, then the slices summed in order (bd_reduce's) -> (los, his)
    a tile, [8 row pairs, NG] (PR_L)."""
    one_a = gl.full([4, 32, 4], 127, gl.uint8, SL4A)
    one_b = gl.full([4, 16, 4], 127, gl.uint8, SL4B)
    acc = zacc([4, 32, NG], MF4)
    for r in gl.static_range(4):
        bv = xs._reinterpret(gl.uint8, [4, NG, 4, 128], BSW3).index(r).permute((1, 2, 0)).load(B4)
        acc = cdna4.mfma_scaled(aq4[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
    gl.barrier()  # the previous group's reads of ``red`` done
    # LDS [slice, tile, row parity, token, row pair]: a tile's (lo, hi) read in PR_L
    red.store(gl.reshape(gl.permute(gl.reshape(acc, [4, 2, 8, 2, NG]), (0, 1, 3, 4, 2)), [16, NG, 8]))
    gl.barrier()
    N: gl.constexpr = NO if NO > 0 else NG  # the columns read back (NO: a narrower build's)
    los = ()
    his = ()
    for tt in gl.static_range(2):
        lo = gl.zeros([8, N], gl.float32, PR_L)
        hi = gl.zeros([8, N], gl.float32, PR_L)
        for p in gl.static_range(4):
            r2 = red._reinterpret(gl.float32, [16, NG, 8], FLAT2)
            lo = lo + r2.index(4 * p + 2 * tt).permute((1, 0)).slice(0, N, dim=1).load(PR_L)
            hi = hi + r2.index(4 * p + 2 * tt + 1).permute((1, 0)).slice(0, N, dim=1).load(PR_L)
        los = los + (lo,)
        his = his + (hi,)
    return los, his


@gluon.jit
def o_owner_pk(t, cols, live, jr, rank, attn_own):
    """o_owner's dense loads issued (rank t % 4's; elsewhere masked off): o_owner_fin lands them."""
    own_d = dptr(attn_own, ATTND_OFF)
    blk = cols * HPAIRS + 16 * t
    mine = live & (rank == t % 4)
    opk = ()
    for tt in gl.static_range(2):
        for r in gl.static_range(4):
            opk = opk + (peek_dense(own_d, r * (MAX_S * HPAIRS) + blk + 2 * jr + tt, mine),)
    return opk


@gluon.jit
def o_owner_fin(t, cols, live, jr, rank, attn_own, attn_peer, errors, bid, opk):
    """Dense owner step (rank t % 4) from o_owner_pk's loads."""
    if rank == t % 4:
        own_d = dptr(attn_own, ATTND_OFF)
        aagd = dshift(attn_peer, AAGD_OFF)
        blk = cols * HPAIRS + 16 * t
        oit = ()
        for tt in gl.static_range(2):
            for r in gl.static_range(4):
                oit = oit + ((own_d, r * (MAX_S * HPAIRS) + blk + 2 * jr + tt, opk[4 * tt + r]),)
        ows, err = wait_dense(oit, 1)
        note_err(errors, bid, err, E_DOWN)
        osw = ()
        for tt in gl.static_range(2):
            o_lo = bf_lo(ows[4 * tt])
            o_hi = bf_hi(ows[4 * tt])
            for r in gl.static_range(1, 4):
                o_lo = o_lo + bf_lo(ows[4 * tt + r])
                o_hi = o_hi + bf_hi(ows[4 * tt + r])
            osw = osw + (bf_pair(o_lo, o_hi),)
        for r in gl.static_range(4):
            put_dense2(aagd[r], blk // 2 + jr, osw[0], osw[1], mask=live)
        gl.barrier()  # every replica's words read before the resets (see prims' dense hand-offs)
        for r in gl.static_range(4):
            reset_dense2(own_d, (r * (MAX_S * HPAIRS) + blk) // 2 + jr, mask=live)


@gluon.jit
def o_land_pk(t, cols, live, jr, attn_own, sc, tag):
    """o_land's dense loads (and h) issued: o_land_fin lands them."""
    aagd = dptr(attn_own, AAGD_OFF)
    blk = cols * HPAIRS + 16 * t
    pks = ()
    for tt in gl.static_range(2):
        pw = cols * HPAIRS + 16 * t + 8 * tt + jr
        pks = pks + (peek_dense(aagd, blk + 2 * jr + tt, live), peek(rgn(sc, R_H), pw, tag, live))
    return pks


@gluon.jit
def o_land_fin(t, cols, live, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw, pks):
    """Dense o_land from o_land_pk's loads."""
    aagd = dptr(attn_own, AAGD_OFF)
    blk = cols * HPAIRS + 16 * t
    dws, err = wait_dense(((aagd, blk + 2 * jr, pks[0]), (aagd, blk + 2 * jr + 1, pks[2])), 1)
    note_err(errors, bid, err, E_DLAND)
    hws = wait_group(((rgn(sc, R_H), cols * HPAIRS + 16 * t + jr, pks[1]),
                      (rgn(sc, R_H), cols * HPAIRS + 16 * t + 8 + jr, pks[3])), tag, errors, bid, E_DH, 1)
    ws = (dws[0], hws[0], dws[1], hws[1])
    gl.barrier()  # every replica's words read before the reset
    reset_dense2(aagd, blk // 2 + jr, mask=live)
    o_land_out(cols, live, jr, t, sc, tag, hm_out, aslot, outw, ws)


@gluon.jit
def o_land_out(cols, live, jr, t, sc, tag, hm_out, aslot, outw, ws):
    for tt in gl.static_range(2):
        pw = cols * HPAIRS + 16 * t + 8 * tt + jr
        aw = ws[2 * tt]
        put(rgn(sc, R_A), pw, aw, tag, mask=live)
        hmw = bf_pair(bf_lo(aw) + bf_lo(ws[2 * tt + 1]), bf_hi(aw) + bf_hi(ws[2 * tt + 1]))
        if PLAIN_IN:
            cdna4.buffer_store(hmw, i32p(rgn(sc, R_HM)), pw, mask=live)
        else:
            put(rgn(sc, R_HM), pw, hmw, tag, mask=live)
        if outw:
            cdna4.buffer_store(hmw, i32p(hm_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)


@gluon.jit
def o_owner(t, cols, live, jr, rank, attn_own, attn_peer, tag, errors, bid, DA: gl.constexpr = False):
    """AG_RS owner step of o task t for token columns ``cols`` (rank t % 4 only): the ranks'
    partials summed in rank order -> every rank's aag (DA: the dense regions, a 16-pair block's
    pair 8 tt + j at word 2 j + tt: a lane's two pairs one 8-byte store)."""
    if DA:
        if rank == t % 4:
            o_owner_fin(t, cols, live, jr, rank, attn_own, attn_peer, errors, bid,
                        o_owner_pk(t, cols, live, jr, rank, attn_own))
    elif rank == t % 4:
        aag_peer = peer_shift(attn_peer, AAG_OFF)
        opk = ()
        for tt in gl.static_range(2):
            pw = cols * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                opk = opk + (peek(attn_own, r * (MAX_S * HPAIRS) + pw, tag, live),)
        oit = ()
        for tt in gl.static_range(2):
            pw = cols * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                oit = oit + ((attn_own, r * (MAX_S * HPAIRS) + pw, opk[4 * tt + r]),)
        ows = wait_group(oit, tag, errors, bid, E_ATTN, 1)
        for tt in gl.static_range(2):
            pw = cols * HPAIRS + 16 * t + 8 * tt + jr
            o_lo = bf_lo(ows[4 * tt])
            o_hi = bf_hi(ows[4 * tt])
            for r in gl.static_range(1, 4):
                o_lo = o_lo + bf_lo(ows[4 * tt + r])
                o_hi = o_hi + bf_hi(ows[4 * tt + r])
            osw = bf_pair(o_lo, o_hi)
            for r in gl.static_range(4):
                put_sys(aag_peer[r], pw, osw, tag, mask=live)


@gluon.jit
def o_land(t, cols, live, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw, DA: gl.constexpr = False):
    """AG_RS landing of o task t for token columns ``cols``: the all-gathered sums (and h) ->
    "a", h_mid = bf16(a + h) -> "hm" (and hm_out at the last / aux layers)."""
    if DA:
        o_land_fin(t, cols, live, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw,
                   o_land_pk(t, cols, live, jr, attn_own, sc, tag))
    else:
        aag_own = (attn_own.to(gl.int64, bitcast=True) + AAG_OFF).to(gl.pointer_type(gl.int64), bitcast=True)
        pks = ()
        for tt in gl.static_range(2):
            pw = cols * HPAIRS + 16 * t + 8 * tt + jr
            pks = pks + (peek(aag_own, pw, tag, live), peek(rgn(sc, R_H), pw, tag, live))
        items = ()
        for tt in gl.static_range(2):
            pw = cols * HPAIRS + 16 * t + 8 * tt + jr
            items = items + ((aag_own, pw, pks[2 * tt]), (rgn(sc, R_H), pw, pks[2 * tt + 1]))
        ws = wait_group(items, tag, errors, bid, E_ATTN, 1)
        o_land_out(cols, live, jr, t, sc, tag, hm_out, aslot, outw, ws)


@gluon.jit
def o_land1s(t, cols, live, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw):
    """Dense one-shot landing of o task t for token columns ``cols``: the 4 ranks' partials (a
    16-pair block's pair 8 tt + j at word 2 j + tt) summed in rank order (and h) -> "a", h_mid
    -> "hm" (and hm_out), the partial slots reset."""
    own_d = dptr(attn_own, ATTND_OFF)
    blk = cols * HPAIRS + 16 * t
    pks = ()
    for tt in gl.static_range(2):
        for r in gl.static_range(4):
            pks = pks + (peek_dense(own_d, r * (MAX_S * HPAIRS) + blk + 2 * jr + tt, live),)
    hpk = ()
    for tt in gl.static_range(2):
        hpk = hpk + (peek(rgn(sc, R_H), blk + 8 * tt + jr, tag, live),)
    its = ()
    for tt in gl.static_range(2):
        for r in gl.static_range(4):
            its = its + ((own_d, r * (MAX_S * HPAIRS) + blk + 2 * jr + tt, pks[4 * tt + r]),)
    ows, err = wait_dense(its, 1)
    note_err(errors, bid, err, E_DLAND)
    hws = wait_group(((rgn(sc, R_H), blk + jr, hpk[0]), (rgn(sc, R_H), blk + 8 + jr, hpk[1])), tag, errors, bid,
                     E_DH, 1)
    gl.barrier()  # every replica's words read before the resets
    for r in gl.static_range(4):
        reset_dense2(own_d, (r * (MAX_S * HPAIRS) + blk) // 2 + jr, mask=live)
    for tt in gl.static_range(2):
        pw = blk + 8 * tt + jr
        o_lo = bf_lo(ows[4 * tt])
        o_hi = bf_hi(ows[4 * tt])
        for r in gl.static_range(1, 4):
            o_lo = o_lo + bf_lo(ows[4 * tt + r])
            o_hi = o_hi + bf_hi(ows[4 * tt + r])
        aw = bf_pair(o_lo, o_hi)
        put(rgn(sc, R_A), pw, aw, tag, mask=live)
        hmw = bf_pair(bf_lo(aw) + bf_lo(hws[tt]), bf_hi(aw) + bf_hi(hws[tt]))
        if PLAIN_IN:
            cdna4.buffer_store(hmw, i32p(rgn(sc, R_HM)), pw, mask=live)
        else:
            put(rgn(sc, R_HM), pw, hmw, tag, mask=live)
        if outw:
            cdna4.buffer_store(hmw, i32p(hm_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)


@gluon.jit
def o_words(sc, k0, S: gl.constexpr):
    """o tasks past 16 rows: tokens k0 .. + 16's FP8 o inputs (plain words, past the scales' gate)."""
    k3 = k0 + ar(16, layout=gl.SliceLayout(1, gl.SliceLayout(2, XW3)))[:, None, None]
    h3 = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, XW3)))[None, :, None]
    w3 = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, XW3)))[None, None, :]
    return cdna4.buffer_load(i32p(rgn(sc, R_OQ)), k3 * (O_K // 4) + h3 * 32 + w3, mask=(k3 < S) & (w3 >= 0),
                             other=0, cache=".cv")


@gluon.jit
def stage_o(t, li, ptab, n_layers, sc, peers, peer_local, pbase, rank, hm_out, tag, errors, bid, S: gl.constexpr,
            FFN_OFF: gl.constexpr, tr_=None, NOAR: gl.constexpr = False, RAAG: gl.constexpr = False):
    """o task t: hidden rows 32 t .. + 32."""
    AGRS: gl.constexpr = S >= AG_RS_FROM
    # h_mid out at the last layer (slot 0) and EAGLE3's aux layers (slot > 0)
    aslot = lrow(ptab, li, (19, 0))[0].to(gl.int32) >> 8
    if NOAUX:
        aslot = 0
    outw = (li == n_layers - 1) | (aslot > 0)
    e = lrow(ptab, li, (7, 8))
    w_o = lp(e[0], gl.uint8)
    s_o = lp(e[1], gl.float32)
    attn_peer, _, attn_own, _ = peer_regions(peers, peer_local, pbase, FFN_OFF)
    # weights first: tile tt = rows 16 (2 t + tt) .., round r, wave (tt, p) step 4 r + p
    m = w8_rows()
    wv = m // 16
    # batched o GEMM (o_mma4): the pipelined path past 16 rows, and up to 16 rows
    OB4: gl.constexpr = O_B4 and ((O_PIPE and (S > 16) and AGRS and (not NOAR)) or ((S <= 16) and (S > SMALL_FROM)))
    aq = ()
    if OB4:
        for r in gl.static_range(4):
            aq = aq + (w8_tile4(w_o, (2 * t + (m // 16) % 2) * (16 * 2048) + (4 * r + m // 32) * 2048 + (m % 16) * 16),)
    else:
        for r in gl.static_range(4):
            aq = aq + (w8_tile(w_o, (2 * t + wv // 4) * (16 * 2048) + (4 * r + wv % 4) * 2048 + (m % 16) * 16),)
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    ws = ()
    for tt in gl.static_range(2):
        ws = ws + ((gl.load(s_o + 32 * t + 16 * tt + 2 * jr), gl.load(s_o + 32 * t + 16 * tt + 2 * jr + 1)),)
    m = pin(m)
    # every token's FP8 row in LDS [4 head groups, 4 tokens x 4 heads, 128]
    NS: gl.constexpr = ncols(S)
    NG: gl.constexpr = 16 if (S > 16 or OB4) else NS  # the MFMA's tokens (past 16 rows, or batched: 16)
    xs = gl.allocate_shared_memory(gl.uint8, [4, 4 * NG, 128], BSW)
    nn = ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :]
    live = nn < S
    PIPE: gl.constexpr = O_PIPE and (S > 16) and AGRS and (not NOAR)
    DA: gl.constexpr = PIPE and DENSE_ATTN and HYB
    D1: gl.constexpr = DA and DENSE_1S
    OVL: gl.constexpr = DA and (not D1) and O_OVL
    LAG: gl.constexpr = O_LAG if OVL else 2  # groups between a group's GEMM and its landing
    attnd = dshift(attn_peer, ATTND_OFF)
    if PIPE:
        # past 16 rows, the all-reduce pipelined behind the GEMM: iteration g computes group g
        # (gated on its 16 tokens' merges only) and sends its partials, then the owner step of
        # group g - 1, the next group's gate + input loads, and the landing of group g - 2
        # (whose loads overlap the input's flight)
        G: gl.constexpr = (S + 15) // 16
        lt = ar(THREADS, T1)
        one_a = unit_scales([128, 4], A_L)
        one_b = unit_scales([4 * NG, 4], B_L)
        if O_B4:
            aq4 = aq
            red = gl.allocate_shared_memory(gl.float32, [16, NG, 8], FLAT3)
        else:
            red = gl.allocate_shared_memory(gl.float32, [128, NG], FLAT2)
        c16 = ar(NG, layout=gl.SliceLayout(0, PR_L))[None, :]
        gate_poll(rgn(sc, R_OQS), lt, (lt < 16) & (lt < S), tag, errors, bid, E_OQ)
        wv = o_words(sc, 0, S)
        for g in gl.static_range(G):
            ng = 16 * g + c16
            isc = ng + jr * 0
            pks = peek(rgn(sc, R_OQS), isc, tag, isc < S)
            if OVL:
                # this iteration's hand-off loads all in flight under the GEMM (one wait after it,
                # instead of the owner's, the gate's and the landing's round trips in a row)
                if g >= 1:
                    pg = 16 * (g - 1) + c16
                    opk = o_owner_pk(t, pg, pg < S, jr, rank, attn_own)
                if g >= LAG:
                    lg = 16 * (g - LAG) + c16
                    lpk = o_land_pk(t, lg, lg < S, jr, attn_own, sc, tag)
                if g + 1 < G:
                    gi = 16 * (g + 1) + lt
                    gpk = peek(rgn(sc, R_OQS), gi, tag, (lt < 16) & (gi < S))
            if TRACE == 3 and O_TST and g == 2:
                stamp(tr_, bid, 14)
            b0 = (wv & 255).to(gl.uint8)
            b1 = ((wv >> 8) & 255).to(gl.uint8)
            b2 = ((wv >> 16) & 255).to(gl.uint8)
            b3 = ((wv >> 24) & 255).to(gl.uint8)
            x = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [16, 4, 4, 128]), (1, 0, 2, 3))
            gl.barrier()  # the previous group's B reads done
            xs.store(gl.reshape(x, [4, 64, 128]))
            gl.barrier()
            if TRACE == 3 and O_TST and g == 2:
                stamp(tr_, bid, 15)
            xsc = wait_group(((rgn(sc, R_OQS), isc, pks),), tag, errors, bid, E_OQ, 1)[0].to(gl.float32, bitcast=True)
            if TRACE == 3 and O_TST and g == 2:
                stamp(tr_, bid, 16)
            if TRACE == 3:  # (diagnostics) group 0's / the last group's input landed
                if g == 0:
                    stamp(tr_, bid, 30)
                if g == G - 1:
                    stamp(tr_, bid, 56)
            if O_B4:
                los, his = o_mma4(aq4, xs, red, NG)
            else:
                acc = zacc([128, 4 * NG], MFMA)
                for r in gl.static_range(4):
                    bv = rnd(xs, r).permute((1, 0)).load(B_L)
                    acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                            bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
                los, his = bd_reduce(acc, 2, 4, red)
            if TRACE == 3 and O_TST and g == 2:
                stamp(tr_, bid, 26)
            if DA:
                pw2 = ()
                for tt in gl.static_range(2):
                    pw2 = pw2 + (bf_pair(bf16_round(los[tt] * xsc * ws[tt][0]), bf16_round(his[tt] * xsc * ws[tt][1])),)
                if D1:
                    for r in gl.static_range(4):
                        put_dense2(attnd[r], ((rank * MAX_S + ng) * HPAIRS + 16 * t) // 2 + jr, pw2[0], pw2[1],
                                   mask=ng < S)
                else:
                    put_dense2(dpick(attnd, t % 4), ((rank * MAX_S + ng) * HPAIRS + 16 * t) // 2 + jr, pw2[0], pw2[1],
                               mask=ng < S)
            else:
                for tt in gl.static_range(2):
                    lo = bf16_round(los[tt] * xsc * ws[tt][0])
                    hi = bf16_round(his[tt] * xsc * ws[tt][1])
                    at = (rank * MAX_S + ng) * HPAIRS + 16 * t + 8 * tt + jr
                    put_sys(peer_pick(attn_peer, t % 4), at, bf_pair(lo, hi), tag, mask=ng < S)
            if OVL:
                if TRACE == 3 and O_TST and g == 2:
                    stamp(tr_, bid, 27)
                if g + 1 < G:
                    wait_group(((rgn(sc, R_OQS), gi, gpk),), tag, errors, bid, E_OQ, 1)
                    gl.barrier()
                    wv = o_words(sc, 16 * (g + 1), S)
                if TRACE == 3 and O_TST and g == 2:
                    stamp(tr_, bid, 28)
                if g >= 1:
                    o_owner_fin(t, pg, pg < S, jr, rank, attn_own, attn_peer, errors, bid, opk)
                if TRACE == 3 and O_TST and g == 2:
                    stamp(tr_, bid, 34)
                if g >= LAG:
                    o_land_fin(t, lg, lg < S, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw, lpk)
                if TRACE == 3 and O_TST and g == 2:
                    stamp(tr_, bid, 35)
            elif D1:
                pass
            elif g >= 1:
                pg = 16 * (g - 1) + c16
                o_owner(t, pg, pg < S, jr, rank, attn_own, attn_peer, tag, errors, bid, DA)
            if OVL:
                pass
            elif g + 1 < G:
                gate_poll(rgn(sc, R_OQS), 16 * (g + 1) + lt, (lt < 16) & (16 * (g + 1) + lt < S), tag, errors, bid,
                          E_OQ)
                wv = o_words(sc, 16 * (g + 1), S)
            if OVL:
                pass
            elif D1:
                # one-shot: group g - 1's partials (sent a GEMM ago) landed while g's input flies
                if g >= 1:
                    lg = 16 * (g - 1) + c16
                    o_land1s(t, lg, lg < S, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw)
            elif g >= 2:
                lg = 16 * (g - 2) + c16
                o_land(t, lg, lg < S, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw, DA)
        stamp(tr_, bid, 25)
        if D1:
            lg = 16 * (G - 1) + c16
            o_land1s(t, lg, lg < S, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw)
        elif OVL:
            pg = 16 * (G - 1) + c16
            opk = o_owner_pk(t, pg, pg < S, jr, rank, attn_own)
            lpks = ()
            for g in gl.static_range(max(G - LAG, 0), G - 1):
                lg = 16 * g + c16
                lpks = lpks + (o_land_pk(t, lg, lg < S, jr, attn_own, sc, tag),)
            o_owner_fin(t, pg, pg < S, jr, rank, attn_own, attn_peer, errors, bid, opk)
            for g in gl.static_range(max(G - LAG, 0), G - 1):
                lg = 16 * g + c16
                o_land_fin(t, lg, lg < S, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw,
                           lpks[g - max(G - LAG, 0)])
            lg = 16 * (G - 1) + c16
            o_land(t, lg, lg < S, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw, DA)
        else:
            pg = 16 * (G - 1) + c16
            o_owner(t, pg, pg < S, jr, rank, attn_own, attn_peer, tag, errors, bid, DA)
            for g in gl.static_range(max(G - 2, 0), G):
                lg = 16 * g + c16
                o_land(t, lg, lg < S, jr, attn_own, sc, tag, errors, bid, hm_out, aslot, outw, DA)
        stamp(tr_, bid, 29)
    elif S > 16:
        lt = ar(THREADS, T1)
        gate_poll(rgn(sc, R_OQS), lt, lt < S, tag, errors, bid, E_OQ)
        one_a = unit_scales([128, 4], A_L)
        one_b = unit_scales([4 * NG, 4], B_L)
        red = gl.allocate_shared_memory(gl.float32, [128, NG], FLAT2)
        if O_PF:
            wv_next = o_words(sc, 0, S)
        for g in gl.static_range((S + 15) // 16):
            if O_PF:
                # group g + 1's words issued before group g's math (its round trip overlaps it)
                wv = wv_next
                if g + 1 < (S + 15) // 16:
                    wv_next = o_words(sc, 16 * (g + 1), S)
            else:
                wv = o_words(sc, 16 * g, S)
            ng = 16 * g + ar(NG, layout=gl.SliceLayout(0, PR_L))[None, :]
            isc = ng + jr * 0
            pks = peek(rgn(sc, R_OQS), isc, tag, isc < S)
            b0 = (wv & 255).to(gl.uint8)
            b1 = ((wv >> 8) & 255).to(gl.uint8)
            b2 = ((wv >> 16) & 255).to(gl.uint8)
            b3 = ((wv >> 24) & 255).to(gl.uint8)
            x = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [16, 4, 4, 128]), (1, 0, 2, 3))
            gl.barrier()  # the previous group's B reads done
            xs.store(gl.reshape(x, [4, 64, 128]))
            gl.barrier()
            xsc = wait_group(((rgn(sc, R_OQS), isc, pks),), tag, errors, bid, E_OQ, 1)[0].to(gl.float32, bitcast=True)
            acc = zacc([128, 4 * NG], MFMA)
            for r in gl.static_range(4):
                bv = rnd(xs, r).permute((1, 0)).load(B_L)
                acc = cdna4.mfma_scaled(aq[r].to(gl.float8e4nv, bitcast=True), one_a, "e4m3",
                                        bv.to(gl.float8e4nv, bitcast=True), one_b, "e4m3", acc)
            los, his = bd_reduce(acc, 2, 4, red)
            for tt in gl.static_range(2):
                lo = bf16_round(los[tt] * xsc * ws[tt][0])
                hi = bf16_round(his[tt] * xsc * ws[tt][1])
                at = (rank * MAX_S + ng) * HPAIRS + 16 * t + 8 * tt + jr
                if NOAR:  # (ablation: the GEMM's output to local scratch, no all-reduce)
                    cdna4.buffer_store(bf_pair(lo, hi), i32p(rgn(sc, R_A)), ng * HPAIRS + 16 * t + 8 * tt + jr,
                                       mask=ng < S)
                elif AGRS:
                    put_sys(peer_pick(attn_peer, t % 4), at, bf_pair(lo, hi), tag, mask=ng < S)
                else:
                    for r in gl.static_range(4):
                        put_sys(attn_peer[r], at, bf_pair(lo, hi), tag, mask=ng < S)
        stamp(tr_, bid, 25)
    else:
        o_task_mma(t, sc, tag, errors, bid, aq, ws, xs, nn, jr, rank, attn_peer, S, NS, AGRS, tr_)
    if NOAR or PIPE:
        pass
    elif AGRS:
        # reduce-scatter + all-gather (FlyDSL's AG_RS): task t's owner (rank t % 4) sums the
        # ranks' partials in rank order and puts the bf16 sums to every rank's aag; every rank
        # then reads them back (half the link bytes, one more hop)
        aag_own = (attn_own.to(gl.int64, bitcast=True) + AAG_OFF).to(gl.pointer_type(gl.int64), bitcast=True)
        if rank == t % 4:
            aag_peer = peer_shift(attn_peer, AAG_OFF)
            opk = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                for r in gl.static_range(4):
                    opk = opk + (peek(attn_own, r * (MAX_S * HPAIRS) + pw, tag, live),)
            oit = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                for r in gl.static_range(4):
                    oit = oit + ((attn_own, r * (MAX_S * HPAIRS) + pw, opk[4 * tt + r]),)
            ows = wait_group(oit, tag, errors, bid, E_ATTN, 1)
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                o_lo = bf_lo(ows[4 * tt])
                o_hi = bf_hi(ows[4 * tt])
                for r in gl.static_range(1, 4):
                    o_lo = o_lo + bf_lo(ows[4 * tt + r])
                    o_hi = o_hi + bf_hi(ows[4 * tt + r])
                osw = bf_pair(o_lo, o_hi)
                for r in gl.static_range(4):
                    put_sys(aag_peer[r], pw, osw, tag, mask=live)
        if RAAG:
            pass  # (the router tasks read aag and write h_mid themselves)
        else:
            pks = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                pks = pks + (peek(aag_own, pw, tag, live), peek(rgn(sc, R_H), pw, tag, live))
            items = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                items = items + ((aag_own, pw, pks[2 * tt]), (rgn(sc, R_H), pw, pks[2 * tt + 1]))
            ws = wait_group(items, tag, errors, bid, E_ATTN, 1)
            stamp(tr_, bid, 29)
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                aw = ws[2 * tt]
                put(rgn(sc, R_A), pw, aw, tag, mask=live)
                hmw = bf_pair(bf_lo(aw) + bf_lo(ws[2 * tt + 1]), bf_hi(aw) + bf_hi(ws[2 * tt + 1]))
                if PLAIN_IN:
                    cdna4.buffer_store(hmw, i32p(rgn(sc, R_HM)), pw, mask=live)
                else:
                    put(rgn(sc, R_HM), pw, hmw, tag, mask=live)
                if outw:
                    cdna4.buffer_store(hmw, i32p(hm_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)
    elif PUSH_W and (NS == 4):
        pw, ok, t_lo, t_hi, hw = ar_recv(attn_own, t, tag, errors, bid, S, E_ATTN, rgn(sc, R_H))
        stamp(tr_, bid, 29)
        put(rgn(sc, R_A), pw, bf_pair(t_lo, t_hi), tag, mask=ok)
        hmw = bf_pair(bf16_round(t_lo) + bf_lo(hw), bf16_round(t_hi) + bf_hi(hw))
        if PLAIN_IN:
            cdna4.buffer_store(hmw, i32p(rgn(sc, R_HM)), pw, mask=ok)
        else:
            put(rgn(sc, R_HM), pw, hmw, tag, mask=ok)
        if outw:
            cdna4.buffer_store(hmw, i32p(hm_out), aslot * (MAX_S * HPAIRS) + pw, mask=ok)
    else:
        # this task's rows of the all-reduce: every rank's partials in rank order (and h),
        # all the loads issued before any wait
        pks = ()
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                pks = pks + (peek(attn_own, r * (MAX_S * HPAIRS) + pw, tag, live),)
            pks = pks + (peek(rgn(sc, R_H), pw, tag, live),)
        items = ()
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                items = items + ((attn_own, r * (MAX_S * HPAIRS) + pw, pks[5 * tt + r]),)
            items = items + ((rgn(sc, R_H), pw, pks[5 * tt + 4]),)
        ws = wait_group(items, tag, errors, bid, E_ATTN, 1)
        stamp(tr_, bid, 29)
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            t_lo = bf_lo(ws[5 * tt])
            t_hi = bf_hi(ws[5 * tt])
            for r in gl.static_range(1, 4):
                t_lo = t_lo + bf_lo(ws[5 * tt + r])
                t_hi = t_hi + bf_hi(ws[5 * tt + r])
            h_lo = bf_lo(ws[5 * tt + 4])
            h_hi = bf_hi(ws[5 * tt + 4])
            put(rgn(sc, R_A), pw, bf_pair(t_lo, t_hi), tag, mask=live)
            hmw = bf_pair(bf16_round(t_lo) + h_lo, bf16_round(t_hi) + h_hi)
            if PLAIN_IN:
                cdna4.buffer_store(hmw, i32p(rgn(sc, R_HM)), pw, mask=live)
            else:
                put(rgn(sc, R_HM), pw, hmw, tag, mask=live)
            if outw:
                cdna4.buffer_store(hmw, i32p(hm_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)


# ============================================================ router
# FlyDSL's scheme: a wave per expert (gate row in 16 B chunks, bf16 dot2, a wave sum: no
# barrier), the row normed once (one CTA reduction) and shared through LDS, the MXFP8
# slice and the scores published straight from registers (no layout conversions)
GL = gl.constexpr(gl.BlockedLayout([1, 4], [1, 64], [8, 1], [1, 0]))  # [8 experts, pairs]: wave e, 4 pairs a lane
NA = gl.constexpr(gl.BlockedLayout([4], [64], [8], [0]))  # pairs 0 .. 2048: thread t pairs 4 t ..
NB = gl.constexpr(gl.BlockedLayout([2], [64], [8], [0]))  # pairs 2048 .. 3072: thread t pairs 2048 + 2 t ..
SCL = gl.constexpr(gl.BlockedLayout([1, 1], [1, 64], [8, 1], [1, 0]))  # [8 experts, 8 copies]: wave e
XROW = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))


@gluon.jit
def router_gate(gate, pair0: gl.constexpr, NP: gl.constexpr, e0):
    """Gate rows e0 .. e0 + 8 (bf16 pairs pair0 .. pair0 + NP), wave e row e0 + e."""
    e = ar(8, layout=gl.SliceLayout(1, GL))[:, None]
    q = ar(NP, layout=gl.SliceLayout(0, GL))[None, :]
    return gl.load(gate + (e0 + e) * HPAIRS + pair0 + q)


@gluon.jit
def dot2_bf16(a, b):
    """a.lo b.lo + a.hi b.hi (bf16 pairs as i32) in f32: v_dot2_f32_bf16."""
    return gl.inline_asm_elementwise("v_dot2_f32_bf16 $0, $1, $2, 0", "=v,v,v", [a, b], dtype=gl.float32,
                                     is_pure=True, pack=1)


@gluon.jit
def stage_router(tr, li, ptab, sc, tag, errors, bid, eps, S: gl.constexpr, tr_=None, moe_x=0,
                 NORM_ONLY: gl.constexpr = False, asrc=None, RAAG: gl.constexpr = False, n_layers=0, hm_out=0):
    """Router task tr = (tokens TPR (tr // 16) .. + TPR, experts 8 (tr % 16) ..): past 16
    rows a task routes several tokens (its gate rows loaded once). ``asrc``: where the
    all-reduced attention rows land (the "a" region, or with RAAG the owners' gathered sums
    aag, read here directly as FlyDSL's router does: the o tasks then land nothing)."""
    TPR: gl.constexpr = (S + 15) // 16
    if asrc is None:
        asrc = rgn(sc, R_A)
    tq = tr % 16
    e0 = tq * 8
    e = lrow(ptab, li, (10, 11, 9))
    gate = lp(e[0], gl.int32)
    bias = lp(e[1], gl.float32)
    g_post = lp(e[2], gl.int32)
    e1 = ar(8, layout=gl.SliceLayout(1, GL))
    if NORM_ONLY:  # (ablation: no gate rows)
        gA = gl.zeros([8, 2048], gl.int32, GL)
        gB = gl.zeros([8, 1024], gl.int32, GL)
        bias_e = gl.zeros([8], gl.float32, gl.SliceLayout(1, GL))
    else:
        gA = router_gate(gate, 0, 2048, e0)
        gB = router_gate(gate, 2048, 1024, e0)
        bias_e = gl.load(bias + e0 + e1)
    qa = ar(2048, layout=NA)
    if ROUTER_ONEWAIT:
        # part B as wide as A (pairs 3072 .. masked): every word in one batched wait,
        # as FlyDSL's single poll (two waits: the second's loads are a round trip stale)
        qb = 2048 + ar(2048, layout=NA)
        okb = qb < 3072
    else:
        qb = 2048 + ar(1024, layout=NB)
        okb = qb >= 0
    gpA = gl.load(g_post + qa)
    gpB = gl.load(g_post + qb, mask=okb, other=0)
    if not NORM_ONLY:
        gA = pin(gA)
    xs = gl.allocate_shared_memory(gl.int32, [4096], XROW)
    if ROUTER_PF and (TPR > 1):
        # token i + 1's a, h loads issued before token i's wait and math: its round trip
        # overlaps them (the gate for the first token only; later words re-polled if late)
        k0 = (tr // 16) * TPR
        if ROUTER_GATE:
            lt = ar(THREADS, T1)
            gate_poll(asrc, k0 * HPAIRS + 16 * lt + 15, (lt < N_O) & (k0 < S), tag, errors, bid, E_A)
        nxt = router_peeks(k0, qa, qb, okb, sc, tag, S, asrc)
        for i in gl.static_range(TPR):
            cur = nxt
            if i + 1 < TPR:
                nxt = router_peeks(k0 + i + 1, qa, qb, okb, sc, tag, S, asrc)
            if k0 + i < S:
                if i > 0:
                    gl.barrier()  # xs: the previous token's reads done
                router_tok(k0 + i, tq, e0, e1, gA, gB, bias_e, qa, qb, okb, gpA, gpB, xs, sc, tag, errors, bid, eps,
                           S, tr_, moe_x, tr, NORM_ONLY, cur, asrc, RAAG, li, n_layers, hm_out, ptab)
    for i in gl.static_range(TPR if not (ROUTER_PF and (TPR > 1)) else 0):
        k = (tr // 16) * TPR + i
        if TPR == 1:
            router_tok(k, tq, e0, e1, gA, gB, bias_e, qa, qb, okb, gpA, gpB, xs, sc, tag, errors, bid, eps, S, tr_,
                       moe_x, tr, NORM_ONLY, None, asrc, RAAG, li, n_layers, hm_out, ptab)
        else:
            if k < S:
                gl.barrier()  # xs: the previous token's reads done
                router_tok(k, tq, e0, e1, gA, gB, bias_e, qa, qb, okb, gpA, gpB, xs, sc, tag, errors, bid, eps, S,
                           tr_, moe_x, tr, NORM_ONLY, None, asrc, RAAG, li, n_layers, hm_out, ptab)


@gluon.jit
def router_peeks(k, qa, qb, okb, sc, tag, S: gl.constexpr, asrc):
    """Token k's a, h loads issued (router_tok's peeks)."""
    pwA = k * HPAIRS + qa
    pwB = k * HPAIRS + qb
    live = k < S
    return (peek(asrc, pwA, tag, (qa >= 0) & live), peek(rgn(sc, R_H), pwA, tag, (qa >= 0) & live),
            peek(asrc, pwB, tag, okb & live), peek(rgn(sc, R_H), pwB, tag, okb & live))


@gluon.jit
def router_tok(k, tq, e0, e1, gA, gB, bias_e, qa, qb, okb, gpA, gpB, xs, sc, tag, errors, bid, eps, S: gl.constexpr,
               tr_, moe_x, tr, NORM_ONLY: gl.constexpr = False, pre=None, asrc=None, RAAG: gl.constexpr = False,
               li=0, n_layers=0, hm_out=0, ptab=0):
    """Router task (token k, experts e0 ..): a, h polled, the norm, xn out, logits, scores."""
    pwA = k * HPAIRS + qa
    pwB = k * HPAIRS + qb
    if asrc is None:
        asrc = rgn(sc, R_A)
    hpre = ()
    if ROUTER_HPRE and (pre is None):
        # FlyDSL's router: the residual h (put by this layer's norm tasks, long landed) read
        # before the attention wait, so only a is polled -- one batch, its words each a tag
        phA = peek(rgn(sc, R_H), pwA, tag, qa >= 0)
        phB = peek(rgn(sc, R_H), pwB, tag, okb)
        hA = wait_group(((rgn(sc, R_H), pwA, phA),), tag, errors, bid, E_A, 4)[0]
        hB = wait_group(((rgn(sc, R_H), pwB, phB),), tag, errors, bid, E_A, 4 if ROUTER_ONEWAIT else 2)[0]
        if ROUTER_GATE:
            lt = ar(THREADS, T1)
            gate_poll(asrc, k * HPAIRS + 16 * lt + 15, lt < N_O, tag, errors, bid, E_A)
        paA = peek(asrc, pwA, tag, qa >= 0)
        paB = peek(asrc, pwB, tag, okb)
        hpre = (hA, hB)
        phA = paA
        phB = paB
    elif pre is not None:  # (stage_router issued them: ROUTER_PF)
        paA, phA, paB, phB = pre
    else:
        if ROUTER_GATE:
            # the o tasks' last words (task t: pairs 16 t .. + 16 of the token), then every
            # word fresh: lands at its first check (no CTA-wide spin through the attention)
            lt = ar(THREADS, T1)
            gate_poll(asrc, k * HPAIRS + 16 * lt + 15, lt < N_O, tag, errors, bid, E_A)
        paA = peek(asrc, pwA, tag, qa >= 0)
        phA = peek(rgn(sc, R_H), pwA, tag, qa >= 0)
        paB = peek(asrc, pwB, tag, okb)
        phB = peek(rgn(sc, R_H), pwB, tag, okb)
    stamp(tr_, bid, 42)
    if ROUTER_HPRE and (pre is None):
        if ROUTER_ONEWAIT:
            w2 = wait_group(((asrc, pwA, paA), (asrc, pwB, paB)), tag, errors, bid, E_A, 4)
            wa = (w2[0], hpre[0])
            wb = (w2[1], hpre[1])
        else:
            wa = (wait_group(((asrc, pwA, paA),), tag, errors, bid, E_A, 4)[0], hpre[0])
            stamp(tr_, bid, 43)
            wb = (wait_group(((asrc, pwB, paB),), tag, errors, bid, E_A, 2)[0], hpre[1])
    elif ROUTER_ONEWAIT:
        w4 = wait_group(((asrc, pwA, paA), (rgn(sc, R_H), pwA, phA), (asrc, pwB, paB),
                         (rgn(sc, R_H), pwB, phB)), tag, errors, bid, E_A, 4)
        wa = (w4[0], w4[1])
        wb = (w4[2], w4[3])
    else:
        wa = wait_group(((asrc, pwA, paA), (rgn(sc, R_H), pwA, phA)), tag, errors, bid, E_A, 4)
        stamp(tr_, bid, 43)
        wb = wait_group(((asrc, pwB, paB), (rgn(sc, R_H), pwB, phB)), tag, errors, bid, E_A, 2)
    vA_lo = bf_lo(wa[0]) + bf_lo(wa[1])
    vA_hi = bf_hi(wa[0]) + bf_hi(wa[1])
    vB_lo = bf_lo(wb[0]) + bf_lo(wb[1])
    vB_hi = bf_hi(wb[0]) + bf_hi(wb[1])
    stamp(tr_, bid, 17)
    if RAAG:
        # FlyDSL's: this task writes its slice (pairs 192 tq .. + 192) of h_mid = bf16(a + h),
        # the same f32 sums as the o tasks' landing rounded the same way (bit-identical)
        XSL: gl.constexpr = HPAIRS // 16
        hmA = bf_pair(vA_lo, vA_hi)
        hmB = bf_pair(vB_lo, vB_hi)
        mA = (qa >= XSL * tq) & (qa < XSL * tq + XSL)
        mB = okb & (qb >= XSL * tq) & (qb < XSL * tq + XSL)
        if PLAIN_IN:
            cdna4.buffer_store(hmA, i32p(rgn(sc, R_HM)), pwA, mask=mA)
            cdna4.buffer_store(hmB, i32p(rgn(sc, R_HM)), pwB, mask=mB)
        else:
            put(rgn(sc, R_HM), pwA, hmA, tag, mask=mA)
            put(rgn(sc, R_HM), pwB, hmB, tag, mask=mB)
        aslot = lrow(ptab, li, (19, 0))[0].to(gl.int32) >> 8
        if NOAUX:
            aslot = 0
        if (li == n_layers - 1) | (aslot > 0):
            cdna4.buffer_store(hmA, i32p(hm_out), aslot * (MAX_S * HPAIRS) + pwA, mask=mA)
            cdna4.buffer_store(hmB, i32p(hm_out), aslot * (MAX_S * HPAIRS) + pwB, mask=mB)
    # sum of squares: each thread's 12 in registers, then one CTA reduction
    sA = gl.sum(gl.reshape(vA_lo * vA_lo + vA_hi * vA_hi, [512, 4]), axis=1)
    if ROUTER_ONEWAIT:
        sB = gl.sum(gl.reshape(vB_lo * vB_lo + vB_hi * vB_hi, [512, 4]), axis=1)
    else:
        sB = gl.sum(gl.reshape(vB_lo * vB_lo + vB_hi * vB_hi, [512, 2]), axis=1)
    ss = gl.sum(sA + gl.convert_layout(sB, sA.type.layout), axis=0)
    rstd = hw_rsq(ss / 6144.0 + eps)
    stamp(tr_, bid, 46)
    xA_lo = bf16_round(vA_lo * rstd * (bf_lo(gpA) + 1.0))
    xA_hi = bf16_round(vA_hi * rstd * (bf_hi(gpA) + 1.0))
    xB_lo = bf16_round(vB_lo * rstd * (bf_lo(gpB) + 1.0))
    xB_hi = bf16_round(vB_hi * rstd * (bf_hi(gpB) + 1.0))
    xs.slice(0, 2048).store(bf_pair(xA_lo, xA_hi))
    if ROUTER_ONEWAIT:
        xs.slice(2048, 2048).store(bf_pair(xB_lo, xB_hi))
    else:
        xs.slice(2048, 1024).store(bf_pair(xB_lo, xB_hi))
    # this task's slice of xn as MXFP8 (aiter's rule): blocks 12 tq .. + 12, from registers.
    # The wide MoE's xn is plain words and a flag, after the scores: it routes first and
    # fetches xn after its tables (the drain then costs the scores nothing)
    PL: gl.constexpr = S >= WIDE_FROM
    if HYB:
        # AITER's MoE input: this task's pairs 192 tq .. + 192 of xn, bf16
        xo = i32p(moe_x)
        cdna4.buffer_store(bf_pair(xA_lo, xA_hi), xo, k * HPAIRS + qa, mask=(qa >= 192 * tq) & (qa < 192 * tq + 192))
        cdna4.buffer_store(bf_pair(xB_lo, xB_hi), xo, k * HPAIRS + qb,
                           mask=okb & (qb >= 192 * tq) & (qb < 192 * tq + 192))
        stamp(tr_, bid, 52)
    elif (not PL) and (not XN_SLICE):
        mx8_put(gl.reshape(xA_lo, [128, 16]), gl.reshape(xA_hi, [128, 16]), 0, tq, sc, k, tag)
        if ROUTER_ONEWAIT:
            mx8_put(gl.reshape(xB_lo, [128, 16]), gl.reshape(xB_hi, [128, 16]), 128, tq, sc, k, tag)
        else:
            mx8_put(gl.reshape(xB_lo, [64, 16]), gl.reshape(xB_hi, [64, 16]), 128, tq, sc, k, tag)
    stamp(tr_, bid, 47)
    if (not PL) and (not HYB) and XN_SLICE:
        # FlyDSL's: only this task's slice (blocks 12 tq .. + 12) quantized, read back from
        # the row in LDS
        gl.barrier()
        X1: gl.constexpr = gl.BlockedLayout([2], [64], [8], [0])
        iq = ar(256, layout=X1)  # block iq // 16 (of the 12), pair iq % 16
        w1 = xs.gather(gl.where(iq < 192, 12 * tq * 16 + iq, 0), 0)
        wq = gl.reshape(w1, [16, 16])
        mx8_put(bf_lo(wq), bf_hi(wq), 0, tq, sc, k, tag, False, 12 * tq)
    if PL and (not HYB) and (not NORM_ONLY) and (ROUTER_XN_EARLY == 1):
        # FlyDSL's order: this task's xn slice stored before the logits (in flight through
        # the dot products), its flag after the scores with no vmcnt drain (the barrier
        # alone: write-through stores, then one flag store -- the probe's 0 stale words)
        mx8_put(gl.reshape(xA_lo, [128, 16]), gl.reshape(xA_hi, [128, 16]), 0, tq, sc, k, tag, True)
        mx8_put(gl.reshape(xB_lo, [64, 16]), gl.reshape(xB_hi, [64, 16]), 128, tq, sc, k, tag, True)
    if not NORM_ONLY:
        gl.barrier()
        xa = xs.slice(0, 2048).load(gl.SliceLayout(0, GL))
        xb = xs.slice(2048, 1024).load(gl.SliceLayout(0, GL))
        logit = gl.sum(dot2_bf16(gA, xa[None, :]), axis=1) + gl.sum(dot2_bf16(gB, xb[None, :]), axis=1)
        stamp(tr_, bid, 18)
        sig = hw_rcp(1.0 + hw_exp2(-bf16_round(logit) * LOG2E))
        rk = route_key(sig + bias_e, e0 + e1)
        # copy c of (key, sigmoid) of expert e from wave e's lane c
        c2 = ar(SCORE_COPIES, layout=gl.SliceLayout(0, SCL))[None, :]
        e2 = ar(8, layout=gl.SliceLayout(1, SCL))[:, None]
        rk2 = gl.convert_layout(rk, gl.SliceLayout(1, SCL))[:, None] + c2 * 0
        sg2 = gl.convert_layout(sig, gl.SliceLayout(1, SCL))[:, None] + c2 * 0.0
        at = ((c2 * MAX_S + k) * N_ROUTED + e0 + e2) * 2
        scr = rgn(sc, R_SC)
        if SC_LINE and PL:
            # one wave writes each copy's whole line, (key, tag, sigmoid, tag) x 8 experts =
            # 128 B from 8 lanes: the 8 waves' 16 partial writes a line (read-modify-writes
            # at the memory side, serialized: ~2 us to drain at S = 8, the pollers' reads of
            # the line queued among them) become one full-line write a copy
            sc_s = gl.allocate_shared_memory(gl.int32, [16], FLAT1)
            sc_s.slice(0, 8).store(rk)
            sc_s.slice(8, 8).store(sig.to(gl.int32, bitcast=True))
            gl.barrier()
            L4: gl.constexpr = gl.BlockedLayout([4], [64], [8], [0])
            i4 = ar(2048, layout=L4)
            j4 = i4 % 4
            e4 = (i4 // 4) % 8
            c4 = (i4 // 32) % SCORE_COPIES
            v4 = sc_s.gather(e4 + gl.where(j4 >= 2, 8, 0), 0)
            v4 = gl.where((j4 % 2) == 1, tag + 0 * j4, v4)
            at4 = ((c4 * MAX_S + k) * N_ROUTED + e0 + e4) * 4 + j4
            cdna4.buffer_store(v4, i32p(scr), at4, mask=i4 < 32 * SCORE_COPIES, cache=PUT_CM)
        elif SC_PUT_DEV:
            # device scope (sc1 alone, FlyDSL's for its scratch mailboxes): the system-scope
            # write-through pairs took ~3.5 us to land at the (gated) pollers at S = 8
            put_dev(scr, at, rk2, tag)
            put_dev(scr, at + 1, sg2, tag)
        else:
            put(scr, at, rk2, tag)
            put(scr, at + 1, sg2, tag)
        if PL:
            stamp(tr_, bid, 12)  # (wide) the scores put
        if PL and not HYB:
            if ROUTER_XN_EARLY == 1:
                ft = ar(THREADS, T1)
                gl.barrier()  # every wave's xn stores issued (FlyDSL's xndone: no vmcnt drain)
            elif ROUTER_XN_EARLY == 2:
                # the scores' puts first (the critical path), then the xn stores, then the flag
                # behind a barrier alone (no vmcnt drain)
                mx8_put(gl.reshape(xA_lo, [128, 16]), gl.reshape(xA_hi, [128, 16]), 0, tq, sc, k, tag, True)
                mx8_put(gl.reshape(xB_lo, [64, 16]), gl.reshape(xB_hi, [64, 16]), 128, tq, sc, k, tag, True)
                ft = ar(THREADS, T1)
                gl.barrier()
            elif XN_LDS_W:
                # FlyDSL's: the task's slice (blocks 12 tq .. + 12) read back from the row in
                # LDS by 128 consecutive threads -> 96 consecutive words, full-line writes
                # (from the registers' layout they are 4 B partial writes to the 3 lines)
                X2: gl.constexpr = gl.BlockedLayout([2], [64], [8], [0])
                iq = ar(2048, layout=X2)
                w1 = xs.gather(gl.where(iq < 192, 12 * tq * 16 + iq, 0), 0)
                wq = gl.reshape(w1, [128, 16])
                mx8_put(bf_lo(wq), bf_hi(wq), 0, tq, sc, k, tag, True, 12 * tq)
                ft = drain(ar(THREADS, T1))
            else:
                mx8_put(gl.reshape(xA_lo, [128, 16]), gl.reshape(xA_hi, [128, 16]), 0, tq, sc, k, tag, True)
                mx8_put(gl.reshape(xB_lo, [64, 16]), gl.reshape(xB_hi, [64, 16]), 128, tq, sc, k, tag, True)
                ft = drain(ar(THREADS, T1))  # the plain words stored before their flag (FlyDSL's xndone)
            put(rgn(sc, R_XNF), tr + ft * 0, ft * 0 + 1, tag, mask=ft == 0)
    stamp(tr_, bid, 19)


@gluon.jit
def mx8_put(lo, hi, blk0: gl.constexpr, tq, sc, k, tag, PLAIN: gl.constexpr = False, b_rt=0):
    """[NB blocks, 16 pairs] of xn (blocks blk0 ..) -> MXFP8 words (4 fp8 a word) and E8M0
    codes of the task's blocks 12 tq .. + 12, in the tensors' own layout."""
    layout: gl.constexpr = lo.type.layout
    NB: gl.constexpr = lo.shape[0]
    amax = gl.max(gl.maximum(gl.abs(lo), gl.abs(hi)), axis=1)
    code = code_ceil(amax * (1.0 / 448.0))
    inv = hw_rcp(((code << 23).to(gl.float32, bitcast=True)))[:, None]
    q_lo = clamp_fp8(lo * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    q_hi = clamp_fp8(hi * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
    w2 = q_lo | (q_hi << 8)  # [NB, 16]: pair j's two fp8
    a, c = gl.split(gl.reshape(w2, [NB, 8, 2]))
    word = a | (c << 16)  # [NB, 8]: 4 fp8 a word
    WL: gl.constexpr = word.type.layout
    bw = b_rt + blk0 + ar(NB, layout=gl.SliceLayout(1, WL))[:, None]
    jj = ar(8, layout=gl.SliceLayout(0, WL))[None, :]
    b = b_rt + blk0 + ar(NB, layout=gl.SliceLayout(1, layout))
    if PLAIN:
        cdna4.buffer_store(word, i32p(rgn(sc, R_XNP)), k * (HIDDEN // 4) + bw * 8 + jj,
                           mask=(bw >= 12 * tq) & (bw < 12 * tq + 12), cache=".wt")
        cdna4.buffer_store(code.to(gl.uint8), u8p(rgn(sc, R_XSCP)), k * (HIDDEN // 32) + b,
                           mask=(b >= 12 * tq) & (b < 12 * tq + 12), cache=".wt")
    else:
        put(rgn(sc, R_XN8), k * (HIDDEN // 4) + bw * 8 + jj, word, tag, mask=(bw >= 12 * tq) & (bw < 12 * tq + 12))
        put(rgn(sc, R_XSC), k * (HIDDEN // 32) + b, code, tag, mask=(b >= 12 * tq) & (b < 12 * tq + 12))


# ============================================================ MoE (narrow: S <= 4)
RT = gl.constexpr(gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0]))  # [8 tokens, 128 experts]
UG_DEPTH = gl.constexpr(8)  # up / gate rounds in flight (16 B a lane each)
REP8 = gl.constexpr(gl.BlockedLayout([8], [64], [8], [0]))  # [8], every thread holding all 8
XW3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 2, 32], [1, 8, 1], [2, 1, 0]))  # [S, steps, 32 words]
CL3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 16, 4], [1, 8, 1], [2, 1, 0]))  # [S, steps, 4 codes]
MD3 = gl.constexpr(gl.BlockedLayout([1, 1, 2], [1, 2, 32], [1, 8, 1], [2, 1, 0]))  # [S, 32 entries, 64 pairs]
INT_MIN = gl.constexpr(-2147483648)


# [4 tokens, 128 experts] held whole by every wave (warp bases 0), the tokens in registers: each
# wave routes every token itself, its picks uniform in the wave (no cross-wave exchange)
RTW = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [1, 0], [2, 0]], lane_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]],
    warp_bases=[[0, 0], [0, 0], [0, 0]], block_bases=[], shape=[4, 128]))
ROUTE_W = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_ROUTE_W", "0")))  # S <= 4: every wave routes


@gluon.jit
def route_wave(sc, bid, tag, errors, route_scale, pred, S: gl.constexpr, tr_=None):
    """``route_all`` for S <= 4: every wave polls and routes every token (copy bid %
    SCORE_COPIES) -- the picks are wave-uniform registers, read out without LDS."""
    kk = ar(4, layout=gl.SliceLayout(1, RTW))[:, None]
    e = ar(128, layout=gl.SliceLayout(0, RTW))[None, :]
    at = (((bid % SCORE_COPIES) * MAX_S + kk) * N_ROUTED + e) * 2
    ok = (kk < S) & pred
    scr = rgn(sc, R_SC)
    pk_k = peek(scr, at, tag, ok)
    pk_s = peek(scr, at + 1, tag, ok)
    stamp(tr_, bid, 40)
    rk, sw = wait_group(((scr, at, pk_k), (scr, at + 1, pk_s)), tag, errors, bid, E_SC, 8)
    stamp(tr_, bid, 41)
    sig = sw.to(gl.float32, bitcast=True)
    keys = gl.where(ok, rk, INT_MIN)
    ids = ()
    ws = ()
    for p in gl.static_range(TOP_K):
        mx = gl.max(keys, axis=1)
        keys = gl.where(keys == mx[:, None], INT_MIN, keys)
        idp = 127 - (mx & 127)
        ids = ids + (idp,)
        ws = ws + (gl.sum(gl.where(e == idp[:, None], sig, 0.0), axis=1),)
    scale = route_scale / gl.maximum((ws[0] + ws[1]) + (ws[2] + ws[3]), 1e-20)
    stamp(tr_, bid, 30)
    k1 = ar(4, layout=gl.SliceLayout(1, RTW))
    E = ()
    W = ()
    for k in gl.static_range(S):
        for p in gl.static_range(TOP_K):
            ik = gl.sum(gl.where(k1 == k, ids[p], 0), axis=0)
            wk = gl.sum(gl.where(k1 == k, ws[p] * scale, 0.0), axis=0)
            E = E + (gl.minimum(gl.maximum(uniform(ik), 0), N_ROUTED - 1),)
            W = W + (uniform(wk.to(gl.int32, bitcast=True)).to(gl.float32, bitcast=True),)
    return E, W


@gluon.jit
def route_all(sc, bid, tag, errors, route_scale, pred, S: gl.constexpr, tr_=None):
    """Every token's top-4 routed experts from the router's (key, sigmoid) pairs (copy
    bid % SCORE_COPIES) -> (ids, weights): a [8] tensor (token rows) a slot. The weights
    are the picks' sigmoids renormalized and scaled by route_scale (ATOM's)."""
    if ROUTE_GATE and S <= 2:
        # (S = 4 with the lean wait: a wild expert id, timing-dependent -- open)
        # one word a router task (token k, experts 8 tq ..): its last expert's sigmoid
        lt = ar(THREADS, T1)
        gk = lt // 16
        gate_poll(rgn(sc, R_SC), (((bid % SCORE_COPIES) * MAX_S + gk) * N_ROUTED + 8 * (lt % 16) + 7) * 2 + 1,
                  (lt < 16 * S) & pred, tag, errors, bid, E_SC)
    kk = ar(8, layout=gl.SliceLayout(1, RT))[:, None]
    e = ar(128, layout=gl.SliceLayout(0, RT))[None, :]
    at = (((bid % SCORE_COPIES) * MAX_S + kk) * N_ROUTED + e) * 2
    ok = (kk < S) & pred
    scr = rgn(sc, R_SC)
    pk_k = peek(scr, at, tag, ok)
    pk_s = peek(scr, at + 1, tag, ok)
    stamp(tr_, bid, 40)
    rk, sw = wait_group(((scr, at, pk_k), (scr, at + 1, pk_s)), tag, errors, bid, E_SC, 2)
    stamp(tr_, bid, 41)
    sig = sw.to(gl.float32, bitcast=True)
    keys = gl.where(ok, rk, INT_MIN)
    ids = ()
    ws = ()
    for p in gl.static_range(TOP_K):
        mx = gl.max(keys, axis=1)
        keys = gl.where(keys == mx[:, None], INT_MIN, keys)
        idp = 127 - (mx & 127)
        ids = ids + (idp,)
        ws = ws + (gl.sum(gl.where(e == idp[:, None], sig, 0.0), axis=1),)
    scale = route_scale / gl.maximum((ws[0] + ws[1]) + (ws[2] + ws[3]), 1e-20)
    stamp(tr_, bid, 30)
    # token k's slot s as CTA-uniform scalars E[4 k + s], W[4 k + s]: a lookup is then a
    # scalar select. Wave k holds token k's picks: one LDS exchange (word 8 k + p: id, 8 k +
    # 4 + p: weight), one barrier, every thread reading all 64 words
    xch = gl.allocate_shared_memory(gl.int32, [8, 8], FLAT2)
    RT8: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [8, 1], [1, 0])  # [8 tokens, 8 words]
    cw = ar(8, layout=gl.SliceLayout(0, RT8))[None, :]
    tile = gl.zeros([8, 8], gl.int32, RT8)
    for p in gl.static_range(TOP_K):
        idp = gl.convert_layout(ids[p], gl.SliceLayout(1, RT8))[:, None]
        wp = gl.convert_layout((ws[p] * scale).to(gl.int32, bitcast=True), gl.SliceLayout(1, RT8))[:, None]
        tile = gl.where(cw == p, idp, gl.where(cw == 4 + p, wp, tile))
    xch.store(tile)
    gl.barrier()
    xch = xch.reshape([64])
    # all 64 words into every thread's registers (wide reads, in flight together), the
    # picks register selects, then the scalars 8 a readfirstlane block
    R64: gl.constexpr = gl.BlockedLayout([64], [64], [8], [0])
    allw = xch.load(R64)
    ii = ar(64, layout=R64)
    raw = ()
    for k in gl.static_range(S):
        for p in gl.static_range(TOP_K):
            raw = raw + (gl.sum(gl.where(ii == 8 * k + p, allw, 0), axis=0),)
    for k in gl.static_range(S):
        for p in gl.static_range(TOP_K):
            raw = raw + (gl.sum(gl.where(ii == 8 * k + 4 + p, allw, 0), axis=0),)
    un = uniforms(raw)
    E = ()
    W = ()
    for i in gl.static_range(S * TOP_K):
        E = E + (gl.minimum(gl.maximum(un[i], 0), N_ROUTED - 1),)
        W = W + (un[S * TOP_K + i].to(gl.float32, bitcast=True),)
    return E, W


@gluon.jit
def route_out(g, sc, bid, tag, errors, route_scale, shared_weight, moe_ids, moe_w, S: gl.constexpr):
    """Hybrid: tokens 8 g .. + 8's top-4 (route_all's rule) -> AITER's topk ids / weights
    [S, 5] (slot 4: the shared expert N_ROUTED, weight shared_weight)."""
    kk = ar(8, layout=gl.SliceLayout(1, RT))[:, None]
    e = ar(128, layout=gl.SliceLayout(0, RT))[None, :]
    ok = (8 * g + kk) < S
    at = (((bid % SCORE_COPIES) * MAX_S + 8 * g + kk) * N_ROUTED + e) * 2
    scr = rgn(sc, R_SC)
    pk_k = peek(scr, at, tag, ok)
    pk_s = peek(scr, at + 1, tag, ok)
    rk, sw = wait_group(((scr, at, pk_k), (scr, at + 1, pk_s)), tag, errors, bid, E_SC, 2)
    sig = sw.to(gl.float32, bitcast=True)
    keys = gl.where(ok, rk, INT_MIN)
    ids = ()
    ws = ()
    for p in gl.static_range(TOP_K):
        mx = gl.max(keys, axis=1)
        keys = gl.where(keys == mx[:, None], INT_MIN, keys)
        idp = 127 - (mx & 127)
        ids = ids + (idp,)
        ws = ws + (gl.sum(gl.where(e == idp[:, None], sig, 0.0), axis=1),)
    scale = route_scale / gl.maximum((ws[0] + ws[1]) + (ws[2] + ws[3]), 1e-20)
    tk = 8 * g + ar(8, layout=gl.SliceLayout(1, RT))
    okt = tk < S
    for p in gl.static_range(TOP_K):
        gl.store(moe_ids + tk * (TOP_K + 1) + p, ids[p], mask=okt)
        gl.store(moe_w + tk * (TOP_K + 1) + p, ws[p] * scale, mask=okt)
    gl.store(moe_ids + tk * (TOP_K + 1) + TOP_K, tk * 0 + N_ROUTED, mask=okt)
    gl.store(moe_w + tk * (TOP_K + 1) + TOP_K, tk * 0.0 + shared_weight, mask=okt)


@gluon.jit
def stage_far(t, li, n_layers, moe_out, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid,
              S: gl.constexpr, FFN_OFF: gl.constexpr, tr_=None, aslot=0):
    """Hybrid: layer li's FFN all-reduce, down task t's rows (pairs 16 t .. + 16) of AITER's
    output (this rank's partial, plain bf16) through ffn_tail."""
    NS: gl.constexpr = ncols(S)
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    nn = ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :]
    live = nn < S
    los = ()
    his = ()
    for tt in gl.static_range(2):
        w = cdna4.buffer_load(i32p(moe_out), nn * HPAIRS + 16 * t + 8 * tt + jr, mask=live, other=0)
        los = los + (bf_lo(w),)
        his = his + (bf_hi(w),)
    ffn_tail(t, li, n_layers, los, his, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, S, FFN_OFF,
             tr_, aslot)


@gluon.jit
def tok_of(vec, k):
    """Row k of a [8] route tensor (REP8: in every thread), as a scalar."""
    kk = gl.arange(0, 8, layout=REP8)
    return gl.sum(gl.where(kk == k, vec, 0 * vec), axis=0)


@gluon.jit
def expert_of(E, k, slot, S: gl.constexpr):
    """Token k's slot-``slot`` expert (k, slot runtime scalars)."""
    e = E[0]
    for i in gl.static_range(1, 4 * S):
        e = gl.where((k == i // 4) & (slot == i % 4), E[i], e)
    return uniform(e)


@gluon.jit
def gate_poll(ptr, idx, pred, tag, errors, bid, code: gl.constexpr):
    """FlyDSL's gated wait: a few lanes (``pred``; [512] T1 ``idx``) poll their producers'
    last words, then the CTA waits at a barrier: 512 threads of many CTAs spinning on the
    whole of a region load the memory system the producers and other hand-offs need."""
    pk = peek(ptr, idx, tag, pred)
    wait_group(((ptr, idx, pk),), tag, errors, bid, code, 1)
    gl.barrier()


@gluon.jit
def stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, pred, S: gl.constexpr, k0=0, ONE: gl.constexpr = False):
    """Every token's MXFP8 xn -> LDS [steps, 16 tokens, 128 B] (steps 0..31 in xsa,
    32..47 in xsb), its codes -> [steps, 16, 4] (csa, csb). ONE: token k0's alone, at
    token position 0 (a CTA whose routed tasks are all token k0's: FlyDSL's fetch_xn)."""
    SP: gl.constexpr = 1 if ONE else triton.next_power_of_2(S)
    NT: gl.constexpr = 1 if ONE else S
    if XN_GATE:
        # router task tq's slice: blocks 12 tq .. + 12 of 8 words; lane (k, tq) its last word
        lt = ar(THREADS, T1)
        gate_poll(rgn(sc, R_XN8), (k0 + lt // 16) * (HIDDEN // 4) + 96 * (lt % 16) + 95, (lt < 16 * NT) & pred, tag,
                  errors, bid, E_XN)
    xn_part(sc, tag, errors, bid, xsa, csa, pred, 0, 32, SP, NT, k0)
    xn_part(sc, tag, errors, bid, xsb, csb, pred, 32, 16, SP, NT, k0)


@gluon.jit
def xn_part(sc, tag, errors, bid, xs8, cs8, pred, st0: gl.constexpr, NST: gl.constexpr, SP: gl.constexpr,
            S: gl.constexpr, k0=0, G: gl.constexpr = 4):
    k3 = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XW3)))[:, None, None]
    st3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, XW3)))[None, :, None]
    w3 = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, XW3)))[None, None, :]
    iw = (k0 + k3) * (HIDDEN // 4) + st3 * 32 + w3
    kc = ar(SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, CL3)))[:, None, None]
    sc3 = st0 + ar(NST, layout=gl.SliceLayout(0, gl.SliceLayout(2, CL3)))[None, :, None]
    g3 = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(1, CL3)))[None, None, :]
    ic = (k0 + kc) * (HIDDEN // 32) + sc3 * 4 + g3
    pkw = peek(rgn(sc, R_XN8), iw, tag, (k3 < S) & pred)
    pkc = peek(rgn(sc, R_XSC), ic, tag, (kc < S) & pred)
    wv = wait_group(((rgn(sc, R_XN8), iw, pkw),), tag, errors, bid, E_XN, SP * NST // 16)[0]
    b0 = (wv & 255).to(gl.uint8)
    b1 = ((wv >> 8) & 255).to(gl.uint8)
    b2 = ((wv >> 16) & 255).to(gl.uint8)
    b3 = ((wv >> 24) & 255).to(gl.uint8)
    x = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [SP, NST // G, G, 128]), (1, 0, 2, 3))
    xs8.slice(0, SP * G, dim=1).store(gl.reshape(x, [NST // G, SP * G, 128]))
    cv = wait_group(((rgn(sc, R_XSC), ic, pkc),), tag, errors, bid, E_XN, SP)[0]
    cq = gl.permute(gl.reshape(cv.to(gl.uint8), [SP, NST // G, G, 4]), (1, 0, 2, 3))
    cs8.slice(0, SP * G, dim=1).store(gl.reshape(cq, [NST // G, SP * G, 4]))


@gluon.jit
def ug_round(e, cg, r: gl.constexpr, w13, s13, offs):
    """Round r of up / gate tile pair (expert e, gate rows 16 cg .., up rows 768 + 16 cg ..):
    wave (tile, p) takes 128-k step 4 r + p -> (A tile, A scales), in flight. The expert,
    row group and round live in the scalar bases, the lane offsets ``offs`` (``ug_offs``)
    are loop-invariant. A task past the end is clamped by the caller (its result dropped)."""
    base = w13 + e.to(gl.int64) * (2 * INTER * (HIDDEN // 2)) + cg * (16 * (HIDDEN // 2)) + (4 * r) * 1024
    a = w4_tile(base, offs[0])
    sca = cdna4.buffer_load(s13 + ((e * 48 + cg // 2) * 6144 + r * 512 + cg % 2), offs[1])
    return a, sca


@gluon.jit
def ug_bround(r: gl.constexpr, xsa, xsb, csa, csb):
    """Round r's B operand (every token's xn, 4 steps) and its scales, from LDS."""
    BS: gl.constexpr = cdna4.get_mfma_scale_layout(B_L, [16, 4])
    if r < 8:
        bv = rnd(xsa, r).permute((1, 0)).load(B_L)
        bs = rnd(csa, r).load(BS)
    else:
        bv = rnd(xsb, r - 8).permute((1, 0)).load(B_L)
        bs = rnd(csb, r - 8).load(BS)
    return bv, bs


@gluon.jit
def ug_loads(e, cg, w13, s13, offs):
    """Up / gate tile pair of expert e: gate rows 16 cg .. + 16 (tile 0), up rows 768 +
    16 cg .. (tile 1); round r, wave (tile, p) takes 128-k step 4 r + p (in flight)."""
    aq = ()
    sq = ()
    for r in gl.static_range(12):
        a, sca = ug_round(e, cg, r, w13, s13, offs)
        aq = aq + (a,)
        sq = sq + (sca,)
    return aq, sq


@gluon.jit
def ug_mma(aq, sq, xsa, xsb, csa, csb, red):
    """The tile pair against every token's xn -> (gate (lo, hi), up (lo, hi)) [8 pairs, 16 t]."""
    acc = zacc([128, 16], MFMA)
    BS: gl.constexpr = cdna4.get_mfma_scale_layout(B_L, [16, 4])
    for r in gl.static_range(12):
        if r < 8:
            bv = rnd(xsa, r).permute((1, 0)).load(B_L)
            bs = rnd(csa, r).load(BS)
        else:
            bv = rnd(xsb, r - 8).permute((1, 0)).load(B_L)
            bs = rnd(csb, r - 8).load(BS)
        acc = cdna4.mfma_scaled(aq[r], sq[r], "e2m1", bv.to(gl.float8e4nv, bitcast=True), bs, "e4m3", acc)
    los, his = bd_reduce(acc, 2, 4, red)
    return los[0], his[0], los[1], his[1]


@gluon.jit
def swiglu(g, u, limit):
    """aiter's swiglu_mul: min(g, L) * sigmoid(1.702 min(g, L)) * (clamp(u, -L, L) + 1)."""
    gc = gl.minimum(g, limit)
    uc = gl.maximum(gl.minimum(u, limit), -limit)
    sig = hw_rcp(1.0 + hw_exp2((1.702 * gc) * -LOG2E))
    return gc * sig * (uc + 1.0)


MD2 = gl.constexpr(gl.BlockedLayout([1, 2], [2, 32], [8, 1], [1, 0]))  # [entries, 64 pairs]


@gluon.jit
def mids_peek(sc, tag, k, slot0: gl.constexpr, NE: gl.constexpr):
    """Token k's mid chunks of slots slot0 .., entry 8 (slot - slot0) + chunk (chunks 6, 7
    zero): the loads issued -> (pair index, live mask, peek)."""
    en = ar(NE, layout=gl.SliceLayout(1, MD2))[:, None]
    j = ar(64, layout=gl.SliceLayout(0, MD2))[None, :]
    sl = slot0 + en // 8
    ch = en % 8
    ok = (sl < SLOTS) & (ch < 6)
    im = (k * SLOTS + sl) * (INTER // 2) + ch * 64 + j
    return im, ok, peek(rgn(sc, R_MID), im, tag, ok)


@gluon.jit
def mids_part(sc, tag, errors, bid, k, NE: gl.constexpr, buf, cbuf, pk, tr_=None):
    """A ``mids_peek`` landed -> MXFP8 -> LDS ``buf`` [NE / 4, 4, MAX_S, 128] row k (codes
    ``cbuf``)."""
    im, ok, pm = pk
    PK: gl.constexpr = (NE // 16 if NE >= 16 else 1) * 2
    mv = wait_group(((rgn(sc, R_MID), im, pm),), tag, errors, bid, E_MID, PK)[0]
    if tr_ is not None:
        stamp(tr_, bid, 23)
    lo = bf_lo(mv)
    hi = bf_hi(mv)
    lo4 = gl.reshape(gl.where(ok, lo, 0.0), [NE, 4, 16])
    hi4 = gl.reshape(gl.where(ok, hi, 0.0), [NE, 4, 16])
    amax = gl.max(gl.maximum(gl.abs(lo4), gl.abs(hi4)), axis=2)
    code = code_ceil(amax * (1.0 / 448.0))
    inv = hw_rcp((code << 23).to(gl.float32, bitcast=True))[:, :, None]
    q_lo = clamp_fp8(lo4 * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    q_hi = clamp_fp8(hi4 * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    buf.slice(4 * k, 4, dim=1).store(gl.reshape(gl.join(q_lo, q_hi), [NE // 4, 4, 128]))
    cbuf.slice(4 * k, 4, dim=1).store(gl.reshape(code.to(gl.uint8), [NE // 4, 4, 4]))


@gluon.jit
def stage_mids(sc, tag, errors, bid, ms8, mc8, msh, mch, S: gl.constexpr, tr_=None):
    """Every token's mid rows -> MXFP8 (aiter's rule) in LDS [entries, 16 tokens, 128 B]
    (codes [entries, 4, 4]): routed slot s's 128-k chunk c at entry 8 s + c of ms8 (c < 6;
    chunks 6, 7 zero), the shared slot's at entry c of msh. Every load issued before any
    wait (one round trip for all)."""
    if MIDS_GATE:
        # one word an up / gate task (token k, slot, rows 16 cg ..): its last pair; the
        # tokens a CTA's lanes reach (2), the rest by the full wait
        NPT: gl.constexpr = SLOTS * UG_TASKS
        lt = ar(THREADS, T1)
        gk = lt // NPT
        gr = lt % NPT
        gate_poll(rgn(sc, R_MID), (gk * SLOTS + gr // UG_TASKS) * (INTER // 2) + 8 * (gr % UG_TASKS) + 7,
                  (lt < NPT * S) & (gk < S), tag, errors, bid, E_MID)
    pks = ()
    for k in gl.static_range(S):
        pks = pks + (mids_peek(sc, tag, k, 0, 32), mids_peek(sc, tag, k, TOP_K, 8))
    for k in gl.static_range(S):
        mids_part(sc, tag, errors, bid, k, 32, ms8, mc8, pks[2 * k], tr_ if k == 0 else None)
        mids_part(sc, tag, errors, bid, k, 8, msh, mch, pks[2 * k + 1])


@gluon.jit
def down_round(i: gl.constexpr, t, ids, w2, s2, S: gl.constexpr, DBG: gl.constexpr, offs):
    """Round i's A tiles and scales of down task t (rows 32 t .. + 32): token i // 8's
    routed slot (i % 8) // 2, chunks 4 (i % 2) + q on wave (tile, q) for i < 8 S, then
    the shared expert's chunks 4 (i - 8 S) + q; chunks past 6 zero. Scalar bases, lane
    offsets ``offs`` (``dn_offs``)."""
    if i < 8 * S:
        if DBG & 2097152:  # debug: slot s's rounds read slot (s + 1) % 4's expert
            e = ids[4 * (i // 8) + ((i % 8) // 2 + 1) % 4]
        else:
            e = ids[4 * (i // 8) + (i % 8) // 2]
        half: gl.constexpr = i % 2
    else:
        e = ids[0] * 0 + N_ROUTED
        half: gl.constexpr = i - 8 * S
    if DN_UNI:
        # the expert id as an SGPR: LLVM otherwise sees it divergent here and wraps every
        # weight / scale load of the round in a readfirstlane waterfall loop (ATT: 64 loops)
        e = uniform(e)
    t = gl.minimum(t, N_DN - 1)  # a CTA without a down task streams task N_DN - 1's (dropped)
    base = w2 + e.to(gl.int64) * (HIDDEN * (INTER // 2)) + t * (32 * (INTER // 2)) + (4 * half) * 1024
    sb = s2 + ((e * 192 + t) * 768 + half * 512)
    if DBG & 512:
        a = w4_tile(base, offs[half], t < 0)
        sca = cdna4.buffer_load(sb, offs[2 + half], mask=offs[2 + half] < 0, other=0x7F)
    else:
        a = w4_tile(base, offs[half])
        sca = cdna4.buffer_load(sb, offs[2 + half])
    return a, sca


@gluon.jit
def down_wcol(i: gl.constexpr, wts, shared_weight, S: gl.constexpr):
    """Round i's route weight of each accumulator column (token n = column // 4); only
    the diagonal blocks are read (``bd_reduce``)."""
    n = ar(16, layout=gl.SliceLayout(0, MFMA)) // 4
    if i < 8 * S:
        w = gl.where(n == i // 8, wts[4 * (i // 8) + (i % 8) // 2], 0.0)
    else:
        w = gl.where(n < S, shared_weight, 0.0)
    return w[None, :]


@gluon.jit
def stage_down(t, li, n_layers, ids, wts, w2, s2, ms8, mc8, msh, mch, red, sc, peers, peer_local, pbase, rank, ar_out,
               tag, errors, bid, shared_weight, S: gl.constexpr, FFN_OFF: gl.constexpr, DBG: gl.constexpr = 0,
               tr_=None, pre=None, pf_q=0, pf_o=0):
    """Down task t: rows 32 t .. + 32, every token's route-weighted units -> bf16
    partials to every rank -> rank-ordered sum -> ar. ``pre``: rounds 0 .. DN_DEPTH - 1
    already in flight (streamed in the up / gate loop's tail), as (A tiles, scales)."""
    NRND: gl.constexpr = 8 * S + 2
    DEPTH: gl.constexpr = DN_DEPTH
    doffs = dn_offs()
    aw = ()
    asc = ()
    if DBG & 262144:  # debug A: every earlier memory op done first
        drain(tids())
    if DBG & 524288:  # debug B: weights only after the mids
        stage_mids(sc, tag, errors, bid, ms8, mc8, msh, mch, S)
        gl.barrier()
    if pre is not None:
        aw, asc = pre
    else:
        for i in gl.static_range(DEPTH):
            if i < NRND:
                a, sca = down_round(i, t, ids, w2, s2, S, DBG, doffs)
                aw = aw + (a,)
                asc = asc + (sca,)
    pin(tids())
    if (DBG & (1024 | 524288)) == 0:
        stamp(tr_, bid, 22)
        stage_mids(sc, tag, errors, bid, ms8, mc8, msh, mch, S, tr_)
    stamp(tr_, bid, 9)
    gl.barrier()
    acc = gl.zeros([128, 16], gl.float32, MFMA)
    BS: gl.constexpr = cdna4.get_mfma_scale_layout(B_L, [16, 4])
    for i in gl.static_range(NRND if (DBG & 16384) == 0 else 0):
        pin(tids())  # a round's LDS reads and next loads stay in it (else all hoisted: spills)
        if i + DEPTH < NRND:
            a, sca = down_round(i + DEPTH, t, ids, w2, s2, S, DBG, doffs)
            aw = aw + (a,)
            asc = asc + (sca,)
        if DBG & 32768:
            bv = gl.full([128, 16], 0x38, gl.uint8, B_L)
            bs = gl.full([16, 4], 127, gl.uint8, BS)
        elif i < 8 * S:
            bv = rnd(ms8, i % 8).permute((1, 0)).load(B_L)
            bs = rnd(mc8, i % 8).load(BS)
        else:
            bv = rnd(msh, i - 8 * S).permute((1, 0)).load(B_L)
            bs = rnd(mch, i - 8 * S).load(BS)  # 2 rounds of msh
        res = cdna4.mfma_scaled(aw[i], asc[i], "e2m1", bv.to(gl.float8e4nv, bitcast=True), bs, "e4m3",
                                zacc([128, 16], MFMA))
        acc = acc + res * down_wcol(i, wts, shared_weight, S)
    if DBG & 8192:
        return
    los, his = bd_reduce(acc, 2, 4, red)
    stamp(tr_, bid, 10)
    if DBG & 2048:
        return
    ffn_tail(t, li, n_layers, los, his, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, S, FFN_OFF,
             tr_, pf_q=pf_q, pf_o=pf_o)


@gluon.jit
def ffn_tail(t, li, n_layers, los, his, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid,
             S: gl.constexpr, FFN_OFF: gl.constexpr, tr_=None, aslot=0, pf_q=0, pf_o=0):
    if DENSE_FFN and HYB and (S > 16):
        ffn_tail_dense(t, li, n_layers, los, his, peers, peer_local, pbase, rank, ar_out, errors, bid, S, FFN_OFF,
                       tr_, aslot)
    else:
        ffn_tail_tagged(t, li, n_layers, los, his, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, S,
                        FFN_OFF, tr_, aslot, pf_q, pf_o, PF_NEXT)


@gluon.jit
def ffn_tail_dense(t, li, n_layers, los, his, peers, peer_local, pbase, rank, ar_out, errors, bid,
                   S: gl.constexpr, FFN_OFF: gl.constexpr, tr_=None, aslot=0):
    """ffn_tail past 16 rows (hybrid, AG_RS), dense: partials to the owner, its sums to every
    rank's fagd, read by the next layer's norm tasks (which also write the aux layers' ar
    out); the last layer's here."""
    NS: gl.constexpr = los[0].shape[1]
    if NOAUX:
        aslot = 0
    attn_peer, _, attn_own, _ = peer_regions(peers, peer_local, pbase, FFN_OFF)
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    nn = ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :]
    live = nn < S
    # dense: partials to the owner, its sums to every rank's fagd, read by the next layer's
    # norm tasks (which also write the aux layers' ar out); the last layer's here
    ffnd = dshift(attn_peer, FFND_OFF)
    for tt in gl.static_range(2):
        word = bf_pair(bf16_round(los[tt]), bf16_round(his[tt]))
        if DENSE_1S:  # one-shot: to every rank (its norm tasks sum the 4)
            for r in gl.static_range(4):
                put_dense(ffnd[r], (rank * MAX_S + nn) * HPAIRS + 16 * t + 8 * tt + jr, word, mask=live)
        else:
            put_dense(dpick(ffnd, t % 4), (rank * MAX_S + nn) * HPAIRS + 16 * t + 8 * tt + jr, word, mask=live)
    stamp(tr_, bid, 38)
    if TRACE == 3:  # (diagnostics) the puts acknowledged
        jr = wave_drain(jr)
        stamp(tr_, bid, 37)
    if DENSE_1S:
        if li == n_layers - 1:  # the last layer's sum here (no norm task reads it)
            own_d = dptr(attn_own, FFND_OFF)
            pks = ()
            for tt in gl.static_range(2):
                for r in gl.static_range(4):
                    pks = pks + (peek_dense(own_d, r * (MAX_S * HPAIRS) + nn * HPAIRS + 16 * t + 8 * tt + jr, live),)
            its = ()
            for tt in gl.static_range(2):
                for r in gl.static_range(4):
                    its = its + ((own_d, r * (MAX_S * HPAIRS) + nn * HPAIRS + 16 * t + 8 * tt + jr, pks[4 * tt + r]),)
            ws, err = wait_dense(its, 1)
            note_err(errors, bid, err, E_FFN)
            stamp(tr_, bid, 39)
            gl.barrier()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                for r in gl.static_range(4):
                    reset_dense(own_d, r * (MAX_S * HPAIRS) + pw, mask=live)
                t_lo = bf_lo(ws[4 * tt])
                t_hi = bf_hi(ws[4 * tt])
                for r in gl.static_range(1, 4):
                    t_lo = t_lo + bf_lo(ws[4 * tt + r])
                    t_hi = t_hi + bf_hi(ws[4 * tt + r])
                cdna4.buffer_store(bf_pair(t_lo, t_hi), i32p(ar_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)
    elif rank == t % 4:
        own_d = dptr(attn_own, FFND_OFF)
        fagd = dshift(attn_peer, FAGD_OFF)
        opk = ()
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                opk = opk + (peek_dense(own_d, r * (MAX_S * HPAIRS) + pw, live),)
        oit = ()
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                oit = oit + ((own_d, r * (MAX_S * HPAIRS) + pw, opk[4 * tt + r]),)
        ows, err = wait_dense(oit, 1)
        note_err(errors, bid, err, E_FFN)
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            o_lo = bf_lo(ows[4 * tt])
            o_hi = bf_hi(ows[4 * tt])
            for r in gl.static_range(1, 4):
                o_lo = o_lo + bf_lo(ows[4 * tt + r])
                o_hi = o_hi + bf_hi(ows[4 * tt + r])
            osw = bf_pair(o_lo, o_hi)
            for r in gl.static_range(4):
                put_dense(fagd[r], pw, osw, mask=live)
        gl.barrier()  # every replica's words read before the resets (NS < 64 replicates)
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                reset_dense(own_d, r * (MAX_S * HPAIRS) + pw, mask=live)
    if (not DENSE_1S) and (li == n_layers - 1):
        fagd_own = dptr(attn_own, FAGD_OFF)
        pks = ()
        for tt in gl.static_range(2):
            pks = pks + (peek_dense(fagd_own, nn * HPAIRS + 16 * t + 8 * tt + jr, live),)
        ws, err = wait_dense(((fagd_own, nn * HPAIRS + 16 * t + jr, pks[0]),
                              (fagd_own, nn * HPAIRS + 16 * t + 8 + jr, pks[1])), 1)
        note_err(errors, bid, err, E_FFN)
        stamp(tr_, bid, 39)
        gl.barrier()
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            reset_dense(fagd_own, pw, mask=live)
            cdna4.buffer_store(ws[tt], i32p(ar_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)


@gluon.jit
def ffn_tail_tagged(t, li, n_layers, los, his, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid,
             S: gl.constexpr, FFN_OFF: gl.constexpr, tr_=None, aslot=0, pf_q=0, pf_o=0, PF: gl.constexpr = 0):
    """Down task t's route-weighted sums (``los`` / ``his``: [8 row pairs, NS tokens] a
    16-row tile, PR_L) -> bf16 partials to every rank -> rank-ordered sum -> ar. From
    AG_RS_FROM rows: to the rows' owner (rank t % 4) only, which puts the sums to every rank."""
    NS: gl.constexpr = los[0].shape[1]
    AGRS: gl.constexpr = S >= AG_RS_FROM
    if NOAUX:
        aslot = 0
    outw = (li == n_layers - 1) | (aslot > 0)  # ar out: the last layer (slot 0), EAGLE3's aux layers
    _, ffn_peer, _, ffn_own = peer_regions(peers, peer_local, pbase, FFN_OFF)
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    nn = ar(NS, layout=gl.SliceLayout(0, PR_L))[None, :]
    live = nn < S
    if PUSH_W and (NS == 4) and (not AGRS):
        push_peers(bf_pair(bf16_round(los[0]), bf16_round(his[0])), bf_pair(bf16_round(los[1]), bf16_round(his[1])),
                   ffn_peer, rank, t, tag, S)
        if PF:
            # PF_NEXT: the next layer's weight lines touched now (the receive below is wave 0's);
            # their values are consumed after the receive so the loads stay in flight across it
            pt_ = ar(THREADS, T1)
            pfm = pt_ >= 64
            pq_ = lp(pf_q, gl.int32)
            pv0 = gl.load(pq_ + pt_ * 32, mask=pfm & (pf_q != 0))
            pv1 = gl.load(pq_ + (pt_ + THREADS) * 32, mask=pfm & (pf_q != 0) & (pt_ + THREADS < 768))
            if PF >= 2:
                po_ = lp(pf_o, gl.int32)
                pv2 = gl.load(po_ + pt_ * 32, mask=pfm & (pf_o != 0))
            else:
                pv2 = pv0 * 0
    else:
        for tt in gl.static_range(2):
            word = bf_pair(bf16_round(los[tt]), bf16_round(his[tt]))
            at = (rank * MAX_S + nn) * HPAIRS + 16 * t + 8 * tt + jr
            if AGRS:
                put_sys(peer_pick(ffn_peer, t % 4), at, word, tag, mask=live)
            else:
                for r in gl.static_range(4):
                    put_sys(ffn_peer[r], at, word, tag, mask=live)
    stamp(tr_, bid, 38)
    if AGRS:
        fag_own = (ffn_own.to(gl.int64, bitcast=True) + (FAG_OFF - FFN_OFF)).to(gl.pointer_type(gl.int64),
                                                                                bitcast=True)
        if rank == t % 4:
            fag_peer = peer_shift(ffn_peer, FAG_OFF - FFN_OFF)
            opk = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                for r in gl.static_range(4):
                    opk = opk + (peek(ffn_own, r * (MAX_S * HPAIRS) + pw, tag, live),)
            oit = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                for r in gl.static_range(4):
                    oit = oit + ((ffn_own, r * (MAX_S * HPAIRS) + pw, opk[4 * tt + r]),)
            ows = wait_group(oit, tag, errors, bid, E_FFN, 1)
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                o_lo = bf_lo(ows[4 * tt])
                o_hi = bf_hi(ows[4 * tt])
                for r in gl.static_range(1, 4):
                    o_lo = o_lo + bf_lo(ows[4 * tt + r])
                    o_hi = o_hi + bf_hi(ows[4 * tt + r])
                osw = bf_pair(o_lo, o_hi)
                for r in gl.static_range(4):
                    put_sys(fag_peer[r], pw, osw, tag, mask=live)
        # hybrid: the next layer's readers poll fag itself (no copy to ar): the last
        # layer's output only
        if (HYB == 0) | outw:
            pks = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                pks = pks + (peek(fag_own, pw, tag, live),)
            items = ()
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                items = items + ((fag_own, pw, pks[tt]),)
            ws = wait_group(items, tag, errors, bid, E_FFN, 1)
            stamp(tr_, bid, 39)
            for tt in gl.static_range(2):
                pw = nn * HPAIRS + 16 * t + 8 * tt + jr
                word = ws[tt]
                if PLAIN_IN and not HYB:
                    cdna4.buffer_store(word, i32p(rgn(sc, R_AR)), pw, mask=live)
                else:
                    put(rgn(sc, R_AR), pw, word, tag, mask=live)
                if outw:
                    cdna4.buffer_store(word, i32p(ar_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)
    elif PUSH_W and (NS == 4):
        pw, ok, t_lo, t_hi, _ = ar_recv(ffn_own, t, tag, errors, bid, S, E_FFN)
        stamp(tr_, bid, 39)
        if PF and (not AGRS):
            gl.store(errors + bid + pt_ * 0, pv0 + pv1 + pv2, mask=(pt_ < 0) & (bid < 0))  # (never: keeps the loads)
        word = bf_pair(t_lo, t_hi)
        if PLAIN_IN and not HYB:
            cdna4.buffer_store(word, i32p(rgn(sc, R_AR)), pw, mask=ok)
        else:
            put(rgn(sc, R_AR), pw, word, tag, mask=ok)
        if outw:
            cdna4.buffer_store(word, i32p(ar_out), aslot * (MAX_S * HPAIRS) + pw, mask=ok)
    else:
        pks = ()
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                pks = pks + (peek(ffn_own, r * (MAX_S * HPAIRS) + pw, tag, live),)
        items = ()
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            for r in gl.static_range(4):
                items = items + ((ffn_own, r * (MAX_S * HPAIRS) + pw, pks[4 * tt + r]),)
        ws = wait_group(items, tag, errors, bid, E_FFN, 1)
        stamp(tr_, bid, 39)
        for tt in gl.static_range(2):
            pw = nn * HPAIRS + 16 * t + 8 * tt + jr
            t_lo = bf_lo(ws[4 * tt])
            t_hi = bf_hi(ws[4 * tt])
            for r in gl.static_range(1, 4):
                t_lo = t_lo + bf_lo(ws[4 * tt + r])
                t_hi = t_hi + bf_hi(ws[4 * tt + r])
            word = bf_pair(t_lo, t_hi)
            if PLAIN_IN and not HYB:
                cdna4.buffer_store(word, i32p(rgn(sc, R_AR)), pw, mask=live)
            else:
                put(rgn(sc, R_AR), pw, word, tag, mask=live)
            if outw:
                cdna4.buffer_store(word, i32p(ar_out), aslot * (MAX_S * HPAIRS) + pw, mask=live)


@gluon.jit
def mid_put(sc, k_or_all, slot, cg, g_lo, g_hi, u_lo, u_hi, limit, tag, S: gl.constexpr, col=-1):
    """A task's 16 mid values (rows 16 cg ..) -> bf16 pairs of token k (k_or_all >= 0; its
    B columns' group ``col``, default k) or of every token (the shared expert, k_or_all < 0)."""
    jr = ar(8, layout=gl.SliceLayout(1, PR_L))[:, None]
    nn = ar(4, layout=gl.SliceLayout(0, PR_L))[None, :]
    word = bf_pair(bf16_round(swiglu(g_lo, u_lo, limit)), bf16_round(swiglu(g_hi, u_hi, limit)))
    cs = gl.where(col < 0, k_or_all, col)
    mine = gl.where(k_or_all < 0, nn < S, nn == cs)
    dst = gl.where(k_or_all < 0, nn, k_or_all)
    put(rgn(sc, R_MID), (dst * SLOTS + slot) * (INTER // 2) + 8 * cg + jr, word, tag, mask=mine)


SH_HALF = gl.constexpr(int(os.environ.get("VLLM_M3_GLUON_SH_HALF", "0")))  # 4 rows: 96 shared half tasks
SHH_BASE = gl.constexpr(160)  # the half tasks' CTAs (160 .. 255)


@gluon.jit
def shh_offs():
    """A shared half task's loop-invariant lane offsets (FlyDSL's sh_load: gate rows 8 j .. + 8
    and up rows 768 + 8 j .. in one 16-row tile, the waves splitting k). A tile [128, 64 B]:
    virtual row m = 16 p + rr (wave p takes 128-k step 8 r + p of round r); rr < 8: gate row
    8 j + rr, else up row 768 + 8 j + rr - 8 -> (rr / 8) 48 row groups + p 1024 + (rr % 8) 16
    + 256 c + byte (j and r in the scalar base). Scales: (expert row, col) per element."""
    m = w4_rows()[:, None, None]
    c = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(2, W4_L)))[None, :, None]
    b = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(1, W4_L)))[None, None, :]
    rr = m & 15
    off = (rr >> 3) * (48 * 16 * (HIDDEN // 2)) + (m >> 4) * 1024 + (rr & 7) * 16 + c * 256 + b
    ASL: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
    sm = ar(128, layout=gl.SliceLayout(1, ASL))[:, None]
    sg = ar(4, layout=gl.SliceLayout(0, ASL))[None, :]
    return off, sm, sg


@gluon.jit
def shh_loads(j, w13, s13, offs):
    """Half task j's 6 rounds of (A tile, A scales), in flight."""
    off, sm, sg = offs
    srr = sm & 15
    srow = N_ROUTED * (2 * INTER) + gl.where(srr < 8, 8 * j + srr, INTER + 8 * j + srr - 8)
    base = w13 + N_ROUTED * (2 * INTER * (HIDDEN // 2)) + (j // 2) * (16 * (HIDDEN // 2)) + (j % 2) * 128
    aq = ()
    sq = ()
    for r in gl.static_range(6):
        aq = aq + (w4_tile(base + (8 * r) * 1024, off),)
        sq = sq + (a_scales(s13, srow, (8 * r + (sm >> 4)) * 4 + sg, HIDDEN // 32),)
    return aq, sq


@gluon.jit
def shh_mma(j, aq, sq, xha, xhb, cha, chb, red, sc, limit, tag, S: gl.constexpr):
    """Half task j against every token's xn ([round, token 8 + wave, 128 B]) -> swiglu -> the
    shared slot's mid rows 8 j .. + 8 of every token (gate row r with up row r + 8)."""
    acc = zacc([128, 32], MFMA)
    BS: gl.constexpr = cdna4.get_mfma_scale_layout(B_L, [32, 4])
    for r in gl.static_range(6):
        if r < 4:
            bv = rnd(xha, r).permute((1, 0)).load(B_L)
            bs = rnd(cha, r).load(BS)
        else:
            bv = rnd(xhb, r - 4).permute((1, 0)).load(B_L)
            bs = rnd(chb, r - 4).load(BS)
        acc = cdna4.mfma_scaled(aq[r], sq[r], "e2m1", bv.to(gl.float8e4nv, bitcast=True), bs, "e4m3", acc)
    # bd_reduce's first half (wave p's column group p kept, [128, 4 tokens] through LDS), then
    # the gate rows 2 q, 2 q + 1 and the up rows 8 + 2 q, 9 + 2 q, the waves summed in order
    c = ar(32, layout=gl.SliceLayout(0, MFMA))[None, :]
    m = ar(128, layout=gl.SliceLayout(1, MFMA))[:, None]
    mine = gl.where(c % 8 == (m // 16) % 8, acc, 0.0)
    rsum = gl.sum(gl.reshape(mine, [128, 4, 8]), axis=2)
    gl.barrier()
    red.store(rsum)
    gl.barrier()
    WL: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [8, 1], [1, 0])
    q = ar(4, layout=gl.SliceLayout(1, WL))[:, None]
    nn = ar(4, layout=gl.SliceLayout(0, WL))[None, :]
    g_lo = gl.zeros([4, 4], gl.float32, WL)
    g_hi = gl.zeros([4, 4], gl.float32, WL)
    u_lo = gl.zeros([4, 4], gl.float32, WL)
    u_hi = gl.zeros([4, 4], gl.float32, WL)
    for p in gl.static_range(8):
        g_lo = g_lo + red.gather(p * 16 + 2 * q + nn * 0, 0)
        g_hi = g_hi + red.gather(p * 16 + 2 * q + 1 + nn * 0, 0)
        u_lo = u_lo + red.gather(p * 16 + 8 + 2 * q + nn * 0, 0)
        u_hi = u_hi + red.gather(p * 16 + 9 + 2 * q + nn * 0, 0)
    word = bf_pair(bf16_round(swiglu(g_lo, u_lo, limit)), bf16_round(swiglu(g_hi, u_hi, limit)))
    put(rgn(sc, R_MID), (nn * SLOTS + TOP_K) * (INTER // 2) + 4 * j + q, word, tag, mask=nn < S)


@gluon.jit
def stage_moe(li, ptab, n_layers, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, route_scale,
              shared_weight, swiglu_limit, S: gl.constexpr, FFN_OFF: gl.constexpr, STAGES: gl.constexpr = 511,
              tr_=None):
    NR: gl.constexpr = 4 * UG_TASKS * S  # routed up / gate tasks
    MAXT: gl.constexpr = (NR + BLOCKS - 1) // BLOCKS
    has_r = (bid < NR) & ((STAGES & 128) != 0)
    has_dn = (bid < N_DN) & ((STAGES & 256) != 0) & ((bid > 0) | ((STAGES & 65536) == 0))
    SHH: gl.constexpr = SH_HALF and (S == 4)
    sh_j = bid - SHARED_BASE
    has_sh = (sh_j >= 0) & (sh_j < UG_TASKS) & ((STAGES & 64) != 0) & (not SHH)
    hj = bid - SHH_BASE
    has_hh = (hj >= 0) & (hj < 2 * UG_TASKS) & ((STAGES & 64) != 0) & SHH
    if has_r | has_dn | has_sh | has_hh:
        e = lrow(ptab, li, (12, 13, 14, 15))  # w13, s13, w2, s2: one scalar round trip
        w13 = lp(e[0], gl.uint8)
        s13 = lp(e[1], gl.uint8)
        xsa = gl.allocate_shared_memory(gl.uint8, [8, 16, 128], BSW)
        xsb = gl.allocate_shared_memory(gl.uint8, [4, 16, 128], BSW)
        csa = gl.allocate_shared_memory(gl.uint8, [8, 16, 4], FLAT3)
        csb = gl.allocate_shared_memory(gl.uint8, [4, 16, 4], FLAT3)
        red = gl.allocate_shared_memory(gl.float32, [128, 4], FLAT2)
        if SHH:
            xha = gl.allocate_shared_memory(gl.uint8, [4, 32, 128], BSW)
            xhb = gl.allocate_shared_memory(gl.uint8, [2, 32, 128], BSW)
            cha = gl.allocate_shared_memory(gl.uint8, [4, 32, 4], FLAT3)
            chb = gl.allocate_shared_memory(gl.uint8, [2, 32, 4], FLAT3)
            hoffs = shh_offs()
        uoffs = ug_offs()
        # SH_AFTER (4 rows, FlyDSL's shared_after_routing): the shared tasks' weights and xn go
        # out first, but their GEMV runs after the routing, under the first routed weights
        SH_AFTER: gl.constexpr = SHARED_AFTER and (S == 4)
        shj = gl.minimum(gl.maximum(sh_j, 0), UG_TASKS - 1)
        SH_LATE: gl.constexpr = SH_LOAD_LATE or SH_RFIRST
        if has_sh:
            # the shared expert needs no routing: its weights, then every token's xn (SH_AFTER:
            # the weights after the routing -- issued before the score polls, the in-order
            # vmcnt would hold the routing until they land)
            if SH_AFTER and SH_RFIRST:
                # (ATT: ~400 loads a wave ahead of the score poll held the routing 3-4 us) the
                # routing first: the shared weights and every token's xn after it
                pass
            elif SH_AFTER and SH_LATE:
                stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, sh_j >= 0, S)
            elif SH_AFTER:
                aq_s, sq_s = ug_loads(sh_j * 0 + N_ROUTED, shj, w13, s13, uoffs)
                stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, sh_j >= 0, S)
            else:
                aq_s, sq_s = ug_loads(sh_j * 0 + N_ROUTED, shj, w13, s13, uoffs)
                stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, sh_j >= 0, S)
                gl.barrier()
                g_lo, g_hi, up_lo, up_hi = ug_mma(aq_s, sq_s, xsa, xsb, csa, csb, red)
                mid_put(sc, -1, TOP_K, shj, g_lo, g_hi, up_lo, up_hi, swiglu_limit, tag, S)
                stamp(tr_, bid, 12)
        else:
            if SH_AFTER and (not SH_LATE):
                ASE: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
                aq_s = ()
                sq_s = ()
                for r in gl.static_range(12):
                    aq_s = aq_s + (gl.full([128, 64], 0, gl.uint8, A_L),)
                    sq_s = sq_s + (gl.full([128, 4], 0, gl.uint8, ASE),)
        # TOKC (4 rows): CTA b's routed tasks are token b // 64's (FlyDSL's token-contiguous
        # up / gate map): a CTA without a shared task stages that token's xn alone
        TOKC: gl.constexpr = UG_TOKC and (S == 4)
        tokc = bid // 64
        colr = gl.where(has_sh, tokc, 0)
        if SHH:
            # the half tasks' weights issued early (as the full shared task's)
            ASL6: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
            if has_hh:
                aq_h, sq_h = shh_loads(gl.minimum(gl.maximum(hj, 0), 2 * UG_TASKS - 1), w13, s13, hoffs)
            else:
                aq_h = ()
                sq_h = ()
                for r in gl.static_range(6):
                    aq_h = aq_h + (gl.full([128, 64], 0, gl.uint8, A_L),)
                    sq_h = sq_h + (gl.full([128, 4], 0, gl.uint8, ASL6),)
        if XN_FIRST:
            if has_r & (not has_sh):
                # FlyDSL's order: xn (published before the scores) staged while the routing
                # is still out -- after the weights it would wait for all of them (in-order vmcnt)
                if TOKC:
                    stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, bid >= 0, S, tokc, True)
                else:
                    stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, bid >= 0, S)
        if has_r | has_dn:
            if ROUTE_W and (S <= 4):
                ids, wts = route_wave(sc, bid, tag, errors, route_scale, bid >= 0, S, tr_)
            else:
                ids, wts = route_all(sc, bid, tag, errors, route_scale, bid >= 0, S, tr_)
            stamp(tr_, bid, 6)
            if STAGES & 131072:  # debug: every wave's view of the expert ids -> trace[bid, 4 w + i]
                tt_ = tids()
                for i in gl.static_range(4):
                    gl.store(tr_ + bid * 32 + (tt_ // 64) * 4 + i, (tt_ * 0 + ids[i]).to(gl.int64), mask=tt_ % 64 == 0)
            rbid = bid
            if STAGES & 1048576:  # debug: routed tasks rotated by 48 CTAs
                rbid = (bid + 48) % NR
            if has_r:
                # the CTA's routed tasks bid, bid + 256, .. as one stream of 12-round tile
                # pairs, UG_DEPTH rounds in flight across task boundaries
                TOT: gl.constexpr = MAXT * 12
                if SH_AFTER and SH_LATE:
                    if has_sh:
                        aq_s, sq_s = ug_loads(sh_j * 0 + N_ROUTED, shj, w13, s13, uoffs)
                    else:
                        ASL: gl.constexpr = cdna4.get_mfma_scale_layout(A_L, [128, 4])
                        aq_s = ()
                        sq_s = ()
                        for r in gl.static_range(12):
                            aq_s = aq_s + (gl.full([128, 64], 0, gl.uint8, A_L),)
                            sq_s = sq_s + (gl.full([128, 4], 0, gl.uint8, ASL),)
                aw = ()
                asc = ()
                for j in gl.static_range(UG_DEPTH):
                    if j < TOT:
                        uj = gl.minimum(rbid + (j // 12) * BLOCKS, NR - 1)
                        if TOKC:
                            uj = tokc * 192 + (bid % 64) + 64 * (j // 12)
                        ej = expert_of(ids, uj // 192, (uj % 192) // 48, S)
                        if STAGES & 4194304:  # debug: each wave's expert of rounds 0..3 -> trace[bid, 4 w + j]
                            if j < 4:
                                tt2 = tids()
                                gl.store(tr_ + bid * 32 + (tt2 // 64) * 4 + j, (tt2 * 0 + ej).to(gl.int64),
                                         mask=tt2 % 64 == 0)
                        a, sca = ug_round(ej, uj % 48, j % 12, w13, s13, uoffs)
                        aw = aw + (a,)
                        asc = asc + (sca,)
                pin(tids())
                if not has_sh:
                    if not XN_FIRST:
                        stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, bid >= 0, S)
                    gl.barrier()
                elif SH_AFTER and SH_RFIRST:
                    stage_xn(sc, tag, errors, bid, xsa, xsb, csa, csb, sh_j >= 0, S)
                if SH_AFTER:
                    if has_sh:
                        gl.barrier()
                        g_lo, g_hi, up_lo, up_hi = ug_mma(aq_s, sq_s, xsa, xsb, csa, csb, red)
                        mid_put(sc, -1, TOP_K, shj, g_lo, g_hi, up_lo, up_hi, swiglu_limit, tag, S)
                        stamp(tr_, bid, 12)
                if SHH:
                    if has_hh:
                        # every token's xn again (landed long ago), the half tasks' layout
                        xn_part(sc, tag, errors, bid, xha, cha, bid >= 0, 0, 32, 4, S, 0, 8)
                        xn_part(sc, tag, errors, bid, xhb, chb, bid >= 0, 32, 16, 4, S, 0, 8)
                        gl.barrier()
                        shh_mma(gl.minimum(gl.maximum(hj, 0), 2 * UG_TASKS - 1), aq_h, sq_h, xha, xhb, cha, chb,
                                red, sc, swiglu_limit, tag, S)
                        stamp(tr_, bid, 12)
                stamp(tr_, bid, 7)
                # a down CTA streams its first DN_DEPTH down rounds into the slots the loop's
                # last UG_DEPTH rounds free (as FlyDSL): they land before the mids do
                DNB: gl.constexpr = STAGES & 3145216
                w2p = lp(e[2], gl.uint8)
                s2p = lp(e[3], gl.uint8)
                daw = ()
                dsc = ()
                doffs = dn_offs()
                acc = zacc([128, 16], MFMA)
                for j in gl.static_range(TOT):
                    pin(tids())
                    if j + UG_DEPTH < TOT:
                        jn = j + UG_DEPTH
                        un = gl.minimum(rbid + (jn // 12) * BLOCKS, NR - 1)
                        if TOKC:
                            un = tokc * 192 + (bid % 64) + 64 * (jn // 12)
                        a, sca = ug_round(expert_of(ids, un // 192, (un % 192) // 48, S), un % 48, jn % 12, w13, s13,
                                          uoffs)
                        aw = aw + (a,)
                        asc = asc + (sca,)
                    elif DN_STREAM and (j + UG_DEPTH - TOT < DN_DEPTH):
                        a, sca = down_round(j + UG_DEPTH - TOT, bid, ids, w2p, s2p, S, DNB, doffs)
                        daw = daw + (a,)
                        dsc = dsc + (sca,)
                    if j % 12 == 0:
                        acc = zacc([128, 16], MFMA)
                    bv, bs = ug_bround(j % 12, xsa, xsb, csa, csb)
                    acc = cdna4.mfma_scaled(aw[j], asc[j], "e2m1", bv.to(gl.float8e4nv, bitcast=True), bs, "e4m3",
                                            acc)
                    if j % 12 == 11:
                        u = rbid + (j // 12) * BLOCKS
                        los, his = bd_reduce(acc, 2, 4, red)
                        if TOKC:
                            u = tokc * 192 + (bid % 64) + 64 * (j // 12)
                            mid_put(sc, tokc, (u % 192) // 48, u % 48, los[0], his[0], los[1], his[1],
                                    swiglu_limit, tag, S, colr)
                        elif u < NR:
                            mid_put(sc, u // 192, (u % 192) // 48, u % 48, los[0], his[0], los[1], his[1],
                                    swiglu_limit, tag, S)
                stamp(tr_, bid, 8)
                if has_dn:
                    if DN_STREAM:
                        for d in gl.static_range(UG_DEPTH, DN_DEPTH):
                            a, sca = down_round(d, bid, ids, w2p, s2p, S, DNB, doffs)
                            daw = daw + (a,)
                            dsc = dsc + (sca,)
                        run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag,
                                 errors, shared_weight, S, FFN_OFF, STAGES, tr_, (daw, dsc), ptab)
                    else:
                        run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag,
                                 errors, shared_weight, S, FFN_OFF, STAGES, tr_, None, ptab)
            if has_dn & ~has_r:  # (debug STAGES only: a down CTA is always a routed one)
                run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag, errors,
                         shared_weight, S, FFN_OFF, STAGES, tr_, None, ptab)


@gluon.jit
def run_down(bid, li, n_layers, ids, wts, e, red, sc, peers, peer_local, pbase, rank, ar_out, tag, errors,
             shared_weight, S: gl.constexpr, FFN_OFF: gl.constexpr, STAGES: gl.constexpr, tr_, pre=None, ptab=0):
    """The CTA's down task (``e``: the MoE stage's pointer row w13, s13, w2, s2)."""
    w2 = lp(e[2], gl.uint8)
    s2 = lp(e[3], gl.uint8)
    ms8 = gl.allocate_shared_memory(gl.uint8, [8, 16, 128], BSW)
    mc8 = gl.allocate_shared_memory(gl.uint8, [8, 16, 4], FLAT3)
    msh = gl.allocate_shared_memory(gl.uint8, [2, 16, 128], BSW)
    mch = gl.allocate_shared_memory(gl.uint8, [2, 16, 4], FLAT3)
    pf_q = 0
    pf_o = 0
    if PF_NEXT:
        nli = gl.minimum(li + 1, n_layers - 1)
        have = (li + 1) < n_layers
        en = lrow(ptab, nli, (0, 7))
        pf_q = gl.where(have & (bid < ROWS16), en[0] + bid * (16 * 6144), 0)
        if PF_NEXT >= 2:
            pf_o = gl.where(have & (bid < N_O), en[1] + bid * (2 * 16 * 2048), 0)
    if (STAGES & 4096) == 0:
        stage_down(bid, li, n_layers, ids, wts, w2, s2, ms8, mc8, msh, mch, red, sc, peers, peer_local, pbase,
                   rank, ar_out, tag, errors, bid, shared_weight, S, FFN_OFF, STAGES & 3145216, tr_, pre, pf_q, pf_o)


# ============================================================ MoE (wide: S >= WIDE_FROM)
# FlyDSL's expert-grouped MoE: every CTA routes every token and lists the step's distinct
# experts in expert order, the shared one last ("units"). Up / gate tasks are (unit, 32-row
# block) pairs on every CTA, a B column a token, each publishing its MXFP8 mid rows of the
# tokens routed there at row 5 k + slot; a down task (32 rows) runs every unit, its B
# gathered per column from those rows (the zero row where the token is not routed there).
# A wave takes its own K slice of a 16-row tile -- a batched MFMA, a batch a wave, so the
# B columns are the tokens (no K slices packed into them) -- and the waves' partials are
# summed in wave order (FlyDSL's row_sum).
MF3 = gl.constexpr(cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1, 1]))
A3 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MF3, k_width=16))
B3 = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MF3, k_width=16))
# the scaled MFMA's scales [8 waves, 16 rows / columns, 4 32-blocks]: lane = row + 16 block
SL3 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[], lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
    warp_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0]], block_bases=[], shape=[8, 16, 4]))
# a down B gathered as int64 words of 128-B lines: [128 = (wave, column), 16 words], lane =
# 16 (16-B chunk % 4) + column, registers the chunk's 2 words and chunk + 4's -- B3 once
# split into bytes; and its codes [128 = (wave, column), 4] (SL3 flattened)
G64 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 8]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 2], [0, 4]],
    warp_bases=[[16, 0], [32, 0], [64, 0]], block_bases=[], shape=[128, 16]))
G8 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
    warp_bases=[[16, 0], [32, 0], [64, 0]], block_bases=[], shape=[128, 4]))
# G64's bytes: [128 = (wave, column), 128 B], a thread's 16-B chunks (bytes 16 c .. + 16 and
# 64 + 16 c ..): B3 once reshaped [8, 16, 128] and permuted -- loaded as bytes, no unpacking
G8B = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 64]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 32]],
    warp_bases=[[16, 0], [32, 0], [64, 0]], block_bases=[], shape=[128, 128]))
RT16 = gl.constexpr(gl.BlockedLayout([2, 2], [1, 64], [8, 1], [1, 0]))  # [16 tokens, 128 experts]
UT = gl.constexpr(gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0]))  # [units, 128 experts]
TT = gl.constexpr(gl.BlockedLayout([1, 1], [4, 16], [8, 1], [1, 0]))  # [units, 16 columns]
RL2 = gl.constexpr(gl.BlockedLayout([1, 1], [16, 4], [1, 8], [0, 1]))  # [16 rows, 16 columns]: one a thread (x2)
XQ3 = gl.constexpr(gl.BlockedLayout([1, 1, 4], [1, 8, 8], [8, 1, 1], [2, 1, 0]))  # [tokens, 8 waves, 32 words]
CQ3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [2, 8, 4], [8, 1, 1], [2, 1, 0]))  # [tokens, 8 waves, 4 codes]
MW3 = gl.constexpr(gl.BlockedLayout([1, 1, 4], [1, 8, 8], [8, 1, 1], [2, 1, 0]))  # [mid rows, 8 lines, 32 words]
MC3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [2, 8, 4], [8, 1, 1], [2, 1, 0]))  # [mid rows, 8 lines, 4 codes]
XSW4 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=8, order=[3, 2, 1, 0]))
XSW3 = gl.constexpr(gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=8, order=[2, 1, 0]))
W13_BYTES = gl.constexpr(2 * INTER * (HIDDEN // 2))
W2_BYTES = gl.constexpr(HIDDEN * (INTER // 2))
MID_WORDS = gl.constexpr(INTER // 4)
MID_CODES = gl.constexpr(INTER // 32)
UG_PAIRS = gl.constexpr(INTER // 32)  # 32-row blocks a unit


@gluon.constexpr_function
def units_max(S):
    """Units a step can have: every token's 4 routed experts distinct, + the shared one."""
    S = getattr(S, "value", S)
    return min(128, 4 * S) + 1


@gluon.constexpr_function
def units_pad(S):
    """Table rows: units_max rounded up to 2^n (always > units_max: row UP - 1 is the dump)."""
    S = getattr(S, "value", S)
    return 1 << min(128, 4 * S).bit_length()


@gluon.constexpr_function
def mid_rows(S):
    """LDS mid rows of a down task: the S 5 (token, slot) rows, padded to 2^n, the last zero."""
    S = getattr(S, "value", S)
    return 1 << (S * 5).bit_length()


@gluon.constexpr_function
def mid_lines(S):
    """LDS 128-B mid lines of a down task: line 6 row + chunk of the S 5 rows, padded to 2^n
    (the last line zero: unrouted columns and missing chunks read it)."""
    S = getattr(S, "value", S)
    return 1 << (S * 30).bit_length()


@gluon.constexpr_function
def wide_pairs(S):
    """Up / gate pairs a CTA runs at most."""
    S = getattr(S, "value", S)
    return -(-(min(128, 4 * S) + 1) * 24 // 256)


@gluon.jit
def lds_row(buf, u):
    """Row u of a 2-D LDS table (``index`` wants a rank-1 layout view first)."""
    return buf._reinterpret(buf.dtype, buf.shape, FLAT1).index(u)


@gluon.jit
def route_wide(sc, bid, tag, errors, route_scale, S: gl.constexpr, tr_=None):
    """Every token's top-4 (route_all's picks) -> (ids, weights): TOP_K [16] tensors of
    token rows (RT16), the weights the picks' sigmoids renormalized and scaled by route_scale."""
    kk = ar(16, layout=gl.SliceLayout(1, RT16))[:, None]
    e = ar(128, layout=gl.SliceLayout(0, RT16))[None, :]
    at = (((bid % SCORE_COPIES) * MAX_S + kk) * N_ROUTED + e) * 2
    ok = (kk < S) & (e >= 0)
    scr = rgn(sc, R_SC)
    if ROUTE_GATE_W:
        # one word a router task (token k, experts 8 tq ..: its last expert's sigmoid) polled
        # by 16 S lanes, then the table once: 256 CTAs spinning on the whole 16 KB table
        # (S = 8) saturate the fabric, the scores then landing 4-5 us after they were put
        lt = ar(THREADS, T1)
        gate_poll(scr, (((bid % SCORE_COPIES) * MAX_S + lt // 16) * N_ROUTED + 8 * (lt % 16) + 7) * 2 + 1,
                  lt < 16 * S, tag, errors, bid, E_SC)
    pk_k = peek(scr, at, tag, ok)
    pk_s = peek(scr, at + 1, tag, ok)
    stamp(tr_, bid, 40)
    rk, sw = wait_group(((scr, at, pk_k), (scr, at + 1, pk_s)), tag, errors, bid, E_SC, 4)
    stamp(tr_, bid, 41)
    sig = sw.to(gl.float32, bitcast=True)
    keys = gl.where(ok, rk, INT_MIN)
    ids = ()
    ws = ()
    for p in gl.static_range(TOP_K):
        mx = gl.max(keys, axis=1)
        keys = gl.where(keys == mx[:, None], INT_MIN, keys)
        idp = 127 - (mx & 127)
        ids = ids + (idp,)
        ws = ws + (gl.sum(gl.where(e == idp[:, None], sig, 0.0), axis=1),)
    scale = route_scale / gl.maximum((ws[0] + ws[1]) + (ws[2] + ws[3]), 1e-20)
    wsc = ()
    for p in gl.static_range(TOP_K):
        wsc = wsc + (ws[p] * scale,)
    return ids, wsc


@gluon.jit
def ballot_rank(nz):
    """[128] 0/1 (SliceLayout(0, RT16): lane l experts 2 l, 2 l + 1, in every warp) -> each
    expert's rank among the set ones (an exclusive prefix count): two ballots and mbcnt."""
    return gl.inline_asm_elementwise(
        "s_nop 4\n"
        "v_cmp_ne_u32_e64 s[92:93], 0, $2\n"
        "v_cmp_ne_u32_e64 s[94:95], 0, $3\n"
        "s_nop 4\n"
        "v_mbcnt_lo_u32_b32 $0, s92, 0\n"
        "v_mbcnt_hi_u32_b32 $0, s93, $0\n"
        "v_mbcnt_lo_u32_b32 $0, s94, $0\n"
        "v_mbcnt_hi_u32_b32 $0, s95, $0\n"
        "v_add_u32_e32 $1, $0, $2",
        "=&v,=&v,v,v,~{s92},~{s93},~{s94},~{s95}", [nz], dtype=gl.int32, is_pure=True, pack=2)


@gluon.jit
def wide_tables(ids, ws, shared_weight, eu_s, urow_s, uwt_s, S: gl.constexpr, tr_=None, bid=0):
    """The step's units -> nu (routed units; the shared expert is unit nu). LDS (row UP - 1,
    never a unit, the dump of the scatters): eu_s [UP] unit -> expert (-1 past nu); urow_s /
    uwt_s [UP, 16]
    (unit, column) -> the column's mid row 5 k + slot and route weight (the zero row and 0
    where it is not routed there). Ranks by ballot, tables by LDS scatter: no layout
    conversion through LDS (each a barrier round trip)."""
    UP: gl.constexpr = units_pad(S)
    ZROW: gl.constexpr = mid_rows(S) - 1
    kk = ar(16, layout=gl.SliceLayout(1, RT16))[:, None]
    e = ar(128, layout=gl.SliceLayout(0, RT16))[None, :]
    hit = e == ids[0][:, None]
    for p in gl.static_range(1, TOP_K):
        hit = hit | (e == ids[p][:, None])
    hit = hit & (kk < S)
    stamp(tr_, bid, 44)
    nz = (gl.sum(gl.where(hit, 1, 0), axis=0) != 0).to(gl.int32)  # [128]: some token picked e
    rank = ballot_rank(nz)
    nu = gl.sum(nz, axis=0)
    stamp(tr_, bid, 45)
    rk_s = gl.allocate_shared_memory(gl.int32, [128], FLAT1)
    rk_s.store(rank)
    pid_s = gl.allocate_shared_memory(gl.int32, [TOP_K, 16], FLAT2)
    pws_s = gl.allocate_shared_memory(gl.float32, [TOP_K, 16], FLAT2)
    for p in gl.static_range(TOP_K):
        lds_row(pid_s, p).store(ids[p])
        lds_row(pws_s, p).store(ws[p])
    eu_s.store(gl.full([UP], -1, gl.int32, gl.BlockedLayout([1], [64], [8], [0])))
    urow_s.store(gl.full([UP, 16], ZROW, gl.int32, TT))
    uwt_s.store(gl.zeros([UP, 16], gl.float32, TT))
    gl.barrier()
    e1 = ar(128, layout=gl.SliceLayout(0, RT16))
    DUMP: gl.constexpr = UP - 1
    eu_s.scatter(e1, gl.where(nz != 0, rank, DUMP), 0)
    stamp(tr_, bid, 46)
    cc = ar(16, layout=gl.SliceLayout(0, TT))
    for p in gl.static_range(TOP_K):
        u = rk_s.gather(lds_row(pid_s, p).load(gl.SliceLayout(0, TT)), 0)
        tgt = gl.where(cc < S, u, DUMP)[None, :]
        urow_s.scatter((cc * SLOTS + p)[None, :], tgt, 0)
        uwt_s.scatter(lds_row(pws_s, p).load(gl.SliceLayout(0, TT))[None, :], tgt, 0)
    tsh = gl.where(cc < S, nu, DUMP)[None, :]
    urow_s.scatter((cc * SLOTS + TOP_K)[None, :], tsh, 0)
    uwt_s.scatter(gl.full([1, 16], 0.0, gl.float32, TT) + shared_weight, tsh, 0)
    one = gl.full([1], 0, gl.int32, gl.BlockedLayout([1], [64], [8], [0]))
    eu_s.scatter(one + N_ROUTED, one + nu, 0)
    stamp(tr_, bid, 47)
    gl.barrier()
    stamp(tr_, bid, 48)
    return nu


@gluon.jit
def _add_i32(a, b):
    return a + b


@gluon.jit
def unit_expert_v(eu_s, u, L: gl.constexpr):
    """Unit u's expert (>= 0) in every lane, as an [8] tensor of layout L (a per-wave index
    slice of an offset layout): one broadcast LDS read, no scalar move (each v_readfirstlane
    waits for its read: ~2.5 us for a down task's 33 units)."""
    return gl.maximum(eu_s.gather(gl.full([8], 0, gl.int32, L) + u, 0), 0)


AW_L = gl.constexpr(gl.SliceLayout(1, gl.SliceLayout(2, A3)))  # [8 waves] of an A3 tensor
SW_L = gl.constexpr(gl.SliceLayout(1, gl.SliceLayout(2, SL3)))  # [8 waves] of an SL3 tensor


@gluon.jit
def uniform_p(x):
    """``uniform`` as a pure asm: LLVM may then issue every LDS read feeding a batch of them
    first (a side-effecting asm keeps each read behind the previous one: a latency each)."""
    return gl.inline_asm_elementwise("s_nop 4\nv_readfirstlane_b32 $0, $1\ns_nop 4", "=s,v", [x], dtype=gl.int32,
                                     is_pure=True, pack=1)


@gluon.jit
def unit_experts(eu_s, U: gl.constexpr):
    """Units 0 .. U - 1's experts (>= 0) as scalars: the whole table read into every thread
    (unit u is then register u: the select-sum folds away), then a v_readfirstlane each."""
    UP: gl.constexpr = eu_s.shape[0]
    C: gl.constexpr = min(UP, 64)
    L: gl.constexpr = gl.BlockedLayout([C], [64], [8], [0])
    i = gl.arange(0, C, layout=L)
    out = ()
    for c0 in gl.static_range(0, U, C):
        v = gl.maximum(eu_s.slice(c0, C).load(L), 0)
        for u in gl.static_range(c0, min(c0 + C, U)):
            out = out + (uniform_p(gl.sum(gl.where(i == u - c0, v, 0), axis=0)),)
    return out


@gluon.jit
def unit_expert(eu_s, u):
    """Unit u's expert as a scalar (-1 past the last unit)."""
    one = gl.full([1], 0, gl.int32, gl.BlockedLayout([1], [64], [8], [0])) + u
    return uniform(gl.sum(eu_s.gather(one, 0), axis=0))


@gluon.jit
def xn_issue(sc, tag, errors, bid, S: gl.constexpr):
    """Every token's MXFP8 xn (FlyDSL's fetch): every router task's flag (one a lane), a
    barrier, then the plain words (16 B loads) and codes of every round, issued only:
    ahead of the first weights they return first (in-order vmcnt)."""
    NS: gl.constexpr = max(ncols(S), 8)  # (a swizzled tile is sliced in whole phases)
    lt = ar(THREADS, T1)
    gate_poll(rgn(sc, R_XNF), lt, lt < 16 * S, tag, errors, bid, E_XN)
    k3 = ar(NS, layout=gl.SliceLayout(1, gl.SliceLayout(2, XQ3)))[:, None, None]
    w3 = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, XQ3)))[None, :, None]
    j3 = ar(32, layout=gl.SliceLayout(0, gl.SliceLayout(1, XQ3)))[None, None, :]
    kc = ar(NS, layout=gl.SliceLayout(1, gl.SliceLayout(2, CQ3)))[:, None, None]
    wc = ar(8, layout=gl.SliceLayout(0, gl.SliceLayout(2, CQ3)))[None, :, None]
    gc = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(1, CQ3)))[None, None, :]
    xnp = i32p(rgn(sc, R_XNP))
    xsp = u8p(rgn(sc, R_XSCP))
    wvs = ()
    cvs = ()
    for i in gl.static_range(6):
        wvs = wvs + (cdna4.buffer_load(xnp, k3 * (HIDDEN // 4) + (6 * w3 + i) * 32 + j3,
                                       mask=(k3 < S) & (j3 >= 0), other=0, cache=".cv"),)
        cvs = cvs + (cdna4.buffer_load(xsp, kc * (HIDDEN // 32) + (6 * wc + i) * 4 + gc, mask=(kc < S) & (gc >= 0),
                                       other=0, cache=".cv"),)
    return wvs, cvs


@gluon.jit
def xn_finish(xv, xra, xrb, cra, crb, S: gl.constexpr):
    """``xn_issue``'s words -> LDS rounds (xra: 0..3, xrb: 4, 5) [8 waves, 16 columns, 128 B]:
    wave w's round-i chunk 6 w + i, column = token; codes -> cra / crb [8, 16, 4]."""
    NS: gl.constexpr = max(ncols(S), 8)
    wvs, cvs = xv
    for i in gl.static_range(6):
        wv = wvs[i]
        b0 = (wv & 255).to(gl.uint8)
        b1 = ((wv >> 8) & 255).to(gl.uint8)
        b2 = ((wv >> 16) & 255).to(gl.uint8)
        b3 = ((wv >> 24) & 255).to(gl.uint8)
        x = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [NS, 8, 128]), (1, 0, 2))
        cq = gl.permute(cvs[i], (1, 0, 2))
        if i < 4:
            xra.index(i).slice(0, NS, dim=1).store(x)
            cra.index(i).slice(0, NS, dim=1).store(cq)
        else:
            xrb.index(i - 4).slice(0, NS, dim=1).store(x)
            crb.index(i - 4).slice(0, NS, dim=1).store(cq)


@gluon.jit
def wug_aoff():
    """A batched MXFP4 A tile [8 waves, 16 rows, 64 B]'s lane byte offsets in aiter's
    shuffled layout (16-row group: 128-k chunk c at 1 KB c, row r 16 B at 16 r + 256 g):
    wave w at chunk 6 w (the round's chunk in the scalar base)."""
    w = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, A3)))[:, None, None]
    r = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, A3)))[None, :, None]
    b = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, A3)))[None, None, :]
    return (6 * w) * 1024 + r * 16 + (b // 16) * 256 + (b % 16)


@gluon.jit
def wug_soff():
    """The scale dword of (wave w's chunk pair, row r, 32-block g) of aiter's shuffle_scale
    (FlyDSL's group_scale_byte / 4): (3 w) 64 + r + 16 g past the scalar base."""
    w = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, SL3)))[:, None, None]
    r = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, SL3)))[None, :, None]
    g = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(1, SL3)))[None, None, :]
    return (3 * w) * 64 + r + 16 * g


@gluon.jit
def wug_loads(eu_s, u, blk, h, valid, w13, s13w, aoff, soff, SCALES: gl.constexpr):
    """Half h (rows 16 h .. of the block) of pair (expert e, 32-row block blk): its gate and
    up A tiles, 6 rounds (wave w: chunk 6 w + round), and (SCALES) the pair's scale dwords
    (3 a tile: chunks 2 j, 2 j + 1 of both halves); masked off past the last pair."""
    base = w13 + (2 * blk + h) * (16 * (HIDDEN // 2))
    ao = aoff + unit_expert_v(eu_s, u, AW_L)[:, None, None] * W13_BYTES
    vm = (aoff >= 0) & valid
    ag = ()
    au = ()
    for i in gl.static_range(6):
        ag = ag + (wload(base + i * 1024, ao, vm, 0),)
        au = au + (wload(base + (INTER * (HIDDEN // 2) + i * 1024), ao, vm, 0),)
    sg = ()
    su = ()
    if SCALES:
        svm = (soff >= 0) & valid
        so = soff + unit_expert_v(eu_s, u, SW_L)[:, None, None] * (48 * 24 * 64)
        sb = blk * (24 * 64)
        for j in gl.static_range(3):
            sg = sg + (cdna4.buffer_load(s13w + (sb + j * 64), so, mask=svm, other=0),)
            su = su + (cdna4.buffer_load(s13w + (sb + (24 * 24 + j) * 64), so, mask=svm, other=0),)
    return ag, au, sg, su


@gluon.jit
def wide_red(redb, w: gl.constexpr):
    """Wave w's partial [16 rows, 16 columns] of an LDS [8 waves, 16 columns, 16 rows] buffer."""
    return redb._reinterpret(gl.float32, [8, 16, 16], FLAT2).index(w).permute((1, 0))


@gluon.jit
def wug_half(ag, au, sg, su, h: gl.constexpr, xra, xrb, cra, crb, redg, redu):
    """A half task's gate / up sums [16 rows, 16 columns] (RL2): each wave's 6 chunks, then
    the waves' partials in wave order from 0 (FlyDSL's row_sum)."""
    accg = zacc([8, 16, 16], MF3)
    accu = zacc([8, 16, 16], MF3)
    for i in gl.static_range(6):
        if i < 4:
            bv = xra.index(i).permute((0, 2, 1)).load(B3)
            bs = cra.index(i).load(SL3)
        else:
            bv = xrb.index(i - 4).permute((0, 2, 1)).load(B3)
            bs = crb.index(i - 4).load(SL3)
        sh = (i % 2) * 16 + h * 8
        sgb = ((sg[i // 2] >> sh) & 255).to(gl.uint8)
        sub = ((su[i // 2] >> sh) & 255).to(gl.uint8)
        b8 = bv.to(gl.float8e4nv, bitcast=True)
        accg = cdna4.mfma_scaled(ag[i], sgb, "e2m1", b8, bs, "e4m3", accg)
        accu = cdna4.mfma_scaled(au[i], sub, "e2m1", b8, bs, "e4m3", accu)
    # [wave, column, row] in LDS: a thread's 8 reads (one output) conflict-free
    redg.permute((0, 2, 1)).store(accg)
    redu.permute((0, 2, 1)).store(accu)
    gl.barrier()
    gs = gl.zeros([16, 16], gl.float32, RL2)
    us = gl.zeros([16, 16], gl.float32, RL2)
    for w in gl.static_range(8):
        gs = gs + wide_red(redg, w).load(RL2)
        us = us + wide_red(redu, w).load(RL2)
    gl.barrier()
    return gs, us


@gluon.jit
def wug_publish(u, blk, y_lo, y_hi, urow_s, sc, tag, S: gl.constexpr):
    """Pair (unit u, block blk)'s mids [16 rows, 16 columns] a half -> bf16 -> MXFP8 per
    (token, 32-row block) (aiter's rule, as the narrow down quantizes them) -> words of
    the routed tokens' rows 5 k + slot (words 8 blk + 4 half + g), and their codes."""
    ZROW: gl.constexpr = mid_rows(S) - 1
    lo = bf16_round(y_lo)
    hi = bf16_round(y_hi)
    amax = gl.maximum(gl.max(gl.abs(lo), axis=0), gl.max(gl.abs(hi), axis=0))
    code = code_ceil(amax * (1.0 / 448.0))
    inv = hw_rcp((code << 23).to(gl.float32, bitcast=True))[None, :]
    row = lds_row(urow_s, u).load(gl.SliceLayout(0, RL2))
    routed = row != ZROW
    put(rgn(sc, R_MIDWSC), row * MID_CODES + blk, code, tag, mask=routed)
    for hv in gl.static_range(2):
        v = lo if hv == 0 else hi
        q = clamp_fp8(v * inv).to(gl.float8e4nv).to(gl.uint8, bitcast=True).to(gl.int32)
        q3 = gl.reshape(q, [4, 4, 16])
        r3 = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(2, q3.type.layout)))[None, :, None]
        word = gl.sum(q3 << (8 * r3), axis=1)  # [4 groups, 16 columns]: rows 4 g .. + 4
        g = ar(4, layout=gl.SliceLayout(1, word.type.layout))[:, None]
        rw = gl.convert_layout(row, gl.SliceLayout(0, word.type.layout))[None, :]
        put(rgn(sc, R_MIDW), rw * MID_WORDS + blk * 8 + hv * 4 + g, word, tag, mask=rw != ZROW)


@gluon.jit
def bytes8(v):
    """int64 [..] -> uint8 [.., 2, 2, 2]: byte 4 i + 2 j + l at [.., i, j, l] (reshape to [.., 8 n])."""
    b = ()
    for j in gl.static_range(8):
        b = b + (((v >> (8 * j)) & 255).to(gl.uint8),)
    x = gl.join(gl.join(b[0], b[4]), gl.join(b[2], b[6]))
    y = gl.join(gl.join(b[1], b[5]), gl.join(b[3], b[7]))
    return gl.join(x, y)


@gluon.jit
def wdn_offs():
    """Down A tile / scale lane offsets of wave w = (tile w / 4, chunk q = w % 4): (A, scale
    half 0, scale half 1). A: aiter's shuffle, 16-row tiles 6 KB apart (a half's chunk 4
    + q on top; q >= 2 there reads the next tile's chunk -- finite, and its B is zero);
    scales: chunk q, and 4 + min(q, 1) (a missing chunk reads chunk 5's)."""
    w = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, A3)))[:, None, None]
    r = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, A3)))[None, :, None]
    b = ar(64, layout=gl.SliceLayout(0, gl.SliceLayout(1, A3)))[None, None, :]
    ao = (w // 4) * (16 * (INTER // 2)) + (w % 4) * 1024 + r * 16 + (b // 16) * 256 + (b % 16)
    ws = ar(8, layout=gl.SliceLayout(1, gl.SliceLayout(2, SL3)))[:, None, None]
    rs = ar(16, layout=gl.SliceLayout(0, gl.SliceLayout(2, SL3)))[None, :, None]
    gs = ar(4, layout=gl.SliceLayout(0, gl.SliceLayout(1, SL3)))[None, None, :]
    q = ws % 4
    rest = rs * 4 + gs * 64 + ws // 4
    q1 = gl.minimum(q, 1)
    # half 1 = chunk 4 + q: waves q >= 2 have none (6 chunks a 32-row block); DN_NOWASTE masks
    # their A loads off (they read the next tile's chunk otherwise: a third more down bytes)
    if DN_NOWASTE == 3:
        # waves q >= 2 re-read wave 1's chunk 5 (a cache hit: no new HBM lines, no mask)
        ao1 = (w // 4) * (16 * (INTER // 2)) + gl.minimum(w % 4, 1) * 1024 + r * 16 + (b // 16) * 256 + (b % 16)
    else:
        ao1 = gl.where((w % 4) < 2, ao, -1)  # (DN_NOWASTE=1: exec-masked loads -- pathological, +28 us/layer)
    return ao, (q // 2) * 256 + (q % 2) * 2 + rest, 512 + q1 * 2 + rest, ao1


@gluon.jit
def wdn_round(i: gl.constexpr, td, eus, nu, w2, s2, offs):
    """Down round i = (unit i / 2, half i % 2): the batched A tile and scales (masked off
    past the last unit)."""
    u: gl.constexpr = i // 2
    h: gl.constexpr = i % 2
    e = eus[u]
    ok = u <= nu
    base = w2 + e.to(gl.int64) * W2_BYTES + td * (32 * (INTER // 2)) + h * 4096
    if DN_NOMASK:
        # no exec-masked loads in the weight stream: a phantom unit (u > nu, its B zero) reads
        # expert 0's block 0 like every other CTA (scalar base select: L2 hits, no new lines)
        base = gl.where(ok, base, w2 + h * 4096)
        sbase = gl.where(ok, s2 + (e * 192 + td) * 768, s2)
        if DN_NOWASTE and (h == 1):
            a = wload(base, offs[3])
        else:
            a = wload(base, offs[0])
        sa = cdna4.buffer_load(sbase, offs[1 + h])
        return a, sa
    if DN_NOWASTE and (h == 1):
        a = wload(base, offs[3], (offs[3] >= 0) & ok, 0)
    else:
        a = wload(base, offs[0], (offs[0] >= 0) & ok, 0)
    sa = cdna4.buffer_load(s2 + (e * 192 + td) * 768, offs[1 + h], mask=(offs[1 + h] >= 0) & ok, other=127)
    return a, sa


@gluon.jit
def wide_down(td, li, n_layers, nu, eu_s, urow_s, uwt_s, w2, s2, sc, peers, peer_local, pbase, rank, ar_out, tag,
              errors, bid, S: gl.constexpr, FFN_OFF: gl.constexpr, tr_=None, eus_pre=0):
    """Down task td (rows 32 td .. + 32) over every unit (FlyDSL's wide_down): wave (tile,
    q) takes chunks q and 4 + q of each, B its columns' mid rows gathered from LDS, route-
    weighted per column, the 4 waves of a tile summed in order -> the FFN all-reduce."""
    U: gl.constexpr = units_max(S)
    NR: gl.constexpr = 2 * U
    MR: gl.constexpr = mid_rows(S)
    DEPTH: gl.constexpr = DN_DEPTH
    NS: gl.constexpr = ncols(S)
    if DN_EUS_EARLY:
        eus = eus_pre
    else:
        eus = unit_experts(eu_s, U)
    stamp(tr_, bid, 58)
    # every (token, slot) mid row -> LDS 128-B lines (line 6 row + chunk: also the global
    # words' order), and their codes; lines past the step's rows zero
    NL: gl.constexpr = mid_lines(S)
    ML2: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [8, 1], [1, 0])
    MC2: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0])
    ml = ar(NL, layout=gl.SliceLayout(1, ML2))[:, None]
    mj = ar(32, layout=gl.SliceLayout(0, ML2))[None, :]
    iw = ml * 32 + mj
    okw = (ml < S * 30) & (mj >= 0)
    cl = ar(NL, layout=gl.SliceLayout(1, MC2))[:, None]
    cg = ar(4, layout=gl.SliceLayout(0, MC2))[None, :]
    ic = cl * 4 + cg
    okc = (cl < S * 30) & (cg >= 0)
    if UG_FLAG:
        # the pairs' flags first (a few lanes): with DN_MIDS_FIRST the mids are then read once,
        # ahead of the weight prefetch (a wave's loads return in order: behind DEPTH rounds,
        # ~100 KB a CTA, the mids land ~3 us later)
        lt_ = ar(THREADS, T1)
        npairs_ = (nu + 1) * UG_PAIRS
        gate_poll(rgn(sc, R_UGF), lt_, lt_ < npairs_, tag, errors, bid, E_MID)
        if units_max(S) * UG_PAIRS > THREADS:
            gate_poll(rgn(sc, R_UGF), lt_ + THREADS, lt_ + THREADS < npairs_, tag, errors, bid, E_MID)
    if DN_MIDS_FIRST:
        pkw = peek(rgn(sc, R_MIDW), iw, tag, okw)
        pkc = peek(rgn(sc, R_MIDWSC), ic, tag, okc)
    offs = wdn_offs()
    aw = ()
    asc = ()
    for i in gl.static_range(DEPTH):
        if i < NR:
            a, sa = wdn_round(i, td, eus, nu, w2, s2, offs)
            aw = aw + (a,)
            asc = asc + (sa,)
    pin(tids())
    stamp(tr_, bid, 59)
    if not DN_MIDS_FIRST:
        pkw = peek(rgn(sc, R_MIDW), iw, tag, okw)
        pkc = peek(rgn(sc, R_MIDWSC), ic, tag, okc)
    stamp(tr_, bid, 22)
    wv = wait_group(((rgn(sc, R_MIDW), iw, pkw),), tag, errors, bid, E_MID, NL * 32 // THREADS)[0]
    cv = wait_group(((rgn(sc, R_MIDWSC), ic, pkc),), tag, errors, bid, E_MID, (NL * 4 + THREADS - 1) // THREADS)[0]
    stamp(tr_, bid, 23)
    w_lo, w_hi = gl.split(gl.reshape(wv, [NL, 16, 2]))
    v64 = w_lo.to(gl.uint32, bitcast=True).to(gl.int64) | (w_hi.to(gl.int64) << 32)
    lines = gl.allocate_shared_memory(gl.int64, [NL, 16], FLAT2, v64)
    lsc = gl.allocate_shared_memory(gl.uint8, [NL, 4], FLAT2, cv.to(gl.uint8))
    gl.barrier()
    stamp(tr_, bid, 9)
    wn = ar(128, layout=gl.SliceLayout(1, G64))
    k8 = ar(16, layout=gl.SliceLayout(0, G64))[None, :]
    ch = (wn // 16) % 4
    wn8 = ar(128, layout=gl.SliceLayout(1, G8))
    g8 = ar(4, layout=gl.SliceLayout(0, G8))[None, :]
    ch8 = (wn8 // 16) % 4
    acc = gl.zeros([8, 16, 16], gl.float32, MF3)
    if DN_LEAN:  # (defined ahead of the loop: a round's tables are reassigned at each unit's first half)
        rw = lds_row(urow_s, 0).gather(wn % 16, 0)
        rw8 = lds_row(urow_s, 0).gather(wn8 % 16, 0)
        wt = lds_row(uwt_s, 0).load(gl.SliceLayout(0, gl.SliceLayout(1, MF3)))[None, None, :]
    for i in gl.static_range(NR):
        pin(tids())
        if i + DEPTH < NR:
            a, sa = wdn_round(i + DEPTH, td, eus, nu, w2, s2, offs)
            aw = aw + (a,)
            asc = asc + (sa,)
        u = i // 2
        h = i % 2
        if DN_LEAN:
            # the unit's tables once (both halves), the MFMA only for real units (the rounds
            # are unrolled to the step's maximum), the route weight fused into the sum
            if (h == 0) and (u > 0):
                rw = lds_row(urow_s, u).gather(wn % 16, 0)
                rw8 = lds_row(urow_s, u).gather(wn8 % 16, 0)
                wt = lds_row(uwt_s, u).load(gl.SliceLayout(0, gl.SliceLayout(1, MF3)))[None, None, :]
            if u <= nu:
                ln = gl.where((rw < S * SLOTS) & (ch + 4 * h < 6), rw * 6 + ch + 4 * h, NL - 1)
                v = lines.gather(ln[:, None] + k8 * 0, 0)
                bv = gl.convert_layout(gl.permute(gl.reshape(bytes8(v), [8, 16, 128]), (0, 2, 1)), B3,
                                       assert_trivial=True)
                ln8 = gl.where((rw8 < S * SLOTS) & (ch8 + 4 * h < 6), rw8 * 6 + ch8 + 4 * h, NL - 1)
                sb = gl.convert_layout(gl.reshape(lsc.gather(ln8[:, None] + g8 * 0, 0), [8, 16, 4]), SL3,
                                       assert_trivial=True)
                res = cdna4.mfma_scaled(aw[i], asc[i], "e2m1", bv.to(gl.float8e4nv, bitcast=True), sb, "e4m3",
                                        zacc([8, 16, 16], MF3))
                acc = gl.fma(res, wt + res * 0.0, acc)
        else:
            rw = lds_row(urow_s, u).gather(wn % 16, 0)
            ln = gl.where((rw < S * SLOTS) & (ch + 4 * h < 6), rw * 6 + ch + 4 * h, NL - 1)
            v = lines.gather(ln[:, None] + k8 * 0, 0)
            bv = gl.convert_layout(gl.permute(gl.reshape(bytes8(v), [8, 16, 128]), (0, 2, 1)), B3, assert_trivial=True)
            rw8 = lds_row(urow_s, u).gather(wn8 % 16, 0)
            ln8 = gl.where((rw8 < S * SLOTS) & (ch8 + 4 * h < 6), rw8 * 6 + ch8 + 4 * h, NL - 1)
            sb = gl.convert_layout(gl.reshape(lsc.gather(ln8[:, None] + g8 * 0, 0), [8, 16, 4]), SL3, assert_trivial=True)
            wt = lds_row(uwt_s, u).load(gl.SliceLayout(0, gl.SliceLayout(1, MF3)))[None, None, :]
            res = cdna4.mfma_scaled(aw[i], asc[i], "e2m1", bv.to(gl.float8e4nv, bitcast=True), sb, "e4m3",
                                    zacc([8, 16, 16], MF3))
            acc = acc + res * wt
    stamp(tr_, bid, 10)
    # [wave, column, row] in LDS; a thread reads its rows 2 j, 2 j + 1 of a column from the
    # tile's 4 waves (8 B a read, all issued together), summed in wave order
    # LDS [wave, row parity, column, row pair]: the epilogue's (lo, hi) [8 row pairs, NS] tiles
    # read in its own layout (PR_L), the tile's 4 waves summed in order
    red = gl.allocate_shared_memory(gl.float32, [8, 2, 16, 8], FLAT4)
    red.store(gl.permute(gl.reshape(acc, [8, 8, 2, 16]), (0, 2, 3, 1)))
    gl.barrier()
    los = ()
    his = ()
    for tt in gl.static_range(2):
        lo = gl.zeros([8, NS], gl.float32, PR_L)
        hi = gl.zeros([8, NS], gl.float32, PR_L)
        for q in gl.static_range(4):
            rw = red._reinterpret(gl.float32, [8, 2, 16, 8], FLAT3).index(4 * tt + q)
            lo = lo + rw._reinterpret(gl.float32, [2, 16, 8], FLAT2).index(0).permute((1, 0)).slice(0, NS, dim=1).load(PR_L)
            hi = hi + rw._reinterpret(gl.float32, [2, 16, 8], FLAT2).index(1).permute((1, 0)).slice(0, NS, dim=1).load(PR_L)
        los = los + (lo,)
        his = his + (hi,)
    ffn_tail(td, li, n_layers, los, his, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, S, FFN_OFF,
             tr_)


@gluon.jit
def stage_moe_wide(li, ptab, n_layers, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, route_scale,
                   shared_weight, swiglu_limit, S: gl.constexpr, FFN_OFF: gl.constexpr, tr_=None):
    UP: gl.constexpr = units_pad(S)
    MAXP: gl.constexpr = wide_pairs(S)
    NH: gl.constexpr = 2 * MAXP
    e = lrow(ptab, li, (12, 13, 14, 15))
    w13 = lp(e[0], gl.uint8)
    s13w = lp(e[1], gl.int32)
    w2 = lp(e[2], gl.uint8)
    s2 = lp(e[3], gl.uint8)
    # every token's routing -> the units and the down tables
    ids, ws = route_wide(sc, bid, tag, errors, route_scale, S, tr_)
    eu_s = gl.allocate_shared_memory(gl.int32, [UP], FLAT1)
    urow_s = gl.allocate_shared_memory(gl.int32, [UP, 16], FLAT2)
    uwt_s = gl.allocate_shared_memory(gl.float32, [UP, 16], FLAT2)
    nu = uniform(wide_tables(ids, ws, shared_weight, eu_s, urow_s, uwt_s, S, tr_, bid))
    gl.barrier()
    stamp(tr_, bid, 6)
    if DN_EUS_EARLY:
        eus_pre = unit_experts(eu_s, units_max(S))  # (SGPRs through the up / gate loop)
    else:
        eus_pre = 0
    n_pairs = (nu + 1) * UG_PAIRS
    aoff = wug_aoff()
    soff = wug_soff()
    # every token's xn (its flags set while the tables were built), then pair j = CTA bid +
    # j 256's half 0 weights
    xv = xn_issue(sc, tag, errors, bid, S)
    stamp(tr_, bid, 62)
    pr0 = gl.minimum(bid, n_pairs - 1)
    cur = wug_loads(eu_s, pr0 // UG_PAIRS, pr0 % UG_PAIRS, 0, bid < n_pairs, w13, s13w, aoff, soff, True)
    pin(tids())
    xa = gl.allocate_shared_memory(gl.uint8, [4, 8, 16, 128], XSW4)
    xb = gl.allocate_shared_memory(gl.uint8, [2, 8, 16, 128], XSW4)
    ca = gl.allocate_shared_memory(gl.uint8, [4, 8, 16, 4], FLAT4)
    cb = gl.allocate_shared_memory(gl.uint8, [2, 8, 16, 4], FLAT4)
    xra = xa._reinterpret(gl.uint8, [4, 8, 16, 128], XSW3)
    xrb = xb._reinterpret(gl.uint8, [2, 8, 16, 128], XSW3)
    cra = ca._reinterpret(gl.uint8, [4, 8, 16, 4], FLAT3)
    crb = cb._reinterpret(gl.uint8, [2, 8, 16, 4], FLAT3)
    stamp(tr_, bid, 60)
    xn_finish(xv, xra, xrb, cra, crb, S)
    stamp(tr_, bid, 61)
    gl.barrier()
    stamp(tr_, bid, 7)
    redg = gl.allocate_shared_memory(gl.float32, [8, 16, 16], FLAT3)
    redu = gl.allocate_shared_memory(gl.float32, [8, 16, 16], FLAT3)
    sg = cur[2]
    su = cur[3]
    y_lo = gl.zeros([16, 16], gl.float32, RL2)
    for it in gl.static_range(NH):
        pr = bid + (it // 2) * BLOCKS
        if it + 1 < NH:
            prn = bid + ((it + 1) // 2) * BLOCKS
            prc = gl.minimum(prn, n_pairs - 1)
            nxt = wug_loads(eu_s, prc // UG_PAIRS, prc % UG_PAIRS, (it + 1) % 2, prn < n_pairs, w13, s13w, aoff, soff,
                            (it + 1) % 2 == 0)
            pin(tids())  # the next half's loads stay issued here (LLVM sinks them to their use)
        if it % 2 == 0:
            sg = cur[2]
            su = cur[3]
        if pr < n_pairs:
            gs, us = wug_half(cur[0], cur[1], sg, su, it % 2, xra, xrb, cra, crb, redg, redu)
            if it < 8:
                stamp(tr_, bid, 49 + it)
            y = swiglu(gs, us, swiglu_limit)
            if it % 2 == 0:
                y_lo = y
            else:
                wug_publish(pr // UG_PAIRS, pr % UG_PAIRS, y_lo, y, urow_s, sc, tag, S)
                if UG_FLAG:
                    # the pair's flag behind a barrier (every wave's mid puts issued; FlyDSL's
                    # ugdone, no vmcnt drain): the down CTAs gate on these few words, then
                    # read the mids once instead of re-polling the whole tagged table
                    lt_ = ar(THREADS, T1)
                    gl.barrier()
                    put(rgn(sc, R_UGF), pr + lt_ * 0, lt_ * 0 + 1, tag, mask=lt_ == 0)
                stamp(tr_, bid, 57)
        if it + 1 < NH:
            cur = nxt
    stamp(tr_, bid, 8)
    if bid < N_DN:
        wide_down(bid, li, n_layers, nu, eu_s, urow_s, uwt_s, w2, s2, sc, peers, peer_local, pbase, rank, ar_out, tag,
                  errors, bid, S, FFN_OFF, tr_, eus_pre)
        stamp(tr_, bid, 11)


# ============================================================ the launch
@gluon.jit
def peer_regions(peers, peer_local, pbase, FFN_OFF: gl.constexpr):
    """The 4 ranks' peer regions of this parity (attn, ffn) and this rank's own: the peer
    bases by scalar loads in one asm (one scalar round trip; a vector load + readfirstlane
    each is a vector one, in order behind every load in flight). The outputs are early-clobber:
    an output allocated over the table pointer $4 is written whenever its load returns, and a
    later load of the four then reads its address from that peer base (the peer's tagged pairs
    became rank addresses: GPU memory faults at "tag:value" addresses)."""
    pb = gl.inline_asm_elementwise(
        "s_load_dwordx2 $0, $4, 0\ns_load_dwordx2 $1, $4, 8\ns_load_dwordx2 $2, $4, 16\n"
        "s_load_dwordx2 $3, $4, 24\ns_waitcnt lgkmcnt(0)", "=&s,=&s,=&s,=&s,s", [peers.to(gl.int64, bitcast=True)],
        dtype=(gl.int64, gl.int64, gl.int64, gl.int64), is_pure=False, pack=1)
    attn_b = ()
    ffn_b = ()
    for r in gl.static_range(4):
        b = pb[r] + pbase
        attn_b = attn_b + (b.to(gl.pointer_type(gl.int64), bitcast=True),)
        ffn_b = ffn_b + ((b + FFN_OFF).to(gl.pointer_type(gl.int64), bitcast=True),)
    own = peer_local + pbase
    return attn_b, ffn_b, own.to(gl.pointer_type(gl.int64), bitcast=True), \
        (own + FFN_OFF).to(gl.pointer_type(gl.int64), bitcast=True)


@gluon.jit
def dshift(bases, OFF: gl.constexpr):
    """A parity's peer bases (``*i64``, offset 0) -> their dense region at OFF (``*i32``)."""
    out = ()
    for i in gl.static_range(4):
        out = out + (dptr(bases[i], OFF),)
    return out


@gluon.jit
def dptr(base, OFF: gl.constexpr):
    return (base.to(gl.int64, bitcast=True) + OFF).to(gl.pointer_type(gl.int32), bitcast=True)


@gluon.jit
def dpick(bases, r):
    """bases[r] (``*i32``) for a runtime (CTA-uniform) rank r."""
    b = bases[0].to(gl.int64, bitcast=True)
    for i in gl.static_range(1, 4):
        b = gl.where(r == i, bases[i].to(gl.int64, bitcast=True), b)
    return b.to(gl.pointer_type(gl.int32), bitcast=True)


@gluon.jit
def peer_pick(bases, r):
    """bases[r] for a runtime (CTA-uniform) rank r."""
    b = bases[0].to(gl.int64, bitcast=True)
    for i in gl.static_range(1, 4):
        b = gl.where(r == i, bases[i].to(gl.int64, bitcast=True), b)
    return b.to(gl.pointer_type(gl.int64), bitcast=True)


@gluon.jit
def peer_shift(bases, OFF: gl.constexpr):
    out = ()
    for i in gl.static_range(4):
        out = out + ((bases[i].to(gl.int64, bitcast=True) + OFF).to(gl.pointer_type(gl.int64), bitcast=True),)
    return out


@gluon.jit
def layer_body(li, ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in, res_in,
               ar_out, hm_out, scratch, peer_local, peers, errors, trace, rank, sm_scale, eps, route_scale,
               shared_weight, swiglu_limit, ep, bid, tls, moe_x, moe_ids, moe_w, moe_out,
               S: gl.constexpr, HALF: gl.constexpr, PHALF: gl.constexpr, FFN_OFF: gl.constexpr, STAGES: gl.constexpr,
               QL: gl.constexpr = 1, qkv_out=0, attn_in=0, PHASE: gl.constexpr = 0):
    """Sparse layer li of the step (tag ep * TAGS + li + 1, scratch / peer half li % 2)."""
    tag = ep * TAGS + li + 1
    ptag = tag - 1
    par = li % 2
    sc = scratch + par * HALF
    psc = scratch + (1 - par) * HALF
    pbase = (par * PHALF).to(gl.int64)
    tr_ = trace + li * (BLOCKS * TRACE_SLOTS)
    if TRACE == 3:
        if bid >= 0:  # (diagnostics) where CTA bid runs this layer: HW_ID, XCC_ID -> past the stamps
            t1_ = ar(THREADS, T1)
            hw_ = gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s,v", [t1_], dtype=gl.int32,
                                            is_pure=False, pack=1)
            xc_ = gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s,v", [t1_],
                                            dtype=gl.int32, is_pure=False, pack=1)
            hwp = trace + n_layers * (BLOCKS * TRACE_SLOTS) + (li * BLOCKS + bid) * 2
            gl.store(hwp + t1_ * 0, hw_.to(gl.int64), mask=t1_ == 0)
            gl.store(hwp + 1 + t1_ * 0, xc_.to(gl.int64), mask=t1_ == 0)
        tr_ = (tr_, tls)
        stamp_init(tr_)
    stamp(tr_, bid, 0)
    arsrc = rgn(psc, R_AR)
    PA: gl.constexpr = PHASE != 2  # split hybrid: launch A = the FFN all-reduce .. qkv
    PATT: gl.constexpr = PHASE == 0  # the in-kernel attention (else vLLM's, between the launches)
    PB: gl.constexpr = PHASE != 1  # launch B = the o input .. top-k
    if HYB:
        ppb = ((1 - par) * PHALF).to(gl.int64)
        if PA and (STAGES & 1024) == 0 and (li > 0) & (bid < N_DN):  # (bit 1024: ablation, no all-reduce)
            stage_far(bid, li - 1, n_layers, moe_out, psc, peers, peer_local, ppb, rank, ar_out, ptag, errors, bid,
                      S, FFN_OFF, tr_, lrow(ptab, li - 1, (19, 0))[0].to(gl.int32) >> 8)
        if S >= AG_RS_FROM:
            # the owners' all-gathered sums, read where they land
            _, _, _, ffn_own_p = peer_regions(peers, peer_local, ppb, FFN_OFF)
            arsrc = (ffn_own_p.to(gl.int64, bitcast=True) + (FAG_OFF - FFN_OFF)).to(gl.pointer_type(gl.int64),
                                                                                    bitcast=True)
        stamp(tr_, bid, 24)
    topk = lrow(ptab, li, (19, 0))[0].to(gl.int32) & 255  # (bits 8 ..: the EAGLE3 aux slot)
    nq = gl.where(topk != 0, ROWS16, NQ_REUSE)
    nt_ = bid - norm_base(S)
    # (bit 2048: ablation, the input norm tasks alone)
    if not PA:
        pass
    elif NORM_WAVE and (S > 16):
        if ((STAGES & 2049) != 0) and (nt_ >= 0) & (nt_ < (S + 7) // 8):
            stage_norm8(nt_, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, arsrc, S, tr_)
    elif NORM_P3 and HYB and (S > 16):
        if ((STAGES & 2049) != 0) and (nt_ >= 0) & (nt_ < S):
            if DENSE_FFN:
                attn_pp, _, attn_op, _ = peer_regions(peers, peer_local, ((1 - par) * PHALF).to(gl.int64), FFN_OFF)
                stage_norm_p3(nt_, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps,
                              dptr(attn_op, FFND_OFF if DENSE_1S else FAGD_OFF), tr_, ar_out, True)
            else:
                stage_norm_p3(nt_, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, arsrc, tr_)
    elif ((STAGES & 2049) != 0) and (NORM_TASKS and S > 2) and (nt_ >= 0) & (nt_ < S):
        stage_norm(nt_, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, arsrc, tr_)
    if PA and ((STAGES & (1 | AB_QKV)) != 0) and bid < nq:
        stage_qkv(bid, li, ptab, hid_in, res_in, sc, psc, tag, ptag, errors, bid, eps, S, tr_, arsrc, qkv_out, PHASE)
    stamp(tr_, bid, 1)
    if not PATT:
        pass
    elif S <= 8:
        if (STAGES & 2) and (bid == ROWS16) & (topk != 0):
            stage_ik(li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps, S)
    else:
        # a task a token (CTAs 16 ..: after their qkv tasks): one CTA looping every token
        # would hold up its split and merge tasks
        ik_ = bid - 16
        if (STAGES & 2) and (ik_ >= 0) & (ik_ < S) & (topk != 0):
            stage_ik(li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps, S, ik_, 1)
    if S <= 8:
        ok_ = bid - (ROWS16 + 1)
    else:
        ok_ = bid  # past 8 rows the split takes CTAs 128 ..: the own keys after CTAs 0 ..'s qkv tasks
    if PATT and ((STAGES & (4 | AB_OWN)) != 0) and (ok_ >= 0) & (ok_ < S):
        stage_own(ok_, li, ptab, cos_sin, positions, slots, sc, tag, errors, bid, eps, tr_, own_split(S))
    if own_split(S) and PATT and ((STAGES & (4 | AB_OWN)) != 0) and (nt_ >= 0) & (nt_ < S):
        stage_ownv(nt_, li, ptab, slots, sc, tag, errors, bid, tr_)
    if PATT and (S <= LONG_S) and ((STAGES & (4 | AB_SPLIT)) != 0) and (topk != 0):
        stage_score(li, ptab, cos_sin, positions, seq_lens, block_table, bt_stride, sc, tag, errors, bid, eps,
                    sm_scale, S, QL, tr_)
    st = BLOCKS - 1 - bid
    if (S <= 4) or (MERGE_ON_SPLIT == 0):
        mt = bid - merge_base(S)
    else:
        # past 4 rows every non-split CTA has an o task: token k's merge runs on its split
        # partition 7's CTA (248 - 8 k), after that split (past the context, or the last)
        mt = gl.where((248 - bid) % 8 == 0, (248 - bid) // 8, -1)
    # past 32 rows (MERGE_SPREAD): token k's merge right after its partition-7 split, before
    # this CTA's wrapped split (token k + 32's), and token k + 32's merge on a one-split CTA
    # (past 48 rows, MERGE_SPREAD 2: up to 60 rows, where the one-split CTAs run out)
    SPREAD: gl.constexpr = MERGE_SPREAD and (S > BLOCKS // N_SPLIT) and (S <= (48 if MERGE_SPREAD == 1 else 60))
    if PATT and ((STAGES & (4 | AB_SPLIT)) != 0) and st < N_SPLIT * S:
        stage_split(st, li, ptab, cos_sin, positions, slots, block_table, bt_stride, seq_lens, sc, tag, errors,
                    bid, sm_scale, eps, tr_, QL, topk, scratch, S)
    if SPREAD and PATT:
        if ((STAGES & (8 | AB_MERGE)) != 0) and (mt >= 0) & (mt < S):
            stage_merge(mt, sc, tag, errors, bid, tr_)
    if PATT and (S > BLOCKS // N_SPLIT):
        # past 32 rows the split tasks wrap: task st + 256 on the same CTA
        if ((STAGES & (4 | AB_SPLIT)) != 0) and st + BLOCKS < N_SPLIT * S:
            stage_split(st + BLOCKS, li, ptab, cos_sin, positions, slots, block_table, bt_stride, seq_lens, sc, tag,
                        errors, bid, sm_scale, eps, tr_, QL, topk, scratch, S)
    stamp(tr_, bid, 2)
    if not PATT:
        if PHASE == 2:
            # vLLM's attention output -> the o input: token k on CTA 255 - k (router CTAs, past
            # the o tasks, so no o task waits behind one)
            ko = BLOCKS - 1 - bid
            if ((STAGES & (8 | AB_MERGE)) != 0) and ko < S:
                stage_oin(ko, attn_in, sc, tag)
    elif SPREAD:
        # tokens 32 ..: CTA 124 - 8 j (one split only: tokens 16 .. 31's partitions)
        if S <= 48:
            m2 = gl.where((bid <= 124) & ((124 - bid) % 8 == 0), 32 + (124 - bid) // 8, -1)
        else:
            # the CTAs left with one split (bid < NF), but not 248 - 8 k (token k's merge): 7 NF / 8
            NF: gl.constexpr = 2 * BLOCKS - N_SPLIT * S
            MST: gl.constexpr = max(1, (7 * NF // 8) // (S - BLOCKS // N_SPLIT))
            q_ = NF - 1 - bid
            i_ = q_ - (q_ + 1) // 8
            m2 = gl.where((bid < NF) & (q_ % 8 != 7) & (i_ % MST == 0), 32 + i_ // MST, -1)
        if ((STAGES & (8 | AB_MERGE)) != 0) and (m2 >= 32) & (m2 < S):
            stage_merge(m2, sc, tag, errors, bid, tr_)
    else:
        if ((STAGES & (8 | AB_MERGE)) != 0) and (MERGE_TASKS and S > 1) and (mt >= 0) & (mt < S):
            stage_merge(mt, sc, tag, errors, bid, tr_)
        if S > BLOCKS // N_SPLIT:
            # token k + 32's partition 7 wrapped onto token k's CTA
            if ((STAGES & (8 | AB_MERGE)) != 0) and (mt >= 0) & (mt + BLOCKS // N_SPLIT < S):
                stage_merge(mt + BLOCKS // N_SPLIT, sc, tag, errors, bid, tr_)
    # AG_RS up to 16 rows: the router tasks read the owners' gathered sums where they land
    # (FlyDSL's), no o task waits for them (the "a" re-put was a local hop on the critical path)
    RAAG: gl.constexpr = ROUTER_AAG and (S >= AG_RS_FROM) and (S <= 16) and ((STAGES & AB_ONOAR) == 0) and (not HYB)
    if PB and ((STAGES & (8 | AB_O | AB_ONOAR)) != 0) and bid < N_O:
        stage_o(bid, li, ptab, n_layers, sc, peers, peer_local, pbase, rank, hm_out, tag, errors, bid, S, FFN_OFF, tr_,
                (STAGES & AB_ONOAR) != 0, RAAG)
    stamp(tr_, bid, 3)
    tr = (bid + BLOCKS - ROUTER_BASE) % BLOCKS
    if RAAG:
        _, _, attn_own_r, _ = peer_regions(peers, peer_local, pbase, FFN_OFF)
        asrc = (attn_own_r.to(gl.int64, bitcast=True) + AAG_OFF).to(gl.pointer_type(gl.int64), bitcast=True)
    else:
        asrc = rgn(sc, R_A)
    # (bit 4096 without 16: ablation, the router tasks' post-attention norm / xn alone)
    if PB and ((STAGES & (4112 | AB_ROUTER)) != 0) and tr < 16 * ((S + (S + 15) // 16 - 1) // ((S + 15) // 16)):
        stage_router(tr, li, ptab, sc, tag, errors, bid, eps, S, tr_, moe_x, (STAGES & (16 | AB_ROUTER)) == 0, asrc,
                     RAAG, n_layers, hm_out)
    if HYB:
        if PB and ((STAGES & (16 | AB_TOPK)) != 0) and (tr % 32 == 0) & (tr // 32 < (S + 7) // 8):
            route_out(tr // 32, sc, bid, tag, errors, route_scale, shared_weight, moe_ids, moe_w, S)
    stamp(tr_, bid, 4)
    if (STAGES & 32) and not HYB:
        if S >= WIDE_FROM:
            stage_moe_wide(li, ptab, n_layers, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid,
                           route_scale, shared_weight, swiglu_limit, S, FFN_OFF, tr_)
        else:
            stage_moe(li, ptab, n_layers, sc, peers, peer_local, pbase, rank, ar_out, tag, errors, bid, route_scale,
                      shared_weight, swiglu_limit, S, FFN_OFF, STAGES, tr_)
    stamp(tr_, bid, 5)
    stamp_flush(tr_, bid)


# Every scalar and per-step pointer argument unspecialized: Triton would otherwise key the
# binary on li0 == 1, multiples of 16 and pointer alignment -- a recompile (10-20 s, per
# rank) inside a step's first launches, the other ranks spinning on its mailboxes meanwhile
@gluon.jit(do_not_specialize=["n_layers", "bt_stride", "rank", "li0"],
           do_not_specialize_on_alignment=["positions", "slots", "block_table", "seq_lens", "hid_in", "res_in",
                                           "ar_out", "hm_out", "cos_sin", "peer_local", "trace", "qkv_out",
                                           "attn_in"])
def m3_step_kernel(ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in, res_in,
                   ar_out, hm_out, scratch, peer_local, peers, epoch, errors, trace, rank, sm_scale, eps,
                   route_scale, shared_weight, swiglu_limit, li0, moe_x, moe_ids, moe_w, moe_out, qkv_out, attn_in,
                   S: gl.constexpr, HALF: gl.constexpr, PHALF: gl.constexpr, FFN_OFF: gl.constexpr,
                   STAGES: gl.constexpr = 511, LOOP: gl.constexpr = False, QL: gl.constexpr = 1,
                   PHASE: gl.constexpr = 0, NL: gl.constexpr = 1):
    """LOOP: every layer in this launch; else layer li0 only (a launch a layer, the step's
    last moving the epoch on)."""
    bid = gl.program_id(0)
    tid = tids()
    cos_sin = i32p(cos_sin)  # [positions, 64] bf16 as pairs
    if EP_SLOAD and (not LOOP):
        lrow0 = gl.minimum(li0, n_layers - 1)
        ep = gl.inline_asm_elementwise(_EP_ASM, "=&s,=&s,=&s,=&s,=&s,s,s",
                                       [epoch.to(gl.int64, bitcast=True),
                                        (ptab + lrow0 * NPTR).to(gl.int64, bitcast=True)],
                                       dtype=(gl.int32, gl.int64, gl.int64, gl.int64, gl.int64), is_pure=False,
                                       pack=1)[0]
    else:
        ep = uniform(gl.load(epoch).to(gl.int32))  # (i32: a 64-bit readfirstlane does not assemble)
    marks = epoch.to(gl.pointer_type(gl.int64), bitcast=True) + EP_MARK_PAIRS
    tls = 0
    if TRACE == 3:
        tls = gl.allocate_shared_memory(gl.int64, [TRACE_SLOTS], FLAT1)
    if LOOP:
        put(marks, bid + tid * 0, tid * 0 + 1, ep + 1, mask=tid == 0)
        for li in range(n_layers):
            layer_body(li, ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in,
                       res_in, ar_out, hm_out, scratch, peer_local, peers, errors, trace, rank, sm_scale, eps,
                       route_scale, shared_weight, swiglu_limit, ep, bid, tls, moe_x, moe_ids, moe_w, moe_out, S,
                       HALF, PHALF, FFN_OFF, STAGES, QL)
        last = bid >= 0
    elif HYB:
        if li0 < n_layers:
            layer_body(li0, ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens, hid_in,
                       res_in, ar_out, hm_out, scratch, peer_local, peers, errors, trace, rank, sm_scale, eps,
                       route_scale, shared_weight, swiglu_limit, ep, bid, tls, moe_x, moe_ids, moe_w, moe_out, S,
                       HALF, PHALF, FFN_OFF, STAGES, QL, qkv_out, attn_in, PHASE)
        else:
            # the last layer's FFN all-reduce (into ar_out) alone
            lp_ = li0 - 1
            if bid < N_DN:
                stage_far(bid, lp_, n_layers, moe_out, scratch + (lp_ % 2) * HALF, peers, peer_local,
                          ((lp_ % 2) * PHALF).to(gl.int64), rank, ar_out, ep * TAGS + lp_ + 1, errors, bid, S,
                          FFN_OFF, None, lrow(ptab, lp_, (19, 0))[0].to(gl.int32) >> 8)
        last = li0 == n_layers
        if last:
            put(marks, bid + tid * 0, tid * 0 + 1, ep + 1, mask=tid == 0)
    else:
        for j in gl.static_range(NL):
            if (NL == 1) or (li0 + j < n_layers):
                layer_body(li0 + j, ptab, n_layers, cos_sin, positions, slots, block_table, bt_stride, seq_lens,
                           hid_in, res_in, ar_out, hm_out, scratch, peer_local, peers, errors, trace, rank, sm_scale,
                           eps, route_scale, shared_weight, swiglu_limit, ep, bid, tls, moe_x, moe_ids, moe_w,
                           moe_out, S, HALF, PHALF, FFN_OFF, STAGES, QL)
        last = li0 + NL >= n_layers
        if last:
            put(marks, bid + tid * 0, tid * 0 + 1, ep + 1, mask=tid == 0)
    # every CTA read this step's epoch: CTA 0 moves it on
    if last & (bid == 0):
        _, err = pollv1(marks, tid, ep + 1, tid < BLOCKS)
        note_err(errors, bid, err, E_EPOCH)
        gl.barrier()
        gl.store(epoch + tid * 0, ep + 1 + tid * 0, mask=tid == 0)
