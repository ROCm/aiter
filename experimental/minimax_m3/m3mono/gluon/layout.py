# SPDX-License-Identifier: MIT
"""Host-side shapes and memory layout of the Gluon MiniMax-M3 step kernel.

One launch runs every sparse layer (3..59) of a decode step: layer i's hand-offs live
in the half of the scratch / peer buffers of parity i % 2 (every stage of layer i + 2
waits on all of layer i + 1, which waits on all of layer i: no CTA reaches a half
while a peer still reads the previous layer's pairs there). A tagged pair is 8 bytes
(tag << 32 | value); bf16 data travel two to a pair. Regions are 256-byte aligned.
"""

import os
from dataclasses import dataclass

TP = 4
HIDDEN = 6144
HEAD_DIM = 128
HEADS = 16  # q heads a rank
QKV_ROWS = (HEADS + 4) * HEAD_DIM  # q | k | v | index q | index k: 2560
ROWS16 = QKV_ROWS // 16  # 160 GEMV tasks
NQ_REUSE = (HEADS + 2) * HEAD_DIM // 16  # 144: a layer reusing a selection skips the index rows
O_K = HEADS * HEAD_DIM  # 2048
N_ROUTED, TOP_K, SLOTS = 128, 4, 5
INTER = 768  # expert intermediate a rank
SPLIT_KEYS = 256
N_SPLIT = 8  # partitions of a <= 2048-key context
MAX_SHORT = N_SPLIT * SPLIT_KEYS  # contexts up to this attend every key (no indexer)
# the indexer (contexts past MAX_SHORT, rows <= LONG_S): FlyDSL's selection, 128-key blocks
SPARSE_BLOCK = 128
TOPK_BLOCKS = 16  # selected blocks (= 2 a split partition)
LONG_FROM_BLOCKS = 512  # past this a request's blocks are ranked in N_SPLIT shares
MAX_INDEX_BLOCKS = 8192  # 1M context
LONG_S = 16  # rows a long-context step may have (the score region's rows)
SHARE_BATCH = 2  # blocks a thread keys in a long share (N_SPLIT * THREADS * 2 >= MAX_INDEX_BLOCKS)
STAB_WORDS = 32  # a token's sparse table: 16 block ids, the sparse context length (padded)
O_ROWS = 32
N_O = HIDDEN // O_ROWS  # 192
N_ROUTER = 16  # router tasks a token (8 experts each, one a wave)
XN_SLICE = HIDDEN // N_ROUTER  # 384
UG_ROWS = 16  # gate rows (and as many up rows) a routed up / gate task
UG_TASKS = INTER // UG_ROWS  # 48 a slot
DN_ROWS = 32
N_DN = HIDDEN // DN_ROWS  # 192
SCORE_COPIES = 8
BLOCKS, THREADS = 256, 512
PAIR = 8
MAX_S = 64  # step rows the scratch / peer buffers hold (the row stride); the narrow MoE serves S <= 4
WIDE_FROM = 5  # rows from here run the expert-grouped (wide) MoE
SUPPORTED_S = 64  # rows the kernel builds for (past FUSED_MAX_S: hybrid only)
FUSED_MAX_S = 16  # rows the in-kernel (wide) MoE serves
AG_RS_FROM = 8  # rows from here reduce-scatter + all-gather the all-reduces (FlyDSL's: half the link bytes)

# the epoch buffer, int32 words: the epoch, then a tagged pair a CTA (it read it)
EP_EPOCH = 0
EP_MARKS = 64
EPOCH_WORDS = EP_MARKS + 2 * BLOCKS
TAGS_PER_STEP = 64  # tag = epoch * 64 + layer + 1


def cdiv(a, b):
    return -(-a // b)


def padded(n, a):
    return cdiv(n, a) * a


def _lay(items, align=256):
    out, off = {}, 0
    for name, n in items:
        off = padded(off, align)
        out[name] = (off, n)
        off += n
    return out, padded(off, align)


@dataclass(frozen=True)
class StepLayout:
    """Scratch of one parity (``half`` bytes; parity p at p * half) for MAX_S rows."""

    def regions(self) -> dict:
        s = MAX_S
        items = [
            ("qkv", s * QKV_ROWS // 2 * PAIR),  # GEMV rows, bf16 pairs
            ("spo", s * N_SPLIT * O_K // 2 * PAIR),  # split outputs, bf16 pairs
            ("spm", s * N_SPLIT * HEADS * PAIR),
            ("spl", s * N_SPLIT * HEADS * PAIR),
            ("a", s * HIDDEN // 2 * PAIR),  # attention all-reduce sum, bf16 pairs
            ("h", s * HIDDEN // 2 * PAIR),  # the layer's residual in, bf16(ar + res)
            ("hm", s * HIDDEN // 2 * PAIR),  # h_mid = bf16(bf16(a) + h): next layer's residual
            ("sc", SCORE_COPIES * s * N_ROUTED * 2 * PAIR),  # (routing key, sigmoid)
            ("xn8", s * HIDDEN // 4 * PAIR),  # MXFP8 xn, 4 a pair
            ("xsc", s * HIDDEN // 32 * PAIR),  # its E8M0 codes, one a pair
            ("mid", s * SLOTS * INTER // 2 * PAIR),  # swiglu outputs, bf16 pairs
            ("ar", s * HIDDEN // 2 * PAIR),  # FFN all-reduce: next layer's hidden in
            ("own", s * 2 * 64 * PAIR),  # the own-key task's fp8 K then V (2 bytes a pair)
            ("oq", s * O_K // 4 * PAIR),  # merge tasks (S > 1): a token's merged FP8 o input, 4 a pair
            ("oqs", s * PAIR),  # its per-token scale
            ("x8", s * HIDDEN // 4 * PAIR),  # norm tasks (S > 2): a token's FP8 qkv input, 4 a pair
            ("x8s", s * PAIR),  # its per-token scale
            # qkv K-split (S > 16): tile t's K-slice c partial [tile][slice][token][16 rows], f32 tagged
            ("qkvp", ROWS16 * 8 * s * 16 * PAIR),
            # wide MoE: the MXFP8 mid rows (token k, slot) = row 5 k + slot, 4 fp8 a pair, and
            # their E8M0 codes (one a pair, 24 a row)
            # and its xn plain (the router's words then a flag a router task; 16 B loads)
            ("xnp", s * HIDDEN),  # MXFP8 xn bytes, row k
            ("xscp", s * HIDDEN // 32),  # its E8M0 codes, a byte each
            ("xnf", s * 16 * PAIR),  # router task (k, tq)'s flag
            ("ugf", (min(128, 4 * s) + 1) * (INTER // 32) * PAIR),  # wide up / gate pair's mids-out flag
            ("midw", s * SLOTS * INTER // 4 * PAIR),
            ("midwsc", s * SLOTS * (INTER // 32) * PAIR),
            # the indexer (long contexts): token k's block b score at pair k * MAX_INDEX_BLOCKS + b
            ("isc", LONG_S * MAX_INDEX_BLOCKS * PAIR),
            # a long request's share candidates: (token k, share j, rank r) -> pairs 2 i, 2 i + 1
            # (key, block id), i = (k * N_SPLIT + j) * TOPK_BLOCKS + r
            ("cand", LONG_S * N_SPLIT * TOPK_BLOCKS * 2 * PAIR),
            ("ikf", s * PAIR),  # token k's index key in the cache (stage_ik's flag)
        ]
        return _lay(items)[0]

    def half(self) -> int:
        return _lay(list((k, v[1]) for k, v in self.regions().items()))[1]

    def stab(self) -> int:
        """Byte offset of the step-global sparse tables (past both halves): token k's at
        word k * STAB_WORDS, written by a selecting layer, read by the reusing ones (later
        launches)."""
        return 2 * self.half()

    def nbytes(self) -> int:
        return 2 * self.half() + padded(LONG_S * STAB_WORDS * 4, 256)


def region(name: str) -> int:
    return StepLayout().regions()[name][0]


def peer_regions() -> dict:
    """One parity's peer regions: attn / ffn [src rank][MAX_S][HIDDEN / 2] pairs each, then
    (AG_RS_FROM rows on) the owners' all-gathered sums aag / fag [MAX_S][HIDDEN / 2] pairs."""
    n = TP * MAX_S * HIDDEN // 2 * PAIR
    m = MAX_S * HIDDEN // 2 * PAIR
    # dense (DENSE_AR: plain bf16 pairs, 4 bytes, DSENT until written) twins of the four
    d, e = n // 2, m // 2
    b = 2 * n + 2 * m
    return {"attn": 0, "ffn": n, "aag": 2 * n, "fag": 2 * n + m,
            "attnd": b, "ffnd": b + d, "aagd": b + 2 * d, "fagd": b + 2 * d + e, "_half": b + 2 * d + 2 * e}


def dense_spans() -> list:
    """(byte offset, bytes) of one parity's dense regions (filled with 0xFF)."""
    pr = peer_regions()
    return [(pr["attnd"], pr["_half"] - pr["attnd"])]


def peer_half() -> int:
    return peer_regions()["_half"]


# CTA map (S <= 4): task t of a stage on CTA (base + t) % BLOCKS
def split_cta(task: int) -> int:
    """Split task (token k, partition p) = k * N_SPLIT + p on the top CTAs, downward."""
    return BLOCKS - 1 - task


ROUTER_BASE = N_O  # the router tasks on the CTAs past the o tasks (192 ..)


def norm_base(S: int) -> int:
    """S > 2: token k's norm task on CTA norm_base + k (past the own-key tasks 161 ..);
    past 16 rows (hybrid) on the CTAs past the FFN all-reduce tasks (192 ..)."""
    return N_DN if S > 16 else ROWS16 + 1 + max(S, 4)


def merge_base(S: int) -> int:
    """S > 1: token k's merge task on CTA merge_base + k: a router CTA, idle in the split,
    up to 4 rows (past them the kernel puts it on token k's partition-7 split CTA, 248 - 8 k)."""
    return N_O
SHARED_BASE = int(os.environ.get("VLLM_M3_GLUON_SHARED_BASE", str(N_O)))  # the shared expert's up / gate tasks
# there too, after the router (192); 128: off the router CTAs (on o / down ones)
