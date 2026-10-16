# SPDX-License-Identifier: MIT
"""Host-side shapes and memory layout of the Gluon MiniMax-M3 step kernel.

One launch runs every sparse layer (3..59) of a decode step: layer i's hand-offs live
in the half of the scratch / peer buffers of parity i % 2 (every stage of layer i + 2
waits on all of layer i + 1, which waits on all of layer i: no CTA reaches a half
while a peer still reads the previous layer's pairs there). A tagged pair is 8 bytes
(tag << 32 | value); bf16 data travel two to a pair. Regions are 256-byte aligned.
"""

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
MAX_SHORT = N_SPLIT * SPLIT_KEYS  # contexts past this need the indexer (not in this build)
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
MAX_S = 4  # step rows this build serves (the narrow MoE)

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
        ]
        return _lay(items)[0]

    def half(self) -> int:
        return _lay(list((k, v[1]) for k, v in self.regions().items()))[1]

    def nbytes(self) -> int:
        return 2 * self.half()


def region(name: str) -> int:
    return StepLayout().regions()[name][0]


def peer_regions() -> dict:
    """One parity's peer regions: [src rank][MAX_S][HIDDEN / 2] pairs each."""
    n = TP * MAX_S * HIDDEN // 2 * PAIR
    return {"attn": 0, "ffn": n, "_half": 2 * n}


def peer_half() -> int:
    return peer_regions()["_half"]


# CTA map (S <= 4): task t of a stage on CTA (base + t) % BLOCKS
def split_cta(task: int) -> int:
    """Split task (token k, partition p) = k * N_SPLIT + p on the top CTAs, downward."""
    return BLOCKS - 1 - task


ROUTER_BASE = N_O  # the router tasks on the CTAs past the o tasks (192 ..)
SHARED_BASE = N_O  # the shared expert's up / gate tasks there too, after the router
