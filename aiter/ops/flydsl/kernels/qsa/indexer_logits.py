# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA indexer prefill logits for gfx942 (FlyDSL): relu(Q_h . K) summed over 4 heads, times scale.

One 4-wave CTA owns up to 128 query rows (32 per wave) of one request and a contiguous key
range. Q stays in VGPRs (4 heads x 64 BF16 per lane). Keys are read straight from the compressed
pool: key j of request s is slot token_slot_table[s, 4j] / 4 (a group's compressed slot is its
first raw slot / 4), looked up two blocks ahead of use. Each 32-key block is staged once per CTA
into double-buffered, XOR-swizzled LDS and fed to v_mfma_f32_32x32x8_bf16 with keys as the
MFMA rows, so every lane holds 4 consecutive keys of one query row (16-byte stores). Lane half
h holds head dims [64h, 64h + 64) of its key/query row, so k-step s pairs dims {4s..4s+3} and
{64+4s..64+4s+3}: a fixed permutation of the D=128 reduction (FP32 order differs from the
Torch einsum reference only in association). Two CTAs share a CU.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import gpu, rocdl

from aiter.ops.flydsl.kernels.qsa.indexer_topk import (
    _load as _lds_load_word,
)
from aiter.ops.flydsl.kernels.qsa.indexer_topk import (
    _store as _lds_store_word,
)
from aiter.ops.flydsl.kernels.qsa.indexer_topk import (
    _umax,
    _umin,
)
from aiter.ops.flydsl.kernels.qsa.sparse_prefill_common import (
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _min,
    _uniform,
)

HEADS, DIM, BLOCK, ITEM = 4, 128, 32, 6
# Keys per item at most, and the LDS bytes of one key block.
ITEM_KEYS, BLOCK_BYTES = 512, BLOCK * DIM * 2
# Byte offset that fails every per-wave range check.
_DROP = 0x40000000


def _buffer_word(resource, offset):
    return _buffer_words(resource, fx.Int32(offset), words=1)[0]


def _load_slots(slots, start, end, tid):
    # Raw slots TABLE[request, 4 * key] of keys start + tid + 256 i; keys past `end` repeat the last.
    keys = [start + tid + 256 * i for i in range(ITEM_KEYS // 256)]
    keys = [(key < end).select(key, end - 1) for key in keys]
    return [_buffer_word(slots, key * 16) for key in keys]


def _stage_slots(offsets, tid, raw):
    # Byte offsets of the keys in the compressed pool (compressed slot = raw slot / 4).
    for i, slot in enumerate(raw):
        _lds_store_word(offsets + (tid + 256 * i) * 4, (slot >> 2) * (DIM * 2))


def _lds(address):
    return fx.inttoptr(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16),
        fx.Int32(address),
    )


def _lds_load(address):
    return fx.Vector(fx.generic_load(_lds(address), count=4))


def _lds_store(address, value):
    fx.generic_store(_lds(address), fx.Vector(value))


def _step(words, s):
    pair = (words[2], words[3]) if s & 1 else (words[0], words[1])
    return fx.Vector.from_elements(list(pair), fx.Int32).bitcast(fx.Int16)


def _mfma(a, b, c):
    # Preserve the packed SSA fragment ABI; fx.gemm requires tensor fragments.
    return fx.Vector(
        rocdl.mfma_f32_32x32x8bf16_1k(
            ir.VectorType.get([16], fx.Float32.ir_type),
            [a.ir_value(), b.ir_value(), c.ir_value(), 0, 0, 0],
        )
    )


def _relu(values):
    # Signed-integer max on the FP32 bits: -0 and negatives become +0.
    bits = fx.Vector(values).bitcast(fx.Int32)
    zero = fx.Int32(0)
    return fx.Vector.from_elements(
        [(bits[i] > zero).select(bits[i], zero) for i in range(bits.numel)], fx.Int32
    ).bitcast(fx.Float32)


def _scores(key, query):
    # Keys are the MFMA rows and query rows the columns: lane (column, half) ends up holding keys
    # block + 8j + 4 * half + [0, 4) of query row `column`, i.e. one 16-byte store per j.
    acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(HEADS)]
    for s in range(16):
        for h in range(HEADS):
            acc[h] = _mfma(_step(key[s // 2], s), _step(query[h][s // 2], s), acc[h])
    return [_relu(value) for value in acc]


def _store(relu, output, block, end, column, half, start, pitch, scale, low, high):
    # Also folds the bits of every key the tile writes for this row (all logits are >= 0, so
    # unsigned bit order is float order and any NaN sorts above +inf) into its running min/max.
    for j in range(4):
        words = []
        for p in range(2):
            i = 4 * j + 2 * p
            value = fx.Vector.from_elements([relu[0][i], relu[0][i + 1]], fx.Float32)
            for h in range(1, HEADS):
                value = value + fx.Vector.from_elements(
                    [relu[h][i], relu[h][i + 1]], fx.Float32
                )
            value = value * fx.Vector.from_elements([scale, scale], fx.Float32)
            for e in range(2):
                bits = value.bitcast(fx.Int32)[e]
                low, high = _umin(low, bits), _umax(high, bits)
                words.append(bits)
        # A 4-key group past `end` is dropped; a partial group only spills into columns
        # [end, end + 3] of the item's key block, which no row reads as a causal logit.
        group = block + 8 * j + 4 * half
        offset = (group < end).select(
            (column * pitch + group - start) * 4, fx.Int32(_DROP)
        )
        _buffer_store_words(fx.Vector.from_elements(words, fx.Int32), output, offset)
    return low, high


def _fold(stats, offset, value):
    # BufferAtomic exposes Min but no explicit unsigned-min operation. Stats
    # compare FP32 bit patterns including 0xffffffff, so retain buffer_atomic_umin.
    rocdl.raw_ptr_buffer_atomic_umin(
        fx.Int32.ir_type,
        fx.Int32(value).ir_value(),
        rocdl.get_buffer_rsrc(stats),
        offset.ir_value(),
        fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
    )


@flyc.kernel(name="qsa_indexer_logits", known_block_size=[256, 1, 1])
def _kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    TABLE: fx.Tensor,
    LOGITS: fx.Tensor,
    ITEMS: fx.Tensor,
    STATS: fx.Tensor,
    table_stride: fx.Int32,
    key_bytes: fx.Int32,
    block_stride: fx.Int32,
    pitch: fx.Int32,
    scale: fx.Float32,
):
    # Key c's 16-byte chunk j lives at c * 16 + (j ^ (c & 7)): conflict-free ds_read_b128. The
    # item's key byte offsets in the compressed pool follow the two key blocks.
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, 2 * BLOCK_BYTES + 4 * ITEM_KEYS, 16])
        .peek()
    )
    shared = fx.Int32(
        fx.ptrtoint(
            fx.get_iter(
                storage.view(fx.make_layout(2 * BLOCK_BYTES + 4 * ITEM_KEYS, 1))
            )
        )
    )
    offsets = shared + 2 * BLOCK_BYTES
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    column, half = lane & 31, lane >> 5
    entry = fx.block_idx.x * ITEM
    row0, local0, rows = (
        _uniform(ITEMS[entry]),
        _uniform(ITEMS[entry + 1]),
        _uniform(ITEMS[entry + 2]),
    )
    sequence, start, end = (
        _uniform(ITEMS[entry + 3]),
        _uniform(ITEMS[entry + 4]),
        _uniform(ITEMS[entry + 5]),
    )
    # Key j of the request is compressed slot TABLE[sequence, 4j] / 4; stage this item's slots
    # (keys past `end` repeat the last one) before the queries load.
    slots = _buffer(
        fx.make_view(
            fx.get_iter(TABLE) + fx.Int64(sequence) * fx.Int64(table_stride),
            fx.make_layout(1, 1),
        ),
        end * 16,
    )
    raw_slots = _load_slots(slots, start, end, tid)
    wave_rows = rows - 32 * wave
    wave_rows = (wave_rows > 32).select(
        fx.Int32(32), (wave_rows < 0).select(fx.Int32(0), wave_rows)
    )

    queries = _buffer(
        fx.make_view(
            fx.get_iter(Q) + fx.Int64(row0) * (HEADS * DIM), fx.make_layout(1, 1)
        ),
        rows * (HEADS * DIM * 2),
    )
    row = 32 * wave + column
    row = (row < rows).select(row, rows - 1)
    query = [
        [
            _buffer_words(
                queries, row * (HEADS * DIM * 2) + half * 128 + h * (DIM * 2) + i * 16
            )
            for i in range(8)
        ]
        for h in range(HEADS)
    ]

    # Buffer stores are range checked against the wave's rows, so padded rows are dropped without
    # branches; 4-key groups starting past `end` push their offset out of range. An item covers one
    # 512-key block, so its rows' logits are one contiguous stretch.
    output = _buffer(
        fx.make_view(
            fx.get_iter(LOGITS)
            + fx.Int64(start // ITEM_KEYS) * fx.Int64(block_stride)
            + fx.Int64(local0 + 32 * wave) * fx.Int64(pitch),
            fx.make_layout(1, 1),
        ),
        wave_rows * pitch * 4,
    )
    # Thread t copies 32 bytes (chunks 2(t%8), 2(t%8)+1) of key t/8 of each block into LDS.
    copy_key, copy_chunk = tid >> 3, (tid & 7) * 2
    keys = _buffer(fx.make_view(fx.get_iter(K), fx.make_layout(1, 1)), key_bytes)
    stage0 = copy_key * 256 + ((copy_chunk ^ (copy_key & 7)) * 16)
    stage1 = copy_key * 256 + (((copy_chunk + 1) ^ (copy_key & 7)) * 16)
    read = [column * 256 + (((8 * half + i) ^ (column & 7)) * 16) for i in range(8)]
    _stage_slots(offsets, tid, raw_slots)
    gpu.barrier()

    source = _lds_load_word(offsets + copy_key * 4) + copy_chunk * 16
    upcoming = _lds_load_word(offsets + (BLOCK + copy_key) * 4)
    staged0, staged1 = _buffer_words(keys, source), _buffer_words(keys, source + 16)
    _lds_store(shared + stage0, staged0)
    _lds_store(shared + stage1, staged1)
    low, high = fx.Int32(-1), fx.Int32(0)
    gpu.barrier()
    buffer = fx.Int32(0)
    for block in range(start, end, fx.Int32(BLOCK)):
        current = fx.Int32(block)
        more = current + BLOCK < end
        if more:
            source = upcoming + copy_chunk * 16
            staged0, staged1 = _buffer_words(keys, source), _buffer_words(
                keys, source + 16
            )
            # The slot of the block after next; past the item it is never used.
            upcoming = _lds_load_word(
                offsets
                + _min(current + 2 * BLOCK - start + copy_key, ITEM_KEYS - 1) * 4
            )
        base = shared + buffer * (BLOCK * 256)
        key = [_lds_load(base + address) for address in read]
        low, high = _store(
            _scores(key, query),
            output,
            current,
            end,
            column,
            half,
            start,
            pitch,
            scale,
            low,
            high,
        )
        if more:
            base = shared + (buffer ^ 1) * (BLOCK * 256)
            _lds_store(base + stage0, staged0)
            _lds_store(base + stage1, staged1)
        gpu.barrier()
        buffer = buffer ^ 1
    # The row's (min bits, ~max bits) fold with unsigned min; both lane halves hold the row.
    low = _umin(low, low.shuffle_xor(32, 64))
    high = _umax(high, high.shuffle_xor(32, 64))
    mine = 32 * wave + column
    stats = _buffer(
        fx.make_view(fx.get_iter(STATS), fx.make_layout(1, 1)), (row0 + rows) * 8
    )
    offset = ((half == 0) & (mine < rows)).select((row0 + mine) * 8, fx.Int32(_DROP))
    _fold(stats, offset, low)
    _fold(stats, offset + 4, ~high)


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    TABLE: fx.Tensor,
    LOGITS: fx.Tensor,
    ITEMS: fx.Tensor,
    STATS: fx.Tensor,
    table_stride: fx.Int32,
    key_bytes: fx.Int32,
    block_stride: fx.Int32,
    pitch: fx.Int32,
    scale: fx.Float32,
    items: fx.Int32,
    stream: fx.Stream,
):
    if items > 0:
        _kernel(
            Q,
            K,
            TABLE,
            LOGITS,
            ITEMS,
            STATS,
            table_stride,
            key_bytes,
            block_stride,
            pitch,
            scale,
            value_attrs={"rocdl.waves_per_eu": 2},
        ).launch(grid=(items, 1, 1), block=(256, 1, 1), stream=stream)
