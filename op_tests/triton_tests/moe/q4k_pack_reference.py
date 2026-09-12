# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Pure-numpy GGUF Q4_K_M packer + dequant reference, used by
# test_moe_q4k_streaming.py to construct ground-truth inputs and outputs.
#
# The packed bytes produced by ``pack_block`` are byte-exact to llama.cpp's
# Q4_K_M layout (validated against ``dequantize_row_q4_K`` in libggml.so;
# see test_q4k_vs_ggml.py in the development workspace).

from __future__ import annotations

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
BLOCK_BYTES = 4 + K_SCALE_SIZE + QK_K // 2  # = 144


def pack_scale_min(sc: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Encode 8 (sc, m) 6-bit pairs into 12 bytes."""
    assert sc.shape == (8,) and m.shape == (8,)
    assert (sc < 64).all() and (m < 64).all(), "sc/m must fit in 6 bits"
    sc = sc.astype(np.uint8)
    m = m.astype(np.uint8)
    out = np.zeros(K_SCALE_SIZE, dtype=np.uint8)
    for j in range(4):
        out[j] = sc[j] | ((sc[j + 4] >> 4) << 6)
        out[j + 4] = m[j] | ((m[j + 4] >> 4) << 6)
    for j in range(4, 8):
        out[j + 4] = (sc[j] & 0x0F) | ((m[j] & 0x0F) << 4)
    return out


def unpack_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_scale_min, mirrors the C ``q4k_unpack_scale_min``."""
    assert scales.shape == (K_SCALE_SIZE,) and scales.dtype == np.uint8
    sc = np.zeros(8, dtype=np.uint8)
    m = np.zeros(8, dtype=np.uint8)
    for j in range(4):
        sc[j] = scales[j] & 0x3F
        m[j] = scales[j + 4] & 0x3F
    for j in range(4, 8):
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, m


def pack_block(
    dall: float, dmin: float, sc: np.ndarray, m: np.ndarray, q: np.ndarray
) -> bytes:
    """Construct a valid Q4_K_M block from raw values for testing.

    Args:
        dall, dmin: fp16-representable super-block scales
        sc, m: shape (8,), 6-bit values in [0, 63]
        q:     shape (256,), 4-bit values in [0, 15]
    Returns:
        144-byte Q4_K_M block.
    """
    assert q.shape == (QK_K,)
    assert (q < 16).all(), "nibbles must be 4-bit"

    out = bytearray(BLOCK_BYTES)
    out[0:4] = np.array([dall, dmin], dtype=np.float16).tobytes()
    out[4:16] = pack_scale_min(sc, m).tobytes()

    # Vectorized nibble packing. The qs[128]-byte layout interleaves two
    # consecutive sub-blocks per byte: byte 32*il+pos holds q[64*il+pos] in
    # the low nibble and q[64*il+pos+32] in the high nibble, for il in 0..3
    # and pos in 0..31.
    q_2d = q.reshape(4, 2, 32).astype(np.uint8) & 0x0F
    low = q_2d[:, 0, :]   # (4, 32)
    high = q_2d[:, 1, :]  # (4, 32)
    qs = (low | (high << 4)).reshape(-1)
    out[16:144] = qs.tobytes()
    return bytes(out)


def dequant_block(block: bytes) -> np.ndarray:
    """Dequantize a single Q4_K_M block to 256 float32 values."""
    assert len(block) == BLOCK_BYTES
    arr = np.frombuffer(block, dtype=np.uint8)
    dall, dmin = np.frombuffer(arr[0:4].tobytes(), dtype=np.float16).astype(np.float32)
    sc, m = unpack_scale_min(arr[4:16])
    sc_f = dall * sc.astype(np.float32)
    m_f = dmin * m.astype(np.float32)
    qs = arr[16:144]
    out = np.zeros(QK_K, dtype=np.float32)
    for idx in range(128):
        il = idx // 32
        pos = idx % 32
        b = qs[32 * il + pos]
        out[64 * il + pos] = sc_f[2 * il] * (b & 0x0F) - m_f[2 * il]
        out[64 * il + pos + 32] = sc_f[2 * il + 1] * (b >> 4) - m_f[2 * il + 1]
    return out


def dequant_expert(expert_bytes: bytes, n_dim_in: int, n_dim_out: int) -> np.ndarray:
    """Dequantize a full expert weight matrix to fp32 ``(n_dim_out, n_dim_in)``."""
    assert n_dim_in % QK_K == 0
    n_blocks = n_dim_in // QK_K
    bytes_per_row = n_blocks * BLOCK_BYTES
    assert len(expert_bytes) == n_dim_out * bytes_per_row
    out = np.empty((n_dim_out, n_dim_in), dtype=np.float32)
    for r in range(n_dim_out):
        row = expert_bytes[r * bytes_per_row : (r + 1) * bytes_per_row]
        for bi in range(n_blocks):
            chunk = row[bi * BLOCK_BYTES : (bi + 1) * BLOCK_BYTES]
            out[r, bi * QK_K : (bi + 1) * QK_K] = dequant_block(chunk)
    return out


def moe_matvec_scattered_ref(
    expert_bufs: list[bytes],
    remap: np.ndarray,
    x: np.ndarray,
    n_dim_out: int,
) -> np.ndarray:
    """Reference scattered-pointer MoE matvec.

    Returns dst with shape ``(n_tokens, n_used_per_token, n_dim_out)`` fp32.
    """
    n_tokens, n_used = remap.shape
    n_dim_in = x.shape[1]
    out = np.zeros((n_tokens, n_used, n_dim_out), dtype=np.float32)
    for t in range(n_tokens):
        for u in range(n_used):
            slot = int(remap[t, u])
            W = dequant_expert(expert_bufs[slot], n_dim_in, n_dim_out)
            out[t, u] = W @ x[t]
    return out


def build_random_expert(rng, n_dim_in: int, n_dim_out: int) -> bytes:
    """Generate a random expert weight blob (raw bytes) for tests/benchmarks.

    Used by correctness tests, where each block needs to be independently
    randomized to exercise per-block scale/nibble paths. For benchmarks where
    realistic byte content is unimportant, prefer ``build_pattern_expert``.
    """
    n_blocks = n_dim_in // QK_K
    rows = bytearray()
    for _ in range(n_dim_out):
        for _ in range(n_blocks):
            sc = rng.integers(1, 32, size=8, dtype=np.uint8)
            m = rng.integers(0, 16, size=8, dtype=np.uint8)
            q = rng.integers(0, 16, size=QK_K, dtype=np.uint8)
            dall = float(rng.uniform(0.01, 0.5))
            dmin = float(rng.uniform(0.0, 0.05))
            rows += pack_block(dall, dmin, sc, m, q)
    return bytes(rows)


def build_pattern_expert(rng, n_dim_in: int, n_dim_out: int) -> bytes:
    """Build a valid expert weight blob by replicating one random block.

    Faster than ``build_random_expert`` for large shapes (skips the per-block
    Python loop). Suitable for benchmarks where memory access pattern matters
    but actual byte content does not.
    """
    n_blocks_per_row = n_dim_in // QK_K
    sc = rng.integers(1, 32, size=8, dtype=np.uint8)
    m = rng.integers(0, 16, size=8, dtype=np.uint8)
    q = rng.integers(0, 16, size=QK_K, dtype=np.uint8)
    blk = pack_block(float(rng.uniform(0.01, 0.5)),
                     float(rng.uniform(0.0, 0.05)), sc, m, q)
    return blk * (n_blocks_per_row * n_dim_out)
