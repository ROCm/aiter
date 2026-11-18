# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test shapes for the Triton GEMM tests.

Every GEMM test reads its shapes from here. The lists are built from the shared
pieces below, in the order each test runs them. Shapes that pin one kernel's
bug or config stay in that kernel's test file.

GEMM shapes are (M, N, K); batched ones (B, M, N, K); the fused two-output
GEMMs (M, N1, N2, K); GMM (M, K, N, G).
"""

import functools

# Shared pieces
# ------------------------------------------------------------------------------

MINIMAL = [(1, 1, 1)]
IRREGULAR = [(3, 5, 2)]


def squares(multiples):
    """Square (M, N, K) = 1024 * v for each v."""
    return [(1024 * v, 1024 * v, 1024 * v) for v in multiples]


# M from decode to prefill, for the low-N/high-K and high-N/low-K pairs below.
DECODE_M = (1, 32, 64, 128, 192, 256, 320, 512, 1024, 2048, 4096, 8192, 16384)
DECODE_1280_8192 = [(m, 1280, 8192) for m in DECODE_M]
DECODE_8192_1024 = [(m, 8192, 1024) for m in DECODE_M]

# Model GEMMs at M = 32, 64, 128, 256.
M_POW2_32_256 = [2**i for i in range(5, 9)]
DSR1_ROUTER = [(m, 256, 7168) for m in M_POW2_32_256]
# GPT-OSS-120B attention projections.
GPTOSS_QKV = [(m, 5120, 2880) for m in M_POW2_32_256]
GPTOSS_O = [(m, 2880, 4096) for m in M_POW2_32_256]
GPTOSS_ROUTER = [(m, 128, 2880) for m in M_POW2_32_256]
# DSv4 router gate: num_tokens x 384 x 7168.
DSV4_ROUTER = [(m, 384, 7168) for m in M_POW2_32_256]


def ll3_405b_fc1(ms):
    """LL3 405B FC1 at the given M."""
    return [(m, 106496, 16384) for m in ms]


# M = 128, 192, 4096, 8000 over a model (N, K).
M_MIXED = (128, 192, 4096, 8000)


def mixed_m(n, k):
    return [(m, n, k) for m in M_MIXED]


# Smaller set for testing activations, setting the output tensor and dtype.
FEWER = [(16, 1024, 1024), (128, 8192, 512), (256, 512, 8192)] + squares((1, 5, 8))

# Tuned model (N, K) pairs
# ------------------------------------------------------------------------------

# GPT-OSS on gfx1250 (tuned a16w16 configs).
GPTOSS_NK = [
    (5120, 2880),  # TP1 QKV
    (2880, 4096),  # TP1 O
    (128, 2880),  # router
    (2560, 2880),  # TP2 QKV
    (2880, 2048),  # TP2 O
    (201088, 2880),  # LM head, TP1
    (100544, 2880),  # LM head, TP2 (partial last N tile)
]

# (N, K) of the linear layers of Llama-3-8B (also Mistral-7B) and Qwen2.5-7B:
# fused qkv, o, fused gate_up and down projections (a16w4).
A16W4_MODEL_NK = [
    (6144, 4096),
    (4096, 4096),
    (28672, 4096),
    (4096, 14336),
    (4608, 3584),
    (3584, 3584),
    (37888, 3584),
    (3584, 18944),
]

# DeepSeek-V3 (N, K) pairs at M = 1, 32, 64, 128, 256, 1024 for a8w8 blockscale.
A8W8_BLOCKSCALE_NK = [
    (8192, 1024),
    (4096, 8192),
    (4096, 4096),
    (4096, 2048),
    (1536, 4096),
    (32768, 1024),
    (8192, 1536),
    (7168, 4096),
    (1536, 7168),
    (7168, 768),
    (2048, 7168),
    (16384, 1536),
    (65536, 1536),
    (7168, 16384),
    (6144, 7168),
    (7168, 3072),
]

# DeepSeek-V3 (N, K) pairs for afp8wfp8: N % 128 == 0 and K % 128 == 0 to fit the
# 128x128 W-scale layout.
AFP8WFP8_NK = [
    (2048, 7168),
    (65536, 1536),
    (16384, 1536),
    (8192, 1536),
    (7168, 16384),
    (7168, 4096),
    (6144, 7168),
    (7168, 3072),
    (1536, 7168),
    (7168, 768),
]

# Basic GEMMs: (M, N, K)
# ------------------------------------------------------------------------------

GEMM_A16W16 = (
    MINIMAL
    + IRREGULAR
    + squares((1, 2, 4, 5, 8))
    + DSR1_ROUTER
    + GPTOSS_QKV
    + [(2048, 5120, 2880), (8192, 5120, 2880)]
    + GPTOSS_O
    + GPTOSS_ROUTER
    + ll3_405b_fc1((256, 4096))
)

GEMM_A16W4 = [(1, 64, 128), (3, 72, 256), (17, 200, 384), (130, 520, 1024)] + [
    (m, n, k)
    for n, k in A16W4_MODEL_NK
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 8192)
]

GEMM_A16W8_BLOCKSCALE = (
    MINIMAL
    + IRREGULAR
    + squares((1, 2, 4, 5, 8))
    + DSR1_ROUTER
    + GPTOSS_O
    + ll3_405b_fc1((256, 4096))
    # gfx950's default config splits these K into 256-wide partitions, so the
    # last one runs past K (K=640 -> 3 x 256)
    + [(m, 6144, k) for m in (1, 16, 64) for k in (640, 896)]
)

GEMM_A16WFP4 = squares((1, 2, 4, 5, 8)) + mixed_m(128, 512) + mixed_m(2112, 7168)

GEMM_A8W8 = (
    MINIMAL
    + IRREGULAR
    + squares((1, 2, 4, 5, 8))
    + ll3_405b_fc1((190, 256, 4096))
    + [(m, 10240, 8192) for m in (256, 4096, 8000)]  # LL3 70B QKV input projection
)

GEMM_A8W8_SPLITK = (
    [(m, 1280, 8192) for m in (1, 32, 64, 128, 256)]
    + [(m, 8192, 1024) for m in (1, 32, 64, 128, 256)]
    + [
        (1024, 1024, 1000),
        (1024, 1024, 1024),
        (1024, 1024, 4096),
        (1024, 4096, 4096),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
)

GEMM_A8W8_BLOCKSCALE = (
    squares((1, 2, 4, 5, 8))
    + ll3_405b_fc1((256, 4096))
    + mixed_m(9216, 7168)
    + mixed_m(7168, 4608)
    + mixed_m(8192, 512)
    # Small-K shapes that exercise the gluon wind-down's num_k_iter guards
    # (BLOCK_SIZE_K=128; K in {128,192,256,320} -> num_k_iter in {1,2,2,3}).
    # K<BLOCK_SIZE_K isn't supported by the gluon wrapper (GROUP_K assert).
    + [(512, 512, k) for k in (128, 192, 256, 320)]
    + [(m, n, k) for n, k in A8W8_BLOCKSCALE_NK for m in (1, 32, 64, 128, 256, 1024)]
    # gfx942's tuned preshuffle config splits K=11008 into four 2816-wide
    # partitions, so the last one runs 256 past K
    + [(16, 4096, 11008)]
)

GEMM_A8W8_PER_TOKEN_SCALE = (
    MINIMAL
    + IRREGULAR
    + squares((1, 2, 4, 5, 8))
    + ll3_405b_fc1((256, 4096))
    + mixed_m(5120, 2880)  # GPT-OSS QKV input projection
    + mixed_m(2880, 4096)  # GPT-OSS output projection
)

GEMM_A8WFP4 = (
    squares((1, 2, 4, 5, 8))
    + DSR1_ROUTER
    + GPTOSS_QKV
    + GPTOSS_O
    + GPTOSS_ROUTER
    + [(m, 57344, 8192) for m in (128, 2048)]  # LL3 405B FC1 (reduced)
)

GEMM_AFP4WFP4 = (
    squares((1, 2, 4, 5, 8))
    + ll3_405b_fc1((150, 256, 4096, 8000))
    + mixed_m(9216, 7168)
    + mixed_m(7168, 4608)
    + mixed_m(2112, 7168)
    + mixed_m(8192, 512)
    + [(2048, 7168, 16384), (2304, 24576, 1536)]
    + [(1, 256, 512), (16, 256, 256), (31, 7168, 4608)]  # M < 32 case
    + [(m, 96, 5120) for m in (32, 128, 192, 256, 8000)]
)

GEMM_AFP8WFP8 = [(m, n, k) for m in (16, 32, 64, 512) for n, k in AFP8WFP8_NK]

# Batched GEMMs: (B, M, N, K)
# ------------------------------------------------------------------------------


def cycle_batch(shapes, batch_sizes, single_batch_k=None):
    """(B, M, N, K) with B taken round-robin from batch_sizes; B = 1 when K > single_batch_k."""
    out = []
    for i, (m, n, k) in enumerate(shapes):
        b = batch_sizes[i % len(batch_sizes)]
        if single_batch_k is not None and k > single_batch_k:
            b = 1
        out.append((b, m, n, k))
    return out


def smallest(shapes, num_vals=20):
    """The num_vals shapes with the fewest elements (M * N * K), for quick subsets."""
    num_ops = [(s, functools.reduce(lambda x, y: x * y, s)) for s in shapes]
    return [s for s, _ in sorted(num_ops, key=lambda x: x[1])[:num_vals]]


# MLA batched shapes: B = 1..16, M = 1..256, (N, K) = (512, 128) and (128, 512).
MLA_BATCHED = [
    (b, 2**m, n, k)
    for b in range(1, 17)
    for m in range(9)
    for n, k in [(512, 128), (128, 512)]
]

# DSR1 MLA absorbed BMMs: W_UK (N=512, K=128) and W_UV (N=128, K=512), 128 heads.
BATCHED_GEMM_A16W8_MLA = [
    (128, m, n, k)
    for m in [1, 7, 64, 256, 1536, 3000]
    for n, k in [(512, 128), (128, 512)]
]
BATCHED_GEMM_A16W8_GENERIC = [
    (16, 1, 128, 64),
    (16, 100, 384, 320),
    (4, 257, 72, 1024),
    (8, 1024, 1024, 1024),
]

BATCHED_GEMM_A16WFP4 = (
    MLA_BATCHED
    # This mirrors GLM MLA's qk_nope_head_dim=192.
    + [(16, 32, 512, 192)]
    # K = 35 * 32 leaves a ragged tail under any BLOCK_SIZE_K >= 64, after
    # several full K iterations rather than within the first one.
    + [(2, 7, 512, 1120), (3, 16, 128, 1120)]
)

BATCHED_GEMM_A8W8 = (
    squares(range(1, 9))
    + [(4864, 4096, 8192), (9728, 8192, 65000), (4864, 8192, 4160)]
    + DECODE_1280_8192
    + DECODE_8192_1024
    + MINIMAL
)

BATCHED_GEMM_A8W8_PREQUANT = (
    squares(range(1, 9))
    + DECODE_1280_8192
    + DECODE_8192_1024
    + [(v**2, 128, 512) for v in range(7)]
    + [(v**2, 512, 128) for v in range(7)]
    + [(1, 128, 1)]  # minimal case
)

# TODO: There's a known bug for large test cases (e.g (9728, 8192, 65536))
# That will cause a failure on the next test. My best guess is that we're not
# overwriting something we should when we get a big chunk of uninitialized data
# in torch.empty().
BATCHED_GEMM_AFP4WFP4 = (
    cycle_batch(
        squares(range(1, 9))
        + [(4864, 4096, 8192), (4864, 8192, 4160)]
        + DECODE_1280_8192
        + DECODE_8192_1024
        + [(2 ** (v - 1), 4096 * v, 4096 * v) for v in range(1, 6)]
        + [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
        + [(256, 3584, 2112)]
        + [(1, 1, 32)],  # minimal case
        [1, 2, 3, 5, 7, 8],
        single_batch_k=16384,
    )
    + MLA_BATCHED
)

BATCHED_GEMM_BF16 = (
    squares(range(1, 4))
    + [(m, 1024, 4096) for m in (1, 32, 64, 128, 256, 16384)]
    + MINIMAL
)

# Fused GEMMs
# ------------------------------------------------------------------------------

# (M, N1, N2, K)
FUSED_GEMM_A16W16_DUAL_SPLITK = (
    [(1, 1, 1, 128)]  # minimal case
    + [
        (m, n1, n2, k)
        for k in [1024, 5120]
        for n1, n2 in [(256, 32), (4096, 96)]
        for m in [1, 8, 64, 128]
    ]
    + [(m, 16384, 96, 5120) for m in [1, 4, 16, 64, 128]]  # tuned shape
)

# (M, N1, N2, K)
FUSED_GEMM_A8W8_BLOCKSCALE_A16W16 = [(1, 1, 1, 128)] + [  # minimal case
    (m, n1, n2, k)
    for k in [1024, 8192, 7168]
    for n2 in [256, 512]
    for n1 in [256, 512]
    for m in [1, 8, 32, 64, 128, 8192]
]

# (M, N1, N2, K)
FUSED_GEMM_AFP4WFP4_A16W16 = FUSED_GEMM_A8W8_BLOCKSCALE_A16W16[1:]

# (M, N, K)
FUSED_GEMM_A16W16_MUL_ADD_2D = (
    MINIMAL
    + IRREGULAR
    + [(16, 1024, 1024), (128, 8192, 512), (1024, 1024, 1024), (4096, 1152, 1152)]
)

# (B, S, N, K)
FUSED_GEMM_A16W16_MUL_ADD_3D = [
    (2, 77, 640, 320),  # S is not a multiple of the tile
    (2, 333, 2432, 2432),  # SD3.5-Large context stream
    (2, 1024, 1152, 1152),  # PixArt-XL at 512px
    (1, 4096, 1536, 1536),  # Wan2.1-1.3B
]

# (M, N, K)
FUSED_GEMM_A16W16_QUANT_X = squares((1, 2)) + DSV4_ROUTER + DSR1_ROUTER
FUSED_GEMM_A16W16_QUANT_X_FEWER = FEWER[:3] + squares((1,))

# (M, N, K)
# TODO M = 1 triton upstream compilation error without AMDGCN_USE_BUFFER_OPS=0
FUSED_GEMM_A8W8_BLOCKSCALE_MUL_ADD = [(m, 7168, 256) for m in [2, 4, 8, 16, 1024]]
FUSED_GEMM_AFP4WFP4_MUL_ADD = [(m, 7168, 256) for m in [1, 2, 4, 8, 16, 31, 32, 1024]]

# (M, N, K)
FUSED_GEMM_A8W8_BLOCKSCALE_SPLIT_CAT = [(4096, 256, 512)]

# (M, N, K)
FUSED_GEMM_AFP4WFP4_SPLIT_CAT = (
    squares(range(1, 4))
    + [(m, 1280, 8192) for m in (1, 32, 64, 128, 192, 320, 8192)]
    + [(m, 8192, 1024) for m in (1, 32, 64, 128, 192, 320, 8192)]
    + [(2048, 2048, 2049), (159, 17389, 597), (16, 576, 7168)]
    + [(m, 4096, 512) for m in (16, 32, 64, 128, 256, 1024)]
)

# Grouped GEMM: (M, K, N, G)
# ------------------------------------------------------------------------------

# Shapes used only for test purposes.
# fmt: off
GMM_TEST_ONLY_SHAPES: list[tuple[int, int, int, int]] = [
    #  M,    K,    N,   G
    ( 10,    2,    3,   4),
    ( 32,   16,    8,   4),
]
# fmt: on

# Real shapes, used by real models.
# fmt: off
GMM_REAL_SHAPES: list[tuple[int, int, int, int]] = [
    #      M,     K,     N,   G
    (  49152,  1408,  2048,  64),  # deepseekv2-16B
    (3145728,  2048,  1408,   8),  # deepseekv2-16B
    (  32768,  6144, 16384,   8),  # Mixtral 8x22B
    (  32768, 16384,  6144,   8),  # Mixtral 8x22B
]
# fmt: on

# Test shapes are test only + real ones.
GMM_SHAPES: list[tuple[int, int, int, int]] = GMM_TEST_ONLY_SHAPES + GMM_REAL_SHAPES
