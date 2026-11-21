"""The shapes every Triton GEMM test runs.

All GEMM tests call get_shapes(). Kernel limits come in as arguments, so each
kernel runs the shapes it supports. Shapes that pin one kernel's bug or config
stay in that kernel's test file.
"""

# (M, N, K)
EDGE_CASES = [
    (1, 1, 1),
    (3, 5, 2),
    (17, 200, 384),
    (31, 7168, 4608),
    (159, 17389, 597),
    # Small K: the gluon blockscale wind-down's num_k_iter guards
    # (BLOCK_SIZE_K=128; K in {128,192,256,320} -> num_k_iter in {1,2,2,3}).
    (512, 512, 128),
    (512, 512, 192),
    (512, 512, 256),
    (512, 512, 320),
    # gfx942's tuned a8w8 blockscale preshuffle config splits K=11008 into four
    # 2816-wide partitions, so the last one runs 256 past K.
    (16, 4096, 11008),
    # gfx950's a16w8 blockscale default config splits these K into 256-wide
    # partitions, so the last one runs past K (K=640 -> 3 x 256).
    (1, 6144, 640),
    (1, 6144, 896),
    (16, 6144, 640),
    (16, 6144, 896),
    (64, 6144, 640),
    (64, 6144, 896),
    # K = 35 * 32 leaves a ragged tail under any BLOCK_SIZE_K >= 64, after
    # several full K iterations rather than within the first one.
    (7, 512, 1120),
    (16, 128, 1120),
]

# (M, N, K), M <= 256
DECODE = [
    # DSR1 router
    (1, 256, 7168),
    (32, 256, 7168),
    (128, 256, 7168),
    # Low N, high K (split-K)
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    # High N, low K
    (1, 8192, 1024),
    (32, 8192, 1024),
    (128, 8192, 1024),
    (1, 5120, 2880),
    (32, 5120, 2880),
    (128, 5120, 2880),
    (1, 512, 128),
    (32, 512, 128),
    (128, 512, 128),
    (1, 128, 512),
    (32, 128, 512),
    (128, 128, 512),
    # GLM MLA: qk_nope_head_dim=192
    (32, 512, 192),
    # M < 32
    (1, 256, 512),
    (16, 256, 256),
    # very large N
    (256, 106496, 16384),
]

# (M, N, K), M >= 1024
PREFILL = [
    (1024, 256, 7168),
    (1024, 1280, 8192),
    (1024, 8192, 1024),
    (1024, 5120, 2880),
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (8192, 5120, 2880),
]

SHAPES = EDGE_CASES + DECODE + PREFILL


def get_shapes(n_mult=1, k_mult=1, min_k=1, max_size=None, batch=None, n2=None):
    """SHAPES that fit a kernel.

    n_mult, k_mult, min_k: keep shapes with N % n_mult == 0, K % k_mult == 0, K >= min_k.
    max_size: keep shapes with M * N * K <= max_size.
    batch: return (B, M, N, K), B taken round-robin from this list.
    n2: return (M, N, N2, K) for two-output GEMMs, N2 taken round-robin from this list.
    """
    shapes = [
        (m, n, k)
        for m, n, k in SHAPES
        if n % n_mult == 0
        and k % k_mult == 0
        and k >= min_k
        and (max_size is None or m * n * k <= max_size)
    ]
    if batch is not None:
        return [(batch[i % len(batch)], m, n, k) for i, (m, n, k) in enumerate(shapes)]
    if n2 is not None:
        return [(m, n, n2[i % len(n2)], k) for i, (m, n, k) in enumerate(shapes)]
    return shapes
