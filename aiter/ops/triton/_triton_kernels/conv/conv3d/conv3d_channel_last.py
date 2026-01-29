import triton.language as tl
import triton

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_conv3d_forward_repr = make_kernel_repr(
    "_conv3d_channel_last_kernel",
    [
        "K_C",
        "K_D",
        "K_H",
        "K_W",
        "STRIDE_D",
        "STRIDE_H",
        "STRIDE_W",
        "PAD_D",
        "PAD_H",
        "PAD_W",
        "DIL_D",
        "DIL_H",
        "DIL_W",
        "GROUPS",
        "BLOCK_N",
        "BLOCK_CI",
        "BLOCK_CO",
    ],
)


@triton.jit(repr=_conv3d_forward_repr)
def _conv3d_channel_last_kernel(
    # Pointers to tensors
    x_ptr,
    w_ptr,
    y_ptr,
    b_ptr,
    # Tensor dimensions
    N,
    D,
    H,
    W,
    OC,
    OD,
    OH,
    OW,
    # Strides
    s_x_n,
    s_x_c,
    s_x_d,
    s_x_h,
    s_x_w,
    s_w_o,
    s_w_c,
    s_w_d,
    s_w_h,
    s_w_w,
    s_y_n,
    s_y_c,
    s_y_d,
    s_y_h,
    s_y_w,
    # Meta-parameters
    K_C: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,  # M: NI_OD_OH_OW
    BLOCK_CI: tl.constexpr,  # K: The inner dimension block size for dot
    BLOCK_CO: tl.constexpr,  # N: CO per group
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_g = tl.program_id(2)

    # calculate n/od/oh/ow in-kernel
    n_odoho_owo = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_odoho = n_odoho_owo // OW
    n_odo = n_odoho // OH
    n_idx = n_odo // OD
    od_idx = n_odo % OD
    oh_idx = n_odoho % OH
    ow_idx = n_odoho_owo % OW

    # input: [N, G, C, D, H, W], weight: [G, OC, C, KD, KH, KW]
    out_per_g = OC // GROUPS
    co_off = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    x_ptr += (s_x_n * n_idx + s_x_c * pid_g * K_C)[:, None]
    w_ptr += (s_w_o * co_off + s_w_o * pid_g * out_per_g)[None, :]

    acc = tl.zeros((BLOCK_N, BLOCK_CO), dtype=tl.float32)
    CI_TILES = (K_C + BLOCK_CI - 1) // BLOCK_CI
    for dhwc in range(K_D * K_H * K_W * CI_TILES):
        c0 = (dhwc % CI_TILES) * BLOCK_CI
        dhw = dhwc // CI_TILES
        dh = dhw // K_W
        d = dh // K_H
        h = dh % K_H
        w = dhw % K_W

        ci_off = c0 + tl.arange(0, BLOCK_CI)
        in_d_off = d * DIL_D - PAD_D + STRIDE_D * od_idx
        in_h_off = h * DIL_H - PAD_H + STRIDE_H * oh_idx
        in_w_off = w * DIL_W - PAD_W + STRIDE_W * ow_idx

        x_blk_ptr = (
            x_ptr
            + (s_x_c * ci_off)[None, :]
            + (s_x_d * in_d_off)[:, None]
            + (s_x_h * in_h_off)[:, None]
            + (s_x_w * in_w_off)[:, None]
        )
        w_blk_ptr = (
            w_ptr + (s_w_c * ci_off)[:, None] + (s_w_d * d) + (s_w_h * h) + (s_w_w * w)
        )

        x_mask = (
            (n_idx < N)[:, None]
            & (ci_off < K_C)[None, :]
            & (0 <= in_d_off)[:, None]
            & (in_d_off < D)[:, None]
            & (0 <= in_h_off)[:, None]
            & (in_h_off < H)[:, None]
            & (0 <= in_w_off)[:, None]
            & (in_w_off < W)[:, None]
        )
        w_mask = (ci_off < K_C)[:, None] & (co_off < out_per_g)[None, :]

        x_blk = tl.load(x_blk_ptr, mask=x_mask)
        w_blk = tl.load(w_blk_ptr, mask=w_mask)

        acc += tl.dot(x_blk, w_blk, allow_tf32=False)

    b_ptr += (pid_g[None] * out_per_g)[None, :] + co_off[None, :]
    b_mask = (co_off < out_per_g)[None, :]
    bias = tl.load(b_ptr, b_mask).to(tl.float32)
    acc += bias

    y_ptr += (
        (s_y_n * n_idx)[:, None]
        + (s_y_c * (pid_g * out_per_g + co_off))[None, :]
        + (s_y_d * od_idx)[:, None]
        + (s_y_h * oh_idx)[:, None]
        + (s_y_w * ow_idx)[:, None]
    )
    y_mask = (
        (n_idx < N)[:, None]
        & (co_off < out_per_g)[None, :]
        & (od_idx < OD)[:, None]
        & (oh_idx < OH)[:, None]
        & (ow_idx < OW)[:, None]
    )

    tl.store(y_ptr, acc, mask=y_mask)
