import torch
import triton
from .topk_details._topk_forward import _topk_forward
from .topk_details._topk_backward import _topk_backward
from .tensor import Bitmatrix


def topk_forward(x, k, apply_softmax=True, dim=1, return_bitmatrix=True, y_indx=None, HIST_BLOCK_M=32):
    x_shape = [x.shape[0], x.shape[1]]
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_S = 128
    BLOCK_SP = 128
    assert len(x.shape) == 2
    assert x_shape[-1] < 32768
    assert dim == 1
    assert return_bitmatrix
    n_rows, n_cols = x_shape
    dev = x.device
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals = torch.empty((n_rows, k), dtype=x.dtype, device=dev)
    if y_indx is not None:
        use_provided_indx = True
    else:
        y_indx = torch.empty((n_rows, k), dtype=torch.int16, device=dev)
        use_provided_indx = False
    # create bitmatrix in transposed memory layout:
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix = torch.empty((n_cols_words, cdiv(n_rows, 32) * 32), dtype=torch.uint32, device=dev)
    bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows]
    s_blocks = cdiv(n_cols, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
    TILE_SIZE = 8 
    BLOCK_MM = HIST_BLOCK_M * TILE_SIZE
    pids_x = cdiv(n_rows, BLOCK_MM)
    pids_y = cdiv(n_cols, 32)
    scratchpad_partials = torch.empty((pids_y * 32, pids_x * TILE_SIZE), device=dev, dtype=torch.int32)
    scratchpad_partials = torch.transpose(scratchpad_partials, 0, 1)
    sp_size = torch.numel(scratchpad_partials)
    sp_blocks = cdiv(sp_size, BLOCK_SP)
    pids = max(cdiv(n_rows, BLOCK_M), s_blocks + sp_blocks)
    _topk_forward[(pids, )](
        x, x.stride(0),  # inputs
        y_vals, y_indx, y_vals.stride(0), use_provided_indx,  # output [topk]
        bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
        n_rows, n_cols,  # shapes
        scratchpad, BLOCK_S, s_blocks,  # thing to memset to zero
        scratchpad_partials, BLOCK_SP, sp_blocks, sp_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
        num_warps=8
    )
    bitmatrix_shape = [n_rows, n_cols_words * 32]
    bitmatrix = Bitmatrix(bitmatrix, shape=bitmatrix_shape, scratchpad=scratchpad, scratchpad_partials=scratchpad_partials)
    return y_vals, y_indx, bitmatrix


def topk_backward(x, y_indx, dy_vals, k, n_rows, apply_softmax):
    assert dy_vals.shape[-1] == k
    n_expts_pad = triton.next_power_of_2(x.shape[-1])
    dx = torch.empty_like(x)
    _topk_backward[(dy_vals.shape[0], )](
        y_indx, y_indx.stride(0), dy_vals, dy_vals.stride(0), x, x.stride(0),  # inputs
        dx,  # outputs
        dx.stride(0), x.shape[0], n_rows, x.shape[-1], APPLY_SOFTMAX=apply_softmax, N_EXPTS_ACT=k,
        N_EXPTS_PAD=n_expts_pad)
    return dx


def topk(x, k, apply_softmax=True, dim=1, return_bitmatrix=True, y_indx=None, HIST_BLOCK_M=32):
    ret = topk_forward(x, k, apply_softmax, dim, return_bitmatrix, y_indx, HIST_BLOCK_M)
    return ret


# x = torch.randn((32, 32), dtype=torch.float16, device="cuda")
# print(topk(x, 4))
