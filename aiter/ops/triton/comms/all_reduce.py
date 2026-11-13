import ctypes

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _all_reduce_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_LINES = 64  # lines of a chunk: 2 for each lane of one wave
_MAXIT = 20_000_000  # poll limit, some seconds: a peer can still be compiling
_PROGRAMS = 2048  # receivers: more polling receivers can fill the GPU
_POOLS = {}  # the IPC pool of each inbox, by its address: the pool owns the inbox


def norm_waves(hidden: int) -> int:
    """The waves of an all_reduce_rmsnorm program for rows of hidden values, 2 chunks
    a wave, or 0 if it does not support hidden. Hidden 6144 is 8 waves, as the 8
    waves of add_rmsnorm_quant_kernel<256, 24>, so that the results have the same bits.
    """
    waves = triton.next_power_of_2(triton.cdiv(hidden, 12 * _LINES))
    return waves if hidden > 0 and hidden % 8 == 0 and waves <= 32 else 0


def all_reduce_init(group, max_bytes):
    """
    Allocates the inbox of this rank and maps the inboxes of the 4 ranks of
    ``group``, for tensors of at most ``max_bytes``. All ranks must call it.
    Needs ROCm 7.14 or later: the inboxes are shared with hipIpc.

    Returns:
        tuple: The inbox addresses of ranks 0 to 3, and the int32 call counters
        of this rank. calls[-1] counts the polls that timed out.
    """
    from aiter.dist.device_communicators.custom_all_reduce import (
        IPCBufferPool,
        _loaded_hip_runtime,
        _should_use_vmm,
    )

    assert get_arch() == "gfx1250" and dist.get_world_size(group) == 4
    assert not _should_use_vmm(True), "needs ROCm 7.14 or later for hipIpc"
    # A chunk of 64 lines of 16 bytes and 2 call counters for each 512 bytes of x.
    chunks = triton.cdiv(max_bytes, 8 * _LINES)
    size = 2 * 4 * chunks * _LINES * 16  # 2 banks x 4 senders
    device = torch.device("cuda", torch.cuda.current_device())
    pool = IPCBufferPool(device, group)
    # Cached memory (hipMalloc): the polls are slower on uncached memory.
    inbox = pool.create("inbox", size, raw_cached=True)
    err = _loaded_hip_runtime().hipMemset(
        ctypes.c_void_p(inbox.data_ptr), 0, ctypes.c_size_t(size)
    )
    assert err == 0, f"hipMemset failed: {err}"  # 0 is never a tag
    torch.cuda.synchronize()
    inboxes = tuple(pool.open_peer_buffers("inbox"))
    _POOLS[inboxes[dist.get_rank(group)]] = pool
    calls = torch.zeros(2 * chunks + 1, dtype=torch.int32, device=device)
    return inboxes, calls


def all_reduce_free(group, inboxes):
    """Unmaps the peer inboxes and frees the inbox of this rank. All ranks must
    call it."""
    pool = _POOLS.pop(inboxes[dist.get_rank(group)])
    torch.cuda.synchronize()
    dist.barrier(group=group)
    pool.close_peer_buffers()
    dist.barrier(group=group)  # no peer uses the inbox of this rank now
    pool.close()


def all_reduce(x: torch.Tensor, group, inboxes, calls) -> torch.Tensor:
    """
    Returns the sum of the bf16 tensor x over the 4 ranks of ``group`` on gfx1250,
    with one kernel launch. All ranks get the same bits. All ranks must make the
    same calls in the same order. HIP graphs are supported.

    Args:
        x (torch.Tensor): Contiguous bf16, 16-byte aligned, a multiple of 4
            values and at most max_bytes.
        inboxes, calls: The result of ``all_reduce_init``. If a poll times out,
            the sum is wrong and calls[-1] is more than 0.
    """
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() % 4 == 0
    chunks = (calls.numel() - 1) // 2
    n = x.numel()
    assert x.data_ptr() % 16 == 0 and n <= 6 * _LINES * chunks

    _LOGGER.info("ALL_REDUCE [gluon]: x=%s", x.shape)

    y = torch.empty_like(x)
    num_chunks = triton.cdiv(n, 6 * _LINES)  # 6 values a line
    programs = min(num_chunks, _PROGRAMS)
    _all_reduce_kernel[(num_chunks + programs,)](
        x,
        y,
        inboxes,
        calls,
        calls[-1:],
        n,
        programs,
        x,  # residual, weight, out, scale: not used without NORM
        x,
        x,
        x,
        RANK=dist.get_rank(group),
        LINES=_LINES,
        SLOT=4 * _LINES * chunks,
        MAXIT=_MAXIT,
        NORM=0,
        HIDDEN=0,
        GEMMA=False,
        EPS=0.0,
        num_warps=1,
    )
    return y


def all_reduce_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    quant: bool,
    group,
    inboxes,
    calls,
):
    """
    Returns RMSNorm(residual_out) and residual_out = all_reduce(x) + residual over
    the 4 ranks of ``group`` on gfx1250, with one launch of the all-reduce kernel.
    For hidden 6144, the bits are those of ``all_reduce`` followed by
    ``aiter.add_rmsnorm`` (or ``aiter.add_rmsnorm_quant(..., round_before_quant=True)``
    with quant), for finite values that are not fp32 subnormals; other sizes add the
    squares in another order. All ranks must make the same calls in the same order.
    HIP graphs are supported.

    Args:
        x, residual (torch.Tensor): Contiguous bf16 (tokens, hidden), 16-byte
            aligned, with norm_waves(hidden) > 0: hidden is a multiple of 8, at most
            24576.
        weight (torch.Tensor): Contiguous bf16 (hidden,). With gemma, the norm uses
            1 + weight.
        eps (float): The epsilon of the norm. Each value compiles a kernel.
        quant (bool): Return out as per-token FP8 (float8_e4m3fn) and also its fp32
            scale of shape (tokens, 1). Else out is bf16.
        inboxes, calls: The result of ``all_reduce_init``, shared with ``all_reduce``.

    Returns:
        tuple: (out, residual_out), or (out, residual_out, scale) with quant. If a
        poll times out, the results are wrong and calls[-1] is more than 0.
    """
    tokens, hidden = x.shape
    waves = norm_waves(hidden)
    assert waves and tokens > 0
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert residual.shape == x.shape and weight.shape == (hidden,)
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    for t in (x, residual, weight):
        assert t.data_ptr() % 16 == 0
    chunks = (calls.numel() - 1) // 2
    # A row is whole chunks; its last chunk can be partial.
    num_chunks = tokens * triton.cdiv(hidden, 6 * _LINES)
    assert num_chunks <= chunks

    _LOGGER.info("ALL_REDUCE_RMSNORM [gluon]: x=%s quant=%s", x.shape, quant)

    residual_out = torch.empty_like(x)
    if quant:
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((tokens, 1), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty_like(x)
        scale = residual_out  # not used without quant
    _all_reduce_kernel[(triton.cdiv(num_chunks, waves) + tokens,)](
        x,
        residual_out,
        inboxes,
        calls,
        calls[-1:],
        x.numel(),
        tokens,
        residual,
        weight,
        out,
        scale,
        RANK=dist.get_rank(group),
        LINES=_LINES,
        SLOT=4 * _LINES * chunks,
        MAXIT=_MAXIT,
        NORM=2 if quant else 1,
        HIDDEN=hidden,
        GEMMA=bool(gemma),
        EPS=float(eps),
        num_warps=waves,
        # No fma: the HIP kernel rounds each square before it adds it.
        enable_fp_fusion=False,
    )
    if quant:
        return out, residual_out, scale
    return out, residual_out
