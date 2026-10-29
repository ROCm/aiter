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
        RANK=dist.get_rank(group),
        LINES=_LINES,
        SLOT=4 * _LINES * chunks,
        MAXIT=_MAXIT,
        num_warps=1,
    )
    return y
