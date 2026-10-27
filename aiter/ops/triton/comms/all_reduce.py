import ctypes

import torch
import torch.distributed as dist
import triton

from aiter.dist.cuda_wrapper import find_loaded_library
from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _all_reduce_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_LINES = 64  # lines of one program: 2 for each lane of one wave
_MAXIT = 20_000_000  # poll limit, some seconds: a peer can still be compiling


class _IpcHandle(ctypes.Structure):  # hipIpcMemHandle_t
    _fields_ = [("data", ctypes.c_byte * 64)]


def _check(err):
    assert err == 0, f"HIP error {err}"


def all_reduce_init(group, max_bytes):
    """
    Allocates the inbox of this rank and maps the inboxes of the 4 ranks of
    ``group``, for tensors of at most ``max_bytes``. All ranks must call it.
    Needs ROCm 7.14 or later: the inboxes are shared with hipIpc.

    Returns:
        tuple: The inbox addresses of ranks 0 to 3, and the int32 call counters
        of this rank. calls[-1] counts the polls that timed out.
    """
    from aiter.dist.device_communicators.custom_all_reduce import _should_use_vmm

    assert get_arch() == "gfx1250" and dist.get_world_size(group) == 4
    # As in the custom all-reduce: on gfx1250, hipIpc works from ROCm 7.14.
    assert not _should_use_vmm(True), "needs ROCm 7.14 or later for hipIpc"
    hip = ctypes.CDLL(find_loaded_library("libamdhip64"))
    chunks = triton.cdiv(max_bytes, 8 * _LINES)
    size = ctypes.c_size_t(2 * 4 * chunks * _LINES * 16)  # 2 banks x 4 senders
    inbox = ctypes.c_void_p()
    handle = _IpcHandle()
    _check(hip.hipMalloc(ctypes.byref(inbox), size))
    _check(hip.hipMemset(inbox, 0, size))  # 0 is never a call number
    _check(hip.hipIpcGetMemHandle(ctypes.byref(handle), inbox))
    torch.cuda.synchronize()

    handles = [None] * 4
    dist.all_gather_object(handles, bytes(handle), group=group)
    inboxes = []
    for rank, h in enumerate(handles):
        if rank == dist.get_rank(group):
            inboxes.append(inbox.value)
            continue
        peer = ctypes.c_void_p()
        h = _IpcHandle.from_buffer_copy(h)
        _check(hip.hipIpcOpenMemHandle(ctypes.byref(peer), h, 1))
        inboxes.append(peer.value)

    device = torch.cuda.current_device()
    calls = torch.zeros(2 * chunks + 1, dtype=torch.int32, device=device)
    return tuple(inboxes), calls


def all_reduce_free(group, inboxes):
    """Unmaps the peer inboxes and frees the inbox of this rank. All ranks must
    call it."""
    hip = ctypes.CDLL(find_loaded_library("libamdhip64"))
    rank = dist.get_rank(group)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    for r, address in enumerate(inboxes):
        if r != rank:
            _check(hip.hipIpcCloseMemHandle(ctypes.c_void_p(address)))
    dist.barrier(group=group)  # no peer uses the inbox of this rank now
    _check(hip.hipFree(ctypes.c_void_p(inboxes[rank])))


def all_reduce(x: torch.Tensor, group, inboxes, calls) -> torch.Tensor:
    """
    Returns the sum of the bf16 tensor x over the 4 ranks of ``group`` on gfx1250,
    with one kernel launch. All ranks get the same bits. For small tensors: RCCL
    is faster at 1 MB and from 4 MB. All ranks must make the same calls in the
    same order. HIP graphs are supported.

    Args:
        x (torch.Tensor): Contiguous bf16, 16-byte aligned, a multiple of 4
            values and at most max_bytes.
        inboxes, calls: The result of ``all_reduce_init``. If a poll times out,
            the sum is wrong and calls[-1] is more than 0.
    """
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() % 4 == 0
    chunks = (calls.numel() - 1) // 2
    n_words = x.numel() // 2  # the kernel moves bf16 values in pairs
    assert x.data_ptr() % 16 == 0 and n_words <= 2 * _LINES * chunks

    _LOGGER.info("ALL_REDUCE [gluon]: x=%s", x.shape)

    y = torch.empty_like(x)
    _all_reduce_kernel[(2 * triton.cdiv(n_words, 2 * _LINES),)](
        x.view(-1).view(torch.int32),  # flat first: the last dim can be odd
        y.view(-1).view(torch.int32),
        inboxes,
        calls,
        calls[-1:],
        n_words,
        RANK=dist.get_rank(group),
        LINES=_LINES,
        SLOT=4 * _LINES * chunks,
        MAXIT=_MAXIT,
        num_warps=1,
    )
    return y
