import ctypes

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _all_reduce_kernel,
    _all_reduce_rsag_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_MAXIT = 20_000_000  # poll limit, some seconds: a peer can still be compiling
_VARIANTS = {}
_CONFIGS = {}
_POOLS = {}  # the IPC pool of each inbox, by its address: the pool owns the inbox


def _get_config():
    path = resolve_config_dir("comms", "ALL-REDUCE", backend="gluon")
    return load_config_json(f"{path}/DEFAULT.json").copy()


def all_reduce_init(group, max_bytes, config=None):
    """
    Allocates the inbox of this rank and maps the inboxes of the 4 ranks of
    ``group``, for tensors of at most ``max_bytes``. All ranks must call it.
    Needs ROCm 7.14 or later: the inboxes are shared with hipIpc.
    The default config selects variants by message size. Each variant has separate
    IPC banks and counters, so mixed-size HIP graphs can reuse the communicator.
    An explicit config fixes one variant; all ranks must use the same config.

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
    use_variants = config is None
    config = _get_config() if config is None else config.copy()
    lines = config["LINES"]
    assert lines >= 32 and lines & (lines - 1) == 0
    assert config["num_warps"] in (1, 2, 4, 8)
    assert config["NUM_PROGRAMS"] > 0
    # The config is fixed for this communicator: chunk counters and slots depend on it.
    chunks = triton.cdiv(max_bytes, 8 * lines)
    size = (
        (2 if config["RSAG"] else 1) * 2 * 4 * chunks * lines * 16
    )  # 2 banks x 4 senders
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
    _CONFIGS[inboxes[dist.get_rank(group)]] = config
    calls = torch.zeros(2 * chunks + 1, dtype=torch.int32, device=device)
    key = inboxes[dist.get_rank(group)]
    _VARIANTS[key] = []
    if use_variants:
        variants = config["VARIANTS"]
        for i, variant in enumerate(variants):
            minimum = variant["MIN_NUMEL"]
            if 2 * minimum > max_bytes:
                break
            capacity = max_bytes
            if i + 1 < len(variants):
                capacity = min(capacity, 2 * variants[i + 1]["MIN_NUMEL"])
            peer_inboxes, peer_calls = all_reduce_init(
                group, capacity, variant["config"]
            )
            _VARIANTS[key].append((minimum, peer_inboxes, peer_calls))
    return inboxes, calls


def all_reduce_free(group, inboxes):
    """Unmaps the peer inboxes and frees the inbox of this rank. All ranks must
    call it."""
    for _, peer_inboxes, _ in _VARIANTS.pop(inboxes[dist.get_rank(group)]):
        all_reduce_free(group, peer_inboxes)
    pool = _POOLS.pop(inboxes[dist.get_rank(group)])
    _CONFIGS.pop(inboxes[dist.get_rank(group)])
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
    rank = dist.get_rank(group)
    key = inboxes[rank]
    n = x.numel()
    error_counter = calls[-1:]
    for minimum, peer_inboxes, peer_calls in _VARIANTS[key]:
        if n >= minimum:
            inboxes, calls = peer_inboxes, peer_calls
    config = _CONFIGS[inboxes[rank]]
    lines = config["LINES"]
    chunks = (calls.numel() - 1) // 2
    assert x.data_ptr() % 16 == 0 and n <= 6 * lines * chunks

    _LOGGER.info("ALL_REDUCE [gluon]: x=%s", x.shape)

    y = torch.empty_like(x)
    num_chunks = triton.cdiv(n, 6 * lines)  # 6 values a line
    programs = min(num_chunks, config["NUM_PROGRAMS"])
    kernel = _all_reduce_kernel
    grid = num_chunks + programs
    if config["RSAG"]:
        kernel = _all_reduce_rsag_kernel
        owned = triton.cdiv(num_chunks - rank, 4)
        grid = num_chunks + owned + programs
    kernel[(grid,)](
        x,
        y,
        inboxes,
        calls,
        error_counter,
        n,
        programs,
        RANK=rank,
        LINES=lines,
        SLOT=4 * lines * chunks,
        MAXIT=_MAXIT,
        num_warps=config["num_warps"],
    )
    return y
