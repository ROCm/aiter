# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared assets for all-reduce host wrappers."""

import ctypes
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from .allreduce_policy import (
    INBOX_MEMORY_MODES,
    MAX_PAYLOAD_BYTES,
    PAYLOAD_ALIGN_BYTES,
    SUPPORTED_ARCHS,
    SUPPORTED_WORLDS,
    default_inbox_memory,
    resolve_link,
)
from .ipc_heap import UncachedIpcHeap
from .kernels.tensor_shim import _preload_compiled

logger = logging.getLogger("aiter")


def _cuda_index(device, *, owner: str) -> int:
    """The CUDA device index of *device*. *owner* names the caller in errors."""
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"{owner} requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


def _resolve_inbox_flags(
    mode: str, world_size: int, link: str | None = None
) -> tuple[int, str]:
    """(hipExtMallocWithFlags mode, resolved name) for an ``inbox_memory``.

    ``"auto"`` resolves to ``default_inbox_memory``.
    """
    if mode not in INBOX_MEMORY_MODES:
        raise ValueError(
            f"inbox_memory must be one of {INBOX_MEMORY_MODES}, got {mode!r}"
        )
    if mode == "auto":
        mode = default_inbox_memory(world_size, resolve_link(link))
    flags = {
        "uncached": UncachedIpcHeap._HIP_DEVICE_MALLOC_UNCACHED,
        "finegrained": UncachedIpcHeap._HIP_DEVICE_MALLOC_FINEGRAINED,
    }[mode]
    return flags, mode


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(
        self,
        *,
        spec,
        group,
        rank: int,
        world_size: int,
        inbox_flags: int,
    ):
        self.spec = spec
        self.launch = spec["launch"]
        self.super_tile = spec["super_tile"]
        self.grid = spec["grid"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.block = spec["block"]
        self._peer_bases = [None] * world_size
        self._buf_ptr = None
        self._meta_ptr = None
        self._gpu_peer_ptrs = None
        self._colors = None
        # Split-build exchange state inside the meta block; 0 when there is none.
        self._xchg = 0
        # Setup has two collectives, the handle exchange and a final agreement.
        # A rank that fails must still take part in both, and report the
        # failure through them. If it raised at once, its peers would wait in a
        # collective that it never joins, and they would hang. The error is
        # raised on every rank after the agreement.
        error = None
        try:
            # The inbox is the only allocation peers write into, so it is the
            # only one whose memory type matters for fabric throughput.
            self._buf_ptr = UncachedIpcHeap.alloc(
                self.buf_bytes, inbox_flags, fill=spec.get("inbox_fill", 0)
            )
            my_meta = (UncachedIpcHeap.get_mem_handle_bytes(self._buf_ptr), 0)
        except Exception as e:  # noqa: BLE001
            error = e
            my_meta = (None, 0)
        try:
            all_meta = UncachedIpcHeap.gather_object_list_via_broadcast(group, my_meta)
        except Exception:
            self.close()
            raise

        if error is None:
            try:
                if any(handle is None for handle, _off in all_meta):
                    raise RuntimeError(
                        "a peer rank failed to allocate its inbox, which has "
                        f"{self.buf_bytes} B"
                    )
                self._open_peers(all_meta, rank=rank, world_size=world_size)
            except Exception as e:  # noqa: BLE001
                error = e

        ok = torch.tensor([0 if error is not None else 1], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if error is not None or not ok.item():
            self.close()
            raise error or RuntimeError("a peer rank failed to set up its engine")

    def _open_peers(self, all_meta, *, rank: int, world_size: int):
        """Open the peers' inboxes and upload the pointer table and the colours."""
        peer_ptrs = [0] * world_size
        for r in range(world_size):
            handle, off = all_meta[r]
            if r == rank:
                peer_ptrs[r] = self._buf_ptr + off
            else:
                base = int(UncachedIpcHeap.open_mem_handle(bytes(handle)))
                self._peer_bases[r] = base
                peer_ptrs[r] = base + off

        peer_bytes = world_size * 8
        color_bytes = self.grid * 4
        # Peer-pointer table and per-block colours: written by the host once
        # and by this rank's own kernel, never by a peer. Stays uncached in
        # every mode -- no cross-GPU visibility question, and it is a few
        # KiB.
        #
        # A split fused one-shot's exchange words and per-workgroup ``prev``
        # values follow, line-aligned after the colours. Like them, only this
        # rank's own kernel touches it; ``alloc`` zeroes it, and zero words with
        # zero ``prev`` are a consistent start.
        xchg_bytes = int(self.spec.get("xchg_bytes", 0))
        xchg_off = (peer_bytes + color_bytes + 127) // 128 * 128
        meta_bytes = xchg_off + xchg_bytes if xchg_bytes else peer_bytes + color_bytes
        self._meta_ptr = UncachedIpcHeap.alloc_uncached(meta_bytes)
        self._gpu_peer_ptrs = self._meta_ptr
        self._colors = self._meta_ptr + peer_bytes
        UncachedIpcHeap.copy_host_to_device(
            self._gpu_peer_ptrs,
            (ctypes.c_int64 * world_size)(*peer_ptrs),
            peer_bytes,
        )
        UncachedIpcHeap.copy_host_to_device(
            self._colors,
            (ctypes.c_int32 * self.grid)(*([1] * self.grid)),
            color_bytes,
        )
        if xchg_bytes:
            self._xchg = self._meta_ptr + xchg_off

    def close(self):
        for b in self._peer_bases:
            if b is not None:
                try:
                    UncachedIpcHeap.close_mem_handle(int(b))
                except RuntimeError:
                    pass
        self._peer_bases = []
        if self._meta_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._meta_ptr)
            except RuntimeError:
                pass
            self._meta_ptr = None
            self._gpu_peer_ptrs = None
            self._colors = None
            self._xchg = 0
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:
                pass
            self._buf_ptr = None


def check_fused_aliasing(inp, residual_in, out, residual_out, live_bytes: int) -> None:
    """Raise unless the fused call's destinations are safe to write.

    ``out`` and ``residual_out`` are written, ``inp`` and ``residual_in`` are
    only read. Two ways to overlap are allowed:

    * ``inp`` with ``residual_in``: both are read-only.
    * ``residual_out`` is ``residual_in``: each thread reads
      its residual atoms before it stores the sum back to the same bytes.
    """

    def span(t):
        p = int(t.data_ptr())
        return p, p + live_bytes

    def overlap(a, b):
        (a0, a1), (b0, b1) = span(a), span(b)
        return max(a0, b0) < min(a1, b1)

    in_place = residual_out.data_ptr() == residual_in.data_ptr()
    for wname, w in (("out", out), ("residual_out", residual_out)):
        for rname, r in (("inp", inp), ("residual_in", residual_in)):
            if in_place and wname == "residual_out" and rname == "residual_in":
                continue
            if overlap(w, r):
                raise ValueError(
                    f"{wname} overlaps {rname}; only residual_out aliasing "
                    "residual_in exactly is allowed"
                )
    if overlap(out, residual_out):
        raise ValueError("out and residual_out overlap")


def require_arch(*, owner: str) -> str:
    """This device's ``gfx`` arch. Raises if the kernels do not support it."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch not in SUPPORTED_ARCHS:
        raise RuntimeError(f"{owner} supports {', '.join(SUPPORTED_ARCHS)}, got {arch}")
    return arch


def check_world_size(world_size: int) -> None:
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )


def check_payload(inp, out, *, device_index: int, owner: str) -> int:
    """The payload size of an all-reduce of *inp* into *out*, in bytes.

    Raises unless both are contiguous bf16 CUDA tensors on *device_index*, of
    the same size, aligned, not overlapping, and within ``MAX_PAYLOAD_BYTES``.
    """
    align = PAYLOAD_ALIGN_BYTES
    if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
        raise TypeError(f"{owner} requires torch.Tensor input/output")
    if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"{owner} supports bf16 input/output")
    if not inp.is_cuda or not out.is_cuda:
        raise ValueError(f"{owner} requires CUDA tensors")
    if inp.device.index != device_index or out.device.index != device_index:
        raise ValueError(
            f"inp/out must be on cuda:{device_index}, got {inp.device} / {out.device}"
        )
    if not inp.is_contiguous() or not out.is_contiguous():
        raise ValueError(f"{owner} requires contiguous input/output")
    inp_ptr = int(inp.data_ptr())
    out_ptr = int(out.data_ptr())
    if inp_ptr % align != 0 or out_ptr % align != 0:
        raise ValueError(f"{owner} requires {align}-byte-aligned input/output")
    live_bytes = int(inp.numel()) * int(inp.element_size())
    if live_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError(f"{owner} payload must not exceed the 4 GiB buffer window")
    if live_bytes % align != 0:
        raise ValueError(f"byte size must be a multiple of {align} ({align // 2} bf16)")
    if int(out.numel()) * int(out.element_size()) != live_bytes:
        raise ValueError("inp/out byte size mismatch")
    if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
        raise ValueError(f"{owner} requires non-overlapping input/output")
    return live_bytes


def eligible_payload_bytes(inp, *, device: torch.device) -> int | None:
    """The payload size of *inp* in bytes if ``check_payload`` accepts it as the
    input of an all-reduce on *device*, otherwise None. Never raises.

    The dispatchers call this before they pick a FlyDSL engine. The output
    checks of ``check_payload`` do not apply: the dispatchers allocate ``out``.
    """
    align = PAYLOAD_ALIGN_BYTES
    if not isinstance(inp, torch.Tensor) or inp.dtype is not torch.bfloat16:
        return None
    if inp.device != device:
        return None
    if not inp.is_contiguous() or int(inp.data_ptr()) % align:
        return None
    nbytes = int(inp.numel()) * int(inp.element_size())
    if nbytes % align or nbytes > MAX_PAYLOAD_BYTES:
        return None
    return nbytes


def _validate_ipc_process_group(
    group, *, rank: int, world_size: int, owner: str
) -> None:
    """Reject groups that do not match *rank* and *world_size*, or that cannot
    exchange HIP IPC handles or CPU-side metadata."""
    # Keep parallel_state lazy: this module is imported while aiter's AOT setup
    # is still initializing the top-level package.
    from aiter.dist.parallel_state import in_the_same_node_as

    group_world = dist.get_world_size(group=group)
    group_rank = dist.get_rank(group=group)
    if group_world != int(world_size):
        raise ValueError(
            f"world_size={world_size} does not match group size {group_world}"
        )
    if group_rank != int(rank):
        raise ValueError(f"rank={rank} does not match group rank {group_rank}")

    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        raise ValueError(
            f"{owner} does not support NCCL process groups (got "
            f"{backend!r} on group rank {rank}): IPC handle exchange requires "
            "CPU-side broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            f"{owner} does not support multi-node process groups: HIP "
            f"IPC handles are node-local (ranks not on rank 0's node: {off_node})."
        )


# KFD io-link type for xGMI, from include/uapi/linux/kfd_sysfs.h. PCIe is 2.
_HSA_IOLINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def probe_xgmi_peer_links() -> bool | None:
    """Whether any GPU-to-GPU link on this host is xGMI rather than PCIe, from
    the KFD topology. None when the topology cannot be read.

    Arch is not enough to make this call: an MI350X (xGMI) and an MI350P
    (PCIe-only) both report ``gfx950``, and the right inbox memory type is
    opposite on the two. KFD exposes the real link type per peer pair, so read
    that instead of guessing from the SKU.

    Both ``io_links`` and ``p2p_links`` have to be scanned. KFD only populates
    ``p2p_links`` for peers reachable indirectly (through a host bridge), so on
    a directly-connected mesh it holds nothing but the PCIe links to the CPU
    nodes and the xGMI peers appear solely under ``io_links``. Reading
    ``p2p_links`` alone reports "PCIe" on an 8-GPU all-xGMI MI350X, which flips
    ``inbox_memory="auto"`` to the fine-grained heap and silently costs the
    uncached fanout the kernel was designed around.

    The function returns None, not False, when it finds no link ``properties``
    file at all or cannot read one.

    The answer is per host, not per process group: one xGMI link anywhere
    classifies every group on the node as xGMI. That assumes a node is uniformly
    xGMI or uniformly PCIe, which holds for the single-node systems this targets.
    """
    try:
        found = False
        for _src, _dst, props in kfd_links():
            found = True
            if props.get("type") == _HSA_IOLINK_TYPE_XGMI:
                return True
        return False if found else None
    except (OSError, ValueError, KeyError):
        logger.debug("FlyDSL all-reduce: cannot read KFD topology", exc_info=True)
        return None


def kfd_props(path: Path) -> dict[str, int]:
    """The integer fields of one KFD ``properties`` file."""
    out = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(" ")
        try:
            out[key] = int(value)
        except ValueError:
            continue
    return out


def kfd_links(root: Path = _KFD_NODES):
    """Every KFD link as ``(node_from, node_to, props)``, from both
    ``io_links`` and ``p2p_links``. Raises OSError or ValueError if 
    the topology cannot be read."""
    for subdir in ("io_links", "p2p_links"):
        for props_file in sorted(Path(root).glob(f"*/{subdir}/*/properties")):
            props = kfd_props(props_file)
            yield props["node_from"], props["node_to"], props


def kfd_node_of_device(device_index: int, root: Path = _KFD_NODES) -> int | None:
    """The KFD topology node of HIP device *device_index*, matched on its PCI
    domain and location (``bus << 8 | device << 3``, function 0). None if no
    node matches or the topology cannot be read.

    torch reports the physical PCI address, so this holds whatever
    ``HIP_VISIBLE_DEVICES`` selects.
    """
    try:
        prop = torch.cuda.get_device_properties(int(device_index))
        location = (int(prop.pci_bus_id) << 8) | (int(prop.pci_device_id) << 3)
        domain = int(prop.pci_domain_id)
    except (RuntimeError, AttributeError, AssertionError):
        logger.debug("FlyDSL topology: no PCI address for the device", exc_info=True)
        return None
    try:
        for node in Path(root).iterdir():
            if not node.name.isdigit():
                continue
            props = kfd_props(node / "properties")
            if props.get("location_id") == location and props.get("domain") == domain:
                return int(node.name)
    except (OSError, ValueError):
        logger.debug("FlyDSL topology: cannot read KFD nodes", exc_info=True)
    return None


def kfd_link_weights(
    nodes: list[int], root: Path = _KFD_NODES
) -> list[list[int]] | None:
    """KFD link weights between *nodes*: ``[i][j]`` is the weight of the link
    from ``nodes[i]`` to ``nodes[j]``, 0 on the diagonal. A lower weight is a 
    closer peer. None if a pair has no link or the topology cannot be read."""
    index = {int(n): i for i, n in enumerate(nodes)}
    n = len(nodes)
    weights = [[0 if i == j else None for j in range(n)] for i in range(n)]
    try:
        for src, dst, props in kfd_links(root):
            if src in index and dst in index and src != dst:
                weights[index[src]][index[dst]] = int(props.get("weight", 0))
    except (OSError, ValueError, KeyError):
        logger.debug("FlyDSL topology: cannot read KFD links", exc_info=True)
        return None
    if any(w is None for row in weights for w in row):
        return None
    return weights


def socket_groups(weights) -> tuple[tuple[int, ...], ...]:
    """The groups of ranks that *weights* splits into, or one group if it does
    not split uniformly.

    Ranks joined by the lowest off-diagonal weight form a group. The split is
    accepted only if there are at least two groups, all of the same size of at
    least two, every pair inside a group has that lowest weight, and every pair
    across groups has one same, higher weight. Anything else -- a flat mesh, or
    a host that is not symmetric -- is one group.
    """
    n = len(weights)
    flat = (tuple(range(n)),)
    if n < 4:
        return flat
    off = [weights[i][j] for i in range(n) for j in range(n) if i != j]
    near = min(off)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(n):
            if i != j and weights[i][j] == near:
                parent[find(i)] = find(j)
    by_root: dict[int, list[int]] = {}
    for i in range(n):
        by_root.setdefault(find(i), []).append(i)
    groups = tuple(sorted(tuple(sorted(g)) for g in by_root.values()))
    if len(groups) < 2 or len({len(g) for g in groups}) != 1 or len(groups[0]) < 2:
        return flat
    group_of = {r: gi for gi, g in enumerate(groups) for r in g}
    far = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if group_of[i] == group_of[j]:
                if weights[i][j] != near:
                    return flat
            else:
                far.add(weights[i][j])
    if len(far) != 1:
        return flat
    return groups


GROUPS_ENV = "AITER_FLY_AR_GROUPS"


def parse_groups_env(text: str, world_size: int) -> tuple[tuple[int, ...], ...]:
    """Groups from ``AITER_FLY_AR_GROUPS``, e.g. ``"0,1,2,3;4,5,6,7"``."""
    try:
        groups = tuple(
            tuple(sorted(int(r) for r in part.split(",") if r.strip()))
            for part in text.split(";")
            if part.strip()
        )
    except ValueError as exc:
        raise ValueError(
            f"{GROUPS_ENV}={text!r} is not a ';'-list of ','-lists"
        ) from exc
    if sorted(r for g in groups for r in g) != list(range(int(world_size))):
        raise ValueError(
            f"{GROUPS_ENV}={text!r} must hold every rank of world_size={world_size} once"
        )
    return tuple(sorted(groups))


@dataclass(frozen=True)
class PeerTopology:
    """How the ranks of one process group sit on the host's fabric.

    ``weights[i][j]`` is the KFD link weight from rank ``i``'s GPU to rank
    ``j``'s (None when it was not read, e.g. under an override). ``groups`` are
    the ranks of each socket group (sorted). A single group means the host is
    flat for this process group.
    """

    groups: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[int, ...], ...] | None = None
    kfd_nodes: tuple[int, ...] | None = None
    source: str = "kfd"

    @property
    def is_hierarchical(self) -> bool:
        return len(self.groups) >= 2

    def group_of(self, rank: int) -> int:
        return next(gi for gi, g in enumerate(self.groups) if int(rank) in g)

    def local_index(self, rank: int) -> int:
        return self.groups[self.group_of(rank)].index(int(rank))

    def partner(self, rank: int) -> int:
        """The rank at the same local index in the other group (two groups)."""
        if len(self.groups) != 2:
            raise ValueError(f"partner needs exactly two groups, got {self.groups}")
        return self.groups[1 - self.group_of(rank)][self.local_index(rank)]


def probe_peer_topology(
    group, *, rank: int, world_size: int, device_index: int
) -> PeerTopology:
    """The ``PeerTopology`` of *group*, identical on every rank.

    A collective over *group* (which must carry CPU objects, i.e. gloo): each
    rank finds its own GPU's KFD node, the nodes are gathered, and rank 0 reads
    the link weights and broadcasts the result. ``AITER_FLY_AR_GROUPS``
    overrides the detected groups (read on rank 0). An unreadable topology is
    one flat group.
    """
    # Never raises: every rank has to reach the collectives below.
    node = kfd_node_of_device(device_index)
    nodes = IpcHeap.gather_object_list_via_broadcast(group, node)
    topo = None
    if int(rank) == 0:
        override = os.environ.get(GROUPS_ENV, "").strip()
        weights = None
        if all(n is not None for n in nodes):
            weights = kfd_link_weights([int(n) for n in nodes])
        groups, source = (tuple(range(int(world_size))),), "unknown"
        if weights is not None:
            groups, source = socket_groups(weights), "kfd"
        if override:
            try:
                groups, source = parse_groups_env(override, world_size), "env"
            except ValueError:
                # Rank 0 alone reads it; a bad value must still reach the
                # broadcast below, or the other ranks would wait in it.
                logger.warning("ignoring %s", GROUPS_ENV, exc_info=True)
        topo = PeerTopology(
            groups=groups,
            weights=None if weights is None else tuple(tuple(r) for r in weights),
            kfd_nodes=None if None in nodes else tuple(int(n) for n in nodes),
            source=source,
        )
    holder = [topo]
    src = dist.get_process_group_ranks(group=group)[0]
    dist.broadcast_object_list(holder, src=src, group=group, device="cpu")
    return holder[0]


def kernel_symbol(launch) -> str:
    """The JIT symbol a kernel factory stamped on its launch wrapper."""
    name = getattr(getattr(launch, "func", None), "__name__", None)
    if not name:
        return "?"
    return name.removeprefix("launch_")


def _launch_artifacts(launch) -> list:
    """The compiled artifacts of a ``@flyc.jit`` launch in this process."""
    cache = getattr(launch, "_mem_cache", None)
    if not isinstance(cache, dict):
        raise TypeError(
            f"cannot reach the compiled artifacts of {kernel_symbol(launch)}: "
            "this FlyDSL version has no JitFunction._mem_cache"
        )
    return list(cache.values())


def preload_launcher(launch, *args) -> None:
    """Compile launch for args and load it onto the current device,
    without launching it.

    The module loads into the current device, so the caller must make the
    engine's device current.
    """
    _preload_compiled(launch, *args)
    artifacts = _launch_artifacts(launch)
    if not artifacts:
        raise RuntimeError(f"FlyDSL compiled no artifact for {kernel_symbol(launch)}")
    for artifact in artifacts:
        # Builds the ExecutionEngine, which loads the module, on first use.
        artifact._get_func_exe()


def launcher_loaded(launch) -> bool:
    """Whether every compiled artifact of *launch* is loaded, so that its
    next launch does no JIT and no module load. False if none is compiled."""
    artifacts = _launch_artifacts(launch)
    return bool(artifacts) and all(a._engine is not None for a in artifacts)


def check_build_outside_capture(*, owner: str, hidden: int) -> None:
    """Raise if building the engines of hidden would run inside an open HIP
    graph capture.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{owner}: hidden={hidden} is not built, and building it exchanges "
            "IPC handles with every rank, which cannot run under HIP graph "
            f"capture; pass hiddens=({hidden},) to the constructor or call "
            f"preload({hidden}) on every rank before the capture"
        )


def check_launch_outside_capture(launch, *, owner: str) -> None:
    """Raise if launching *launch* would JIT-compile it or load its module
    inside an open HIP graph capture.
    """
    if not launcher_loaded(launch) and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{owner}: {kernel_symbol(launch)} is not compiled and loaded, and "
            "neither can run under HIP graph capture; call preload(hidden) on "
            "every rank before the capture"
        )


def payload_probes(floors, lo: int, hi: int) -> tuple[int, ...]:
    """Payload sizes in ``lo..hi`` bytes (inclusive) that together select every
    rung a ladder can use in that range.

    *floors* are the rungs' ``min_bytes``. The result is both ends of the range
    plus the valid sizes on either side of each floor inside it, all multiples
    of ``PAYLOAD_ALIGN_BYTES``, sorted. It is ``()`` if the range holds no valid
    size.
    """
    step = PAYLOAD_ALIGN_BYTES
    lo = max(step, -(-int(lo) // step) * step)
    hi = int(hi) // step * step
    if lo > hi:
        return ()
    probes = {lo, hi}
    for floor in floors:
        first = -(-int(floor) // step) * step
        if lo < first <= hi:
            probes.update((first - step, first))
    return tuple(sorted(probes))
