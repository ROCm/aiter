import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import torch

from aiter.ops.triton.utils._triton import arch_info

_DEFAULT_TARGET_ROT_POOL_SZ_MB_BY_ARCH = {
    "gfx950": 256.0,
    "gfx942": 256.0,
    "gfx1250": 196.0,
}


def _run_input_generator(
    gen: Callable[[], tuple], fields: Sequence[str]
) -> dict[str, Any]:
    values = gen()
    if len(values) != len(fields):
        raise ValueError(
            f"{gen!r} returned {len(values)} values, expected {len(fields)}: {fields}"
        )
    return {name: val for name, val in zip(fields, values) if name != "_"}


def _count_tensor_bytes(v, seen_ptrs: set, total: int) -> int:
    if isinstance(v, torch.Tensor):
        storage = v.untyped_storage()
        ptr = storage.data_ptr()
        # Aliased views (e.g. a reshape of another returned tensor) share
        # backing storage; count it once, by its real size, not once per
        # view, or pool sizing undercounts/overcounts the true footprint.
        if ptr in seen_ptrs:
            return total
        seen_ptrs.add(ptr)
        return total + storage.nbytes()
    elif isinstance(v, dict):
        for item in v.values():
            total = _count_tensor_bytes(item, seen_ptrs, total)
        return total
    elif isinstance(v, (list, tuple)):
        # A list/tuple/dict of tensors (e.g. one entry per candidate) is
        # real backing memory too -- walk it instead of skipping it, or
        # sizing undercounts and pool_len blows up to cover a
        # phantom-small entry.
        for item in v:
            total = _count_tensor_bytes(item, seen_ptrs, total)
        return total
    return total


def _rotating_buffers_fns(
    fn: Callable, rot_bufs: list[dict[str, torch.Tensor]], n: int
):
    """``n`` callables that rotate through ``rot_bufs`` in a shuffled order.

    The shuffle keeps a full cycle hitting every pool entry once, but not in
    the same fixed sequential order every capture -- useful when per-set
    sizes are small relative to the overall pool target size.
    """
    pool_len = len(rot_bufs)
    order = list(range(pool_len))
    random.Random(0).shuffle(order)

    def _make_fn(idx):
        return lambda: fn(**(rot_bufs[order[idx % pool_len]]))

    return [_make_fn(i) for i in range(n)]


class RotatingBufferBench:
    """Builder-style: build_pool() constructs and stores a rotating-buffer
    pool; bench()/bench_synced() time a callable against it. Each method
    takes optional overrides only for the config it owns; omit to keep
    whatever __init__/from_dict set, or a previous call left in place.
    """

    def __init__(self):
        # Pool-building defaults (owned by build_pool)
        self.min_pool_len = 4
        self.max_pool_len = 1000
        self.target_rot_pool_sz_mb = self._default_target_rot_pool_sz_mb()
        self.total_pool_sz_gb = self._default_total_pool_sz_gb()
        self.print_info = False
        # Timing defaults (owned by bench/bench_synced)
        self.warmup_iters = 3
        self.warmup_replays = 3
        self.kernels_per_graph = 1000
        self.num_replays = 10
        # Set by build_pool(), consumed by bench()/bench_synced()
        self.rot_bufs: list[dict[str, torch.Tensor]] | None = None

    @staticmethod
    def _default_target_rot_pool_sz_mb() -> float:
        """How much distinct memory the rotating pool should span, so
        replaying through it defeats cache residency instead of timing a
        resident buffer repeatedly. This is a manually chosen value per
        supported arch instead because there is no way to directly query
        these values"""
        arch = arch_info.get_arch()
        return _DEFAULT_TARGET_ROT_POOL_SZ_MB_BY_ARCH.get(arch, 256.0)

    @staticmethod
    def _default_total_pool_sz_gb(safety: float = 0.9) -> float:
        """Cap on total pool memory: a safety-margined fraction of currently
        free device memory, in GB (fractional allowed), so sizing never aims
        past what's actually available right now (this process's own prior
        allocations included)."""
        free_bytes, _ = torch.cuda.mem_get_info()
        return safety * free_bytes / (1024**3)

    @classmethod
    def from_dict(cls, config: dict) -> "RotatingBufferBench":
        """__init__'s defaults, overridden by whichever keys ``config``
        provides (a ``None`` value means "not specified", keep the default).
        ``config`` is typically
        ``get_rot_bufs_kwargs(args) | get_do_bench_graph_kwargs(args)``."""
        bencher = cls()
        for key, value in config.items():
            if value is not None:
                setattr(bencher, key, value)
        return bencher

    def build_pool(
        self,
        inp_gen_rot_fns: Sequence[Callable],
        inp_rot_keys: Sequence[Sequence[str]],
        inp_gen_fixed_fns: Sequence[Callable] | None = None,
        inp_fixed_keys: Sequence[Sequence[str]] | None = None,
        print_info: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Build a pool of per-call kwargs, plus an optional fixed (built-once) set.

        ``inp_gen_rot_fns``/``inp_rot_keys`` are the rotating side: each gen
        runs once per pool entry (``pool_len`` times total), so replaying
        through the pool during timing defeats cache residency instead of
        timing one fixed buffer repeatedly. ``target_rot_pool_sz_mb``/
        ``pool_len`` sizing is computed from their bytes alone. There is no
        exclude-from-sizing escape hatch: if a kernel does not read a field
        at all, do not generate it here -- write (or wrap) the generator so
        it only returns what that call path actually uses.

        ``inp_gen_fixed_fns``/``inp_fixed_keys`` are the fixed side: each gen
        runs exactly once, and its result is merged into every pool entry by
        reference -- the same tensors on every call, not regenerated per
        entry. Use this for an input a kernel reads through a small index
        gather out of a much larger structure (e.g. a paged KV cache read
        through topk indices): build one cache sized for the gather
        diversity you want, instead of paying for ``pool_len`` redundant
        full-size copies of mostly bytes the kernel never touches. Fixed
        fields never count toward sizing -- their cost is paid once, not per
        entry, so target/pool_len math does not apply to them; size them
        directly in the fixed gen itself.

        A fixed field and a rotating field cannot share a name: the merge
        order (fixed first, then rotating) would silently let one clobber
        the other.

        ``min_pool_len``/``max_pool_len``/``target_rot_pool_sz_mb``/
        ``total_pool_sz_gb`` are fixed for this instance's lifetime --
        set them via ``__init__``/``from_dict`` before the first
        ``build_pool()`` call in a shape sweep, not per call, or an
        override meant for one shape would silently persist onto every
        later shape that omits it. ``print_info`` given here overrides
        (persists onto) this instance's value; the rest stay whatever
        __init__/from_dict set or a previous call left in place.

        Builds and stores ``self.rot_bufs``; also returns it, since callers
        need direct pool access for metrics/correctness checks.
        """
        if print_info is not None:
            self.print_info = print_info

        if self.total_pool_sz_gb * 1024 < self.target_rot_pool_sz_mb:
            raise ValueError(
                f"total_pool_sz_gb ({self.total_pool_sz_gb:.3f} GB = "
                f"{self.total_pool_sz_gb * 1024:.1f} MB) must be >= "
                f"target_rot_pool_sz_mb ({self.target_rot_pool_sz_mb:.1f} MB)"
            )

        inp_gen_fixed_fns = list(inp_gen_fixed_fns or ())
        inp_fixed_keys = list(inp_fixed_keys or ())

        def _field_names(vals_keys):
            return {name for fields in vals_keys for name in fields if name != "_"}

        rotating_names = _field_names(inp_rot_keys)
        fixed_names = _field_names(inp_fixed_keys)
        overlap = rotating_names & fixed_names
        if overlap:
            raise ValueError(
                f"fields cannot be both rotating and fixed: {sorted(overlap)}"
            )

        rot_per_set_bytes = 0
        fixed_bytes = 0
        seen_rot_ptrs = set()
        seen_fixed_ptrs = set()
        # Keep every generator's output alive for the whole sizing loop below: once
        # a kwargs dict is dropped, its tensors' storages can be freed and the
        # caching allocator can hand the same address to a later gen's unrelated
        # tensor, which would then look like an already-counted alias and get
        # skipped. Release these extra references once sizing is done, before
        # building the real pool.
        keepalive = []

        fixed_entry = {}
        for gen, fields in zip(inp_gen_fixed_fns, inp_fixed_keys):
            fixed_entry.update(_run_input_generator(gen, fields))
        for v in fixed_entry.values():
            fixed_bytes = _count_tensor_bytes(v, seen_fixed_ptrs, fixed_bytes)

        for gen, fields in zip(inp_gen_rot_fns, inp_rot_keys):
            kwargs = _run_input_generator(gen, fields)
            keepalive.append(kwargs)
            for v in kwargs.values():
                rot_per_set_bytes = _count_tensor_bytes(
                    v, seen_rot_ptrs, rot_per_set_bytes
                )
        del keepalive

        if rot_per_set_bytes == 0:
            raise ValueError(
                "rot_per_set_bytes=0, specify rotating gens that generate tensors "
                "with size > 0 (fixed gens never count toward sizing)"
            )

        del kwargs, v

        pool_len = max(
            self.min_pool_len,
            math.ceil(self.target_rot_pool_sz_mb * 1024 * 1024 / rot_per_set_bytes),
        )

        # max_pool_len is needed in case the size of input set is small, then covering
        # target_rot_pool_sz_mb might create pool_len in order of millions. This will
        # require bench() to capture one kernel launch per pool entry i.e. a long
        # graph capture. So, we need to bound this.
        pool_len = min(pool_len, max(self.max_pool_len, self.min_pool_len))

        # Bound total real memory (fixed once + pool_len rotating copies) by the
        # configured budget, so a generous target (or a large fixed set) cannot
        # try to allocate past what's actually available.
        max_gpu_mem_mb = self.total_pool_sz_gb * 1024
        fixed_mb = fixed_bytes / (1024 * 1024)
        remaining_mb = max_gpu_mem_mb - fixed_mb
        if remaining_mb <= 0:
            raise ValueError(
                f"fixed fields alone ({fixed_mb:.1f} MB) exceed the "
                f"total_pool_sz_gb budget ({self.total_pool_sz_gb:.3f} GB); raise "
                "total_pool_sz_gb or shrink inp_gen_fixed_fns"
            )
        pool_len = min(
            pool_len,
            max(self.min_pool_len, int(remaining_mb * 1024 * 1024 / rot_per_set_bytes)),
        )

        pool = []
        for _ in range(pool_len):
            entry = dict(fixed_entry)
            for gen, fields in zip(inp_gen_rot_fns, inp_rot_keys):
                entry.update(_run_input_generator(gen, fields))
            pool.append(entry)

        if self.print_info:
            print(
                f"Rotating Buffers, fixed_sz(MB):{round(fixed_mb, 2)}"
                + f" rot_per_set_sz(MB):{round(rot_per_set_bytes / (1024 * 1024), 2)}"
                + f" target_sz(MB):{round(self.target_rot_pool_sz_mb, 2)}"
                + f" total_pool_sz(GB):{round(self.total_pool_sz_gb, 3)} pool_len:{pool_len}"
            )

        self.rot_bufs = pool
        return pool

    def bench(
        self,
        fn: Callable,
        warmup_iters: int | None = None,
        warmup_replays: int | None = None,
        kernels_per_graph: int | None = None,
        num_replays: int | None = None,
    ) -> float:
        """Rotating-buffer CUDA graph benchmark, single process.

        Captures ``max(kernels_per_graph, len(self.rot_bufs))`` calls to
        ``fn`` into one CUDA graph, rotating through ``self.rot_bufs`` so the
        whole pool is covered -- a CUDA graph replays the exact calls
        captured, so a pool bigger than ``kernels_per_graph`` would
        otherwise leave entries unused. Returns the mean per-kernel latency
        in ms.

        Any of warmup_iters/warmup_replays/kernels_per_graph/num_replays
        given here overrides (persists onto) this instance's value.
        """
        if self.rot_bufs is None:
            raise RuntimeError("call build_pool() before bench()")
        if warmup_iters is not None:
            self.warmup_iters = warmup_iters
        if warmup_replays is not None:
            self.warmup_replays = warmup_replays
        if kernels_per_graph is not None:
            self.kernels_per_graph = kernels_per_graph
        if num_replays is not None:
            self.num_replays = num_replays

        rot_bufs = self.rot_bufs

        # warmup
        for _ in range(self.warmup_iters):
            fn(**(rot_bufs[0]))
        torch.cuda.synchronize()

        pool_len = len(rot_bufs)
        actual_kernels_per_graph = max(self.kernels_per_graph, pool_len)
        fns = _rotating_buffers_fns(fn, rot_bufs, actual_kernels_per_graph)

        # capture graph
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for f in fns[: min(3, len(fns))]:
                f()
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                for f in fns:
                    f()
        torch.cuda.synchronize()

        # warmup replays
        for _ in range(self.warmup_replays):
            g.replay()
        torch.cuda.synchronize()

        # Timed replays
        start_events = []
        end_events = []
        for _ in range(self.num_replays):
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            g.replay()
            ee.record()
            start_events.append(se)
            end_events.append(ee)
        torch.cuda.synchronize()

        times_ms = [se.elapsed_time(ee) for se, ee in zip(start_events, end_events)]
        per_kernel_ms = [t / actual_kernels_per_graph for t in times_ms]

        return sum(per_kernel_ms) / len(per_kernel_ms)

    def bench_synced(
        self,
        fn: Callable,
        dist_group,
        warmup_iters: int | None = None,
        warmup_replays: int | None = None,
        kernels_per_graph: int | None = None,
        num_replays: int | None = None,
    ) -> float:
        """Rotating-buffer CUDA graph benchmark, synced across ranks.

        Use this instead of :meth:`bench` when ``fn`` performs real
        cross-rank communication (e.g. a combine/all-to-all step), so every
        rank's capture and replay phases line up. ``dist_group`` is a
        ``torch.distributed`` process group; it is only a barrier at each
        phase boundary (warmup, capture, replay) -- unlike the old
        ``do_bench_cudagraph_synced``, no cross-rank latency estimate needs
        to be agreed on, because ``kernels_per_graph`` is a fixed constant
        and is therefore already identical on every rank.

        Ranks may still generate differently sized rotating buffer pools
        (e.g. differing local shapes), so growing the per-graph call count
        from the local pool size alone would desync ranks on replay. This
        method agrees the call count across ranks itself with one extra
        all-reduce before capture; callers do not need to pre-align pool
        sizes.

        The returned latency is the local rank's own measurement; callers
        that want one number across ranks must reduce it themselves (e.g.
        ``dist.all_reduce(..., op=MAX)``).

        dist_group stays a required, explicit, per-call argument (it names
        which process group this call targets, not a tunable default). Any
        of warmup_iters/warmup_replays/kernels_per_graph/num_replays given
        here overrides (persists onto) this instance's value.
        """
        import torch.distributed as dist

        if self.rot_bufs is None:
            raise RuntimeError("call build_pool() before bench_synced()")
        if warmup_iters is not None:
            self.warmup_iters = warmup_iters
        if warmup_replays is not None:
            self.warmup_replays = warmup_replays
        if kernels_per_graph is not None:
            self.kernels_per_graph = kernels_per_graph
        if num_replays is not None:
            self.num_replays = num_replays

        rot_bufs = self.rot_bufs

        def _barrier():
            dist.barrier(group=dist_group)

        # warmup
        _barrier()
        for _ in range(self.warmup_iters):
            fn(**(rot_bufs[0]))
        torch.cuda.synchronize()

        pool_len = len(rot_bufs)
        actual_kernels_per_graph = max(self.kernels_per_graph, pool_len)

        # Ranks can have differently sized pools; agree on one call count so
        # every rank captures/replays the same number of fn() calls, or
        # collectives inside fn would desync ranks whose pools differ.
        # Keep on the CPU: the all-reduce uses the gloo (CPU) group.
        agreed = torch.tensor(
            [actual_kernels_per_graph], dtype=torch.int64, device="cpu"
        )
        dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=dist_group)
        actual_kernels_per_graph = int(agreed[0])

        fns = _rotating_buffers_fns(fn, rot_bufs, actual_kernels_per_graph)

        # capture graph
        _barrier()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for f in fns[: min(3, len(fns))]:
                f()
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                for f in fns:
                    f()
        torch.cuda.synchronize()

        # warmup replays
        _barrier()
        for _ in range(self.warmup_replays):
            g.replay()
        torch.cuda.synchronize()

        # Timed replays
        start_events = []
        end_events = []
        for _ in range(self.num_replays):
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            g.replay()
            ee.record()
            start_events.append(se)
            end_events.append(ee)
        torch.cuda.synchronize()

        times_ms = [se.elapsed_time(ee) for se, ee in zip(start_events, end_events)]
        per_kernel_ms = [t / actual_kernels_per_graph for t in times_ms]

        return sum(per_kernel_ms) / len(per_kernel_ms)
