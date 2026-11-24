# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Times the RDNA attention kernels the way a server runs them.

`Attention.forward` serves one layer's call of a batch as vLLM's
ROCM_RDNA_ATTN does: uniform decodes of up to MAX_M query tokens on the decode
kernel, anything else on the prefill kernel (a batch through a plan), and
leading decodes beside prefills on both, the decodes on a side stream.

`run` times a batch spec over several layers, each with its own Q, KV cache
and output, enough of them that the KV outgrows the last-level cache: decodes
under a CUDA graph, prefills without. The tables were tuned on these numbers.

`kernels` is what picks and loads the builds: `aiter.ops.rdna_attn` (the
tables' builds) or jit.py (standalone builds with forced knobs).
"""

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from aiter.ops import rdna_attn as R
from batch_spec import parse_batch_spec

# Split-KV scratch of the decode builds, shared by the layers, by
# (variant, device): the sequences it holds and the tensors. A batch of more
# sequences than the budget holds is not served.
_SCRATCH: dict = {}
_SCRATCH_BUDGET = 64 * 1024**2
# Scratch of the prefill builds, by (shape, device).
_PREFILL_SCRATCH: dict = {}
# Side stream for a mixed batch's decodes, per device.
_SIDE_STREAMS: dict = {}


def clear_scratch() -> None:
    """Free the scratch the layers share."""
    _SCRATCH.clear()
    _PREFILL_SCRATCH.clear()


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    if device not in _SIDE_STREAMS:
        _SIDE_STREAMS[device] = torch.cuda.Stream(device)
    return _SIDE_STREAMS[device]


@dataclass
class Batch:
    """One step's batch: its requests on the host and on the device."""

    # Query starts and sequence lengths on the host, for the prefill plan.
    query_start_loc_cpu: np.ndarray
    seq_lens_cpu: np.ndarray
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    # (requests, blocks) int32.
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    # Leading decodes of one query length, and their tokens.
    num_decodes: int = 0
    num_decode_tokens: int = 0
    # (window, first sequence) -> (build, plan, head group, query starts),
    # made by the first layer and shared by the rest.
    plans: dict = field(default_factory=dict)


def split_decodes(q_lens: list[int], threshold: int) -> tuple[int, int]:
    """(leading decodes of one query length, their tokens) of a batch ordered
    decodes first; a decode has at most `threshold` query tokens."""
    first = q_lens[0]
    if max(q_lens) <= threshold <= 1:
        return len(q_lens), sum(q_lens)
    if first > threshold:
        return 0, 0
    # Padding (no query tokens) counts as a decode.
    if all(q in (first, 0) for q in q_lens):
        return len(q_lens), sum(q_lens)
    decodes = next(i for i, q in enumerate(q_lens) if q not in (first, 0))
    return decodes, sum(q_lens[:decodes])


class Attention:
    """A layer's attention on the RDNA kernels."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        window: int = 0,
        max_model_len: int = 0,
        max_num_seqs: int = 256,
        kernels: Any = R,
    ):
        self.num_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = head_size**-0.5
        self.window = window
        # A window no shorter than the longest sequence is full attention:
        # without the window's builds, full attention's serve it.
        self._window_covers_seqs = bool(window) and window >= max_model_len > 0
        self._max_seqs = max_num_seqs
        self.kernels = kernels
        self._variant = None
        self._built = None
        self._cap = 0
        # Why the last unserved call was refused.
        self.rejected: Optional[str] = None
        self.decode_calls = 0
        self.prefill_calls = 0
        self.unserved_calls = 0

    def path(self) -> str:
        """What ran so far: "kernel" (decode), "prefill" or "unserved"."""
        if self.unserved_calls or not (self.decode_calls or self.prefill_calls):
            return "unserved"
        return "prefill" if self.prefill_calls else "kernel"

    def _served_tensors(self, q, out, kv_cache, head_sizes) -> bool:
        if q.dtype not in (torch.float16, torch.bfloat16):
            self.rejected = f"kernel is fp16 or bf16 only, got {q.dtype}"
        elif not (q.is_contiguous() and out.is_contiguous()):
            self.rejected = "query or output is not contiguous"
        elif kv_cache.dtype != q.dtype:
            self.rejected = f"KV cache is {kv_cache.dtype}, query is {q.dtype}"
        elif head_sizes is not None and self.head_size not in head_sizes:
            self.rejected = f"head_size {self.head_size} not built"
        elif self.num_heads % self.num_kv_heads:
            self.rejected = "q heads must divide evenly over kv heads"
        elif kv_cache.shape[2] % 16:
            self.rejected = f"block size {kv_cache.shape[2]} is not a multiple of 16"
        else:
            return True
        return False

    def _prepare_decode(self, q, out, kv_cache, seq_lens, max_query_len):
        """The decode build and its scratch, or None."""
        nseq = seq_lens.shape[0]
        if q.shape[0] != nseq * max_query_len:
            self.rejected = "sequences of unequal query length"
            return None
        if max_query_len > R.MAX_M:
            self.rejected = f"{max_query_len} query tokens, more than {R.MAX_M}"
            return None
        if not self._served_tensors(q, out, kv_cache, R.SUPPORTED_HEAD_SIZES):
            return None
        shape = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            max_query_len,
            kv_cache.shape[2],
            0 if kv_cache.stride(1) < kv_cache.stride(2) else 1,
        )
        win = self.window
        variant = self.kernels.variant_for(*shape, win, q.dtype, nseq > 1)
        if variant is None and win and self._window_covers_seqs:
            variant = self.kernels.variant_for(*shape, 0, q.dtype, nseq > 1)
        if variant is None:
            self.rejected = "no build for this shape and layout"
            return None
        expected = R.expected_kv_cache_strides(variant)
        actual = (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2))
        if actual != expected:
            self.rejected = f"KV strides {actual} != {expected} for this layout"
            return None
        if self._variant != variant:
            module = self.kernels.load(variant)
            if module is None:
                self.rejected = f"{variant.name} does not load"
                return None
            key = (variant, q.device)
            if key not in _SCRATCH:
                cap = 1
                if variant.batched:
                    per_seq = R.scratch_bytes(variant)
                    cap = min(self._max_seqs, max(1, _SCRATCH_BUDGET // per_seq))
                _SCRATCH[key] = (
                    cap,
                    R.make_scratch(variant, q.device, cap if variant.batched else 0),
                )
            self._cap, scratch = _SCRATCH[key]
            self._built = (module, scratch)
            self._variant = variant
        if nseq > self._cap:
            self.rejected = f"{nseq} sequences, the decode scratch holds {self._cap}"
            return None
        return self._built

    def _launch_decode(self, built, q, out, kv_cache, seq_lens, block_table) -> None:
        module, scratch = built
        # seq_lens is a device tensor: CUDA graphs follow the sequence length.
        module.decode_attn(
            q, kv_cache, block_table, out, *scratch, seq_lens, self.scale
        )

    def forward(
        self, q: torch.Tensor, kv_cache: torch.Tensor, batch: Batch, out: torch.Tensor
    ) -> bool:
        """Serve the call; False if the kernels cannot (see `rejected`)."""
        n, nseq = q.shape[0], batch.seq_lens.shape[0]
        m = batch.max_query_len
        if m <= R.MAX_M and n == nseq * m:
            # Decode: every sequence the same M.
            built = self._prepare_decode(q, out, kv_cache, batch.seq_lens, m)
            if built is not None:
                self.decode_calls += 1
                self._launch_decode(
                    built, q, out, kv_cache, batch.seq_lens, batch.block_table
                )
                return True
        elif self._serve_mixed(q, out, kv_cache, batch) or self._serve_prefill(
            q, out, kv_cache, batch
        ):
            self.prefill_calls += 1
            return True
        self.unserved_calls += 1
        return False

    def _serve_mixed(self, q, out, kv_cache, batch: Batch) -> bool:
        """Leading decodes on the decode kernel, the rest on the prefill
        kernel; False if a kernel cannot take its part."""
        nd, ndt = batch.num_decodes, batch.num_decode_tokens
        nseq = batch.seq_lens.shape[0]
        if not 0 < nd < nseq or torch.cuda.is_current_stream_capturing():
            return False
        built = self._prepare_decode(
            q[:ndt], out[:ndt], kv_cache, batch.seq_lens[:nd], ndt // nd
        )
        if built is None:
            return False
        launch_prefills = self._prepare_prefill(
            q[ndt:], out[ndt:], kv_cache, batch, first=nd
        )
        if launch_prefills is None:
            return False
        # Decodes on a side stream fill the SIMDs a narrow prefill leaves.
        main = torch.cuda.current_stream()
        side = _side_stream(main.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            self._launch_decode(
                built,
                q[:ndt],
                out[:ndt],
                kv_cache,
                batch.seq_lens[:nd],
                batch.block_table[:nd],
            )
        launch_prefills()
        main.wait_stream(side)
        return True

    def _serve_prefill(self, q, out, kv_cache, batch: Batch) -> bool:
        launch = self._prepare_prefill(q, out, kv_cache, batch)
        if launch is None:
            return False
        launch()
        return True

    def _prepare_prefill(
        self, q, out, kv_cache, batch: Batch, first: int = 0
    ) -> Optional[Callable[[], None]]:
        """The prefill launch for the sequences from `first` on, or None.

        Not under CUDA-graph capture: the launch depends on the host's M and S.
        """
        if torch.cuda.is_current_stream_capturing():
            self.rejected = "prefill under CUDA-graph capture"
            return None
        if not self._served_tensors(q, out, kv_cache, None):
            return None
        block_size = kv_cache.shape[2]
        kv_row = 2 * self.head_size
        hnd = (block_size * self.num_kv_heads * kv_row, block_size * kv_row, kv_row)
        if (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2)) != hnd:
            self.rejected = "prefill: KV cache is not the packed HND layout"
            return None
        hq, hkv, d = self.num_heads, self.num_kv_heads, self.head_size
        win = self.window
        if (
            win
            and self._window_covers_seqs
            and self.kernels.prefill_variant_for(
                hq, hkv, d, block_size, 1, win, q.dtype, R.PREFILL_MIN_M
            )
            is None
        ):
            win = 0
        m = q.shape[0]
        seq_lens = batch.seq_lens[first:]
        block_table = batch.block_table[first:]
        nseq = seq_lens.shape[0]
        # A batch, or fewer tokens than the first build serves: a plan.
        planned = nseq > 1 or m < R.PREFILL_MIN_M
        if planned:
            variant, plan, head_group, starts = self._plan(
                batch, block_size, win, first, q
            )
        else:
            variant = self.kernels.prefill_variant_for(
                hq, hkv, d, block_size, 1, win, q.dtype, m
            )
        module = self.kernels.load_prefill(variant) if variant is not None else None
        if variant is None or module is None:
            self.rejected = (
                f"prefill of {m} query tokens: no build for {hq}/{hkv}/{d} "
                f"window {win} page {block_size} {q.dtype}"
            )
            return None
        # Sized for every build of the shape.
        shape = (hq, hkv, d, block_size, win, q.dtype)
        scratch_key = (shape, q.device)
        scratch = _PREFILL_SCRATCH.get(scratch_key)
        if scratch is None or scratch[1].numel() < 2 * variant.scratch_rows(q.device):
            scratch = R.make_prefill_scratch(
                [*R.prefill_variants(*shape), variant], q.device
            )
            _PREFILL_SCRATCH[scratch_key] = scratch
        args = (q, kv_cache, block_table, out, *scratch, seq_lens)
        if not planned:
            # Only sizes the KV split.
            return lambda: module.prefill_attn(
                *args, max(batch.max_seq_len, m), self.scale
            )
        if not plan.shape[0]:
            return lambda: None
        return lambda: module.prefill_attn(
            *args,
            batch.max_seq_len,
            self.scale,
            plan=plan,
            query_start_loc=starts,
            head_group=head_group,
        )

    def _plan(self, batch: Batch, block_size: int, win: int, first: int, q):
        """This step's build, plan, head group and device query starts for the
        sequences from `first` on."""
        key = (win, first)
        cached = batch.plans.get(key)
        if cached is None:
            base = int(batch.query_start_loc_cpu[first])
            starts = batch.query_start_loc_cpu[first:] - base
            seq_lens = batch.seq_lens_cpu[first:]
            variant, head_group = self.kernels.prefill_batch_variant_for(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                block_size,
                1,
                win,
                q.dtype,
                starts,
                seq_lens,
            )
            plan = None
            if variant is not None:
                plan, head_group = R.plan_prefill(
                    starts,
                    seq_lens,
                    variant,
                    win,
                    R.prefill_target_workgroups(q.device),
                    head_group,
                )
                plan = torch.from_numpy(plan).to(q.device)
            device_starts = batch.query_start_loc[first:] - base
            cached = batch.plans[key] = (variant, plan, head_group, device_starts)
        return cached


# ------------------------------------------------------------------ timing --


def do_bench_cudagraph(fn: Callable[[], None], rep: int = 20) -> list[float]:
    """Milliseconds of `fn`, ten samples: a CUDA graph of about `rep` ms of
    calls, replayed (as triton.testing.do_bench_cudagraph)."""
    with torch.cuda.stream(torch.cuda.Stream()):
        fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        torch.cuda.synchronize()
        estimate_ms = start.elapsed_time(end) / 5
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(n_repeat):
                fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / n_repeat)
    return times


_flush: Optional[torch.Tensor] = None


def do_bench(fn: Callable[[], None], warmup: int = 25, rep: int = 100) -> list[float]:
    """Milliseconds of each call of `fn` over about `rep` ms, after `warmup`
    ms, with a 256 MiB buffer rewritten before each call to empty the caches
    (as triton.testing.do_bench)."""
    global _flush
    fn()
    torch.cuda.synchronize()
    if _flush is None:
        _flush = torch.empty(256 * 1024 * 1024 // 8, dtype=torch.int64, device="cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        _flush.zero_()
        fn()
    end.record()
    torch.cuda.synchronize()
    estimate_ms = start.elapsed_time(end) / 5
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for _ in range(n_warmup):
        fn()
    for i in range(n_repeat):
        _flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def median_us(fn: Callable[[], None], warmup: int = 5, rep: int = 40) -> float:
    """Median microseconds of one call of `fn`."""
    return statistics.median(do_bench(fn, warmup=warmup, rep=rep)) * 1e3


def _block_rows(
    q_lens: list[int], kv_lens: list[int], block_size: int, window: int
) -> tuple[list[list[int]], int]:
    """Block table rows and the KV blocks to allocate: each request's own
    blocks (those in its window), the rest block 0, as a server does."""
    max_blocks = -(-max(kv_lens) // block_size)
    rows, nxt = [], 1
    for q, kv in zip(q_lens, kv_lens):
        first = max(0, kv - q - (window - 1)) // block_size if window else 0
        live = -(-kv // block_size) - first
        rows.append(
            [0] * first
            + list(range(nxt, nxt + live))
            + [0] * (max_blocks - first - live)
        )
        nxt += live
    return rows, nxt


@dataclass
class Result:
    """A timed batch spec, per layer."""

    median_us: float
    std_us: float
    # "kernel" (decode), "prefill" or "unserved".
    path: str
    # Why, if unserved.
    rejected: Optional[str]
    num_layers: int

    @property
    def spread_pct(self) -> float:
        return self.std_us / self.median_us * 100


def run(
    spec: str,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    *,
    dtype: torch.dtype = torch.float16,
    block_size: int = 16,
    window: int = 0,
    num_layers: int = 10,
    min_working_set_mb: float = 96,
    use_cuda_graphs: bool = True,
    max_model_len: Optional[int] = None,
    spec_tokens: int = 0,
    max_num_seqs: int = 256,
    kernels: Any = R,
    device: str = "cuda",
) -> Result:
    """Time one attention call of a batch spec, per layer.

    Args:
        spec: The batch (batch_spec.py), decodes first.
        num_q_heads: Query heads.
        num_kv_heads: KV heads.
        head_size: Head dimension.
        dtype: Element type of Q, the KV cache and the output.
        block_size: KV cache page size.
        window: Sliding window in keys, 0 for full attention.
        num_layers: Fewest layers, each with its own tensors.
        min_working_set_mb: More layers until their KV reaches this size.
        use_cuda_graphs: Time a CUDA graph of the layers (decodes).
        max_model_len: Longest sequence served; the spec's by default.
        spec_tokens: Requests of up to 1 + this many tokens are decodes.
        max_num_seqs: Most sequences of a batch.
        kernels: What picks and loads the builds.
        device: The GPU.

    """
    dev = torch.device(device)
    requests = parse_batch_spec(spec)
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)
    rows, num_blocks = _block_rows(q_lens, kv_lens, block_size, window)

    # More layers until the KV working set outgrows the cache.
    read_lens = [
        min(kv, window + q - 1) if window else kv for kv, q in zip(kv_lens, q_lens)
    ]
    itemsize = torch.finfo(dtype).bits // 8
    kv_bytes = 2 * sum(read_lens) * num_kv_heads * head_size * itemsize
    if min_working_set_mb and kv_bytes > 0:
        num_layers = max(num_layers, -(-int(min_working_set_mb * 1024**2) // kv_bytes))

    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    seq_lens = np.asarray(kv_lens, dtype=np.int32)
    num_decodes, num_decode_tokens = split_decodes(
        q_lens, min(1 + spec_tokens, R.MAX_M)
    )
    batch = Batch(
        query_start_loc_cpu=starts,
        seq_lens_cpu=seq_lens,
        query_start_loc=torch.from_numpy(starts).to(dev),
        seq_lens=torch.from_numpy(seq_lens).to(dev),
        block_table=torch.tensor(rows, dtype=torch.int32, device=dev),
        max_query_len=max(q_lens),
        max_seq_len=max(kv_lens),
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
    )
    attn = Attention(
        num_q_heads,
        num_kv_heads,
        head_size,
        window,
        max_model_len or max(kv_lens),
        max_num_seqs,
        kernels,
    )
    # Random KV: zeros make every score tie and run unrealistically fast.
    caches = torch.empty(
        num_layers,
        num_blocks,
        num_kv_heads,
        block_size,
        2 * head_size,
        dtype=dtype,
        device=dev,
    ).normal_()
    q_list = [
        torch.randn(total_q, num_q_heads, head_size, device=dev, dtype=dtype)
        for _ in range(num_layers)
    ]
    # One output per layer, like the inputs: a single buffer rewritten by every
    # layer would stay in the cache and its writes would not reach DRAM.
    out_list = [torch.empty_like(q) for q in q_list]

    def layers() -> None:
        for i in range(num_layers):
            attn.forward(q_list[i], caches[i], batch, out_list[i])

    if use_cuda_graphs:
        times = do_bench_cudagraph(layers)
        torch.cuda.empty_cache()
    else:
        times = do_bench(layers)
    torch.cuda.synchronize()
    us = [t * 1e3 / num_layers for t in times]
    return Result(
        median_us=statistics.median(us),
        std_us=statistics.stdev(us) if len(us) > 1 else 0.0,
        path=attn.path(),
        rejected=attn.rejected if attn.unserved_calls else None,
        num_layers=num_layers,
    )
