import warnings
from collections.abc import Callable

import torch

from flykat.tools.bench.models import MeasureConfig, TimingStats


class EmptyGraphCaptureError(RuntimeError):
    """Graph capture recorded no work, so the timed graph is empty.

    A FlyDSL launcher that does not declare a ``stream: fx.Stream`` parameter runs on
    the HIP NULL stream; ``torch.cuda.graph`` captures a side stream and drops those
    launches, leaving an empty graph that would otherwise time as a meaningless ~1 us.
    Such a kernel also cannot run inside a graph-based inference server, so this is a
    real defect, not a measurement nuisance.
    """

    def __init__(self) -> None:
        super().__init__("graph capture recorded zero nodes: the kernel launched on the NULL stream and was not captured. Declare a 'stream: fx.Stream' parameter on the launcher, pass torch.cuda.current_stream() from run(), and forward it to .launch(..., stream=stream) so the kernel is graph-capturable.")


def _captured_node_count(graph: object) -> int | None:
    """Captured node count if the torch CUDAGraph binding exposes it, else None.

    The attribute is build-dependent (method or plain attr across ROCm torch builds), so
    probe defensively; None means the count cannot be determined and the caller falls
    back to the runtime's empty-graph warning.
    """
    for attr in ("num_nodes", "_num_nodes"):
        probe = getattr(graph, attr, None)
        if probe is None:
            continue
        value = probe() if callable(probe) else probe
        if isinstance(value, int):
            return value
    return None


def bench_graph(closure: Callable[[], None], measure: MeasureConfig) -> TimingStats:
    # Warm on a dedicated capture stream. The floor of max(3, warmup_iters) ensures at
    # least one call lands after the FlyDSL cache miss (which runs the kernel twice: once
    # for the real output, then flyc.compile runs one canary launch), so the cache is
    # fully populated before capture.
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(max(3, measure.warmup_iters)):
            closure()
    torch.cuda.synchronize()

    # Capture one launcher call. torch.cuda.graph makes capture_stream current for the
    # block, so a launcher that threads torch.cuda.current_stream() enqueues on it and is
    # recorded. A launcher that omits the stream param runs on the NULL stream and is
    # dropped, leaving an empty graph; detect that and raise rather than time ~1 us.
    g = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.cuda.graph(g, stream=capture_stream):
            closure()

    node_count = _captured_node_count(g)
    empty_warning = any("empty" in str(w.message).lower() for w in caught)
    if node_count == 0 or (node_count is None and empty_warning):
        raise EmptyGraphCaptureError()

    # keepalive holds the runner that owns the FlyDSL engine the captured nodes point
    # into. If the caller's reference to closure is the only one and Python GCs it,
    # graph.replay() becomes a use-after-free. Keeping this reference prevents that.
    keepalive = closure  # noqa: F841

    # Each sample brackets graph_replay_iters back-to-back replays under one event pair
    # and divides by the replay count. Replays are serialized by the stream FIFO, so
    # this is serial device time — not a throughput/overlap measurement. Bracketing N
    # replays together removes per-replay event-record overhead and the per-replay host
    # dispatch overhead where the GPU would otherwise wait for the host to issue the next
    # replay().
    samples: list[float] = []
    for _ in range(measure.bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measure.graph_replay_iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / measure.graph_replay_iters * 1000.0)  # ms -> us

    return TimingStats(warmup_iters=measure.warmup_iters, samples_us=samples, strategy="graph")
