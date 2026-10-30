"""Per-test GPU memory trace (diagnostic, opt-in via AITER_GPU_MEMTRACE=1).

Decomposes device memory each test into torch-allocated vs. non-torch "used"
so we can tell whether the ~110 GiB non-torch occupancy seen in DPX OOMs is
in-process accumulation (non_torch climbs test-by-test) or external/sibling-
partition contention (non_torch high/noisy from the first test, independent of
this process). Lines are tagged [memtrace] for easy grepping in pytest.log.
"""

import os

import pytest

_TRACE_ENABLED = bool(os.environ.get("AITER_GPU_MEMTRACE"))

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _HAS_CUDA = False


def _mem_snapshot() -> str:
    gib = 1024**3
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    used = total - free
    non_torch = used - reserved
    return (
        f"total={total / gib:.1f} used={used / gib:.1f} free={free / gib:.1f} "
        f"torch_alloc={allocated / gib:.2f} torch_reserved={reserved / gib:.2f} "
        f"non_torch={non_torch / gib:.2f}"
    )


@pytest.fixture(autouse=True)
def _gpu_mem_trace(request):
    if not (_TRACE_ENABLED and _HAS_CUDA):
        yield
        return
    node = request.node.nodeid
    try:
        print(f"[memtrace] BEFORE {node} :: {_mem_snapshot()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[memtrace] BEFORE {node} :: snapshot_failed {exc!r}", flush=True)
    try:
        yield
    finally:
        try:
            print(f"[memtrace] AFTER  {node} :: {_mem_snapshot()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[memtrace] AFTER  {node} :: snapshot_failed {exc!r}", flush=True)
