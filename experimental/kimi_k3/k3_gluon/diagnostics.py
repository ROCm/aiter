# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E731
"""Explicit stream delays for reproducing peer wait failures on idle GPUs."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def stream_delay(iterations):
    for _ in range(iterations):
        gl.inline_asm_elementwise(
            "s_sleep 1\n s_mov_b32 $0, 0",
            "=s",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )


def calibrated_delay(milliseconds):
    """Return a warmed launch; measure s_sleep instead of assuming clock rate."""
    import torch

    launch = lambda count: stream_delay[(1,)](count, num_warps=4)
    launch(4096)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    launch(4096)
    end.record()
    end.synchronize()
    measured = start.elapsed_time(end)
    count = max(1, round(4096 * milliseconds / measured))
    start.record()
    launch(count)
    end.record()
    end.synchronize()
    return (lambda: launch(count)), start.elapsed_time(end)


@g.jit
def cta_timestamp(Progress, epoch, pid, stage: gl.constexpr, TARGET: gl.constexpr):
    if epoch == TARGET:
        ticks = gl.inline_asm_elementwise(
            "s_memrealtime $0\n s_waitcnt lgkmcnt(0)",
            "=s,~{memory}",
            [],
            gl.uint64,
            False,
            1,
        )
        gl.store(Progress + pid * 3 + stage, ticks)


class LimitedCUStream:
    """A diagnostic mask on our own HIP stream; never changes GPU settings."""

    def __init__(self, count):
        import ctypes as C
        from pathlib import Path
        import torch

        mapped = [
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libamdhip64.so" in line
        ]
        self.hip = C.CDLL(mapped[0])
        self.hip.hipExtStreamCreateWithCUMask.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_uint32,
            C.POINTER(C.c_uint32),
        ]
        self.hip.hipExtStreamGetCUMask.argtypes = [
            C.c_void_p,
            C.c_uint32,
            C.POINTER(C.c_uint32),
        ]
        self.hip.hipStreamDestroy.argtypes = [C.c_void_p]
        self.ptr = C.c_void_p()
        mask = (C.c_uint32 * 8)(
            *[
                sum(1 << bit for bit in range(32) if word * 32 + bit < count)
                for word in range(8)
            ]
        )
        assert self.hip.hipExtStreamCreateWithCUMask(C.byref(self.ptr), 8, mask) == 0
        actual = (C.c_uint32 * 8)()
        assert self.hip.hipExtStreamGetCUMask(self.ptr, 8, actual) == 0
        self.actual_mask = list(actual)
        assert sum(word.bit_count() for word in actual) == count, self.actual_mask
        self.stream = torch.cuda.ExternalStream(self.ptr.value)

    def close(self):
        self.stream.synchronize()
        assert self.hip.hipStreamDestroy(self.ptr) == 0
