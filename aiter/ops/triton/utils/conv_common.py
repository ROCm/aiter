import torch
import triton
import math
import functools
from typing import Optional, Callable

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER: AiterTritonLogger = AiterTritonLogger()


def padding_same(i, w, stride, dilation):
    return math.ceil((stride * (i - 1) + 1 + dilation * (w - 1) - i) / 2)


def out_padding(i, w, padding, stride, dilation):
    return int((i + 2 * padding - dilation * (w - 1) - 1) / stride + 1)


def padding_type(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: torch.Tensor,
    dilation: torch.Tensor,
    padding: str,
) -> list[int, int, int]:
    """
    Determines padding-mode padding

    Args:
        x (torch.Tensor): Input tensor shaped (N, C, D, H, W).
        w (torch.Tensor): Weight tensor shaped (C_out, C_in/groups, Kd, Kh, Kw).
        stride (torch.Tensor | tuple[int, int, int]): Strides for D, H, W.
        dilation (torch.Tensor | tuple[int, int, int]): Dilations for Kd, Kh, Kw.

    Returns:
        tuple[list[int, int, int], list[int, int, int]]:
            - padding_inp: padding for (D, H, W)
            - padding_out: output sizes for (D, H, W)

    Notes:
        SAME mode only supports stride 1 for all dimensions.
    """
    padding = padding.upper()
    padding_val = [0, 0, 0]  # default valid padding

    if padding == "SAME":
        _LOGGER.info("Using SAME padding algorithm.")
        assert all(
            s == 1 for s in stride
        ), "SAME padding only supports stride of 1. recieved stride: {stride}"
        padding_val = [
            padding_same(i, w, s, d)
            for i, w, s, d in zip(x.shape[-3:], w.shape[-3:], stride, dilation)
        ]

    return padding_val


def conv3d_output_shape(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    """
    Calculate the output shape of a 3D convolution.

    Args:
        x (torch.Tensor): Input tensor shaped (N, C_in, D_in, H_in, W_in).
        w (torch.Tensor): Weight tensor shaped (C_out, C_in/groups, Kd, Kh, Kw).
        stride (tuple[int, int, int]): Strides for D, H, W.
        padding (tuple[int, int, int]): Paddings for D, H, W.
        dilation (tuple[int, int, int]): Dilations for Kd, Kh, Kw.

    Returns:
        tuple[int, int, int, int, int]: Output tensor shape (N, C_out, D_out, H_out, W_out).
    """
    N = x.shape[0]
    C_out = w.shape[0]
    D_out, H_out, W_out = [
        out_padding(i, w, p, s, d)
        for i, w, p, s, d in zip(x.shape[-3:], w.shape[-3:], padding, stride, dilation)
    ]

    return (N, C_out, D_out, H_out, W_out)


def conv3d_total_flops(*args, **kwargs):
    """Total FLOPs for a conv3d.

    Accepts the same ``*args, **kwargs`` as the triton conv3d kernel::

        (x_ptr, w_ptr, y_ptr, b_ptr,
         N, D, H, W, OC, OD, OH, OW,
         ...strides...,
         K_C, K_D, K_H, K_W,
         ...conv params...,
         GROUPS=1, BLOCK_N=..., ...)
    """
    # positions: 4=N, 8=OC, 9=OD, 10=OH, 11=OW, 27=K_C, 28=K_D, 29=K_H, 30=K_W
    N, OC = args[4], args[8]
    OD, OH, OW = args[9], args[10], args[11]
    K_C, K_D, K_H, K_W = args[27], args[28], args[29], args[30]
    GROUPS = kwargs.get("GROUPS", 1)
    in_c = K_C * GROUPS
    return 2 * N * OC * OD * OH * OW * in_c * K_D * K_H * K_W


def conv3d_total_bytes(*args, **kwargs):
    """Total bytes transferred for a conv3d.

    Accepts the same ``*args, **kwargs`` as the triton conv3d kernel.
    Assumes bf16 (2 bytes per element).
    """
    # positions: 4=N, 5=D, 6=H, 7=W, 8=OC, 9=OD, 10=OH, 11=OW
    #            27=K_C, 28=K_D, 29=K_H, 30=K_W
    N, D, H, W = args[4], args[5], args[6], args[7]
    OC, OD, OH, OW = args[8], args[9], args[10], args[11]
    K_C, K_D, K_H, K_W = args[27], args[28], args[29], args[30]
    GROUPS = kwargs.get("GROUPS", 1)
    dtype_size = 2  # bf16
    in_c = K_C * GROUPS
    input_bytes = N * in_c * D * H * W * dtype_size
    weight_bytes = OC * in_c * K_D * K_H * K_W * dtype_size
    bias_bytes = OC * dtype_size
    output_bytes = N * OC * OD * OH * OW * dtype_size
    return input_bytes + weight_bytes + bias_bytes + output_bytes


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 0.2,
    atol: float = 0.02,
    msg: str = "",
) -> None:
    """GPU-side tensor comparison -- orders of magnitude faster than
    ``torch.testing.assert_close`` on large tensors.

    Falls back to ``torch.testing.assert_close`` when tensors are not
    on a CUDA device.

    An element passes if ``|actual - expected| <= atol`` **or**
    ``|actual - expected| / (|expected| + eps) <= rtol``.
    The assertion fails only when *both* conditions are violated for
    at least one element.

    On failure the error message includes statistics similar to
    ``torch.testing.assert_close``: number / percentage of mismatched
    elements, greatest absolute and relative differences with their
    locations.

    Args:
        actual:   Tensor to check.
        expected: Reference tensor (same shape / device).
        rtol:     Maximum acceptable relative tolerance.
        atol:     Maximum acceptable absolute tolerance.
        msg:      Optional prefix for the error message.
    """
    # Fallback to torch when not on GPU
    if not actual.is_cuda or not expected.is_cuda:
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, msg=msg or None
        )
        return

    def _flat_to_index(flat_idx: int, shape: torch.Size) -> tuple:
        idx = []
        for s in reversed(shape):
            idx.append(flat_idx % s)
            flat_idx //= s
        return tuple(reversed(idx))

    diff = (actual - expected).abs()

    # 1: all elements within absolute tolerance
    max_atol = diff.max().item()
    if max_atol <= atol:
        return

    # 2: element OK if diff <= rtol * |expected|
    # Uses multiplication instead of division to avoid bf16->float32 promotion
    bad = (diff > atol) & (diff > rtol * expected.abs())
    num_mismatch = bad.sum().item()
    del bad

    if num_mismatch == 0:
        return

    numel = actual.numel()
    pct = num_mismatch / numel * 100

    # Greatest absolute difference
    abs_flat = diff.reshape(-1)
    abs_argmax = abs_flat.argmax().item()
    abs_max = abs_flat[abs_argmax].item()
    abs_idx = _flat_to_index(abs_argmax, actual.shape)
    abs_actual = actual.reshape(-1)[abs_argmax].item()
    abs_expected = expected.reshape(-1)[abs_argmax].item()

    # Greatest relative difference (only computed on failure path)
    rel = diff / (expected.abs() + 1e-6)
    rel_flat = rel.reshape(-1)
    rel_argmax = rel_flat.argmax().item()
    rel_max = rel_flat[rel_argmax].item()
    rel_idx = _flat_to_index(rel_argmax, actual.shape)
    rel_actual = actual.reshape(-1)[rel_argmax].item()
    rel_expected = expected.reshape(-1)[rel_argmax].item()

    prefix = f"{msg}\n" if msg else ""
    raise AssertionError(
        f"{prefix}"
        f"Mismatched elements: {num_mismatch} / {numel} ({pct:.2f}%)\n"
        f"Greatest absolute difference: {abs_max:.6f} (up to {atol} allowed)\n"
        f"  at index {abs_idx}: actual={abs_actual}, expected={abs_expected}\n"
        f"Greatest relative difference: {rel_max:.6f} (up to {rtol} allowed)\n"
        f"  at index {rel_idx}: actual={rel_actual}, expected={rel_expected}"
    )


class GPUTimer:
    """GPU kernel timer -- usable as a **decorator** or standalone.

    As decorator (auto-captures triton kernel launches inside the function):

        @GPUTimer(warmup=5, rep=20)
        def conv3d_channel_last(x, weight, ...):
            _kernel[grid](...)          # <-- captured & benchmarked
            return output

        output = conv3d_channel_last(x, weight, ...)
        # [INFO] conv3d_channel_last::_kernel  median=1.23 ms, p20=1.20 ms, p80=1.25 ms

    Standalone:

        t = GPUTimer.bench(lambda: kernel(...), warmup=5, rep=100)
        print(t.elapsed, t.elapsed_p20, t.elapsed_p80)
    """

    def __init__(
        self,
        warmup: int = 0,
        rep: int = 1,
        total_flops: Optional[Callable] = None,
        total_bytes: Optional[Callable] = None,
    ):
        """
        Args:
            warmup:      Number of warmup iterations.
            rep:         Number of timed repetitions.
            total_flops: callable(*kernel_args, **kernel_kwargs) -> int.
                         Called with the captured triton kernel's launch
                         arguments to compute total FLOPs.
            total_bytes: callable(*kernel_args, **kernel_kwargs) -> int.
                         Same as total_flops but for total memory bytes.
        """
        self.warmup = warmup
        self.rep = max(rep, 1)
        self._total_flops = (
            total_flops  # callable(*kernel_args, **kernel_kwargs) -> int
        )
        self._total_bytes = (
            total_bytes  # callable(*kernel_args, **kernel_kwargs) -> int
        )
        self.total_flops: Optional[int] = None
        self.total_bytes: Optional[int] = None
        self.elapsed_ms = 0.0
        self.elapsed_p20 = 0.0
        self.elapsed_p80 = 0.0
        self._timings: list[float] = []

    # ------------------------------------------------------------------ #
    #  Core benchmark logic                                               #
    # ------------------------------------------------------------------ #
    def _run_bench(self, fn, warmup: int = 0, rep: int = 1):
        """Run *fn* with warmup/rep, populate timing fields on *self*."""
        rep = max(rep, 1)

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        timings = []
        for _ in range(rep):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))

        timings.sort()
        self._timings = timings
        n = len(timings)
        self.elapsed_ms = timings[n // 2]  # median
        self.elapsed_p20 = timings[int(n * 0.2)]
        self.elapsed_p80 = timings[int(n * 0.8)]
        return self

    @staticmethod
    def bench(fn, warmup: int = 0, rep: int = 1):
        """Standalone benchmark -- return a new GPUTimer with results."""
        return GPUTimer(warmup=warmup, rep=rep)._run_bench(fn, warmup, rep)

    # ------------------------------------------------------------------ #
    #  Decorator mode                                                     #
    # ------------------------------------------------------------------ #
    def __call__(self, fn):
        """Decorate *fn* -- intercept triton kernel launches and benchmark them."""
        warmup = self.warmup
        rep = self.rep
        flops_spec = self._total_flops
        bytes_spec = self._total_bytes

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            captured: list[tuple] = []
            JITFunc = triton.runtime.jit.JITFunction
            orig_run = JITFunc.run

            # Hook: run kernel normally, then record its launch args
            def _hooked_run(kernel_self, *ka, **kw):
                ret = orig_run(kernel_self, *ka, **kw)
                captured.append((kernel_self, ka, kw))
                return ret

            JITFunc.run = _hooked_run
            try:
                result = fn(*args, **kwargs)
            finally:
                JITFunc.run = orig_run

            # Replay each captured kernel for benchmarking
            # ka = positional kernel args (ptrs, dims, strides, meta-params)
            # kw = keyword kernel args  (GROUPS=..., BLOCK_N=..., etc.)
            for ko, ka, kw in captured:
                t = GPUTimer.bench(
                    lambda _ko=ko, _ka=ka, _kw=kw: orig_run(_ko, *_ka, **_kw),
                    warmup=warmup,
                    rep=rep,
                )
                if flops_spec:
                    t.total_flops = flops_spec(*ka, **kw)
                if bytes_spec:
                    t.total_bytes = bytes_spec(*ka, **kw)
                name = getattr(getattr(ko, "fn", ko), "__name__", str(ko))
                _LOGGER.info(f"{fn.__name__}::{name} {t._format_stats()}")

            return result

        return wrapper

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #
    @property
    def elapsed(self) -> float:
        return self.elapsed_ms

    @property
    def timings(self) -> list[float]:
        return self._timings

    @property
    def tflops(self) -> Optional[float]:
        """TFLOPS based on total_flops and median latency. None if total_flops not set."""
        if self.total_flops is None or self.elapsed_ms <= 0:
            return None
        return self.total_flops / (self.elapsed_ms / 1e3) / 1e12

    @property
    def gbps(self) -> Optional[float]:
        """GB/s based on total_bytes and median latency. None if total_bytes not set."""
        if self.total_bytes is None or self.elapsed_ms <= 0:
            return None
        return self.total_bytes / (self.elapsed_ms / 1e3) / 1e9

    def _format_stats(self) -> str:
        """Format timing + optional tflops/gbps into a log string."""
        parts = [
            f"median={self.elapsed:.4f} ms",
            f"p20={self.elapsed_p20:.4f} ms",
            f"p80={self.elapsed_p80:.4f} ms",
        ]
        if self.tflops is not None:
            parts.append(f"tflops={self.tflops:.2f}")
        if self.gbps is not None:
            parts.append(f"gbps={self.gbps:.2f}")
        return ", ".join(parts)
