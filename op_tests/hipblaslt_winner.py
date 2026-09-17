# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load hipBLASLt paths into the Torch benchmark process.

The explicit-winner path uses a generated Tensile library. The public path asks
the linked hipBLASLt library for its heuristic selection.
"""

import copy
import ctypes
import functools
import importlib.util
import pathlib

import torch


def external_perftest(num_iters=100, num_warmup=2, num_rotate_args=0, **options):
    """Use the same rotation loop without nesting Torch and rocprof profilers.

    Timing is reported by rocprof, so the ordinary summary's time is NaN.
    This helper is only selected with GEMM_BENCH_EXTERNAL=1.
    """
    test_graph = options.pop("testGraph", False)
    if options:
        raise ValueError(f"Unsupported external benchmark options: {options}")

    def decorate(function):
        def run(*args, **kwargs):
            count = min(num_rotate_args or num_iters, num_iters)
            buffers = [
                (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(count - 1)
            ] + [(args, kwargs)]
            for _ in range(num_warmup):
                function(*args, **kwargs)
            torch.cuda.synchronize()
            if test_graph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for index in range(num_iters):
                        call_args, call_kwargs = buffers[index % count]
                        result = function(*call_args, **call_kwargs)
                graph.replay()
            else:
                for index in range(num_iters):
                    call_args, call_kwargs = buffers[index % count]
                    result = function(*call_args, **call_kwargs)
            torch.cuda.synchronize()
            return result, float("nan")

        return run

    return decorate


def pack_scales(scale: torch.Tensor, repeat_rows: int = 1) -> torch.Tensor:
    """Expand block-128 E8M0 bytes to equivalent Tensile MX32 scale storage."""
    scale = scale.view(torch.uint8).repeat_interleave(repeat_rows, dim=0)
    rows, groups = scale.shape
    # gfx1250 TN format is [K/128, rows, four MX32 scales].
    return scale[:, :, None].expand(rows, groups, 4).permute(1, 0, 2).contiguous()


class HipblasltWinner:
    """Own the host adapter; tensors and the current stream remain Torch-owned."""

    def __init__(self, root: str, bridge: str, m: int, n: int, k: int):
        self._handle = None
        root_path = pathlib.Path(root)
        matches = []
        for ini in root_path.rglob("ClientParameters.ini"):
            params = dict(
                line.split("=", 1)
                for line in ini.read_text().splitlines()
                if "=" in line
            )
            if params.get("problem-size") == f"{m},{n},1,{k}":
                matches.append(pathlib.Path(params["library-file"]))
        matches = sorted(set(matches))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one winner for {m,n,k} under {root}; found {matches}"
            )
        self._lib = ctypes.CDLL(bridge)
        pointer = ctypes.c_void_p
        self._lib.WinnerCreate.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.WinnerCreate.restype = pointer
        self._lib.WinnerError.restype = ctypes.c_char_p
        self._lib.WinnerError.argtypes = []
        self._lib.WinnerName.argtypes = [pointer]
        self._lib.WinnerName.restype = ctypes.c_char_p
        self._lib.WinnerWorkspaceSize.argtypes = [pointer]
        self._lib.WinnerWorkspaceSize.restype = ctypes.c_size_t
        self._lib.WinnerLaunch.argtypes = [pointer] * 9
        self._lib.WinnerLaunch.restype = ctypes.c_int
        self._lib.WinnerDestroy.argtypes = [pointer]
        self._lib.WinnerDestroy.restype = None
        self._handle = self._lib.WinnerCreate(
            str(matches[0]).encode(), str(matches[0].parent).encode(), m, n, k
        )
        if not self._handle:
            raise RuntimeError(self._lib.WinnerError().decode())
        self.name = self._lib.WinnerName(self._handle).decode()
        self.workspace_size = self._lib.WinnerWorkspaceSize(self._handle)

    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        synchronizer: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the winner without packing, copies, or synchronization."""
        status = self._lib.WinnerLaunch(
            self._handle,
            a.data_ptr(),
            b.data_ptr(),
            scale_a.data_ptr(),
            scale_b.data_ptr(),
            out.data_ptr(),
            workspace.data_ptr(),
            synchronizer.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(self._lib.WinnerError().decode())
        return out

    def close(self) -> None:
        """Release the adapter after its launches have completed."""
        if self._handle:
            self._lib.WinnerDestroy(self._handle)
            self._handle = None


@functools.lru_cache(maxsize=None)
def _load_public_bridge(path: str):
    """Load the native pybind module once per shared-object path."""
    spec = importlib.util.spec_from_file_location("hipblaslt_public_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load hipBLASLt public bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HipblasltPublic:
    """Own public hipBLASLt descriptors; tensors and stream remain Torch-owned."""

    def __init__(self, bridge: str, m: int, n: int, k: int):
        module = _load_public_bridge(str(pathlib.Path(bridge).resolve()))
        self._gemm = module.PublicGemm(m, n, k)
        self.index = self._gemm.index
        self.name = self._gemm.solution_name
        self.kernel_name = self._gemm.kernel_name
        self.workspace_size = self._gemm.workspace_size

    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the public heuristic without packing, copies, or synchronization."""
        self._gemm.launch(
            a.data_ptr(),
            b.data_ptr(),
            scale_a.data_ptr(),
            scale_b.data_ptr(),
            out.data_ptr(),
            workspace.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    def close(self) -> None:
        """Release the hipBLASLt descriptors after launches have completed."""
        self._gemm = None
