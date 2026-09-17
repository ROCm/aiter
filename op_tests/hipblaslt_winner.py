# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load generated Tensile or public hipBLASLt into the Torch benchmark process.

The direct path selects a supplied winner. The public path asks the linked
hipBLASLt library for its normal heuristic selection.
"""

import functools
import importlib.util
import pathlib

import torch


def pack_scales(scale: torch.Tensor, repeat_rows: int = 1) -> torch.Tensor:
    """Expand block-128 E8M0 bytes to equivalent Tensile MX32 scale storage."""
    scale = scale.view(torch.uint8).repeat_interleave(repeat_rows, dim=0)
    rows, groups = scale.shape
    # gfx1250 TN format is [K/128, rows, four MX32 scales].
    return scale[:, :, None].expand(rows, groups, 4).permute(1, 0, 2).contiguous()


@functools.lru_cache(maxsize=None)
def _load_bridge(path: str):
    """Load the direct-winner pybind module once per shared-object path."""
    spec = importlib.util.spec_from_file_location("hipblaslt_winner_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load direct winner bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HipblasltWinner:
    """Own the exact generated winner while Torch owns tensors and the stream."""

    def __init__(self, root: str, bridge: str, m: int, n: int, k: int):
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
        module = _load_bridge(str(pathlib.Path(bridge).resolve()))
        self._winner = module.Winner(str(matches[0]), str(matches[0].parent), m, n, k)
        self.name = self._winner.solution_name
        self.workspace_size = self._winner.workspace_size

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
        self._winner.launch(
            a.data_ptr(),
            b.data_ptr(),
            scale_a.data_ptr(),
            scale_b.data_ptr(),
            out.data_ptr(),
            workspace.data_ptr(),
            synchronizer.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    def close(self) -> None:
        """Release the adapter after its launches have completed."""
        self._winner = None


@functools.lru_cache(maxsize=None)
def _load_public_bridge(path: str):
    """Load the public hipBLASLt pybind module once per shared-object path."""
    spec = importlib.util.spec_from_file_location("hipblaslt_public_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load hipBLASLt public bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HipblasltPublic:
    """Own public hipBLASLt descriptors while Torch owns tensors and the stream."""

    def __init__(self, bridge: str, m: int, n: int, k: int):
        module = _load_public_bridge(str(pathlib.Path(bridge).resolve()))
        self._gemm = module.PublicGemm(m, n, k)
        self.index = self._gemm.index
        self.name = self._gemm.solution_name
        self.kernel_name = self._gemm.kernel_name
        self.workspace_size = self._gemm.workspace_size
        self.scale_a_mode = self._gemm.scale_a_mode
        self.scale_b_mode = self._gemm.scale_b_mode

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
