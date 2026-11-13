# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The roofline's TransferBench settings, in one place.

These are the knobs that can change without changing what a roof means.
Two settings are fixed in ``patterns`` instead, because the method depends
on them: ``GFX_SE_TYPE=1`` (the wave split counts single-wave
sub-executors) and ``NUM_SUBITERATIONS=1`` (more passes per launch let
cacheable peer memory absorb the writes in L2; see ``patterns``).
"""

from __future__ import annotations

from dataclasses import dataclass

from aiter.ops.flydsl.comm_traffic import PeerMem


@dataclass(frozen=True)
class RoofSettings:
    # Timed and untimed TransferBench iterations per test
    # (NUM_ITERATIONS, NUM_WARMUPS). The CLI's --tb-iters / --tb-warmup.
    iterations: int = 20
    warmups: int = 3
    # Seconds one TransferBench invocation may take.
    timeout_s: float = 900.0
    # Waves per transfer every roof tries besides the op's own grid.
    wave_sweep: tuple[int, ...] = (16, 32, 64, 128, 256)
    # The workgroup (waves of 64 threads; 4 is RCCL's 256) and the peer
    # memory types the RCCL roof and the fabric peak try.
    rccl_waves_per_workgroup: int = 4
    rccl_mems: tuple[PeerMem, ...] = (PeerMem.UNCACHED, PeerMem.FINE, PeerMem.COARSE)
    # The fabric-peak probes move this many bytes per (rank, peer) pair:
    # enough to saturate the links, so the probe measures bandwidth.
    peak_bytes_per_peer: int = 64 << 20
    # Optional datasheet rate of one link, GB/s per direction. The nominal
    # per-rank peak is this times the links a rank can use: N - 1 on an xGMI
    # mesh, 1 on PCIe. None: no nominal peak in the report.
    link_gbps: float | None = None
    # Further TransferBench environment for every test. Validation walks
    # every element on the host, which costs far more than the transfers.
    env: tuple[tuple[str, str], ...] = (
        ("USE_HIP_EVENTS", "1"),
        ("ALWAYS_VALIDATE", "-1"),
    )

    def __post_init__(self):
        if self.iterations < 1 or self.warmups < 0:
            raise ValueError(f"need iterations >= 1 and warmups >= 0, got {self}")
        if not self.wave_sweep or min(self.wave_sweep) < 1:
            raise ValueError(f"wave_sweep needs positive widths, got {self}")
        if self.peak_bytes_per_peer < 4:
            raise ValueError(f"peak_bytes_per_peer must be >= 4, got {self}")
        if self.link_gbps is not None and self.link_gbps <= 0:
            raise ValueError(f"link_gbps must be positive, got {self}")
        if not self.rccl_mems or self.rccl_waves_per_workgroup < 1:
            raise ValueError(f"need RCCL memory types and block waves, got {self}")


DEFAULT = RoofSettings()
