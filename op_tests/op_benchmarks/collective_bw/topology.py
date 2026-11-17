# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The GPU driver's per-link peak bandwidth, from the KFD topology.

The same sysfs tree ``probe_xgmi_peer_links`` reads for the link type
(``/sys/class/kfd/kfd/topology/nodes``). Every node has ``io_links`` (and
``p2p_links``), each with a ``properties`` file whose ``max_bandwidth`` is
the driver's peak for that link in MB/s:

* PCIe (``type 2``): the GPU's link to its CPU, its one uplink. The driver
  computes it as lanes x signalling rate / 8, before line encoding: Gen5
  x16 reads 64000, where 128b/130b leaves 63.0 GB/s of payload.
* xGMI (``type 11``): one entry per directly connected peer GPU.

A value of 0 means the driver does not know (on PCIe, for example, the
``p2p_links`` between GPUs on different CPU sockets).

The answer is per host, not per rank: the slowest link of its kind over every
GPU. That assumes a node is uniform, as ``probe_xgmi_peer_links`` does, and
holds whatever ``HIP_VISIBLE_DEVICES`` selects.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("aiter")

KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")

# KFD io-link types, from include/uapi/linux/kfd_sysfs.h.
IOLINK_TYPE_PCIE = 2
IOLINK_TYPE_XGMI = 11


def _props(path: Path) -> dict[str, int]:
    from aiter.ops.flydsl.comm_collectives_host_shared import kfd_props

    return kfd_props(path)


def _is_gpu(node: Path) -> bool:
    gpu_id = node / "gpu_id"
    if gpu_id.is_file():
        return gpu_id.read_text().strip() not in ("", "0")
    return _props(node / "properties").get("simd_count", 0) > 0


def driver_link_gbps(link: str, root: Path = KFD_NODES) -> float | None:
    """The driver's peak rate of one *link* (``"pcie"`` or ``"xgmi"``) in
    GB/s per direction: the slowest such link of any GPU. None if the
    topology is unreadable, *link* is something else, or the driver reports
    no rate.

    PCIe: each GPU's link to its CPU node. xGMI: each GPU's links to its
    peer GPUs."""
    want = {"pcie": IOLINK_TYPE_PCIE, "xgmi": IOLINK_TYPE_XGMI}.get(link)
    if want is None:
        return None
    try:
        nodes = {p.name: p for p in Path(root).iterdir() if p.name.isdigit()}
        gpus = {name for name, p in nodes.items() if _is_gpu(p)}
        rates = []
        for name in sorted(gpus, key=int):
            for props_file in (nodes[name] / "io_links").glob("*/properties"):
                props = _props(props_file)
                if props.get("type") != want:
                    continue
                peer_is_gpu = str(props.get("node_to")) in gpus
                if peer_is_gpu != (want == IOLINK_TYPE_XGMI):
                    continue
                if props.get("max_bandwidth", 0) > 0:
                    rates.append(props["max_bandwidth"])
    except (OSError, ValueError):
        logger.debug("collective_bw: cannot read the KFD topology", exc_info=True)
        return None
    return min(rates) / 1e3 if rates else None
