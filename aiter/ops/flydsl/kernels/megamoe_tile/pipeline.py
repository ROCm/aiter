# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from .config import HierMegaMoETileConfig
from .markers import roctx_range
from .topology import LogicalTopology, RoutePlan, build_route_plan
from .transport import CopyTransport
from .workspace import HierWorkspace


class HierMegaMoETilePrototype:
    """First executable seam for the two fused kernels.

    Today this object validates topology, workspace, copy transport and the
    accepted-route snapshot.  Compute is delegated to AITER's A4W4 wrapper;
    subsequent revisions move these phases into resident roles in H1/H2.
    """

    def __init__(self, config: HierMegaMoETileConfig, *, device: torch.device | str):
        self.config = config
        self.device = torch.device(device)
        self.topology = LogicalTopology(
            config.world_size, config.logical_gpus_per_node
        )
        self.workspace = HierWorkspace.allocate(config, device=self.device)
        self.transport = CopyTransport()

    def plan(self, topk_ids: torch.Tensor) -> RoutePlan:
        with roctx_range("megamoeTile.h1.route_plan"):
            return build_route_plan(
                topk_ids,
                num_experts=self.config.num_experts,
                topology=self.topology,
            )

    def copy_put_signal(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        nbytes: int,
        signal: torch.Tensor,
        generation: int,
    ) -> None:
        with roctx_range("megamoeTile.transport.copy_stub"):
            self.transport.put_signal(
                destination, source, nbytes, signal, generation
            )
