# SPDX-License-Identifier: MIT
"""Experimental public-path factory for the Stage2 route-store reducer."""

from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import (
    MegaMoETileA4W4,
)


class MegaMoETileA4W4RouteStore(MegaMoETileA4W4):
    """Keep the public MegaMoE API while selecting the route-store Stage2."""

    def __init__(
        self,
        *args,
        stage2_node_reduce_blocks: int = 16,
        stage2_node_reduce_vec_bytes: int = 8,
        stage2_node_reduce_schedule: str = "token",
        stage2_node_reduce_load_schedule: str = "interleaved",
        **kwargs,
    ):
        if int(stage2_node_reduce_vec_bytes) not in (4, 8):
            raise ValueError(
                "route_store stage2_node_reduce_vec_bytes must be 4 or 8"
            )
        self.stage2_node_accumulation_mode = "route_store"
        self.stage2_node_reduce_blocks = int(stage2_node_reduce_blocks)
        self.stage2_node_reduce_vec_bytes = int(stage2_node_reduce_vec_bytes)
        self.stage2_node_reduce_schedule = str(stage2_node_reduce_schedule)
        self.stage2_node_reduce_load_schedule = str(
            stage2_node_reduce_load_schedule
        )
        self.stage2_worker_blocks = 160 + self.stage2_node_reduce_blocks
        super().__init__(*args, **kwargs)

    def _validate_launcher_contracts(self) -> None:
        super()._validate_launcher_contracts()
        expected = {
            "node_accumulation_mode": "route_store",
            "node_reduce_blocks": self.stage2_node_reduce_blocks,
            "node_reduce_vec_bytes": self.stage2_node_reduce_vec_bytes,
            "node_reduce_schedule": self.stage2_node_reduce_schedule,
            "node_reduce_load_schedule": self.stage2_node_reduce_load_schedule,
        }
        launcher_mismatch = {
            name: (getattr(self._stage2, name, "<missing>"), value)
            for name, value in expected.items()
            if getattr(self._stage2, name, "<missing>") != value
        }
        manifest = getattr(self._stage2, "architecture_contract", {})
        manifest_mismatch = {
            name: (manifest.get(name, "<missing>"), value)
            for name, value in expected.items()
            if manifest.get(name, "<missing>") != value
        }
        if launcher_mismatch or manifest_mismatch:
            raise RuntimeError(
                "route-store Stage2 launcher contract mismatch: "
                f"launcher={launcher_mismatch}, manifest={manifest_mismatch}"
            )


__all__ = ["MegaMoETileA4W4RouteStore"]
