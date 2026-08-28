# SPDX-License-Identifier: MIT
"""Experimental public-path factory for rank-local Stage2 accumulation."""

from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import (
    MegaMoETileA4W4,
)


class MegaMoETileA4W4RankLocal(MegaMoETileA4W4):
    """Keep the public MegaMoE API while selecting rank-local Stage2."""

    def __init__(
        self,
        *args,
        stage2_node_reduce_blocks: int = 16,
        stage2_node_reduce_vec_bytes: int = 8,
        stage2_node_reduce_schedule: str = "token",
        stage2_node_reduce_load_schedule: str = "interleaved",
        stage2_node_reduce_work_schedule: str = "static_strided",
        stage2_node_reduce_rejoin_blocks: int = 0,
        stage2_rank_epilogue_lds_addressing: str = "expanded",
        stage2_rank_accumulation_mode: str = "atomic",
        stage2_return_chunk_tokens: int = 16,
        stage2_rail_return_schedule: str = "compact",
        **kwargs,
    ):
        if int(stage2_node_reduce_blocks) not in (8, 16, 32, 56):
            raise ValueError(
                "stage2_node_reduce_blocks must be one of 8,16,32,56"
            )
        if int(stage2_node_reduce_vec_bytes) not in (4, 8, 16):
            raise ValueError(
                "stage2_node_reduce_vec_bytes must be 4, 8, or 16"
            )
        if stage2_node_reduce_work_schedule not in (
            "static_strided",
            "dynamic_head",
        ):
            raise ValueError(
                "stage2_node_reduce_work_schedule must be static_strided "
                "or dynamic_head"
            )
        stage2_node_reduce_rejoin_blocks = int(
            stage2_node_reduce_rejoin_blocks
        )
        if stage2_node_reduce_rejoin_blocks not in (0, 8, 16, 32):
            raise ValueError(
                "stage2_node_reduce_rejoin_blocks must be one of 0,8,16,32"
            )
        if stage2_node_reduce_rejoin_blocks > 0 and (
            stage2_rail_return_schedule != "compact"
            or stage2_node_reduce_work_schedule != "dynamic_head"
        ):
            raise ValueError(
                "node-reduce GMM rejoin requires rank_local, compact return, "
                "persistent_queue GMM, and dynamic_head reduction"
            )
        if stage2_rank_epilogue_lds_addressing not in ("expanded", "dynamic_base"):
            raise ValueError(
                "stage2_rank_epilogue_lds_addressing must be expanded or dynamic_base"
            )
        if stage2_rank_epilogue_lds_addressing == "dynamic_base" and not (
            int(stage2_node_reduce_vec_bytes) == 8
            and stage2_node_reduce_load_schedule == "load_first"
            and stage2_node_reduce_work_schedule == "static_strided"
            and stage2_node_reduce_rejoin_blocks == 0
        ):
            raise ValueError(
                "dynamic_base LDS addressing requires vec8, load_first, "
                "static_strided reduction, and rejoin_blocks=0"
            )
        if stage2_rank_accumulation_mode not in ("atomic", "staged_reduce", "staged_ring"):
            raise ValueError(
                "stage2_rank_accumulation_mode must be atomic, staged_reduce, or staged_ring"
            )
        if stage2_rank_accumulation_mode == "staged_reduce" and not (
            int(stage2_node_reduce_vec_bytes) == 8
            and stage2_node_reduce_load_schedule == "load_first"
            and stage2_node_reduce_work_schedule == "static_strided"
            and stage2_node_reduce_rejoin_blocks == 0
            and stage2_rank_epilogue_lds_addressing == "expanded"
        ):
            raise ValueError(
                "staged_reduce requires vec8/load_first/static_strided "
                "reduction and rejoin_blocks=0"
            )
        if stage2_rank_accumulation_mode == "staged_ring" and not (
            int(stage2_node_reduce_vec_bytes) == 8
            and stage2_node_reduce_load_schedule == "load_first"
            and stage2_node_reduce_work_schedule == "static_strided"
            and stage2_node_reduce_rejoin_blocks == 0
            and stage2_rank_epilogue_lds_addressing == "expanded"
        ):
            raise ValueError(
                "staged_ring requires vec8/load_first/static_strided reduction "
                "and rejoin_blocks=0"
            )
        self.stage2_node_accumulation_mode = "rank_local"
        self.stage2_node_reduce_blocks = int(stage2_node_reduce_blocks)
        self.stage2_node_reduce_vec_bytes = int(stage2_node_reduce_vec_bytes)
        self.stage2_node_reduce_schedule = str(stage2_node_reduce_schedule)
        self.stage2_node_reduce_load_schedule = str(
            stage2_node_reduce_load_schedule
        )
        self.stage2_node_reduce_work_schedule = str(
            stage2_node_reduce_work_schedule
        )
        self.stage2_node_reduce_rejoin_blocks = (
            stage2_node_reduce_rejoin_blocks
        )
        self.stage2_rank_epilogue_lds_addressing = str(
            stage2_rank_epilogue_lds_addressing
        )
        self.stage2_rank_accumulation_mode = str(stage2_rank_accumulation_mode)
        self.stage2_return_chunk_tokens = int(stage2_return_chunk_tokens)
        self.stage2_rail_return_schedule = str(stage2_rail_return_schedule)
        # One RAIL role, the configured node reducers, 14 final-combine roles,
        # and the original 145 persistent GMM2 workers.
        self.stage2_worker_blocks = (
            160
            + self.stage2_node_reduce_blocks
            + (1 if stage2_rank_accumulation_mode == "staged_ring" else 0)
        )
        gmm_cta_count = self.stage2_worker_blocks - (
            1
            + self.stage2_node_reduce_blocks
            + 14
            + (1 if stage2_rank_accumulation_mode == "staged_ring" else 0)
        )
        if self.stage2_node_reduce_rejoin_blocks > gmm_cta_count:
            raise ValueError(
                "stage2_node_reduce_rejoin_blocks exceeds the available "
                f"GMM2 CTA count ({gmm_cta_count})"
            )
        super().__init__(*args, **kwargs)

    def _validate_launcher_contracts(self) -> None:
        super()._validate_launcher_contracts()
        expected = {
            "node_accumulation_mode": "rank_local",
            "node_reduce_blocks": self.stage2_node_reduce_blocks,
            "node_reduce_vec_bytes": self.stage2_node_reduce_vec_bytes,
            "node_reduce_schedule": self.stage2_node_reduce_schedule,
            "node_reduce_load_schedule": self.stage2_node_reduce_load_schedule,
            "node_reduce_work_schedule": self.stage2_node_reduce_work_schedule,
            "node_reduce_rejoin_blocks": self.stage2_node_reduce_rejoin_blocks,
            "rank_epilogue_lds_addressing": self.stage2_rank_epilogue_lds_addressing,
            "rank_accumulation_mode": self.stage2_rank_accumulation_mode,
            "return_chunk_tokens": self.stage2_return_chunk_tokens,
            "rail_return_schedule": self.stage2_rail_return_schedule,
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
                "rank-local Stage2 launcher contract mismatch: "
                f"launcher={launcher_mismatch}, manifest={manifest_mismatch}"
            )


__all__ = ["MegaMoETileA4W4RankLocal"]
