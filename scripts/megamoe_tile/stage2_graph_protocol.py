"""Validate recorded Graph protocol state without treating missing data as zero."""


def validate_protocol_snapshot(snapshot, *, expected_generation, readiness, tokens,
                               accumulation_mode="atomic"):
    expected = {
        "generation": expected_generation,
        "stage1_error_count": 0,
        "stage2_error_count": 0,
    }
    if accumulation_mode == "reduce_push":
        if snapshot.get("rank_accumulation_mode") != "reduce_push":
            raise ValueError("protocol snapshot missing or incorrect rank_accumulation_mode")
        if readiness != "group":
            raise ValueError("reduce_push protocol requires group readiness")
        expected.update({
            "rank_push_invalid_key_count": 0,
            "rank_push_duplicate_key_count": 0,
            "rank_push_row_map_mismatch": 0,
            "rank_push_route_count_mismatch": 0,
            "rank_push_local_arrival_mismatch": 0,
            "rank_push_node_mask_invalid": 0,
            "rank_push_node_arrival_mismatch": 0,
            "node_ready_mask_full_count": 2 * tokens,
            "tile_partial_ready_count": 2 * tokens,
        })
    elif snapshot.get("rank_accumulation_mode") == "reduce_push":
        raise ValueError("reduce_push snapshot requires the reduce_push validator")
    elif readiness == "group":
        expected.update({
            "rank_group_pending_mismatch": 0,
            "rank_group_ready_missing": 0,
            "node_ready_mask_full_count": 2 * tokens,
            "tile_partial_ready_count": 2 * tokens,
        })
    values = {}
    if accumulation_mode == "reduce_push":
        values["rank_accumulation_mode"] = accumulation_mode
    for name, value in expected.items():
        if name not in snapshot:
            raise ValueError(f"protocol snapshot missing {name}")
        values[name] = int(snapshot[name])
        if values[name] != value:
            raise ValueError(f"protocol {name}={values[name]}, expected {value}")
    return values
