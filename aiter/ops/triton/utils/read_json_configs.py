import json
import re
from typing import Any, Dict, List, Optional
import triton


def read_configs_from_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_triton_configs_from_json(json_path: str):

    raw_configs = read_configs_from_json(json_path)
    triton_configs = []

    for rc in raw_configs:
        cfg_dict = {
            "BLOCK_SIZE_M": rc.get("BLOCK_SIZE_M"),
            "BLOCK_SIZE_N": rc.get("BLOCK_SIZE_N"),
            "BLOCK_SIZE_K": rc.get("BLOCK_SIZE_K"),
            "GROUP_SIZE_M": rc.get("GROUP_SIZE_M"),
            "waves_per_eu": rc.get("waves_per_eu"),
            "matrix_instr_nonkdim": rc.get("matrix_instr_nonkdim"),
            "cache_modifier": rc.get("cache_modifier"),
        }

        if not cfg_dict["cache_modifier"]:
            cfg_dict["cache_modifier"] = None

        num_warps = rc.get("num_warps")
        num_stages = rc.get("num_stages")

        triton_configs.append(
            triton.Config(cfg_dict, num_warps=num_warps, num_stages=num_stages)
        )

    return triton_configs
