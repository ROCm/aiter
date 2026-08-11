import functools
import json
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_TRITON_OPS_PATH = os.path.abspath(f"{this_dir}/../")
AITER_TRITON_CONFIGS_PATH = os.path.abspath(f"{this_dir}/../configs")

# This flag should be set to True, unless it is being used for debugging.
# When False, config JSON files are re-read on every call, so live edits to
# the JSON are picked up.
USE_LRU_CACHE = True


@functools.lru_cache(maxsize=None if USE_LRU_CACHE else 0)
def load_config_json(fpath: str, required: bool = True) -> dict | None:
    """Load a config JSON file, cached per path (including negative results --
    add config files before process start, or call
    ``load_config_json.cache_clear()``). Raises FileNotFoundError if the file
    doesn't exist, consistently on every call (exceptions are never cached);
    pass required=False for probe/fallback lookups to get None instead.

    The returned dict is the shared cached object -- copy before mutating:
    a shallow ``.copy()`` suffices for flat bucket dicts (scalar values),
    ``copy.deepcopy`` when nested sub-dicts will be mutated."""
    try:
        with open(fpath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(
                f"Required config file doesn't exist: {fpath}"
            ) from None
        return None
