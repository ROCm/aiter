
from triton.knobs import amd
from functools import lru_cache

@lru_cache(maxsize=1)
def get_env():
    return amd.knobs