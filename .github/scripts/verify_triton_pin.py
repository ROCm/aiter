"""Assert installed triton meets the minimum version requirement (>=3.6.0)."""

import sys
from importlib.metadata import version

v = version("triton")
parts = tuple(int(p) for p in v.split("+")[0].split("-")[0].split(".") if p.isdigit())
if parts < (3, 6, 0):
    sys.exit(f"triton>=3.6.0 is required, found {v}")
print(f"triton {v}")
