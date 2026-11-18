"""Assert installed amd-triton meets the minimum version requirement (>=3.6.0)
and that its module files are actually importable."""

import sys
from importlib.metadata import version

v = version("amd-triton")
parts = tuple(int(p) for p in v.split("+")[0].split("-")[0].split(".") if p.isdigit())
if parts < (3, 6, 0):
    sys.exit(f"amd-triton>=3.6.0 is required, found {v}")

try:
    import triton.language  # noqa: F401
except ModuleNotFoundError:
    sys.exit(
        f"amd-triton {v} metadata is present but triton.language cannot be imported. "
        "This usually means pip uninstall of an old triton variant deleted shared files."
    )

print(f"amd-triton {v}")
