"""Assert installed triton matches the pin in the given requirements file."""

import re
import sys
from importlib.metadata import version

req_path = sys.argv[1]
with open(req_path) as f:
    m = re.search(r"^triton==(\S+)", f.read(), re.M)
if not m:
    sys.exit(f"no triton pin found in {req_path}")
expected = m.group(1)

got = version("triton")
if got != expected:
    sys.exit(f"triton mismatch: expected {expected}, got {got}")
print(f"triton {got}")
