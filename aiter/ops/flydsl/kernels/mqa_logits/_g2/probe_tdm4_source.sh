#!/bin/bash
set -euo pipefail
STAGE=/data/blyu/qsa/work/opt-tdm4-cpu-trace-20261031
mkdir -p "$STAGE"
docker run --rm --entrypoint bash rocm/fw-bringup-mlperf:gfx1250-atom-20261015 -c 'python3 - <<'"'"'PY'"'"'
import importlib.metadata as m
import inspect
from flydsl.expr.rocdl import tdm_ops
print("FlyDSL", m.version("flydsl"))
assert m.version("flydsl") == "0.3.3.dev886"
print("SOURCE_PATH", inspect.getsourcefile(tdm_ops))
print(inspect.getsource(tdm_ops.compute_padding_encoding))
print(inspect.getsource(tdm_ops.make_tensor_gather_descriptor))
print("FP8_ENCODING", tdm_ops.compute_padding_encoding(128,16,8))
assert tdm_ops.compute_padding_encoding(128,16,8)==(4,3)
print("BF16_ENCODING", tdm_ops.compute_padding_encoding(128,8,16))
assert tdm_ops.compute_padding_encoding(128,8,16)==(5,3)
PY' > "$STAGE/stock-padding-source.txt"
cat "$STAGE/stock-padding-source.txt"
