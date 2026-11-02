python3 - <<'PY' || exit 1
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys

wheel = Path("flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl")
try:
    installed = version("flydsl")
except PackageNotFoundError:
    if not wheel.is_file():
        raise SystemExit(f"FlyDSL is not installed and local wheel is missing: {wheel}")
    print(f"FlyDSL not found; installing local wheel: {wheel}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
        check=True,
    )
    installed = version("flydsl")
if installed != "0.3.3.dev882":
    raise SystemExit(
        f"Expected flydsl==0.3.3.dev882, found {installed}. "
        f"Install locally with: python3 -m pip install --no-deps ./{wheel}"
    )
PY

export AITER_FLYDSL_MMA_FIRST_GROUP=4
export AITER_FLYDSL_MMA_GROUP=4
export AITER_FLYDSL_DS_FIRST_N=8



  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=1 \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --scenario kernel \
    --data-format a4w4 \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048 \
    --act silu \
    --no-bias \
    --iters 128 \
    --no-check-aot-cache \
    --data-init ${AITER_DATA_INIT:-uniform} \
    --scale-init ${AITER_SCALE_INIT:-uniform}
