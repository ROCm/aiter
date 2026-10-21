import json
import subprocess
import sys
from pathlib import Path

work = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021/release")
mode, case, kind = sys.argv[1:4]
tag = sys.argv[4] if len(sys.argv) > 4 else "baseline"
label = f"{tag}_{mode}_{case}_{kind}"
if mode == "pr":
    label += "_v2"
out = work / label
assert not out.exists(), label
regex = (
    "fused_dsv41_fp8_qk_decode_64"
    if mode != "pr"
    else "(mla_v41|q_prepare|inverse_rope)"
)
cmd = [
    "rocprofv3",
    "--kernel-include-regex",
    regex,
    "--kernel-iteration-range",
    "3-3",
    "-d",
    str(out),
    "-o",
    "profile",
    "-f",
    "csv",
]
groups = {
    "inst": ["SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_INSTS_LDS", "SQ_WAVES"],
    "vmem": ["SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_SALU"],
    "occ": ["LDSBankConflict", "MeanOccupancyPerCU", "OccupancyPercent"],
    "busy": [
        "GRBM_GUI_ACTIVE",
        "SQ_BUSY_CU_CYCLES",
        "SQ_ACTIVE_INST_ANY",
        "SQ_ACTIVE_INST_MFMA",
    ],
    "l2": ["TCC_HIT", "TCC_MISS"],
}
if kind == "att":
    cmd += [
        "--att",
        "--att-library-path",
        "/opt/rocm/lib",
        "--att-target-cu",
        "1",
        "--att-shader-engine-mask",
        "0x1",
        "--att-buffer-size",
        "134217728",
        "--att-activity",
        "8",
    ]
elif kind == "trace":
    cmd += ["--kernel-trace", "--stats"]
else:
    cmd += ["--pmc", *groups[kind]]
cmd += ["--", sys.executable, str(work / "workload.py"), mode, case, tag]
print("RUN", cmd, flush=True)
result = subprocess.run(
    cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300
)
(work / (label + ".log")).write_text(result.stdout)
(work / (label + ".json")).write_text(
    json.dumps(dict(command=cmd, exit_code=result.returncode), indent=2)
)
print(result.stdout[-6000:])
result.check_returncode()
print(
    "FILES",
    [
        (str(p.relative_to(out)), p.stat().st_size)
        for p in out.rglob("*")
        if p.is_file()
    ][:60],
)
