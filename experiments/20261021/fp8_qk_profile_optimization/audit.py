# Reuse pinned local fixture/adapter source to preserve the comparison contract.
# ruff: noqa: S102
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
work = base / "fp8-qk-profile-opt-20261021/release"
tag = sys.argv[1]
(work / tag).mkdir(exist_ok=True)
src = (base / "fp8-qk-20261021/audit_isa.py").read_text()
src = src.replace(
    "work=Path('/home/jun_chen2_qle/yzhou_aiter/fp8-qk-20261021')",
    f"work=Path({str(work)!r})",
)
src = src.replace(
    "work/'jit/v4/module_fused_dsv41_mxfp8.so'",
    f"work/'jit/{tag}/module_fused_dsv41_mxfp8.so'",
)
src = src.replace("out=work/'isa'", f"out=work/'{tag}/isa'")
src = src.replace("print(json.dumps(report,indent=2))", "")
src = src.replace("fused_dsv41_fp8_qk_decode_64", "fused_dsv41_fp8_qk_decode_")
src = src.replace(
    "assert len(instructions)==12,len(instructions)",
    "assert len(instructions) in (12,24),len(instructions)",
)
exec(compile(src, "audit_current", "exec"), ns := {})
report = ns["report"]
assert len(report["instructions"]) in (12, 24)
assert all(
    r["scaled_fp8"] > 0 and r["bf16"] > 0 and r["scratch"] == 0
    for r in report["instructions"].values()
)
for r in report["resources"]:
    if "<true, 0, false" in r["symbol"] or "<true, 128, false" in r["symbol"]:
        print(
            "RESOURCE",
            r["symbol"],
            {
                k: r[k]
                for k in [
                    ".vgpr_count",
                    ".agpr_count",
                    ".sgpr_count",
                    ".private_segment_fixed_size",
                    ".group_segment_fixed_size",
                ]
            },
        )
print("COUNTS", report["instructions"])
