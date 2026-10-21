import hashlib
import json
import re
import subprocess
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-pv-opt-20261021"
work = base / "fp8-qk-pv-opt-20261021/release"
print(subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True))
print(subprocess.check_output(["git", "diff", "--stat"], cwd=repo, text=True))
print(
    subprocess.check_output(
        [
            "git",
            "diff",
            "--",
            "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp",
            "csrc/kernels/mla/fused_dsv41_mxfp8.cu",
        ],
        cwd=repo,
        text=True,
    )
)
prov = json.loads((work / "final_validation/provenance.json").read_text())
for name, digest in prov["source_sha256"].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
out = {}
for tag, path in [
    ("baseline", base / "fp8-qk-profile-opt-20261021/release/final/isa/fp8_qk.isa"),
    ("final", work / "final/isa/fp8_qk.isa"),
]:
    if not path.exists():
        continue
    parts = re.split(r"^([0-9a-f]+ <.*>):\n", path.read_text(), flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        if "fused_dsv41_fp8_qk_decode_64<true, 0, false, false>" not in parts[i]:
            continue
        body = parts[i + 1]
        prep = body[: body.index("v_mfma_scale")]
        out[tag] = {
            k: len(re.findall(r"\b" + k + r"\b", prep))
            for k in [
                "ds_bpermute_b32",
                "v_cvt_scalef32_pk_fp8_f32",
                "global_load_dwordx4",
                "global_load_dwordx2",
            ]
        }
if len(out) == 2:
    (work / "qdirect_isa_comparison.json").write_text(json.dumps(out, indent=2))
print("QDIRECT_ISA", json.dumps(out))
print("SOURCE_HASHES", json.dumps(prov["source_sha256"]))
