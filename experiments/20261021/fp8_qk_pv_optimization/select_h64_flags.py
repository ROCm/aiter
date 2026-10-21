import re
import shutil
from pathlib import Path

root = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-pv-opt-20261021")
out = root / "h64_flags_prefetch"
out.mkdir(exist_ok=True)
assert not any(out.iterdir())
h = (root / "flags_prefetch/decode_fp8_qk.hpp").read_text()
c = (root / "flags_prefetch/fused_dsv41_mxfp8.cu").read_text()
h, count = re.subn(
    r"        const auto& flags\s*=\s*tile_flags;",
    """        vector_t<uint32_t, 4> flags;
        if constexpr(LateCommit)
            flags = tile_flags;
        else
            flags = *reinterpret_cast<OPUS_LDS_ADDR const vector_t<uint32_t, 4>*>(raw_cur + 32 * 528);""",
    h,
)
assert count == 1, count
start = h.index("        // Issue validity before K operands.")
end = h.index("        compute_qk(v_s);", start)
block = h[start:end]
h = (
    h[:start]
    + "        if constexpr(LateCommit)\n        {\n"
    + block
    + "        }\n"
    + h[end:]
)
# The same existing H64 specialization now controls commit, scale reuse and
# early flags; name the private template parameter for its complete role.
h = h.replace("LateCommit", "H64Schedule")
c = c.replace("LateCommit", "H64Schedule")
(out / "decode_fp8_qk.hpp").write_text(h)
(out / "fused_dsv41_mxfp8.cu").write_text(c)
if (root / "release" / out.name).exists():
    for p in out.iterdir():
        shutil.copy2(p, root / "release" / out.name / p.name)
print("SELECTED", out.name)
