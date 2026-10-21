"""Combine independently validated improvements, retaining native scale reads for H128."""

import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
work = base / "fp8-qk-pv-opt-20261021"
tag = sys.argv[1]
out = work / tag
out.mkdir()
h = (work / "cached_scales/decode_fp8_qk.hpp").read_text()
c = (work / "qdirect/fused_dsv41_mxfp8.cu").read_text()
if "valid" in tag:
    v = (work / "valid_fast/decode_fp8_qk.hpp").read_text()
    a = h.index("    template <class RawPointer>")
    z = h.index("    template <class Pointer, class RawPointer>", a)
    va = v.index("    template <class RawPointer>")
    vz = v.index("    template <class Pointer, class RawPointer>", va)
    h = h[:a] + v[va:vz] + h[z:]
    a = h.index("    auto mask_scores =")
    z = h.index("    // K is A and Q is B:", a)
    va = v.index("    auto mask_scores =")
    vz = v.index("    // K is A and Q is B:", va)
    h = h[:a] + v[va:vz] + h[z:]
h = h.replace(
    "template <class Pointer, class RawPointer>",
    "template <bool CacheScales, class Pointer, class RawPointer>",
)
old = "float scale = __uint_as_float(((scale_words[int(j)] >> (((lane % 8) / 2) * 8)) & 255u) << 23);"
assert old in h
h = h.replace(
    old,
    """float scale;
            if constexpr(CacheScales)
                scale = __uint_as_float(((scale_words[int(j)] >> (((lane % 8) / 2) * 8)) & 255u) << 23);
            else
                scale = __uint_as_float(uint32_t(bytes[row * 528 + 512 + d / 32]) << 23);""",
)
h = h.replace(
    "current_scales = initial.scales;",
    "if constexpr(LateCommit)\n            current_scales = initial.scales;",
)
h = h.replace(
    "        current_scales = next_raw.scales;",
    "        if constexpr(LateCommit)\n            current_scales = next_raw.scales;",
)
h = h.replace(
    "source.dequant_current(raw_cur, s_kv.ptr, warp_id, lane_id, current_scales);",
    """// The H64 schedule carries scale words across iterations; H128 keeps
        // native LDS scale reads to avoid its measured register-pressure cost.
        source.template dequant_current<LateCommit>(raw_cur, s_kv.ptr, warp_id, lane_id, current_scales);""",
)
(out / "decode_fp8_qk.hpp").write_text(h)
(out / "fused_dsv41_mxfp8.cu").write_text(c)
print("COMBINED", tag)
