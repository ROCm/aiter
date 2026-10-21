"""Clarify the manual LDS schedule before the final formatted rebuild."""

from pathlib import Path

w = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021/release")
p = w / "dequant16_qkds_storeflip_h64late/decode_fp8_qk.hpp"
h = p.read_text()
h = h.replace(
    "                vector_t<uint32_t, 4> words;",
    """                // Explicit DS reads keep LLVM from strengthening the pipelined
                // partial waits below. Immediate offsets avoid per-load VALU work.
                vector_t<uint32_t, 4> words;""",
)
h = h.replace(
    "            const bool flip = (lane & 4) != 0;",
    """            // The two lane groups write opposite halves first to reduce LDS bank
            // overlap. Both stores retain exactly the original BF16 V layout.
            const bool flip = (lane & 4) != 0;""",
)
p.write_text(h)
print("COMMENTS_READY")
