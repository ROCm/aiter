# SPDX-License-Identifier: MIT
"""Numerically gate the public M3 Gluon dispatch before model startup."""

import importlib
from unittest.mock import patch

from test_unified_attention_scale_cache import check_case


def main():
    ua = importlib.import_module("aiter.ops.triton.attention.unified_attention")
    assert ua.DEVICE_ARCH == "gfx950", "This gate must run on the target GFX950 GPU"
    assert ua._unified_attention_kernel_gfx950 is not None
    original = ua._unified_attention_gfx950
    original_kernel = ua._unified_attention_kernel_gfx950

    class GlobalLoads:
        # Real draft pools exceed the buffer loader's 2 GiB address range.
        # Exercise that loader on small physical pools against the same oracle.
        def __getitem__(self, grid):
            def launch(**kwargs):
                kwargs["USE_LOAD_BUFFER_OP"] = False
                return original_kernel[grid](**kwargs)

            return launch

    launches = []

    def record(*args, **kwargs):
        launches.append(kwargs.get("NUM_SEGMENTS", 1))
        return original(*args, **kwargs)

    def forbid_fallback(*args, **kwargs):
        raise AssertionError("Gluon experiment silently fell back to Triton attention")

    cases = [
        {"heads": 64, "kv_heads": kv, "q_lens": [1, 4, 19], "k_lens": [257, 2049, 3073]}
        for kv in (1, 2, 4, 8, 16, 32, 64)
    ]
    cases += [
        {"heads": 16, "kv_heads": 1, "q_lens": [q], "k_lens": [16385]}
        for q in (1, 2, 4)
    ]
    cases += [
        {
            "heads": 16,
            "kv_heads": kv,
            "q_lens": [1, 17, 255, 256, 513],
            "k_lens": [257, 273, 1023, 1037, 2049],
        }
        for kv in (1, 16)
    ]
    cases += [
        {"heads": 16, "kv_heads": 1, "q_lens": [1, 4], "k_lens": [65539, 131073]},
        {"heads": 16, "kv_heads": 16, "q_lens": [1024, 1536], "k_lens": [2049, 3073]},
        {"heads": 16, "kv_heads": 1, "q_lens": [1, 3], "k_lens": [7, 31]},
    ]
    with (
        patch.object(ua, "_unified_attention_gfx950", record),
        patch.object(ua, "_unified_attention_2d_triton", forbid_fallback),
        patch.object(ua, "_unified_attention_3d_triton", forbid_fallback),
    ):
        for case in cases:
            before = len(launches)
            check_case("shuffle", "token", False, page=128, backend="gluon", **case)
            assert len(launches) == before + 1, "Expected an actual Gluon launch"
        with patch.object(ua, "_unified_attention_kernel_gfx950", GlobalLoads()):
            for case in (cases[8], cases[13]):
                check_case("shuffle", "token", False, page=128, backend="gluon", **case)
    assert 1 in launches and any(n > 1 for n in launches), launches
    print(
        f"[gluon-c32-gate] {len(cases) + 2} numerical cases passed; "
        f"real Gluon split counts={launches}; no Triton attention fallback",
        flush=True,
    )


if __name__ == "__main__":
    main()
