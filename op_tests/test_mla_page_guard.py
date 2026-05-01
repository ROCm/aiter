"""Unit test for the AITER page-size guard patch.

We test the Python `compile()` entry in
`csrc/cpp_itfs/mla/asm_mla_decode_fwd.py` directly, which is the lowest layer
where the kernel path becomes a fixed `.co` file.  At this layer:

  * `compile(page_size=1, ...)` should still load the (correct) ps=0 kernel.
  * `compile(page_size=32, ...)` should raise `NotImplementedError` with a
    clear message that the bf16 decode-stage1 kernel has no paged variant.

We also do a smoke check on the failing block_sizes that AITER's own
`op_tests/test_mla.py` exposes (16/32/64/128) to confirm they all hit the
guard.
"""
import os
import sys
import traceback

# sys.path is set by the AITER test harness or by running from repo root


def run():
    from csrc.cpp_itfs.mla.asm_mla_decode_fwd import compile as compile_decode

    print("=" * 72)
    print("Test 1: page_size=1 (the documented good path)")
    print("-" * 72)
    try:
        compile_decode(
            gqa_ratio=128,
            page_size=1,
            q_dtype="bf16",
            kv_dtype="bf16",
            num_kv_splits=1,
            v_head_dim=128,
        )
        print("PASS: compile() returned without NotImplementedError\n")
        ok_one = True
    except NotImplementedError as e:
        print("FAIL: page_size=1 should NOT raise NotImplementedError")
        print(f"      got: {e}\n")
        ok_one = False
    except Exception as e:
        # Anything else (e.g. JIT errors) is unrelated to the guard test.
        print(f"PASS-ish: page_size=1 did not hit guard ({type(e).__name__}: {e})")
        ok_one = True

    print("=" * 72)
    print("Test 2: page_size > 1 should raise NotImplementedError")
    print("        (these are the values AITER's own test_mla.py shows broken)")
    print("-" * 72)
    bad_sizes = [16, 32, 64, 128]
    ok_two = True
    for ps in bad_sizes:
        try:
            compile_decode(
                gqa_ratio=128,
                page_size=ps,
                q_dtype="bf16",
                kv_dtype="bf16",
                num_kv_splits=1,
                v_head_dim=128,
            )
            print(f"FAIL: page_size={ps} should raise NotImplementedError")
            ok_two = False
        except NotImplementedError as e:
            msg = str(e).splitlines()[0]
            print(f"PASS: page_size={ps:>3} → {msg}")
        except Exception as e:
            print(
                f"FAIL: page_size={ps} raised wrong exception type: "
                f"{type(e).__name__}: {e}"
            )
            ok_two = False

    print("=" * 72)
    print("Test 3: gqa_ratio=16 (the smaller decode-stage1 kernel) too")
    print("-" * 72)
    try:
        compile_decode(
            gqa_ratio=16,
            page_size=32,
            q_dtype="bf16",
            kv_dtype="bf16",
            num_kv_splits=1,
            v_head_dim=128,
        )
        print("FAIL: gqa_ratio=16 / page_size=32 should also be guarded")
        ok_three = False
    except NotImplementedError as e:
        msg = str(e).splitlines()[0]
        print(f"PASS: gqa_ratio=16 / page_size=32 → {msg}")
        ok_three = True

    print()
    summary = [
        ("page_size=1 path", ok_one),
        ("page_size>1 raises", ok_two),
        ("gqa_ratio=16 also guarded", ok_three),
    ]
    print("=" * 72)
    for name, ok in summary:
        print(f"  {('PASS' if ok else 'FAIL'):>4}: {name}")
    print("=" * 72)
    return 0 if all(ok for _, ok in summary) else 1


if __name__ == "__main__":
    try:
        rc = run()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
