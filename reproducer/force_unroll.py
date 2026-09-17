"""Pin the KV loop's unroll factor so it cannot vary between the arms under test.

    import force_unroll
    force_unroll.enable(1)    # the setting every arm shares a loop body under
    force_unroll.enable(2)    # what aiter ships

aiter passes `UNROLL=2`, and the kernel maps it to `tl.range(..., loop_unroll_factor=...)`:

    unroll: gl.constexpr = UNROLL if UNROLL > 1 else None

so `UNROLL<=1` leaves the factor to the backend and `UNROLL=2` pins it at two.
Neither is automatically the same loop on both compilers: at the shipped
`UNROLL=2`, 3.7.0 produced 8 `v_mfma` per body with packing on and 16 with it
off, so the loop shape was itself a variable in the comparison.

`UNROLL=1` turns out to be the stable choice -- measured, not assumed: all four
arms (2 versions x 2 packing settings) compile to a 4-`v_mfma` body with no
further unrolling from LLVM. That makes the VALU counts directly comparable
without normalising.

This overrides the constexpr at the launch site, so the kernel under test is
byte-for-byte what aiter ships.
"""
import aiter.ops.triton.attention.fp8_mqa_logits as _mod

_installed = False
_value = 1


def enable(value=1):
    """Force UNROLL to `value` on every launch of the fp8 MQA-logits kernel.

    1 is what this reproducer uses: it is the setting under which every arm
    compiles to the same loop body. 2 is what aiter ships.
    """
    global _installed, _value
    _value = value
    if _installed:
        return
    kern = getattr(_mod, "_gluon_fp8_mqa_logits_kernel", None)
    if kern is None:
        raise RuntimeError("_gluon_fp8_mqa_logits_kernel not importable; "
                           "is this a gfx950 build?")
    orig = kern.run

    def patched(*args, **kwargs):
        if "UNROLL" in kwargs:
            kwargs["UNROLL"] = _value
        return orig(*args, **kwargs)

    kern.run = patched
    _installed = True


def value():
    return _value
