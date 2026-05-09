"""Compatibility patches for FlyDSL ROCDL helpers used by AITER kernels."""

from flydsl._mlir.ir import IntegerType
from flydsl.expr import arith as _arith_ext
from flydsl.expr import rocdl
from flydsl.expr.meta import traced_op


def _unwrap_mfma_operand(value, *, loc=None):
    unwrap = getattr(rocdl, "_unwrap_mfma_operand", None)
    if unwrap is not None:
        return unwrap(value, loc=loc)
    if isinstance(value, int):
        return _arith_ext.unwrap(
            _arith_ext.constant(
                value,
                type=IntegerType.get_signless(32),
                loc=loc,
            ),
            loc=loc,
        )
    return _arith_ext.unwrap(value, loc=loc)


def ensure_mfma_scale_f32_32x32x64_f8f6f4():
    """Install the traced wrapper for the MFMA32/K64 FP4 scale op if needed.

    Some FlyDSL builds expose only the generated ODS op class. AITER calls this
    op with the same list-style operand convention as the 16x16x128 wrapper, so
    patch the wrapper locally instead of requiring users to edit site-packages.
    """
    name = "mfma_scale_f32_32x32x64_f8f6f4"
    ods_name = f"_ods_{name}"
    current = getattr(rocdl, name, None)

    # A traced wrapper is a Python function produced by functools.wraps.
    if current is not None and not isinstance(current, type):
        return current

    ods_op = getattr(rocdl, ods_name, None) or current or getattr(rocdl, f"{name}_", None)

    @traced_op
    def mfma_scale_f32_32x32x64_f8f6f4(result_type, operands, *, loc=None, ip=None):
        if ods_op is None:
            raise AttributeError("ROCDL op not found: mfma_scale_f32_32x32x64_f8f6f4(_)")
        a = _unwrap_mfma_operand(operands[0], loc=loc)
        b = _unwrap_mfma_operand(operands[1], loc=loc)
        c = _unwrap_mfma_operand(operands[2], loc=loc)
        cbsz = int(operands[3]) if len(operands) > 3 else 0
        blgp = int(operands[4]) if len(operands) > 4 else 0
        opsel_a = int(operands[5]) if len(operands) > 5 else 0
        scale_a = _unwrap_mfma_operand(operands[6], loc=loc) if len(operands) > 6 else a
        opsel_b = int(operands[7]) if len(operands) > 7 else 0
        scale_b = _unwrap_mfma_operand(operands[8], loc=loc) if len(operands) > 8 else b
        return ods_op(
            result_type,
            a,
            b,
            c,
            cbsz,
            blgp,
            opsel_a,
            scale_a,
            opsel_b,
            scale_b,
            loc=loc,
            ip=ip,
        ).result

    if ods_op is not None and getattr(rocdl, ods_name, None) is None:
        setattr(rocdl, ods_name, ods_op)
    setattr(rocdl, name, mfma_scale_f32_32x32x64_f8f6f4)
    return mfma_scale_f32_32x32x64_f8f6f4
