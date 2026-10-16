import triton, inspect
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4
print("triton", triton.__version__)
print([n for n in dir(gl) if n in ("fma","to_linear_layout","split","join","gather","inline_asm_elementwise","reduce","static_range","umulhi","div_rn","sqrt_rn")])
print([n for n in dir(cdna4) if not n.startswith("_")])
from triton.experimental.gluon.language import _layouts as L
print([n for n in dir(gl) if "layout" in n.lower()])
MFMA = cdna4.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[8, 1])
A_L = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=16)
B_L = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=16)
try:
    from triton._C.libtriton.gluon_ir import compute_linear_layout  # noqa
    print("has compute_linear_layout")
except Exception as e:
    print("no compute_linear_layout", e)
print(inspect.signature(gl.to_linear_layout) if hasattr(gl, "to_linear_layout") else "no to_linear_layout")
