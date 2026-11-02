"""QB native tile with a whole-CTA barrier in the four-wave parent."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from native_gemm import build_native_gemm_body
QB,QB_LDS=build_native_gemm_body(128,512,4,2,parent_waves=4)
def ptr(value,dtype=fx.Uint8):
 return fx.inttoptr(fx.PointerType.get(dtype.ir_type,address_space=fx.AddressSpace.Global,alignment=dtype.width//8),value)

@flyc.jit
def qb_body(out:fx.Uint64,x:fx.Uint64,w:fx.Uint64,s:fx.Uint64,ws:fx.Uint64,logical:fx.Int32):
 QB(ptr(out),ptr(x),ptr(w),ptr(s),ptr(ws),fx.Int32(16),fx.Int32(32768),fx.Int32(1024),fx.Int32(16),fx.Int32(1024),fx.Int32(32768),logical,fx.get_dyn_shared())
