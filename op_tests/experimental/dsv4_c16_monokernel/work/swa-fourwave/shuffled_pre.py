import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
G=BoundExternFunction('mhc_native_pre_gemm_shuffled',['uint64']*4+['uint32']*3,'void',bitcode_path='/work/swa-fourwave/pre_stage.bc')
@flyc.jit
def gemm(o:fx.Uint64,sq:fx.Uint64,x:fx.Uint64,w:fx.Uint64,logical:fx.Int32):
 G(o,sq,x,w,fx.Uint32(0),fx.Uint32(logical%2),fx.Uint32(logical//2))
