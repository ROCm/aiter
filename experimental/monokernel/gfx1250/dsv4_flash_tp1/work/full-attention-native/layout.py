from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_store

@g.jit
def unshuffle_body(R,RPlain):
    pid=gl.program_id(0)
    if pid<64:
        SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,64],[1,0])
        LL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
        sm=gl.allocate_shared_memory(R.dtype.element_ty,[4,64],SL)
        desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=R+pid*256,shape=[4,64],strides=[64,1],block_shape=[4,64],layout=SL)
        gl.amd.gfx1250.tdm.async_load(desc,[0,0],sm)
        gl.amd.gfx1250.tdm.async_wait(0)
        v=sm.load(LL)
        h=gl.arange(0,4,layout=gl.SliceLayout(1,LL));k=gl.arange(0,64,layout=gl.SliceLayout(0,LL))
        dev_store(v,RPlain,h[:,None]*4096+pid*64+k[None,:])

@g.jit
def unshuffle_kernel(R,RPlain):
    unshuffle_body(R,RPlain)
