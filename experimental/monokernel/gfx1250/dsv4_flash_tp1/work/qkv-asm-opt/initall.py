from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_store
@g.jit
def initialize_all(Acc,Count):
 i=gl.program_id(0)*256+gl.arange(0,256,layout=gl.BlockedLayout([1],[32],[4],[0]))
 dev_store(0.,Acc,i,i<1536)
 dev_store(0,Count,i,i<1536)
