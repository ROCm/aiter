import sys,torch,json
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native/prefix']
from attention_batch import attention_body
from attention_matched import attention_kernel
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def batch(Q,KV,C,I,P,S,MAX,SUM,A,O,COUNT,COUNTER,H,M:gl.constexpr,T:gl.constexpr,SPLITS:gl.constexpr):
 attention_body(Q,KV,C,I,P,S,MAX,SUM,A,O,COUNT,COUNTER,H,M,T,SPLITS)
def scratch(m,t,s):
 return [torch.empty((m,s,64),device='cuda'),torch.empty((m,s,64),device='cuda'),torch.empty((m,s,64,512),device='cuda'),torch.empty((m,64,512),device='cuda',dtype=torch.bfloat16),torch.zeros(4,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32)]
