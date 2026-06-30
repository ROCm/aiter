# SPDX-License-Identifier: MIT
# CUDA-graph end-to-end bench: fused gdn_chunk_prepare (1 kernel) vs Triton
# non-fused k14 (chunk_local_cumsum + chunk_scaled_dot_kkt_fwd + solve_tril +
# recompute_w_u_fwd, 4 kernels). Also reports precision alignment vs Triton.
import torch, torch.nn.functional as F
from aiter.ops.gdn_chunk_prepare import gdn_chunk_prepare_fwd
from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
    chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd,
)

def make(B,T,H,K=128,V=128,seed=0):
    torch.manual_seed(seed); dev="cuda"
    k=F.normalize(torch.randn(B,T,H,K,dtype=torch.bfloat16,device=dev),p=2,dim=-1)
    v=torch.randn(B,T,H,V,dtype=torch.bfloat16,device=dev)*0.5
    g=F.logsigmoid(torch.randn(B,T,H,dtype=torch.float32,device=dev))
    beta=torch.rand(B,T,H,dtype=torch.float32,device=dev)
    return k,v,g,beta

def triton_k14(k,v,g,beta):
    g_cs=chunk_local_cumsum(g,chunk_size=64,cu_seqlens=None)
    A=chunk_scaled_dot_kkt_fwd(k=k,g=g_cs,beta=beta,cu_seqlens=None,output_dtype=torch.float32)
    A=solve_tril(A=A,cu_seqlens=None,output_dtype=k.dtype)
    w,u=recompute_w_u_fwd(k=k,v=v,beta=beta,A=A,g=g_cs,cu_seqlens=None)
    return g_cs,w,u

def opus_k14(k,v,g,beta,w,u,gc):
    gdn_chunk_prepare_fwd(k,v,g,beta,w,u,gc)

def graph_time(fn, iters=50):
    # warmup on side stream (required before capture)
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/iters*1e3  # us

def rell2(a,b):
    a,b=a.float(),b.float()
    return (torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12)).item()

print(f"{'config':>16} | {'opus(us)':>9} {'triton(us)':>10} {'speedup':>8} | {'w relL2':>9} {'u relL2':>9}")
print("-"*72)
for (B,T,H) in [(1,1024,4),(1,2048,16),(1,4096,8),(1,8192,32),(1,16384,16),(4,2048,8),(1,32768,16)]:
    k,v,g,beta=make(B,T,H)
    K=k.shape[-1]; V=v.shape[-1]
    w=torch.empty(B,T,H,K,dtype=torch.bfloat16,device="cuda")
    u=torch.empty(B,T,H,V,dtype=torch.bfloat16,device="cuda")
    gc=torch.empty(B,T,H,dtype=torch.float32,device="cuda")
    # precision vs triton non-fused
    gT,wT,uT=triton_k14(k,v,g,beta)
    opus_k14(k,v,g,beta,w,u,gc)
    dw,du=rell2(w,wT),rell2(u,uT)
    # cudagraph timing
    to=graph_time(lambda: opus_k14(k,v,g,beta,w,u,gc))
    tt=graph_time(lambda: triton_k14(k,v,g,beta))
    print(f"B{B}T{T}H{H:<2}".rjust(16)+f" | {to:9.1f} {tt:10.1f} {tt/to:7.2f}x | {dw:9.2e} {du:9.2e}",flush=True)
