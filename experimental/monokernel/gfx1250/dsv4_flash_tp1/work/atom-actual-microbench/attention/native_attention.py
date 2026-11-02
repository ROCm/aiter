"""Actual ATOM BF16 Flash C1 SWA attention dispatch, max_num_seqs=128.
The upstream qk_norm_rope kernel has already written the current cache row.
"""
import torch
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
class NativeAttention:
 def __init__(self, cache, sink, length=128, index_capacity=16384):
  self.cache,self.sink=cache,sink
  self.indices=torch.zeros(index_capacity,device=cache.device,dtype=torch.int32)
  self.indices[:length]=torch.arange(length,device=cache.device,dtype=torch.int32)
  self.indptr=torch.tensor([0,length],device=cache.device,dtype=torch.int32)
 def __call__(self,q):
  return sparse_attn_v4_paged_decode(q.view(1,64,512),self.cache,self.indices,self.indptr,self.sink,512**-.5)
