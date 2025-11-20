#!/usr/bin/env python3
"""
Simple script to reproduce attention precision issue using dumped tensors.
Can be run directly without pytest.
"""

import os
import torch


def main():
    # dump_path = "/root/test/attention_debug_dump/inputs_rank5_1763115285598.pt"

    # dump_path = "/root/test/attention_debug_dump/inputs_rank3_1763021869575.pt"
    # data = torch.load(dump_path, map_location='cpu')
    
    # print(f"   Keys: {list(data.keys())}")
    
    # print(f"   TP Rank: {data['tp_rank']}/{data['tp_world_size']}")
    # print(f"   Return LSE: {data['return_softmax_lse']}")
    # print(f"   Other kwargs: {data['other_kwargs']}")
    
    # print(f"   q: shape={data['q'].shape}, dtype={data['q'].dtype}, stride={data['q'].stride()}")
    # print(f"   k: shape={data['k'].shape}, dtype={data['k'].dtype}, stride={data['k'].stride()}")
    # print(f"   v: shape={data['v'].shape}, dtype={data['v'].dtype}, stride={data['v'].stride()}")
    
    # print(f"   softmax_scale: {data['softmax_scale']}")
    # print(f"   cu_seqlens_q: shape={data['cu_seqlens_q'].shape}, values={data['cu_seqlens_q'].tolist()}")
    # print(f"   cu_seqlens_k: shape={data['cu_seqlens_k'].shape}, values={data['cu_seqlens_k'].tolist()}")
    # print(f"   max_seqlen_q: {data['max_seqlen_q']}")
    # print(f"   max_seqlen_k: {data['max_seqlen_k']}")
    
    # batch_size = data['cu_seqlens_q'].shape[0] - 1
    # print(f"   batch_size: {batch_size}")
    
    # for name in ['q', 'k', 'v']:
    #     tensor = data[name]
    #     print(f"   {name}:")
    #     print(f"     has_nan={torch.isnan(tensor).any().item()}, has_inf={torch.isinf(tensor).any().item()}")
        
    # device = torch.device('cuda')
    
    # # Move tensors to GPU
    # q = data['q'].to(device)
    # k = data['k'].to(device)
    # v = data['v'].to(device)
    # cu_seqlens_q = data['cu_seqlens_q'].to(device)
    # cu_seqlens_k = data['cu_seqlens_k'].to(device)
    
    # import sys
    # sys.path.insert(0, '/root/test/rocm-vllm')
    
    from aiter import flash_attn_varlen_func
    from aiter.ops.triton.mha import flash_attn_varlen_func as flash_attn_triton
    call_num = "call000095"

    q = torch.load(f'./dump_data/{call_num}_q.pt', map_location='cuda')
    k = torch.load(f'./dump_data/{call_num}_k.pt', map_location='cuda')
    v = torch.load(f'./dump_data/{call_num}_v.pt', map_location='cuda')
    cu_seqlens_q = torch.load(f'./dump_data/{call_num}_cu_seqlens_q.pt', map_location='cuda')
    cu_seqlens_k = torch.load(f'./dump_data/{call_num}_cu_seqlens_k.pt', map_location='cuda')
    print(f"q shape is {q.shape}, stride is {q.stride()}")
    print(f"k shape is {k.shape}, stride is {k.stride()}")
    print(f"v shape is {v.shape}, stride is {v.stride()}")
    print(f"cu_seqlens_q is {cu_seqlens_q}")
    print(f"cu_seqlens_k is {cu_seqlens_k}")
    # print(f"k is {k}")
    kwargs = {
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_k,
        'max_seqlen_q': 2048,
        'max_seqlen_k': 2048,
        'causal': True,
        # **data['other_kwargs']
    }
    out = torch.randn(
        q.shape[0], q.shape[1], v.shape[2], device="cuda", dtype=q.dtype, requires_grad=True
    )
    output_flash = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=0.1352337788608801,
        return_lse=False,
        **kwargs,
        min_seqlen_q=1,
        # window_size=(-1,0),
        out=out
    )

    attn_out_flash = output_flash[0] if isinstance(output_flash, tuple) else output_flash
    attn_softmax = output_flash[1] if isinstance(output_flash, tuple) else output_flash
    # attn_out_flash = attn_out_flash[:283, :, :]

    print(attn_softmax)
    print(f"attn_softmax shape is {attn_softmax.shape}, stride is {attn_softmax.stride()}")

    output_triton = flash_attn_triton(
        q=q,
        k=k,
        v=v,
        softmax_scale=0.1352337788608801,
        return_lse=False,
        **kwargs,
    )
    
    attn_out_ref = output_triton[0] if isinstance(output_triton, tuple) else output_triton      
    attn_softmax_ref = output_triton[1] if isinstance(output_triton, tuple) else output_triton     
    print(attn_softmax_ref) 
    print(f"attn_softmax_ref shape is {attn_softmax_ref.shape}, stride is {attn_softmax_ref.stride()}")
    # attn_out_ref = attn_out_ref[:283, :, :]
    for name in ['attn_out_flash', 'attn_out_ref']:
        tensor = locals()[name]
        print(f"   {name}:")
        print(f"     has_nan={torch.isnan(tensor).any().item()}, has_inf={torch.isinf(tensor).any().item()}")  
    
    v_dim = v.shape[-1]
    attn_out_flash_valid = attn_out_flash[..., :v_dim]

    max_abs_diff = torch.max(torch.abs(attn_out_ref.float() - attn_out_flash_valid.float())).item()
    mean_abs_diff = torch.mean(torch.abs(attn_out_ref.float() - attn_out_flash_valid.float())).item()
    max_ref = torch.max(torch.abs(attn_out_ref.float())).item()
    rel_diff = max_abs_diff / (max_ref + 1e-8)

    # out_diff = (out - out_ref).abs().max().item()
    # ref_diff = (out_pt - out_ref).abs().max().item()
    # print(f"Output max diff: {out_diff}")
    # print(f"Output Pytorch max diff: {ref_diff}")
    # out_tol = max(4 * ref_diff, 0.01)
    # assert out_diff <= out_tol, f"forward diff {out_diff} exceeds tolerance {out_tol}"
    
    ATOL = 0.1
    RTOL = 0.1
    is_close = torch.allclose(attn_out_ref.float(), attn_out_flash_valid.float(), atol=ATOL, rtol=RTOL)
    
    print(f"   torch.allclose(atol={ATOL}, rtol={RTOL}): {is_close}")
    print(f"   Max absolute difference: {max_abs_diff:.6e}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"   Max reference value: {max_ref:.6e}")
    print(f"   Relative difference: {rel_diff:.6e}")
    
    if is_close:
        print("\nPASS!!!!")
        return 0
    else:
        print("\nFAIL!!!!")
        print(f"   Expected tolerance: atol={ATOL}, rtol={RTOL}")
        print(f"   Actual difference exceeds tolerance")
        return 1
            

if __name__ == "__main__":
    exit(main())

