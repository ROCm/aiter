import torch
from op_tests.op_benchmarks.triton.utils.quantization import quantize_fp8


def torch_moe(
    a,
    b,
    c,
    a_scale,
    b_scale,
    b_zp,
    group_size,
    topk_ids,
    topk_weights,
    routed_weight,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    dtype,
    fp8_w8a8,
    int8_w8a16,
    int4_w4a16,
    gelu=False,
):
    if fp8_w8a8:
        a, _, a_scale = quantize_fp8(a)

    M, top_k, N = c.shape
    _, K = a.shape

    if int4_w4a16:
        b = torch.repeat_interleave(b, repeats=2, dim=2)  # Expand to (E, N, K)
        b_shifter = ((torch.arange(0, K, device=b.device) % 2) * 4)[None, None, :]
        b = (b >> b_shifter) & 0xF
        b_scale = torch.repeat_interleave(
            b_scale, repeats=group_size, dim=2
        )  # (E, N, K)
        if b_zp is not None:
            b_zp = torch.repeat_interleave(
                b_zp, repeats=2, dim=1
            )  # (E,N//2,K//group_size) -> (E, N, K // group_size)
            b_zp = torch.repeat_interleave(
                b_zp, repeats=group_size, dim=2
            )  # (E,N,K//group_size) -> (E, N, K)
            b_zp_shifter = ((torch.arange(0, N, device=b.device) % 2) * 4)[
                None, :, None
            ]
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b = (b - b_zp) * b_scale
        else:
            b = (b - 8) * b_scale

    # Repeat a -> (M, top_k, K)
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
    # (M, top_k, N, K)
    if fp8_w8a8:
        b_indexed = b.half()[topk_ids]
    else:
        topk_ids = topk_ids.to(torch.int32)
        print(b.dtype, topk_ids.dtype)
        b_indexed = b[topk_ids]

    c = torch.einsum("mek,menk->men", a_expanded.to(dtype), b_indexed.to(dtype))

    if routed_weight:
        c *= topk_weights.unsqueeze(-1)

    if not routed_weight and gelu:
        c = 0.5 * c * (1.0 + torch.tanh(0.7978845608 * (c + 0.044715 * c * c * c)))

    if fp8_w8a8:
        c = c * b_scale[topk_ids].unsqueeze(-1)
        c = c * a_scale
        c = c.to(dtype)

    if int8_w8a16:
        c = c * b_scale[topk_ids].unsqueeze(-1)
        c = c.to(dtype)

    return c
