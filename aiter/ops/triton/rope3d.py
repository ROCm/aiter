import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.rope import _rope_fwd_3d_kernel

def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # complex
    return freqs

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def rope_apply_triton(x, grid_sizes: tl.constexpr, freqs: tl.constexpr, sp_size: tl.constexpr, sp_rank: tl.constexpr):
    B, s, n_heads, C = x.shape
    c_total = C // 2  # 64
    c1 = c_total - 2 * (c_total // 3)  # 22
    c2 = c_total // 3                  # 21
    c3 = c_total // 3                  # 21
    device = x.device

    grid_sizes = grid_sizes.to(device=device, dtype=torch.int32).contiguous()

    freqs_real = freqs.real.to(dtype=torch.float32, device=device).contiguous()
    freqs_imag = freqs.imag.to(dtype=torch.float32, device=device).contiguous()
    out = torch.empty_like(x, dtype=torch.float32, device=device)

    BLOCK_L, BLOCK_N, BLOCK_C = 32, 4, 64

    grid = (
        B,
        n_heads,  
        triton.cdiv(s, BLOCK_L)
    )

    num_warps = 4
    waves_per_eu = 1

    _rope_fwd_3d_kernel[grid](
        x, freqs_real, freqs_imag, grid_sizes, out, 
        *x.stride(),
        freqs_real.stride(0), freqs_real.stride(1),
        *grid_sizes.stride(),
        *out.stride(),
        s, n_heads, C, c_total,
        sp_size, sp_rank,
        freqs.shape[0], s,
        1.0, 0.0,
        BLOCK_L=BLOCK_L, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
        C1=c1, C2=c2,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out

def rope_apply_original(x, grid_sizes, freqs, sp_size, sp_rank):
    B = x.size(0)
    s = x.size(1)
    n = x.size(2)
    c = x.size(3) // 2

    c1 = c - 2 * (c // 3)
    c2 = (c // 3)
    c3 = (c // 3)
    freqs = freqs.split([c1, c2, c3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))

        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        merged_real_sum = freqs_i.real.sum()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]

        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        output.append(x_i)

    out = torch.stack(output).float()
    return out

def test_rope_consistency():
    B, s, n, C = 1, 9450, 40, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp_size = 8
    max_seq_len = 1024

    x = torch.arange(B*s*n*C, dtype=torch.float32, device=device).reshape(B, s, n, C)
    x = x / (B*s*n*C)

    grid_sizes = torch.tensor([[21, 45, 80]], dtype=torch.int32, device=device)

    d_total = 128
    d1 = d_total - 4 * (d_total // 6)
    d2 = 2 * (d_total // 6)
    d3 = 2 * (d_total // 6)

    freqs_f = rope_params(max_seq_len, d1)
    freqs_h = rope_params(max_seq_len, d2)
    freqs_w = rope_params(max_seq_len, d3)
    freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=1).to(device)

    sp_rank = 0
    out_orig = rope_apply_original(x.clone(), grid_sizes.clone(), freqs.clone(), sp_size, sp_rank)


    out_triton = rope_apply_triton(x.clone(), grid_sizes.clone(), freqs.clone(), sp_size, sp_rank)

    print(f"the result compare: sp_rank={sp_rank}")
    print("="*50)
    shape_ok = (out_orig.shape == out_triton.shape)
    sum_orig = out_orig.sum().item()
    sum_triton = out_triton.sum().item()
    sum_diff = abs(sum_orig - sum_triton) / abs(sum_orig)
    sum_ok = sum_diff < 1e-2
    feat_orig = out_orig[0,0,0,:4]
    feat_triton = out_triton[0,0,0,:4]
    feat_diff = torch.abs(feat_orig - feat_triton).max().item()
    feat_ok = feat_diff < 1e-3

    print(f"shape same {'yes' if shape_ok else 'no'}")
    print(f"(sum diff<1%): {'yes' if sum_ok else 'no'}")
    print(f"   - Original sum: {sum_orig:.6f}")
    print(f"   - Triton sum:   {sum_triton:.6f}")
    print(f"   - corellation diff %:     {sum_diff*100:.2f}%")
    print(f"fisrt 4 tensor same {'yes' if feat_ok else 'no'}")
    print(f"   - Original: {feat_orig.cpu().numpy()}")
    print(f"   - Triton:   {feat_triton.cpu().numpy()}")
    print(f"   - max diff: {feat_diff:.6f}")


    if shape_ok and sum_ok and feat_ok:
        print(f"\n sp_rank={sp_rank} test success")
    else:
        print(f"\n sp_rank={sp_rank} test failed")
    print("="*60)


if __name__ == "__main__":
    test_rope_consistency()

