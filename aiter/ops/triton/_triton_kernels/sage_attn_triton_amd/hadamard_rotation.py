import torch
import triton
import triton.language as tl

def create_hadamard_matrix(block_size, device="cuda", dtype=torch.float32):
    """
    Create an orthogonal Hadamard matrix of size block_size x block_size.
    Uses Sylvester's recursive construction and normalizes to be orthogonal.
    
    Args:
        block_size: Size of the matrix (must be a power of 2)
    
    Returns:
        Orthogonal Hadamard matrix of shape (block_size, block_size)
        Satisfies: H @ H.T = I (identity matrix)
    
    Example:
        H_2 = [[1,  1],
               [1, -1]] / sqrt(2)
        
        H_4 = [[1,  1,  1,  1],
               [1, -1,  1, -1],
               [1,  1, -1, -1],
               [1, -1, -1,  1]] / 2
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"
    
    # Base case: H_1 = [1]
    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    
    # Recursive construction: H_{2n} = [H_n   H_n  ]
    #                                   [H_n  -H_n ]
    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)
    
    # Build the full matrix (unnormalized)
    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half
    
    # Normalize to make it orthogonal: H @ H.T = I
    # The unnormalized matrix satisfies H_unnorm @ H_unnorm.T = block_size * I
    # So divide by sqrt(block_size) to get orthogonal matrix
    H = H / (2.0 ** 0.5)  # Divide by sqrt(2) since we doubled the size
    
    return H

def create_random_rotation(block_size, device="cuda", dtype=torch.float32):
    generator = torch.Generator(device=device)
    generator.manual_seed(1000)

    A = torch.randn(block_size, block_size, device=device, dtype=torch.float32, generator=generator)

    Q, R = torch.linalg.qr(A)

    return Q.to(dtype)

@triton.jit
def _hadamard_rotation_kernel(
    input_ptr,
    r_ptr,
    output_ptr,
    stride_m,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Tile size along M dimension
    BLOCK_SIZE_D: tl.constexpr,  # Block size for Hadamard transform (must be power of 2)
):
    """
    Simplified kernel that processes one block at a time.
    """
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Row indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D
    mask_all = mask_m[:, None] & mask_d[None, :]
    
    input_offsets = offs_m[:, None] * stride_m + offs_d[None, :]
    input = tl.load(input_ptr + input_offsets, mask=mask_all, other=0.0)

    offs_r = tl.arange(0, BLOCK_SIZE_D)

    if (pid_d + 1) * BLOCK_SIZE_D > D:
        # using Identity matrix
        r = (offs_r[:, None] == offs_r[None, :]).to(r_ptr.dtype.element_ty)
    else:
        r_offsets = offs_r[:, None] * BLOCK_SIZE_D + offs_r[None, :]
        r = tl.load(r_ptr + r_offsets)

    output = tl.dot(input, r, out_dtype=tl.float32).to(output_ptr.dtype.element_ty)
    
    # Store result (transformed elements + unchanged remainder)
    tl.store(output_ptr + input_offsets, output, mask=mask_all)


def apply_hadamard_rotation(input, rotation, BLK, block_size=32):
    """
    Apply block-diagonal Hadamard rotation to input tensor.
    
    Args:
        input: Input tensor of shape (M, D) where M is sequence length, D is feature dim
        rotation: orthogonal matrix with block_size by block_size
        block_size: Size of each Hadamard block (must be power of 2)
    
    Returns:
        rotated_tensor: Rotated tensor of same shape
    """
    assert input.ndim == 2, "Input must be 2D (M, D)"
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert rotation.shape == (block_size, block_size), f"shape of rotation matrix must be {(block_size, block_size)}"
    
    M, D = input.shape
    output = torch.empty_like(input)
    stride_m = input.stride(0)
    
    # Calculate grid dimensions
    num_blocks_d = triton.cdiv(D, block_size)
    
    grid = (triton.cdiv(M, BLK), num_blocks_d)
    
    _hadamard_rotation_kernel[grid](
        input,
        rotation,
        output,
        stride_m,
        M=M,
        D=D,
        BLOCK_SIZE_M=BLK,
        BLOCK_SIZE_D=block_size,
    )
    
    return output


def apply_hadamard_rotation_qk(q, k, BLKM=128, BLKN=64, block_size=32):
    """
    Apply the same block-diagonal Hadamard rotation to both Q and K.
    
    Args:
        q: Query tensor of shape (..., D)
        k: Key tensor of shape (..., D)
        block_size: Size of each Hadamard block (must be power of 2)
    
    Returns:
        q_rotated, k_rotated: Rotated Q and K tensors
        
    Note: The rotation preserves dot products: q' @ k'.T = q @ k.T
    """
    assert q.shape[-1] == k.shape[-1], "q and k must have same feature dimension"
    q_shape = q.shape
    k_shape = k.shape
    q = q.view(-1, q_shape[-1])
    k = k.view(-1, k_shape[-1])

    r = create_hadamard_matrix(block_size, q.device, q.dtype)
    #r = create_random_rotation(block_size, q.device, q.dtype)
    
    q_rotated = apply_hadamard_rotation(q, r, BLKM, block_size=block_size)
    k_rotated = apply_hadamard_rotation(k, r, BLKN, block_size=block_size)
    q_rotated = q_rotated.reshape(q_shape)
    k_rotated = k_rotated.reshape(k_shape)
    
    return q_rotated, k_rotated

if __name__ == "__main__":

    def compare_accuracy(current, reference):
        """Print quick statistics comparing FP8 and SageAttn tensors."""
        current_f = current.float()
        reference_f = reference.float()
        abs_diff = torch.abs(reference_f - current_f)

        print("Output Tensor Stats:")
        print(
            f"  Reference ({tuple(reference_f.shape)}): min={reference_f.min().item():.6f}, max={reference_f.max().item():.6f}, "
            f"mean={reference_f.mean().item():.6f}, std={reference_f.std().item():.6f}"
        )
        print(
            f"  Test      ({tuple(current_f.shape)}): min={current_f.min().item():.6f}, max={current_f.max().item():.6f}, "
            f"mean={current_f.mean().item():.6f}, std={current_f.std().item():.6f}"
        )
        
        print("Correctness Comparison:")
        print(f"  Mean Absolute Error: {abs_diff.mean().item():.6e}")
        print(f"  Max Absolute Error: {abs_diff.max().item():.6e}")
        print(f"  Std Absolute Error: {abs_diff.std().item():.6e}")
        ref_flat = reference_f.reshape(-1)
        test_flat = current_f.reshape(-1)
        cos_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), test_flat.unsqueeze(0))
        print(f"  Cosine Similarity: {cos_sim.item():.8f}")

    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(4, 5, 512, 128, device=device, dtype=dtype)
    k = torch.randn(4, 5, 1024, 128, device=device, dtype=dtype)

    q_rotated, k_rotated = apply_hadamard_rotation_qk(q, k)
    
    ref = torch.einsum("bhsd,bhtd->bhst", q.to(torch.float32), k.to(torch.float32))
    out = torch.einsum("bhsd,bhtd->bhst", q_rotated.to(torch.float32), k_rotated.to(torch.float32))

    compare_accuracy(out, ref)

    #r = create_hadamard_matrix(32, device=device, dtype=dtype)
    #rr = r @ r.T  # Matrix multiplication to check orthogonality: should equal I
    #I = 
    #compare_accuracy(rr, I)
