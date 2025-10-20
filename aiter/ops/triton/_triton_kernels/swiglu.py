import triton
import triton.language as tl
# [ \text{SwiGLU}(x, W, V, b, c) = (\text{SiLU}(x*W + b)) \odot (x * V + c) ]

@triton.jit
def _swiglu(
    output_ptr, #[M,N]
    input_ptr, #[M,2K]
    W_ptr, #[K,N]
    V_ptr, #[K,N]
    b_ptr, #[N]
    c_ptr, #[N]
    stride_xm,stride_xk,
    stride_ym,stride_yn,
    stride_Wk,stride_Wn,
    stride_vk,stride_vn,
    M, K, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offm = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offn = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    mask_m = offm < M 
    mask_n = offn < N 

    firstPart = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),tl.float32) #(x*W)
    secondPart = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),tl.float32) #(x*V)

    #add biases (b and c)
    b = tl.load(b_ptr + offm,mask=mask_m, other=0.0)
    c = tl.load(c_ptr + offn,mask=mask_n, other=0.0)
    for k in range(0, K, BLOCK_SIZE_K): #xW
        offk = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offk < K

        xTemp = input_ptr + offm[:, None] * stride_xm + offk[None, :] * stride_xk 
        WTemp = W_ptr + offk[:, None] * stride_Wk + offn[None, :] * stride_Wn
        x = tl.load(xTemp, mask=mask_m[:, None] & mask_k[None, :])
        W = tl.load(WTemp, mask=mask_k[:, None] & mask_n[None, :])
        firstPart += tl.dot(x, W)

        displace = offk[None, :] + K #second part of input
        xTemp = input_ptr + offm[:, None] * stride_xm + displace * stride_xk  
        VTemp = V_ptr + offk[:, None] * stride_vk + offn[None, :] * stride_vn
        x = tl.load(xTemp, mask=mask_m[:, None] & mask_k[None, :])
        V = tl.load(VTemp, mask=mask_k[:, None] & mask_n[None, :])
        secondPart += tl.dot(x, V)

    firstPart = firstPart * tl.sigmoid(firstPart)
    firstPart += b[:, None]#xW + b
    secondPart += c[None, :] #xV + c

    res = firstPart * secondPart
    outTemp = output_ptr + offm[:, None] * stride_ym + offn[None, :] * stride_yn
    tl.store(outTemp, res, mask=mask_m[:,None] & mask_n[None,:])