import triton
import triton.language as tl
# [ \text{SwiGLU}(x, W, V, b, c) = (\text{SiLU}(x*W + b)) \odot (x * V + c) ]

def _swiglu(
    output_ptr, #[M,N]
    input_ptr, #[M,K]
    W_ptr,V_ptr,b_ptr,c_ptr, #trainable parameters, shape [K,N]
    M, K, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offRow = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offCol = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    mask_m = offRow < M 
    mask_n = offCol < N 
    firstPart = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),tl.float32) #(x*W)
    secondPart = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),tl.float32) #(x*V)

    #add biases (b and c)
    b = tl.load(b_ptr + offRow,mask=mask_m)
    c = tl.load(c_ptr + offCol,mask=mask_n)
    for k in range(0,K): #xW
        xTemp = input_ptr + offRow + k 
        WTemp = W_ptr + offCol + k 
        x = tl.load(xTemp, mask=mask_m)
        W = tl.load(WTemp, mask=mask_n)
    
        firstPart += x * W
    firstPart = firstPart * tl.sigmoid(firstPart)
    firstPart += b[None, :] #xW + b
    for k in range(0,K): #xV
        xTemp = input_ptr + offRow + k 
        vTemp = V_ptr + offCol + k 
        x = tl.load(xTemp, mask=mask_m)
        V = tl.load(vTemp, mask=mask_n)
        
        secondPart += x * V 
    secondPart += c[None, :] #xV + c

    res = firstPart * secondPart
    out_ptrs = output_ptr + offRow[:, None] + offCol[None, :]
    tl.store(out_ptrs, res)