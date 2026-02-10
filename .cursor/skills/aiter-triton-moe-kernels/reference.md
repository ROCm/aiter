# Triton MoE Kernel Reference

## Token Sorting and Alignment

MoE sorting prepares tokens for grouped GEMM:

```python
from aiter import moe_sorting_fwd

# Input: topk_ids (M, top_k) - which experts each token routes to
# Output:
#   sorted_token_ids - tokens reordered by expert
#   expert_ids - expert ID per block
#   num_tokens_post_padded - total tokens after padding to block_size
sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
    topk_ids, block_size=128, num_experts=64
)
```

The `moe_align_block_size` Triton kernel:
1. Builds per-expert histogram
2. Pads each expert's token count to `block_size` multiple
3. Outputs sorted token IDs (invalid slots get `num_valid_tokens` sentinel)

## Fused SiLU MoE Pattern

Combines GEMM + SiLU activation in one kernel:

```python
@triton.jit
def _moe_silu_fused_kernel(...):
    # Compute gate = A @ W_gate and up = A @ W_up
    # out = silu(gate) * up
    # Avoids materializing intermediate gate/up tensors
```

## Quantized MoE (A8W8)

```python
# Weights: w1 (E, 2*inter_dim, dim) as INT8, w2 (E, dim, inter_dim) as INT8
# Scales: w1_scale (E, 2*inter_dim, 1), w2_scale (E, dim, 1) 
# Activation quantization: a1_scale per-token, a2_scale per-token
```

## MoE Config CSV Format

`aiter/configs/tuned_fmoe.csv`:

```csv
cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,impl,splitK,pad_m,pad_n
256,1,7168,2048,256,8,SiLu,bf16,ck_2stages,0,128,128
256,32,7168,2048,256,8,SiLu,bf16,triton_2stages,0,64,128
```

Fields:
- `cu_num`: Compute unit count (device-specific)
- `token`: Batch size (M)
- `model_dim`, `inter_dim`: Model dimensions
- `expert`, `topk`: MoE configuration
- `act_type`: Activation (SiLu, GeLu, etc.)
- `impl`: Backend (ck_2stages, triton_2stages, asm)
- `splitK`, `pad_m`, `pad_n`: Tuning parameters

## Routing Patterns

### Softmax Top-K (standard)
```python
scores = router(hidden_states)  # (M, E)
topk_weights, topk_ids = torch.topk(torch.softmax(scores, dim=-1), top_k, dim=-1)
```

### Sigmoid Top-1 Fused
```python
# Fused kernel: sigmoid + top-1 selection + normalization
# Located in moe_routing_sigmoid_top1_fused.py
```

## Persistent MoE Kernels

For small M (decode-like), persistent kernels keep all CUs occupied:

```python
@triton.jit
def _moe_persistent_kernel(..., NUM_SMS: tl.constexpr):
    pid = tl.program_id(0)
    # Each SM iterates over multiple tiles
    num_tiles = tl.cdiv(EM, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    for tile_id in range(pid, num_tiles, NUM_SMS):
        # Process tile_id
        ...
```

Grid: `(NUM_SMS,)` where `NUM_SMS = get_num_sms()`

## Benchmark Model Configs

Common MoE models in `model_configs.json`:

| Model | Experts | Top-K | Hidden | Intermediate |
|-------|---------|-------|--------|-------------|
| Mixtral 8x7B | 8 | 2 | 4096 | 14336 |
| DeepSeek V3 | 256 | 8 | 7168 | 2048 |
| Qwen3 235B | 128 | 8 | 4096 | 2048 |

## Test Input Helpers

```python
def input_helper_e2e(M, model_dim, inter_dim, E, top_k, dtype, quant_type):
    """End-to-end MoE test helper with quantization."""
    hidden = torch.randn(M, model_dim, dtype=dtype, device="cuda")
    w1 = torch.randn(E, 2 * inter_dim, model_dim, dtype=dtype, device="cuda")
    w2 = torch.randn(E, model_dim, inter_dim, dtype=dtype, device="cuda")
    scores = torch.randn(M, E, device="cuda")
    topk_weights, topk_ids = torch.topk(torch.softmax(scores, -1), top_k)
    if quant_type != QuantType.No:
        w1, w1_scale = quantize_fp8(w1)
        w2, w2_scale = quantize_fp8(w2)
    return hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale

def quantize_fp8(x):
    """Quantize to FP8 with per-tensor scale."""
    scale = x.abs().amax() / torch.finfo(torch.float8_e4m3fn).max
    return (x / scale).to(torch.float8_e4m3fn), scale
```
