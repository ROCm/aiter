## Practices for writing Triton tests:

### Performance

1. Use `torch.testing.assert_close` for tensor comparisons over `triton.testing.assert_close`, as it's significantly faster.
Triton's implementation uses numpy under the hood. Switch back to Triton if you're experiencing OOM issues.

2. When possible, generate test inputs directly on the GPU (e.g with `torch.randn((M, K), device="cuda")` as opposed to `torch.randn((M, K)).cuda()`).
It's ~2 orders of magnitude faster for large test cases. 