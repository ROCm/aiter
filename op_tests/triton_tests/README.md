# Triton tests

## Rules

1. Put each test in the folder that matches the wrapper folder in
   `aiter/ops/triton/`. Example: the test of `gemm/basic/gemm_a8w8.py` is
   `gemm/basic/test_gemm_a8w8.py`.
2. Call the public wrapper. Do not launch kernels from `_triton_kernels/` or
   `_gluon_kernels/`.
3. Check the results with asserts. Do not print results.
4. Do not put configs in a test and do not autotune in a test. The wrapper
   loads its config.

## Speed

1. Use `torch.testing.assert_close`, not `triton.testing.assert_close`. It is
   much faster. Use the Triton version only if you get out-of-memory errors.
2. Make inputs on the GPU: `torch.randn((M, K), device="cuda")`, not
   `torch.randn((M, K)).cuda()`. For large inputs, this is approximately 100
   times faster.

## Logging

1. Use the aiter logger (`from aiter import logger`), not `print`. Under
   pytest, `__init__.py` sets `AITER_LOG_LEVEL=WARNING`.
2. Use `%` placeholders, not f-strings: `logger.info("shape=%s", x.shape)`.
3. Use `%d` and `%f` only for numbers that are never `None`. Use `%s` for
   tensors, shapes, tuples, booleans, strings and values that can be `None`.
4. Use `logger.debug(...)` for detail. Turn it on from the environment:

   ```bash
   AITER_LOG_LEVEL=DEBUG pytest op_tests/triton_tests/rope/test_rope.py -s
   ```

   Library code logs through `AiterTritonLogger`. To see its detail, set
   `AITER_TRITON_LOG_LEVEL=DEBUG` and add `--log-cli-level=DEBUG` to pytest.

   Do not call `logger.setLevel(...)` or `logging.basicConfig(...)`.
5. Log failure detail at WARNING or higher, or put it in the assert message.
   CI does not show INFO. A test that only logs a result and does not assert
   it cannot fail.
