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

## Device and dtype

These rules also apply to the benchmarks in `op_tests/op_benchmarks/triton/`
and to the multi-GPU tests in `op_tests/multigpu_tests/triton_test/`.

1. Make each tensor directly on the GPU with `device="cuda"`. Do not make a
   tensor on the CPU and then move it with `.cuda()`, `.to("cuda")` or
   `.to(device)`. A multi-GPU test uses the CUDA device of its rank. A
   benchmark with a `--device` option uses that option.
2. Give each floating-point tensor an explicit `dtype`, also random tensors:
   `torch.randn((M, K), dtype=torch.float32, device="cuda")`. Do not use the
   default dtype. Integer constructors such as `arange`, `randint`,
   `randperm` and `torch.tensor` of Python ints infer `int64`. They do not
   need a `dtype`.
3. Do not call `torch.set_default_device`, `torch.set_default_dtype` or
   `torch.set_default_tensor_type`. These calls change the defaults for all
   tests that run after them in the same pytest process. The only exception
   is a context manager that sets a default and restores the previous one on
   exit. Use it only to test that library code does not depend on the default
   (`_with_default_device` in `attention/test_mha_dao_ai.py`,
   `with_default_dtype` in `op_tests/triton_tests/utils/types.py`).
4. Give `torch.load` an explicit `map_location`.
5. Use a CPU tensor only when the test needs one, for example to check that
   the wrapper rejects CPU inputs. Write `device="cpu"`.

## Speed

1. Use `torch.testing.assert_close`, not `triton.testing.assert_close`. It is
   much faster. Use the Triton version only if you get out-of-memory errors.
2. Make inputs directly on the GPU (see [Device and dtype](#device-and-dtype)).
   For large inputs, this is approximately 100 times faster than
   `torch.randn((M, K)).cuda()`.

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
