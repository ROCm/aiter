## Practices for writing Triton tests:

### Performance

1. Use `torch.testing.assert_close` for tensor comparisons over `triton.testing.assert_close`, as it's significantly faster.
Triton's implementation uses numpy under the hood. Switch back to Triton if you're experiencing OOM issues.

2. When possible, generate test inputs directly on the GPU (e.g with `torch.randn((M, K), device="cuda")` as opposed to `torch.randn((M, K)).cuda()`).
It's ~2 orders of magnitude faster for large test cases.


### Logging

1. Don't `print(...)`. A test that needs to say something uses the aiter logger
(`from aiter import logger`), which is quiet under pytest — `op_tests/triton_tests/__init__.py`
pins `AITER_LOG_LEVEL=WARNING` there — and still prints when the file is run directly.

2. Pass values to the logger, don't format them in. `logger.info(f"shape={x.shape}")` builds the
string before the level check, so the work happens even when nothing is emitted. Write
`logger.info("shape=%s", x.shape)` instead.

3. Match the placeholder to the value: `%d` for counts and dimensions, `%f` for thresholds and
real scalars, `%s` for tensors, `torch.Size` shapes, tuples, bools and strings. `%d` or `%f` on
`None` or on a tuple raises when the record is emitted, and logging turns that into
`--- Logging error ---` on stderr rather than a failed test — so when a value is sometimes
`None`, use `%s`.

4. Use `logger.debug(...)` for the verbose stuff and turn it on from the environment — there is
no `DEBUG_MODE` constant to flip:

```bash
AITER_LOG_LEVEL=DEBUG pytest op_tests/triton_tests/rope/test_rope.py -s
AITER_LOG_LEVEL=DEBUG python op_tests/triton_tests/rope/test_rope.py
```

`aiter/__init__.py` reads that variable once, on first import, and applies it to the logger *and*
to its console handler. Don't try to lower the level from inside a test with
`logger.setLevel(logging.DEBUG)`: that moves the logger but leaves the handler at
`AITER_LOG_LEVEL`, and a handler drops any record below its own level, so nothing is printed.
Don't call `logging.basicConfig(...)` either — it reconfigures the root logger for the whole
process at import.

5. Anything that explains a failure goes at `logger.warning(...)` or above, and the detail a reader
needs in order to act goes in the assertion message. Under pytest the aiter handler sits at
`WARNING`, so an `INFO` line is invisible in CI precisely when it matters — a mismatch, a NaN, the
coordinates behind a failed comparison. And a test that *logs* its verdict rather than asserting it
cannot fail at all; the log is not a check.
