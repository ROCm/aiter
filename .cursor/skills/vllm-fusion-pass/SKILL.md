---
name: vllm-fusion-pass
description: Implement kernel fusion passes in vLLM's torch.compile pipeline using PatternMatcherPass. Use when writing, debugging, or refactoring fusion passes, custom op registration, or investigating unexpected Triton kernels in vLLM traces.
---

# vLLM Fusion Pass Development

## Compilation Pipeline

vLLM uses `torch.compile` with a post-grad pass manager (`vllm/compilation/pass_manager.py`). Pass ordering:

```
NoOpEliminationPass          (removes redundant reshapes/slices)
  -> Fusion passes            (PatternMatcherPass-based, registered here)
    -> PostCleanupPass
      -> FixFunctionalizationPass  (defunctionalizes auto_functionalized nodes)
```

All fusion passes operate on a **functionalized** FX graph.

## Key Files

| File | Role |
|------|------|
| `vllm/compilation/pass_manager.py` | Registers and orders passes |
| `vllm/compilation/fusion.py` | Generic RMSNorm+Quant fusions (CUDA, uses `auto_functionalized`) |
| `vllm/compilation/rocm_aiter_fusion.py` | ROCm AITER fusions |
| `vllm/compilation/matcher_utils.py` | Reusable matchers (`MatcherRMSNorm`, `MatcherQuantFP8`, etc.) |
| `vllm/compilation/vllm_inductor_pass.py` | `VllmInductorPass`, `VllmPatternMatcherPass` base classes |
| `vllm/compilation/inductor_pass.py` | `enable_fake_mode` decorator |
| `vllm/compilation/noop_elimination.py` | Removes reshapes/slices that break pattern matching |
| `vllm/compilation/fix_functionalization.py` | Converts `auto_functionalized` back to direct calls |

## How to Implement a Fusion

### 1. Register the fused custom op

Register in the appropriate ops file (e.g., `_aiter_ops.py`, `_C/__init__.pyi`):

```python
direct_register_custom_op(
    op_name="my_fused_op",
    op_func=_my_fused_op_impl,    # actual GPU kernel call
    mutates_args=[],               # list arg names modified in-place
    fake_impl=_my_fused_op_fake,   # FakeTensor impl for shape/stride inference
    dispatch_key="hip",            # or current_platform.dispatch_key
)
```

### 2. Write a pattern class

Study existing patterns in `rocm_aiter_fusion.py` and `fusion.py` for reference.

```python
class MyFusionPattern:
    FUSED_OP = torch.ops.vllm.my_fused_op.default
    OP_A = torch.ops.vllm.op_a.default
    OP_B = torch.ops.vllm.op_b.default

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            torch.empty(5, 16, dtype=torch.bfloat16, device="cuda"),
            torch.empty(16, dtype=torch.bfloat16, device="cuda"),
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        epsilon = self.epsilon
        OP_A, OP_B, FUSED_OP = self.OP_A, self.OP_B, self.FUSED_OP

        def pattern(input, weight):
            x = OP_A(input, weight, epsilon)
            return OP_B(x=x)

        def replacement(input, weight):
            return FUSED_OP(x=input, weight=weight, epsilon=epsilon)

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )
```

### 3. Write the pass class

```python
class MyFusionPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns = PatternMatcherPass(pass_name="my_fusion_pass")
        # Register for each relevant epsilon
        MyFusionPattern(epsilon=1e-6).register(self.patterns)
        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)

    def uuid(self) -> str:
        return self.hash_source(self, MyFusionPattern)
```

### 4. Register in pass_manager.py

Add under the appropriate config gate (e.g., `fuse_norm_quant`, `fuse_act_quant`).

## Critical Rules

### Always use PatternMatcherPass, never manual graph walks

`pm.register_replacement` handles pattern matching, node insertion, user rewiring, FakeTensor metadata propagation, and dead code elimination. Writing manual `VllmInductorPass` graph walks reimplements all of this. This leads to hundreds of lines of redundant code that duplicates what `NoOpEliminationPass` and `FixFunctionalizationPass` already handle.

### Never implement fusion by modifying model executors

Do not add if-else branches in model code (e.g., `mla.py`, `deepseek_v2.py`) to call fused kernels. This causes `torch.compile` to generate wrapper Triton kernels around your fused op and does not compose with other passes.

### Use high-level PyTorch ops in pattern/replacement functions

```python
# GOOD
q_c, kv = qkv.split([q_rank, kv_rank], dim=-1)
return result[0], result[1]

# BAD
split = torch.ops.aten.split_with_sizes.default(qkv, [...], -1)
q_c = operator.getitem(split, 0)
```

The FX tracer decomposes high-level ops to ATen internally; the pattern matcher handles the lowering.

### @enable_fake_mode is mandatory on __init__

Pattern tracing happens during `__init__`. Without `@enable_fake_mode`, the tracing will fail because it tries to allocate real CUDA tensors for `get_inputs()`.

### Scalar constants are matched literally

Epsilon, booleans, dtypes, split sizes are embedded as literals in the traced pattern graph. The matcher only matches subgraphs with identical constant values. Register multiple patterns when multiple values are possible:

```python
for eps in [1e-5, 1e-6]:
    MyPattern(eps).register(self.patterns)
```

For model-specific dimensions, read them from `config.model_config.hf_config` with `getattr(..., None)` guards.

## get_inputs() Rules

PyTorch only traces input types/dtypes using FakeTensors for pattern matching, not actual shapes. Use simple dummy shapes.

**Exception**: if the pattern contains shape-dependent ops like `split()`, dummy input dimensions must be compatible with the split sizes during tracing. For example, if you split dim=-1 into `[1536, 576]`, the last dim must be `2112`. Use the sum of split sizes for the relevant dimension; other dimensions can be arbitrary.

## Fake (Meta) Kernel Gotchas

The fake implementation infers output shapes, dtypes, **and strides** at compile time. Mistakes cause hard-to-debug failures.

### Stride mismatch -> extra copy kernels

If the fake returns strides different from the real impl, Inductor inserts a `triton_poi_fused_*` copy kernel to reconcile them at runtime. This is the most common source of unexpected Triton kernels.

**Fix**: make the fake return tensors with the exact same strides as the real impl. For transposed outputs:

```python
def _my_op_fake(x):
    M, N = x.shape
    N_s = (N + 31) // 32
    return torch.empty((N_s, M), dtype=torch.uint8, device=x.device).T
```

### Stride consistency across fused variants

If standalone `op_a` and fused `op_a_b` both produce outputs consumed by the same downstream ops, their fakes must return identical strides. Otherwise, the compiler inserts stride-conversion copies when the pattern matcher swaps one for the other.

### Non-contiguous outputs

Some ops produce non-contiguous views. The fake must replicate this:

```python
def _my_op_fake(x, full_dim, slice_dim):
    M = x.shape[0]
    buf = torch.empty((M, full_dim), dtype=torch.bfloat16, device=x.device)
    return buf[..., :slice_dim]  # non-contiguous view
```

### Stride assertion errors

`AssertionError: expected size X, stride Y at dim Z` at runtime means the fake's strides don't match the real kernel's strides. Compare both and fix the fake.

## auto_functionalized vs Pure Ops

Ops with `mutates_args=["residual"]` appear in the graph wrapped in `auto_functionalized`:

```
at = auto_functionalized(fused_add_rms_norm, input=x, residual=r)
result = getitem(at, 1)
residual = getitem(at, 2)
```

Fusion replacements for in-place ops must use `auto_functionalized` in the replacement (see `fusion.py`). If your fused op also has `mutates_args`, add a case to `fix_functionalization.py` to defunctionalize it.

Pure ops (`mutates_args=[]`) appear as direct calls and are simpler to match.

## Pass Composition

Passes compose: a simpler pass creates fused ops, a later pass matches those fused ops in a larger pattern. Register passes in correct order in `pass_manager.py`.

`NoOpEliminationPass` must run first -- it removes redundant reshapes/slices that `apply_fp8_linear` inserts. Without it, patterns won't match due to extra nodes.

## Debugging

### Pattern doesn't match (0 matches logged)

- Extra reshape/slice nodes between ops -> ensure `NoOpEliminationPass` runs first
- Wrong op target -> graph may use `auto_functionalized(op)` instead of `op` directly
- Constant mismatch -> epsilon or split sizes differ from model's values
- Keyword vs positional args -> check op schema calling convention
- Inspect the FX graph: set `VLLM_TORCH_COMPILE_DEBUG_DUMP_PATH=/tmp/vllm_debug`

### Unexpected triton_poi_fused_* kernels in traces

These indicate stride mismatches. To debug:
1. Find the kernel source under `/tmp/torchinductor_root/`
2. Read the `.py` wrapper to see which op's output it copies
3. Compare fake vs real strides and fix the fake

### Enable pattern match debug

Set `VLLM_PATTERN_MATCH_DEBUG=1` to see which patterns matched.

## Testing and Trace Verification

### Run the test

```bash
# Terminal 1
cd /app/scripts && bash run_server.sh

# Terminal 2 (after "Application startup complete.")
cd /app/scripts && bash run_client.sh
```

### Analyze traces

Write a script to parse `*.pt.trace.json.gz` files. The script should:
- `zcat` and `json.load` the trace file
- Count kernel names from events where `event["cat"] == "kernel"`
- Print sorted kernel counts
- Flag any `triton_poi_fused_*` kernels not in a known-benign allowlist (allreduce fusion, RoPE, pre-existing non-MLA scale transposes)
- Check that expected fused kernel names appear with nonzero counts

### What to verify

1. Fused kernel name appears with expected count (layers x decode_steps)
2. No unexpected `triton_poi_fused_*` copy kernels
3. No regressions in TPOT vs baseline

### Reinstall loop

```bash
cd /path/to/vllm-source && python3 setup.py develop
```

Then restart the server.
