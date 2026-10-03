# Opus MoE backward

Native BF16 MoE backward for AMD Instinct MI355X (gfx950). It consumes forward
sorting metadata and saved SwiGLU preactivations, with no backward TopK or host
readback. Expert/input gradients are BF16, GEMM accumulation and route reduction
are FP32, and router-score gradients are FP32.

## Supported functionality

| Entry | Contract |
|---|---|
| `opus_moe_backward` | Fixed K in {1,2,4,8}, concat gate/up SwiGLU, optional expert biases |
| `opus_moe_varlen_backward` | Compact token-major route IDs and token offsets, including tokens with no routes |
| `opus_moe_attach_backward` | Attach two native backward Functions to forward-owned saved tensors; prunes unrequested gradients |
| `opus_moe_selected_softmax` / varlen equivalent | Softmax over already-selected logits, native Jacobian/scatter backward |
| Single-family wrappers | Checked K1, K2/K3, K4, K5, bias and router launches for validation and tuning |

All expert-path tensors must be contiguous BF16 on one gfx950 device, with
`W1=[E,2I,D]`, `W2=[E,D,I]`, `D%128==0`, `I%128==0` and sorting
`block_m=32`. Higher-order autograd, unaligned dimension tails, other GPU
architectures and distributed expert parallelism are outside this implementation.
Router gradients implement selected-logit softmax, not full-softmax-before-TopK.

## Fixed pipeline and saved state

| Stage | Operation | Output |
|---|---|---|
| K1 down backward | dO @ W2; SwiGLU Jacobian; route-score gradient | dZ, a_scaled, dScores |
| K2 route input backward | dZ @ W1 per expert | route dX |
| K3 route reduction | Sum top-k route gradients per token | dX |
| K4 up-weight backward | dZ transpose @ X per expert | dW1 |
| K5 down-weight backward | dO transpose @ a_scaled per expert | dW2 |

Fixed sorted IDs pack a 24-bit token and an 8-bit top-k slot. Compact IDs refer
to token-major logical routes; a compile-time layout trait separates the ABIs.
`num_valid_ids[0]` is the active padded prefix, and expert offsets delimit its
32-row intervals. Empty experts produce exact-zero weight gradients. The caller
must supply valid metadata from the same forward; tensor validation does not
inspect every device-side route value.

The large working-set policy keeps dZ consumers adjacent. For D in [1536,2048]
and at least 512 MiB dZ, the order is K1/K4/K2/K3/K5; with both forward caches and
at least 1 GiB dZ, K5 runs first. Smaller cases use K1/K2/K3/K4/K5.

Forward can save `a_scaled=score*SwiGLU(Z)` in padded sorted order and sorted X
instead of repeating scale/store and route gathers during backward. Padding rows
must be zero, and the cache must use the exact same metadata. Creating these
caches inside backward is not part of the optimized performance contract.

`opus_moe_gather_x_blocked_g2` produces the private blocked-G2 sorted-X cache.
`saved_x_sorted_blocked_g2=True` selects a coupled K1/K2/K4 layout and requires
saved a_scaled, D divisible by 512 and I divisible by 256. The default uses K1
kid 19 and K2 kid 21 for buffers below 4 GiB, their flat-address counterparts
above that bound, and K4 kid 25. K3 selects a legal full-row or general reducer.
Public autograd attachment supports row-major caches; the coupled blocked path
is currently exposed by the full-chain API.

## Code ownership

- `aiter/ops/opus/moe_backward_types.py`: metadata and output containers.
- `aiter/ops/opus/moe_backward.py`: checked Python wrappers, JIT bindings,
  allocation and autograd attachment; existing import paths remain available.
- `opus_moe_backward_common.py`: registry with stable explicit IDs, auto
  targets and fallbacks. Shared defaults keep instance geometry visible.
- `gen_instances.py`: validated deterministic family manifest.
- `include/opus_moe_backward_launch.cuh`: host checks, auto policy, checked
  family launches and full-chain scheduling. No extra arch forwarding layers.
- `include/opus_moe_backward_host_impl.cuh`: tensor validation and kargs setup;
  included exactly once by `opus_moe_backward.cu`.
- `include/gfx950/opus_moe_backward_dispatch_gfx950.cuh`: generated exact-ID
  tables and concrete gfx950 launchers, not host shape-selection policy.
- `include/gfx950/bf16/*traits*`: compile-time layout/tile policy.
- `include/gfx950/bf16/*pipeline*`: device helpers, process_tile and launchers.
- `include/gfx950/bf16/opus_moe_reduction_pipeline_gfx950.cuh`: non-GEMM
  reductions: bias, route-to-token dX and selected-softmax Jacobian/scatter.
- `csrc/pybind/opus_moe_backward_pybind.cu`: allocation-free tensor bindings.
- `aiter/ops/triton/moe/moe_backward.py`: independent Triton comparison path.

The backward implementation has 16 source files (formerly 22): two Python
modules, registry/generator/TU, one pybind TU, four top-level headers, one
gfx950 dispatch header and five BF16 device headers. See the
[Opus architecture map](../../../aiter/ops/opus/README.md) for repository-wide
ownership and the Python interfaces. Header consolidation changes include
boundaries only; kernel bodies, IDs and schedule policy remain unchanged.

The registry retains production auto targets and legal geometry/layout fallbacks.
Retired comparison IDs are not compiled and fail dispatch. Shared base traits
remain where production traits inherit their layout definitions; they do not
create additional compiled kernels. The original implementations are recoverable
from the pre-pruning Git backup, with context in TUNING_HISTORY.md.

The registry has 54 instances (formerly 71). Retired IDs by family are K1
`5/6/8/14/15/17`, K3 `2`, K4 `10/12/17/19/20/21/22/23/24`, and K5 `17`.
K2 `16` remains active for partial autograd; full-chain auto uses K2 `17`.
K1 `12` remains a no-cache auto target but is no longer accepted with
`saved_a_scaled`; use cache auto or `13/16` instead. Removed IDs are not
renumbered or silently mapped to different kernels.

### Configuration and low-level maintenance

Treat the registry as an entry-point list, not a complete device call graph.
K5 kid `10` also instantiates `SwizzledRouteLe30720` and
`SwizzledRouteGt30720` internally; its split uses padded expert route counts.
These helper traits and the long-route wave4x2 geometry are live even without
standalone IDs. Python partial-autograd selection, compact routing, uncached
paths and flat-address fallbacks must be audited alongside the host selector.

Traits express supported configurations rather than an optimization history.
Literal-only intermediate layers may be folded into their consumers while
keeping registered names/IDs stable. Do not mechanically move derived constexpr
expressions: overriding a base parameter does not re-evaluate expressions
already bound in that base. Compare every effective constant/type and generated
kernel before accepting such a change. Geometry/layout bases remain shared where
they carry expressions or invariants; do not grow an arbitrary policy matrix.

Packed BF16 conversion must preserve operand order, rounding, register constraints
and emitted instructions. K1's lane-swap asm is a separate compiler workaround
for two-output register coalescing with MFMA accumulators; do not replace it merely
because another pipeline uses a builtin. Synchronization, cache control and
addressing fallbacks are pipeline semantics, not dead experiment scaffolding.

## Validation and benchmark

Run in a ROCm PyTorch environment on an authorized idle gfx950 GPU:

```bash
PYTHONPATH=. python -m pytest -q op_tests/test_opus_moe_backward.py
PYTHONPATH=. python op_tests/op_benchmarks/bench_opus_moe_backward.py
```

The suite checks fixed K variants, skew/empty experts, unused sorted capacity,
intermediate dZ, cached/full-chain gradients, bias, compact routes, selected
softmax, autograd attachment and 100 graph replays against FP32 PyTorch equations
with explicit BF16 intermediate rounding. The benchmark also compares the branch
Triton path on small, 16K and 32K shapes.

Graph timing excludes Python/allocator work; direct Opus timing includes the
allocating wrapper while Triton uses preallocated outputs. Sorting and forward
cache production are excluded. Synthetic saved Z is not an end-to-end forward
training quality check. GEMM throughput uses 12*T*K*D*I FLOPs.

For cleanup acceptance use separate source worktrees/JIT directories, fresh-build
ISA/resource comparison and parent/candidate/parent measurements on the same GPU.
Historical tuning notes and measurements are in [TUNING_HISTORY.md](TUNING_HISTORY.md).
