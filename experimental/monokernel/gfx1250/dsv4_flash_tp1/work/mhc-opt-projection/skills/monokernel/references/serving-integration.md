# Integrating a monokernel or hybrid glue into a serving framework (vLLM)

## Contents
1. Find the structural facts first
2. The framework contract (spec, guards, eligibility, fallback)
3. CUDA graphs
4. Speculative decoding
5. Hybrid glue runtime patterns
6. Shipping and running end to end
7. Verifying it engaged and is correct

## 1. Structural facts that decide the integration

- Is the model torch-compiled (piecewise graphs, custom ops required) or eager Python captured by full CUDA graphs
  (raw kernels can be launched inline; Python branches on the token count are baked per capture size)?
- Which layer modules own stream logic (e.g. vLLM's fused-MoE runner puts shared experts on `aux_stream()` below a
  token threshold), collectives (`RowParallelLinear(reduce_results=True)`), and quantized weight formats.
- What the model loop needs as tensors between layers (residuals, hidden states captured for a drafter, final
  outputs) — anything you defer must be materialized there.
- Which library kernels a call launches at each bucket (read dispatch code and tuned config tables).

## 2. The framework contract

The vLLM monokernel plumbing (`vllm/models/common/mono/`) is a good template for any framework:
- A per-model spec: layer list, widths, default-deny feature set, which caches it uses, how to take weight views,
  `refuse_config` (before load: CU count = grid, TP size, speculative config), `refuse_model` (after load: every
  weight view's shape, dtype, contiguity, MoE backend), `pack` (per-rank state, collective), `bind` (re-check on cache
  re-bind), `eligible(step)` returning a reason or None, `build(layer, width)` returning a launchable kernel with its
  health words, warm-up keys and compile hook.
- Guards run at config post-init and fail with every reason at once. Model refusals are agreed across TP (a MIN
  all-reduce): one rank opting out alone would leave its peers spinning forever.
- Per-step eligibility must be decided from rank-invariant data only (step width, uniform query length, pure
  speculative-verify step). Fallback reasons are counted in metrics.
- The model loop calls `forward_layer(...)`, which launches the kernel for (layer, width) or runs the framework's
  layer.
- Weight guards matter: a guard that accepts only contiguous bf16 weights refused a recipe that quantized o_proj
  and shared experts to fp8; both arms of the comparison then had to serve bf16 for those layers.

## 3. CUDA graphs

- Require full-graph modes (FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE); piecewise-only steps fall back.
- Make every supported width a capture size so no replay pads rows; the guard trimming capture sizes changes the
  baseline too — run the baseline with the same capture sizes.
- Bake the per-size decision at capture and count replays by size.
- All persistent state is allocated at attach; the epoch bump is a device op inside the graph; weight views are
  taken per launch; re-check caches when pointers change (profiling runs use throwaway caches).
- Compile every build key during warm-up before capture (compile-only launch or `kernel.warmup` with stand-ins of
  exact dtypes and strides). JIT inside capture fails or stalls; a cold JIT cache made startup 610 s vs 260 s.
- Side streams inside capture must fork from and rejoin the capture stream; tensors used on the side stream must
  stay alive until the rejoin.

## 4. Speculative decoding

- Rows per step = concurrency x (k + 1). Widths must be whole requests (multiples of k + 1).
- Speculative verify steps are chained rows of one request (consecutive tokens): stateful attention (linear
  attention / SSM / conv state) must keep per-token state snapshots for rejected drafts; accepted counts and state
  indices are kernel arguments.
- Layers of every attention kind need coverage for the step to run fused (e.g. MoE back half on MLA layers too).
- Recipes change k with concurrency; plan for the buckets the recipe actually uses.

## 5. Hybrid glue runtime patterns

- Producers write into peer memory: replace the projection's forward while the glue path is active with
  `torch.matmul(x, W.t(), out=peer_view)` and skip the framework's all-reduce; give the MoE `output=peer_view`.
  Check which backend `out=` selects and that the library honors the output buffer without a copy.
- Record library arguments: wrap the library MoE call during the profile run and store each layer's arguments
  (weights, scales, activation config); run the glue for a layer only once its arguments are recorded.
- Pending residual across layers: keep the final all-reduce un-materialized and fuse it into the next layer's
  input-side residual mix and norm; materialize at the last layer, at layers whose hidden states the framework
  captures, and before any layer that falls back.
- Reproduce the bypassed layer's stream schedule: shared experts on the aux stream with `aux.wait_stream(main)` at
  the fork and `main.wait_stream(aux)` before their output is consumed; do not fuse a routed-path op with a
  shared-path op. Place the fork where the side work overlaps latency-bound kernels.
- Eligibility per layer (shapes, activation constants, scaling factors, TP size, rows <= the glue's max); exclusive
  with the monokernel path for the same layer; bump the epoch at the first glue layer of the forward.

## 6. Shipping and running end to end

- Deliver as a patch over a pinned framework commit (and the reference PR if it is not merged): generate with
  `git add -N <new dirs>; git diff -- <paths>`; rebase by diffing the PR from its true merge-base restricted to its
  files and 3-way merging per file. Regenerate the patch after every change and make the launch script fail if
  `git apply` fails.
- An env switch selects the implementation (e.g. `VLLM_<MODEL>_GLUE=1`, `..._MONO_K2=gluon`); size shared scratch
  and peer buffers to the max over implementations.
- Keep a run script per mode (baseline / reference / yours) with identical server flags; profile mode with
  `--profiler-config`; detect device faults in logs and stop.
- Check flags across framework versions (`--kernel-config '{"monokernel":true}'` vs `--monokernel`); synthetic
  acceptance options may not combine with every rejection method.

## 7. Verifying it engaged and is correct

- Log lines naming the layers and widths that engaged, metrics counters per path and per fallback reason; a "fast"
  run that was mostly per-op is a common surprise.
- Speculative acceptance equal to baseline; a consistent drop means a numerical or semantic difference.
- Live side-by-side checks for a few steps: fused routing vs the framework's top-k on the same logits; fused
  all-reduce + residual + norm vs the per-op sequence.
- Profile both arms and compare per-layer windows (see `llm-serving-perf-analysis`).
