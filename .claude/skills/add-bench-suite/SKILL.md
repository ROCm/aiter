---
name: add-bench-suite
description: Add a model suite (a --<model> flag) to op_tests/bench_gfx1250_combo.py so its operators are benchmarked on gfx1250 at the shapes one GPU sees when the model is served, then validate it with the run-bench skill. Use when asked to add a new model's microbench, to cover a model's MoE / Mega MoE / MLA / indexer shapes, or to move an existing suite to a new serving recipe.
---

# Add a bench suite

A suite is a list of combo ops run at one model's per-GPU serving shapes. The
`--dsr1` and `--dsv41` suites are worked examples; read them before starting.

## 1. Collect the facts

Take every number from a source, not from memory:

- **Model** - the checkpoint's `config.json` (`text_config` for multimodal ones):
  hidden size, MoE intermediate size, routed / shared experts, top-k, layers,
  attention heads, head dims, index top-k, quantization (expert and non-expert
  dtypes, block size), draft-model MoE if any.
- **Serving recipe** - the serve script in the image (e.g.
  `/app/endpoints-recipe-dsv41/serve_dep4.sh`, or `/app/scripts/<model>/`):
  TP / EP / DP attention, `--max-num-seqs`, `--max-num-batched-tokens`, prefill
  chunk, speculative tokens, KV and index cache dtype, and the MoE / MegaMoE
  environment (dispatch wire, fusion switches, AITER_FORCE_A8W4).
- **Steady state** - what one rank actually runs under load, from a server log or
  trace when one exists (running sequences per rank, prefill batch tokens).

## 2. Turn them into per-GPU shapes

- Experts per GPU = routed experts / EP, plus the shared expert only if the
  recipe fuses it into the MoE. MoE rows per GPU = tokens per rank x top-k.
- Heads per GPU: all of them under DP attention, heads / TP otherwise.
- Tokens per rank: decode = sequences x (1 + speculative tokens); prefill is
  capped by the chunk.
- Sweep powers of 2 up to the prefill chunk. Ops whose cost depends on the batch
  shape (MLA decode/prefill) also get the recipe's steady-state decode points;
  MoE and Mega MoE do not need them. Stop decode-only sweeps at the largest
  decode a rank can carry.
- Attention KV per token is the compressed history (context / each compression
  ratio the model uses) plus the sliding window; index top-k only at the model's
  value (V4.1: 512).
- Only the precisions the model runs (e.g. V4.1 MLA: bf16 and fp4, no fp8).

Write the shapes down with their source before coding, and get them confirmed.

## 3. Wire the suite into the combo

- Constants next to the existing ones: `_<MODEL>_TOKENS`, `_<MODEL>_MOE_CONFIG`,
  per-op case lists. Comment each with where it comes from.
- In each `run_<op>`, branch on `args.suite == "<model>"`; leave the other
  suites' paths unchanged.
- In-process UT calls go inside `_ut_timing()`. A child UT that times through
  `aiter.test_common` (`run_perftest` / `@perftest`) is launched with
  `_run_child(..., ut_timing=True)`; a bench with its own timing (e.g.
  `bench_mega_moe.py`, cuda.Event) needs neither. This keeps every case timed as
  its UT times it alone.
- Pass several sizes to one child when the UT takes a list (`-n 1 2 4 ...`)
  instead of starting a process per size.
- A UT whose sweep runs at import can still be called in-process: load it only
  up to its argument parser (`_load_ut_head`), as `topk` does.
- UT interfaces drift on main. When a child fails on its arguments or prints no
  JSON table, look for an open aiter PR that changes the UT before adapting the
  call; if the suite needs it, validate on top of it and name it in the PR.
- Add `<MODEL>_OPS`, the `--<model>` flag, the `args.suite` mapping, the usage
  lines at the top of the file and the sweep entry in its Environment section.
- `ruff check --no-cache`, `black --check` and `python3 -m py_compile` on the
  file.

## 4. Validate on gfx1250

1. Fresh container, aiter at the commit under test, JIT caches cleared, FlyDSL
   at the `requirements.txt` pin. Ops that need four GPUs (`mega_moe`) need a
   four-GPU node; use `gpu-lock` where the node has it.
2. Smoke run with a few sizes:
   `AITER_BENCH_TOKENS=1,384,1536 bash .claude/skills/run-bench/run.sh <model> <out> 0,1,2,3`.
   Every op must produce its tables with no `err_msg`; fix what fails before the
   full sweep.
3. Full run with run-bench. Check exit code 0, failed cases 0, the expected case
   counts per table, and kernel names where the op reports them.
4. If the suite needs reading notes, add a `### <model>` section to the run-bench
   SKILL.md - only what a reader needs to interpret the numbers.
