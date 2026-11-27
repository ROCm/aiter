# Curated pytest manifests

Each `.txt` file here is a manifest for `.github/scripts/run_pytest_manifest.py`:
one pytest selector per line, optionally followed by the exact number of nodes
it must collect. The runner runs each test file in its own pytest process and
fails unless every selector collects its count and every node passes setup,
call, and teardown with no skip, xfail, or xpass. See the runner's docstring
for the full contract and its artifacts.

```bash
python3 .github/scripts/run_pytest_manifest.py \
  --manifest .github/test-manifests/mi450-gfx1250-p0.txt \
  --require-gfx gfx1250 \
  --artifact-dir test-reports/mi450-gfx1250/p0
```

`--collect-only` checks collection without running tests. `--expect-failure`
inverts the per-node rule for a manifest of known bugs. The selftest
(`.github/scripts/run_pytest_manifest_selftest.py`) needs no GPU.

When a count stops matching, review what upstream added, removed, or changed
in that test before editing the count. Never edit a count, or drop a node, to
turn a run green.

## MI450 (gfx1250) manifests

| Manifest | Nodes | Contents |
| --- | ---: | --- |
| `mi450-gfx1250-p0.txt` | 38 | Architecture and stepping guards, gfx1250 code-object loading and dispatch, A16W16 contracts, MoE routing, DSV4 fused QK norm/RoPE, DCP decode top-k merge |
| `mi450-gfx1250-p1.txt` | 229 | Triton/Gluon MLA prefill and decode, sparse paged-attention prefill and decode, MXFP4 GEMM |
| `mi450-gfx1250-known-failures.txt` | 11 | Filed gfx1250 bugs (#7536, #7777, #7778, #7779), run with `--expect-failure` so a fix is noticed |

Every P0 and P1 node passed on gfx1250 hardware, and every known failure still
failed as filed, at the time of selection. The selections run nightly on
internal MI450 CI, which reports results outside this repository. Each
manifest's header lists what was left out and why.

Two cautions for a gfx1250 runner:

- A kernel that never finishes can hold the GPU until the host is
  power-cycled. That is why `test_unified_attention.py` (#7776) is not
  selected.
- A GPU page fault aborts the whole pytest process. `test_mla_flydsl` (#7781)
  and the unshuffled MLA ring-wraparound test (#8078) do this intermittently
  and are not selected; the runner's one-process-per-file isolation keeps such
  a crash from hiding other files' results.
