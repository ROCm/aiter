# AITER Release Checklist

## Pre-Release

### Branch & Code
- [ ] Release branch created from agreed commit (`git checkout -b release/vX.Y.Z <commit>`)
- [ ] Known blockers resolved (check GitHub issues labeled `blocker`)
- [ ] Release notes generated (`./scripts/generate_changelog.sh <prev_tag> <release_branch>`)
- [ ] Highlights section reviewed and filled in

### CI Validation (on release branch)
- [ ] `aiter-test` workflow green (build + unit tests)
- [ ] `triton-test` workflow green
- [ ] `atom-test` workflow green

### Downstream Validation
- [ ] SGLang downstream test pass (`sglang_downstream.yaml`)
  - [ ] DeepSeek-R1-MXFP4 accuracy >= 0.93 (GSM8K 200-question)
  - [ ] GPT-OSS functional test pass
- [ ] vLLM benchmark no regression (`vllm_benchmark.yaml`)
- [ ] ATOM E2E test pass

### Build Validation
- [ ] whl build success — Python 3.10 (gfx942;gfx950)
- [ ] whl build success — Python 3.12 (gfx942;gfx950)
- [ ] Import smoke test: `pip install dist/*.whl && python -c "import aiter; print(aiter.__version__)"`

## Release

- [ ] Tag created: `git tag vX.Y.Z && git push origin vX.Y.Z`
- [ ] GitHub release published with release notes and whl artifacts
- [ ] Release branch pushed to origin (ROCm/aiter)

## Post-Release

- [ ] Notify downstream teams (SGLang, vLLM, ATOM) of new release
- [ ] Update downstream pinned AITER versions where applicable
- [ ] Close related GitHub issues

## Downstream Test Coverage Matrix

| Project | CI Workflow | Status | Notes |
|---------|-----------|--------|-------|
| SGLang | `sglang_downstream.yaml` | Active | Accuracy + functional tests |
| vLLM | `vllm_benchmark.yaml` | Active | Benchmark only, no accuracy |
| ATOM | `atom-test.yaml` | Active | E2E model tests |
| TE | None | GAP | Need to scope |
| Megatron | None | GAP | Need to scope |
| Primus | None | GAP | Need to scope |
| hipblaslt | None | GAP | Need to scope |
| Flashinfer | None | GAP | Need to scope |
