<!-- Maintained by the AITER team. Please open an issue or ask a listed contact to change this page. -->

# AITER — Area Contacts

> **DRAFT** — this page is being trialed and may change. Feedback is welcome in an issue.

This page tells contributors who to contact for a pull request to AITER. Each area has a contact and one or more backups. If a change touches several areas, start with the area of the main change; for MoE, pick the row of the backend you changed. The contact makes sure your pull request gets a reviewer; they may hand the review to someone else on the team.

## How to get your pull request reviewed

1. Open the pull request against `main` and make it ready for review (see [Code Review Process](CONTRIBUTE.md#code-review-process)).
   Pull requests into other branches, drafts, and titles marked WIP or DO NOT MERGE are not picked up.
2. A reviewer is usually assigned within one working day after the pull request is ready.
   You can also mention the contact for the area your change touches (see the table below);
   if it touches several areas, mention the contact of the main one.
3. Most pull requests get a first review within a few working days. The exact time depends on the size and
   risk of the change and on current project priorities; fixes for regressions and release blockers come first.
4. If nobody has responded for about a week, mention the area contact, then the backup contact.

## Areas

| Area | Paths | Contact | Backup(s) |
|---|---|---|---|
| Attention — MLA kernels (HIP / OPUS) | `aiter/mla.py`, `aiter/ops/mla*`, `csrc/kernels/mla/`, `hsa/*/mla_opus/`, `hsa/*/mla_v4/` | Kai Yang ([@kaiyang-1](https://github.com/kaiyang-1)) | Mengdie Min ([@minmengdie](https://github.com/minmengdie)) |
| Attention — MLA ASM kernels | `hsa/*/mla/`, `csrc/py_itfs_cu/asm_mla.cu` | Shengnan Xu ([@shengnxu](https://github.com/shengnxu)) | Mengdie Min ([@minmengdie](https://github.com/minmengdie)) |
| Attention — MHA forward | `aiter/ops/mha*`, `csrc/cpp_itfs/mha_fwd*`, `hsa/*/fmha_v3_fwd/`, `hsa/*/fmha_fwd*/` | Xiangxiang Li ([@shay-li77](https://github.com/shay-li77)) | Mengdie Min ([@minmengdie](https://github.com/minmengdie)) |
| Attention — MHA backward | `csrc/cpp_itfs/mha_bwd*`, `csrc/py_itfs_cu/asm_mha*bwd*`, `hsa/*/fmha_v3_bwd/` | Jim Guo ([@slippedJim](https://github.com/slippedJim)) | Xiangxiang Li ([@shay-li77](https://github.com/shay-li77)) |
| Attention — paged attention | `aiter/ops/pa*`, `csrc/cpp_itfs/pa/` | Rui Tang ([@amd-ruitang3](https://github.com/amd-ruitang3)) | Fang Che ([@fangche123](https://github.com/fangche123)) |
| Attention — MQA logits / sparse indexer | `csrc/paged_mqa_logits/`, `csrc/kernels/opus_mqa_logits/`, `csrc/kernels/sparse_attn/` | Rui Tang ([@amd-ruitang3](https://github.com/amd-ruitang3)) | Kai Yang ([@kaiyang-1](https://github.com/kaiyang-1)) |
| MoE — CK / ASM / HIP / OPUS | `aiter/fused_moe*`, `csrc/ck_gemm_moe_2stages_codegen/`, `csrc/ck_tile_gemm_moe_2stages/`, `csrc/opus_moe/`, `hsa/*/fmoe*` | Zufa Yu ([@zufayu](https://github.com/zufayu)) | Zoe Zhou ([@yzhou103](https://github.com/yzhou103)) |
| MoE — FlyDSL | `aiter/ops/flydsl/kernels/*moe*` | Yan Dai ([@yadaish](https://github.com/yadaish)) | Felix Li ([@coderfeli](https://github.com/coderfeli)) |
| MoE — Triton | `aiter/ops/triton/moe/`, `aiter/ops/triton/_triton_kernels/moe/` | Vinayak Gokhale ([@vgokhale](https://github.com/vgokhale)) | Ali Zaidy ([@azaidy](https://github.com/azaidy)), Dewei Wang ([@Dewei-Wang-sh](https://github.com/Dewei-Wang-sh)) |
| FlyDSL kernels (other than MoE) | `aiter/ops/flydsl/`, `aiter/aot/flydsl/` | Felix Li ([@coderfeli](https://github.com/coderfeli)) | Yan Dai ([@yadaish](https://github.com/yadaish)) |
| Triton / Gluon kernels (other than MoE) | `aiter/ops/triton/`, `op_tests/triton_tests/` | Vinayak Gokhale ([@vgokhale](https://github.com/vgokhale)) | Ali Zaidy ([@azaidy](https://github.com/azaidy)), Dewei Wang ([@Dewei-Wang-sh](https://github.com/Dewei-Wang-sh)) |
| GEMM — OPUS | `csrc/opus_gemm/`, `aiter/ops/opus/` | Binglue Lyu ([@demonsan](https://github.com/demonsan)) | Yifeng Huang ([@yifehuan](https://github.com/yifehuan)) |
| GEMM kernels (CK / ASM / HIP) | `csrc/ck_gemm_*/`, `csrc/gemm_*/`, `hsa/*/*gemm*/` | Yifeng Huang ([@yifehuan](https://github.com/yifehuan)) | Binglue Lyu ([@demonsan](https://github.com/demonsan)) |
| Tuning tools and tuned configs | `aiter/configs/`, `gradlib/` | Zoe Zhou ([@yzhou103](https://github.com/yzhou103)) | Yifeng Huang ([@yifehuan](https://github.com/yifehuan)) |
| Communication / all-reduce | `aiter/dist/`, `csrc/kernels/custom_all_reduce*`, `op_tests/multigpu_tests/` | Tenny Wang ([@TennyWang1223](https://github.com/TennyWang1223)) | Zufa Yu ([@zufayu](https://github.com/zufayu)) |
| Norm / RoPE / quantization / top-k / mHC ops | `aiter/ops/norm*`, `aiter/ops/rope*`, `aiter/ops/quant*`, `aiter/ops/topk*`, `aiter/ops/mhc*` | Zufa Yu ([@zufayu](https://github.com/zufayu)) | Jun Chen ([@junhaha666](https://github.com/junhaha666)) |
| JIT and build | `aiter/jit/`, `setup.py`, `csrc/cpp_itfs/`, `csrc/pybind/` | Rui Tang ([@amd-ruitang3](https://github.com/amd-ruitang3)) | Zufa Yu ([@zufayu](https://github.com/zufayu)) |
| CI and test infrastructure | `.github/`, `scripts/` | Xin Huang ([@gyohuangxin](https://github.com/gyohuangxin)) | Zufa Yu ([@zufayu](https://github.com/zufayu)) |
| Overall coordination, and anything not listed above | — | Zufa Yu ([@zufayu](https://github.com/zufayu)) | Lingpeng Jin ([@valarLip](https://github.com/valarLip)) |

## Community reviewers

None yet. Contributors who want to help review are welcome — see [REVIEWERS.md](REVIEWERS.md).
