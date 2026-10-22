See [report](../../../docs/fused_dsv41_pr_v41_fusion.md) for the selected implementation.

`executed_runners.tar.gz` preserves the executed experiment scripts exactly, including source-fragment substitutions that must not be reformatted. These phase-specific scripts retain the pinned remote paths and require the earlier benchmark fixture archives and retained baseline binary. Use the public pytest files for current correctness validation. The kernel rebuild script under csrc/kernels/mla/fused_dsv41/pr_v41 is self-contained within the checkout apart from the documented compiler toolchain.
