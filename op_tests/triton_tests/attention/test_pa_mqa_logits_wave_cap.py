"""Paged MQA logits: the gfx950 plain Gluon kernel caps its waves_per_eu hint.

A WavePerEU above the tuned cap for a ChunkK would make the kernel spill, so the
compile hint is lowered to the cap. The grid is still sized from the caller's
WavePerEU, and the logits must not change.
"""

import pytest
import torch
import triton

from aiter import dtypes
from aiter.ops.triton.attention import pa_mqa_logits as pa_mqa_logits_module
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    enable_jit_gluon_pa_mqa_logits_kernel,
)
from aiter.ops.triton.utils._triton import arch_info


def _expected_split_kv(wave_per_eu, cu_count, tile_q_count, max_model_len, chunk_k):
    # Mirrors the non-gfx1250 SplitKV sizing in deepgemm_fp8_paged_mqa_logits.
    split_kv = max(1, triton.cdiv(wave_per_eu * cu_count, tile_q_count))
    chunks = max(1, triton.cdiv(max_model_len, chunk_k))
    split_kv = min(max(split_kv, triton.cdiv(chunks, 32)), split_kv * 2)
    return min(split_kv, chunks)


@pytest.mark.parametrize("chunk_k", [64, 128, 256])
@torch.inference_mode()
def test_plain_gluon_wave_cap(chunk_k: int, monkeypatch) -> None:
    if arch_info.get_arch() != "gfx950":
        pytest.skip("The wave cap is tuned for gfx950 only")
    if not enable_jit_gluon_pa_mqa_logits_kernel:
        pytest.skip("Requires the Gluon JIT paged MQA kernel")
    cap_table = pa_mqa_logits_module._plain_wave_cap()
    assert cap_table is not None
    cap = cap_table["max_waves_per_eu_by_chunk_k"][str(chunk_k)]
    wave_per_eu = cap + 1

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(950)
    batch, next_n, heads, hidden_dim, block_size = 4, 1, 64, 128, 1
    max_model_len = 2048
    num_blocks = batch * max_model_len

    q = torch.randn(
        batch, next_n, heads, hidden_dim, generator=generator, device=device
    ).to(dtypes.fp8)
    kv = torch.randn(
        num_blocks, block_size, hidden_dim, generator=generator, device=device
    ).to(dtypes.fp8)
    scales = 0.25 + torch.rand(
        num_blocks, block_size, generator=generator, device=device
    )
    weights = torch.randn(batch * next_n, heads, generator=generator, device=device)
    packed = torch.empty(
        num_blocks, block_size * (hidden_dim + 4), dtype=torch.uint8, device=device
    )
    value_bytes = block_size * hidden_dim
    packed[:, :value_bytes] = kv.reshape(num_blocks, -1).view(torch.uint8)
    packed[:, value_bytes:] = scales.view(torch.uint8)
    cache = packed.view(num_blocks, block_size, 1, hidden_dim + 4)
    block_tables = (
        torch.randperm(num_blocks, generator=generator, device=device)
        .view(batch, max_model_len)
        .to(torch.int32)
    )
    context_lens = torch.tensor((2048, 1500, 333, 7), dtype=torch.int32, device=device)

    launches = []
    compile_kernel = pa_mqa_logits_module._compile_deepgemm_fp8_paged_mqa_logits

    class _Spy:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launches.append((self.kernel, grid))
            return self.kernel[grid]

        def __getattr__(self, name):
            return getattr(self.kernel, name)

    def spy_compile(*args, **kwargs):
        return _Spy(compile_kernel(*args, **kwargs))

    monkeypatch.setattr(
        pa_mqa_logits_module, "_compile_deepgemm_fp8_paged_mqa_logits", spy_compile
    )

    def run(wave):
        out = torch.full((batch * next_n, max_model_len), float("-inf"), device=device)
        deepgemm_fp8_paged_mqa_logits(
            q,
            cache,
            weights,
            out,
            context_lens,
            block_tables,
            max_model_len,
            ChunkK=chunk_k,
            WavePerEU=wave,
        )
        return out

    expected = run(1)
    got = run(wave_per_eu)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    kernel, grid = launches[-1]
    assert kernel.metadata.waves_per_eu == cap
    cu_count = pa_mqa_logits_module.get_num_sms()
    split_kv = _expected_split_kv(
        wave_per_eu, cu_count, batch * next_n, max_model_len, chunk_k
    )
    assert grid[0] == batch * next_n * split_kv
