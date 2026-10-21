"""Paged MQA logits with a (B, next_n) per-row context_lens table.

vLLM passes context_lens as a (B, next_n) table, the layout DeepGEMM takes;
row b ends with sequence b's full length. Every Triton/Gluon entry point must
give the same logits for that table as for the per-sequence [B] lengths, and
must reject layouts it would read wrongly.
"""

import pytest
import torch
import triton
from packaging.version import Version

from aiter import dtypes
from aiter.ops.triton.attention import pa_mqa_logits as pa_mqa_logits_module
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    deepgemm_fp8_paged_mqa_logits_persistent_schedule,
    deepgemm_fp8_paged_mqa_logits_schedule,
    deepgemm_fp8_paged_mqa_logits_stage1,
    enable_jit_gluon_pa_mqa_logits_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_weight

# entry -> arches it runs on. The gfx1250 base Gluon kernel needs KVBlockSize=1
# and ignores VarCtxSchedule, so only its preshuffle (TDM) path is covered.
ENTRIES = {
    "gluon": ("gfx942", "gfx950"),
    "gluon_preshuffle": ("gfx942", "gfx950", "gfx1250"),
    "gluon_varctx": ("gfx942", "gfx950"),
    "gluon_persistent": ("gfx942", "gfx950"),
    "triton": ("gfx942", "gfx950"),
    "stage1": ("gfx942", "gfx950"),
}
GLUON_ENTRIES = ("gluon", "gluon_preshuffle", "gluon_varctx", "gluon_persistent")
# stage1 and the varctx kernel write whole tail tiles past the end of the
# output (known bug, separate fix): up to ChunkK (256) columns for stage1 and
# up to KVBlockSize for varctx. Keep the row width a multiple of both until then.
ROW_ALIGN = 256


@pytest.mark.parametrize("entry", list(ENTRIES))
@torch.inference_mode()
def test_paged_mqa_logits_2d_context_lens(entry: str, monkeypatch) -> None:
    arch = arch_info.get_arch()
    if arch not in ENTRIES[entry]:
        pytest.skip(f"{entry} is not covered on {arch}")
    if entry in GLUON_ENTRIES and not enable_jit_gluon_pa_mqa_logits_kernel:
        pytest.skip("Requires the Gluon JIT paged MQA kernel")
    if entry == "gluon_persistent" and Version(triton.__version__) < Version("3.5.0"):
        pytest.skip("PersistentSchedule requires Triton >= 3.5")
    if entry == "triton":
        # The Triton kernel is the fallback when Gluon is off.
        monkeypatch.setattr(pa_mqa_logits_module, "enable_gluon_pa_mqa_logits", False)

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(6021)
    next_n, heads, hidden_dim, block_size = 3, 32, 128, 64
    context_lengths = (3000, 257, 64, 3)
    batch = len(context_lengths)
    max_model_len = -(-max(context_lengths) // ROW_ALIGN) * ROW_ALIGN
    max_pages = max_model_len // block_size
    num_blocks = batch * max_pages
    preshuffle = entry in ("gluon_preshuffle", "gluon_varctx")

    q = torch.randn(
        batch,
        next_n,
        heads,
        hidden_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(dtypes.fp8)
    kv = torch.randn(
        num_blocks,
        block_size,
        hidden_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(dtypes.fp8)
    if preshuffle:
        kv = shuffle_weight(kv, layout=(16, 16))
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
        .view(batch, max_pages)
        .to(torch.int32)
    )
    seq_lens = torch.tensor(context_lengths, dtype=torch.int32, device=device)
    row_lens = (
        seq_lens[:, None]
        - next_n
        + 1
        + torch.arange(next_n, dtype=torch.int32, device=device)
    )

    def run(context_lens):
        if entry == "stage1":
            out = torch.full(
                (heads, batch * next_n, max_model_len), float("-inf"), device=device
            )
            # ChunkQ is the head count here, not a tuning choice: the default
            # (64) is wider than the 32 heads of this shape.
            deepgemm_fp8_paged_mqa_logits_stage1(
                q,
                cache,
                weights,
                out,
                context_lens,
                block_tables,
                max_model_len,
                ChunkQ=heads,
            )
            return out
        schedule = persistent = None
        if entry == "gluon_varctx":
            schedule = deepgemm_fp8_paged_mqa_logits_schedule(
                batch, next_n, context_lens, max_model_len
            )
        if entry == "gluon_persistent":
            persistent = deepgemm_fp8_paged_mqa_logits_persistent_schedule(
                batch, next_n, context_lens, max_model_len
            )
        out = torch.full((batch * next_n, max_model_len), float("-inf"), device=device)
        deepgemm_fp8_paged_mqa_logits(
            q,
            cache,
            weights,
            out,
            context_lens,
            block_tables,
            max_model_len,
            Preshuffle=preshuffle,
            KVBlockSize=block_size,
            VarCtxSchedule=schedule,
            PersistentSchedule=persistent,
        )
        return out

    expected = run(seq_lens)
    for context_lens in (row_lens, seq_lens[:, None]):
        torch.testing.assert_close(run(context_lens), expected, rtol=0, atol=0)
    if entry == "gluon_varctx":
        torch.testing.assert_close(
            deepgemm_fp8_paged_mqa_logits_schedule(
                batch, next_n, row_lens, max_model_len
            ),
            deepgemm_fp8_paged_mqa_logits_schedule(
                batch, next_n, seq_lens, max_model_len
            ),
            rtol=0,
            atol=0,
        )

    # A row count other than B, a column count other than 1 or next_n, or a
    # tensor on another device are rejected. The persistent schedule makes its
    # own contiguous int32 copy; every other entry also rejects a strided [B]
    # or [B, n] and a dtype other than int32.
    bad_inputs = [
        seq_lens[:-1],
        torch.cat([seq_lens, seq_lens[:1]]),
        row_lens[:-1],
        torch.cat([row_lens, row_lens[:1]]),
        row_lens[:, :0],
        row_lens[:, 1:].contiguous(),
        row_lens.cpu(),
    ]
    if entry != "gluon_persistent":
        bad_inputs += [
            row_lens[:, -1],
            row_lens.t().contiguous().t(),
            row_lens.long(),
        ]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            run(bad)
