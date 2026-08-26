import pytest
import torch


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(props, "gcnArchName", "").split(":")[0] == "gfx950"


requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(), reason="FP4 KV-cache kernel requires gfx950"
)

_NUM_TOKENS = 17
_HEAD_NUM = 1
_DIM = 128
_ROPE_DIM = 64
_KV_BLOCK_SIZE = 64
_NUM_BLOCKS = 20
_PAYLOAD_SENTINEL = 0xA5
_SCALE_SENTINEL = 0x5A
_SLOTS = (
    327,
    63,
    578,
    159,
    764,
    -1,
    272,
    493,
    67,
    668,
    251,
    524,
    434,
    9,
    705,
    362,
    132,
)


def _make_inputs():
    generator = torch.Generator(device="cuda").manual_seed(20260826)
    input = torch.randn(
        (_NUM_TOKENS, _HEAD_NUM, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        (_DIM,), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    positions = torch.tensor(
        [3, 17, 5, 29, 11, 23, 2, 31, 7, 19, 13, 0, 27, 9, 21, 15, 25],
        dtype=torch.int64,
        device="cuda",
    )
    freqs = torch.randn(
        (32, _ROPE_DIM // 2),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    cos = torch.cos(freqs).to(torch.bfloat16)
    sin = torch.sin(freqs).to(torch.bfloat16)
    return input, norm_weight, cos, sin, positions


def _make_cache():
    from aiter import dtypes

    payload = torch.full(
        (_NUM_BLOCKS, _DIM // 128, 4, _KV_BLOCK_SIZE, 16),
        _PAYLOAD_SENTINEL,
        dtype=torch.uint8,
        device="cuda",
    ).view(dtypes.fp4x2)
    scale = torch.full(
        (_NUM_BLOCKS, _DIM // 128, 4, _KV_BLOCK_SIZE),
        _SCALE_SENTINEL,
        dtype=torch.uint8,
        device="cuda",
    )
    return payload, scale


def _run_kernel(
    payload,
    scale,
    input,
    norm_weight,
    cos,
    sin,
    positions,
    slot_mapping,
):
    import aiter

    aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache(
        payload,
        scale,
        input,
        norm_weight,
        cos,
        sin,
        positions,
        slot_mapping,
        1e-6,
        _ROPE_DIM,
        _KV_BLOCK_SIZE,
        group_size=32,
        shuffle_scale=True,
        do_rotate_act=True,
    )


def _assert_exact_bytes(actual, expected):
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


def _assert_unaddressed_bytes_are_sentinel(payload, scale):
    payload_bytes = payload.view(torch.uint8)
    payload_written = torch.zeros_like(payload_bytes, dtype=torch.bool)
    scale_written = torch.zeros_like(scale, dtype=torch.bool)

    for slot in _SLOTS:
        if slot < 0:
            continue
        block, position = divmod(slot, _KV_BLOCK_SIZE)
        payload_written[block, :, :, position, :] = True
        scale_position = (position % 16) * (_KV_BLOCK_SIZE // 16) + position // 16
        scale_written[block, :, :, scale_position] = True

    assert not torch.any(payload_bytes[~payload_written] != _PAYLOAD_SENTINEL).item()
    assert not torch.any(scale[~scale_written] != _SCALE_SENTINEL).item()
    assert torch.any(payload_bytes[payload_written] != _PAYLOAD_SENTINEL).item()
    assert torch.any(scale[scale_written] != _SCALE_SENTINEL).item()


@requires_gfx950
def test_scattered_slots_match_single_row_calls():
    input, norm_weight, cos, sin, positions = _make_inputs()
    slot_mapping = torch.tensor(_SLOTS, dtype=torch.int64, device="cuda")
    batched_payload, batched_scale = _make_cache()
    rowwise_payload, rowwise_scale = _make_cache()

    assert input.shape == (_NUM_TOKENS, _HEAD_NUM, _DIM)
    assert any(slot < 0 for slot in _SLOTS)
    assert len({slot // _KV_BLOCK_SIZE for slot in _SLOTS if slot >= 0}) >= 8

    _run_kernel(
        batched_payload,
        batched_scale,
        input,
        norm_weight,
        cos,
        sin,
        positions,
        slot_mapping,
    )

    for row, slot in enumerate(_SLOTS):
        if slot < 0:
            payload_before = rowwise_payload.view(torch.uint8).clone()
            scale_before = rowwise_scale.clone()
        _run_kernel(
            rowwise_payload,
            rowwise_scale,
            input[row : row + 1],
            norm_weight,
            cos,
            sin,
            positions[row : row + 1],
            slot_mapping[row : row + 1],
        )
        if slot < 0:
            _assert_exact_bytes(rowwise_payload, payload_before)
            _assert_exact_bytes(rowwise_scale, scale_before)

    _assert_exact_bytes(batched_payload, rowwise_payload)
    _assert_exact_bytes(batched_scale, rowwise_scale)
    _assert_unaddressed_bytes_are_sentinel(batched_payload, batched_scale)
    _assert_unaddressed_bytes_are_sentinel(rowwise_payload, rowwise_scale)
