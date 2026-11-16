# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regression tests for grouped GEMM's host C ABI (issue #7241)."""

import ctypes
import inspect

import pytest
import torch

pytest.importorskip("flydsl.compiler")

from flydsl.compiler.jit_executor import CallState
from flydsl.compiler.jit_function import _build_call_spec

from aiter.ops.flydsl import grouped_gemm_mxfp4 as grouped
from aiter.ops.flydsl.kernels import tensor_shim
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext
from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm

_ADDRESS_ARGS = (
    "arg_c",
    "arg_a",
    "arg_b",
    "arg_scale_a",
    "arg_scale_b",
    "arg_m_tile_map",
    "arg_bias",
    "arg_quant_scale",
    "arg_ep_row_map",
)


@pytest.fixture
def dispatch(monkeypatch):
    """Exercise the real host C ABI while replacing GPU compilation/execution."""
    signature = inspect.signature(launch_gemm_a8w4_tdm.func)
    names = tuple(signature.parameters)
    compiled = []
    launches = []

    def compile_and_run(exe, *args):
        assert exe is launch_gemm_a8w4_tdm
        slots = _build_call_spec(signature, args)
        address_slots = {
            names[index]: ctype
            for index, ctype, _fill in slots
            if index >= 0 and names[index] in _ADDRESS_ARGS
        }
        state = CallState(slots, lambda _packed: None)

        def dispatch(*live_args):
            state(live_args)
            launches.append(
                {
                    names[index]: storage.value
                    for (index, _ctype, _fill), storage in zip(
                        slots, state._tls.storages
                    )
                    if index >= 0
                }
            )

        compiled.append(dispatch)
        dispatch(*args)
        assert address_slots == dict.fromkeys(_ADDRESS_ARGS, ctypes.c_void_p)
        assert not any(
            index >= 0 and names[index] == "ep_arena_handle"
            for index, _ctype, _fill in slots
        )
        return dispatch

    monkeypatch.setattr(tensor_shim, "_compile_and_run", compile_and_run)
    monkeypatch.setattr(
        launch_gemm_a8w4_tdm, "_cf_by_specialization", {}, raising=False
    )
    return compiled, launches


@pytest.mark.parametrize("ep_quant_bits", [None, 0, 4, 8])
@pytest.mark.parametrize("use_default_buffers", [False, True])
def test_large_buffers_pack_on_first_and_cached_calls(
    dispatch, ep_quant_bits, use_default_buffers
):
    # Meta tensors exercise multi-billion-element dimensions without allocating
    # GPU memory. Arena addresses remain constexprs and are not ABI slots.
    compiled, launches = dispatch
    arena_handle = 132319086288896

    def meta(shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, device="meta")

    source_map = meta((64, 8192), torch.int32)
    scatter = (
        None
        if ep_quant_bits is None
        else Stage2ScatterContext(
            arena_handle=arena_handle,
            combine_input_offset=1611085056,
            slot_stride_bytes=16384,
            max_tokens_per_rank=8192,
            world_size=4,
            source_token_map=source_map,
            combine_quant_bits=ep_quant_bits,
        )
    )
    for shape in [(64, 8192, 7168), (64, 1, 7168)]:
        out = meta(shape, torch.bfloat16)
        assert (
            grouped.flydsl_grouped_gemm_a8w4_masked(
                out,
                meta((64, 256)),
                meta((64, 7168, 128)),
                meta((2**33,)),  # int32 view still has 2**31 elements
                meta((2**33,)),
                meta((64,), torch.int32),
                n_experts=64,
                contiguous_m=shape[0] * shape[1],
                N=7168,
                K=256,
                stream=0,
                quant_scale=None if use_default_buffers else meta((2**31,)),
                ep_row_map=(
                    None if use_default_buffers else meta((2**31,), torch.int32)
                ),
                stage2_scatter=scatter,
            )
            is out
        )

    assert len(compiled) == 1
    assert len(launches) == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.uint8])
def test_pitched_and_offset_addresses_survive_cached_dispatch(dispatch, dtype):
    compiled, launches = dispatch
    for rows in (64, 128):
        out = torch.empty((rows, 256), dtype=dtype)
        a = torch.empty((rows, 260), dtype=torch.uint8)[:, 4:]
        w = torch.empty((1, 256, 128), dtype=torch.uint8)
        # Compact scales are a pitched window with a nonzero storage offset.
        storage = torch.empty(rows * 512 + 4, dtype=torch.uint8)
        a_scales = torch.as_strided(storage, (rows, 224), (512, 1), storage_offset=4)
        w_scales = torch.empty((256, 8), dtype=torch.uint8)
        tile_map = torch.empty(1, dtype=torch.int32)
        quant_scale = torch.empty((rows, 8), dtype=torch.uint8)[:, 4:]
        row_map = torch.empty(rows + 1, dtype=torch.int32)[1:]
        assert (
            grouped.flydsl_grouped_gemm_a8w4_masked(
                out,
                a,
                w,
                a_scales,
                w_scales,
                tile_map,
                n_experts=1,
                contiguous_m=rows,
                N=256,
                K=256,
                stream=0,
                out_is_f16=int(dtype == torch.float16),
                stage1_act=int(dtype == torch.uint8),
                stage1_quant_out=int(dtype == torch.uint8),
                quant_scale=quant_scale,
                ep_row_map=row_map,
            )
            is out
        )
        expected = {
            "arg_c": out,
            "arg_a": a,
            "arg_b": w,
            "arg_scale_a": a_scales,
            "arg_scale_b": w_scales,
            "arg_m_tile_map": tile_map,
            "arg_bias": a,
            "arg_quant_scale": quant_scale,
            "arg_ep_row_map": row_map,
        }
        assert {name: launches[-1][name] for name in expected} == {
            name: tensor.data_ptr() for name, tensor in expected.items()
        }
        assert launches[-1]["i32_m"] == rows
    assert len(compiled) == 1
    assert len(launches) == 2


def test_large_device_pointer_is_preserved_by_the_cabi():
    address = 132319086288896
    pointer = tensor_shim.flyc.from_c_void_p(tensor_shim.fx.Uint8, address)

    def launcher(buffer):
        pass

    slots = _build_call_spec(inspect.signature(launcher), (pointer,))
    state = CallState(slots, lambda _packed: None)
    state((pointer,))
    assert state._tls.storages[0].value == address


def test_changing_arena_selects_a_new_specialization(dispatch):
    compiled, launches = dispatch
    out = torch.empty((64, 256), dtype=torch.bfloat16, device="meta")
    data = torch.empty((64, 256), dtype=torch.uint8, device="meta")
    ids = torch.empty(64, dtype=torch.int32, device="meta")
    for handle in (132319086288896, 132319086288896, 132319086292992):
        scatter = Stage2ScatterContext(handle, 1611085056, 16384, 8192, 4, ids)
        grouped.flydsl_grouped_gemm_a8w4_masked(
            out,
            data,
            data,
            data,
            data,
            ids,
            n_experts=1,
            contiguous_m=64,
            N=256,
            K=256,
            stream=0,
            stage2_scatter=scatter,
        )
    assert len(compiled) == 2
    assert len(launches) == 3
