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
            if index >= 0 and names[index] in _ADDRESS_ARGS and ctype is ctypes.c_void_p
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
                    if index >= 0 and hasattr(storage, "value")
                }
            )
            launches[-1]["layout_buffers"] = {
                names[index]: bytes(storage)
                for (index, _ctype, _fill), storage in zip(slots, state._tls.storages)
                if index >= 0 and isinstance(storage, ctypes.Array)
            }

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
@pytest.mark.parametrize("a_preshuffled", [0, 1])
def test_large_buffers_pack_on_first_and_cached_calls(
    dispatch, ep_quant_bits, use_default_buffers, a_preshuffled
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
    shapes = [(64, 8192, 7168), (64, 1, 7168)]
    for shape in shapes * 2:
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
                a_is_fp4=a_preshuffled,
                a_preshuffled=a_preshuffled,
                num_buffers=2,
                stream=0,
                quant_scale=None if use_default_buffers else meta((2**31,)),
                ep_row_map=(
                    None if use_default_buffers else meta((2**31,), torch.int32)
                ),
                stage2_scatter=scatter,
            )
            is out
        )

    # Every layout uses Pointer packing, including dimensions beyond i32.
    assert len(compiled) == 1
    assert len(launches) == 4


@pytest.mark.parametrize("a_preshuffled", [0, 1])
def test_output_dimension_boundary_preserves_cached_abis(dispatch, a_preshuffled):
    compiled, launches = dispatch

    def meta(shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, device="meta")

    for elements in (2**31 - 1, 2**31) * 2:
        out = meta((elements,), torch.bfloat16)
        grouped.flydsl_grouped_gemm_a8w4_masked(
            out,
            meta((64, 256)),
            meta((1, 256, 128)),
            meta((256,)),
            meta((256,)),
            meta((1,), torch.int32),
            n_experts=1,
            contiguous_m=64,
            N=256,
            K=256,
            a_is_fp4=a_preshuffled,
            a_preshuffled=a_preshuffled,
            num_buffers=2,
            stream=0,
            quant_scale=meta((1,)),
            ep_row_map=meta((64,), torch.int32),
        )
        layout = launches[-1]["layout_buffers"].get("arg_c")
        assert layout is None
        assert (launches[-1]["arg_c"] or 0) == out.data_ptr()
    assert len(compiled) == 1
    assert len(launches) == 4


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.uint8])
@pytest.mark.parametrize("elements", [3, 4])
def test_default_optional_buffers_allow_odd_output_sizes(dispatch, dtype, elements):
    compiled, launches = dispatch
    for _ in range(2):
        out = torch.empty(elements, dtype=dtype)
        grouped.flydsl_grouped_gemm_a8w4_masked(
            out,
            torch.empty((elements, 256), dtype=torch.uint8),
            torch.empty((1, 1, 128), dtype=torch.uint8),
            torch.empty(256, dtype=torch.uint8),
            torch.empty(256, dtype=torch.uint8),
            torch.empty(1, dtype=torch.int32),
            n_experts=1,
            contiguous_m=elements,
            N=1,
            K=256,
            stream=0,
            out_is_f16=int(dtype == torch.float16),
            stage1_quant_out=int(dtype == torch.uint8),
        )
        for name in ("arg_c", "arg_quant_scale", "arg_ep_row_map"):
            assert launches[-1][name] == out.data_ptr()
        row_layout = launches[-1]["layout_buffers"].get("arg_ep_row_map")
        assert row_layout is None
    assert len(compiled) == 1
    assert len(launches) == 2


@pytest.mark.parametrize("a_preshuffled", [0, 1])
def test_scale_byte_offset_changes_cached_abi(dispatch, a_preshuffled):
    compiled, launches = dispatch
    for offset in (1, 4) * 2:
        storage = torch.empty(256 + offset, dtype=torch.uint8)
        scale_a = storage[offset:]
        out = torch.empty((64, 256), dtype=torch.bfloat16)
        grouped.flydsl_grouped_gemm_a8w4_masked(
            out,
            torch.empty((64, 256), dtype=torch.uint8),
            torch.empty((1, 256, 128), dtype=torch.uint8),
            scale_a,
            torch.empty(256, dtype=torch.uint8),
            torch.empty(1, dtype=torch.int32),
            n_experts=1,
            contiguous_m=64,
            N=256,
            K=256,
            a_is_fp4=a_preshuffled,
            a_preshuffled=a_preshuffled,
            num_buffers=2,
            stream=0,
        )
        assert launches[-1]["arg_scale_a"] == scale_a.data_ptr()
        layout = launches[-1]["layout_buffers"].get("arg_scale_a")
        assert layout is None
    assert len(compiled) == 1
    assert len(launches) == 4


@pytest.mark.parametrize("native_quant_output", [False, True])
def test_quant_output_layout_preserves_size_and_offset_abis(
    dispatch, native_quant_output
):
    compiled, launches = dispatch
    for _ in range(2):
        for elements in (2**31 - 1, 2**31):
            for offset in (1, 4):
                out = torch.empty(elements, dtype=torch.uint8, device="meta")
                storage = torch.empty(256 + offset, dtype=torch.uint8)
                scale_a = storage[offset:]
                grouped.flydsl_grouped_gemm_a8w4_masked(
                    out,
                    torch.empty((64, 256), dtype=torch.uint8, device="meta"),
                    torch.empty((1, 256, 128), dtype=torch.uint8, device="meta"),
                    scale_a,
                    torch.empty(256, dtype=torch.uint8, device="meta"),
                    torch.empty(1, dtype=torch.int32, device="meta"),
                    n_experts=1,
                    contiguous_m=64,
                    N=256,
                    K=256,
                    stream=0,
                    a_is_fp4=1,
                    stage1_act=1,
                    stage1_quant_out=1,
                    quant_payload_preshuffled=native_quant_output,
                    quant_scale=torch.empty(512, dtype=torch.uint8, device="meta"),
                    ep_row_map=torch.empty(64, dtype=torch.int32, device="meta"),
                )
                launch = launches[-1]
                assert (launch["arg_c"] or 0) == out.data_ptr()
                assert launch["arg_scale_a"] == scale_a.data_ptr()
                assert launch["layout_buffers"].get("arg_c") is None
                assert launch["layout_buffers"].get("arg_scale_a") is None
    assert len(compiled) == 1
    assert len(launches) == 8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.uint8])
def test_persistent_buffers_keep_tensor_dimensions_bounded(dispatch, dtype):
    compiled, launches = dispatch

    def meta(elements, dtype=torch.uint8):
        return torch.empty(elements, dtype=dtype, device="meta")

    for elements in (2**31 - 1, 2**31) * 2:
        out = meta(elements, dtype)
        grouped.flydsl_grouped_gemm_a8w4_masked(
            out,
            meta(256 * 3584),
            meta(256 * 3584),
            meta(2**33),
            meta(2**33),
            meta(256, torch.int32),
            n_experts=64,
            contiguous_m=256,
            N=7168,
            K=7168,
            tile_m=256,
            tile_n=256,
            tile_k=256,
            m_warp=2,
            n_warp=2,
            num_buffers=4,
            cluster_m=4,
            cluster_n=4,
            persistent_workers=256,
            a_is_fp4=1,
            a_preshuffled=1,
            stream=0,
            out_is_f16=int(dtype == torch.float16),
            stage1_act=int(dtype == torch.uint8),
            stage1_quant_out=int(dtype == torch.uint8),
        )
        launch = launches[-1]
        assert (launch["arg_c"] or 0) == out.data_ptr()
        # Runtime Tensor layout dimensions occupy signed i32 slots. The
        # actual buffers can be larger; these address-only views always fit.
        for name in (
            "arg_c",
            "arg_scale_a",
            "arg_scale_b",
            "arg_quant_scale",
            "arg_ep_row_map",
        ):
            assert (
                int.from_bytes(launch["layout_buffers"][name], "little", signed=True)
                == 1
            )
    assert len(compiled) == 1
    assert len(launches) == 4


def test_persistent_pitched_scale_offsets_select_compatible_cached_abis(dispatch):
    compiled, launches = dispatch
    for offset in (1, 4) * 2:
        out = torch.empty((256, 7168), dtype=torch.bfloat16, device="meta")
        storage = torch.empty(256 * 512 + offset, dtype=torch.uint8)
        scale = storage.as_strided((256, 225), (512, 1), offset)
        data = torch.empty(256 * 1024, dtype=torch.uint8, device="meta")
        grouped.flydsl_grouped_gemm_a8w4_masked(
            out,
            data,
            data,
            scale,
            data,
            torch.empty(256, dtype=torch.int32),
            n_experts=64,
            contiguous_m=256,
            N=7168,
            K=2048,
            tile_m=256,
            tile_n=256,
            tile_k=256,
            m_warp=2,
            n_warp=2,
            num_buffers=4,
            cluster_m=4,
            cluster_n=4,
            persistent_workers=256,
            a_is_fp4=1,
            a_preshuffled=1,
            stream=0,
        )
        assert launches[-1]["arg_scale_a"] == scale.data_ptr()
        layout = launches[-1]["layout_buffers"].get("arg_scale_a")
        assert (layout is None) == (offset == 1)
    assert len(compiled) == 2
    assert len(launches) == 4


def test_persistent_empty_and_nonempty_buffers_keep_compatible_cached_abis(dispatch):
    compiled, launches = dispatch
    for elements in (0, 256) * 2:
        out = torch.empty(elements, dtype=torch.bfloat16, device="meta")
        data = torch.empty(elements, dtype=torch.uint8, device="meta")
        grouped.flydsl_grouped_gemm_a8w4_masked(
            out,
            data,
            data,
            data,
            data,
            torch.empty(64, dtype=torch.int32),
            n_experts=64,
            contiguous_m=elements,
            N=7168,
            K=2048,
            tile_m=256,
            tile_n=256,
            tile_k=256,
            m_warp=2,
            n_warp=2,
            num_buffers=4,
            cluster_m=4,
            cluster_n=4,
            persistent_workers=256,
            a_is_fp4=1,
            a_preshuffled=1,
            stream=0,
        )
        layout = launches[-1]["layout_buffers"]["arg_c"]
        assert int.from_bytes(layout, "little", signed=True) == int(elements != 0)
        assert (launches[-1]["arg_c"] or 0) == out.data_ptr()
    assert len(compiled) == 2
    assert len(launches) == 4
