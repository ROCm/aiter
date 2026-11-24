# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regression tests for grouped GEMM's host C ABI (issue #7241)."""

import ctypes
import inspect
import types

import pytest
import torch

pytest.importorskip("flydsl.compiler")

from flydsl.compiler.jit_executor import CallState
from flydsl.compiler.jit_function import _build_call_spec

from aiter.ops.flydsl import grouped_gemm_mxfp4 as grouped
from aiter.ops.flydsl import grouped_moe_gfx1250 as moe
from aiter.ops.flydsl import iod_locality
from aiter.ops.flydsl import moe_kernels
from aiter.ops.flydsl.kernels import moe_apreshuffled_quant_scatter as apre_scatter
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
    "arg_loc_phase",
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


@pytest.mark.parametrize(
    "overrides,env,expected_workers,expected_buffers",
    [
        ({}, {}, 256, 3),
        ({"persistent_workers": 128}, {}, 128, 3),
        ({"persistent_workers": 512, "num_buffers": 2}, {}, 512, 3),
        ({}, {"AITER_FLYDSL_EP_SCATTER_OVERLAP": "0"}, 0, 4),
        (
            {"persistent_workers": 512, "num_buffers": 3},
            {"AITER_FLYDSL_EP_SCATTER_OVERLAP": "0"},
            512,
            3,
        ),
        (
            {"persistent_workers": 128},
            {"AITER_FLYDSL_PERSISTENT_WORKERS2": "64"},
            64,
            3,
        ),
        ({}, {"AITER_FLYDSL_PERSISTENT_WORKERS": "128"}, 128, 3),
        ({}, {"AITER_FLYDSL_PERSISTENT_WORKERS2": "0"}, 256, 3),
        ({"a_preshuffled": 0}, {}, 0, 4),
        ({"tile_m": 128}, {}, 0, 4),
        ({"next_stage_prefetch": 0}, {}, 0, 4),
        ({}, {"AITER_TDM_NEXT_STAGE_PREFETCH": "0"}, 0, 4),
        ({"tdm_as_in_prologue": 1}, {}, 0, 4),
        ({}, {"AITER_GROUPED_GEMM_AS_PROLOGUE": "1"}, 0, 4),
        ({"K": 512}, {}, 0, 2),
        ({"stage2_scatter": None}, {}, 0, 4),
    ],
)
def test_overlap_defaults_preserve_launch_configuration(
    monkeypatch, overrides, env, expected_workers, expected_buffers
):
    """Inspect actual launch arguments without compiling or executing GPU code."""
    for name in (
        "AITER_FLYDSL_PERSISTENT_WORKERS",
        "AITER_FLYDSL_PERSISTENT_WORKERS1",
        "AITER_FLYDSL_PERSISTENT_WORKERS2",
        "AITER_FLYDSL_MXFP4_CLUSTER_M",
        "AITER_FLYDSL_MXFP4_CLUSTER_N",
        "AITER_TDM_NEXT_STAGE_PREFETCH",
        "AITER_GROUPED_GEMM_AS_PROLOGUE",
        "AITER_FLYDSL_EP_SCATTER_OVERLAP",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    calls = []
    signature = inspect.signature(launch_gemm_a8w4_tdm.func)

    def capture_launch(exe, *args, specialization_key):
        assert exe is launch_gemm_a8w4_tdm
        assert specialization_key
        calls.append(signature.bind(*args).arguments)

    monkeypatch.setattr(grouped, "_run_compiled", capture_launch)
    out = torch.empty((256, 1024), dtype=torch.bfloat16, device="meta")
    data = torch.empty((256, 1024), dtype=torch.uint8, device="meta")
    ids = torch.empty(64, dtype=torch.int32, device="meta")
    config = {
        "n_experts": 64,
        "contiguous_m": 256,
        "N": 1024,
        "K": 2048,
        "tile_m": 256,
        "tile_n": 256,
        "tile_k": 256,
        "m_warp": 2,
        "n_warp": 2,
        "a_is_fp4": 1,
        "a_preshuffled": 1,
        "num_buffers": 4,
        "next_stage_prefetch": 1,
        "cluster_m": 1,
        "cluster_n": 4,
        "stream": 0,
        "stage2_scatter": Stage2ScatterContext(123456, 0, 16384, 8192, 4, ids),
    }
    config.update(overrides)
    grouped.flydsl_grouped_gemm_a8w4_masked(out, data, data, data, data, ids, **config)
    assert len(calls) == 1
    selected = calls[0]
    assert selected["persistent_workers"] == expected_workers
    assert selected["num_buffers"] == expected_buffers
    assert selected["ep_scatter_overlap"] == int(
        env.get("AITER_FLYDSL_EP_SCATTER_OVERLAP", "1")
    )
    for name in (
        "tile_m",
        "tile_n",
        "tile_k",
        "cluster_m",
        "cluster_n",
        "m_warp",
        "n_warp",
        "a_preshuffled",
    ):
        assert selected[name] == config[name]


def test_scatter_overlap_override_separates_cached_launches(dispatch, monkeypatch):
    """The overlap ABI must not reuse an ordinary-scatter compiled handle."""
    compiled, launches = dispatch
    data = torch.empty((256, 1024), dtype=torch.uint8, device="meta")
    ids = torch.empty(64, dtype=torch.int32, device="meta")
    scatter = Stage2ScatterContext(123456, 0, 16384, 8192, 4, ids)
    for overlap in ("0", "1", "0"):
        monkeypatch.setenv("AITER_FLYDSL_EP_SCATTER_OVERLAP", overlap)
        grouped.flydsl_grouped_gemm_a8w4_masked(
            data,
            data,
            data,
            data,
            data,
            ids,
            n_experts=64,
            contiguous_m=256,
            N=1024,
            K=2048,
            tile_m=256,
            tile_n=256,
            tile_k=256,
            m_warp=2,
            n_warp=2,
            a_is_fp4=1,
            a_preshuffled=1,
            num_buffers=3,
            next_stage_prefetch=1,
            cluster_m=1,
            cluster_n=4,
            persistent_workers=512,
            stream=0,
            stage2_scatter=scatter,
        )
    assert len(compiled) == 2
    assert len(launches) == 3


@pytest.mark.parametrize(
    "cluster_m,requested,expected",
    [
        (1, 0, 1024),
        (2, 0, 1024),
        (4, 0, 1024),
        (1, 1000, 992),
        (2, 1000, 992),
        (4, 1000, 960),
        (1, 1, 16),
        (2, 1, 32),
        (4, 1, 64),
    ],
)
def test_iod_workers_count_all_cluster_peers(
    monkeypatch, cluster_m, requested, expected
):
    """M clustering must preserve total workers and eight-cluster IOD groups."""
    from aiter.jit.utils import chip_info

    for name in (
        "AITER_FLYDSL_MXFP4_CLUSTER_M",
        "AITER_FLYDSL_MXFP4_CLUSTER_N",
        "AITER_FLYDSL_PERSISTENT_WORKERS",
        "AITER_FLYDSL_PERSISTENT_WORKERS1",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AITER_MOE_LOC_PW", str(requested))
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    data = torch.empty((256, 1024), dtype=torch.uint8, device="meta")
    ids = torch.empty(64, dtype=torch.int32, device="meta")
    monkeypatch.setattr(iod_locality, "lookup", lambda *args: (ids, 1, 2))
    monkeypatch.setattr(iod_locality, "mode", lambda: 1)
    calls = []
    signature = inspect.signature(launch_gemm_a8w4_tdm.func)

    def capture_launch(exe, *args, specialization_key):
        assert exe is launch_gemm_a8w4_tdm
        calls.append(signature.bind(*args).arguments)

    monkeypatch.setattr(grouped, "_run_compiled", capture_launch)
    grouped.flydsl_grouped_gemm_a8w4_masked(
        data,
        data,
        data,
        data,
        data,
        ids,
        n_experts=64,
        contiguous_m=256,
        N=4096,
        K=7168,
        tile_m=256,
        tile_n=256,
        tile_k=256,
        m_warp=2,
        n_warp=2,
        waves_per_tensor_tdm=2,
        a_is_fp4=1,
        a_preshuffled=1,
        stage1_act=1,
        cluster_m=cluster_m,
        cluster_n=2,
        stream=0,
    )
    assert len(calls) == 1
    selected = calls[0]
    assert selected["loc_sched"] == 1
    assert selected["cluster_m"] == cluster_m
    assert selected["persistent_workers"] == expected
    grid_x = selected["persistent_workers"] // cluster_m
    assert grid_x * cluster_m == expected
    assert grid_x % (8 * selected["cluster_n"]) == 0


def test_compact_preshuffle_cache_separates_layout_and_scale_rep():
    """Transport metadata and scale grouping must not reuse a stale kernel."""
    factory = moe_kernels._get_compiled_apre_scatter
    ordinary = factory(7168, 8)
    compact = factory(7168, 8, 3840, 3840, 8, 64)
    other_rep = factory(7168, 1, 3840, 3840, 8, 64)
    other_topk = factory(7168, 8, 3840, 3840, 4, 64)
    assert len({id(ordinary), id(compact), id(other_rep), id(other_topk)}) == 4
    assert compact is factory(7168, 8, 3840, 3840, 8, 64)


def test_compact_preshuffle_lds_bank_mapping():
    """Audit the reused async scatter's payload and scale bank mappings."""
    payload_stride = apre_scatter._PAYLOAD_LDS_ROW_DWORDS
    scale_stride = apre_scatter._SCALE_LDS_ROW_DWORDS
    scale_slice = apre_scatter._SCALE_LDS_SLICE_DWORDS
    for first_lane in range(0, 512, 16):
        write_banks, read_banks = [], []
        for thread in range(first_lane, first_lane + 16):
            for word in range(4):
                write_banks.append(
                    (thread // 32 * payload_stride + thread % 32 * 4 + word) % 64
                )
                read_banks.append(
                    (thread % 16 * payload_stride + thread // 16 * 4 + word) % 64
                )
        assert len(set(write_banks)) == 64
        assert len(set(read_banks)) == 64
    for row in range(16):
        write_banks = [
            (column // 8 * scale_slice + row * scale_stride + column % 8) % 64
            for column in range(32)
        ]
        assert len(set(write_banks)) == 32
    for column in range(8):
        read_banks = [(row * scale_stride + column) % 64 for row in range(16)]
        assert len(set(read_banks)) == 16


def test_compact_preshuffle_rejects_fused_stage1(monkeypatch):
    """A fused producer cannot silently ignore the requested GEMM1 layout."""
    monkeypatch.setattr(
        moe,
        "_flydsl_dispatch_context",
        lambda: types.SimpleNamespace(compact_layout=True, compact_row_gather=False),
    )
    data = torch.empty((16, 64), device="meta")
    ids = torch.empty((16, 8), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="AITER_MEGA_S1F=0"):
        moe._grouped_a8w4_tdm_moe(
            data,
            data,
            data,
            ids,
            ids,
            E=1,
            model_dim=128,
            inter_dim=128,
            dtype=torch.bfloat16,
            activation=None,
            w1_scale=None,
            w2_scale=None,
            bias1=None,
            bias2=None,
            swiglu_limit=None,
            doweight_stage1=False,
            data_format="fp4",
            a_preshuffle=1,
        )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"wmma_rep": 0}, "positive wmma_rep"),
        ({"wmma_rep": 3}, "positive wmma_rep"),
        ({"wire_row_stride": 129}, "K128 MXFP4"),
        ({"masked_m": None}, "matching expert counts/psum"),
        ({}, "wire views on one GPU"),
    ],
)
def test_compact_preshuffle_validates_before_launch(overrides, message):
    data = torch.empty((16, 128), dtype=torch.uint8, device="cpu")
    rowmap = torch.empty((129, 2), dtype=torch.int32, device="cpu")
    counts = torch.empty(1, dtype=torch.int32, device="cpu")
    kwargs = {
        "wire_row_stride": 128,
        "topk": 8,
        "masked_m": counts,
        "psum": counts,
        "payload_preshuffled": True,
        "wmma_rep": 8,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        moe.flydsl_moe_scatter_copy_compact(
            data[:, :64], data[:, 64:68], rowmap, **kwargs
        )
