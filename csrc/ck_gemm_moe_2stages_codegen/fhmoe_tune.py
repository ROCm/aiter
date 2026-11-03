# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os
import re
import tempfile
from typing import Any, ClassVar

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fhmoe import _use_fhmoe_wrappers
from aiter.fused_moe import _fused_moe_impl, fused_moe, fused_topk, get_padded_M
from aiter.jit.core import AITER_CONFIG_FHMOE, AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.moe_kernels import (
    get_flydsl_stage1_kernels,
    get_flydsl_stage2_kernels,
)
from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.utility.base_tuner import TunerCommon
from aiter.utility.mp_tuner import mp_tuner
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
    TUNE_MOE_EXPERT_BALANCE,
    FmoeTuner,
    cosine_diff_compare,
    tensor_compare_diagnostics,
    torch_dynamic_mxfp8_quant,
)


def closest_reference_cosine_diff_compare(refs, res, msg="", printLog=True):
    # compare_fn only sees the reference and the result, so an output with more
    # than one valid reference gets them all, stacked on dim 0, and is judged
    # against the closest.
    return min(
        cosine_diff_compare(ref, res, msg=msg, printLog=printLog) for ref in refs
    )


class FhmoeTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_FHMOE}",
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/untuned_fhmoe.csv",
        # max accepted cosine_diff; 0.1 rejects broken kernels
        "errRatio": 0.1,
        "config_env_name": "AITER_CONFIG_FHMOE",
    }

    def _setup_specific_arguments(self):
        pass

    def _clear_op_caches(self):
        import aiter.fused_moe as fmoe_module

        fmoe_module.cfg_2stages = None
        fmoe_module.cfg_2stages_by_file.clear()
        fmoe_module.get_2stage_cfgs.cache_clear()

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
            return
        self.untunedf = self.get_untuned_gemm_list(args.untune_file)
        self.tunedf = self.get_tuned_gemm_list(self.get_out_file(args.tune_file))
        self.untunedf["gfx"] = self.get_gfx()
        self.untunedf["cu_num"] = self.get_cu_num()
        dedup_cols = [
            col
            for col in self.keys
            if col in self.untunedf.columns and col in self.tunedf.columns
        ]
        if len(dedup_cols) == len(self.keys):
            tuned = self.tunedf[dedup_cols].apply(tuple, axis=1)
            mask = self.untunedf[dedup_cols].apply(tuple, axis=1).isin(tuned)
            self.untunedf = self.untunedf[~mask]

    def get_untuned_gemm_list(self, untuned_gemm_file):
        df = super().get_untuned_gemm_list(untuned_gemm_file)
        # A tuned file read back for an in-place retune holds the rows of every
        # GPU and is rewritten whole, so its other targets must survive.
        if "kernelName1" in df.columns:
            return df
        live = {"gfx": self.get_gfx(), "cu_num": self.get_cu_num()}
        foreign = pd.Series(False, index=df.index)
        for col, value in live.items():
            if col not in df.columns:
                continue
            if col == "cu_num":
                declared = pd.to_numeric(df[col], errors="coerce")
            else:
                declared = df[col].astype(str).str.strip()
            foreign |= df[col].notna() & (declared != value)
        if foreign.any():
            print(
                f"[fhmoe] skipping {int(foreign.sum())} rows pinned to another "
                f"target (live GPU is {live['gfx']} with {live['cu_num']} CUs)",
                flush=True,
            )
        df = df[~foreign].reset_index(drop=True)
        for col, value in live.items():
            if col in df.columns:
                df[col] = value
        return df

    # _kernel_pairs only builds MXFP4-weight pairs; tuned_fhmoe.csv also holds
    # HY4 FP8-weight rows owned by op_tests/tuners/tune_fhmoe.py.
    _OWNED_Q_DTYPE_W = "torch.float4_e2m1fn_x2"

    @classmethod
    def _owned_rows(cls, df):
        owned = df["q_dtype_w"].astype(str) == cls._OWNED_Q_DTYPE_W
        skipped = int((~owned).sum())
        if skipped:
            print(
                f"[fhmoe] skipping {skipped} non-MXFP4-weight rows "
                "(tuned by op_tests/tuners/tune_fhmoe.py)",
                flush=True,
            )
        return df[owned].reset_index(drop=True)

    @staticmethod
    def _parse_fhmoe_row(row):
        return {
            "gfx": str(row["gfx"]),
            "cu_num": int(row["cu_num"]),
            "token": int(row["token"]),
            "model_dim": int(row["model_dim"]),
            "inter_dim": int(row["inter_dim"]),
            "expert": int(row["expert"]),
            "topk": int(row["topk"]),
            "shared_expert_id": int(row["shared_expert_id"]),
            "act_type": eval(row["act_type"]),
            "dtype": eval(row["dtype"]),
            "q_dtype_a": eval(row["q_dtype_a"]),
            "q_dtype_w": eval(row["q_dtype_w"]),
            "q_type": eval(row["q_type"]),
            "use_g1u1": int(row["use_g1u1"]),
            "doweight_stage1": int(row["doweight_stage1"]),
            "hidden_pad": int(row["hidden_pad"]),
            "intermediate_pad": int(row["intermediate_pad"]),
            "gate_mode": eval(row["gate_mode"]),
        }

    @staticmethod
    def _pad_dummy_expert(tensor, n_experts=None):
        if tensor.ndim == 2:
            rows_per_expert = tensor.shape[0] // n_experts
            dummy_shape = (rows_per_expert, tensor.shape[1])
        else:
            dummy_shape = (1, *tensor.shape[1:])
        dummy = tensor.new_empty(dummy_shape)
        dummy.view(torch.uint8).zero_()
        return torch.cat([tensor, dummy], dim=0)

    @staticmethod
    def generate_fhmoe_data(shape, block_m=32, device="cuda"):
        expert = shape["expert"]
        topk = shape["topk"]
        shared_expert_id = shape["shared_expert_id"]
        if topk < 2:
            raise ValueError(f"FHMoE topk must include the shared expert; got {topk=}")

        routed_expert = expert - 1
        routed_topk = topk - 1
        token = shape["token"]
        model_dim = shape["model_dim"]
        inter_dim = shape["inter_dim"]
        dtype = shape["dtype"]
        q_type = shape["q_type"]
        q_dtype_w = shape["q_dtype_w"]
        torch.manual_seed(0)
        hidden = torch.randn((token, model_dim), dtype=dtype, device="cpu") / 10
        if shape["use_g1u1"]:
            w1_dense = (
                torch.randn(
                    (routed_expert, inter_dim * 2, model_dim),
                    dtype=dtype,
                    device="cpu",
                )
                / 10
            )
        else:
            w1_dense = (
                torch.randn(
                    (routed_expert, inter_dim, model_dim),
                    dtype=dtype,
                    device="cpu",
                )
                / 10
            )
        w2_dense = torch.randn(
            (routed_expert, model_dim, inter_dim), dtype=dtype, device="cpu"
        )
        w1_shape = w1_dense.shape
        w2_shape = w2_dense.shape
        with torch.device("cpu"):
            w1, w1_scale = FmoeTuner.weight_quant(
                w1_dense, q_type, quant_dtype=q_dtype_w
            )
            w2, w2_scale = FmoeTuner.weight_quant(
                w2_dense, q_type, quant_dtype=q_dtype_w
            )
        del w1_dense, w2_dense
        if q_dtype_w is not dtypes.fp4x2:
            w1 = w1.view(w1_shape)
            w2 = w2.view(w2_shape)
        else:
            w1 = w1.view(w1_shape[0], w1_shape[1], w1_shape[2] // 2)
            w2 = w2.view(w2_shape[0], w2_shape[1], w2_shape[2] // 2)
        if TUNE_MOE_EXPERT_BALANCE:
            score = torch.zeros((token, routed_expert), dtype=dtype, device="cpu")
            start_col = 0
            end_col = routed_topk
            for token_id in range(token):
                score[token_id, start_col:end_col] = 1.0
                start_col = end_col % routed_expert
                end_col = start_col + routed_topk
        else:
            routing_seed = os.environ.get("TUNE_MOE_ROUTING_SEED")
            if routing_seed is not None:
                torch.manual_seed(int(routing_seed))
            score = torch.randn((token, routed_expert), dtype=dtype, device="cpu")
        shared_w1_dense = (
            torch.randn((1, 2 * inter_dim, model_dim), dtype=dtype, device="cpu") / 10
        )
        shared_w2_dense = (
            torch.randn((1, model_dim, inter_dim), dtype=dtype, device="cpu") / 10
        )
        # Shared expert is FP8 in the FHMoE contract (aiter/fhmoe.py), not a
        # column of the CSV. Routed weights use shape["q_dtype_w"] above.
        with torch.device("cpu"):
            shared_w1, shared_w1_scale = FmoeTuner.weight_quant(
                shared_w1_dense, q_type, quant_dtype=dtypes.fp8
            )
            shared_w2, shared_w2_scale = FmoeTuner.weight_quant(
                shared_w2_dense, q_type, quant_dtype=dtypes.fp8
            )
        del shared_w1_dense, shared_w2_dense
        topk_weights, topk_ids = fused_topk(
            hidden.to(device), score.to(device), routed_topk, True
        )
        del score
        topk_weights = topk_weights.cpu()
        topk_ids = topk_ids.cpu()

        # Unshuffled copies for the torch reference. Shuffle below is kernel ABI.
        w1_qt, w2_qt = w1, w2
        w1_scale_qt, w2_scale_qt = w1_scale, w2_scale
        shared_w1_qt, shared_w2_qt = shared_w1, shared_w2
        shared_w1_scale_qt, shared_w2_scale_qt = shared_w1_scale, shared_w2_scale

        w1 = FhmoeTuner._pad_dummy_expert(w1)
        w2 = FhmoeTuner._pad_dummy_expert(w2)
        w1_scale = FhmoeTuner._pad_dummy_expert(w1_scale, routed_expert)
        w2_scale = FhmoeTuner._pad_dummy_expert(w2_scale, routed_expert)

        shared_ids = torch.full(
            (token, 1),
            shared_expert_id,
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        shared_weights = torch.ones(
            (token, 1),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        topk_ids = torch.cat([topk_ids, shared_ids], dim=1)
        topk_weights = torch.cat([topk_weights, shared_weights], dim=1)

        if shape["gate_mode"] == GateMode.INTERLEAVE:
            w1 = shuffle_weight_a16w4(w1, 16, True)
            w1_scale = shuffle_scale_a16w4(w1_scale, expert, True)
            w2 = shuffle_weight_a16w4(w2, 16, False)
            w2_scale = shuffle_scale_a16w4(w2_scale, expert, False)
            shared_w1 = shuffle_weight_a16w4(shared_w1, 16, True)
            shared_w1_scale = shuffle_scale_a16w4(shared_w1_scale, 1, True)
            shared_w2 = shuffle_weight_a16w4(shared_w2, 16, False)
            shared_w2_scale = shuffle_scale_a16w4(shared_w2_scale, 1, False)
        else:
            w1 = shuffle_weight(w1, layout=(16, 16))
            w1_scale = shuffle_scale(w1_scale)
            w2 = shuffle_weight(w2, layout=(16, 16))
            w2_scale = shuffle_scale(w2_scale)
            shared_w1 = shuffle_weight(shared_w1, layout=(16, 16))
            shared_w1_scale = shuffle_scale(shared_w1_scale)
            shared_w2 = shuffle_weight(shared_w2, layout=(16, 16))
            shared_w2_scale = shuffle_scale(shared_w2_scale)
        w1.is_shuffled = True
        w2.is_shuffled = True

        return {
            "hidden": hidden,
            "w1": w1,
            "w2": w2,
            "w1_scale": w1_scale,
            "w2_scale": w2_scale,
            "w1_qt": w1_qt,
            "w2_qt": w2_qt,
            "w1_scale_qt": w1_scale_qt,
            "w2_scale_qt": w2_scale_qt,
            "shared_w1": shared_w1,
            "shared_w2": shared_w2,
            "shared_w1_scale": shared_w1_scale,
            "shared_w2_scale": shared_w2_scale,
            "shared_w1_qt": shared_w1_qt,
            "shared_w2_qt": shared_w2_qt,
            "shared_w1_scale_qt": shared_w1_scale_qt,
            "shared_w2_scale_qt": shared_w2_scale_qt,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "a1_qt": hidden,
            "a1_scale": None,
            "gate_mode": shape["gate_mode"],
            "hidden_pad": shape["hidden_pad"],
            "intermediate_pad": shape["intermediate_pad"],
            "shared_expert_id": shared_expert_id,
        }

    BLOCK_MS: ClassVar[tuple[int, ...]] = (16, 32, 64, 128)

    @staticmethod
    def _csv_cell(value):
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, GateMode):
            return f"GateMode.{value.name}"
        if isinstance(value, ActivationType):
            return f"ActivationType.{value.name}"
        if isinstance(value, QuantType):
            return f"QuantType.{value.name}"
        return value

    @staticmethod
    def _kernel_pairs(shape):
        model_dim = shape["model_dim"]
        inter_dim = shape["inter_dim"]
        # Same map as FmoeTuner.gen_flydsl_2stages_task.
        a_dtype_map = {
            dtypes.fp8: "fp8",
            dtypes.fp4x2: "fp4",
            dtypes.fp16: "fp16",
            dtypes.bf16: "bf16",
        }
        a_dtype = a_dtype_map.get(shape["q_dtype_a"], "fp8")
        b_dtype = "fp8" if shape["q_dtype_w"] == dtypes.fp8 else "fp4"
        out_dtype = "bf16" if shape["dtype"] == dtypes.bf16 else "f16"
        s1_all = get_flydsl_stage1_kernels(a_dtype, b_dtype, out_dtype)
        s2_all = get_flydsl_stage2_kernels(a_dtype, b_dtype, out_dtype)
        pairs = []
        for block_m in FhmoeTuner.BLOCK_MS:
            s1_names = []
            for kname, kparams in s1_all.items():
                if kparams.get("tile_m") != block_m:
                    continue
                k_batch = kparams.get("k_batch", 1)
                k_wave = kparams.get("k_wave", 1)
                tile_k = kparams["tile_k"]
                # Same stage1 k_batch cut as FmoeTuner.gen_flydsl_2stages_task:
                # drop splits that do not divide K.
                if model_dim % k_batch != 0:
                    continue
                k_per_batch = model_dim // k_batch
                if k_per_batch % tile_k != 0:
                    continue
                if k_wave > 1 and (
                    k_per_batch % k_wave != 0 or (k_per_batch // k_wave) % tile_k != 0
                ):
                    continue
                # Trailing _fp8/_fp4: fused_moe fuses inter-stage quant into
                # stage1; bare kname: quant after. Same as FmoeTuner.
                fused = kname + ("_fp8" if a_dtype == "fp8" else "_fp4")
                if k_batch > 1:
                    s1_names.append(fused)
                else:
                    s1_names.extend((kname, fused))
            s2_names = []
            for kname, kparams in s2_all.items():
                if kparams.get("tile_m") != block_m:
                    continue
                # Same stage2 cuts as FmoeTuner.gen_flydsl_2stages_task:
                #  - tile_k must divide inter_dim (K); tile_k=256 on non-256 inter
                #    (e.g. 384) is parsed verbatim by the wrapper -> OOB/wrong out;
                #  - LDS is tile_m*(tile_k*2 + tile_n*4); over 160 KiB (gfx950
                #    workgroup LDS) the build fails.
                if inter_dim % kparams["tile_k"] != 0:
                    continue
                lds = block_m * (kparams["tile_k"] * 2 + kparams["tile_n"] * 4)
                if lds > 160 * 1024:
                    continue
                s2_names.append(kname)
            for kn1 in s1_names:
                for kn2 in s2_names:
                    pairs.append((block_m, kn1, kn2))
        kernel_regex = os.environ.get("TUNE_MOE_KERNEL_REGEX")
        if kernel_regex:
            pattern = re.compile(kernel_regex)
            pairs = [p for p in pairs if pattern.search(f"{p[1]} {p[2]}")]
        return pairs

    @staticmethod
    def _lookup_token(token):
        # Serving get_2stage_cfgs looks up get_padded_M(token_num), not the
        # catalogue token. Time and publish under that same key.
        return get_padded_M(int(token))

    @staticmethod
    def _write_candidate_csv(path, shape, block_m, kn1, kn2):
        gate_mode = shape["gate_mode"]
        row = {
            "gfx": shape["gfx"],
            "cu_num": shape["cu_num"],
            "token": FhmoeTuner._lookup_token(shape["token"]),
            "model_dim": shape["model_dim"],
            "inter_dim": shape["inter_dim"],
            "expert": shape["expert"],
            "topk": shape["topk"],
            "act_type": FhmoeTuner._csv_cell(shape["act_type"]),
            "dtype": FhmoeTuner._csv_cell(shape["dtype"]),
            "q_dtype_a": FhmoeTuner._csv_cell(shape["q_dtype_a"]),
            "q_dtype_w": FhmoeTuner._csv_cell(shape["q_dtype_w"]),
            "q_type": FhmoeTuner._csv_cell(shape["q_type"]),
            "use_g1u1": int(shape["use_g1u1"]),
            "doweight_stage1": int(shape["doweight_stage1"]),
            "shared_expert_id": shape["shared_expert_id"],
            "hidden_pad": shape["hidden_pad"],
            "intermediate_pad": shape["intermediate_pad"],
            "gate_mode": FhmoeTuner._csv_cell(gate_mode),
            "block_m": block_m,
            "ksplit": 0,
            "kernelName1": kn1,
            "kernelName2": kn2,
        }
        pd.DataFrame([row]).to_csv(path, index=False)

    @staticmethod
    def _run_fhmoe(shape, data, kn1, kn2, block_m):
        gate_mode = shape["gate_mode"]
        if not isinstance(gate_mode, str):
            gate_mode = gate_mode.value
        q_dtype_a = shape["q_dtype_a"]
        to_dev = FhmoeTuner._to_ref_device

        w1 = to_dev(data["w1"])
        w2 = to_dev(data["w2"])
        w1.is_shuffled = True
        w2.is_shuffled = True
        import aiter.fused_moe as fmoe_mod

        cfg_fd, cfg_path = tempfile.mkstemp(suffix="_fhmoe_tune.csv")
        os.close(cfg_fd)
        try:
            FhmoeTuner._write_candidate_csv(cfg_path, shape, block_m, kn1, kn2)
            return _fused_moe_impl(
                hidden_states=to_dev(data["hidden"]),
                w1=w1,
                w2=w2,
                topk_weight=to_dev(data["topk_weights"]),
                topk_ids=to_dev(data["topk_ids"]),
                activation=shape["act_type"].value,
                quant_type=shape["q_type"].value,
                doweight_stage1=bool(shape["doweight_stage1"]),
                w1_scale=to_dev(data["w1_scale"]),
                w2_scale=to_dev(data["w2_scale"]),
                block_size_M=block_m,
                dtype=shape["dtype"],
                hidden_pad=shape["hidden_pad"],
                intermediate_pad=shape["intermediate_pad"],
                gate_mode=gate_mode,
                _q_dtype_a=q_dtype_a,
                _metadata_transform=_use_fhmoe_wrappers,
                _metadata_config_file=cfg_path,
                _stage1_extra_args={
                    "shared_w1": to_dev(data["shared_w1"]),
                    "shared_w1_scale": to_dev(data["shared_w1_scale"]),
                    "shared_expert_id": data["shared_expert_id"],
                },
                _stage2_extra_args={
                    "shared_w2": to_dev(data["shared_w2"]),
                    "shared_w2_scale": to_dev(data["shared_w2_scale"]),
                    "shared_expert_id": data["shared_expert_id"],
                },
            )
        finally:
            fmoe_mod.cfg_2stages_by_file.pop(cfg_path, None)
            os.unlink(cfg_path)

    _FHMOE_MP_DATA_KEYS = (
        "hidden",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
        "w1_scale",
        "w2_scale",
        "shared_w1",
        "shared_w2",
        "shared_w1_scale",
        "shared_w2_scale",
    )
    _FHMOE_REF_DATA_KEYS = (
        "hidden",
        "w1_qt",
        "w2_qt",
        "w1_scale_qt",
        "w2_scale_qt",
        "shared_w1_qt",
        "shared_w2_qt",
        "shared_w1_scale_qt",
        "shared_w2_scale_qt",
        "topk_weights",
        "topk_ids",
    )

    @staticmethod
    def _run_fhmoe_mp(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
        shape,
        kn1,
        kn2,
        block_m,
    ):
        return FhmoeTuner._run_fhmoe(
            shape,
            {
                "hidden": hidden,
                "w1": w1,
                "w2": w2,
                "topk_weights": topk_weights,
                "topk_ids": topk_ids,
                "w1_scale": w1_scale,
                "w2_scale": w2_scale,
                "shared_w1": shared_w1,
                "shared_w2": shared_w2,
                "shared_w1_scale": shared_w1_scale,
                "shared_w2_scale": shared_w2_scale,
                "shared_expert_id": shape["shared_expert_id"],
            },
            kn1,
            kn2,
            block_m,
        )

    @staticmethod
    def _to_ref_device(tensor):
        if tensor is None or not torch.is_tensor(tensor):
            return tensor
        if tensor.is_cuda:
            return tensor
        if torch.cuda.is_available():
            return tensor.to(torch.device("cuda"))
        return tensor

    @staticmethod
    def run_torch_fhmoe(
        hidden,
        w1_qt,
        w2_qt,
        w1_scale_qt,
        w2_scale_qt,
        shared_w1_qt,
        shared_w2_qt,
        shared_w1_scale_qt,
        shared_w2_scale_qt,
        topk_weights,
        topk_ids,
        shape,
        quantize_fp32=False,
    ):
        """Homogeneous torch refs for routed experts and shared FP8, then summed.

        Matches the tuner launch: same tensors, no swiglu_limit (the candidate
        path does not pass one). Rows whose q_dtype_a is MXFP8 quantize the
        hidden states and the stage-1 result to MXFP8, so the reference does
        too. A fused-quant stage 1 (kernelName1 ending in _fp8) quantizes its
        fp32 result; the unfused path quantizes it after rounding to dtype.
        Other activation dtypes are not modeled; they keep bf16.
        """
        hidden = FhmoeTuner._to_ref_device(hidden)
        w1_qt = FhmoeTuner._to_ref_device(w1_qt)
        w2_qt = FhmoeTuner._to_ref_device(w2_qt)
        w1_scale_qt = FhmoeTuner._to_ref_device(w1_scale_qt)
        w2_scale_qt = FhmoeTuner._to_ref_device(w2_scale_qt)
        shared_w1_qt = FhmoeTuner._to_ref_device(shared_w1_qt)
        shared_w2_qt = FhmoeTuner._to_ref_device(shared_w2_qt)
        shared_w1_scale_qt = FhmoeTuner._to_ref_device(shared_w1_scale_qt)
        shared_w2_scale_qt = FhmoeTuner._to_ref_device(shared_w2_scale_qt)
        topk_weights = FhmoeTuner._to_ref_device(topk_weights)
        topk_ids = FhmoeTuner._to_ref_device(topk_ids)

        routed_ids = topk_ids[:, :-1]
        routed_w = topk_weights[:, :-1]
        shared_ids = torch.zeros(
            (topk_ids.shape[0], 1),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        shared_w = topk_weights[:, -1:]
        dtype = shape["dtype"]
        activation = shape["act_type"]
        quant_type = shape["q_type"]
        doweight_stage1 = bool(shape["doweight_stage1"])
        quantize_mxfp8 = shape["q_dtype_a"] == dtypes.fp8
        stage1_dtype = dtypes.fp32 if quantize_mxfp8 and quantize_fp32 else dtype

        a1_scale = None
        if quantize_mxfp8:
            a1_bytes, a1_scale = torch_dynamic_mxfp8_quant(hidden)
            hidden = a1_bytes.view(dtypes.fp8)

        def quantize_a2(ref1):
            if not quantize_mxfp8:
                return ref1, None
            a2_bytes, a2_scale = torch_dynamic_mxfp8_quant(ref1)
            return a2_bytes.view(dtypes.fp8), a2_scale

        ref1 = FmoeTuner.run_torch_moe_stage1(
            hidden,
            w1_qt,
            w2_qt,
            routed_w,
            routed_ids,
            a1_scale,
            w1_scale_qt,
            dtype=stage1_dtype,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            topk=routed_ids.shape[1],
        )
        a2, a2_scale = quantize_a2(ref1)
        routed = FmoeTuner.run_torch_moe_stage2(
            a2,
            w1_qt,
            w2_qt,
            routed_w,
            routed_ids,
            a2_scale,
            w2_scale_qt,
            dtype=dtype,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
        )
        shared_ref1 = FmoeTuner.run_torch_moe_stage1(
            hidden,
            shared_w1_qt,
            shared_w2_qt,
            shared_w,
            shared_ids,
            a1_scale,
            shared_w1_scale_qt,
            dtype=stage1_dtype,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            topk=1,
        )
        shared_a2, shared_a2_scale = quantize_a2(shared_ref1)
        shared = FmoeTuner.run_torch_moe_stage2(
            shared_a2,
            shared_w1_qt,
            shared_w2_qt,
            shared_w,
            shared_ids,
            shared_a2_scale,
            shared_w2_scale_qt,
            dtype=dtype,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
        )
        return routed + shared

    @staticmethod
    def _warn_unquantized_reference(shape, label):
        if shape["q_dtype_a"] != dtypes.fp8:
            aiter.logger.warning(
                f"FHMoE {label}: the reference keeps fp4 activations in bf16, so "
                "outputs are checked against an unquantized reference"
            )

    @staticmethod
    def run_torch_fhmoe_refs(*args):
        """run_torch_fhmoe stacked for closest_reference_cosine_diff_compare: the
        stage-1 result quantized after rounding to dtype, then, for MXFP8
        activations, quantized from fp32 as a fused-quant stage 1 does."""
        shape = args[-1]
        refs = [FhmoeTuner.run_torch_fhmoe(*args)]
        if shape["q_dtype_a"] == dtypes.fp8:
            refs.append(FhmoeTuner.run_torch_fhmoe(*args, quantize_fp32=True))
        return torch.stack(refs)

    def _fhmoe_tune_tasks(self, untunedf, args):
        measure_kwargs = {
            "num_warmup": int(args.warmup),
            "num_iters": int(args.iters),
        }
        tasks = []
        in_datas = []
        for _, row in untunedf.iterrows():
            shape = self._parse_fhmoe_row(row)
            padded = self._lookup_token(shape["token"])
            if padded != shape["token"]:
                raise ValueError(
                    f"FHMoE token {shape['token']} is not a serving bucket "
                    f"(get_padded_M -> {padded}); use power-of-two tokens"
                )
            info = tuple(shape[k] for k in self.keys)
            pairs = self._kernel_pairs(shape)
            print(
                f"\nStart tuning FHMoE {info} candidates={len(pairs)}",
                flush=True,
            )
            if not pairs:
                continue
            self._warn_unquantized_reference(shape, info)
            for block_m, kn1, kn2 in pairs:
                tasks.append(
                    (
                        (info, kn1, kn2, block_m),
                        FhmoeTuner.generate_fhmoe_data,
                        (shape,),
                        FhmoeTuner._run_fhmoe_mp,
                        (list(self._FHMOE_MP_DATA_KEYS), shape, kn1, kn2, block_m),
                        measure_kwargs,
                        FhmoeTuner.run_torch_fhmoe_refs,
                        (list(self._FHMOE_REF_DATA_KEYS), shape),
                        {},
                        None,
                        0.01,
                        0.01,
                        closest_reference_cosine_diff_compare,
                    )
                )
            in_datas.append((len(pairs), ()))
        return tasks, in_datas

    def run_config(self, args):
        """Public fused_moe + shared FP8, not FmoeTuner's homogeneous path."""
        from aiter.test_common import run_perftest

        results = []
        for _, row in self._owned_rows(self.untunedf).iterrows():
            shape = self._parse_fhmoe_row(row)
            shape_str = (
                f"(token={shape['token']}, dim={shape['model_dim']}, "
                f"inter={shape['inter_dim']}, E={shape['expert']}, "
                f"topk={shape['topk']}, shared={shape['shared_expert_id']}, "
                f"{self._csv_cell(shape['act_type'])}, "
                f"{self._csv_cell(shape['dtype'])}, "
                f"{self._csv_cell(shape['q_dtype_a'])}, "
                f"{self._csv_cell(shape['q_dtype_w'])}, "
                f"{self._csv_cell(shape['q_type'])}, "
                f"{self._csv_cell(shape['gate_mode'])})"
            )
            self._warn_unquantized_reference(shape, f"run_config {shape_str}")
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            kernel_us = None
            if "us" in row and pd.notna(row["us"]):
                try:
                    kernel_us = float(row["us"])
                except (TypeError, ValueError):
                    kernel_us = None
            try:
                data = self.generate_fhmoe_data(shape)
                w1 = self._to_ref_device(data["w1"])
                w2 = self._to_ref_device(data["w2"])
                w1.is_shuffled = True
                w2.is_shuffled = True
                gate_mode = shape["gate_mode"]
                if not isinstance(gate_mode, str):
                    gate_mode = gate_mode.value
                out, us = run_perftest(
                    fused_moe,
                    self._to_ref_device(data["hidden"]),
                    w1,
                    w2,
                    self._to_ref_device(data["topk_weights"]),
                    self._to_ref_device(data["topk_ids"]),
                    activation=shape["act_type"],
                    quant_type=shape["q_type"],
                    doweight_stage1=bool(shape["doweight_stage1"]),
                    w1_scale=self._to_ref_device(data["w1_scale"]),
                    w2_scale=self._to_ref_device(data["w2_scale"]),
                    dtype=shape["dtype"],
                    hidden_pad=shape["hidden_pad"],
                    intermediate_pad=shape["intermediate_pad"],
                    gate_mode=gate_mode,
                    shared_w1=self._to_ref_device(data["shared_w1"]),
                    shared_w2=self._to_ref_device(data["shared_w2"]),
                    shared_w1_scale=self._to_ref_device(data["shared_w1_scale"]),
                    shared_w2_scale=self._to_ref_device(data["shared_w2_scale"]),
                    shared_expert_id=data["shared_expert_id"],
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                reference = functools.partial(
                    self.run_torch_fhmoe,
                    *[data[k] for k in self._FHMOE_REF_DATA_KEYS],
                    shape,
                )
                ref = reference()
                if out.count_nonzero() == 0 and ref.count_nonzero() > 0:
                    diag = tensor_compare_diagnostics(ref, out)
                    status = (
                        "error:output is all zeros (kernel produced no output); "
                        f"{diag}"
                    )
                else:
                    err_ratio = cosine_diff_compare(ref, out, printLog=False)
                    if (
                        err_ratio > allowed_err_ratio
                        and shape["q_dtype_a"] == dtypes.fp8
                    ):
                        # A fused-quant stage 1 quantizes its fp32 result where the
                        # reference above rounds it to dtype first.
                        ref_fp32 = reference(quantize_fp32=True)
                        err_fp32 = cosine_diff_compare(ref_fp32, out, printLog=False)
                        if err_fp32 < err_ratio:
                            ref, err_ratio = ref_fp32, err_fp32
                    aiter.logger.info(
                        f"run_config {shape_str} cosine_diff={err_ratio:.6g}"
                    )
                    if err_ratio <= allowed_err_ratio:
                        status = "ok"
                    else:
                        diag = tensor_compare_diagnostics(ref, out)
                        status = (
                            f"mismatch:cosine_diff={err_ratio:.6g}"
                            f"(>{allowed_err_ratio_desc}); {diag}"
                        )
                results.append(
                    {
                        "shape": shape_str,
                        "e2e_us": us,
                        "kernel_us": kernel_us,
                        "status": status,
                    }
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    {
                        "shape": shape_str,
                        "e2e_us": -1,
                        "kernel_us": kernel_us,
                        "status": f"error:{e}",
                    }
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    def tune(self, untunedf, tunedf, args):
        del tunedf
        tasks, in_datas = self._fhmoe_tune_tasks(self._owned_rows(untunedf), args)
        if not tasks:
            return []
        return mp_tuner(
            tasks,
            in_datas,
            args.mp,
            True,
            True,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def post_process(self, results, args, topk=-1, fast_mode=False):
        del topk, fast_mode
        from collections import defaultdict

        grouped = defaultdict(list)
        profile_rows = []
        for info, us, err in results:
            key = tuple(info[0])
            kn1, kn2, block_m = info[1], info[2], info[3]
            grouped[key].append((kn1, kn2, block_m, us, err))
            profile_row = {
                **dict(zip(self.keys, key)),
                "block_m": block_m,
                "ksplit": 0,
                "kernelName1": kn1,
                "kernelName2": kn2,
                "us": us,
                "err": err,
            }
            profile_row["token"] = self._lookup_token(profile_row["token"])
            profile_rows.append(profile_row)
        if args.profile_file:
            profile_df = pd.DataFrame(profile_rows)
            for col in (
                "act_type",
                "dtype",
                "q_dtype_a",
                "q_dtype_w",
                "q_type",
                "gate_mode",
            ):
                if col in profile_df.columns:
                    profile_df[col] = profile_df[col].map(self._csv_cell)
            if os.path.exists(args.profile_file):
                profile_df = pd.concat(
                    [pd.read_csv(args.profile_file), profile_df], ignore_index=True
                )
            profile_df.to_csv(args.profile_file, index=False)
        bests = []
        for key, rets in grouped.items():
            valid = [
                r
                for r in rets
                if r[3] not in (self.INVALID_TIME, self.INF_TIME)
                and r[3] >= 0
                and r[4] <= args.errRatio
            ]
            if not valid:
                print(
                    f"Tuning result for {key} is none, please check errRatio={args.errRatio}",
                    flush=True,
                )
                continue
            kn1, kn2, block_m, us, err = min(valid, key=lambda r: r[3])
            row = dict(zip(self.keys, key))
            row["token"] = self._lookup_token(row["token"])
            for col in (
                "act_type",
                "dtype",
                "q_dtype_a",
                "q_dtype_w",
                "q_type",
                "gate_mode",
            ):
                row[col] = self._csv_cell(row[col])
            row.update(
                {
                    "block_m": block_m,
                    "ksplit": 0,
                    "kernelName1": kn1,
                    "kernelName2": kn2,
                    "us": us,
                }
            )
            print(
                f"Tuning result for {key} is {block_m, kn1, kn2} {us} us err={err}",
                flush=True,
            )
            bests.append(row)
        # us selects valid rows in result_to_csv; update_tunedf keeps only
        # self.columns, so it is not written to tuned_fhmoe.csv.
        columns = self.columns + ["us"]
        if not bests:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(bests, columns=columns)

    def result_to_csv(self, results, file, concat=False):
        del concat
        old_tunedf = self.get_tuned_gemm_list(file, self.columns)
        for col in self.columns:
            if col not in old_tunedf.columns:
                old_tunedf[col] = ""
        valid = results[
            (results["us"] != self.INVALID_TIME) & (results["us"] != self.INF_TIME)
        ]
        invalid = results[
            (results["us"] == self.INVALID_TIME) | (results["us"] == self.INF_TIME)
        ]
        resultdf = self.update_tunedf(old_tunedf, valid)
        self.success = pd.concat([self.success, valid], ignore_index=True)
        self.failed = pd.concat([self.failed, invalid], ignore_index=True)
        resultdf = resultdf.astype(str).drop_duplicates(subset=self.keys, keep="last")
        resultdf.to_csv(file, index=False)


if __name__ == "__main__":
    key = [
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
        "shared_expert_id",
        "hidden_pad",
        "intermediate_pad",
        "gate_mode",
    ]
    resultList = [
        "block_m",
        "ksplit",
        "kernelName1",
        "kernelName2",
    ]
    if get_gfx() != "gfx950":
        raise SystemExit("the FHMoE tuner is only supported on gfx950")
    tuner = FhmoeTuner("fhmoeTuner", key, resultList, "fhmoe tuner")
    args = tuner.parse_args()
    if args.e2e_tune:
        tuner.parser.error("--e2e_tune is not supported by the FHMoE tuner")
    tuner.run(args, False)
