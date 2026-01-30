#!/usr/bin/env python3

import pandas as pd
import subprocess
import sys
import os
import re
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
import time

# Import aiter and torch
import torch
import aiter
from aiter import QuantType, ActivationType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting


class FlyDSLBatchTuner:
    def __init__(
        self,
        untuned_csv: str,
        candidate_csv: str,
        output_csv: str,
        cu_num: int = 256,
    ):
        self.untuned_csv = untuned_csv
        self.candidate_csv = candidate_csv
        self.output_csv = output_csv
        self.cu_num = cu_num

        self.untuned_configs = pd.read_csv(untuned_csv)
        print(f"✅ Loaded {len(self.untuned_configs)} input configurations")

        # Read candidate tile configurations
        self.candidate_tiles = pd.read_csv(candidate_csv)
        print(f"✅ Loaded {len(self.candidate_tiles)} candidate tile configurations")

        self.all_results = []

        # Set torch default device
        torch.set_default_device("cuda")

    @staticmethod
    def weight_quant(weight, q_type, quant_dtype):
        """Weight quantization"""
        E, dim1, dim2 = weight.shape
        if q_type == QuantType.per_1x128:
            weight_qt = (
                weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(E, -1, 128 * 128)
            )
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_qt, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(E, -1)
            weight_qt = (
                weight_qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(E, dim1, dim2)
            )
        else:
            torch_quant = aiter.get_torch_quant(q_type)
            weight_qt, weight_scale = torch_quant(weight, quant_dtype=quant_dtype)
        return weight_qt, weight_scale

    @staticmethod
    def generate_test_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        block_m,
    ):
        """Generate test data - reference tune.py's generate_data"""
        torch.manual_seed(0)

        # Generate input and weights
        input_data = torch.randn((token, model_dim), dtype=dtype) / 10

        if use_g1u1:
            w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
        else:
            w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype) / 10

        w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)

        # Weight quantization
        w1_qt, w1_scale = FlyDSLBatchTuner.weight_quant(
            w1, q_type, quant_dtype=q_dtype_w
        )
        w2_qt, w2_scale = FlyDSLBatchTuner.weight_quant(
            w2, q_type, quant_dtype=q_dtype_w
        )

        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)

        # Generate score and topk
        score = torch.randn((token, expert), dtype=dtype)
        topk_weights, topk_ids = fused_topk(input_data, score, topk, True)

        # Input quantization
        if q_type == QuantType.per_1x128:
            a1_qt, a1_scale = aiter.pertoken_quant(
                input_data.view(token, -1, 128), quant_dtype=q_dtype_a
            )
            a1_qt = a1_qt.view(token, model_dim)
            a1_scale = a1_scale.squeeze(-1)
        else:
            torch_quant = aiter.get_torch_quant(q_type)
            a1_qt, a1_scale = torch_quant(input_data, quant_dtype=q_dtype_a)

        # moe_sorting
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
            moe_sorting(topk_ids, topk_weights, expert, model_dim, dtype, block_m)
        )

        return {
            "input": input_data,
            "a1_qt": a1_qt,
            "w1_qt": w1_qt,
            "w2_qt": w2_qt,
            "sorted_ids": sorted_ids,
            "sorted_weights": sorted_weights,
            "sorted_expert_ids": sorted_expert_ids,
            "num_valid_ids": num_valid_ids,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "moe_buf": moe_buf,
            "a1_scale": a1_scale,
            "w1_scale": w1_scale,
            "w2_scale": w2_scale,
        }

    def convert_value(self, key, value):
        """Convert value format to match lookup key"""
        # Handle act_type
        if key == "act_type":
            if isinstance(value, str):
                if "ActivationType." in value:
                    return value
                return f"ActivationType.{value.capitalize()}"
            return str(value)

        # Handle dtype
        if key == "dtype":
            dtype_map = {
                "bf16": "torch.bfloat16",
                "fp16": "torch.float16",
                "fp32": "torch.float32",
            }
            if isinstance(value, str):
                if "torch." in value:
                    return value
                return dtype_map.get(value, value)
            return str(value)

        # Handle q_dtype_a and q_dtype_w
        if key in ["q_dtype_a", "q_dtype_w"]:
            # Use architecture-specific fp8 type from dtypes module
            # gfx942 (MI300) uses e4m3fnuz, gfx950 uses e4m3fn
            qdtype_map = {
                "fp8": str(dtypes.fp8),  # Auto-select based on GPU architecture
                "int8": "torch.int8",
            }
            if isinstance(value, str):
                if "torch." in value:
                    return value
                return qdtype_map.get(value, value)
            return str(value)

        # Handle q_type
        if key == "q_type":
            qtype_map = {
                "per_token": "QuantType.per_Token",
                "per_1x128": "QuantType.per_1x128",
            }
            if isinstance(value, str):
                if "QuantType." in value:
                    return value
                return qtype_map.get(value, value)
            return str(value)

        # Handle use_g1u1 and doweight_stage1
        if key in ["use_g1u1", "doweight_stage1"]:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(int(value))
            if isinstance(value, str):
                return value.lower() in ["true", "1"]
            return bool(value)

        return value

    def create_tuned_config(self, untuned_row, tile_row):
        """Merge untuned config and tile config to generate complete tuned config"""
        config = {}

        # Get input parameters from untuned_row
        config["cu_num"] = self.cu_num
        config["token"] = int(untuned_row["token"])
        config["model_dim"] = int(untuned_row["model_dim"])
        config["inter_dim"] = int(untuned_row["inter_dim"])
        config["expert"] = int(untuned_row["expert"])
        config["topk"] = int(untuned_row["topk"])
        config["act_type"] = self.convert_value("act_type", untuned_row["act_type"])
        config["dtype"] = self.convert_value("dtype", untuned_row["dtype"])
        config["q_dtype_a"] = self.convert_value("q_dtype_a", untuned_row["q_dtype_a"])
        config["q_dtype_w"] = self.convert_value("q_dtype_w", untuned_row["q_dtype_w"])
        config["q_type"] = self.convert_value("q_type", untuned_row["q_type"])
        config["use_g1u1"] = self.convert_value("use_g1u1", untuned_row["use_g1u1"])
        config["doweight_stage1"] = self.convert_value(
            "doweight_stage1", untuned_row["doweight_stage1"]
        )

        # Get tile parameters from tile_row
        config["block_m"] = int(tile_row["tile_m"])
        config["ksplit"] = 0
        config["us1"] = 0.0
        config["kernelName1"] = tile_row["kernelName1"]
        config["err1"] = "0.0%"
        config["us2"] = 0.0
        config["kernelName2"] = tile_row["kernelName2"]
        config["err2"] = "0.0%"
        config["us"] = 0.0
        config["run_1stage"] = 0
        config["tflops"] = 0.0
        config["bw"] = 0.0

        return config

    def run_single_test(self, config, test_data, tile_idx):
        """Run performance test for a single configuration - using CK MoE measurement method"""
        from aiter.fused_moe import fused_moe_, flydsl_moe_stage1, flydsl_moe_stage2
        from aiter.test_common import run_perftest

        # Extract configuration parameters
        token = config["token"]
        model_dim = config["model_dim"]
        inter_dim = config["inter_dim"]
        expert = config["expert"]
        topk = config["topk"]
        block_m = config["block_m"]

        # Parse dtype and activation
        dtype = (
            eval(config["dtype"])
            if isinstance(config["dtype"], str)
            else config["dtype"]
        )
        q_dtype_a = (
            eval(config["q_dtype_a"])
            if isinstance(config["q_dtype_a"], str)
            else config["q_dtype_a"]
        )
        q_dtype_w = (
            eval(config["q_dtype_w"])
            if isinstance(config["q_dtype_w"], str)
            else config["q_dtype_w"]
        )

        # Parse q_type and act_type
        q_type_str = config["q_type"]
        if isinstance(q_type_str, str) and "QuantType." in q_type_str:
            q_type = eval(q_type_str)
        else:
            q_type = QuantType.per_Token

        act_type_str = config["act_type"]
        if isinstance(act_type_str, str) and "ActivationType." in act_type_str:
            act_type = eval(act_type_str)
        else:
            act_type = ActivationType.Silu

        use_g1u1 = bool(config["use_g1u1"])
        doweight_stage1 = bool(config["doweight_stage1"])

        # 🔧 Create temporary config file and set environment variable
        # This allows get_2stage_cfgs() in fused_moe.py to load the correct tile configuration
        temp_csv_file = None
        old_aiter_config_fmoe = os.environ.get("AITER_CONFIG_FMOE", None)

        try:
            # Create temporary config file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                temp_csv_file = f.name
                # Write header
                header = ",".join(config.keys())
                f.write(header + "\n")
                # Write data
                values = ",".join(str(v) for v in config.values())
                f.write(values + "\n")

            # Set environment variable to let fused_moe.py load this config
            os.environ["AITER_CONFIG_FMOE"] = temp_csv_file

            # CRITICAL: Reset ALL caches to force re-reading config
            # 1. Reset global cfg_2stages cache
            import aiter.fused_moe as fused_moe_module

            fused_moe_module.cfg_2stages = None

            # 2. Clear AITER_CONFIGS.get_config_file lru_cache
            # Without this, get_config_file() returns stale cached path!
            from aiter.jit.core import AITER_CONFIGS

            AITER_CONFIGS.get_config_file.cache_clear()

            # 3. Clear get_2stage_cfgs lru_cache in fused_moe
            if hasattr(fused_moe_module, "get_2stage_cfgs"):
                fused_moe_module.get_2stage_cfgs.cache_clear()

            # Print FlyDSL configuration info (only first time)
            if tile_idx == 0:
                print(f"\n  🔧 FlyDSL Configuration:")
                print(f"     AITER_CONFIG_FMOE = {temp_csv_file}")
                print(
                    f"     kernelName1 = {config['kernelName1']} ← contains 'flydsl', automatically uses FlyDSL"
                )
                print(f"     kernelName2 = {config['kernelName2']}")
                print(f"     block_m = {block_m}\n")

            # Prepare output buffer
            moe_output = torch.zeros((token, model_dim), dtype=dtype, device="cuda")

            # Prepare input quantization (for Stage1 separate test)
            torch_quant = aiter.get_torch_quant(q_type)
            a1_qt, a1_scale = torch_quant(test_data["input"], quant_dtype=q_dtype_a)

            # ========== Measure Stage1 time (using CK MoE method) ==========
            a2_output = torch.zeros(
                (token, topk, inter_dim), dtype=dtype, device="cuda"
            )

            # Use run_perftest for performance testing (default num_iters=101)
            _, us1_time = run_perftest(
                flydsl_moe_stage1,
                test_data["input"],
                test_data["w1_qt"],
                test_data["w2_qt"],  # w2 (needed for inter_dim inference)
                test_data["sorted_ids"],
                test_data["sorted_expert_ids"],
                test_data["num_valid_ids"],
                a2_output,
                topk,
                block_m,
                a1_scale,
                test_data["w1_scale"],
                kernelName=config["kernelName1"],
                sorted_weights=test_data["sorted_weights"],
                quant_type=q_type,
                activation=act_type,
                splitk=0,
                dtype=dtype,
            )

            # ========== Measure Stage2 time (using CK MoE method) ==========
            # Use Stage1 output as Stage2 input
            a2_qt, a2_scale = torch_quant(a2_output, quant_dtype=q_dtype_a)

            # Use run_perftest for performance testing (default num_iters=101)
            _, us2_time = run_perftest(
                flydsl_moe_stage2,
                a2_output,
                test_data["w1_qt"],  # not used in stage2
                test_data["w2_qt"],
                test_data["sorted_ids"],
                test_data["sorted_expert_ids"],
                test_data["num_valid_ids"],
                moe_output,
                topk,
                test_data["w2_scale"],
                a2_scale,
                block_m,
                kernelName=config["kernelName2"],
                sorted_weights=test_data["sorted_weights"],
                quant_type=q_type,
                activation=act_type,
            )

            # ========== Measure complete 2-stage time (using CK MoE method) ==========
            # Use run_perftest for performance testing (default num_iters=101)
            _, us_time = run_perftest(
                fused_moe_,
                test_data["input"],  # ← Unquantized input
                test_data["w1_qt"],
                test_data["w2_qt"],
                test_data["topk_weights"],
                test_data["topk_ids"],
                expert_mask=None,
                activation=act_type.value,
                quant_type=q_type.value,
                doweight_stage1=doweight_stage1,
                w1_scale=test_data["w1_scale"],
                w2_scale=test_data["w2_scale"],
                a1_scale=None,  # ← Let function auto-quantize
                a2_scale=None,
                block_size_M=block_m,
                dtype=dtype,
                use_flydsl=True,
            )

            # ========== Clean up GPU memory to reduce crash probability ==========
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Calculate TFLOPS
            if use_g1u1:
                n = inter_dim * 2
            else:
                n = inter_dim
            flop = (
                token * n * model_dim * topk * 2  # gemm1
                + topk * token * model_dim * inter_dim * 2  # gemm2
            )
            tflops = flop / (us_time * 1e6)

            # Calculate bandwidth (GB/s)
            # Memory access: input + w1 + w2 + output + scales
            dtype_bytes = 2 if dtype == torch.bfloat16 else 4  # bf16=2, fp32=4
            q_dtype_bytes = 1  # fp8=1, int8=1

            input_bytes = token * model_dim * q_dtype_bytes  # Quantized input
            w1_bytes = expert * n * model_dim * q_dtype_bytes
            w2_bytes = expert * model_dim * inter_dim * q_dtype_bytes
            output_bytes = token * model_dim * dtype_bytes
            scale_bytes = (token + expert * (n + model_dim)) * 4  # scales are fp32

            total_bytes = input_bytes + w1_bytes + w2_bytes + output_bytes + scale_bytes
            bw = total_bytes / (us_time * 1e-6) / 1e9  # GB/s

            # Calculate reference result and error
            from aiter.fused_moe import torch_moe

            ref_output = torch_moe(
                test_data["input"],
                test_data["w1_qt"],
                test_data["w2_qt"],
                test_data["topk_weights"],
                test_data["topk_ids"],
                test_data["w1_scale"],
                test_data["w2_scale"],
                None,
                None,
                None,
                act_type,
            )

            # Calculate error
            diff = (moe_output - ref_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            relative_error = (diff / (ref_output.abs() + 1e-6)).mean().item()
            error_rate = relative_error * 100

            return {
                "config": config,
                "us_time": us_time,
                "us1_time": us1_time,
                "us2_time": us2_time,
                "bw": bw,
                "tflops": tflops,
                "error_rate": error_rate,
                "success": True,
                "tile_idx": tile_idx,
            }

        except Exception as e:
            print(f"  ❌ Exception: {e}")
            import traceback

            traceback.print_exc()
            return {
                "config": config,
                "us_time": None,
                "tflops": None,
                "error_rate": None,
                "success": False,
                "tile_idx": tile_idx,
            }
        finally:
            # 🔧 Clean up GPU memory to prevent memory accumulation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # NOTE: Don't delete temp_csv_file here - it may still be accessed by
            # profiler or other async operations. Let OS clean it up at process exit.
            # Also keep AITER_CONFIG_FMOE for cfg_2stages cache consistency.
            pass

    def tune_single_input(self, untuned_idx, untuned_row):
        """Iterate all tile parameters for a single input configuration - reference tune.py's batch test method"""
        print(f"\n{'='*80}")
        print(f"📋 Input configuration #{untuned_idx + 1}/{len(self.untuned_configs)}")
        print(f"{'='*80}")
        print(
            f"  token={untuned_row['token']}, model_dim={untuned_row['model_dim']}, "
            f"inter_dim={untuned_row['inter_dim']}, expert={untuned_row['expert']}, topk={untuned_row['topk']}"
        )
        print(f"  Candidate tile configurations: {len(self.candidate_tiles)}")
        print(f"{'='*80}\n")

        # Parse parameters
        token = int(untuned_row["token"])
        model_dim = int(untuned_row["model_dim"])
        inter_dim = int(untuned_row["inter_dim"])
        expert = int(untuned_row["expert"])
        topk = int(untuned_row["topk"])

        dtype_str = untuned_row["dtype"]
        dtype = (
            eval(dtype_str)
            if isinstance(dtype_str, str) and "torch." in dtype_str
            else dtypes.bf16
        )

        q_dtype_a_str = untuned_row["q_dtype_a"]
        q_dtype_a = (
            eval(q_dtype_a_str)
            if isinstance(q_dtype_a_str, str) and "torch." in q_dtype_a_str
            else dtypes.fp8
        )

        q_dtype_w_str = untuned_row["q_dtype_w"]
        q_dtype_w = (
            eval(q_dtype_w_str)
            if isinstance(q_dtype_w_str, str) and "torch." in q_dtype_w_str
            else dtypes.fp8
        )

        q_type_str = untuned_row["q_type"]
        q_type = (
            eval(q_type_str)
            if isinstance(q_type_str, str) and "QuantType." in q_type_str
            else QuantType.per_Token
        )

        use_g1u1 = bool(untuned_row["use_g1u1"])

        results = []

        # Iterate all tile configurations - generate test data for EACH tile
        # CRITICAL: sorted_ids size = token*topk + expert*block_m - topk
        # Each tile_m requires its own test data to avoid memory access faults!
        for tile_idx, tile_row in self.candidate_tiles.iterrows():
            config = self.create_tuned_config(untuned_row, tile_row)
            block_m = int(tile_row["tile_m"])

            print(
                f"🔧 Testing tile #{tile_idx + 1}/{len(self.candidate_tiles)}: "
                f"{tile_row['tile_m']}x{tile_row['tile_n']}x{tile_row['tile_k']}",
                end=" ",
            )

            # Generate test data with matching block_m for THIS tile configuration
            test_data = self.generate_test_data(
                token=token,
                model_dim=model_dim,
                inter_dim=inter_dim,
                expert=expert,
                topk=topk,
                dtype=dtype,
                q_dtype_a=q_dtype_a,
                q_dtype_w=q_dtype_w,
                q_type=q_type,
                use_g1u1=use_g1u1,
                block_m=block_m,
            )

            result = self.run_single_test(config, test_data, tile_idx)
            results.append(result)

            if result["success"]:
                print(
                    f"✅ {result['us_time']:.2f} us, {result['tflops']:.2f} tflops, "
                    f"{result['bw']:.2f} GB/s, err: {result['error_rate']:.1f}%"
                )
            else:
                print(f"❌ Failed")

        # Filter successful results
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            print(f"\n❌ No successful tile configurations!")
            return None

        # ========== Group results by block_m ==========
        # Key insight: Stage1 and Stage2 MUST use the same block_m for moe_sorting compatibility
        from collections import defaultdict

        results_by_block_m = defaultdict(list)
        for r in successful_results:
            block_m = r["config"]["block_m"]
            results_by_block_m[block_m].append(r)

        print(f"\n📊 Results grouped by block_m:")
        for block_m in sorted(results_by_block_m.keys()):
            print(
                f"   block_m={block_m}: {len(results_by_block_m[block_m])} configurations"
            )

        # ========== For each block_m, find best Stage1 and best Stage2 ==========
        best_configs_by_block_m = {}

        for block_m, block_results in results_by_block_m.items():
            # Find best Stage1 within this block_m group
            best_s1 = min(block_results, key=lambda x: x["us1_time"])
            # Find best Stage2 within this block_m group
            best_s2 = min(block_results, key=lambda x: x["us2_time"])

            # Combined time using best Stage1 + best Stage2 from SAME block_m
            combined_us = best_s1["us1_time"] + best_s2["us2_time"]

            best_configs_by_block_m[block_m] = {
                "block_m": block_m,
                "best_s1": best_s1,
                "best_s2": best_s2,
                "combined_us": combined_us,
            }

        # ========== Find overall best block_m (by combined time) ==========
        best_block_m = min(
            best_configs_by_block_m.keys(),
            key=lambda bm: best_configs_by_block_m[bm]["combined_us"],
        )
        best_combo = best_configs_by_block_m[best_block_m]

        # Sort by overall performance for ranking display
        successful_results.sort(key=lambda x: x["us_time"])

        # Print overall performance ranking (Top 3)
        print(f"\n📊 Overall Performance Ranking (Top 3 - same tile for both stages):")
        print(
            f"{'Rank':<6} {'Tile Config':<20} {'Total(us)':<12} {'Stage1(us)':<12} {'Stage2(us)':<12} {'TFLOPS':<12}"
        )
        print(f"{'-'*80}")

        for rank, result in enumerate(successful_results[:3], 1):
            tile_m = result["config"]["block_m"]
            tile_n = self.candidate_tiles.iloc[result["tile_idx"]]["tile_n"]
            tile_k = self.candidate_tiles.iloc[result["tile_idx"]]["tile_k"]
            tile_str = f"{tile_m}x{tile_n}x{tile_k}"
            print(
                f"{rank:<6} {tile_str:<20} {result['us_time']:<12.2f} {result['us1_time']:<12.2f} "
                f"{result['us2_time']:<12.2f} {result['tflops']:<12.2f}"
            )

        # ========== Print best Stage1 + Stage2 per block_m ==========
        print(f"\n{'='*80}")
        print(f"🎯 Best Stage1 + Stage2 per block_m (SAME block_m constraint):")
        print(f"{'='*80}")

        for block_m in sorted(best_configs_by_block_m.keys()):
            combo = best_configs_by_block_m[block_m]
            best_s1 = combo["best_s1"]
            best_s2 = combo["best_s2"]

            s1_tile_n = self.candidate_tiles.iloc[best_s1["tile_idx"]]["tile_n"]
            s1_tile_k = self.candidate_tiles.iloc[best_s1["tile_idx"]]["tile_k"]
            s2_tile_n = self.candidate_tiles.iloc[best_s2["tile_idx"]]["tile_n"]
            s2_tile_k = self.candidate_tiles.iloc[best_s2["tile_idx"]]["tile_k"]

            marker = " ⭐ BEST" if block_m == best_block_m else ""
            print(f"\n  block_m={block_m}{marker}")
            print(
                f"    Best Stage1: {block_m}x{s1_tile_n}x{s1_tile_k} → {best_s1['us1_time']:.2f} us"
            )
            print(f"      kernelName1: {best_s1['config']['kernelName1']}")
            print(
                f"    Best Stage2: {block_m}x{s2_tile_n}x{s2_tile_k} → {best_s2['us2_time']:.2f} us"
            )
            print(f"      kernelName2: {best_s2['config']['kernelName2']}")
            print(f"    Combined: {combo['combined_us']:.2f} us")

        # ========== Build output config with best block_m's Stage1 + Stage2 ==========
        best_s1 = best_combo["best_s1"]
        best_s2 = best_combo["best_s2"]

        # Start with best_s1's config as base
        best_config = best_s1["config"].copy()

        # Override kernelName2 with best Stage2's kernel (SAME block_m)
        best_config["kernelName2"] = best_s2["config"]["kernelName2"]

        # Record timing info
        best_config["us1"] = round(best_s1["us1_time"], 2)
        best_config["us2"] = round(best_s2["us2_time"], 2)
        best_config["us"] = round(best_combo["combined_us"], 2)

        # Calculate combined performance metrics
        if use_g1u1:
            n = inter_dim * 2
        else:
            n = inter_dim
        flop = (
            token * n * model_dim * topk * 2  # gemm1
            + topk * token * model_dim * inter_dim * 2  # gemm2
        )
        best_config["tflops"] = round(flop / (best_combo["combined_us"] * 1e6), 2)

        # Calculate bandwidth
        dtype_bytes = 2 if dtype == torch.bfloat16 else 4
        q_dtype_bytes = 1
        input_bytes = token * model_dim * q_dtype_bytes
        w1_bytes = expert * n * model_dim * q_dtype_bytes
        w2_bytes = expert * model_dim * inter_dim * q_dtype_bytes
        output_bytes = token * model_dim * dtype_bytes
        scale_bytes = (token + expert * (n + model_dim)) * 4
        total_bytes = input_bytes + w1_bytes + w2_bytes + output_bytes + scale_bytes
        best_config["bw"] = round(
            total_bytes / (best_combo["combined_us"] * 1e-6) / 1e9, 2
        )

        # block_m is the same for both stages now
        best_config["block_m"] = best_block_m

        # Extract tile info for display
        s1_tile_n = self.candidate_tiles.iloc[best_s1["tile_idx"]]["tile_n"]
        s1_tile_k = self.candidate_tiles.iloc[best_s1["tile_idx"]]["tile_k"]
        s2_tile_n = self.candidate_tiles.iloc[best_s2["tile_idx"]]["tile_n"]
        s2_tile_k = self.candidate_tiles.iloc[best_s2["tile_idx"]]["tile_k"]

        print(f"\n{'='*80}")
        print(f"✅ Final Output Configuration (block_m={best_block_m}):")
        print(f"{'='*80}")
        print(f"   kernelName1: {best_config['kernelName1']}")
        print(
            f"      → Stage1 tile: {best_block_m}x{s1_tile_n}x{s1_tile_k}, Time: {best_s1['us1_time']:.2f} us"
        )
        print(f"   kernelName2: {best_config['kernelName2']}")
        print(
            f"      → Stage2 tile: {best_block_m}x{s2_tile_n}x{s2_tile_k}, Time: {best_s2['us2_time']:.2f} us"
        )
        print(f"   Combined Time: {best_combo['combined_us']:.2f} us")
        print(
            f"   TFLOPS: {best_config['tflops']:.2f}, BW: {best_config['bw']:.2f} GB/s"
        )
        print(f"{'='*80}\n")

        return best_config

    def tune_all(self):
        """Iterate all input configurations"""
        print(f"\n{'='*80}")
        print(f"Starting batch FlyDSL MoE Tuning")
        print(f"{'='*80}")
        print(f"Input configurations: {len(self.untuned_configs)}")
        print(f"Candidate tile configurations: {len(self.candidate_tiles)}")
        print(f"Total tests: {len(self.untuned_configs) * len(self.candidate_tiles)}")
        print(f"{'='*80}\n")

        all_best_configs = []

        # Iterate all input configurations
        for idx, row in self.untuned_configs.iterrows():
            best_config = self.tune_single_input(idx, row)
            if best_config:
                all_best_configs.append(best_config)

        if not all_best_configs:
            print("\n❌ No successful configurations!")
            return False

        # Save all best configurations
        df = pd.DataFrame(all_best_configs)
        df.to_csv(self.output_csv, index=False)

        print(f"\n{'='*80}")
        print(f"✅ Batch Tuning Complete")
        print(f"{'='*80}")
        print(
            f"Successful configurations: {len(all_best_configs)}/{len(self.untuned_configs)}"
        )
        print(f"Results saved to: {self.output_csv}")
        print(f"{'='*80}\n")

        # Display summary
        print("📊 All Configurations Summary:")
        print(
            f"{'Token':<8} {'Model':<8} {'Inter':<8} {'Expert':<8} {'TopK':<6} "
            f"{'Best Tile':<16} {'Total(us)':<10} {'S1(us)':<10} {'S2(us)':<10} {'TFLOPS':<10}"
        )
        print(f"{'-'*108}")

        for config in all_best_configs:
            # Extract tile info from kernelName
            kernel_match = re.search(r"(\d+)x(\d+)x(\d+)", config["kernelName1"])
            if kernel_match:
                tile_str = f"{kernel_match.group(1)}x{kernel_match.group(2)}x{kernel_match.group(3)}"
            else:
                tile_str = "N/A"

            print(
                f"{config['token']:<8} {config['model_dim']:<8} {config['inter_dim']:<8} "
                f"{config['expert']:<8} {config['topk']:<6} {tile_str:<16} "
                f"{config['us']:<10.2f} {config['us1']:<10.2f} {config['us2']:<10.2f} {config['tflops']:<10.2f}"
            )

        # Display selected kernels (best Stage1 + best Stage2 with SAME block_m)
        print(f"\n{'='*80}")
        print("🎯 Selected Kernels (Best Stage1 + Best Stage2, SAME block_m):")
        print(f"{'='*80}")
        print(
            f"{'Token':<8} {'block_m':<8} {'kernelName1 (Best Stage1)':<45} {'S1(us)':<10}"
        )
        print(f"{'-'*80}")
        for config in all_best_configs:
            print(
                f"{config['token']:<8} {config['block_m']:<8} {config['kernelName1']:<45} "
                f"{config['us1']:<10.2f}"
            )

        print()
        print(
            f"{'Token':<8} {'block_m':<8} {'kernelName2 (Best Stage2)':<45} {'S2(us)':<10}"
        )
        print(f"{'-'*80}")
        for config in all_best_configs:
            print(
                f"{config['token']:<8} {config['block_m']:<8} {config['kernelName2']:<45} "
                f"{config['us2']:<10.2f}"
            )

        print(f"\n{'='*80}")
        print(
            "✅ Stage1 and Stage2 use the SAME block_m - compatible with moe_sorting!"
        )
        print(f"{'='*80}")

        return True


def print_header(args):
    """Print formatted header and configuration info"""
    print("=" * 80)
    print("FlyDSL MoE Batch Configuration Tuning")
    print("=" * 80)
    print()
    print("File Paths:")
    print(f"  Untuned config:       {args.untuned}")
    print(f"  Candidate tile config: {args.candidate}")
    print(f"  Output config:        {args.output}")
    print()
    print("Environment Variables:")
    print(f"  DSL2_ROOT:            {os.environ.get('DSL2_ROOT', '/data/FlyDSL/')}")
    print(f"  CU Count:             {args.cu_num}")
    print("=" * 80)
    print()


def check_files(args):
    """Check input files and display statistics"""
    # Check untuned file
    if not os.path.exists(args.untuned):
        print(f"❌ Error: Untuned configuration file does not exist: {args.untuned}")
        sys.exit(1)

    # Check candidate file
    if not os.path.exists(args.candidate):
        print(
            f"❌ Error: Candidate configuration file does not exist: {args.candidate}"
        )
        sys.exit(1)

    # Count configurations
    untuned_df = pd.read_csv(args.untuned)
    candidate_df = pd.read_csv(args.candidate)

    untuned_count = len(untuned_df)
    candidate_count = len(candidate_df)
    total_tests = untuned_count * candidate_count

    print("📊 Configuration Statistics:")
    print(f"  Input configurations:  {untuned_count}")
    print(f"  Candidate tiles:       {candidate_count}")
    print(f"  Total tests:           {total_tests}")
    print()

    # Display input configuration preview
    print("📋 Input Configuration List:")
    print(untuned_df.to_string(index=False, max_rows=5))
    if untuned_count > 5:
        print(f"... ({untuned_count} configurations total)")
    print()

    # Time warning
    if total_tests > 50:
        print(f"⚠️  Warning: Will run {total_tests} tests, this may take a while")
        print(f"Estimated time: ~{total_tests * 2 // 60} minutes")
        print()


def print_results(args, success):
    """Print results summary"""
    if not success:
        print()
        print("❌ Batch Tuning Failed!")
        sys.exit(1)

    print()
    print("=" * 80)
    print("✅ Batch Tuning Complete!")
    print("=" * 80)
    print()
    print(f"Results file: {args.output}")
    print()

    # Display results preview
    if os.path.exists(args.output):
        result_df = pd.read_csv(args.output)
        result_count = len(result_df)

        print(f"Successful configurations: {result_count}")
        print()
        print("Results Preview:")
        print(result_df.to_string(index=False, max_rows=3))
        if result_count > 3:
            print(f"... ({result_count} configurations total)")
        print()

    print("=" * 80)
    print("Use Tuned Configuration:")
    print(f'  export AITER_CONFIG_FMOE="{args.output}"')
    print(
        "  python op_tests/test_moe_2stage.py -t <token> -dim <dim> -e <expert> -k <topk> -q 2 -a silu"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="FlyDSL MoE Batch Configuration Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
      Use default parameters, tune all configurations in untuned_fmoe.csv
  
  %(prog)s -u my_configs.csv -o my_tuned.csv
      Use custom input and output files

Untuned CSV Format:
  token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
  128,7168,512,8,2,ActivationType.Silu,torch.bfloat16,fp8,fp8,QuantType.per_Token,1,0
  256,8192,1024,16,4,ActivationType.Silu,torch.bfloat16,fp8,fp8,QuantType.per_Token,1,0
  
  Note: "fp8" will auto-select based on GPU architecture:
    - gfx942 (MI300): torch.float8_e4m3fnuz
    - gfx950: torch.float8_e4m3fn
        """,
    )

    parser.add_argument(
        "-u",
        "--untuned",
        default="/data/aiter/aiter/configs/untuned_fmoe.csv",
        help="Untuned configuration CSV file (input shape list)",
    )
    parser.add_argument(
        "-c",
        "--candidate",
        default="/data/aiter/hsa/gfx942/fmoe_2stages/flydsl_moe_bf16_pertokenFp8_g1u1.csv",
        help="Candidate tile configuration CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/data/aiter/aiter/configs/tuned_flydsl_moe_batch.csv",
        help="Output tuned configuration CSV file",
    )
    parser.add_argument("--cu_num", type=int, default=80, help="CU count (default: 80)")

    args = parser.parse_args()

    # Set environment variable
    if "DSL2_ROOT" not in os.environ:
        os.environ["DSL2_ROOT"] = "/data/FlyDSL/"

    # Change to aiter directory
    os.chdir("/data/aiter")

    # Initialize CUDA
    print("🔧 Initializing CUDA...")
    if not torch.cuda.is_available():
        print("❌ Error: CUDA not available!")
        sys.exit(1)
    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}\n")

    # Print header
    print_header(args)

    # Check files and display statistics
    check_files(args)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 80)
    print("Starting batch tuning...")
    print("=" * 80)
    print()

    # Run batch tuning
    tuner = FlyDSLBatchTuner(
        untuned_csv=args.untuned,
        candidate_csv=args.candidate,
        output_csv=args.output,
        cu_num=args.cu_num,
    )

    success = tuner.tune_all()

    # Print results
    print_results(args, success)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
