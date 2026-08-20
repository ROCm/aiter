# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F
from batched_gemm_bf16_common import kernels_list as ck_kernels_list

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_BF16_BATCHED_GEMM
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner


OPUS_TUNE_ERROR = None
try:
    _opus_csrc = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../opus_gemm")
    )
    if _opus_csrc not in sys.path:
        sys.path.insert(0, _opus_csrc)

    from opus_gemm_tune import (  # type: ignore[import-not-found]
        _ensure_kids_compiled as _opus_ensure_kids_compiled,
    )
    from opus_gemm_tune import (  # type: ignore[import-not-found]
        a16w16_all_kernels as _opus_kernels_list,
    )
    from opus_gemm_tune import (  # type: ignore[import-not-found]
        candidate_kids_for_shape as _opus_candidate_kids_for_shape,
    )
    from opus_gemm_tune import (  # type: ignore[import-not-found]
        candidate_splitK as _opus_candidate_splitK,
    )
    from opus_gemm_tune import (  # type: ignore[import-not-found]
        kid_rejects_shape as _opus_kid_rejects_shape,
    )

    from aiter.ops.opus import opus_bmm as _opus_bmm
    from aiter.ops.opus.policy import resolve_a16w16_tuned_candidate
except Exception as _opus_exc:  # noqa: BLE001
    _opus_bmm = None
    _opus_kernels_list = {}
    _opus_candidate_kids_for_shape = None
    _opus_candidate_splitK = None
    _opus_kid_rejects_shape = None
    _opus_ensure_kids_compiled = None
    resolve_a16w16_tuned_candidate = None
    OPUS_TUNE_ERROR = str(_opus_exc)


def run_torch(x, weight, bias=None, dtype=dtypes.bf16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        b_out = F.linear(x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32))
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


def run_batched_gemm(x, weight, out, kernel_id, splitK=0):
    aiter.batched_gemm_bf16_tune(x, weight, out, kernel_id, splitK)
    return out


def run_opus_batched_gemm(x, weight, out, kernel_id, splitK=0):
    if _opus_bmm is None:
        raise RuntimeError(f"OPUS is unavailable: {OPUS_TUNE_ERROR}")
    _opus_bmm(x, weight, out, kid=int(kernel_id), split_k=int(splitK))
    return out


def generate_data(b, m, n, k, device="cuda"):
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.bf16, device=device)
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.bf16, device=device)
    out = torch.empty(b, m, n, dtype=dtypes.bf16, device=device)
    return {"x": x, "weight": weight, "out": out}


class BatchedGemmBf16Tuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_BF16_BATCHED_GEMM}",
        "untune_file": "aiter/configs/bf16_untuned_batched_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "config_env_name": "AITER_CONFIG_BF16_BATCHED_GEMM",
    }

    def _clear_op_caches(self):
        from aiter.ops.batched_gemm_op_bf16 import (
            _clear_batched_gemm_bf16_config_caches,
        )

        _clear_batched_gemm_bf16_config_caches()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--libtype",
            choices=("ck", "opus", "all"),
            default="ck",
            help=(
                "BF16 BMM backend candidates to tune. 'ck' preserves the "
                "legacy tuner; 'opus' scans exact OPUS A16W16 kids; 'all' "
                "benchmarks both and stores the fastest backend."
            ),
        )

    def pre_process(self, args):
        super().pre_process(args)
        # Migrate legacy CK-only rows in memory.  The next successful write
        # persists explicit libtype=ck while still accepting old CSVs at
        # runtime and in CK codegen.
        if self.tunedf is not None:
            self.tunedf = self.tunedf.copy()
            if "libtype" not in self.tunedf.columns:
                self.tunedf["libtype"] = "ck"
            else:
                self.tunedf["libtype"] = self.tunedf["libtype"].map(
                    lambda value: (
                        "ck"
                        if value is None
                        or (not isinstance(value, str) and pd.isna(value))
                        or str(value).strip().lower()
                        in ("", "0", "nan", "none", "null")
                        else str(value).strip().lower()
                    )
                )

    def run_config(self, args):
        from aiter.ops.batched_gemm_op_bf16 import batched_gemm_bf16_tuned
        from aiter.test_common import checkAllclose, run_perftest

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            B = int(row["B"])
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            shape_str = f"({B}, {M}, {N}, {K})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                gd = generate_data(B, M, N, K)
                x, weight = gd["x"], gd["weight"]
                out, us = run_perftest(
                    batched_gemm_bf16_tuned,
                    x,
                    weight,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = run_torch(x, weight)
                err_ratio = checkAllclose(out, ref, msg=f"run_config {shape_str}")
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:  # noqa: BLE001
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
        return results

    def calculate(self, results, bpes=(2, 2, 2)):
        info, time, _err_ratio = results
        if time == -1:
            return -1, -1
        _gfx, _cu_num, b, m, n, k = info[0]
        flops = m * n * k * 2 * b
        tflops = round(flops / (time * 1000000), 2)
        lhs_bpe, rhs_bpe, out_bpe = bpes
        bw = round(
            b
            * (m * k * lhs_bpe + n * k * rhs_bpe + m * n * out_bpe)
            / (time * 1e-6)
            / 1e9,
            2,
        )
        return tflops, bw

    def getKernelName(self, kernelId):
        kernel = ck_kernels_list.get(int(kernelId))
        return None if kernel is None else kernel.name

    def _kernel_name(self, libtype: str, kernel_id: int) -> str | None:
        if libtype == "ck":
            return self.getKernelName(kernel_id)
        if libtype == "opus":
            kernel = _opus_kernels_list.get(int(kernel_id))
            return None if kernel is None else kernel.name
        return None

    def result_to_df(self, results):
        """Serialize the backend tag alongside the backend-local exact id."""
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, libtype, kernel_id, splitK, kernel_name = info
            if kernel_name == "" or pd.isna(kernel_name):
                kernel_name = self._kernel_name(libtype, kernel_id)
            if kernel_name is None or pd.isna(kernel_name):
                kernel_name = "None"
            tflops, bw = self.calculate(el)
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "libtype": str(libtype),
                    "kernelId": int(kernel_id),
                    "splitK": int(splitK),
                    "us": time,
                    "kernelName": str(kernel_name),
                    "errRatio": err_ratio,
                    "tflops": tflops,
                    "bw": bw,
                }
            )
            if len(results) == self.topk:
                print(
                    "Tuning result for "
                    f"{str(dict(zip(self.keys, keys))).strip('{}')} is "
                    f"libtype={libtype}, kernelId={kernel_id} "
                    f"{kernel_name} splitK={splitK}, {time}us, "
                    f"err_ratio={err_ratio}, tflops={tflops} TFLOPS, "
                    f"bw={bw} GB/s"
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        tune_ck = args.libtype in ("ck", "all")
        tune_opus = args.libtype in ("opus", "all")
        if tune_opus and _opus_bmm is None:
            aiter.logger.warning(
                f"OPUS is unavailable; skipping OPUS BF16 BMM candidates: "
                f"{OPUS_TUNE_ERROR}"
            )
            tune_opus = False

        task = []
        tasks_data = []
        opus_candidate_kids: set[int] = set()
        for i in range(len(untunedf)):
            B = int(untunedf.loc[i, "B"])
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            shape_key = (gfx, cu_num, B, M, N, K)

            print(
                f"tuning B:{B}, M:{M}, N:{N}, K:{K}, "
                f"libtype={args.libtype}"
            )
            total_kernel_nums = 0

            if tune_ck:
                for kid, kernel in sorted(ck_kernels_list.items()):
                    maxsplitK = (
                        aiter.compute_batched_gemm_SplitK(
                            M,
                            N,
                            K,
                            kernel.MPerBLOCK,
                            kernel.NPerBLOCK,
                            kernel.KPerBLOCK,
                        )
                        if useSplitK
                        else 0
                    )
                    for splitK in range(maxsplitK + 1):
                        info = (
                            shape_key,
                            "ck",
                            int(kid),
                            int(splitK),
                            kernel.name,
                        )
                        task.append(
                            (
                                info,
                                generate_data,
                                (B, M, N, K),
                                run_batched_gemm,
                                (["x", "weight", "out"], kid, splitK),
                                {
                                    "num_warmup": args.warmup,
                                    "num_iters": args.iters,
                                },
                                run_torch,
                                (["x", "weight"],),
                                {},
                                None,
                                1e-2,
                                1e-2,
                                None,
                                None,
                                ("out",),
                            )
                        )
                        total_kernel_nums += 1

            if tune_opus:
                assert _opus_candidate_kids_for_shape is not None
                assert _opus_candidate_splitK is not None
                assert _opus_kid_rejects_shape is not None
                assert resolve_a16w16_tuned_candidate is not None
                candidate_kids = _opus_candidate_kids_for_shape(
                    M, N, K, False, cu_num
                )
                seen_pairs: set[tuple[int, int]] = set()
                for kid in sorted(candidate_kids):
                    instance = _opus_kernels_list.get(int(kid))
                    if instance is None:
                        continue
                    if _opus_kid_rejects_shape(instance, M, N, K):
                        continue
                    if instance.splitk_workspace_dtype is None:
                        splitK_range = [0]
                    else:
                        splitK_range = _opus_candidate_splitK(
                            M, N, K, B, cu_num, instance
                        )
                    for splitK in splitK_range:
                        plan = resolve_a16w16_tuned_candidate(
                            arch=gfx,
                            M=M,
                            N=N,
                            K=K,
                            batch=B,
                            cu_num=cu_num,
                            has_bias=False,
                            input_dtype=dtypes.bf16,
                            output_dtype=dtypes.bf16,
                            requested_kid=int(kid),
                            requested_split_k=int(splitK),
                        )
                        if plan is None:
                            continue
                        pair = (int(plan.resolved_kid), int(splitK))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        resolved_instance = _opus_kernels_list.get(pair[0])
                        if resolved_instance is None:
                            continue
                        opus_candidate_kids.add(pair[0])
                        info = (
                            shape_key,
                            "opus",
                            pair[0],
                            pair[1],
                            resolved_instance.name,
                        )
                        task.append(
                            (
                                info,
                                generate_data,
                                (B, M, N, K),
                                run_opus_batched_gemm,
                                (["x", "weight", "out"], pair[0], pair[1]),
                                {
                                    "num_warmup": args.warmup,
                                    "num_iters": args.iters,
                                },
                                run_torch,
                                (["x", "weight"],),
                                {},
                                None,
                                1e-2,
                                1e-2,
                                None,
                                None,
                                ("out",),
                            )
                        )
                        total_kernel_nums += 1

            tasks_data.append((total_kernel_nums, ()))

        if (
            opus_candidate_kids
            and _opus_ensure_kids_compiled is not None
            and _opus_ensure_kids_compiled(opus_candidate_kids)
        ):
            aiter.logger.info(
                "OPUS subset-compile expanded to cover "
                f"{len(opus_candidate_kids)} BF16 BMM candidate kids"
            )

        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )

        return ret


if __name__ == "__main__":
    key = [
        "gfx",
        "cu_num",
        "B",
        "M",
        "N",
        "K",
    ]
    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "errRatio",
        "tflops",
        "bw",
    ]

    tuner = BatchedGemmBf16Tuner(
        "BatchedGemmBf16Tuner",
        key,
        resultList,
        "Tune CK and OPUS batch GEMM BF16 kernels",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
