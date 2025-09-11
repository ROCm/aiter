# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import argparse
import torch
import pandas as pd

from abc import ABC, abstractmethod
from aiter import logger
import traceback
from operator import itemgetter
import time
from aiter import dtypes


class TunerCommon:
    ARG_DEFAULTS = {
        "verbose": False,
        "tune_file": "",
        "untune_file": "",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",  # for all results
    }
    dtype2bpe_dict = {
        dtypes.fp16: 2,
        dtypes.bf16: 2,
        dtypes.i16: 2,
        dtypes.fp8: 1,
        dtypes.fp8_e8m0: 1,
        dtypes.i8: 1,
        dtypes.i32: 4,
        dtypes.i4x2: 1,
        dtypes.fp4x2: 1,
        torch.uint8: 1,
        torch.uint32: 4,
        torch.int4: 1 / 2,
        torch.float8_e4m3fnuz: 1,
        torch.float8_e4m3fn: 1,
    }

    def __init__(self, name, key, resultList, description=None):
        self.parser = argparse.ArgumentParser(description=description)
        self._setup_common_arguments()
        self._setup_specific_arguments()
        self.columns = key + resultList
        self.keys = key
        self.tunedf = None
        self.untunedf = None
        self.name = name
        self.topk = 1

    def get_arg_defaults(self):
        """get default arguments"""
        return self.ARG_DEFAULTS.copy()

    def get_bpe(self, dtype):
        return self.dtype2bpe_dict[eval(dtype)]

    def _setup_common_arguments(self):
        """set common arguments"""
        defaults = self.get_arg_defaults()
        self.parser.add_argument(
            "--verbose", "-v", action="store_true", help="more info"
        )
        self.parser.add_argument(
            "-i",
            "--untune_file",
            default=defaults["untune_file"],
            required=False,
            help="input",
        )
        self.parser.add_argument(
            "-o",
            "--tune_file",
            default=defaults["tune_file"],
            required=False,
            help="output: tuning result store this file",
        )
        self.parser.add_argument(
            "--mp",
            type=int,
            default=torch.cuda.device_count(),
            help="Tuning on multiple GPUs using multiple processes",
        )
        self.parser.add_argument(
            "-k",
            "--splitK",
            action="store_true",
            required=False,
            help="Use splitK kernels",
        )
        self.parser.add_argument(
            "--sort",
            action="store_true",
            required=False,
            help="Arranged according to the keys",
        )
        self.parser.add_argument(
            "--errRatio",
            type=float,
            default=defaults["errRatio"],
            help="tolerable error ratio, default 0.05.",
        )
        self.parser.add_argument(
            "--batch",
            type=int,
            default=defaults["batch"],
            help="split untuned shapes to batches to tune",
        )
        self.parser.add_argument(
            "-o2",
            "--profile_file",
            default=defaults["profile_file"],
            required=False,
            help="output: all tuning results stored in this file",
        )

    def parse_args(self):
        return self.parser.parse_args()

    @abstractmethod
    def _setup_specific_arguments(self):
        """set specific arguments"""
        pass

    @abstractmethod
    def pre_process(self, args):
        """pre_process tunedf and untunedf"""
        pass

    @abstractmethod
    def tune(self, untunedf, tunedf, args):
        """tune process, return all results"""
        pass

    @abstractmethod
    def getKernelName(self, kernel_id):
        """获取kernel name"""
        pass

    @abstractmethod
    def calculate(self, results, inbpe=2, outbpe=2):
        """calculate TFLOPS and bandwidth"""
        pass

    def get_untuned_gemm_list(self, untuned_gemm_file):
        assert os.path.exists(
            untuned_gemm_file
        ), f"Not exist untuned file: {untuned_gemm_file}"
        untunedf = pd.read_csv(untuned_gemm_file)
        filtered_df = untunedf.drop_duplicates().reset_index(drop=True)
        return filtered_df

    def get_tuned_gemm_list(self, tuned_gemm_file):
        if os.path.exists(tuned_gemm_file):
            column_order = pd.read_csv(tuned_gemm_file, nrows=0).columns.tolist()
            tunedf = pd.read_csv(tuned_gemm_file)
            tunedf = tunedf[column_order]
        else:
            print(f"Not exist tuned file: {tuned_gemm_file}")
            tunedf = pd.DataFrame(columns=self.columns)
        return tunedf

    def sortResults(self, tunedf, issorted, values):
        if issorted:
            tunedf = tunedf.sort_values(by=values)
        print(tunedf)
        return tunedf

    def get_cu_num(self):
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        return cu_num

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        """post process, post process all results to return topk results"""
        rets = list(rets)
        if args.profile_file != "":
            self.result_to_csv(sorted(rets, key=itemgetter(0)), args.profile_file)
        if fast_mode or topk == -1:
            return rets
        best_time = -1
        tol_err_ratio = args.errRatio
        from collections import defaultdict

        grouped_rets = defaultdict(list)
        bestConfigs = []

        for info, us, max_err_ratio in rets:
            grouped_rets[info[0]].append((info[1:], us, max_err_ratio))

        grouped_results = list(grouped_rets.items())

        for info_key, time_list in grouped_results:
            sorted_time = sorted(time_list, key=lambda x: x[1])
            filtered_time = [
                (info_ex, round(us, 4), max_err_ratio)
                for info_ex, us, max_err_ratio in sorted_time
                if max_err_ratio <= tol_err_ratio and us != -1 and us != float("inf")
            ]
            if len(filtered_time) == 0:
                logger.error(
                    f"error: no valid candidate found for {info_key}, please check the result or errRatio in all result file running with --profile_file"
                )
            if len(filtered_time) < topk:
                topk = len(filtered_time)
                print(f"choose {topk} kernels")
            self.topk = topk
            best_config = [
                ((info_key, *info_ex), us, max_err_ratio)
                for info_ex, us, max_err_ratio in filtered_time[0:topk]
            ]
            if not best_config:
                logger.info(f"No kernel can be used for {info_key}")
                best_config = [((info_key, *sorted_time[0][0]), -1, 1.0)]
            bestConfigs.extend(best_config)
        return bestConfigs

    @abstractmethod
    def result_to_csv(self, results, file):
        pass

    def update_tflops_bw(self, tune_file):
        """update tflops and bw from old tune_file"""
        pass

    #
    def run(self, args, fast_mode=False):
        """tuner run function"""
        self.pre_process(args)
        if args.verbose:
            logger.info(f"args: {args}")
        if len(self.untunedf) == 0:
            # self.update_tflops_bw(args.tune_file)
            logger.info(f"no shapes to be tuned, skip tuning")
            return self.tunedf if self.tunedf is not None else pd.DataFrame()
        batch_size = min(args.batch, len(self.untunedf))
        total_batches = (len(self.untunedf) + batch_size - 1) // batch_size
        if args.verbose:
            logger.info(
                f"total shapes to be tuned: {len(self.untunedf) }, total_batches: {total_batches}, batch_size: {batch_size}"
            )
        processed_batches = 0
        results = []
        topk = -1 if fast_mode else 1
        start_time = time.time()
        try:
            for i in range(0, len(self.untunedf), batch_size):
                batch = self.untunedf.iloc[i : i + batch_size].reset_index(drop=True)
                processed_batches += 1
                all_results = self.tune(batch, self.tunedf, args)
                if all_results:
                    results = self.post_process(all_results, args, topk)
                    self.result_to_csv(results, args.tune_file)
                    logger.info(
                        f"processed {processed_batches} batches of {total_batches}, Processing Status ====> {round(processed_batches / total_batches,2)*100:.1f}% tuned in {self.name}"
                    )
                else:
                    logger.info("tune result is none or all shape is tuned!")
            logger.info(
                f"Tuning finished. total tuning time is {round(time.time() - start_time,4)} seconds"
            )
            tunedf = self.sortResults(pd.read_csv(args.tune_file), args.sort, self.keys)
            tunedf.to_csv(args.tune_file, index=False, na_rep="Null")
        except KeyboardInterrupt:
            logger.error(
                f"interrupted by user, tuning stopped, {processed_batches-1} batches processed"
            )
        except Exception as e:
            logger.error(
                f"error in batch {processed_batches} of {total_batches}: {str(e)}",
                exc_info=True,
            )


class GemmCommonTuner(TunerCommon):

    def __init__(
        self,
        name,
        key=["cu_num", "M", "N", "K"],
        resultList=[
            "kernelId",
            "splitK",
            "us",
            "kernelName",
            "errRatio",
            "tflops",
            "bw",
        ],
        description=None,
    ):
        super().__init__(name, key, resultList, description)

    def pre_process(self, args):
        self.untunedf = self.get_untuned_gemm_list(args.untune_file)
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        self.untunedf["cu_num"] = self.get_cu_num()
        untunedf_cols = self.untunedf.columns
        if len(self.tunedf) != 0:
            mask = self.untunedf.apply(tuple, axis=1).isin(
                self.tunedf[untunedf_cols].apply(tuple, axis=1)
            )
            self.untunedf = self.untunedf[~mask]

    def calculate(self, results, bpes=(2, 2, 2)):
        """calculate TFLOPS and bandwidth"""
        ### bpes: (inbpe, w_bpe, outbpe)
        ### gemm flops,bw
        info, time, err_ratio = results
        if time == -1:
            return -1, -1
        cu_num, m, n, k = info[0]
        flop = m * n * k * 2
        tflops = round(flop / (time * 1000000), 2)
        lhs_bpe, rhs_bpe, out_bpe = bpes
        bw = round(
            (m * k * lhs_bpe + n * k * rhs_bpe + m * n * out_bpe) / (time * 1e-6) / 1e9,
            2,
        )
        return tflops, bw

    def result_to_csv(self, results, file):
        """post process of tuning results"""
        resultdf = self.get_tuned_gemm_list(file)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName = info
            kernelName = (
                "None"
                if time == "nan"
                else self.getKernelName(kernelId) if kernelName == "" else kernelName
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} {kernelName} {splitK=}, {time}us, {err_ratio=}, {tflops=} TFLOPS, {bw=} GB/s"
                )
            key_dict.update(
                {
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bw],
                }
            )
            temp = pd.DataFrame(key_dict)
            resultdf = pd.concat([resultdf, temp], ignore_index=True)
        resultdf.to_csv(file, index=False, na_rep="Null")

    def update_tflops_bw(self, file):
        resultdf = self.get_tuned_gemm_list(file)
        for i in range(len(resultdf)):
            if len(resultdf.loc[i]) == 8:
                *keys, kernelId, splitK, us, kernelName = tuple(resultdf.loc[i])
            else:
                (
                    *keys,
                    kernelId,
                    splitK,
                    us,
                    kernelName,
                    tflops,
                    bw,
                    errRatio,
                ) = resultdf.iloc[i]
            errRatio = 0
            keys = tuple(keys)
            info = (keys, kernelId, splitK, ""), us, errRatio
            tflops, bw = self.calculate(info)
            resultdf.loc[i, "tflops"] = tflops
            resultdf.loc[i, "bw"] = bw
            resultdf.loc[i, "errRatio"] = 0
        resultdf.to_csv(file, index=False, na_rep="Null")
