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

INVALID_TIME = -1


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
        self.success = pd.DataFrame(columns=self.columns)
        self.failed = pd.DataFrame(columns=self.columns)

    def get_arg_defaults(self):
        """get default arguments"""
        return self.ARG_DEFAULTS.copy()

    def get_bpe(self, dtype):
        return self.dtype2bpe_dict[dtype]

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
            "--all",
            action="store_true",
            required=False,
            help="retune all shapes in tune_file if tune file and untune file are the same, or retune shapes in untune file if tune file and untune file are different",
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

    @abstractmethod
    def result_to_df(self, rets):
        """transfer results to dataframe"""
        pass

    def get_untuned_gemm_list(self, untuned_gemm_file):
        assert os.path.exists(
            untuned_gemm_file
        ), f"Not exist untuned file: {untuned_gemm_file}"
        untunedf = pd.read_csv(untuned_gemm_file)
        filtered_df = untunedf.drop_duplicates().reset_index(drop=True)
        return filtered_df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=[]):
        if os.path.exists(tuned_gemm_file):
            column_order = pd.read_csv(tuned_gemm_file, nrows=0).columns.tolist()
            tunedf = pd.read_csv(tuned_gemm_file)
            tunedf = tunedf[column_order]
        else:
            print(f"Not exist tuned file: {tuned_gemm_file}")
            columns = self.columns if not columns else columns
            tunedf = pd.DataFrame(columns=columns)
        return tunedf

    def get_retune_gemm_list(self, args):
        """get retune gemm list from tune_file and untune_file"""
        if args.untune_file is None:
            raise ValueError("untune_file must be specified for retuning")
        if args.tune_file == args.untune_file:
            # retune all shapes in tune_file
            self.untunedf = self.get_untuned_gemm_list(args.untune_file)
            self.tunedf = self.untunedf[self.untunedf["cu_num"] != self.get_cu_num()]
            self.untunedf = self.untunedf[self.untunedf["cu_num"] == self.get_cu_num()]
            self.untunedf = self.untunedf[self.keys]
        else:
            # retune shapes that are in both untune_file and tune_file
            untunedf = self.get_untuned_gemm_list(args.untune_file)
            if "cu_num" not in untunedf.columns:
                untunedf["cu_num"] = self.get_cu_num()
            else:
                untunedf = untunedf[untunedf["cu_num"] == self.get_cu_num()]
            self.untunedf = untunedf[self.keys]
            self.tunedf = self.get_tuned_gemm_list(args.tune_file)

            untunedf_cols = self.untunedf.columns
            mask = (
                self.tunedf[untunedf_cols]
                .apply(tuple, axis=1)
                .isin(self.untunedf[untunedf_cols].apply(tuple, axis=1))
            )
            self.tunedf = self.tunedf[~mask]

    def update_tunedf(self, df_old, df_updates):
        """update tuned result to old df"""
        """ for shapes already tuned, we update the result inplace"""
        key_columns = self.keys
        df_updates = df_updates.loc[:, self.columns]
        # print(df_updates)
        df_old["_tmp_key"] = df_old[key_columns].apply(tuple, axis=1)
        df_updates["_tmp_key"] = df_updates[key_columns].apply(tuple, axis=1)
        matched_keys = df_updates[df_updates["_tmp_key"].isin(df_old["_tmp_key"])][
            "_tmp_key"
        ].tolist()
        unmatched_keys = df_updates[~df_updates["_tmp_key"].isin(df_old["_tmp_key"])][
            "_tmp_key"
        ].tolist()
        for key in matched_keys:
            df_old.loc[df_old.index[df_old["_tmp_key"] == key][0]] = df_updates.loc[
                df_updates["_tmp_key"] == key
            ].values[0]
        if unmatched_keys:
            unmatched_rows = df_updates[
                df_updates["_tmp_key"].isin(unmatched_keys)
            ].copy()
            df_old = pd.concat([df_old, unmatched_rows], ignore_index=True)
        df_old.drop("_tmp_key", axis=1, inplace=True)
        df_updates.drop("_tmp_key", axis=1, inplace=True)
        return df_old

    def sortResults(self, tune_file, issorted, values):
        tunedf = pd.read_csv(tune_file)
        if issorted:
            tunedf = tunedf.sort_values(by=values)
        tunedf = tunedf.drop_duplicates(
            subset=self.keys,
            keep="last",
        )
        # print(tunedf)
        tunedf.to_csv(tune_file, index=False, na_rep="Null")

    def get_cu_num(self):
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        return cu_num

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        """post process, post process all results to return topk results"""
        rets = list(rets)
        if args.profile_file != "":
            profiledf = self.result_to_df(sorted(rets, key=itemgetter(0)))
            profiledf.to_csv(args.profile_file, index=False, na_rep="Null")
            # self.result_to_csv(sorted(rets, key=itemgetter(0)), args.profile_file, True)
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
                if max_err_ratio <= tol_err_ratio
                and us != INVALID_TIME
                and us != float("inf")
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
                best_config = [((info_key, *sorted_time[0][0]), INVALID_TIME, 1.0)]

            bestConfigs.extend(best_config)
        resultdf = self.result_to_df(bestConfigs)
        return resultdf

    @abstractmethod
    def result_to_csv(self, results, file, concat=False):
        """write result to csv file, all means concat all results to file"""
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
            self.sortResults(args.tune_file, args.sort, self.keys)
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
                    self.result_to_csv(results, args.tune_file, not args.all)
                    logger.info(
                        f"processed {processed_batches} batches of {total_batches}, Processing Status ====> {round(processed_batches / total_batches,2)*100:.1f}% tuned in {self.name}"
                    )
                else:
                    logger.info("tune result is none or all shape is tuned!")
            logger.info(
                f"Tuning finished. tune {len(self.untunedf)} shapes, total tuning time is {round(time.time() - start_time,4)} seconds"
            )
            self.sortResults(args.tune_file, args.sort, self.keys)
        except KeyboardInterrupt:
            logger.error(
                f"interrupted by user, tuning stopped, {processed_batches-1} batches processed"
            )
        except Exception as e:
            logger.error(
                f"error in batch {processed_batches} of {total_batches}: {str(e)}",
                exc_info=True,
            )
        finally:
            logger.info("============= Tuning results Summary: ==============")
            logger.info("Successfully tuned shapes:")
            print(self.success)
            logger.info("Failed tuned shapes:")
            print(self.failed)


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
            "tflops",
            "bw",
            "errRatio",
        ],
        description=None,
    ):
        super().__init__(name, key, resultList, description)

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
        else:
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

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName = info
            kernelName = (
                "None"
                if time == INVALID_TIME
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
        return resultdf

    def result_to_csv(self, resultdf, file, concat=False):
        """post process of tuning results"""
        old_df = self.get_tuned_gemm_list(file)
        self.failed = pd.concat(
            [self.failed, resultdf[resultdf["us"] == INVALID_TIME]], ignore_index=True
        )
        self.success = pd.concat(
            [self.success, resultdf[resultdf["us"] != INVALID_TIME]], ignore_index=True
        )
        update_tunedf = self.success
        if not concat:
            resultdf = self.update_tunedf(old_df, update_tunedf)
        else:
            resultdf = pd.concat([old_df, update_tunedf], ignore_index=True)
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
