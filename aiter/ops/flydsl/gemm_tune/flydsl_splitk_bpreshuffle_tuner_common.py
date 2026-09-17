# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``GemmCommonTuner`` scaffolding shared by the FlyDSL split-K bpreshuffle
tuners -- ``gemm_a8w8_blockscale_bpreshuffle_tune.py`` (blockscale + mx128)
and ``gemm_a4w4_blockscale_bpreshuffle_tune.py`` (mxfp4). Both subclasses'
``tune()``/``result_to_df()`` bodies were byte-identical (~100 lines) except
for which per-shape task-getter method(s) ``tune()`` calls -- one for a4w4,
two for a8w8. That is the only axis of variation, so it is the only thing a
subclass still overrides here: ``_tune_task_getter_names()``, returning the
ordered tuple of its own ``get_flydsl_splitk_*_tune_task`` method names.
Everything else (task-loop shape, ``mp_tuner`` invocation, result-row
assembly) lives once, in ``FlydslSplitKBpreshuffleTuner`` below.

Each subclass's data-gen/run functions and its task-getter method bodies stay
in its own tuner script -- this module only owns the loop that calls them.
"""

import pandas as pd

from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner


class FlydslSplitKBpreshuffleTuner(GemmCommonTuner):
    def _setup_specific_arguments(self):
        """No extra flags: this tuner sweeps exactly one pipeline/libtype."""

    def _tune_task_getter_names(self) -> tuple[str, ...]:
        """Ordered ``get_flydsl_splitk_*_tune_task`` method names this family
        calls per shape. Subclasses override; the base has no tasks of its
        own to contribute."""
        return ()

    def tune(self, untunedf, tunedf, args):
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 0
        getters = [getattr(self, name) for name in self._tune_task_getter_names()]
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            seed = seed + 1
            prev_task_count = len(task)
            info_keys = (gfx, cu_num, M, N, K)
            for getter in getters:
                task.extend(getter(info_keys, seed))
            shape_kernel_nums = len(task) - prev_task_count
            tasks_data.append((shape_kernel_nums, ()))
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

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME
                else (self.getKernelName(kernelId) if kernelName == "" else kernelName)
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))

            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} {kernelName} {splitK=}, {time}us, {err_ratio=}, {tflops=} TFLOPS, {bw=} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
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
            if resultdf.empty:
                resultdf = temp
            else:
                resultdf = pd.concat([resultdf, temp], ignore_index=True)
        return resultdf
