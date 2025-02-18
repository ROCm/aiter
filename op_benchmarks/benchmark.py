# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from typing import Callable
import torch
import torch.profiler as tpf
import os
import numpy as np
import pandas as pd
from aiter import logger



def get_trace_perf(prof, num_iters):
    assert (num_iters > 1)
    num_iters -= 1
    df = []
    cols = ['name', 'self_cpu_time_total', 'self_device_time_total',
            'device_type', 'device_index',]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    df['cnt'] = 1
    rets = []
    for name, d in df.groupby('name', sort=False):
        r = d.iloc[1:][['cnt',
                        'self_cpu_time_total',
                        'self_device_time_total']].sum()
        if not r.empty:
            device_type = str(d['device_type'].iat[0]).split('.')[-1]
            r['name'] = name
            r['device_type'] = device_type
            r['device_index'] = str(d['device_index'].iat[0])
            if device_type == 'CUDA':
                r['device_time_total'] = r['self_device_time_total']
                r['host_time_total'] = 0
            else:
                r['host_time_total'] = r['self_device_time_total']
                r['device_time_total'] = 0

        rets.append(r)
    df = pd.DataFrame(rets)

    cols = ['name', 'cnt', 'host_time_total', 'device_time_total',
            'device_type', 'device_index',]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_total > 0) | (df.device_time_total > 0)]

    timerList = ['host_time_total', 'device_time_total', ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    avg_name = '[avg us/iter]'
    for el in timerList:
        df.at[avg_name, el] = df[el].sum()/num_iters
    if int(os.environ.get('AITER_LOG_MORE', 0)):
        logger.info(f'{df}')
    return df.at[avg_name, 'device_time_total']

def execute_callback(num_iterations: int, func: Callable, *args, **kwargs) -> None:
    for _ in range(num_iterations):
        func(*args, **kwargs)

def profile(
    num_iterations: int,
    num_warmup_iterations:int,
    func: Callable,
    *args,
    **kwargs
):
    #  warmup
    execute_callback(num_warmup_iterations, func, *args, **kwargs)
    with tpf.profile(
        activities=[
            tpf.ProfilerActivity.CPU,
            tpf.ProfilerActivity.CUDA
        ],
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        execute_callback(func, *args, **kwargs)

    avg = get_trace_perf(prof, num_iterations)

def profile_cuda_graph(
    num_iterations: int,
    num_warmup_iterations:int, func: Callable, *args, **kwargs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute_callback(1, func, *args, **kwargs)
    profile(num_iterations, num_warmup_iterations, func, *args, **kwargs)
