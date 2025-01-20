# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: test_common.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-03 15:53:32
# @Last Modified By: valarLip
# @Last Modified At: 2025-01-17 10:02:34
# @Description: This is description.

import torch
import torch.profiler as tpf
import os
import numpy as np
import pandas as pd
from ater import logger


def perftest(num_iters=101, num_warmup=10, testGraph=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_warmup):
                data = func(*args, **kwargs)

            if int(os.environ.get('ATER_LOG_MORE', 0)):
                latencies = []
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(num_iters):
                    start_event.record()
                    data = func(*args, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                avg = np.mean(latencies) * 1000
                logger.info(f'avg: {avg} us/iter from cuda.Event')

            if testGraph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    data = func(*args, **kwargs)
                with tpf.profile(activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                                 profile_memory=True,
                                 with_stack=True,
                                 with_modules=True,
                                 ) as prof:
                    run_iters(num_iters, graph.replay)
                avg = get_trace_perf(prof, num_iters)
                logger.info(f'avg: {avg} us/iter with hipgraph')
            with tpf.profile(activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                             profile_memory=True,
                             with_stack=True,
                             with_modules=True,
                             #  record_shapes=True,
                             #  on_trace_ready=tpf.tensorboard_trace_handler(
                             #      './ater_logs/'),
                             ) as prof:
                data = run_iters(num_iters, func, *args, **kwargs)
            avg = get_trace_perf(prof, num_iters)
            return data, avg
        return wrapper
    return decorator


def run_iters(num_iters, func, *args, **kwargs):
    for _ in range(num_iters):
        data = func(*args, **kwargs)
    return data


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
    if int(os.environ.get('ATER_LOG_MORE', 0)):
        logger.info(f'{df}')
    return df.at[avg_name, 'device_time_total']


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg=''):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        logger.info(f'{msg}[checkAllclose {atol=} {rtol=} passed~]')
    else:
        percent = (a[mask]).numel()/a.numel()
        delta = (a-b)[mask]
        if percent > 0.01:
            logger.info(f'''{msg}[checkAllclose {atol=} {rtol=} failed!]
        a:  {a.shape}
            {a[mask]}
        b:  {b.shape}
            {b[mask]}
    dtlta:
            {delta}''')
        else:
            logger.info(
                f'''{msg}[checkAllclose {atol=} {rtol=} waring!] a and b results are not all close''')
        logger.info(
            f'-->max delta:{delta.max()}, delta details: {percent:.1%} ({(a[mask]).numel()} of {a.numel()}) elements')


def tensor_dump(x: torch.tensor, name: str, dir='./'):
    x_cpu = x.cpu().view(torch.uint8)
    filename = f'{dir}/{name}.bin'
    x_cpu.numpy().tofile(filename)
    logger.info(f'saving {filename} {x.shape}, {x.dtype}')

    with open(f'{dir}/{name}.meta', 'w') as f:
        f.writelines([f'{el}\n' for el in [x.shape, x.dtype]])


def tensor_load(filename: str):
    DWs = np.fromfile(filename, dtype=np.uint32)
    metafile = '.'.join(filename.split('.')[:-1])+'.meta'
    shape, dtype = [eval(line.strip()) for line in open(metafile)]
    return torch.tensor(DWs).view(dtype).view(shape)
