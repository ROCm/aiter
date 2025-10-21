# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import random

import aiter
from aiter.test_common import checkAllclose, benchmark, perftest, run_perftest

max_model_len = 111 * 1000
# for batch_size, next_n in [(64, 1), (64, 2), (128, 1)]:
for batch_size, next_n in [(1, 1), (64, 2), (128, 1)]:
    for heads, index_dim in [(64, 128)]:
        for avg_kv in (8192, 32768):

                num_blocks, blocksize = max_model_len * 3, 64

                q = torch.randn((batch_size, next_n, heads, index_dim), device='cuda', dtype=torch.bfloat16)
                kv_cache = torch.randn((num_blocks, blocksize, 1, index_dim), device='cuda', dtype=torch.bfloat16)
                weights = torch.randn((batch_size * next_n, heads), device='cuda', dtype=torch.float32)

                context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (batch_size, )).cuda().to(torch.int32)
                max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                block_tables = torch.zeros((batch_size, max_block_len), device='cuda', dtype=torch.int32)

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = context_lens[i].item()
                    for j in range((ctx_len + blocksize - 1) // blocksize):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                # q_fp8 = q.to(torch.float8_e4m3fnuz)
                # kv_cache = 
                gpu = torch.cuda.current_device()
                device_properties = torch.cuda.get_device_properties(gpu)
                cu_num = device_properties.multi_processor_count

                schedule_metadata = aiter.get_paged_mqa_logits_metadata(context_lens, blocksize, cu_num)

                import pdb; pdb.set_trace()
