# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes
import random
import itertools
import argparse
import math
import os
import yaml

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False, threshold=torch.inf)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)


class YamlRecorder:
    def __init__(self, type):
        self.yaml_db = []
        self.curr = {}
        self.type = type

    def update_curr(
        self,
        seq_lens_kv,
        num_splits,
        workloads,
        workload_limit_global,
        default_workload_limit_global,
        avg_time_main,
        avg_time_reduce,
    ):
        if self.curr and num_splits == self.curr["num_splits"]:
            self.curr["workload_limit_global"][0] = min(
                self.curr["workload_limit_global"][0], workload_limit_global
            )
            self.curr["workload_limit_global"][1] = max(
                self.curr["workload_limit_global"][1], workload_limit_global
            )
            self.curr["avg_time_main"] = min(self.curr["avg_time_main"], avg_time_main)
            self.curr["avg_time_reduce"] = min(
                self.curr["avg_time_reduce"], avg_time_reduce
            )
        elif not self.curr or (
            (self.curr["avg_time_main"] + self.curr["avg_time_reduce"])
            >= (avg_time_main + avg_time_reduce)
        ):
            self.curr["type"] = self.type
            self.curr["seq_lens_kv"] = seq_lens_kv
            self.curr["num_splits"] = num_splits
            self.curr["workloads"] = workloads
            self.curr["workload_limit_global"] = [
                workload_limit_global,
                workload_limit_global,
            ]
            self.curr["default_workload_limit_global"] = default_workload_limit_global
            self.curr["avg_time_main"] = avg_time_main
            self.curr["avg_time_reduce"] = avg_time_reduce

    def update_range(self, num_splits, workload_limit_global):
        if self.curr and self.curr["num_splits"] == num_splits:
            self.curr["workload_limit_global"][0] = min(
                self.curr["workload_limit_global"][0], workload_limit_global
            )
            self.curr["workload_limit_global"][1] = max(
                self.curr["workload_limit_global"][1], workload_limit_global
            )

    def flush(self):
        if self.curr:
            self.yaml_db.append(self.curr)
        self.curr = {}


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    # if use_fp8:
    #     assert cos_diff < 3e-2
    # else:
    #     assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8=False,
    q_scale=None,
    kv_scale=None,
) -> torch.Tensor:

    if is_fp8:
        scale *= q_scale * kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)

    m = attn_weights.max(-1).values

    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))

    l = attn_weights_exp.sum(-1)

    if is_fp8:
        attn_weights_fp8 = attn_weights_exp.to(torch.float8_e4m3fnuz)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())

    out = out / l.transpose(0, 1).unsqueeze(-1)

    if is_fp8:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    is_fp8 = q.dtype == torch.float8_e4m3fnuz

    if is_fp8:
        q = q.to(torch.float)
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8=is_fp8,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


@benchmark()
def test_mla(
    ctx_lens,
    ctx_lens_lb,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    mtp,
    bf16_db: YamlRecorder = None,
    fp8_db: YamlRecorder = None,
):
    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_kv[i] = random.uniform(ctx_lens_lb, ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)

    kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    kv_indices = torch.randint(0, num_page, (kv_indptr[-1].item(),), dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    max_seqlen_kv = seq_lens_kv.max().item()
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=kvtype,
    )

    # for none absorb (mha)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    us_asm = None
    # if batch_size * ctx_lens * nhead < 32 * 8192 * 16:
    #     us_asm = test_absorb_prefill()
    torch.cuda.empty_cache()
    nhead_kv = 1

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and mtp != 1:
    #     return
    seq_lens_qo.fill_(mtp)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=dtype)

    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    # aiter implementation
    # the tensor's meaning please refer aiter/ops/attention.py
    work_meta_data = torch.empty([10], dtype=torch.uint64, device="cuda")
    work_indptr = torch.empty([cu_num + 1], dtype=torch.int32, device="cuda")
    work_info_set = torch.empty(
        [batch_size * cu_num, 8], dtype=torch.int32, device="cuda"
    )
    reduce_indptr = torch.empty([batch_size + 1], dtype=torch.int32, device="cuda")
    reduce_final_map = torch.empty([batch_size, 2], dtype=torch.int32, device="cuda")
    reduce_partial_map = torch.empty(
        [batch_size * cu_num], dtype=torch.int32, device="cuda"
    )

    # [0]: fixed workload_limit_global. only valid when the fixed value is larger than 0.
    metadata_test_params = torch.tensor(
        [-1, -1, -1, -1], dtype=torch.int32, device="cuda"
    )
    # [0.0]: actual workload_limit_global
    # [0.1]: whether the ugly code is touched
    # [1]: #splits for each batch
    # [2]: workload for each cu
    metadata_test_outputs = torch.empty(
        [3, max(2, batch_size, cu_num)], dtype=torch.int32, device="cuda"
    )

    split_params = {
        "kv_granularity": max(page_size, 16),
        "max_seqlen_qo": max_seqlen_qo,
    }

    meta = aiter.get_mla_metadata_v1_tunable(
        qo_indptr,
        kv_indptr,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        metadata_test_params,
        metadata_test_outputs,
        split_params=split_params,
    )

    # valid_work_cnt = 0
    # for i in range(batch_size * 80):
    #     bid = work_info_set[i][0].item()
    #     if bid >= batch_size or bid < 0:
    #         break
    #     valid_work_cnt = i + 1
    # valid_reduce_partial_cnt = 0
    # for i in range(batch_size * 80):
    #     idx = reduce_partial_map[i].item()
    #     if idx >= 80 * qo_indptr[-1].item() * nhead or idx < 0:
    #         break
    #     valid_reduce_partial_cnt = i + 1
    #     if idx == reduce_partial_map[-1].item():
    #         break
    # print(f"seq_lens_kv({seq_lens_kv.shape}):")
    # print(seq_lens_kv)
    # print(f"kv_indptr({kv_indptr.shape}):")
    # print(kv_indptr)
    # print(f"work_indptr({work_indptr.shape}):")
    # print(work_indptr)
    # print(f"work_info_set({work_info_set.shape}.{valid_work_cnt}):")
    # print(work_info_set[:valid_work_cnt])
    # print(f"reduce_indptr({batch_size + 1}):")
    # print(reduce_indptr[: batch_size + 1])
    # print(f"reduce_final_map({batch_size}):")
    # print(reduce_final_map[:batch_size])
    # print(f"reduce_partial_map({reduce_partial_map.shape}.{valid_reduce_partial_cnt}):")
    # print(reduce_partial_map[:valid_reduce_partial_cnt])
    # print("metadata_test_outputs[1] - #splits for each batch:")
    # print(metadata_test_outputs[1][:batch_size])
    # print("metadata_test_outputs[2] - workload for each cu:")
    # print(metadata_test_outputs[2][:cu_num])
    # print(f"workload_limit_global: {metadata_test_outputs[0][0].item()}")

    default_workload_limit_global = metadata_test_outputs[0][0].item()
    workload_limit_global_min__ = max(
        int(math.ceil(default_workload_limit_global / 4 / 16) * 16), 16
    )
    workload_limit_global_min = min(workload_limit_global_min__, 32)
    workload_limit_global_max = max(
        default_workload_limit_global * 1.5,
        total_kv + batch_size * 128 * (cu_num + 1) / cu_num,
    )
    workload_limit_global_max = int(math.ceil(workload_limit_global_max / 16) * 16) + 16
    print(
        f"[metadata-autotune] workload_limit_global[default, min, max]=[{default_workload_limit_global}, {workload_limit_global_min}, {workload_limit_global_max}]"
    )

    def test_absorb_decode(check_quality):
        # troch implementation
        if check_quality:
            out_ref, lse_ref = torch_mla_extend(
                q,
                kv_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                sm_scale,
                kv_lora_rank,
                qk_rope_head_dim,
                is_causal=True,
                dtype=dtype,
            )

        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=dtype).fill_(-1)

        (attn_logits, attn_lse), us_asm_decode, avg_prof = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        avg_time_main = 0.0
        avg_time_reduce = 0.0
        for el in avg_prof:
            if "aiter::mla_" in el.key:
                avg_time_main = el.device_time
            elif "kn_mla_reduce_v1" in el.key:
                avg_time_reduce = el.device_time

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
        bytes = (
            total_kv * nhead_kv * qk_head_dim
            + total_q * nhead * (qk_head_dim + v_head_dim)
        ) * (torch.finfo(dtype).bits // 8)
        err = True
        if check_quality:
            err = checkAllclose(
                out_ref,
                out_asm,
                msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )
        return err, us_asm_decode, avg_time_main, avg_time_reduce

    def test_absorb_decode_fp8(check_quality):
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=dtype).fill_(-1)

        q_fp8, q_scale = aiter.per_tensor_quant(q, quant_dtype=torch.float8_e4m3fnuz)
        q_scale = q_scale.to(torch.float)

        kv_buffer_fp8 = kv_buffer.to(torch.float8_e4m3fnuz)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        if check_quality:
            out_ref_fp8, lse_ref_fp8 = torch_mla_extend(
                q_fp8,
                kv_buffer_fp8,
                qo_indptr,
                kv_indptr,
                kv_indices,
                sm_scale,
                kv_lora_rank,
                qk_rope_head_dim,
                dtype=dtype,
                is_causal=True,
                q_scale=q_scale,
                kv_scale=kv_scale,
            )

        (attn_logits, attn_lse), us_asm_decode, avg_prof = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        avg_time_main = 0.0
        avg_time_reduce = 0.0
        for el in avg_prof:
            if "aiter::mla_" in el.key:
                avg_time_main = el.device_time
            elif "kn_mla_reduce_v1" in el.key:
                avg_time_reduce = el.device_time

        if check_quality:
            cal_diff(out_ref, out_asm, "out", True)

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
        bytes = (
            total_kv * nhead_kv * qk_head_dim
            + total_q * nhead * (qk_head_dim + v_head_dim)
        ) * (torch.finfo(dtype).bits // 8)

        err = True
        # err = checkAllclose(
        #     out_ref,
        #     out_asm,
        #     msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        # )
        err_fp8 = True
        if check_quality:
            err_fp8 = checkAllclose(
                out_ref_fp8,
                out_asm,
                msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )
        return err, err_fp8, us_asm_decode, avg_time_main, avg_time_reduce

    err, us_asm_decode, avg_time_bf16_main, avg_time_bf16_reduce = (
        test_absorb_decode(False) if bf16_db is not None else (0, 1, 1, 1)
    )
    (
        err_fp8_fp32,
        err_fp8_fp8,
        us_asm_decode_fp8,
        avg_time_fp8_main,
        avg_time_fp8_reduce,
    ) = (
        test_absorb_decode_fp8(False) if fp8_db is not None else (0, 0, 1, 1, 1)
    )

    last_num_splits = None
    test_workload_limit_global = workload_limit_global_min
    stride = 16
    while test_workload_limit_global <= workload_limit_global_max:
        if metadata_test_params[0].item() != test_workload_limit_global:
            metadata_test_params[0] = test_workload_limit_global
            meta = aiter.get_mla_metadata_v1_tunable(
                qo_indptr,
                kv_indptr,
                nhead // nhead_kv,
                nhead_kv,
                True,
                work_meta_data,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                metadata_test_params,
                metadata_test_outputs,
                split_params=split_params,
            )

        num_splits = metadata_test_outputs[1][:batch_size].tolist()
        workloads = metadata_test_outputs[2][:cu_num].tolist()
        workload_limit_global = metadata_test_outputs[0][0].item()
        touch_ugly = metadata_test_outputs[0][1].item() != 0

        # print(f"[metadata-dbg] workload_limit_global={test_workload_limit_global}, touch_ugly={touch_ugly}, same_splits={last_num_splits == num_splits}")

        if (
            last_num_splits is not None and last_num_splits == num_splits
        ) or touch_ugly:
            if bf16_db is not None:
                bf16_db.update_range(num_splits, workload_limit_global)
            if fp8_db is not None:
                fp8_db.update_range(num_splits, workload_limit_global)
            last_test_workload_limit_global = test_workload_limit_global
            test_workload_limit_global = test_workload_limit_global + stride
            if last_test_workload_limit_global < workload_limit_global_max:
                test_workload_limit_global = min(
                    test_workload_limit_global, workload_limit_global_max
                )
            stride *= 2
            continue
        elif stride > 16:
            test_workload_limit_global = last_test_workload_limit_global + 16
            stride = 16
            continue

        # print(f"test_workload_limit_global={test_workload_limit_global}, num_splits={num_splits}")
        last_num_splits = num_splits
        last_test_workload_limit_global = test_workload_limit_global

        if touch_ugly:
            pass
            # print(f"[metadata-dbg] WARNING: touched ugly!")
            # print(f"[metadata-dbg]   workload_limit_global={test_workload_limit_global}, num_splits={num_splits}")
        else:
            (err, us_asm_decode, avg_time_bf16_main, avg_time_bf16_reduce) = (
                0,
                1,
                1,
                1,
            )
            (
                err_fp8_fp32,
                err_fp8_fp8,
                us_asm_decode_fp8,
                avg_time_fp8_main,
                avg_time_fp8_reduce,
            ) = (0, 0, 1, 1, 1)

            if bf16_db is not None:
                (err, us_asm_decode, avg_time_bf16_main, avg_time_bf16_reduce) = (
                    test_absorb_decode(False)
                )

                bf16_db.update_curr(
                    seq_lens_kv.tolist(),
                    num_splits,
                    workloads,
                    workload_limit_global,
                    [
                        default_workload_limit_global,
                        workload_limit_global_min,
                        workload_limit_global_max,
                    ],
                    avg_time_bf16_main,
                    avg_time_bf16_reduce,
                )

            if fp8_db is not None:
                (
                    err_fp8_fp32,
                    err_fp8_fp8,
                    us_asm_decode_fp8,
                    avg_time_fp8_main,
                    avg_time_fp8_reduce,
                ) = test_absorb_decode_fp8(False)

                fp8_db.update_curr(
                    seq_lens_kv.tolist(),
                    num_splits,
                    workloads,
                    workload_limit_global,
                    [
                        default_workload_limit_global,
                        workload_limit_global_min,
                        workload_limit_global_max,
                    ],
                    avg_time_fp8_main,
                    avg_time_fp8_reduce,
                )

            # print(f"[metadata-autotune] workload_limit_global={test_workload_limit_global}, num_splits={num_splits}, bf16 times: {avg_time_bf16_main} + {avg_time_bf16_reduce} = {avg_time_bf16_main+avg_time_bf16_reduce}, fp8 times: {avg_time_fp8_main} + {avg_time_fp8_reduce} = {avg_time_fp8_main+avg_time_fp8_reduce}")

    # print(f"{out_ref.view(total_q, -1)=}")
    # print(f"{out_asm.view(total_q, -1)=}")
    # checkAllclose(logits_ref, attn_logits,
    #               msg=f'attn_logits [golden vs aiter_asm]')
    # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
    flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim + total_q * nhead * (qk_head_dim + v_head_dim)
    ) * (torch.finfo(dtype).bits // 8)

    if bf16_db is not None:
        print(
            f"[metadata-autotune] bf16 best choice: workload_limit_global={bf16_db.curr['workload_limit_global']}, num_splits={bf16_db.curr['num_splits']}"
        )
    if fp8_db is not None:
        print(
            f"[metadata-autotune] fp8 best choice: workload_limit_global={fp8_db.curr['workload_limit_global']}, num_splits={fp8_db.curr['num_splits']}"
        )

    return {
        "decode:flops": flops,
        "decode:bytes": bytes,
        "decode:err": err,
        "decode:asm_576": us_asm_decode,
        "decode:TFLOPS": flops / us_asm_decode / 1e6,
        "decode:TB/s": bytes / us_asm_decode / 1e6,
        "decode_fp8:err vs fp32": err_fp8_fp32,
        "decode_fp8:err vs fp8": err_fp8_fp8,
        "decode_fp8:asm_576": us_asm_decode_fp8,
        "decode_fp8:TFLOPS": flops / us_asm_decode_fp8 / 1e6,
        "decode_fp8:TB/s": bytes / us_asm_decode_fp8 / 1e6,
    }


def DumpYamlFile(db, path, type: str):
    filename = f"metadata_autotune_{type}.yaml"
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as outfile:
        yaml.dump(db, outfile, default_flow_style=None)


kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
block_size = 1
list_dtype = ["bf16"]
l_kv_dtype = ["bf16"]
list_nhead = [(16, 2)]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=512,
    help="""qk nope head dim.
    e.g.: -qn 512""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=512,
    help="""v head dim.
    e.g.: -vh 512""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["bf16"],
    nargs="*",
    default=["bf16"],
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=str,
    choices=["bf16"],
    nargs="*",
    default=["bf16"],
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    # default=[28, 512, 1023, 4888, 12800][1:],  #
    default=[
        35,
        48,
        61,
        78,
        79,
        80,
        81,
        82,
        101,
        241,
        412,
        512,
        777,
        1023,
        2333,
        4888,
        12800,
    ],
    # default=[78],
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    # default=[i for i in range(1, 320, 1)],  # [41],
    default=[i for i in range(1, 11, 1)]
    + [i for i in range(11, 31, 2)]
    + [i for i in range(31, 71, 3)]
    + [i for i in range(71, 91, 1)]
    + [i for i in range(91, 151, 5)]
    + [i for i in range(151, 251, 7)]
    + [i for i in range(251, 331, 11)],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Number of heads.
    e.g.: -n 16,1""",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="",
    help="Output database file in YAML.",
)
parser.add_argument("-t", "--type", type=str, default="all", help="all, bf16, or fp8")

import pandas as pd

args = parser.parse_args()
list_dtype = [dtypes.d_dtypes[key] for key in args.dtype]
l_kv_dtype = [dtypes.d_dtypes[key] for key in args.kv_dtype]
if args.nhead is not None:
    list_nhead = [args.nhead]

for nhead, mtp in list_nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size in itertools.product(
        list_dtype, l_kv_dtype, args.ctxLen, args.batchSize
    ):
        bf16_db = YamlRecorder("bf16")
        fp8_db = YamlRecorder("fp8")

        for ctx_len_lb in (
            5,
            int(ctx_len / 4),
            int(ctx_len / 2),
            int(ctx_len * 3 / 4),
            ctx_len,
        ):
            ret = test_mla(
                ctx_len,
                ctx_len_lb,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                varlen=True,
                mtp=mtp,
                bf16_db=bf16_db,
                fp8_db=fp8_db,
            )
            df.append(ret)
            bf16_db.flush()
            fp8_db.flush()

        bs_str = f"b{str(batch_size).zfill(4)}_s{str(ctx_len).zfill(6)}"
        DumpYamlFile(bf16_db.yaml_db, args.output, bs_str + "_bf16")
        DumpYamlFile(fp8_db.yaml_db, args.output, bs_str + "_fp8")

    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}mtp{mtp}.csv")
    aiter.logger.info(f"summary:\n{df}")
