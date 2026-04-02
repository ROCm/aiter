import torch

import aiter


def make_uniform_indptr(batch_size, seqlen, device="cuda"):
    return torch.arange(
        0,
        (batch_size + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device=device,
    )


def alloc_metadata_tensors(
    batch_size,
    max_seqlen_qo,
    max_seqlen_kv,
    q_tile_size,
    split_k_size,
    is_causal,
    uni_seqlen_qo,
    device="cuda",
):
    (
        (work_metadata_ptrs_size, work_metadata_ptrs_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mha_prefill_splitk_metadata_info_v1(
        batch_size=batch_size,
        max_seqlen_qo=max_seqlen_qo,
        max_seqlen_kv=max_seqlen_kv,
        q_tile_size=q_tile_size,
        split_k_size=split_k_size,
        is_causal=is_causal,
        uni_seqlen_qo=uni_seqlen_qo,
    )

    return (
        torch.empty(work_metadata_ptrs_size, dtype=work_metadata_ptrs_type, device=device),
        torch.empty(work_indptr_size, dtype=work_indptr_type, device=device),
        torch.empty(work_info_set_size, dtype=work_info_set_type, device=device),
        torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device),
        torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device),
        torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=device),
    )


def to_work_list(work_info_set, valid_work_num):
    rows = work_info_set[:valid_work_num].cpu().tolist()
    return [
        {
            "batch_idx": row[0],
            "partial_qo_loc": row[1],
            "qo_start": row[2],
            "qo_end": row[3],
            "kv_start": row[4],
            "kv_end": row[5],
            "kv_offset": row[6],
        }
        for row in rows
    ]


def reference_stage1_single_work(q, k, v, work_item, softmax_scale):
    qo_start = work_item["qo_start"]
    qo_end = work_item["qo_end"]
    kv_start = work_item["kv_start"]
    kv_end = work_item["kv_end"]

    q_tile = q[qo_start:qo_end, 0, :].float()
    k_tile = k[kv_start:kv_end, 0, :].float()
    v_tile = v[kv_start:kv_end, 0, :].float()

    scores = q_tile @ k_tile.transpose(0, 1)
    scores = scores * softmax_scale
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v_tile
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse, scores


def test_mha_prefill_splitk_stage1_opus_smoke():
    device = "cuda"
    batch_size = 1
    q_len = 128
    kv_len = 128
    q_tile_size = 128
    split_k_size = 32
    num_heads = 1
    num_kv_heads = 1
    head_dim = 128

    qo_indptr = make_uniform_indptr(batch_size, q_len, device=device)
    kv_indptr = make_uniform_indptr(batch_size, kv_len, device=device)

    (
        work_metadata_ptrs,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = alloc_metadata_tensors(
        batch_size=batch_size,
        max_seqlen_qo=q_len,
        max_seqlen_kv=kv_len,
        q_tile_size=q_tile_size,
        split_k_size=split_k_size,
        is_causal=True,
        uni_seqlen_qo=q_len,
        device=device,
    )

    aiter.get_mha_prefill_splitk_metadata_v1(
        qo_indptr,
        kv_indptr,
        True,
        q_tile_size,
        split_k_size,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_qo=q_len,
        uni_seqlen_qo=q_len,
    )

    valid_work_num = int(work_indptr.cpu()[1].item())
    work_list = to_work_list(work_info_set, valid_work_num)
    partial_tokens = max(
        item["partial_qo_loc"] + (item["qo_end"] - item["qo_start"])
        for item in work_list
    )

    q = torch.randn((q_len, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((kv_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn((kv_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    split_o = torch.empty((partial_tokens, num_heads, head_dim), dtype=torch.float32, device=device)
    split_lse = torch.empty((partial_tokens, num_heads), dtype=torch.float32, device=device)
    debug_qk_scores = torch.empty((valid_work_num, num_heads, 4, 32, 32), dtype=torch.float32, device=device)

    aiter.mha_prefill_splitk_stage1_opus(
        q,
        k,
        v,
        kv_indptr,
        kv_indptr[:-1],
        1,
        work_indptr,
        work_info_set,
        1.0 / (head_dim**0.5),
        split_o,
        split_lse,
        debug_qk_scores,
    )

    assert torch.isfinite(split_o).all()
    assert torch.isfinite(split_lse).all()


def test_mha_prefill_splitk_stage1_opus_matches_reference_single_work():
    device = "cuda"
    batch_size = 1
    q_len = 128
    kv_len = 128
    q_tile_size = 128
    split_k_size = 32
    num_heads = 1
    num_kv_heads = 1
    head_dim = 128
    softmax_scale = 1.0 / (head_dim**0.5)

    qo_indptr = make_uniform_indptr(batch_size, q_len, device=device)
    kv_indptr = make_uniform_indptr(batch_size, kv_len, device=device)

    (
        work_metadata_ptrs,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = alloc_metadata_tensors(
        batch_size=batch_size,
        max_seqlen_qo=q_len,
        max_seqlen_kv=kv_len,
        q_tile_size=q_tile_size,
        split_k_size=split_k_size,
        is_causal=True,
        uni_seqlen_qo=q_len,
        device=device,
    )

    aiter.get_mha_prefill_splitk_metadata_v1(
        qo_indptr,
        kv_indptr,
        True,
        q_tile_size,
        split_k_size,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_qo=q_len,
        uni_seqlen_qo=q_len,
    )

    valid_work_num = int(work_indptr.cpu()[1].item())
    work_list = to_work_list(work_info_set, valid_work_num)
    work_item = work_list[0]
    partial_tokens = max(
        item["partial_qo_loc"] + (item["qo_end"] - item["qo_start"])
        for item in work_list
    )

    q = torch.randn((q_len, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((kv_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn((kv_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    split_o = torch.empty((partial_tokens, num_heads, head_dim), dtype=torch.float32, device=device)
    split_lse = torch.empty((partial_tokens, num_heads), dtype=torch.float32, device=device)
    debug_qk_scores = torch.empty((valid_work_num, num_heads, 4, 32, 32), dtype=torch.float32, device=device)

    aiter.mha_prefill_splitk_stage1_opus(
        q,
        k,
        v,
        kv_indptr,
        kv_indptr[:-1],
        1,
        work_indptr,
        work_info_set,
        softmax_scale,
        split_o,
        split_lse,
        debug_qk_scores,
    )

    ref_o, ref_lse, ref_scores = reference_stage1_single_work(
        q, k, v, work_item, softmax_scale
    )

    out_token_begin = work_item["partial_qo_loc"]
    out_token_end = out_token_begin + (work_item["qo_end"] - work_item["qo_start"])

    got_o = split_o[out_token_begin:out_token_end, 0, :]
    got_lse = split_lse[out_token_begin:out_token_end, 0]

    wave0_scores = debug_qk_scores[0, 0, 0]
    ref_scores_wave0 = ref_scores[:32, :]
    assert torch.allclose(
        wave0_scores[: ref_scores_wave0.size(0), : ref_scores_wave0.size(1)],
        ref_scores_wave0,
        atol=1e-1,
        rtol=1e-1,
    )
    assert torch.allclose(got_o, ref_o, atol=1e-1, rtol=1e-1)
    assert torch.allclose(got_lse, ref_lse, atol=1e-1, rtol=1e-1)
