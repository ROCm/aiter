// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "prefill_sparge_attention.h"
#include "fmha_fwd_trek.hpp"

torch::Tensor prefill_sparge_attention(
    torch::Tensor &TQ,
    torch::Tensor &TK,
    torch::Tensor &TV,
    torch::Tensor &Tlut,
    torch::Tensor &Tvalid_block_num,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias = std::nullopt,
    std::optional<torch::Tensor> lse = std::nullopt,
    std::optional<torch::Tensor>  seqstart_q = std::nullopt,
    std::optional<torch::Tensor>  seqstart_k = std::nullopt,
    int bias_type = 0,
    int batch = 0,
    int nhead = 0,
    int nhead_k = 0,
    int seqlen_q = 0,
    int seqlen_k = 0,
    int hdim_q = 0,
    int hdim_v = 0,
    float pv_threshold = 50,
    int mode = 0,
    bool i_perm = true, 
    bool o_perm = true,
    int max_seqlen_q = 0,
    int max_seqlen_k = 0,
    bool is_causal = true
){
    std::string data_type = "fp16";
    if (TQ.dtype() == torch::kBFloat16) {
        data_type = "bf16";
    }

    if (max_seqlen_q == 0) max_seqlen_q = seqlen_q;
    if (max_seqlen_k == 0) max_seqlen_k = seqlen_k;
    bool is_v_rowmajor = true;
    int seqlen_knew = 0;
    float scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));
    float scale_p = 1.f;
    float scale_o = 1.f;
    const float logits_soft_cap = 0.0;

    std::string msk_str;
    if (is_causal) {
        msk_str = "b";
    } else {
        msk_str = "0";
    }
    mask_info mask = mask_info::decode(msk_str, seqlen_q, seqlen_k);

    const ck_tile::index_t shape_batch    = (mode == 0 ? batch : 1);
    const ck_tile::index_t shape_seqlen_q = (mode == 0 ? seqlen_q : max_seqlen_q);
    const ck_tile::index_t shape_seqlen_k = (mode == 0 ? seqlen_k : max_seqlen_k);

    ck_tile::stream_config stream_config{nullptr,
                                         false, // time_kernel
                                         1, /* log_level = */
                                         0,
                                         1,
                                         false};

    const auto init_args = [&](auto& args) {
        assert(nhead % nhead_k == 0);
        const ck_tile::index_t stride_q    = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k    = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_knew = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v    = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return (i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k);
        }();
        const ck_tile::index_t stride_vnew = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? seqlen_knew : nhead_k * seqlen_knew;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? max_seqlen_k : 1 * max_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = (hdim_v);
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = i_perm ? shape_seqlen_k * hdim_q : hdim_q;
        const ck_tile::index_t nhead_stride_knew = (i_perm ? seqlen_knew * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v    = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_vnew = [&]() {
            if(is_v_rowmajor)
                return i_perm ? seqlen_knew * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_knew : seqlen_knew;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * max_seqlen_k : 0 * max_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = (shape_seqlen_q);
        const ck_tile::index_t nhead_stride_o_acc   = (shape_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k = nhead_k * shape_seqlen_k * hdim_q;
        const ck_tile::index_t batch_stride_knew = (nhead_k * seqlen_knew * hdim_q);
        const ck_tile::index_t batch_stride_v = nhead_k * hdim_v * shape_seqlen_k;
        const ck_tile::index_t batch_stride_vnew    = (nhead_k * hdim_v * seqlen_knew);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc = (nhead * shape_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o     = (nhead * shape_seqlen_q * hdim_v);
        // const ck_tile::index_t batch_stride_block_table = (max_num_page_blocks / batch);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = (shape_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = (shape_seqlen_q * hdim_v);

        args.q_ptr = TQ.data_ptr();
        args.k_ptr = TK.data_ptr();
        args.v_ptr = TV.data_ptr();
        args.lut_ptr = Tlut.data_ptr();
        args.valid_block_num_ptr = Tvalid_block_num.data_ptr();

        args.batch    = batch;
        args.seqlen_q = shape_seqlen_q; // unused in group mode
        args.hdim_q   = hdim_q;
        args.hdim_v   = hdim_v;
        args.nhead_q  = nhead;
        args.nhead_k  = nhead_k;

        args.stride_q       = stride_q;
        args.stride_k       = stride_k;
        args.stride_v       = stride_v;
        args.nhead_stride_q = nhead_stride_q;
        args.nhead_stride_k = nhead_stride_k;
        args.nhead_stride_v = nhead_stride_v;
        args.batch_stride_q = batch_stride_q;
        args.batch_stride_k = batch_stride_k;
        args.batch_stride_v = batch_stride_v;

        // args.bias_ptr = bias.type == bias_enum::alibi ? alibi_slope_buf.GetDeviceBuffer()
        //                                                 : bias_buf.GetDeviceBuffer();
        args.bias_ptr = bias ? bias->data_ptr() : nullptr;
        args.lse_ptr  = lse ? lse->data_ptr() : nullptr;
        args.o_ptr    = Y.data_ptr();

        args.seqstart_q_ptr =
            (mode == 1 ? seqstart_q->data_ptr() : nullptr);
        args.seqstart_k_ptr =
            (mode == 1 ? seqstart_k->data_ptr() : nullptr);
        args.seqlen_k_ptr =  nullptr;

        args.seqlen_k     = shape_seqlen_k; // unused in group mode (or kvcache enabled)
        args.max_seqlen_q = max_seqlen_q;

        args.pv_threshold = pv_threshold;
        args.scale_s = scale_s;
        args.scale_p = scale_p;
        args.scale_o = scale_o;

        args.logits_soft_cap = logits_soft_cap;

        args.stride_bias =stride_bias;
        args.stride_o          = stride_o;
        args.nhead_stride_bias = nhead_stride_bias;
        args.nhead_stride_lse  = nhead_stride_lse;
        args.nhead_stride_o    = nhead_stride_o;
        args.batch_stride_bias = batch_stride_bias;
        args.batch_stride_lse  = batch_stride_lse;
        args.batch_stride_o    = batch_stride_o;

        args.window_size_left  = mask.left;
        args.window_size_right = mask.right;
        args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

        args.rand_val_ptr = nullptr;

        args.stride_randval       = stride_randval;
        args.nhead_stride_randval = nhead_stride_randval;
        args.batch_stride_randval = batch_stride_randval;

        args.p_drop    = 0.;
        args.s_randval = false;

    };

    const auto init_traits = [&](auto& traits) {
        traits.hdim_q        = hdim_q;
        traits.hdim_v        = hdim_v;
        traits.data_type     = data_type;
        traits.is_v_rowmajor = is_v_rowmajor;


        traits.is_group_mode       = (mode == 1);
        traits.has_logits_soft_cap = 0.f < logits_soft_cap;
        traits.mask_type           = mask.type;
        traits.bias_type           = static_cast<bias_enum>(bias_type);
        traits.has_lse             = lse ? true: false;
        traits.do_fp8_static_quant = false;

        traits.has_dropout = false;

    };

    fmha_sparge_fwd_traits fmha_traits;
    init_traits(fmha_traits);

    fmha_sparge_fwd_args args;
    init_args(args);
    
    fmha_sparge_fwd(fmha_traits, args, stream_config);
    printf("fmha_sparge_fwd\n");

    return Y;
}
