from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, s72: "Sym(s72)", L_input_ids_: "i32[s72]", L_self_modules_embed_parameters_weight_: "bf16[129280, 4096]", L_self_modules_layers_modules_0_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_0_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_0_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_: "bf16[4096]", s80: "Sym(s80)", L_positions_: "i64[s80]", L_self_modules_layers_modules_0_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_0_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_0_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_1_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_1_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_1_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_1_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_1_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_1_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_2_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_2_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_2_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_2_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_2_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_2_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_3_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_3_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_3_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_3_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_3_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_3_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_4_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_4_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_4_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_4_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_4_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_4_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_5_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_5_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_5_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_5_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_5_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_5_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_6_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_6_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_6_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_6_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_6_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_6_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_7_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_7_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_7_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_7_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_7_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_7_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_8_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_8_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_8_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_8_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_8_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_8_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_9_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_9_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_9_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_9_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_9_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_9_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_10_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_10_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_10_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_10_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_10_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_10_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_11_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_11_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_11_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_11_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_11_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_11_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_12_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_12_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_12_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_12_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_12_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_12_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_13_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_13_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_13_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_13_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_13_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_13_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_14_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_14_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_14_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_14_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_14_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_14_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_15_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_15_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_15_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_15_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_15_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_15_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_16_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_16_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_16_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_16_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_16_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_16_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_17_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_17_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_17_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_17_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_17_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_17_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_18_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_18_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_18_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_18_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_18_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_18_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_19_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_19_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_19_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_19_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_19_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_19_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_20_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_20_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_20_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_20_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_20_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_20_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_21_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_21_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_21_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_21_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_21_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_21_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_22_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_22_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_22_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_22_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_22_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_22_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_23_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_23_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_23_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_23_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_23_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_23_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_24_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_24_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_24_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_24_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_24_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_24_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_25_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_25_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_25_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_25_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_25_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_25_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_26_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_26_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_26_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_26_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_26_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_26_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_27_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_27_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_27_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_27_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_27_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_27_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_28_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_28_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_28_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_28_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_28_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_28_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_29_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_29_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_29_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_29_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_29_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_29_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_30_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_30_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_30_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_30_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_30_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_30_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_31_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_31_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_31_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_31_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_31_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_31_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_32_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_32_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_32_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_32_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_32_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_32_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_33_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_33_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_33_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_33_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_33_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_33_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_34_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_34_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_34_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_34_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_34_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_34_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_35_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_35_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_35_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_35_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_35_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_35_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_36_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_36_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_36_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_36_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_36_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_36_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_37_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_37_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_37_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_37_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_37_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_37_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_38_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_38_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_38_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_38_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_38_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_38_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_39_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_39_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_39_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_39_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_39_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_39_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_40_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_40_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_40_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_40_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_40_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_40_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_41_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_41_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_41_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_41_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_41_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_41_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_42_parameters_hc_attn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_42_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_42_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_42_parameters_hc_ffn_fn_: "i32[24, 16384]", L_self_modules_layers_modules_42_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_42_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_: "bf16[4096]"):
        l_input_ids_ = L_input_ids_
        l_self_modules_embed_parameters_weight_ = L_self_modules_embed_parameters_weight_
        l_self_modules_layers_modules_0_parameters_hc_attn_fn_ = L_self_modules_layers_modules_0_parameters_hc_attn_fn_
        l_self_modules_layers_modules_0_parameters_hc_attn_scale_ = L_self_modules_layers_modules_0_parameters_hc_attn_scale_
        l_self_modules_layers_modules_0_parameters_hc_attn_base_ = L_self_modules_layers_modules_0_parameters_hc_attn_base_
        l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_
        l_positions_ = L_positions_
        l_self_modules_layers_modules_0_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_0_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_0_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_0_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_0_parameters_hc_ffn_base_ = L_self_modules_layers_modules_0_parameters_hc_ffn_base_
        l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_1_parameters_hc_attn_fn_ = L_self_modules_layers_modules_1_parameters_hc_attn_fn_
        l_self_modules_layers_modules_1_parameters_hc_attn_scale_ = L_self_modules_layers_modules_1_parameters_hc_attn_scale_
        l_self_modules_layers_modules_1_parameters_hc_attn_base_ = L_self_modules_layers_modules_1_parameters_hc_attn_base_
        l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_1_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_1_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_1_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_1_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_1_parameters_hc_ffn_base_ = L_self_modules_layers_modules_1_parameters_hc_ffn_base_
        l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_2_parameters_hc_attn_fn_ = L_self_modules_layers_modules_2_parameters_hc_attn_fn_
        l_self_modules_layers_modules_2_parameters_hc_attn_scale_ = L_self_modules_layers_modules_2_parameters_hc_attn_scale_
        l_self_modules_layers_modules_2_parameters_hc_attn_base_ = L_self_modules_layers_modules_2_parameters_hc_attn_base_
        l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_2_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_2_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_2_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_2_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_2_parameters_hc_ffn_base_ = L_self_modules_layers_modules_2_parameters_hc_ffn_base_
        l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_3_parameters_hc_attn_fn_ = L_self_modules_layers_modules_3_parameters_hc_attn_fn_
        l_self_modules_layers_modules_3_parameters_hc_attn_scale_ = L_self_modules_layers_modules_3_parameters_hc_attn_scale_
        l_self_modules_layers_modules_3_parameters_hc_attn_base_ = L_self_modules_layers_modules_3_parameters_hc_attn_base_
        l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_3_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_3_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_3_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_3_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_3_parameters_hc_ffn_base_ = L_self_modules_layers_modules_3_parameters_hc_ffn_base_
        l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_4_parameters_hc_attn_fn_ = L_self_modules_layers_modules_4_parameters_hc_attn_fn_
        l_self_modules_layers_modules_4_parameters_hc_attn_scale_ = L_self_modules_layers_modules_4_parameters_hc_attn_scale_
        l_self_modules_layers_modules_4_parameters_hc_attn_base_ = L_self_modules_layers_modules_4_parameters_hc_attn_base_
        l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_4_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_4_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_4_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_4_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_4_parameters_hc_ffn_base_ = L_self_modules_layers_modules_4_parameters_hc_ffn_base_
        l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_5_parameters_hc_attn_fn_ = L_self_modules_layers_modules_5_parameters_hc_attn_fn_
        l_self_modules_layers_modules_5_parameters_hc_attn_scale_ = L_self_modules_layers_modules_5_parameters_hc_attn_scale_
        l_self_modules_layers_modules_5_parameters_hc_attn_base_ = L_self_modules_layers_modules_5_parameters_hc_attn_base_
        l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_5_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_5_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_5_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_5_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_5_parameters_hc_ffn_base_ = L_self_modules_layers_modules_5_parameters_hc_ffn_base_
        l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_6_parameters_hc_attn_fn_ = L_self_modules_layers_modules_6_parameters_hc_attn_fn_
        l_self_modules_layers_modules_6_parameters_hc_attn_scale_ = L_self_modules_layers_modules_6_parameters_hc_attn_scale_
        l_self_modules_layers_modules_6_parameters_hc_attn_base_ = L_self_modules_layers_modules_6_parameters_hc_attn_base_
        l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_6_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_6_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_6_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_6_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_6_parameters_hc_ffn_base_ = L_self_modules_layers_modules_6_parameters_hc_ffn_base_
        l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_7_parameters_hc_attn_fn_ = L_self_modules_layers_modules_7_parameters_hc_attn_fn_
        l_self_modules_layers_modules_7_parameters_hc_attn_scale_ = L_self_modules_layers_modules_7_parameters_hc_attn_scale_
        l_self_modules_layers_modules_7_parameters_hc_attn_base_ = L_self_modules_layers_modules_7_parameters_hc_attn_base_
        l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_7_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_7_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_7_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_7_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_7_parameters_hc_ffn_base_ = L_self_modules_layers_modules_7_parameters_hc_ffn_base_
        l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_8_parameters_hc_attn_fn_ = L_self_modules_layers_modules_8_parameters_hc_attn_fn_
        l_self_modules_layers_modules_8_parameters_hc_attn_scale_ = L_self_modules_layers_modules_8_parameters_hc_attn_scale_
        l_self_modules_layers_modules_8_parameters_hc_attn_base_ = L_self_modules_layers_modules_8_parameters_hc_attn_base_
        l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_8_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_8_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_8_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_8_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_8_parameters_hc_ffn_base_ = L_self_modules_layers_modules_8_parameters_hc_ffn_base_
        l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_9_parameters_hc_attn_fn_ = L_self_modules_layers_modules_9_parameters_hc_attn_fn_
        l_self_modules_layers_modules_9_parameters_hc_attn_scale_ = L_self_modules_layers_modules_9_parameters_hc_attn_scale_
        l_self_modules_layers_modules_9_parameters_hc_attn_base_ = L_self_modules_layers_modules_9_parameters_hc_attn_base_
        l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_9_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_9_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_9_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_9_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_9_parameters_hc_ffn_base_ = L_self_modules_layers_modules_9_parameters_hc_ffn_base_
        l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_10_parameters_hc_attn_fn_ = L_self_modules_layers_modules_10_parameters_hc_attn_fn_
        l_self_modules_layers_modules_10_parameters_hc_attn_scale_ = L_self_modules_layers_modules_10_parameters_hc_attn_scale_
        l_self_modules_layers_modules_10_parameters_hc_attn_base_ = L_self_modules_layers_modules_10_parameters_hc_attn_base_
        l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_10_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_10_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_10_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_10_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_10_parameters_hc_ffn_base_ = L_self_modules_layers_modules_10_parameters_hc_ffn_base_
        l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_11_parameters_hc_attn_fn_ = L_self_modules_layers_modules_11_parameters_hc_attn_fn_
        l_self_modules_layers_modules_11_parameters_hc_attn_scale_ = L_self_modules_layers_modules_11_parameters_hc_attn_scale_
        l_self_modules_layers_modules_11_parameters_hc_attn_base_ = L_self_modules_layers_modules_11_parameters_hc_attn_base_
        l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_11_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_11_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_11_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_11_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_11_parameters_hc_ffn_base_ = L_self_modules_layers_modules_11_parameters_hc_ffn_base_
        l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_12_parameters_hc_attn_fn_ = L_self_modules_layers_modules_12_parameters_hc_attn_fn_
        l_self_modules_layers_modules_12_parameters_hc_attn_scale_ = L_self_modules_layers_modules_12_parameters_hc_attn_scale_
        l_self_modules_layers_modules_12_parameters_hc_attn_base_ = L_self_modules_layers_modules_12_parameters_hc_attn_base_
        l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_12_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_12_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_12_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_12_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_12_parameters_hc_ffn_base_ = L_self_modules_layers_modules_12_parameters_hc_ffn_base_
        l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_13_parameters_hc_attn_fn_ = L_self_modules_layers_modules_13_parameters_hc_attn_fn_
        l_self_modules_layers_modules_13_parameters_hc_attn_scale_ = L_self_modules_layers_modules_13_parameters_hc_attn_scale_
        l_self_modules_layers_modules_13_parameters_hc_attn_base_ = L_self_modules_layers_modules_13_parameters_hc_attn_base_
        l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_13_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_13_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_13_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_13_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_13_parameters_hc_ffn_base_ = L_self_modules_layers_modules_13_parameters_hc_ffn_base_
        l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_14_parameters_hc_attn_fn_ = L_self_modules_layers_modules_14_parameters_hc_attn_fn_
        l_self_modules_layers_modules_14_parameters_hc_attn_scale_ = L_self_modules_layers_modules_14_parameters_hc_attn_scale_
        l_self_modules_layers_modules_14_parameters_hc_attn_base_ = L_self_modules_layers_modules_14_parameters_hc_attn_base_
        l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_14_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_14_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_14_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_14_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_14_parameters_hc_ffn_base_ = L_self_modules_layers_modules_14_parameters_hc_ffn_base_
        l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_15_parameters_hc_attn_fn_ = L_self_modules_layers_modules_15_parameters_hc_attn_fn_
        l_self_modules_layers_modules_15_parameters_hc_attn_scale_ = L_self_modules_layers_modules_15_parameters_hc_attn_scale_
        l_self_modules_layers_modules_15_parameters_hc_attn_base_ = L_self_modules_layers_modules_15_parameters_hc_attn_base_
        l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_15_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_15_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_15_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_15_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_15_parameters_hc_ffn_base_ = L_self_modules_layers_modules_15_parameters_hc_ffn_base_
        l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_16_parameters_hc_attn_fn_ = L_self_modules_layers_modules_16_parameters_hc_attn_fn_
        l_self_modules_layers_modules_16_parameters_hc_attn_scale_ = L_self_modules_layers_modules_16_parameters_hc_attn_scale_
        l_self_modules_layers_modules_16_parameters_hc_attn_base_ = L_self_modules_layers_modules_16_parameters_hc_attn_base_
        l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_16_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_16_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_16_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_16_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_16_parameters_hc_ffn_base_ = L_self_modules_layers_modules_16_parameters_hc_ffn_base_
        l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_17_parameters_hc_attn_fn_ = L_self_modules_layers_modules_17_parameters_hc_attn_fn_
        l_self_modules_layers_modules_17_parameters_hc_attn_scale_ = L_self_modules_layers_modules_17_parameters_hc_attn_scale_
        l_self_modules_layers_modules_17_parameters_hc_attn_base_ = L_self_modules_layers_modules_17_parameters_hc_attn_base_
        l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_17_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_17_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_17_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_17_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_17_parameters_hc_ffn_base_ = L_self_modules_layers_modules_17_parameters_hc_ffn_base_
        l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_18_parameters_hc_attn_fn_ = L_self_modules_layers_modules_18_parameters_hc_attn_fn_
        l_self_modules_layers_modules_18_parameters_hc_attn_scale_ = L_self_modules_layers_modules_18_parameters_hc_attn_scale_
        l_self_modules_layers_modules_18_parameters_hc_attn_base_ = L_self_modules_layers_modules_18_parameters_hc_attn_base_
        l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_18_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_18_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_18_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_18_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_18_parameters_hc_ffn_base_ = L_self_modules_layers_modules_18_parameters_hc_ffn_base_
        l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_19_parameters_hc_attn_fn_ = L_self_modules_layers_modules_19_parameters_hc_attn_fn_
        l_self_modules_layers_modules_19_parameters_hc_attn_scale_ = L_self_modules_layers_modules_19_parameters_hc_attn_scale_
        l_self_modules_layers_modules_19_parameters_hc_attn_base_ = L_self_modules_layers_modules_19_parameters_hc_attn_base_
        l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_19_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_19_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_19_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_19_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_19_parameters_hc_ffn_base_ = L_self_modules_layers_modules_19_parameters_hc_ffn_base_
        l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_20_parameters_hc_attn_fn_ = L_self_modules_layers_modules_20_parameters_hc_attn_fn_
        l_self_modules_layers_modules_20_parameters_hc_attn_scale_ = L_self_modules_layers_modules_20_parameters_hc_attn_scale_
        l_self_modules_layers_modules_20_parameters_hc_attn_base_ = L_self_modules_layers_modules_20_parameters_hc_attn_base_
        l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_20_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_20_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_20_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_20_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_20_parameters_hc_ffn_base_ = L_self_modules_layers_modules_20_parameters_hc_ffn_base_
        l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_21_parameters_hc_attn_fn_ = L_self_modules_layers_modules_21_parameters_hc_attn_fn_
        l_self_modules_layers_modules_21_parameters_hc_attn_scale_ = L_self_modules_layers_modules_21_parameters_hc_attn_scale_
        l_self_modules_layers_modules_21_parameters_hc_attn_base_ = L_self_modules_layers_modules_21_parameters_hc_attn_base_
        l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_21_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_21_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_21_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_21_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_21_parameters_hc_ffn_base_ = L_self_modules_layers_modules_21_parameters_hc_ffn_base_
        l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_22_parameters_hc_attn_fn_ = L_self_modules_layers_modules_22_parameters_hc_attn_fn_
        l_self_modules_layers_modules_22_parameters_hc_attn_scale_ = L_self_modules_layers_modules_22_parameters_hc_attn_scale_
        l_self_modules_layers_modules_22_parameters_hc_attn_base_ = L_self_modules_layers_modules_22_parameters_hc_attn_base_
        l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_22_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_22_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_22_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_22_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_22_parameters_hc_ffn_base_ = L_self_modules_layers_modules_22_parameters_hc_ffn_base_
        l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_23_parameters_hc_attn_fn_ = L_self_modules_layers_modules_23_parameters_hc_attn_fn_
        l_self_modules_layers_modules_23_parameters_hc_attn_scale_ = L_self_modules_layers_modules_23_parameters_hc_attn_scale_
        l_self_modules_layers_modules_23_parameters_hc_attn_base_ = L_self_modules_layers_modules_23_parameters_hc_attn_base_
        l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_23_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_23_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_23_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_23_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_23_parameters_hc_ffn_base_ = L_self_modules_layers_modules_23_parameters_hc_ffn_base_
        l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_24_parameters_hc_attn_fn_ = L_self_modules_layers_modules_24_parameters_hc_attn_fn_
        l_self_modules_layers_modules_24_parameters_hc_attn_scale_ = L_self_modules_layers_modules_24_parameters_hc_attn_scale_
        l_self_modules_layers_modules_24_parameters_hc_attn_base_ = L_self_modules_layers_modules_24_parameters_hc_attn_base_
        l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_24_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_24_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_24_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_24_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_24_parameters_hc_ffn_base_ = L_self_modules_layers_modules_24_parameters_hc_ffn_base_
        l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_25_parameters_hc_attn_fn_ = L_self_modules_layers_modules_25_parameters_hc_attn_fn_
        l_self_modules_layers_modules_25_parameters_hc_attn_scale_ = L_self_modules_layers_modules_25_parameters_hc_attn_scale_
        l_self_modules_layers_modules_25_parameters_hc_attn_base_ = L_self_modules_layers_modules_25_parameters_hc_attn_base_
        l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_25_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_25_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_25_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_25_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_25_parameters_hc_ffn_base_ = L_self_modules_layers_modules_25_parameters_hc_ffn_base_
        l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_26_parameters_hc_attn_fn_ = L_self_modules_layers_modules_26_parameters_hc_attn_fn_
        l_self_modules_layers_modules_26_parameters_hc_attn_scale_ = L_self_modules_layers_modules_26_parameters_hc_attn_scale_
        l_self_modules_layers_modules_26_parameters_hc_attn_base_ = L_self_modules_layers_modules_26_parameters_hc_attn_base_
        l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_26_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_26_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_26_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_26_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_26_parameters_hc_ffn_base_ = L_self_modules_layers_modules_26_parameters_hc_ffn_base_
        l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_27_parameters_hc_attn_fn_ = L_self_modules_layers_modules_27_parameters_hc_attn_fn_
        l_self_modules_layers_modules_27_parameters_hc_attn_scale_ = L_self_modules_layers_modules_27_parameters_hc_attn_scale_
        l_self_modules_layers_modules_27_parameters_hc_attn_base_ = L_self_modules_layers_modules_27_parameters_hc_attn_base_
        l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_27_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_27_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_27_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_27_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_27_parameters_hc_ffn_base_ = L_self_modules_layers_modules_27_parameters_hc_ffn_base_
        l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_28_parameters_hc_attn_fn_ = L_self_modules_layers_modules_28_parameters_hc_attn_fn_
        l_self_modules_layers_modules_28_parameters_hc_attn_scale_ = L_self_modules_layers_modules_28_parameters_hc_attn_scale_
        l_self_modules_layers_modules_28_parameters_hc_attn_base_ = L_self_modules_layers_modules_28_parameters_hc_attn_base_
        l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_28_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_28_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_28_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_28_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_28_parameters_hc_ffn_base_ = L_self_modules_layers_modules_28_parameters_hc_ffn_base_
        l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_29_parameters_hc_attn_fn_ = L_self_modules_layers_modules_29_parameters_hc_attn_fn_
        l_self_modules_layers_modules_29_parameters_hc_attn_scale_ = L_self_modules_layers_modules_29_parameters_hc_attn_scale_
        l_self_modules_layers_modules_29_parameters_hc_attn_base_ = L_self_modules_layers_modules_29_parameters_hc_attn_base_
        l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_29_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_29_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_29_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_29_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_29_parameters_hc_ffn_base_ = L_self_modules_layers_modules_29_parameters_hc_ffn_base_
        l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_30_parameters_hc_attn_fn_ = L_self_modules_layers_modules_30_parameters_hc_attn_fn_
        l_self_modules_layers_modules_30_parameters_hc_attn_scale_ = L_self_modules_layers_modules_30_parameters_hc_attn_scale_
        l_self_modules_layers_modules_30_parameters_hc_attn_base_ = L_self_modules_layers_modules_30_parameters_hc_attn_base_
        l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_30_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_30_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_30_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_30_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_30_parameters_hc_ffn_base_ = L_self_modules_layers_modules_30_parameters_hc_ffn_base_
        l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_31_parameters_hc_attn_fn_ = L_self_modules_layers_modules_31_parameters_hc_attn_fn_
        l_self_modules_layers_modules_31_parameters_hc_attn_scale_ = L_self_modules_layers_modules_31_parameters_hc_attn_scale_
        l_self_modules_layers_modules_31_parameters_hc_attn_base_ = L_self_modules_layers_modules_31_parameters_hc_attn_base_
        l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_31_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_31_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_31_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_31_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_31_parameters_hc_ffn_base_ = L_self_modules_layers_modules_31_parameters_hc_ffn_base_
        l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_32_parameters_hc_attn_fn_ = L_self_modules_layers_modules_32_parameters_hc_attn_fn_
        l_self_modules_layers_modules_32_parameters_hc_attn_scale_ = L_self_modules_layers_modules_32_parameters_hc_attn_scale_
        l_self_modules_layers_modules_32_parameters_hc_attn_base_ = L_self_modules_layers_modules_32_parameters_hc_attn_base_
        l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_32_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_32_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_32_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_32_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_32_parameters_hc_ffn_base_ = L_self_modules_layers_modules_32_parameters_hc_ffn_base_
        l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_33_parameters_hc_attn_fn_ = L_self_modules_layers_modules_33_parameters_hc_attn_fn_
        l_self_modules_layers_modules_33_parameters_hc_attn_scale_ = L_self_modules_layers_modules_33_parameters_hc_attn_scale_
        l_self_modules_layers_modules_33_parameters_hc_attn_base_ = L_self_modules_layers_modules_33_parameters_hc_attn_base_
        l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_33_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_33_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_33_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_33_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_33_parameters_hc_ffn_base_ = L_self_modules_layers_modules_33_parameters_hc_ffn_base_
        l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_34_parameters_hc_attn_fn_ = L_self_modules_layers_modules_34_parameters_hc_attn_fn_
        l_self_modules_layers_modules_34_parameters_hc_attn_scale_ = L_self_modules_layers_modules_34_parameters_hc_attn_scale_
        l_self_modules_layers_modules_34_parameters_hc_attn_base_ = L_self_modules_layers_modules_34_parameters_hc_attn_base_
        l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_34_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_34_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_34_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_34_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_34_parameters_hc_ffn_base_ = L_self_modules_layers_modules_34_parameters_hc_ffn_base_
        l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_35_parameters_hc_attn_fn_ = L_self_modules_layers_modules_35_parameters_hc_attn_fn_
        l_self_modules_layers_modules_35_parameters_hc_attn_scale_ = L_self_modules_layers_modules_35_parameters_hc_attn_scale_
        l_self_modules_layers_modules_35_parameters_hc_attn_base_ = L_self_modules_layers_modules_35_parameters_hc_attn_base_
        l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_35_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_35_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_35_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_35_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_35_parameters_hc_ffn_base_ = L_self_modules_layers_modules_35_parameters_hc_ffn_base_
        l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_36_parameters_hc_attn_fn_ = L_self_modules_layers_modules_36_parameters_hc_attn_fn_
        l_self_modules_layers_modules_36_parameters_hc_attn_scale_ = L_self_modules_layers_modules_36_parameters_hc_attn_scale_
        l_self_modules_layers_modules_36_parameters_hc_attn_base_ = L_self_modules_layers_modules_36_parameters_hc_attn_base_
        l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_36_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_36_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_36_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_36_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_36_parameters_hc_ffn_base_ = L_self_modules_layers_modules_36_parameters_hc_ffn_base_
        l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_37_parameters_hc_attn_fn_ = L_self_modules_layers_modules_37_parameters_hc_attn_fn_
        l_self_modules_layers_modules_37_parameters_hc_attn_scale_ = L_self_modules_layers_modules_37_parameters_hc_attn_scale_
        l_self_modules_layers_modules_37_parameters_hc_attn_base_ = L_self_modules_layers_modules_37_parameters_hc_attn_base_
        l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_37_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_37_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_37_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_37_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_37_parameters_hc_ffn_base_ = L_self_modules_layers_modules_37_parameters_hc_ffn_base_
        l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_38_parameters_hc_attn_fn_ = L_self_modules_layers_modules_38_parameters_hc_attn_fn_
        l_self_modules_layers_modules_38_parameters_hc_attn_scale_ = L_self_modules_layers_modules_38_parameters_hc_attn_scale_
        l_self_modules_layers_modules_38_parameters_hc_attn_base_ = L_self_modules_layers_modules_38_parameters_hc_attn_base_
        l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_38_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_38_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_38_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_38_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_38_parameters_hc_ffn_base_ = L_self_modules_layers_modules_38_parameters_hc_ffn_base_
        l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_39_parameters_hc_attn_fn_ = L_self_modules_layers_modules_39_parameters_hc_attn_fn_
        l_self_modules_layers_modules_39_parameters_hc_attn_scale_ = L_self_modules_layers_modules_39_parameters_hc_attn_scale_
        l_self_modules_layers_modules_39_parameters_hc_attn_base_ = L_self_modules_layers_modules_39_parameters_hc_attn_base_
        l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_39_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_39_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_39_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_39_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_39_parameters_hc_ffn_base_ = L_self_modules_layers_modules_39_parameters_hc_ffn_base_
        l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_40_parameters_hc_attn_fn_ = L_self_modules_layers_modules_40_parameters_hc_attn_fn_
        l_self_modules_layers_modules_40_parameters_hc_attn_scale_ = L_self_modules_layers_modules_40_parameters_hc_attn_scale_
        l_self_modules_layers_modules_40_parameters_hc_attn_base_ = L_self_modules_layers_modules_40_parameters_hc_attn_base_
        l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_40_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_40_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_40_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_40_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_40_parameters_hc_ffn_base_ = L_self_modules_layers_modules_40_parameters_hc_ffn_base_
        l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_41_parameters_hc_attn_fn_ = L_self_modules_layers_modules_41_parameters_hc_attn_fn_
        l_self_modules_layers_modules_41_parameters_hc_attn_scale_ = L_self_modules_layers_modules_41_parameters_hc_attn_scale_
        l_self_modules_layers_modules_41_parameters_hc_attn_base_ = L_self_modules_layers_modules_41_parameters_hc_attn_base_
        l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_41_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_41_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_41_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_41_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_41_parameters_hc_ffn_base_ = L_self_modules_layers_modules_41_parameters_hc_ffn_base_
        l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_42_parameters_hc_attn_fn_ = L_self_modules_layers_modules_42_parameters_hc_attn_fn_
        l_self_modules_layers_modules_42_parameters_hc_attn_scale_ = L_self_modules_layers_modules_42_parameters_hc_attn_scale_
        l_self_modules_layers_modules_42_parameters_hc_attn_base_ = L_self_modules_layers_modules_42_parameters_hc_attn_base_
        l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_42_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_42_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_42_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_42_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_42_parameters_hc_ffn_base_ = L_self_modules_layers_modules_42_parameters_hc_ffn_base_
        l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_

        # No stacktrace found for following nodes
        submod_0 = self.submod_0(l_input_ids_, s72, l_self_modules_embed_parameters_weight_, l_self_modules_layers_modules_0_parameters_hc_attn_fn_, l_self_modules_layers_modules_0_parameters_hc_attn_scale_, l_self_modules_layers_modules_0_parameters_hc_attn_base_, l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_);  l_input_ids_ = l_self_modules_embed_parameters_weight_ = l_self_modules_layers_modules_0_parameters_hc_attn_fn_ = l_self_modules_layers_modules_0_parameters_hc_attn_scale_ = l_self_modules_layers_modules_0_parameters_hc_attn_base_ = l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_ = None
        getitem = submod_0[0]
        getitem_1 = submod_0[1]
        getitem_2 = submod_0[2]
        getitem_3 = submod_0[3];  submod_0 = None
        submod_1 = self.submod_1(getitem, s72, l_positions_, s80);  getitem = None
        submod_2 = self.submod_2(submod_1, s72, getitem_1, getitem_2, getitem_3, l_self_modules_layers_modules_0_parameters_hc_ffn_fn_, l_self_modules_layers_modules_0_parameters_hc_ffn_scale_, l_self_modules_layers_modules_0_parameters_hc_ffn_base_, l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_1_parameters_hc_attn_fn_, l_self_modules_layers_modules_1_parameters_hc_attn_scale_, l_self_modules_layers_modules_1_parameters_hc_attn_base_, l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_);  submod_1 = getitem_1 = getitem_2 = getitem_3 = l_self_modules_layers_modules_0_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_0_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_0_parameters_hc_ffn_base_ = l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_1_parameters_hc_attn_fn_ = l_self_modules_layers_modules_1_parameters_hc_attn_scale_ = l_self_modules_layers_modules_1_parameters_hc_attn_base_ = l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = None
        getitem_4 = submod_2[0]
        getitem_5 = submod_2[1]
        getitem_6 = submod_2[2]
        getitem_7 = submod_2[3];  submod_2 = None
        submod_3 = self.submod_3(getitem_4, s72, l_positions_, s80);  getitem_4 = None
        submod_4 = self.submod_4(submod_3, s72, getitem_5, getitem_6, getitem_7, l_self_modules_layers_modules_1_parameters_hc_ffn_fn_, l_self_modules_layers_modules_1_parameters_hc_ffn_scale_, l_self_modules_layers_modules_1_parameters_hc_ffn_base_, l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_2_parameters_hc_attn_fn_, l_self_modules_layers_modules_2_parameters_hc_attn_scale_, l_self_modules_layers_modules_2_parameters_hc_attn_base_, l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_);  submod_3 = getitem_5 = getitem_6 = getitem_7 = l_self_modules_layers_modules_1_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_1_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_1_parameters_hc_ffn_base_ = l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_2_parameters_hc_attn_fn_ = l_self_modules_layers_modules_2_parameters_hc_attn_scale_ = l_self_modules_layers_modules_2_parameters_hc_attn_base_ = l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = None
        getitem_8 = submod_4[0]
        getitem_9 = submod_4[1]
        getitem_10 = submod_4[2]
        getitem_11 = submod_4[3];  submod_4 = None
        submod_5 = self.submod_5(getitem_8, s72, l_positions_, s80);  getitem_8 = None
        submod_6 = self.submod_6(submod_5, s72, getitem_9, getitem_10, getitem_11, l_self_modules_layers_modules_2_parameters_hc_ffn_fn_, l_self_modules_layers_modules_2_parameters_hc_ffn_scale_, l_self_modules_layers_modules_2_parameters_hc_ffn_base_, l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_3_parameters_hc_attn_fn_, l_self_modules_layers_modules_3_parameters_hc_attn_scale_, l_self_modules_layers_modules_3_parameters_hc_attn_base_, l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_);  submod_5 = getitem_9 = getitem_10 = getitem_11 = l_self_modules_layers_modules_2_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_2_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_2_parameters_hc_ffn_base_ = l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_3_parameters_hc_attn_fn_ = l_self_modules_layers_modules_3_parameters_hc_attn_scale_ = l_self_modules_layers_modules_3_parameters_hc_attn_base_ = l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = None
        getitem_12 = submod_6[0]
        getitem_13 = submod_6[1]
        getitem_14 = submod_6[2]
        getitem_15 = submod_6[3];  submod_6 = None
        submod_7 = self.submod_7(getitem_12, s72, l_positions_, s80);  getitem_12 = None
        submod_8 = self.submod_8(submod_7, s72, getitem_13, getitem_14, getitem_15, l_self_modules_layers_modules_3_parameters_hc_ffn_fn_, l_self_modules_layers_modules_3_parameters_hc_ffn_scale_, l_self_modules_layers_modules_3_parameters_hc_ffn_base_, l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_4_parameters_hc_attn_fn_, l_self_modules_layers_modules_4_parameters_hc_attn_scale_, l_self_modules_layers_modules_4_parameters_hc_attn_base_, l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_);  submod_7 = getitem_13 = getitem_14 = getitem_15 = l_self_modules_layers_modules_3_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_3_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_3_parameters_hc_ffn_base_ = l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_4_parameters_hc_attn_fn_ = l_self_modules_layers_modules_4_parameters_hc_attn_scale_ = l_self_modules_layers_modules_4_parameters_hc_attn_base_ = l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = None
        getitem_16 = submod_8[0]
        getitem_17 = submod_8[1]
        getitem_18 = submod_8[2]
        getitem_19 = submod_8[3];  submod_8 = None
        submod_9 = self.submod_9(getitem_16, s72, l_positions_, s80);  getitem_16 = None
        submod_10 = self.submod_10(submod_9, s72, getitem_17, getitem_18, getitem_19, l_self_modules_layers_modules_4_parameters_hc_ffn_fn_, l_self_modules_layers_modules_4_parameters_hc_ffn_scale_, l_self_modules_layers_modules_4_parameters_hc_ffn_base_, l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_5_parameters_hc_attn_fn_, l_self_modules_layers_modules_5_parameters_hc_attn_scale_, l_self_modules_layers_modules_5_parameters_hc_attn_base_, l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_);  submod_9 = getitem_17 = getitem_18 = getitem_19 = l_self_modules_layers_modules_4_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_4_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_4_parameters_hc_ffn_base_ = l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_5_parameters_hc_attn_fn_ = l_self_modules_layers_modules_5_parameters_hc_attn_scale_ = l_self_modules_layers_modules_5_parameters_hc_attn_base_ = l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = None
        getitem_20 = submod_10[0]
        getitem_21 = submod_10[1]
        getitem_22 = submod_10[2]
        getitem_23 = submod_10[3];  submod_10 = None
        submod_11 = self.submod_11(getitem_20, s72, l_positions_, s80);  getitem_20 = None
        submod_12 = self.submod_12(submod_11, s72, getitem_21, getitem_22, getitem_23, l_self_modules_layers_modules_5_parameters_hc_ffn_fn_, l_self_modules_layers_modules_5_parameters_hc_ffn_scale_, l_self_modules_layers_modules_5_parameters_hc_ffn_base_, l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_6_parameters_hc_attn_fn_, l_self_modules_layers_modules_6_parameters_hc_attn_scale_, l_self_modules_layers_modules_6_parameters_hc_attn_base_, l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_);  submod_11 = getitem_21 = getitem_22 = getitem_23 = l_self_modules_layers_modules_5_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_5_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_5_parameters_hc_ffn_base_ = l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_6_parameters_hc_attn_fn_ = l_self_modules_layers_modules_6_parameters_hc_attn_scale_ = l_self_modules_layers_modules_6_parameters_hc_attn_base_ = l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = None
        getitem_24 = submod_12[0]
        getitem_25 = submod_12[1]
        getitem_26 = submod_12[2]
        getitem_27 = submod_12[3];  submod_12 = None
        submod_13 = self.submod_13(getitem_24, s72, l_positions_, s80);  getitem_24 = None
        submod_14 = self.submod_14(submod_13, s72, getitem_25, getitem_26, getitem_27, l_self_modules_layers_modules_6_parameters_hc_ffn_fn_, l_self_modules_layers_modules_6_parameters_hc_ffn_scale_, l_self_modules_layers_modules_6_parameters_hc_ffn_base_, l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_7_parameters_hc_attn_fn_, l_self_modules_layers_modules_7_parameters_hc_attn_scale_, l_self_modules_layers_modules_7_parameters_hc_attn_base_, l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_);  submod_13 = getitem_25 = getitem_26 = getitem_27 = l_self_modules_layers_modules_6_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_6_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_6_parameters_hc_ffn_base_ = l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_7_parameters_hc_attn_fn_ = l_self_modules_layers_modules_7_parameters_hc_attn_scale_ = l_self_modules_layers_modules_7_parameters_hc_attn_base_ = l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = None
        getitem_28 = submod_14[0]
        getitem_29 = submod_14[1]
        getitem_30 = submod_14[2]
        getitem_31 = submod_14[3];  submod_14 = None
        submod_15 = self.submod_15(getitem_28, s72, l_positions_, s80);  getitem_28 = None
        submod_16 = self.submod_16(submod_15, s72, getitem_29, getitem_30, getitem_31, l_self_modules_layers_modules_7_parameters_hc_ffn_fn_, l_self_modules_layers_modules_7_parameters_hc_ffn_scale_, l_self_modules_layers_modules_7_parameters_hc_ffn_base_, l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_8_parameters_hc_attn_fn_, l_self_modules_layers_modules_8_parameters_hc_attn_scale_, l_self_modules_layers_modules_8_parameters_hc_attn_base_, l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_);  submod_15 = getitem_29 = getitem_30 = getitem_31 = l_self_modules_layers_modules_7_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_7_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_7_parameters_hc_ffn_base_ = l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_8_parameters_hc_attn_fn_ = l_self_modules_layers_modules_8_parameters_hc_attn_scale_ = l_self_modules_layers_modules_8_parameters_hc_attn_base_ = l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = None
        getitem_32 = submod_16[0]
        getitem_33 = submod_16[1]
        getitem_34 = submod_16[2]
        getitem_35 = submod_16[3];  submod_16 = None
        submod_17 = self.submod_17(getitem_32, s72, l_positions_, s80);  getitem_32 = None
        submod_18 = self.submod_18(submod_17, s72, getitem_33, getitem_34, getitem_35, l_self_modules_layers_modules_8_parameters_hc_ffn_fn_, l_self_modules_layers_modules_8_parameters_hc_ffn_scale_, l_self_modules_layers_modules_8_parameters_hc_ffn_base_, l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_9_parameters_hc_attn_fn_, l_self_modules_layers_modules_9_parameters_hc_attn_scale_, l_self_modules_layers_modules_9_parameters_hc_attn_base_, l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_);  submod_17 = getitem_33 = getitem_34 = getitem_35 = l_self_modules_layers_modules_8_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_8_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_8_parameters_hc_ffn_base_ = l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_9_parameters_hc_attn_fn_ = l_self_modules_layers_modules_9_parameters_hc_attn_scale_ = l_self_modules_layers_modules_9_parameters_hc_attn_base_ = l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = None
        getitem_36 = submod_18[0]
        getitem_37 = submod_18[1]
        getitem_38 = submod_18[2]
        getitem_39 = submod_18[3];  submod_18 = None
        submod_19 = self.submod_19(getitem_36, s72, l_positions_, s80);  getitem_36 = None
        submod_20 = self.submod_20(submod_19, s72, getitem_37, getitem_38, getitem_39, l_self_modules_layers_modules_9_parameters_hc_ffn_fn_, l_self_modules_layers_modules_9_parameters_hc_ffn_scale_, l_self_modules_layers_modules_9_parameters_hc_ffn_base_, l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_10_parameters_hc_attn_fn_, l_self_modules_layers_modules_10_parameters_hc_attn_scale_, l_self_modules_layers_modules_10_parameters_hc_attn_base_, l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_);  submod_19 = getitem_37 = getitem_38 = getitem_39 = l_self_modules_layers_modules_9_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_9_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_9_parameters_hc_ffn_base_ = l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_10_parameters_hc_attn_fn_ = l_self_modules_layers_modules_10_parameters_hc_attn_scale_ = l_self_modules_layers_modules_10_parameters_hc_attn_base_ = l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = None
        getitem_40 = submod_20[0]
        getitem_41 = submod_20[1]
        getitem_42 = submod_20[2]
        getitem_43 = submod_20[3];  submod_20 = None
        submod_21 = self.submod_21(getitem_40, s72, l_positions_, s80);  getitem_40 = None
        submod_22 = self.submod_22(submod_21, s72, getitem_41, getitem_42, getitem_43, l_self_modules_layers_modules_10_parameters_hc_ffn_fn_, l_self_modules_layers_modules_10_parameters_hc_ffn_scale_, l_self_modules_layers_modules_10_parameters_hc_ffn_base_, l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_11_parameters_hc_attn_fn_, l_self_modules_layers_modules_11_parameters_hc_attn_scale_, l_self_modules_layers_modules_11_parameters_hc_attn_base_, l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_);  submod_21 = getitem_41 = getitem_42 = getitem_43 = l_self_modules_layers_modules_10_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_10_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_10_parameters_hc_ffn_base_ = l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_11_parameters_hc_attn_fn_ = l_self_modules_layers_modules_11_parameters_hc_attn_scale_ = l_self_modules_layers_modules_11_parameters_hc_attn_base_ = l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = None
        getitem_44 = submod_22[0]
        getitem_45 = submod_22[1]
        getitem_46 = submod_22[2]
        getitem_47 = submod_22[3];  submod_22 = None
        submod_23 = self.submod_23(getitem_44, s72, l_positions_, s80);  getitem_44 = None
        submod_24 = self.submod_24(submod_23, s72, getitem_45, getitem_46, getitem_47, l_self_modules_layers_modules_11_parameters_hc_ffn_fn_, l_self_modules_layers_modules_11_parameters_hc_ffn_scale_, l_self_modules_layers_modules_11_parameters_hc_ffn_base_, l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_12_parameters_hc_attn_fn_, l_self_modules_layers_modules_12_parameters_hc_attn_scale_, l_self_modules_layers_modules_12_parameters_hc_attn_base_, l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_);  submod_23 = getitem_45 = getitem_46 = getitem_47 = l_self_modules_layers_modules_11_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_11_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_11_parameters_hc_ffn_base_ = l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_12_parameters_hc_attn_fn_ = l_self_modules_layers_modules_12_parameters_hc_attn_scale_ = l_self_modules_layers_modules_12_parameters_hc_attn_base_ = l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = None
        getitem_48 = submod_24[0]
        getitem_49 = submod_24[1]
        getitem_50 = submod_24[2]
        getitem_51 = submod_24[3];  submod_24 = None
        submod_25 = self.submod_25(getitem_48, s72, l_positions_, s80);  getitem_48 = None
        submod_26 = self.submod_26(submod_25, s72, getitem_49, getitem_50, getitem_51, l_self_modules_layers_modules_12_parameters_hc_ffn_fn_, l_self_modules_layers_modules_12_parameters_hc_ffn_scale_, l_self_modules_layers_modules_12_parameters_hc_ffn_base_, l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_13_parameters_hc_attn_fn_, l_self_modules_layers_modules_13_parameters_hc_attn_scale_, l_self_modules_layers_modules_13_parameters_hc_attn_base_, l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_);  submod_25 = getitem_49 = getitem_50 = getitem_51 = l_self_modules_layers_modules_12_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_12_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_12_parameters_hc_ffn_base_ = l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_13_parameters_hc_attn_fn_ = l_self_modules_layers_modules_13_parameters_hc_attn_scale_ = l_self_modules_layers_modules_13_parameters_hc_attn_base_ = l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = None
        getitem_52 = submod_26[0]
        getitem_53 = submod_26[1]
        getitem_54 = submod_26[2]
        getitem_55 = submod_26[3];  submod_26 = None
        submod_27 = self.submod_27(getitem_52, s72, l_positions_, s80);  getitem_52 = None
        submod_28 = self.submod_28(submod_27, s72, getitem_53, getitem_54, getitem_55, l_self_modules_layers_modules_13_parameters_hc_ffn_fn_, l_self_modules_layers_modules_13_parameters_hc_ffn_scale_, l_self_modules_layers_modules_13_parameters_hc_ffn_base_, l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_14_parameters_hc_attn_fn_, l_self_modules_layers_modules_14_parameters_hc_attn_scale_, l_self_modules_layers_modules_14_parameters_hc_attn_base_, l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_);  submod_27 = getitem_53 = getitem_54 = getitem_55 = l_self_modules_layers_modules_13_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_13_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_13_parameters_hc_ffn_base_ = l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_14_parameters_hc_attn_fn_ = l_self_modules_layers_modules_14_parameters_hc_attn_scale_ = l_self_modules_layers_modules_14_parameters_hc_attn_base_ = l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = None
        getitem_56 = submod_28[0]
        getitem_57 = submod_28[1]
        getitem_58 = submod_28[2]
        getitem_59 = submod_28[3];  submod_28 = None
        submod_29 = self.submod_29(getitem_56, s72, l_positions_, s80);  getitem_56 = None
        submod_30 = self.submod_30(submod_29, s72, getitem_57, getitem_58, getitem_59, l_self_modules_layers_modules_14_parameters_hc_ffn_fn_, l_self_modules_layers_modules_14_parameters_hc_ffn_scale_, l_self_modules_layers_modules_14_parameters_hc_ffn_base_, l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_15_parameters_hc_attn_fn_, l_self_modules_layers_modules_15_parameters_hc_attn_scale_, l_self_modules_layers_modules_15_parameters_hc_attn_base_, l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_);  submod_29 = getitem_57 = getitem_58 = getitem_59 = l_self_modules_layers_modules_14_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_14_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_14_parameters_hc_ffn_base_ = l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_15_parameters_hc_attn_fn_ = l_self_modules_layers_modules_15_parameters_hc_attn_scale_ = l_self_modules_layers_modules_15_parameters_hc_attn_base_ = l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = None
        getitem_60 = submod_30[0]
        getitem_61 = submod_30[1]
        getitem_62 = submod_30[2]
        getitem_63 = submod_30[3];  submod_30 = None
        submod_31 = self.submod_31(getitem_60, s72, l_positions_, s80);  getitem_60 = None
        submod_32 = self.submod_32(submod_31, s72, getitem_61, getitem_62, getitem_63, l_self_modules_layers_modules_15_parameters_hc_ffn_fn_, l_self_modules_layers_modules_15_parameters_hc_ffn_scale_, l_self_modules_layers_modules_15_parameters_hc_ffn_base_, l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_16_parameters_hc_attn_fn_, l_self_modules_layers_modules_16_parameters_hc_attn_scale_, l_self_modules_layers_modules_16_parameters_hc_attn_base_, l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_);  submod_31 = getitem_61 = getitem_62 = getitem_63 = l_self_modules_layers_modules_15_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_15_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_15_parameters_hc_ffn_base_ = l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_16_parameters_hc_attn_fn_ = l_self_modules_layers_modules_16_parameters_hc_attn_scale_ = l_self_modules_layers_modules_16_parameters_hc_attn_base_ = l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = None
        getitem_64 = submod_32[0]
        getitem_65 = submod_32[1]
        getitem_66 = submod_32[2]
        getitem_67 = submod_32[3];  submod_32 = None
        submod_33 = self.submod_33(getitem_64, s72, l_positions_, s80);  getitem_64 = None
        submod_34 = self.submod_34(submod_33, s72, getitem_65, getitem_66, getitem_67, l_self_modules_layers_modules_16_parameters_hc_ffn_fn_, l_self_modules_layers_modules_16_parameters_hc_ffn_scale_, l_self_modules_layers_modules_16_parameters_hc_ffn_base_, l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_17_parameters_hc_attn_fn_, l_self_modules_layers_modules_17_parameters_hc_attn_scale_, l_self_modules_layers_modules_17_parameters_hc_attn_base_, l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_);  submod_33 = getitem_65 = getitem_66 = getitem_67 = l_self_modules_layers_modules_16_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_16_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_16_parameters_hc_ffn_base_ = l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_17_parameters_hc_attn_fn_ = l_self_modules_layers_modules_17_parameters_hc_attn_scale_ = l_self_modules_layers_modules_17_parameters_hc_attn_base_ = l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = None
        getitem_68 = submod_34[0]
        getitem_69 = submod_34[1]
        getitem_70 = submod_34[2]
        getitem_71 = submod_34[3];  submod_34 = None
        submod_35 = self.submod_35(getitem_68, s72, l_positions_, s80);  getitem_68 = None
        submod_36 = self.submod_36(submod_35, s72, getitem_69, getitem_70, getitem_71, l_self_modules_layers_modules_17_parameters_hc_ffn_fn_, l_self_modules_layers_modules_17_parameters_hc_ffn_scale_, l_self_modules_layers_modules_17_parameters_hc_ffn_base_, l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_18_parameters_hc_attn_fn_, l_self_modules_layers_modules_18_parameters_hc_attn_scale_, l_self_modules_layers_modules_18_parameters_hc_attn_base_, l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_);  submod_35 = getitem_69 = getitem_70 = getitem_71 = l_self_modules_layers_modules_17_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_17_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_17_parameters_hc_ffn_base_ = l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_18_parameters_hc_attn_fn_ = l_self_modules_layers_modules_18_parameters_hc_attn_scale_ = l_self_modules_layers_modules_18_parameters_hc_attn_base_ = l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = None
        getitem_72 = submod_36[0]
        getitem_73 = submod_36[1]
        getitem_74 = submod_36[2]
        getitem_75 = submod_36[3];  submod_36 = None
        submod_37 = self.submod_37(getitem_72, s72, l_positions_, s80);  getitem_72 = None
        submod_38 = self.submod_38(submod_37, s72, getitem_73, getitem_74, getitem_75, l_self_modules_layers_modules_18_parameters_hc_ffn_fn_, l_self_modules_layers_modules_18_parameters_hc_ffn_scale_, l_self_modules_layers_modules_18_parameters_hc_ffn_base_, l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_19_parameters_hc_attn_fn_, l_self_modules_layers_modules_19_parameters_hc_attn_scale_, l_self_modules_layers_modules_19_parameters_hc_attn_base_, l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_);  submod_37 = getitem_73 = getitem_74 = getitem_75 = l_self_modules_layers_modules_18_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_18_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_18_parameters_hc_ffn_base_ = l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_19_parameters_hc_attn_fn_ = l_self_modules_layers_modules_19_parameters_hc_attn_scale_ = l_self_modules_layers_modules_19_parameters_hc_attn_base_ = l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = None
        getitem_76 = submod_38[0]
        getitem_77 = submod_38[1]
        getitem_78 = submod_38[2]
        getitem_79 = submod_38[3];  submod_38 = None
        submod_39 = self.submod_39(getitem_76, s72, l_positions_, s80);  getitem_76 = None
        submod_40 = self.submod_40(submod_39, s72, getitem_77, getitem_78, getitem_79, l_self_modules_layers_modules_19_parameters_hc_ffn_fn_, l_self_modules_layers_modules_19_parameters_hc_ffn_scale_, l_self_modules_layers_modules_19_parameters_hc_ffn_base_, l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_20_parameters_hc_attn_fn_, l_self_modules_layers_modules_20_parameters_hc_attn_scale_, l_self_modules_layers_modules_20_parameters_hc_attn_base_, l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_);  submod_39 = getitem_77 = getitem_78 = getitem_79 = l_self_modules_layers_modules_19_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_19_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_19_parameters_hc_ffn_base_ = l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_20_parameters_hc_attn_fn_ = l_self_modules_layers_modules_20_parameters_hc_attn_scale_ = l_self_modules_layers_modules_20_parameters_hc_attn_base_ = l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = None
        getitem_80 = submod_40[0]
        getitem_81 = submod_40[1]
        getitem_82 = submod_40[2]
        getitem_83 = submod_40[3];  submod_40 = None
        submod_41 = self.submod_41(getitem_80, s72, l_positions_, s80);  getitem_80 = None
        submod_42 = self.submod_42(submod_41, s72, getitem_81, getitem_82, getitem_83, l_self_modules_layers_modules_20_parameters_hc_ffn_fn_, l_self_modules_layers_modules_20_parameters_hc_ffn_scale_, l_self_modules_layers_modules_20_parameters_hc_ffn_base_, l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_21_parameters_hc_attn_fn_, l_self_modules_layers_modules_21_parameters_hc_attn_scale_, l_self_modules_layers_modules_21_parameters_hc_attn_base_, l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_);  submod_41 = getitem_81 = getitem_82 = getitem_83 = l_self_modules_layers_modules_20_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_20_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_20_parameters_hc_ffn_base_ = l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_21_parameters_hc_attn_fn_ = l_self_modules_layers_modules_21_parameters_hc_attn_scale_ = l_self_modules_layers_modules_21_parameters_hc_attn_base_ = l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = None
        getitem_84 = submod_42[0]
        getitem_85 = submod_42[1]
        getitem_86 = submod_42[2]
        getitem_87 = submod_42[3];  submod_42 = None
        submod_43 = self.submod_43(getitem_84, s72, l_positions_, s80);  getitem_84 = None
        submod_44 = self.submod_44(submod_43, s72, getitem_85, getitem_86, getitem_87, l_self_modules_layers_modules_21_parameters_hc_ffn_fn_, l_self_modules_layers_modules_21_parameters_hc_ffn_scale_, l_self_modules_layers_modules_21_parameters_hc_ffn_base_, l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_22_parameters_hc_attn_fn_, l_self_modules_layers_modules_22_parameters_hc_attn_scale_, l_self_modules_layers_modules_22_parameters_hc_attn_base_, l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_);  submod_43 = getitem_85 = getitem_86 = getitem_87 = l_self_modules_layers_modules_21_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_21_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_21_parameters_hc_ffn_base_ = l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_22_parameters_hc_attn_fn_ = l_self_modules_layers_modules_22_parameters_hc_attn_scale_ = l_self_modules_layers_modules_22_parameters_hc_attn_base_ = l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_ = None
        getitem_88 = submod_44[0]
        getitem_89 = submod_44[1]
        getitem_90 = submod_44[2]
        getitem_91 = submod_44[3];  submod_44 = None
        submod_45 = self.submod_45(getitem_88, s72, l_positions_, s80);  getitem_88 = None
        submod_46 = self.submod_46(submod_45, s72, getitem_89, getitem_90, getitem_91, l_self_modules_layers_modules_22_parameters_hc_ffn_fn_, l_self_modules_layers_modules_22_parameters_hc_ffn_scale_, l_self_modules_layers_modules_22_parameters_hc_ffn_base_, l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_23_parameters_hc_attn_fn_, l_self_modules_layers_modules_23_parameters_hc_attn_scale_, l_self_modules_layers_modules_23_parameters_hc_attn_base_, l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_);  submod_45 = getitem_89 = getitem_90 = getitem_91 = l_self_modules_layers_modules_22_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_22_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_22_parameters_hc_ffn_base_ = l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_23_parameters_hc_attn_fn_ = l_self_modules_layers_modules_23_parameters_hc_attn_scale_ = l_self_modules_layers_modules_23_parameters_hc_attn_base_ = l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_ = None
        getitem_92 = submod_46[0]
        getitem_93 = submod_46[1]
        getitem_94 = submod_46[2]
        getitem_95 = submod_46[3];  submod_46 = None
        submod_47 = self.submod_47(getitem_92, s72, l_positions_, s80);  getitem_92 = None
        submod_48 = self.submod_48(submod_47, s72, getitem_93, getitem_94, getitem_95, l_self_modules_layers_modules_23_parameters_hc_ffn_fn_, l_self_modules_layers_modules_23_parameters_hc_ffn_scale_, l_self_modules_layers_modules_23_parameters_hc_ffn_base_, l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_24_parameters_hc_attn_fn_, l_self_modules_layers_modules_24_parameters_hc_attn_scale_, l_self_modules_layers_modules_24_parameters_hc_attn_base_, l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_);  submod_47 = getitem_93 = getitem_94 = getitem_95 = l_self_modules_layers_modules_23_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_23_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_23_parameters_hc_ffn_base_ = l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_24_parameters_hc_attn_fn_ = l_self_modules_layers_modules_24_parameters_hc_attn_scale_ = l_self_modules_layers_modules_24_parameters_hc_attn_base_ = l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_ = None
        getitem_96 = submod_48[0]
        getitem_97 = submod_48[1]
        getitem_98 = submod_48[2]
        getitem_99 = submod_48[3];  submod_48 = None
        submod_49 = self.submod_49(getitem_96, s72, l_positions_, s80);  getitem_96 = None
        submod_50 = self.submod_50(submod_49, s72, getitem_97, getitem_98, getitem_99, l_self_modules_layers_modules_24_parameters_hc_ffn_fn_, l_self_modules_layers_modules_24_parameters_hc_ffn_scale_, l_self_modules_layers_modules_24_parameters_hc_ffn_base_, l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_25_parameters_hc_attn_fn_, l_self_modules_layers_modules_25_parameters_hc_attn_scale_, l_self_modules_layers_modules_25_parameters_hc_attn_base_, l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_);  submod_49 = getitem_97 = getitem_98 = getitem_99 = l_self_modules_layers_modules_24_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_24_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_24_parameters_hc_ffn_base_ = l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_25_parameters_hc_attn_fn_ = l_self_modules_layers_modules_25_parameters_hc_attn_scale_ = l_self_modules_layers_modules_25_parameters_hc_attn_base_ = l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_ = None
        getitem_100 = submod_50[0]
        getitem_101 = submod_50[1]
        getitem_102 = submod_50[2]
        getitem_103 = submod_50[3];  submod_50 = None
        submod_51 = self.submod_51(getitem_100, s72, l_positions_, s80);  getitem_100 = None
        submod_52 = self.submod_52(submod_51, s72, getitem_101, getitem_102, getitem_103, l_self_modules_layers_modules_25_parameters_hc_ffn_fn_, l_self_modules_layers_modules_25_parameters_hc_ffn_scale_, l_self_modules_layers_modules_25_parameters_hc_ffn_base_, l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_26_parameters_hc_attn_fn_, l_self_modules_layers_modules_26_parameters_hc_attn_scale_, l_self_modules_layers_modules_26_parameters_hc_attn_base_, l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_);  submod_51 = getitem_101 = getitem_102 = getitem_103 = l_self_modules_layers_modules_25_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_25_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_25_parameters_hc_ffn_base_ = l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_26_parameters_hc_attn_fn_ = l_self_modules_layers_modules_26_parameters_hc_attn_scale_ = l_self_modules_layers_modules_26_parameters_hc_attn_base_ = l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_ = None
        getitem_104 = submod_52[0]
        getitem_105 = submod_52[1]
        getitem_106 = submod_52[2]
        getitem_107 = submod_52[3];  submod_52 = None
        submod_53 = self.submod_53(getitem_104, s72, l_positions_, s80);  getitem_104 = None
        submod_54 = self.submod_54(submod_53, s72, getitem_105, getitem_106, getitem_107, l_self_modules_layers_modules_26_parameters_hc_ffn_fn_, l_self_modules_layers_modules_26_parameters_hc_ffn_scale_, l_self_modules_layers_modules_26_parameters_hc_ffn_base_, l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_27_parameters_hc_attn_fn_, l_self_modules_layers_modules_27_parameters_hc_attn_scale_, l_self_modules_layers_modules_27_parameters_hc_attn_base_, l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_);  submod_53 = getitem_105 = getitem_106 = getitem_107 = l_self_modules_layers_modules_26_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_26_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_26_parameters_hc_ffn_base_ = l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_27_parameters_hc_attn_fn_ = l_self_modules_layers_modules_27_parameters_hc_attn_scale_ = l_self_modules_layers_modules_27_parameters_hc_attn_base_ = l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_ = None
        getitem_108 = submod_54[0]
        getitem_109 = submod_54[1]
        getitem_110 = submod_54[2]
        getitem_111 = submod_54[3];  submod_54 = None
        submod_55 = self.submod_55(getitem_108, s72, l_positions_, s80);  getitem_108 = None
        submod_56 = self.submod_56(submod_55, s72, getitem_109, getitem_110, getitem_111, l_self_modules_layers_modules_27_parameters_hc_ffn_fn_, l_self_modules_layers_modules_27_parameters_hc_ffn_scale_, l_self_modules_layers_modules_27_parameters_hc_ffn_base_, l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_28_parameters_hc_attn_fn_, l_self_modules_layers_modules_28_parameters_hc_attn_scale_, l_self_modules_layers_modules_28_parameters_hc_attn_base_, l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_);  submod_55 = getitem_109 = getitem_110 = getitem_111 = l_self_modules_layers_modules_27_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_27_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_27_parameters_hc_ffn_base_ = l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_28_parameters_hc_attn_fn_ = l_self_modules_layers_modules_28_parameters_hc_attn_scale_ = l_self_modules_layers_modules_28_parameters_hc_attn_base_ = l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_ = None
        getitem_112 = submod_56[0]
        getitem_113 = submod_56[1]
        getitem_114 = submod_56[2]
        getitem_115 = submod_56[3];  submod_56 = None
        submod_57 = self.submod_57(getitem_112, s72, l_positions_, s80);  getitem_112 = None
        submod_58 = self.submod_58(submod_57, s72, getitem_113, getitem_114, getitem_115, l_self_modules_layers_modules_28_parameters_hc_ffn_fn_, l_self_modules_layers_modules_28_parameters_hc_ffn_scale_, l_self_modules_layers_modules_28_parameters_hc_ffn_base_, l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_29_parameters_hc_attn_fn_, l_self_modules_layers_modules_29_parameters_hc_attn_scale_, l_self_modules_layers_modules_29_parameters_hc_attn_base_, l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_);  submod_57 = getitem_113 = getitem_114 = getitem_115 = l_self_modules_layers_modules_28_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_28_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_28_parameters_hc_ffn_base_ = l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_29_parameters_hc_attn_fn_ = l_self_modules_layers_modules_29_parameters_hc_attn_scale_ = l_self_modules_layers_modules_29_parameters_hc_attn_base_ = l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_ = None
        getitem_116 = submod_58[0]
        getitem_117 = submod_58[1]
        getitem_118 = submod_58[2]
        getitem_119 = submod_58[3];  submod_58 = None
        submod_59 = self.submod_59(getitem_116, s72, l_positions_, s80);  getitem_116 = None
        submod_60 = self.submod_60(submod_59, s72, getitem_117, getitem_118, getitem_119, l_self_modules_layers_modules_29_parameters_hc_ffn_fn_, l_self_modules_layers_modules_29_parameters_hc_ffn_scale_, l_self_modules_layers_modules_29_parameters_hc_ffn_base_, l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_30_parameters_hc_attn_fn_, l_self_modules_layers_modules_30_parameters_hc_attn_scale_, l_self_modules_layers_modules_30_parameters_hc_attn_base_, l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_);  submod_59 = getitem_117 = getitem_118 = getitem_119 = l_self_modules_layers_modules_29_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_29_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_29_parameters_hc_ffn_base_ = l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_30_parameters_hc_attn_fn_ = l_self_modules_layers_modules_30_parameters_hc_attn_scale_ = l_self_modules_layers_modules_30_parameters_hc_attn_base_ = l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_ = None
        getitem_120 = submod_60[0]
        getitem_121 = submod_60[1]
        getitem_122 = submod_60[2]
        getitem_123 = submod_60[3];  submod_60 = None
        submod_61 = self.submod_61(getitem_120, s72, l_positions_, s80);  getitem_120 = None
        submod_62 = self.submod_62(submod_61, s72, getitem_121, getitem_122, getitem_123, l_self_modules_layers_modules_30_parameters_hc_ffn_fn_, l_self_modules_layers_modules_30_parameters_hc_ffn_scale_, l_self_modules_layers_modules_30_parameters_hc_ffn_base_, l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_31_parameters_hc_attn_fn_, l_self_modules_layers_modules_31_parameters_hc_attn_scale_, l_self_modules_layers_modules_31_parameters_hc_attn_base_, l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_);  submod_61 = getitem_121 = getitem_122 = getitem_123 = l_self_modules_layers_modules_30_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_30_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_30_parameters_hc_ffn_base_ = l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_31_parameters_hc_attn_fn_ = l_self_modules_layers_modules_31_parameters_hc_attn_scale_ = l_self_modules_layers_modules_31_parameters_hc_attn_base_ = l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_ = None
        getitem_124 = submod_62[0]
        getitem_125 = submod_62[1]
        getitem_126 = submod_62[2]
        getitem_127 = submod_62[3];  submod_62 = None
        submod_63 = self.submod_63(getitem_124, s72, l_positions_, s80);  getitem_124 = None
        submod_64 = self.submod_64(submod_63, s72, getitem_125, getitem_126, getitem_127, l_self_modules_layers_modules_31_parameters_hc_ffn_fn_, l_self_modules_layers_modules_31_parameters_hc_ffn_scale_, l_self_modules_layers_modules_31_parameters_hc_ffn_base_, l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_32_parameters_hc_attn_fn_, l_self_modules_layers_modules_32_parameters_hc_attn_scale_, l_self_modules_layers_modules_32_parameters_hc_attn_base_, l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_);  submod_63 = getitem_125 = getitem_126 = getitem_127 = l_self_modules_layers_modules_31_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_31_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_31_parameters_hc_ffn_base_ = l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_32_parameters_hc_attn_fn_ = l_self_modules_layers_modules_32_parameters_hc_attn_scale_ = l_self_modules_layers_modules_32_parameters_hc_attn_base_ = l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_ = None
        getitem_128 = submod_64[0]
        getitem_129 = submod_64[1]
        getitem_130 = submod_64[2]
        getitem_131 = submod_64[3];  submod_64 = None
        submod_65 = self.submod_65(getitem_128, s72, l_positions_, s80);  getitem_128 = None
        submod_66 = self.submod_66(submod_65, s72, getitem_129, getitem_130, getitem_131, l_self_modules_layers_modules_32_parameters_hc_ffn_fn_, l_self_modules_layers_modules_32_parameters_hc_ffn_scale_, l_self_modules_layers_modules_32_parameters_hc_ffn_base_, l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_33_parameters_hc_attn_fn_, l_self_modules_layers_modules_33_parameters_hc_attn_scale_, l_self_modules_layers_modules_33_parameters_hc_attn_base_, l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_);  submod_65 = getitem_129 = getitem_130 = getitem_131 = l_self_modules_layers_modules_32_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_32_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_32_parameters_hc_ffn_base_ = l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_33_parameters_hc_attn_fn_ = l_self_modules_layers_modules_33_parameters_hc_attn_scale_ = l_self_modules_layers_modules_33_parameters_hc_attn_base_ = l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_ = None
        getitem_132 = submod_66[0]
        getitem_133 = submod_66[1]
        getitem_134 = submod_66[2]
        getitem_135 = submod_66[3];  submod_66 = None
        submod_67 = self.submod_67(getitem_132, s72, l_positions_, s80);  getitem_132 = None
        submod_68 = self.submod_68(submod_67, s72, getitem_133, getitem_134, getitem_135, l_self_modules_layers_modules_33_parameters_hc_ffn_fn_, l_self_modules_layers_modules_33_parameters_hc_ffn_scale_, l_self_modules_layers_modules_33_parameters_hc_ffn_base_, l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_34_parameters_hc_attn_fn_, l_self_modules_layers_modules_34_parameters_hc_attn_scale_, l_self_modules_layers_modules_34_parameters_hc_attn_base_, l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_);  submod_67 = getitem_133 = getitem_134 = getitem_135 = l_self_modules_layers_modules_33_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_33_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_33_parameters_hc_ffn_base_ = l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_34_parameters_hc_attn_fn_ = l_self_modules_layers_modules_34_parameters_hc_attn_scale_ = l_self_modules_layers_modules_34_parameters_hc_attn_base_ = l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_ = None
        getitem_136 = submod_68[0]
        getitem_137 = submod_68[1]
        getitem_138 = submod_68[2]
        getitem_139 = submod_68[3];  submod_68 = None
        submod_69 = self.submod_69(getitem_136, s72, l_positions_, s80);  getitem_136 = None
        submod_70 = self.submod_70(submod_69, s72, getitem_137, getitem_138, getitem_139, l_self_modules_layers_modules_34_parameters_hc_ffn_fn_, l_self_modules_layers_modules_34_parameters_hc_ffn_scale_, l_self_modules_layers_modules_34_parameters_hc_ffn_base_, l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_35_parameters_hc_attn_fn_, l_self_modules_layers_modules_35_parameters_hc_attn_scale_, l_self_modules_layers_modules_35_parameters_hc_attn_base_, l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_);  submod_69 = getitem_137 = getitem_138 = getitem_139 = l_self_modules_layers_modules_34_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_34_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_34_parameters_hc_ffn_base_ = l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_35_parameters_hc_attn_fn_ = l_self_modules_layers_modules_35_parameters_hc_attn_scale_ = l_self_modules_layers_modules_35_parameters_hc_attn_base_ = l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_ = None
        getitem_140 = submod_70[0]
        getitem_141 = submod_70[1]
        getitem_142 = submod_70[2]
        getitem_143 = submod_70[3];  submod_70 = None
        submod_71 = self.submod_71(getitem_140, s72, l_positions_, s80);  getitem_140 = None
        submod_72 = self.submod_72(submod_71, s72, getitem_141, getitem_142, getitem_143, l_self_modules_layers_modules_35_parameters_hc_ffn_fn_, l_self_modules_layers_modules_35_parameters_hc_ffn_scale_, l_self_modules_layers_modules_35_parameters_hc_ffn_base_, l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_36_parameters_hc_attn_fn_, l_self_modules_layers_modules_36_parameters_hc_attn_scale_, l_self_modules_layers_modules_36_parameters_hc_attn_base_, l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_);  submod_71 = getitem_141 = getitem_142 = getitem_143 = l_self_modules_layers_modules_35_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_35_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_35_parameters_hc_ffn_base_ = l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_36_parameters_hc_attn_fn_ = l_self_modules_layers_modules_36_parameters_hc_attn_scale_ = l_self_modules_layers_modules_36_parameters_hc_attn_base_ = l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_ = None
        getitem_144 = submod_72[0]
        getitem_145 = submod_72[1]
        getitem_146 = submod_72[2]
        getitem_147 = submod_72[3];  submod_72 = None
        submod_73 = self.submod_73(getitem_144, s72, l_positions_, s80);  getitem_144 = None
        submod_74 = self.submod_74(submod_73, s72, getitem_145, getitem_146, getitem_147, l_self_modules_layers_modules_36_parameters_hc_ffn_fn_, l_self_modules_layers_modules_36_parameters_hc_ffn_scale_, l_self_modules_layers_modules_36_parameters_hc_ffn_base_, l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_37_parameters_hc_attn_fn_, l_self_modules_layers_modules_37_parameters_hc_attn_scale_, l_self_modules_layers_modules_37_parameters_hc_attn_base_, l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_);  submod_73 = getitem_145 = getitem_146 = getitem_147 = l_self_modules_layers_modules_36_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_36_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_36_parameters_hc_ffn_base_ = l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_37_parameters_hc_attn_fn_ = l_self_modules_layers_modules_37_parameters_hc_attn_scale_ = l_self_modules_layers_modules_37_parameters_hc_attn_base_ = l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_ = None
        getitem_148 = submod_74[0]
        getitem_149 = submod_74[1]
        getitem_150 = submod_74[2]
        getitem_151 = submod_74[3];  submod_74 = None
        submod_75 = self.submod_75(getitem_148, s72, l_positions_, s80);  getitem_148 = None
        submod_76 = self.submod_76(submod_75, s72, getitem_149, getitem_150, getitem_151, l_self_modules_layers_modules_37_parameters_hc_ffn_fn_, l_self_modules_layers_modules_37_parameters_hc_ffn_scale_, l_self_modules_layers_modules_37_parameters_hc_ffn_base_, l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_38_parameters_hc_attn_fn_, l_self_modules_layers_modules_38_parameters_hc_attn_scale_, l_self_modules_layers_modules_38_parameters_hc_attn_base_, l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_);  submod_75 = getitem_149 = getitem_150 = getitem_151 = l_self_modules_layers_modules_37_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_37_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_37_parameters_hc_ffn_base_ = l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_38_parameters_hc_attn_fn_ = l_self_modules_layers_modules_38_parameters_hc_attn_scale_ = l_self_modules_layers_modules_38_parameters_hc_attn_base_ = l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_ = None
        getitem_152 = submod_76[0]
        getitem_153 = submod_76[1]
        getitem_154 = submod_76[2]
        getitem_155 = submod_76[3];  submod_76 = None
        submod_77 = self.submod_77(getitem_152, s72, l_positions_, s80);  getitem_152 = None
        submod_78 = self.submod_78(submod_77, s72, getitem_153, getitem_154, getitem_155, l_self_modules_layers_modules_38_parameters_hc_ffn_fn_, l_self_modules_layers_modules_38_parameters_hc_ffn_scale_, l_self_modules_layers_modules_38_parameters_hc_ffn_base_, l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_39_parameters_hc_attn_fn_, l_self_modules_layers_modules_39_parameters_hc_attn_scale_, l_self_modules_layers_modules_39_parameters_hc_attn_base_, l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_);  submod_77 = getitem_153 = getitem_154 = getitem_155 = l_self_modules_layers_modules_38_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_38_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_38_parameters_hc_ffn_base_ = l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_39_parameters_hc_attn_fn_ = l_self_modules_layers_modules_39_parameters_hc_attn_scale_ = l_self_modules_layers_modules_39_parameters_hc_attn_base_ = l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_ = None
        getitem_156 = submod_78[0]
        getitem_157 = submod_78[1]
        getitem_158 = submod_78[2]
        getitem_159 = submod_78[3];  submod_78 = None
        submod_79 = self.submod_79(getitem_156, s72, l_positions_, s80);  getitem_156 = None
        submod_80 = self.submod_80(submod_79, s72, getitem_157, getitem_158, getitem_159, l_self_modules_layers_modules_39_parameters_hc_ffn_fn_, l_self_modules_layers_modules_39_parameters_hc_ffn_scale_, l_self_modules_layers_modules_39_parameters_hc_ffn_base_, l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_40_parameters_hc_attn_fn_, l_self_modules_layers_modules_40_parameters_hc_attn_scale_, l_self_modules_layers_modules_40_parameters_hc_attn_base_, l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_);  submod_79 = getitem_157 = getitem_158 = getitem_159 = l_self_modules_layers_modules_39_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_39_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_39_parameters_hc_ffn_base_ = l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_40_parameters_hc_attn_fn_ = l_self_modules_layers_modules_40_parameters_hc_attn_scale_ = l_self_modules_layers_modules_40_parameters_hc_attn_base_ = l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_ = None
        getitem_160 = submod_80[0]
        getitem_161 = submod_80[1]
        getitem_162 = submod_80[2]
        getitem_163 = submod_80[3];  submod_80 = None
        submod_81 = self.submod_81(getitem_160, s72, l_positions_, s80);  getitem_160 = None
        submod_82 = self.submod_82(submod_81, s72, getitem_161, getitem_162, getitem_163, l_self_modules_layers_modules_40_parameters_hc_ffn_fn_, l_self_modules_layers_modules_40_parameters_hc_ffn_scale_, l_self_modules_layers_modules_40_parameters_hc_ffn_base_, l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_41_parameters_hc_attn_fn_, l_self_modules_layers_modules_41_parameters_hc_attn_scale_, l_self_modules_layers_modules_41_parameters_hc_attn_base_, l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_);  submod_81 = getitem_161 = getitem_162 = getitem_163 = l_self_modules_layers_modules_40_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_40_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_40_parameters_hc_ffn_base_ = l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_41_parameters_hc_attn_fn_ = l_self_modules_layers_modules_41_parameters_hc_attn_scale_ = l_self_modules_layers_modules_41_parameters_hc_attn_base_ = l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_ = None
        getitem_164 = submod_82[0]
        getitem_165 = submod_82[1]
        getitem_166 = submod_82[2]
        getitem_167 = submod_82[3];  submod_82 = None
        submod_83 = self.submod_83(getitem_164, s72, l_positions_, s80);  getitem_164 = None
        submod_84 = self.submod_84(submod_83, s72, getitem_165, getitem_166, getitem_167, l_self_modules_layers_modules_41_parameters_hc_ffn_fn_, l_self_modules_layers_modules_41_parameters_hc_ffn_scale_, l_self_modules_layers_modules_41_parameters_hc_ffn_base_, l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_42_parameters_hc_attn_fn_, l_self_modules_layers_modules_42_parameters_hc_attn_scale_, l_self_modules_layers_modules_42_parameters_hc_attn_base_, l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_);  submod_83 = getitem_165 = getitem_166 = getitem_167 = l_self_modules_layers_modules_41_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_41_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_41_parameters_hc_ffn_base_ = l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_42_parameters_hc_attn_fn_ = l_self_modules_layers_modules_42_parameters_hc_attn_scale_ = l_self_modules_layers_modules_42_parameters_hc_attn_base_ = l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_ = None
        getitem_168 = submod_84[0]
        getitem_169 = submod_84[1]
        getitem_170 = submod_84[2]
        getitem_171 = submod_84[3];  submod_84 = None
        submod_85 = self.submod_85(getitem_168, s72, l_positions_, s80);  getitem_168 = l_positions_ = s80 = None
        submod_86 = self.submod_86(submod_85, s72, getitem_169, getitem_170, getitem_171, l_self_modules_layers_modules_42_parameters_hc_ffn_fn_, l_self_modules_layers_modules_42_parameters_hc_ffn_scale_, l_self_modules_layers_modules_42_parameters_hc_ffn_base_, l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_);  submod_85 = s72 = getitem_169 = getitem_170 = getitem_171 = l_self_modules_layers_modules_42_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_42_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_42_parameters_hc_ffn_base_ = l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_ = None
        return (submod_86,)

    class submod_0(torch.nn.Module):
        def forward(self, l_input_ids_: "i32[s72]", s72: "Sym(s72)", l_self_modules_embed_parameters_weight_: "bf16[129280, 4096]", l_self_modules_layers_modules_0_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_0_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_0_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            embedding: "bf16[s72, 4096]" = torch.nn.functional.embedding(l_input_ids_, l_self_modules_embed_parameters_weight_);  l_input_ids_ = l_self_modules_embed_parameters_weight_ = None
            mhc_res_repeat: "bf16[s72, 4, 4096]" = torch.ops.aiter.mhc_res_repeat(embedding, 4, True);  embedding = None
            mhc_pre = torch.ops.aiter.mhc_pre(mhc_res_repeat, l_self_modules_layers_modules_0_parameters_hc_attn_fn_, l_self_modules_layers_modules_0_parameters_hc_attn_scale_, l_self_modules_layers_modules_0_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = 1, res_preshuffle = True);  l_self_modules_layers_modules_0_parameters_hc_attn_fn_ = l_self_modules_layers_modules_0_parameters_hc_attn_scale_ = l_self_modules_layers_modules_0_parameters_hc_attn_base_ = l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            return (getitem_2, mhc_res_repeat, squeeze, getitem_1)

    class submod_1(torch.nn.Module):
        def forward(self, y_1: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_1, l_positions_, 'layers.0.attn', None, None);  y_1 = l_positions_ = None
            return v4_attention_with_output

    class submod_2(torch.nn.Module):
        def forward(self, x: "bf16[s72, 4096]", s72: "Sym(s72)", result: "bf16[s72, 4, 4096]", post_1: "f32[s72, 4]", comb: "f32[s72, 4, 4]", l_self_modules_layers_modules_0_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_0_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_0_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_1_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_1_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_1_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x, result, post_1, comb, l_self_modules_layers_modules_0_parameters_hc_ffn_fn_, l_self_modules_layers_modules_0_parameters_hc_ffn_scale_, l_self_modules_layers_modules_0_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x = result = post_1 = comb = l_self_modules_layers_modules_0_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_0_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_0_parameters_hc_ffn_base_ = l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.0.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_1_parameters_hc_attn_fn_, l_self_modules_layers_modules_1_parameters_hc_attn_scale_, l_self_modules_layers_modules_1_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_1_parameters_hc_attn_fn_ = l_self_modules_layers_modules_1_parameters_hc_attn_scale_ = l_self_modules_layers_modules_1_parameters_hc_attn_base_ = l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_3(torch.nn.Module):
        def forward(self, x_3: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_3, l_positions_, 'layers.1.attn', None, None);  x_3 = l_positions_ = None
            return v4_attention_with_output

    class submod_4(torch.nn.Module):
        def forward(self, x_4: "bf16[s72, 4096]", s72: "Sym(s72)", res_1: "bf16[s72, 4, 4096]", post_5: "f32[s72, 4]", comb_2: "f32[s72, 4, 4]", l_self_modules_layers_modules_1_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_1_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_1_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_2_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_2_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_2_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_4, res_1, post_5, comb_2, l_self_modules_layers_modules_1_parameters_hc_ffn_fn_, l_self_modules_layers_modules_1_parameters_hc_ffn_scale_, l_self_modules_layers_modules_1_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_4 = res_1 = post_5 = comb_2 = l_self_modules_layers_modules_1_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_1_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_1_parameters_hc_ffn_base_ = l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.1.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_2_parameters_hc_attn_fn_, l_self_modules_layers_modules_2_parameters_hc_attn_scale_, l_self_modules_layers_modules_2_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_2_parameters_hc_attn_fn_ = l_self_modules_layers_modules_2_parameters_hc_attn_scale_ = l_self_modules_layers_modules_2_parameters_hc_attn_base_ = l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_5(torch.nn.Module):
        def forward(self, x_7: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_7, l_positions_, 'layers.2.attn', None, None);  x_7 = l_positions_ = None
            return v4_attention_with_output

    class submod_6(torch.nn.Module):
        def forward(self, x_8: "bf16[s72, 4096]", s72: "Sym(s72)", res_3: "bf16[s72, 4, 4096]", post_9: "f32[s72, 4]", comb_4: "f32[s72, 4, 4]", l_self_modules_layers_modules_2_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_2_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_2_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_3_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_3_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_3_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_8, res_3, post_9, comb_4, l_self_modules_layers_modules_2_parameters_hc_ffn_fn_, l_self_modules_layers_modules_2_parameters_hc_ffn_scale_, l_self_modules_layers_modules_2_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_8 = res_3 = post_9 = comb_4 = l_self_modules_layers_modules_2_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_2_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_2_parameters_hc_ffn_base_ = l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.2.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_3_parameters_hc_attn_fn_, l_self_modules_layers_modules_3_parameters_hc_attn_scale_, l_self_modules_layers_modules_3_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_3_parameters_hc_attn_fn_ = l_self_modules_layers_modules_3_parameters_hc_attn_scale_ = l_self_modules_layers_modules_3_parameters_hc_attn_base_ = l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_7(torch.nn.Module):
        def forward(self, x_11: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_11, l_positions_, 'layers.3.attn', None, None);  x_11 = l_positions_ = None
            return v4_attention_with_output

    class submod_8(torch.nn.Module):
        def forward(self, x_12: "bf16[s72, 4096]", s72: "Sym(s72)", res_5: "bf16[s72, 4, 4096]", post_13: "f32[s72, 4]", comb_6: "f32[s72, 4, 4]", l_self_modules_layers_modules_3_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_3_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_3_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_4_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_4_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_4_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_12, res_5, post_13, comb_6, l_self_modules_layers_modules_3_parameters_hc_ffn_fn_, l_self_modules_layers_modules_3_parameters_hc_ffn_scale_, l_self_modules_layers_modules_3_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_12 = res_5 = post_13 = comb_6 = l_self_modules_layers_modules_3_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_3_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_3_parameters_hc_ffn_base_ = l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.3.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_4_parameters_hc_attn_fn_, l_self_modules_layers_modules_4_parameters_hc_attn_scale_, l_self_modules_layers_modules_4_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_4_parameters_hc_attn_fn_ = l_self_modules_layers_modules_4_parameters_hc_attn_scale_ = l_self_modules_layers_modules_4_parameters_hc_attn_base_ = l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_9(torch.nn.Module):
        def forward(self, x_15: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_15, l_positions_, 'layers.4.attn', None, None);  x_15 = l_positions_ = None
            return v4_attention_with_output

    class submod_10(torch.nn.Module):
        def forward(self, x_16: "bf16[s72, 4096]", s72: "Sym(s72)", res_7: "bf16[s72, 4, 4096]", post_17: "f32[s72, 4]", comb_8: "f32[s72, 4, 4]", l_self_modules_layers_modules_4_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_4_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_4_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_5_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_5_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_5_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_16, res_7, post_17, comb_8, l_self_modules_layers_modules_4_parameters_hc_ffn_fn_, l_self_modules_layers_modules_4_parameters_hc_ffn_scale_, l_self_modules_layers_modules_4_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_16 = res_7 = post_17 = comb_8 = l_self_modules_layers_modules_4_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_4_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_4_parameters_hc_ffn_base_ = l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.4.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_5_parameters_hc_attn_fn_, l_self_modules_layers_modules_5_parameters_hc_attn_scale_, l_self_modules_layers_modules_5_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_5_parameters_hc_attn_fn_ = l_self_modules_layers_modules_5_parameters_hc_attn_scale_ = l_self_modules_layers_modules_5_parameters_hc_attn_base_ = l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_11(torch.nn.Module):
        def forward(self, x_19: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_19, l_positions_, 'layers.5.attn', None, None);  x_19 = l_positions_ = None
            return v4_attention_with_output

    class submod_12(torch.nn.Module):
        def forward(self, x_20: "bf16[s72, 4096]", s72: "Sym(s72)", res_9: "bf16[s72, 4, 4096]", post_21: "f32[s72, 4]", comb_10: "f32[s72, 4, 4]", l_self_modules_layers_modules_5_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_5_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_5_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_6_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_6_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_6_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_20, res_9, post_21, comb_10, l_self_modules_layers_modules_5_parameters_hc_ffn_fn_, l_self_modules_layers_modules_5_parameters_hc_ffn_scale_, l_self_modules_layers_modules_5_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_20 = res_9 = post_21 = comb_10 = l_self_modules_layers_modules_5_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_5_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_5_parameters_hc_ffn_base_ = l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.5.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_6_parameters_hc_attn_fn_, l_self_modules_layers_modules_6_parameters_hc_attn_scale_, l_self_modules_layers_modules_6_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_6_parameters_hc_attn_fn_ = l_self_modules_layers_modules_6_parameters_hc_attn_scale_ = l_self_modules_layers_modules_6_parameters_hc_attn_base_ = l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_13(torch.nn.Module):
        def forward(self, x_23: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_23, l_positions_, 'layers.6.attn', None, None);  x_23 = l_positions_ = None
            return v4_attention_with_output

    class submod_14(torch.nn.Module):
        def forward(self, x_24: "bf16[s72, 4096]", s72: "Sym(s72)", res_11: "bf16[s72, 4, 4096]", post_25: "f32[s72, 4]", comb_12: "f32[s72, 4, 4]", l_self_modules_layers_modules_6_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_6_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_6_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_7_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_7_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_7_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_24, res_11, post_25, comb_12, l_self_modules_layers_modules_6_parameters_hc_ffn_fn_, l_self_modules_layers_modules_6_parameters_hc_ffn_scale_, l_self_modules_layers_modules_6_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_24 = res_11 = post_25 = comb_12 = l_self_modules_layers_modules_6_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_6_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_6_parameters_hc_ffn_base_ = l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.6.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_7_parameters_hc_attn_fn_, l_self_modules_layers_modules_7_parameters_hc_attn_scale_, l_self_modules_layers_modules_7_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_7_parameters_hc_attn_fn_ = l_self_modules_layers_modules_7_parameters_hc_attn_scale_ = l_self_modules_layers_modules_7_parameters_hc_attn_base_ = l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_15(torch.nn.Module):
        def forward(self, x_27: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_27, l_positions_, 'layers.7.attn', None, None);  x_27 = l_positions_ = None
            return v4_attention_with_output

    class submod_16(torch.nn.Module):
        def forward(self, x_28: "bf16[s72, 4096]", s72: "Sym(s72)", res_13: "bf16[s72, 4, 4096]", post_29: "f32[s72, 4]", comb_14: "f32[s72, 4, 4]", l_self_modules_layers_modules_7_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_7_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_7_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_8_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_8_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_8_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_28, res_13, post_29, comb_14, l_self_modules_layers_modules_7_parameters_hc_ffn_fn_, l_self_modules_layers_modules_7_parameters_hc_ffn_scale_, l_self_modules_layers_modules_7_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_28 = res_13 = post_29 = comb_14 = l_self_modules_layers_modules_7_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_7_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_7_parameters_hc_ffn_base_ = l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.7.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_8_parameters_hc_attn_fn_, l_self_modules_layers_modules_8_parameters_hc_attn_scale_, l_self_modules_layers_modules_8_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_8_parameters_hc_attn_fn_ = l_self_modules_layers_modules_8_parameters_hc_attn_scale_ = l_self_modules_layers_modules_8_parameters_hc_attn_base_ = l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_17(torch.nn.Module):
        def forward(self, x_31: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_31, l_positions_, 'layers.8.attn', None, None);  x_31 = l_positions_ = None
            return v4_attention_with_output

    class submod_18(torch.nn.Module):
        def forward(self, x_32: "bf16[s72, 4096]", s72: "Sym(s72)", res_15: "bf16[s72, 4, 4096]", post_33: "f32[s72, 4]", comb_16: "f32[s72, 4, 4]", l_self_modules_layers_modules_8_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_8_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_8_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_9_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_9_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_9_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_32, res_15, post_33, comb_16, l_self_modules_layers_modules_8_parameters_hc_ffn_fn_, l_self_modules_layers_modules_8_parameters_hc_ffn_scale_, l_self_modules_layers_modules_8_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_32 = res_15 = post_33 = comb_16 = l_self_modules_layers_modules_8_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_8_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_8_parameters_hc_ffn_base_ = l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.8.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_9_parameters_hc_attn_fn_, l_self_modules_layers_modules_9_parameters_hc_attn_scale_, l_self_modules_layers_modules_9_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_9_parameters_hc_attn_fn_ = l_self_modules_layers_modules_9_parameters_hc_attn_scale_ = l_self_modules_layers_modules_9_parameters_hc_attn_base_ = l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_19(torch.nn.Module):
        def forward(self, x_35: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_35, l_positions_, 'layers.9.attn', None, None);  x_35 = l_positions_ = None
            return v4_attention_with_output

    class submod_20(torch.nn.Module):
        def forward(self, x_36: "bf16[s72, 4096]", s72: "Sym(s72)", res_17: "bf16[s72, 4, 4096]", post_37: "f32[s72, 4]", comb_18: "f32[s72, 4, 4]", l_self_modules_layers_modules_9_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_9_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_9_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_10_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_10_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_10_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_36, res_17, post_37, comb_18, l_self_modules_layers_modules_9_parameters_hc_ffn_fn_, l_self_modules_layers_modules_9_parameters_hc_ffn_scale_, l_self_modules_layers_modules_9_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_36 = res_17 = post_37 = comb_18 = l_self_modules_layers_modules_9_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_9_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_9_parameters_hc_ffn_base_ = l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.9.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_10_parameters_hc_attn_fn_, l_self_modules_layers_modules_10_parameters_hc_attn_scale_, l_self_modules_layers_modules_10_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_10_parameters_hc_attn_fn_ = l_self_modules_layers_modules_10_parameters_hc_attn_scale_ = l_self_modules_layers_modules_10_parameters_hc_attn_base_ = l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_21(torch.nn.Module):
        def forward(self, x_39: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_39, l_positions_, 'layers.10.attn', None, None);  x_39 = l_positions_ = None
            return v4_attention_with_output

    class submod_22(torch.nn.Module):
        def forward(self, x_40: "bf16[s72, 4096]", s72: "Sym(s72)", res_19: "bf16[s72, 4, 4096]", post_41: "f32[s72, 4]", comb_20: "f32[s72, 4, 4]", l_self_modules_layers_modules_10_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_10_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_10_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_11_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_11_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_11_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_40, res_19, post_41, comb_20, l_self_modules_layers_modules_10_parameters_hc_ffn_fn_, l_self_modules_layers_modules_10_parameters_hc_ffn_scale_, l_self_modules_layers_modules_10_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_40 = res_19 = post_41 = comb_20 = l_self_modules_layers_modules_10_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_10_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_10_parameters_hc_ffn_base_ = l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.10.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_11_parameters_hc_attn_fn_, l_self_modules_layers_modules_11_parameters_hc_attn_scale_, l_self_modules_layers_modules_11_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_11_parameters_hc_attn_fn_ = l_self_modules_layers_modules_11_parameters_hc_attn_scale_ = l_self_modules_layers_modules_11_parameters_hc_attn_base_ = l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_23(torch.nn.Module):
        def forward(self, x_43: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_43, l_positions_, 'layers.11.attn', None, None);  x_43 = l_positions_ = None
            return v4_attention_with_output

    class submod_24(torch.nn.Module):
        def forward(self, x_44: "bf16[s72, 4096]", s72: "Sym(s72)", res_21: "bf16[s72, 4, 4096]", post_45: "f32[s72, 4]", comb_22: "f32[s72, 4, 4]", l_self_modules_layers_modules_11_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_11_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_11_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_12_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_12_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_12_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_44, res_21, post_45, comb_22, l_self_modules_layers_modules_11_parameters_hc_ffn_fn_, l_self_modules_layers_modules_11_parameters_hc_ffn_scale_, l_self_modules_layers_modules_11_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_44 = res_21 = post_45 = comb_22 = l_self_modules_layers_modules_11_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_11_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_11_parameters_hc_ffn_base_ = l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.11.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_12_parameters_hc_attn_fn_, l_self_modules_layers_modules_12_parameters_hc_attn_scale_, l_self_modules_layers_modules_12_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_12_parameters_hc_attn_fn_ = l_self_modules_layers_modules_12_parameters_hc_attn_scale_ = l_self_modules_layers_modules_12_parameters_hc_attn_base_ = l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_25(torch.nn.Module):
        def forward(self, x_47: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_47, l_positions_, 'layers.12.attn', None, None);  x_47 = l_positions_ = None
            return v4_attention_with_output

    class submod_26(torch.nn.Module):
        def forward(self, x_48: "bf16[s72, 4096]", s72: "Sym(s72)", res_23: "bf16[s72, 4, 4096]", post_49: "f32[s72, 4]", comb_24: "f32[s72, 4, 4]", l_self_modules_layers_modules_12_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_12_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_12_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_13_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_13_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_13_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_48, res_23, post_49, comb_24, l_self_modules_layers_modules_12_parameters_hc_ffn_fn_, l_self_modules_layers_modules_12_parameters_hc_ffn_scale_, l_self_modules_layers_modules_12_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_48 = res_23 = post_49 = comb_24 = l_self_modules_layers_modules_12_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_12_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_12_parameters_hc_ffn_base_ = l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.12.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_13_parameters_hc_attn_fn_, l_self_modules_layers_modules_13_parameters_hc_attn_scale_, l_self_modules_layers_modules_13_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_13_parameters_hc_attn_fn_ = l_self_modules_layers_modules_13_parameters_hc_attn_scale_ = l_self_modules_layers_modules_13_parameters_hc_attn_base_ = l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_27(torch.nn.Module):
        def forward(self, x_51: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_51, l_positions_, 'layers.13.attn', None, None);  x_51 = l_positions_ = None
            return v4_attention_with_output

    class submod_28(torch.nn.Module):
        def forward(self, x_52: "bf16[s72, 4096]", s72: "Sym(s72)", res_25: "bf16[s72, 4, 4096]", post_53: "f32[s72, 4]", comb_26: "f32[s72, 4, 4]", l_self_modules_layers_modules_13_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_13_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_13_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_14_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_14_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_14_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_52, res_25, post_53, comb_26, l_self_modules_layers_modules_13_parameters_hc_ffn_fn_, l_self_modules_layers_modules_13_parameters_hc_ffn_scale_, l_self_modules_layers_modules_13_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_52 = res_25 = post_53 = comb_26 = l_self_modules_layers_modules_13_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_13_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_13_parameters_hc_ffn_base_ = l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.13.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_14_parameters_hc_attn_fn_, l_self_modules_layers_modules_14_parameters_hc_attn_scale_, l_self_modules_layers_modules_14_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_14_parameters_hc_attn_fn_ = l_self_modules_layers_modules_14_parameters_hc_attn_scale_ = l_self_modules_layers_modules_14_parameters_hc_attn_base_ = l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_29(torch.nn.Module):
        def forward(self, x_55: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_55, l_positions_, 'layers.14.attn', None, None);  x_55 = l_positions_ = None
            return v4_attention_with_output

    class submod_30(torch.nn.Module):
        def forward(self, x_56: "bf16[s72, 4096]", s72: "Sym(s72)", res_27: "bf16[s72, 4, 4096]", post_57: "f32[s72, 4]", comb_28: "f32[s72, 4, 4]", l_self_modules_layers_modules_14_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_14_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_14_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_15_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_15_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_15_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_56, res_27, post_57, comb_28, l_self_modules_layers_modules_14_parameters_hc_ffn_fn_, l_self_modules_layers_modules_14_parameters_hc_ffn_scale_, l_self_modules_layers_modules_14_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_56 = res_27 = post_57 = comb_28 = l_self_modules_layers_modules_14_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_14_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_14_parameters_hc_ffn_base_ = l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.14.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_15_parameters_hc_attn_fn_, l_self_modules_layers_modules_15_parameters_hc_attn_scale_, l_self_modules_layers_modules_15_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_15_parameters_hc_attn_fn_ = l_self_modules_layers_modules_15_parameters_hc_attn_scale_ = l_self_modules_layers_modules_15_parameters_hc_attn_base_ = l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_31(torch.nn.Module):
        def forward(self, x_59: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_59, l_positions_, 'layers.15.attn', None, None);  x_59 = l_positions_ = None
            return v4_attention_with_output

    class submod_32(torch.nn.Module):
        def forward(self, x_60: "bf16[s72, 4096]", s72: "Sym(s72)", res_29: "bf16[s72, 4, 4096]", post_61: "f32[s72, 4]", comb_30: "f32[s72, 4, 4]", l_self_modules_layers_modules_15_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_15_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_15_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_16_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_16_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_16_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_60, res_29, post_61, comb_30, l_self_modules_layers_modules_15_parameters_hc_ffn_fn_, l_self_modules_layers_modules_15_parameters_hc_ffn_scale_, l_self_modules_layers_modules_15_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_60 = res_29 = post_61 = comb_30 = l_self_modules_layers_modules_15_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_15_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_15_parameters_hc_ffn_base_ = l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.15.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_16_parameters_hc_attn_fn_, l_self_modules_layers_modules_16_parameters_hc_attn_scale_, l_self_modules_layers_modules_16_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_16_parameters_hc_attn_fn_ = l_self_modules_layers_modules_16_parameters_hc_attn_scale_ = l_self_modules_layers_modules_16_parameters_hc_attn_base_ = l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_33(torch.nn.Module):
        def forward(self, x_63: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_63, l_positions_, 'layers.16.attn', None, None);  x_63 = l_positions_ = None
            return v4_attention_with_output

    class submod_34(torch.nn.Module):
        def forward(self, x_64: "bf16[s72, 4096]", s72: "Sym(s72)", res_31: "bf16[s72, 4, 4096]", post_65: "f32[s72, 4]", comb_32: "f32[s72, 4, 4]", l_self_modules_layers_modules_16_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_16_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_16_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_17_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_17_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_17_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_64, res_31, post_65, comb_32, l_self_modules_layers_modules_16_parameters_hc_ffn_fn_, l_self_modules_layers_modules_16_parameters_hc_ffn_scale_, l_self_modules_layers_modules_16_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_64 = res_31 = post_65 = comb_32 = l_self_modules_layers_modules_16_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_16_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_16_parameters_hc_ffn_base_ = l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.16.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_17_parameters_hc_attn_fn_, l_self_modules_layers_modules_17_parameters_hc_attn_scale_, l_self_modules_layers_modules_17_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_17_parameters_hc_attn_fn_ = l_self_modules_layers_modules_17_parameters_hc_attn_scale_ = l_self_modules_layers_modules_17_parameters_hc_attn_base_ = l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_35(torch.nn.Module):
        def forward(self, x_67: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_67, l_positions_, 'layers.17.attn', None, None);  x_67 = l_positions_ = None
            return v4_attention_with_output

    class submod_36(torch.nn.Module):
        def forward(self, x_68: "bf16[s72, 4096]", s72: "Sym(s72)", res_33: "bf16[s72, 4, 4096]", post_69: "f32[s72, 4]", comb_34: "f32[s72, 4, 4]", l_self_modules_layers_modules_17_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_17_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_17_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_18_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_18_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_18_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_68, res_33, post_69, comb_34, l_self_modules_layers_modules_17_parameters_hc_ffn_fn_, l_self_modules_layers_modules_17_parameters_hc_ffn_scale_, l_self_modules_layers_modules_17_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_68 = res_33 = post_69 = comb_34 = l_self_modules_layers_modules_17_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_17_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_17_parameters_hc_ffn_base_ = l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.17.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_18_parameters_hc_attn_fn_, l_self_modules_layers_modules_18_parameters_hc_attn_scale_, l_self_modules_layers_modules_18_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_18_parameters_hc_attn_fn_ = l_self_modules_layers_modules_18_parameters_hc_attn_scale_ = l_self_modules_layers_modules_18_parameters_hc_attn_base_ = l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_37(torch.nn.Module):
        def forward(self, x_71: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_71, l_positions_, 'layers.18.attn', None, None);  x_71 = l_positions_ = None
            return v4_attention_with_output

    class submod_38(torch.nn.Module):
        def forward(self, x_72: "bf16[s72, 4096]", s72: "Sym(s72)", res_35: "bf16[s72, 4, 4096]", post_73: "f32[s72, 4]", comb_36: "f32[s72, 4, 4]", l_self_modules_layers_modules_18_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_18_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_18_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_19_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_19_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_19_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_72, res_35, post_73, comb_36, l_self_modules_layers_modules_18_parameters_hc_ffn_fn_, l_self_modules_layers_modules_18_parameters_hc_ffn_scale_, l_self_modules_layers_modules_18_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_72 = res_35 = post_73 = comb_36 = l_self_modules_layers_modules_18_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_18_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_18_parameters_hc_ffn_base_ = l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.18.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_19_parameters_hc_attn_fn_, l_self_modules_layers_modules_19_parameters_hc_attn_scale_, l_self_modules_layers_modules_19_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_19_parameters_hc_attn_fn_ = l_self_modules_layers_modules_19_parameters_hc_attn_scale_ = l_self_modules_layers_modules_19_parameters_hc_attn_base_ = l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_39(torch.nn.Module):
        def forward(self, x_75: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_75, l_positions_, 'layers.19.attn', None, None);  x_75 = l_positions_ = None
            return v4_attention_with_output

    class submod_40(torch.nn.Module):
        def forward(self, x_76: "bf16[s72, 4096]", s72: "Sym(s72)", res_37: "bf16[s72, 4, 4096]", post_77: "f32[s72, 4]", comb_38: "f32[s72, 4, 4]", l_self_modules_layers_modules_19_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_19_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_19_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_20_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_20_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_20_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_76, res_37, post_77, comb_38, l_self_modules_layers_modules_19_parameters_hc_ffn_fn_, l_self_modules_layers_modules_19_parameters_hc_ffn_scale_, l_self_modules_layers_modules_19_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_76 = res_37 = post_77 = comb_38 = l_self_modules_layers_modules_19_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_19_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_19_parameters_hc_ffn_base_ = l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.19.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_20_parameters_hc_attn_fn_, l_self_modules_layers_modules_20_parameters_hc_attn_scale_, l_self_modules_layers_modules_20_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_20_parameters_hc_attn_fn_ = l_self_modules_layers_modules_20_parameters_hc_attn_scale_ = l_self_modules_layers_modules_20_parameters_hc_attn_base_ = l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_41(torch.nn.Module):
        def forward(self, x_79: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_79, l_positions_, 'layers.20.attn', None, None);  x_79 = l_positions_ = None
            return v4_attention_with_output

    class submod_42(torch.nn.Module):
        def forward(self, x_80: "bf16[s72, 4096]", s72: "Sym(s72)", res_39: "bf16[s72, 4, 4096]", post_81: "f32[s72, 4]", comb_40: "f32[s72, 4, 4]", l_self_modules_layers_modules_20_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_20_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_20_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_21_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_21_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_21_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_80, res_39, post_81, comb_40, l_self_modules_layers_modules_20_parameters_hc_ffn_fn_, l_self_modules_layers_modules_20_parameters_hc_ffn_scale_, l_self_modules_layers_modules_20_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_80 = res_39 = post_81 = comb_40 = l_self_modules_layers_modules_20_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_20_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_20_parameters_hc_ffn_base_ = l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.20.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_21_parameters_hc_attn_fn_, l_self_modules_layers_modules_21_parameters_hc_attn_scale_, l_self_modules_layers_modules_21_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_21_parameters_hc_attn_fn_ = l_self_modules_layers_modules_21_parameters_hc_attn_scale_ = l_self_modules_layers_modules_21_parameters_hc_attn_base_ = l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_43(torch.nn.Module):
        def forward(self, x_83: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_83, l_positions_, 'layers.21.attn', None, None);  x_83 = l_positions_ = None
            return v4_attention_with_output

    class submod_44(torch.nn.Module):
        def forward(self, x_84: "bf16[s72, 4096]", s72: "Sym(s72)", res_41: "bf16[s72, 4, 4096]", post_85: "f32[s72, 4]", comb_42: "f32[s72, 4, 4]", l_self_modules_layers_modules_21_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_21_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_21_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_22_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_22_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_22_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_84, res_41, post_85, comb_42, l_self_modules_layers_modules_21_parameters_hc_ffn_fn_, l_self_modules_layers_modules_21_parameters_hc_ffn_scale_, l_self_modules_layers_modules_21_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_84 = res_41 = post_85 = comb_42 = l_self_modules_layers_modules_21_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_21_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_21_parameters_hc_ffn_base_ = l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.21.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_22_parameters_hc_attn_fn_, l_self_modules_layers_modules_22_parameters_hc_attn_scale_, l_self_modules_layers_modules_22_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_22_parameters_hc_attn_fn_ = l_self_modules_layers_modules_22_parameters_hc_attn_scale_ = l_self_modules_layers_modules_22_parameters_hc_attn_base_ = l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_45(torch.nn.Module):
        def forward(self, x_87: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_87, l_positions_, 'layers.22.attn', None, None);  x_87 = l_positions_ = None
            return v4_attention_with_output

    class submod_46(torch.nn.Module):
        def forward(self, x_88: "bf16[s72, 4096]", s72: "Sym(s72)", res_43: "bf16[s72, 4, 4096]", post_89: "f32[s72, 4]", comb_44: "f32[s72, 4, 4]", l_self_modules_layers_modules_22_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_22_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_22_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_23_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_23_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_23_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_88, res_43, post_89, comb_44, l_self_modules_layers_modules_22_parameters_hc_ffn_fn_, l_self_modules_layers_modules_22_parameters_hc_ffn_scale_, l_self_modules_layers_modules_22_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_88 = res_43 = post_89 = comb_44 = l_self_modules_layers_modules_22_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_22_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_22_parameters_hc_ffn_base_ = l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.22.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_23_parameters_hc_attn_fn_, l_self_modules_layers_modules_23_parameters_hc_attn_scale_, l_self_modules_layers_modules_23_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_23_parameters_hc_attn_fn_ = l_self_modules_layers_modules_23_parameters_hc_attn_scale_ = l_self_modules_layers_modules_23_parameters_hc_attn_base_ = l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_47(torch.nn.Module):
        def forward(self, x_91: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_91, l_positions_, 'layers.23.attn', None, None);  x_91 = l_positions_ = None
            return v4_attention_with_output

    class submod_48(torch.nn.Module):
        def forward(self, x_92: "bf16[s72, 4096]", s72: "Sym(s72)", res_45: "bf16[s72, 4, 4096]", post_93: "f32[s72, 4]", comb_46: "f32[s72, 4, 4]", l_self_modules_layers_modules_23_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_23_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_23_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_24_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_24_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_24_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_92, res_45, post_93, comb_46, l_self_modules_layers_modules_23_parameters_hc_ffn_fn_, l_self_modules_layers_modules_23_parameters_hc_ffn_scale_, l_self_modules_layers_modules_23_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_92 = res_45 = post_93 = comb_46 = l_self_modules_layers_modules_23_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_23_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_23_parameters_hc_ffn_base_ = l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.23.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_24_parameters_hc_attn_fn_, l_self_modules_layers_modules_24_parameters_hc_attn_scale_, l_self_modules_layers_modules_24_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_24_parameters_hc_attn_fn_ = l_self_modules_layers_modules_24_parameters_hc_attn_scale_ = l_self_modules_layers_modules_24_parameters_hc_attn_base_ = l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_49(torch.nn.Module):
        def forward(self, x_95: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_95, l_positions_, 'layers.24.attn', None, None);  x_95 = l_positions_ = None
            return v4_attention_with_output

    class submod_50(torch.nn.Module):
        def forward(self, x_96: "bf16[s72, 4096]", s72: "Sym(s72)", res_47: "bf16[s72, 4, 4096]", post_97: "f32[s72, 4]", comb_48: "f32[s72, 4, 4]", l_self_modules_layers_modules_24_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_24_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_24_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_25_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_25_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_25_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_96, res_47, post_97, comb_48, l_self_modules_layers_modules_24_parameters_hc_ffn_fn_, l_self_modules_layers_modules_24_parameters_hc_ffn_scale_, l_self_modules_layers_modules_24_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_96 = res_47 = post_97 = comb_48 = l_self_modules_layers_modules_24_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_24_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_24_parameters_hc_ffn_base_ = l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.24.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_25_parameters_hc_attn_fn_, l_self_modules_layers_modules_25_parameters_hc_attn_scale_, l_self_modules_layers_modules_25_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_25_parameters_hc_attn_fn_ = l_self_modules_layers_modules_25_parameters_hc_attn_scale_ = l_self_modules_layers_modules_25_parameters_hc_attn_base_ = l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_51(torch.nn.Module):
        def forward(self, x_99: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_99, l_positions_, 'layers.25.attn', None, None);  x_99 = l_positions_ = None
            return v4_attention_with_output

    class submod_52(torch.nn.Module):
        def forward(self, x_100: "bf16[s72, 4096]", s72: "Sym(s72)", res_49: "bf16[s72, 4, 4096]", post_101: "f32[s72, 4]", comb_50: "f32[s72, 4, 4]", l_self_modules_layers_modules_25_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_25_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_25_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_26_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_26_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_26_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_100, res_49, post_101, comb_50, l_self_modules_layers_modules_25_parameters_hc_ffn_fn_, l_self_modules_layers_modules_25_parameters_hc_ffn_scale_, l_self_modules_layers_modules_25_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_100 = res_49 = post_101 = comb_50 = l_self_modules_layers_modules_25_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_25_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_25_parameters_hc_ffn_base_ = l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.25.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_26_parameters_hc_attn_fn_, l_self_modules_layers_modules_26_parameters_hc_attn_scale_, l_self_modules_layers_modules_26_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_26_parameters_hc_attn_fn_ = l_self_modules_layers_modules_26_parameters_hc_attn_scale_ = l_self_modules_layers_modules_26_parameters_hc_attn_base_ = l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_53(torch.nn.Module):
        def forward(self, x_103: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_103, l_positions_, 'layers.26.attn', None, None);  x_103 = l_positions_ = None
            return v4_attention_with_output

    class submod_54(torch.nn.Module):
        def forward(self, x_104: "bf16[s72, 4096]", s72: "Sym(s72)", res_51: "bf16[s72, 4, 4096]", post_105: "f32[s72, 4]", comb_52: "f32[s72, 4, 4]", l_self_modules_layers_modules_26_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_26_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_26_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_27_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_27_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_27_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_104, res_51, post_105, comb_52, l_self_modules_layers_modules_26_parameters_hc_ffn_fn_, l_self_modules_layers_modules_26_parameters_hc_ffn_scale_, l_self_modules_layers_modules_26_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_104 = res_51 = post_105 = comb_52 = l_self_modules_layers_modules_26_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_26_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_26_parameters_hc_ffn_base_ = l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.26.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_27_parameters_hc_attn_fn_, l_self_modules_layers_modules_27_parameters_hc_attn_scale_, l_self_modules_layers_modules_27_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_27_parameters_hc_attn_fn_ = l_self_modules_layers_modules_27_parameters_hc_attn_scale_ = l_self_modules_layers_modules_27_parameters_hc_attn_base_ = l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_55(torch.nn.Module):
        def forward(self, x_107: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_107, l_positions_, 'layers.27.attn', None, None);  x_107 = l_positions_ = None
            return v4_attention_with_output

    class submod_56(torch.nn.Module):
        def forward(self, x_108: "bf16[s72, 4096]", s72: "Sym(s72)", res_53: "bf16[s72, 4, 4096]", post_109: "f32[s72, 4]", comb_54: "f32[s72, 4, 4]", l_self_modules_layers_modules_27_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_27_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_27_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_28_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_28_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_28_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_108, res_53, post_109, comb_54, l_self_modules_layers_modules_27_parameters_hc_ffn_fn_, l_self_modules_layers_modules_27_parameters_hc_ffn_scale_, l_self_modules_layers_modules_27_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_108 = res_53 = post_109 = comb_54 = l_self_modules_layers_modules_27_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_27_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_27_parameters_hc_ffn_base_ = l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.27.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_28_parameters_hc_attn_fn_, l_self_modules_layers_modules_28_parameters_hc_attn_scale_, l_self_modules_layers_modules_28_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_28_parameters_hc_attn_fn_ = l_self_modules_layers_modules_28_parameters_hc_attn_scale_ = l_self_modules_layers_modules_28_parameters_hc_attn_base_ = l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_57(torch.nn.Module):
        def forward(self, x_111: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_111, l_positions_, 'layers.28.attn', None, None);  x_111 = l_positions_ = None
            return v4_attention_with_output

    class submod_58(torch.nn.Module):
        def forward(self, x_112: "bf16[s72, 4096]", s72: "Sym(s72)", res_55: "bf16[s72, 4, 4096]", post_113: "f32[s72, 4]", comb_56: "f32[s72, 4, 4]", l_self_modules_layers_modules_28_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_28_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_28_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_29_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_29_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_29_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_112, res_55, post_113, comb_56, l_self_modules_layers_modules_28_parameters_hc_ffn_fn_, l_self_modules_layers_modules_28_parameters_hc_ffn_scale_, l_self_modules_layers_modules_28_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_112 = res_55 = post_113 = comb_56 = l_self_modules_layers_modules_28_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_28_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_28_parameters_hc_ffn_base_ = l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.28.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_29_parameters_hc_attn_fn_, l_self_modules_layers_modules_29_parameters_hc_attn_scale_, l_self_modules_layers_modules_29_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_29_parameters_hc_attn_fn_ = l_self_modules_layers_modules_29_parameters_hc_attn_scale_ = l_self_modules_layers_modules_29_parameters_hc_attn_base_ = l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_59(torch.nn.Module):
        def forward(self, x_115: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_115, l_positions_, 'layers.29.attn', None, None);  x_115 = l_positions_ = None
            return v4_attention_with_output

    class submod_60(torch.nn.Module):
        def forward(self, x_116: "bf16[s72, 4096]", s72: "Sym(s72)", res_57: "bf16[s72, 4, 4096]", post_117: "f32[s72, 4]", comb_58: "f32[s72, 4, 4]", l_self_modules_layers_modules_29_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_29_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_29_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_30_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_30_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_30_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_116, res_57, post_117, comb_58, l_self_modules_layers_modules_29_parameters_hc_ffn_fn_, l_self_modules_layers_modules_29_parameters_hc_ffn_scale_, l_self_modules_layers_modules_29_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_116 = res_57 = post_117 = comb_58 = l_self_modules_layers_modules_29_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_29_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_29_parameters_hc_ffn_base_ = l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.29.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_30_parameters_hc_attn_fn_, l_self_modules_layers_modules_30_parameters_hc_attn_scale_, l_self_modules_layers_modules_30_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_30_parameters_hc_attn_fn_ = l_self_modules_layers_modules_30_parameters_hc_attn_scale_ = l_self_modules_layers_modules_30_parameters_hc_attn_base_ = l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_61(torch.nn.Module):
        def forward(self, x_119: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_119, l_positions_, 'layers.30.attn', None, None);  x_119 = l_positions_ = None
            return v4_attention_with_output

    class submod_62(torch.nn.Module):
        def forward(self, x_120: "bf16[s72, 4096]", s72: "Sym(s72)", res_59: "bf16[s72, 4, 4096]", post_121: "f32[s72, 4]", comb_60: "f32[s72, 4, 4]", l_self_modules_layers_modules_30_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_30_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_30_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_31_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_31_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_31_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_120, res_59, post_121, comb_60, l_self_modules_layers_modules_30_parameters_hc_ffn_fn_, l_self_modules_layers_modules_30_parameters_hc_ffn_scale_, l_self_modules_layers_modules_30_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_120 = res_59 = post_121 = comb_60 = l_self_modules_layers_modules_30_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_30_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_30_parameters_hc_ffn_base_ = l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.30.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_31_parameters_hc_attn_fn_, l_self_modules_layers_modules_31_parameters_hc_attn_scale_, l_self_modules_layers_modules_31_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_31_parameters_hc_attn_fn_ = l_self_modules_layers_modules_31_parameters_hc_attn_scale_ = l_self_modules_layers_modules_31_parameters_hc_attn_base_ = l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_63(torch.nn.Module):
        def forward(self, x_123: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_123, l_positions_, 'layers.31.attn', None, None);  x_123 = l_positions_ = None
            return v4_attention_with_output

    class submod_64(torch.nn.Module):
        def forward(self, x_124: "bf16[s72, 4096]", s72: "Sym(s72)", res_61: "bf16[s72, 4, 4096]", post_125: "f32[s72, 4]", comb_62: "f32[s72, 4, 4]", l_self_modules_layers_modules_31_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_31_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_31_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_32_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_32_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_32_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_124, res_61, post_125, comb_62, l_self_modules_layers_modules_31_parameters_hc_ffn_fn_, l_self_modules_layers_modules_31_parameters_hc_ffn_scale_, l_self_modules_layers_modules_31_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_124 = res_61 = post_125 = comb_62 = l_self_modules_layers_modules_31_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_31_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_31_parameters_hc_ffn_base_ = l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.31.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_32_parameters_hc_attn_fn_, l_self_modules_layers_modules_32_parameters_hc_attn_scale_, l_self_modules_layers_modules_32_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_32_parameters_hc_attn_fn_ = l_self_modules_layers_modules_32_parameters_hc_attn_scale_ = l_self_modules_layers_modules_32_parameters_hc_attn_base_ = l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_65(torch.nn.Module):
        def forward(self, x_127: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_127, l_positions_, 'layers.32.attn', None, None);  x_127 = l_positions_ = None
            return v4_attention_with_output

    class submod_66(torch.nn.Module):
        def forward(self, x_128: "bf16[s72, 4096]", s72: "Sym(s72)", res_63: "bf16[s72, 4, 4096]", post_129: "f32[s72, 4]", comb_64: "f32[s72, 4, 4]", l_self_modules_layers_modules_32_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_32_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_32_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_33_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_33_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_33_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_128, res_63, post_129, comb_64, l_self_modules_layers_modules_32_parameters_hc_ffn_fn_, l_self_modules_layers_modules_32_parameters_hc_ffn_scale_, l_self_modules_layers_modules_32_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_128 = res_63 = post_129 = comb_64 = l_self_modules_layers_modules_32_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_32_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_32_parameters_hc_ffn_base_ = l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.32.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_33_parameters_hc_attn_fn_, l_self_modules_layers_modules_33_parameters_hc_attn_scale_, l_self_modules_layers_modules_33_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_33_parameters_hc_attn_fn_ = l_self_modules_layers_modules_33_parameters_hc_attn_scale_ = l_self_modules_layers_modules_33_parameters_hc_attn_base_ = l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_67(torch.nn.Module):
        def forward(self, x_131: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_131, l_positions_, 'layers.33.attn', None, None);  x_131 = l_positions_ = None
            return v4_attention_with_output

    class submod_68(torch.nn.Module):
        def forward(self, x_132: "bf16[s72, 4096]", s72: "Sym(s72)", res_65: "bf16[s72, 4, 4096]", post_133: "f32[s72, 4]", comb_66: "f32[s72, 4, 4]", l_self_modules_layers_modules_33_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_33_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_33_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_34_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_34_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_34_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_132, res_65, post_133, comb_66, l_self_modules_layers_modules_33_parameters_hc_ffn_fn_, l_self_modules_layers_modules_33_parameters_hc_ffn_scale_, l_self_modules_layers_modules_33_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_132 = res_65 = post_133 = comb_66 = l_self_modules_layers_modules_33_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_33_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_33_parameters_hc_ffn_base_ = l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.33.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_34_parameters_hc_attn_fn_, l_self_modules_layers_modules_34_parameters_hc_attn_scale_, l_self_modules_layers_modules_34_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_34_parameters_hc_attn_fn_ = l_self_modules_layers_modules_34_parameters_hc_attn_scale_ = l_self_modules_layers_modules_34_parameters_hc_attn_base_ = l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_69(torch.nn.Module):
        def forward(self, x_135: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_135, l_positions_, 'layers.34.attn', None, None);  x_135 = l_positions_ = None
            return v4_attention_with_output

    class submod_70(torch.nn.Module):
        def forward(self, x_136: "bf16[s72, 4096]", s72: "Sym(s72)", res_67: "bf16[s72, 4, 4096]", post_137: "f32[s72, 4]", comb_68: "f32[s72, 4, 4]", l_self_modules_layers_modules_34_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_34_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_34_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_35_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_35_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_35_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_136, res_67, post_137, comb_68, l_self_modules_layers_modules_34_parameters_hc_ffn_fn_, l_self_modules_layers_modules_34_parameters_hc_ffn_scale_, l_self_modules_layers_modules_34_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_136 = res_67 = post_137 = comb_68 = l_self_modules_layers_modules_34_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_34_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_34_parameters_hc_ffn_base_ = l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.34.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_35_parameters_hc_attn_fn_, l_self_modules_layers_modules_35_parameters_hc_attn_scale_, l_self_modules_layers_modules_35_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_35_parameters_hc_attn_fn_ = l_self_modules_layers_modules_35_parameters_hc_attn_scale_ = l_self_modules_layers_modules_35_parameters_hc_attn_base_ = l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_71(torch.nn.Module):
        def forward(self, x_139: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_139, l_positions_, 'layers.35.attn', None, None);  x_139 = l_positions_ = None
            return v4_attention_with_output

    class submod_72(torch.nn.Module):
        def forward(self, x_140: "bf16[s72, 4096]", s72: "Sym(s72)", res_69: "bf16[s72, 4, 4096]", post_141: "f32[s72, 4]", comb_70: "f32[s72, 4, 4]", l_self_modules_layers_modules_35_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_35_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_35_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_36_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_36_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_36_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_140, res_69, post_141, comb_70, l_self_modules_layers_modules_35_parameters_hc_ffn_fn_, l_self_modules_layers_modules_35_parameters_hc_ffn_scale_, l_self_modules_layers_modules_35_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_140 = res_69 = post_141 = comb_70 = l_self_modules_layers_modules_35_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_35_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_35_parameters_hc_ffn_base_ = l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.35.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_36_parameters_hc_attn_fn_, l_self_modules_layers_modules_36_parameters_hc_attn_scale_, l_self_modules_layers_modules_36_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_36_parameters_hc_attn_fn_ = l_self_modules_layers_modules_36_parameters_hc_attn_scale_ = l_self_modules_layers_modules_36_parameters_hc_attn_base_ = l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_73(torch.nn.Module):
        def forward(self, x_143: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_143, l_positions_, 'layers.36.attn', None, None);  x_143 = l_positions_ = None
            return v4_attention_with_output

    class submod_74(torch.nn.Module):
        def forward(self, x_144: "bf16[s72, 4096]", s72: "Sym(s72)", res_71: "bf16[s72, 4, 4096]", post_145: "f32[s72, 4]", comb_72: "f32[s72, 4, 4]", l_self_modules_layers_modules_36_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_36_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_36_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_37_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_37_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_37_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_144, res_71, post_145, comb_72, l_self_modules_layers_modules_36_parameters_hc_ffn_fn_, l_self_modules_layers_modules_36_parameters_hc_ffn_scale_, l_self_modules_layers_modules_36_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_144 = res_71 = post_145 = comb_72 = l_self_modules_layers_modules_36_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_36_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_36_parameters_hc_ffn_base_ = l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.36.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_37_parameters_hc_attn_fn_, l_self_modules_layers_modules_37_parameters_hc_attn_scale_, l_self_modules_layers_modules_37_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_37_parameters_hc_attn_fn_ = l_self_modules_layers_modules_37_parameters_hc_attn_scale_ = l_self_modules_layers_modules_37_parameters_hc_attn_base_ = l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_75(torch.nn.Module):
        def forward(self, x_147: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_147, l_positions_, 'layers.37.attn', None, None);  x_147 = l_positions_ = None
            return v4_attention_with_output

    class submod_76(torch.nn.Module):
        def forward(self, x_148: "bf16[s72, 4096]", s72: "Sym(s72)", res_73: "bf16[s72, 4, 4096]", post_149: "f32[s72, 4]", comb_74: "f32[s72, 4, 4]", l_self_modules_layers_modules_37_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_37_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_37_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_38_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_38_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_38_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_148, res_73, post_149, comb_74, l_self_modules_layers_modules_37_parameters_hc_ffn_fn_, l_self_modules_layers_modules_37_parameters_hc_ffn_scale_, l_self_modules_layers_modules_37_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_148 = res_73 = post_149 = comb_74 = l_self_modules_layers_modules_37_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_37_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_37_parameters_hc_ffn_base_ = l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.37.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_38_parameters_hc_attn_fn_, l_self_modules_layers_modules_38_parameters_hc_attn_scale_, l_self_modules_layers_modules_38_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_38_parameters_hc_attn_fn_ = l_self_modules_layers_modules_38_parameters_hc_attn_scale_ = l_self_modules_layers_modules_38_parameters_hc_attn_base_ = l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_77(torch.nn.Module):
        def forward(self, x_151: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_151, l_positions_, 'layers.38.attn', None, None);  x_151 = l_positions_ = None
            return v4_attention_with_output

    class submod_78(torch.nn.Module):
        def forward(self, x_152: "bf16[s72, 4096]", s72: "Sym(s72)", res_75: "bf16[s72, 4, 4096]", post_153: "f32[s72, 4]", comb_76: "f32[s72, 4, 4]", l_self_modules_layers_modules_38_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_38_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_38_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_39_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_39_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_39_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_152, res_75, post_153, comb_76, l_self_modules_layers_modules_38_parameters_hc_ffn_fn_, l_self_modules_layers_modules_38_parameters_hc_ffn_scale_, l_self_modules_layers_modules_38_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_152 = res_75 = post_153 = comb_76 = l_self_modules_layers_modules_38_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_38_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_38_parameters_hc_ffn_base_ = l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.38.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_39_parameters_hc_attn_fn_, l_self_modules_layers_modules_39_parameters_hc_attn_scale_, l_self_modules_layers_modules_39_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_39_parameters_hc_attn_fn_ = l_self_modules_layers_modules_39_parameters_hc_attn_scale_ = l_self_modules_layers_modules_39_parameters_hc_attn_base_ = l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_79(torch.nn.Module):
        def forward(self, x_155: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_155, l_positions_, 'layers.39.attn', None, None);  x_155 = l_positions_ = None
            return v4_attention_with_output

    class submod_80(torch.nn.Module):
        def forward(self, x_156: "bf16[s72, 4096]", s72: "Sym(s72)", res_77: "bf16[s72, 4, 4096]", post_157: "f32[s72, 4]", comb_78: "f32[s72, 4, 4]", l_self_modules_layers_modules_39_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_39_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_39_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_40_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_40_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_40_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_156, res_77, post_157, comb_78, l_self_modules_layers_modules_39_parameters_hc_ffn_fn_, l_self_modules_layers_modules_39_parameters_hc_ffn_scale_, l_self_modules_layers_modules_39_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_156 = res_77 = post_157 = comb_78 = l_self_modules_layers_modules_39_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_39_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_39_parameters_hc_ffn_base_ = l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.39.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_40_parameters_hc_attn_fn_, l_self_modules_layers_modules_40_parameters_hc_attn_scale_, l_self_modules_layers_modules_40_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_40_parameters_hc_attn_fn_ = l_self_modules_layers_modules_40_parameters_hc_attn_scale_ = l_self_modules_layers_modules_40_parameters_hc_attn_base_ = l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_81(torch.nn.Module):
        def forward(self, x_159: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_159, l_positions_, 'layers.40.attn', None, None);  x_159 = l_positions_ = None
            return v4_attention_with_output

    class submod_82(torch.nn.Module):
        def forward(self, x_160: "bf16[s72, 4096]", s72: "Sym(s72)", res_79: "bf16[s72, 4, 4096]", post_161: "f32[s72, 4]", comb_80: "f32[s72, 4, 4]", l_self_modules_layers_modules_40_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_40_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_40_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_41_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_41_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_41_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_160, res_79, post_161, comb_80, l_self_modules_layers_modules_40_parameters_hc_ffn_fn_, l_self_modules_layers_modules_40_parameters_hc_ffn_scale_, l_self_modules_layers_modules_40_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_160 = res_79 = post_161 = comb_80 = l_self_modules_layers_modules_40_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_40_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_40_parameters_hc_ffn_base_ = l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.40.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_41_parameters_hc_attn_fn_, l_self_modules_layers_modules_41_parameters_hc_attn_scale_, l_self_modules_layers_modules_41_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_41_parameters_hc_attn_fn_ = l_self_modules_layers_modules_41_parameters_hc_attn_scale_ = l_self_modules_layers_modules_41_parameters_hc_attn_base_ = l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_83(torch.nn.Module):
        def forward(self, x_163: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_163, l_positions_, 'layers.41.attn', None, None);  x_163 = l_positions_ = None
            return v4_attention_with_output

    class submod_84(torch.nn.Module):
        def forward(self, x_164: "bf16[s72, 4096]", s72: "Sym(s72)", res_81: "bf16[s72, 4, 4096]", post_165: "f32[s72, 4]", comb_82: "f32[s72, 4, 4]", l_self_modules_layers_modules_41_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_41_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_41_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_42_parameters_hc_attn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_42_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_42_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_164, res_81, post_165, comb_82, l_self_modules_layers_modules_41_parameters_hc_ffn_fn_, l_self_modules_layers_modules_41_parameters_hc_ffn_scale_, l_self_modules_layers_modules_41_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_164 = res_81 = post_165 = comb_82 = l_self_modules_layers_modules_41_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_41_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_41_parameters_hc_ffn_base_ = l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.41.ffn');  getitem_2 = None
            mhc_fused_post_pre_1 = torch.ops.aiter.mhc_fused_post_pre(maybe_dual_stream_forward, getitem_3, squeeze, getitem_1, l_self_modules_layers_modules_42_parameters_hc_attn_fn_, l_self_modules_layers_modules_42_parameters_hc_attn_scale_, l_self_modules_layers_modules_42_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = squeeze = getitem_1 = l_self_modules_layers_modules_42_parameters_hc_attn_fn_ = l_self_modules_layers_modules_42_parameters_hc_attn_scale_ = l_self_modules_layers_modules_42_parameters_hc_attn_base_ = l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_ = None
            getitem_4: "f32[s72, 4, 1]" = mhc_fused_post_pre_1[0]
            getitem_5: "f32[s72, 4, 4]" = mhc_fused_post_pre_1[1]
            getitem_6: "bf16[s72, 4096]" = mhc_fused_post_pre_1[2]
            getitem_7: "bf16[s72, 4, 4096]" = mhc_fused_post_pre_1[3];  mhc_fused_post_pre_1 = None
            squeeze_1: "f32[s72, 4]" = getitem_4.squeeze(-1);  getitem_4 = None
            return (getitem_6, getitem_7, squeeze_1, getitem_5)

    class submod_85(torch.nn.Module):
        def forward(self, x_167: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(x_167, l_positions_, 'layers.42.attn', None, None);  x_167 = l_positions_ = None
            return v4_attention_with_output

    class submod_86(torch.nn.Module):
        def forward(self, x_168: "bf16[s72, 4096]", s72: "Sym(s72)", res_83: "bf16[s72, 4, 4096]", post_169: "f32[s72, 4]", comb_84: "f32[s72, 4, 4]", l_self_modules_layers_modules_42_parameters_hc_ffn_fn_: "i32[24, 16384]", l_self_modules_layers_modules_42_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_42_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_: "bf16[4096]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            mhc_fused_post_pre = torch.ops.aiter.mhc_fused_post_pre(x_168, res_83, post_169, comb_84, l_self_modules_layers_modules_42_parameters_hc_ffn_fn_, l_self_modules_layers_modules_42_parameters_hc_ffn_scale_, l_self_modules_layers_modules_42_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_, 1e-06, w_preshuffle_bf16 = True, res_preshuffle = True);  x_168 = res_83 = post_169 = comb_84 = l_self_modules_layers_modules_42_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_42_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_42_parameters_hc_ffn_base_ = l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_fused_post_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_fused_post_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_fused_post_pre[2]
            getitem_3: "bf16[s72, 4, 4096]" = mhc_fused_post_pre[3];  mhc_fused_post_pre = None
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            maybe_dual_stream_forward: "bf16[s72, 4096]" = torch.ops.aiter.maybe_dual_stream_forward(getitem_2, 'layers.42.ffn');  getitem_2 = None
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(getitem_3)
            unsqueeze: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            mhc_post = torch.ops.aiter.mhc_post(empty_like, maybe_dual_stream_forward, getitem_3, unsqueeze, getitem_1, res_preshuffle = True);  maybe_dual_stream_forward = getitem_3 = unsqueeze = getitem_1 = mhc_post = None
            return empty_like
