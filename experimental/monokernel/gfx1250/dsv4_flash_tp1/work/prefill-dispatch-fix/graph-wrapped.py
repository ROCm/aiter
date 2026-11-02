from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, s72: "Sym(s72)", L_input_ids_: "i32[s72]", L_self_modules_embed_parameters_weight_: "bf16[129280, 4096]", s80: "Sym(s80)", L_positions_: "i64[s80]", L_self_modules_layers_modules_0_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_1_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_2_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_3_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_4_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_5_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_6_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_7_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_8_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_9_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_10_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_11_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_12_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_13_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_14_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_15_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_16_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_17_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_18_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_19_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_20_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_21_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_22_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_23_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_24_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_25_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_26_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_27_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_28_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_29_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_30_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_31_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_32_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_33_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_34_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_35_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_36_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_37_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_38_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_39_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_40_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_41_experimental_full_layer_dummy: "bf16[0]", L_self_modules_layers_modules_42_experimental_full_layer_dummy: "bf16[0]"):
        l_input_ids_ = L_input_ids_
        l_self_modules_embed_parameters_weight_ = L_self_modules_embed_parameters_weight_
        l_positions_ = L_positions_
        l_self_modules_layers_modules_0_experimental_full_layer_dummy = L_self_modules_layers_modules_0_experimental_full_layer_dummy
        l_self_modules_layers_modules_1_experimental_full_layer_dummy = L_self_modules_layers_modules_1_experimental_full_layer_dummy
        l_self_modules_layers_modules_2_experimental_full_layer_dummy = L_self_modules_layers_modules_2_experimental_full_layer_dummy
        l_self_modules_layers_modules_3_experimental_full_layer_dummy = L_self_modules_layers_modules_3_experimental_full_layer_dummy
        l_self_modules_layers_modules_4_experimental_full_layer_dummy = L_self_modules_layers_modules_4_experimental_full_layer_dummy
        l_self_modules_layers_modules_5_experimental_full_layer_dummy = L_self_modules_layers_modules_5_experimental_full_layer_dummy
        l_self_modules_layers_modules_6_experimental_full_layer_dummy = L_self_modules_layers_modules_6_experimental_full_layer_dummy
        l_self_modules_layers_modules_7_experimental_full_layer_dummy = L_self_modules_layers_modules_7_experimental_full_layer_dummy
        l_self_modules_layers_modules_8_experimental_full_layer_dummy = L_self_modules_layers_modules_8_experimental_full_layer_dummy
        l_self_modules_layers_modules_9_experimental_full_layer_dummy = L_self_modules_layers_modules_9_experimental_full_layer_dummy
        l_self_modules_layers_modules_10_experimental_full_layer_dummy = L_self_modules_layers_modules_10_experimental_full_layer_dummy
        l_self_modules_layers_modules_11_experimental_full_layer_dummy = L_self_modules_layers_modules_11_experimental_full_layer_dummy
        l_self_modules_layers_modules_12_experimental_full_layer_dummy = L_self_modules_layers_modules_12_experimental_full_layer_dummy
        l_self_modules_layers_modules_13_experimental_full_layer_dummy = L_self_modules_layers_modules_13_experimental_full_layer_dummy
        l_self_modules_layers_modules_14_experimental_full_layer_dummy = L_self_modules_layers_modules_14_experimental_full_layer_dummy
        l_self_modules_layers_modules_15_experimental_full_layer_dummy = L_self_modules_layers_modules_15_experimental_full_layer_dummy
        l_self_modules_layers_modules_16_experimental_full_layer_dummy = L_self_modules_layers_modules_16_experimental_full_layer_dummy
        l_self_modules_layers_modules_17_experimental_full_layer_dummy = L_self_modules_layers_modules_17_experimental_full_layer_dummy
        l_self_modules_layers_modules_18_experimental_full_layer_dummy = L_self_modules_layers_modules_18_experimental_full_layer_dummy
        l_self_modules_layers_modules_19_experimental_full_layer_dummy = L_self_modules_layers_modules_19_experimental_full_layer_dummy
        l_self_modules_layers_modules_20_experimental_full_layer_dummy = L_self_modules_layers_modules_20_experimental_full_layer_dummy
        l_self_modules_layers_modules_21_experimental_full_layer_dummy = L_self_modules_layers_modules_21_experimental_full_layer_dummy
        l_self_modules_layers_modules_22_experimental_full_layer_dummy = L_self_modules_layers_modules_22_experimental_full_layer_dummy
        l_self_modules_layers_modules_23_experimental_full_layer_dummy = L_self_modules_layers_modules_23_experimental_full_layer_dummy
        l_self_modules_layers_modules_24_experimental_full_layer_dummy = L_self_modules_layers_modules_24_experimental_full_layer_dummy
        l_self_modules_layers_modules_25_experimental_full_layer_dummy = L_self_modules_layers_modules_25_experimental_full_layer_dummy
        l_self_modules_layers_modules_26_experimental_full_layer_dummy = L_self_modules_layers_modules_26_experimental_full_layer_dummy
        l_self_modules_layers_modules_27_experimental_full_layer_dummy = L_self_modules_layers_modules_27_experimental_full_layer_dummy
        l_self_modules_layers_modules_28_experimental_full_layer_dummy = L_self_modules_layers_modules_28_experimental_full_layer_dummy
        l_self_modules_layers_modules_29_experimental_full_layer_dummy = L_self_modules_layers_modules_29_experimental_full_layer_dummy
        l_self_modules_layers_modules_30_experimental_full_layer_dummy = L_self_modules_layers_modules_30_experimental_full_layer_dummy
        l_self_modules_layers_modules_31_experimental_full_layer_dummy = L_self_modules_layers_modules_31_experimental_full_layer_dummy
        l_self_modules_layers_modules_32_experimental_full_layer_dummy = L_self_modules_layers_modules_32_experimental_full_layer_dummy
        l_self_modules_layers_modules_33_experimental_full_layer_dummy = L_self_modules_layers_modules_33_experimental_full_layer_dummy
        l_self_modules_layers_modules_34_experimental_full_layer_dummy = L_self_modules_layers_modules_34_experimental_full_layer_dummy
        l_self_modules_layers_modules_35_experimental_full_layer_dummy = L_self_modules_layers_modules_35_experimental_full_layer_dummy
        l_self_modules_layers_modules_36_experimental_full_layer_dummy = L_self_modules_layers_modules_36_experimental_full_layer_dummy
        l_self_modules_layers_modules_37_experimental_full_layer_dummy = L_self_modules_layers_modules_37_experimental_full_layer_dummy
        l_self_modules_layers_modules_38_experimental_full_layer_dummy = L_self_modules_layers_modules_38_experimental_full_layer_dummy
        l_self_modules_layers_modules_39_experimental_full_layer_dummy = L_self_modules_layers_modules_39_experimental_full_layer_dummy
        l_self_modules_layers_modules_40_experimental_full_layer_dummy = L_self_modules_layers_modules_40_experimental_full_layer_dummy
        l_self_modules_layers_modules_41_experimental_full_layer_dummy = L_self_modules_layers_modules_41_experimental_full_layer_dummy
        l_self_modules_layers_modules_42_experimental_full_layer_dummy = L_self_modules_layers_modules_42_experimental_full_layer_dummy

        # No stacktrace found for following nodes
        submod_0 = self.submod_0(l_input_ids_, s72, l_self_modules_embed_parameters_weight_, l_positions_, s80, l_self_modules_layers_modules_0_experimental_full_layer_dummy, l_self_modules_layers_modules_1_experimental_full_layer_dummy, l_self_modules_layers_modules_2_experimental_full_layer_dummy, l_self_modules_layers_modules_3_experimental_full_layer_dummy, l_self_modules_layers_modules_4_experimental_full_layer_dummy, l_self_modules_layers_modules_5_experimental_full_layer_dummy, l_self_modules_layers_modules_6_experimental_full_layer_dummy, l_self_modules_layers_modules_7_experimental_full_layer_dummy, l_self_modules_layers_modules_8_experimental_full_layer_dummy, l_self_modules_layers_modules_9_experimental_full_layer_dummy, l_self_modules_layers_modules_10_experimental_full_layer_dummy, l_self_modules_layers_modules_11_experimental_full_layer_dummy, l_self_modules_layers_modules_12_experimental_full_layer_dummy, l_self_modules_layers_modules_13_experimental_full_layer_dummy, l_self_modules_layers_modules_14_experimental_full_layer_dummy, l_self_modules_layers_modules_15_experimental_full_layer_dummy, l_self_modules_layers_modules_16_experimental_full_layer_dummy, l_self_modules_layers_modules_17_experimental_full_layer_dummy, l_self_modules_layers_modules_18_experimental_full_layer_dummy, l_self_modules_layers_modules_19_experimental_full_layer_dummy, l_self_modules_layers_modules_20_experimental_full_layer_dummy, l_self_modules_layers_modules_21_experimental_full_layer_dummy, l_self_modules_layers_modules_22_experimental_full_layer_dummy, l_self_modules_layers_modules_23_experimental_full_layer_dummy, l_self_modules_layers_modules_24_experimental_full_layer_dummy, l_self_modules_layers_modules_25_experimental_full_layer_dummy, l_self_modules_layers_modules_26_experimental_full_layer_dummy, l_self_modules_layers_modules_27_experimental_full_layer_dummy, l_self_modules_layers_modules_28_experimental_full_layer_dummy, l_self_modules_layers_modules_29_experimental_full_layer_dummy, l_self_modules_layers_modules_30_experimental_full_layer_dummy, l_self_modules_layers_modules_31_experimental_full_layer_dummy, l_self_modules_layers_modules_32_experimental_full_layer_dummy, l_self_modules_layers_modules_33_experimental_full_layer_dummy, l_self_modules_layers_modules_34_experimental_full_layer_dummy, l_self_modules_layers_modules_35_experimental_full_layer_dummy, l_self_modules_layers_modules_36_experimental_full_layer_dummy, l_self_modules_layers_modules_37_experimental_full_layer_dummy, l_self_modules_layers_modules_38_experimental_full_layer_dummy, l_self_modules_layers_modules_39_experimental_full_layer_dummy, l_self_modules_layers_modules_40_experimental_full_layer_dummy, l_self_modules_layers_modules_41_experimental_full_layer_dummy, l_self_modules_layers_modules_42_experimental_full_layer_dummy);  l_input_ids_ = s72 = l_self_modules_embed_parameters_weight_ = l_positions_ = s80 = l_self_modules_layers_modules_0_experimental_full_layer_dummy = l_self_modules_layers_modules_1_experimental_full_layer_dummy = l_self_modules_layers_modules_2_experimental_full_layer_dummy = l_self_modules_layers_modules_3_experimental_full_layer_dummy = l_self_modules_layers_modules_4_experimental_full_layer_dummy = l_self_modules_layers_modules_5_experimental_full_layer_dummy = l_self_modules_layers_modules_6_experimental_full_layer_dummy = l_self_modules_layers_modules_7_experimental_full_layer_dummy = l_self_modules_layers_modules_8_experimental_full_layer_dummy = l_self_modules_layers_modules_9_experimental_full_layer_dummy = l_self_modules_layers_modules_10_experimental_full_layer_dummy = l_self_modules_layers_modules_11_experimental_full_layer_dummy = l_self_modules_layers_modules_12_experimental_full_layer_dummy = l_self_modules_layers_modules_13_experimental_full_layer_dummy = l_self_modules_layers_modules_14_experimental_full_layer_dummy = l_self_modules_layers_modules_15_experimental_full_layer_dummy = l_self_modules_layers_modules_16_experimental_full_layer_dummy = l_self_modules_layers_modules_17_experimental_full_layer_dummy = l_self_modules_layers_modules_18_experimental_full_layer_dummy = l_self_modules_layers_modules_19_experimental_full_layer_dummy = l_self_modules_layers_modules_20_experimental_full_layer_dummy = l_self_modules_layers_modules_21_experimental_full_layer_dummy = l_self_modules_layers_modules_22_experimental_full_layer_dummy = l_self_modules_layers_modules_23_experimental_full_layer_dummy = l_self_modules_layers_modules_24_experimental_full_layer_dummy = l_self_modules_layers_modules_25_experimental_full_layer_dummy = l_self_modules_layers_modules_26_experimental_full_layer_dummy = l_self_modules_layers_modules_27_experimental_full_layer_dummy = l_self_modules_layers_modules_28_experimental_full_layer_dummy = l_self_modules_layers_modules_29_experimental_full_layer_dummy = l_self_modules_layers_modules_30_experimental_full_layer_dummy = l_self_modules_layers_modules_31_experimental_full_layer_dummy = l_self_modules_layers_modules_32_experimental_full_layer_dummy = l_self_modules_layers_modules_33_experimental_full_layer_dummy = l_self_modules_layers_modules_34_experimental_full_layer_dummy = l_self_modules_layers_modules_35_experimental_full_layer_dummy = l_self_modules_layers_modules_36_experimental_full_layer_dummy = l_self_modules_layers_modules_37_experimental_full_layer_dummy = l_self_modules_layers_modules_38_experimental_full_layer_dummy = l_self_modules_layers_modules_39_experimental_full_layer_dummy = l_self_modules_layers_modules_40_experimental_full_layer_dummy = l_self_modules_layers_modules_41_experimental_full_layer_dummy = l_self_modules_layers_modules_42_experimental_full_layer_dummy = None
        return (submod_0,)

    class submod_0(torch.nn.Module):
        def forward(self, l_input_ids_: "i32[s72]", s72: "Sym(s72)", l_self_modules_embed_parameters_weight_: "bf16[129280, 4096]", l_positions_: "i64[s80]", s80: "Sym(s80)", l_self_modules_layers_modules_0_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_1_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_2_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_3_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_4_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_5_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_6_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_7_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_8_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_9_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_10_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_11_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_12_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_13_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_14_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_15_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_16_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_17_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_18_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_19_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_20_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_21_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_22_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_23_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_24_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_25_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_26_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_27_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_28_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_29_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_30_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_31_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_32_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_33_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_34_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_35_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_36_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_37_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_38_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_39_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_40_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_41_experimental_full_layer_dummy: "bf16[0]", l_self_modules_layers_modules_42_experimental_full_layer_dummy: "bf16[0]"):
            # File: /usr/local/lib/python3.12/dist-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
            embedding: "bf16[s72, 4096]" = torch.nn.functional.embedding(l_input_ids_, l_self_modules_embed_parameters_weight_);  l_input_ids_ = l_self_modules_embed_parameters_weight_ = None
            mhc_res_repeat: "bf16[s72, 4, 4096]" = torch.ops.aiter.mhc_res_repeat(embedding, 4, True);  embedding = None
            dsv4_modelwide_state_c1_default = torch.ops.atom.dsv4_modelwide_state_c1.default(mhc_res_repeat, l_positions_, None, None, None, l_self_modules_layers_modules_0_experimental_full_layer_dummy, 'layers.0');  mhc_res_repeat = l_self_modules_layers_modules_0_experimental_full_layer_dummy = None
            getitem: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default[0]
            getitem_1: "f32[s72, 4]" = dsv4_modelwide_state_c1_default[1]
            getitem_2: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default[2]
            getitem_3: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default[3];  dsv4_modelwide_state_c1_default = None
            dsv4_modelwide_state_c1_default_1 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem, l_positions_, getitem_1, getitem_2, getitem_3, l_self_modules_layers_modules_1_experimental_full_layer_dummy, 'layers.1');  getitem = getitem_1 = getitem_2 = getitem_3 = l_self_modules_layers_modules_1_experimental_full_layer_dummy = None
            getitem_4: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_1[0]
            getitem_5: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_1[1]
            getitem_6: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_1[2]
            getitem_7: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_1[3];  dsv4_modelwide_state_c1_default_1 = None
            dsv4_modelwide_state_c1_default_2 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_4, l_positions_, getitem_5, getitem_6, getitem_7, l_self_modules_layers_modules_2_experimental_full_layer_dummy, 'layers.2');  getitem_4 = getitem_5 = getitem_6 = getitem_7 = l_self_modules_layers_modules_2_experimental_full_layer_dummy = None
            getitem_8: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_2[0]
            getitem_9: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_2[1]
            getitem_10: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_2[2]
            getitem_11: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_2[3];  dsv4_modelwide_state_c1_default_2 = None
            dsv4_modelwide_state_c1_default_3 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_8, l_positions_, getitem_9, getitem_10, getitem_11, l_self_modules_layers_modules_3_experimental_full_layer_dummy, 'layers.3');  getitem_8 = getitem_9 = getitem_10 = getitem_11 = l_self_modules_layers_modules_3_experimental_full_layer_dummy = None
            getitem_12: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_3[0]
            getitem_13: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_3[1]
            getitem_14: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_3[2]
            getitem_15: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_3[3];  dsv4_modelwide_state_c1_default_3 = None
            dsv4_modelwide_state_c1_default_4 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_12, l_positions_, getitem_13, getitem_14, getitem_15, l_self_modules_layers_modules_4_experimental_full_layer_dummy, 'layers.4');  getitem_12 = getitem_13 = getitem_14 = getitem_15 = l_self_modules_layers_modules_4_experimental_full_layer_dummy = None
            getitem_16: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_4[0]
            getitem_17: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_4[1]
            getitem_18: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_4[2]
            getitem_19: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_4[3];  dsv4_modelwide_state_c1_default_4 = None
            dsv4_modelwide_state_c1_default_5 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_16, l_positions_, getitem_17, getitem_18, getitem_19, l_self_modules_layers_modules_5_experimental_full_layer_dummy, 'layers.5');  getitem_16 = getitem_17 = getitem_18 = getitem_19 = l_self_modules_layers_modules_5_experimental_full_layer_dummy = None
            getitem_20: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_5[0]
            getitem_21: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_5[1]
            getitem_22: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_5[2]
            getitem_23: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_5[3];  dsv4_modelwide_state_c1_default_5 = None
            dsv4_modelwide_state_c1_default_6 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_20, l_positions_, getitem_21, getitem_22, getitem_23, l_self_modules_layers_modules_6_experimental_full_layer_dummy, 'layers.6');  getitem_20 = getitem_21 = getitem_22 = getitem_23 = l_self_modules_layers_modules_6_experimental_full_layer_dummy = None
            getitem_24: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_6[0]
            getitem_25: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_6[1]
            getitem_26: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_6[2]
            getitem_27: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_6[3];  dsv4_modelwide_state_c1_default_6 = None
            dsv4_modelwide_state_c1_default_7 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_24, l_positions_, getitem_25, getitem_26, getitem_27, l_self_modules_layers_modules_7_experimental_full_layer_dummy, 'layers.7');  getitem_24 = getitem_25 = getitem_26 = getitem_27 = l_self_modules_layers_modules_7_experimental_full_layer_dummy = None
            getitem_28: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_7[0]
            getitem_29: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_7[1]
            getitem_30: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_7[2]
            getitem_31: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_7[3];  dsv4_modelwide_state_c1_default_7 = None
            dsv4_modelwide_state_c1_default_8 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_28, l_positions_, getitem_29, getitem_30, getitem_31, l_self_modules_layers_modules_8_experimental_full_layer_dummy, 'layers.8');  getitem_28 = getitem_29 = getitem_30 = getitem_31 = l_self_modules_layers_modules_8_experimental_full_layer_dummy = None
            getitem_32: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_8[0]
            getitem_33: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_8[1]
            getitem_34: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_8[2]
            getitem_35: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_8[3];  dsv4_modelwide_state_c1_default_8 = None
            dsv4_modelwide_state_c1_default_9 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_32, l_positions_, getitem_33, getitem_34, getitem_35, l_self_modules_layers_modules_9_experimental_full_layer_dummy, 'layers.9');  getitem_32 = getitem_33 = getitem_34 = getitem_35 = l_self_modules_layers_modules_9_experimental_full_layer_dummy = None
            getitem_36: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_9[0]
            getitem_37: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_9[1]
            getitem_38: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_9[2]
            getitem_39: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_9[3];  dsv4_modelwide_state_c1_default_9 = None
            dsv4_modelwide_state_c1_default_10 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_36, l_positions_, getitem_37, getitem_38, getitem_39, l_self_modules_layers_modules_10_experimental_full_layer_dummy, 'layers.10');  getitem_36 = getitem_37 = getitem_38 = getitem_39 = l_self_modules_layers_modules_10_experimental_full_layer_dummy = None
            getitem_40: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_10[0]
            getitem_41: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_10[1]
            getitem_42: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_10[2]
            getitem_43: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_10[3];  dsv4_modelwide_state_c1_default_10 = None
            dsv4_modelwide_state_c1_default_11 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_40, l_positions_, getitem_41, getitem_42, getitem_43, l_self_modules_layers_modules_11_experimental_full_layer_dummy, 'layers.11');  getitem_40 = getitem_41 = getitem_42 = getitem_43 = l_self_modules_layers_modules_11_experimental_full_layer_dummy = None
            getitem_44: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_11[0]
            getitem_45: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_11[1]
            getitem_46: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_11[2]
            getitem_47: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_11[3];  dsv4_modelwide_state_c1_default_11 = None
            dsv4_modelwide_state_c1_default_12 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_44, l_positions_, getitem_45, getitem_46, getitem_47, l_self_modules_layers_modules_12_experimental_full_layer_dummy, 'layers.12');  getitem_44 = getitem_45 = getitem_46 = getitem_47 = l_self_modules_layers_modules_12_experimental_full_layer_dummy = None
            getitem_48: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_12[0]
            getitem_49: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_12[1]
            getitem_50: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_12[2]
            getitem_51: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_12[3];  dsv4_modelwide_state_c1_default_12 = None
            dsv4_modelwide_state_c1_default_13 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_48, l_positions_, getitem_49, getitem_50, getitem_51, l_self_modules_layers_modules_13_experimental_full_layer_dummy, 'layers.13');  getitem_48 = getitem_49 = getitem_50 = getitem_51 = l_self_modules_layers_modules_13_experimental_full_layer_dummy = None
            getitem_52: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_13[0]
            getitem_53: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_13[1]
            getitem_54: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_13[2]
            getitem_55: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_13[3];  dsv4_modelwide_state_c1_default_13 = None
            dsv4_modelwide_state_c1_default_14 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_52, l_positions_, getitem_53, getitem_54, getitem_55, l_self_modules_layers_modules_14_experimental_full_layer_dummy, 'layers.14');  getitem_52 = getitem_53 = getitem_54 = getitem_55 = l_self_modules_layers_modules_14_experimental_full_layer_dummy = None
            getitem_56: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_14[0]
            getitem_57: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_14[1]
            getitem_58: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_14[2]
            getitem_59: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_14[3];  dsv4_modelwide_state_c1_default_14 = None
            dsv4_modelwide_state_c1_default_15 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_56, l_positions_, getitem_57, getitem_58, getitem_59, l_self_modules_layers_modules_15_experimental_full_layer_dummy, 'layers.15');  getitem_56 = getitem_57 = getitem_58 = getitem_59 = l_self_modules_layers_modules_15_experimental_full_layer_dummy = None
            getitem_60: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_15[0]
            getitem_61: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_15[1]
            getitem_62: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_15[2]
            getitem_63: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_15[3];  dsv4_modelwide_state_c1_default_15 = None
            dsv4_modelwide_state_c1_default_16 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_60, l_positions_, getitem_61, getitem_62, getitem_63, l_self_modules_layers_modules_16_experimental_full_layer_dummy, 'layers.16');  getitem_60 = getitem_61 = getitem_62 = getitem_63 = l_self_modules_layers_modules_16_experimental_full_layer_dummy = None
            getitem_64: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_16[0]
            getitem_65: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_16[1]
            getitem_66: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_16[2]
            getitem_67: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_16[3];  dsv4_modelwide_state_c1_default_16 = None
            dsv4_modelwide_state_c1_default_17 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_64, l_positions_, getitem_65, getitem_66, getitem_67, l_self_modules_layers_modules_17_experimental_full_layer_dummy, 'layers.17');  getitem_64 = getitem_65 = getitem_66 = getitem_67 = l_self_modules_layers_modules_17_experimental_full_layer_dummy = None
            getitem_68: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_17[0]
            getitem_69: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_17[1]
            getitem_70: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_17[2]
            getitem_71: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_17[3];  dsv4_modelwide_state_c1_default_17 = None
            dsv4_modelwide_state_c1_default_18 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_68, l_positions_, getitem_69, getitem_70, getitem_71, l_self_modules_layers_modules_18_experimental_full_layer_dummy, 'layers.18');  getitem_68 = getitem_69 = getitem_70 = getitem_71 = l_self_modules_layers_modules_18_experimental_full_layer_dummy = None
            getitem_72: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_18[0]
            getitem_73: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_18[1]
            getitem_74: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_18[2]
            getitem_75: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_18[3];  dsv4_modelwide_state_c1_default_18 = None
            dsv4_modelwide_state_c1_default_19 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_72, l_positions_, getitem_73, getitem_74, getitem_75, l_self_modules_layers_modules_19_experimental_full_layer_dummy, 'layers.19');  getitem_72 = getitem_73 = getitem_74 = getitem_75 = l_self_modules_layers_modules_19_experimental_full_layer_dummy = None
            getitem_76: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_19[0]
            getitem_77: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_19[1]
            getitem_78: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_19[2]
            getitem_79: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_19[3];  dsv4_modelwide_state_c1_default_19 = None
            dsv4_modelwide_state_c1_default_20 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_76, l_positions_, getitem_77, getitem_78, getitem_79, l_self_modules_layers_modules_20_experimental_full_layer_dummy, 'layers.20');  getitem_76 = getitem_77 = getitem_78 = getitem_79 = l_self_modules_layers_modules_20_experimental_full_layer_dummy = None
            getitem_80: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_20[0]
            getitem_81: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_20[1]
            getitem_82: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_20[2]
            getitem_83: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_20[3];  dsv4_modelwide_state_c1_default_20 = None
            dsv4_modelwide_state_c1_default_21 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_80, l_positions_, getitem_81, getitem_82, getitem_83, l_self_modules_layers_modules_21_experimental_full_layer_dummy, 'layers.21');  getitem_80 = getitem_81 = getitem_82 = getitem_83 = l_self_modules_layers_modules_21_experimental_full_layer_dummy = None
            getitem_84: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_21[0]
            getitem_85: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_21[1]
            getitem_86: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_21[2]
            getitem_87: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_21[3];  dsv4_modelwide_state_c1_default_21 = None
            dsv4_modelwide_state_c1_default_22 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_84, l_positions_, getitem_85, getitem_86, getitem_87, l_self_modules_layers_modules_22_experimental_full_layer_dummy, 'layers.22');  getitem_84 = getitem_85 = getitem_86 = getitem_87 = l_self_modules_layers_modules_22_experimental_full_layer_dummy = None
            getitem_88: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_22[0]
            getitem_89: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_22[1]
            getitem_90: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_22[2]
            getitem_91: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_22[3];  dsv4_modelwide_state_c1_default_22 = None
            dsv4_modelwide_state_c1_default_23 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_88, l_positions_, getitem_89, getitem_90, getitem_91, l_self_modules_layers_modules_23_experimental_full_layer_dummy, 'layers.23');  getitem_88 = getitem_89 = getitem_90 = getitem_91 = l_self_modules_layers_modules_23_experimental_full_layer_dummy = None
            getitem_92: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_23[0]
            getitem_93: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_23[1]
            getitem_94: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_23[2]
            getitem_95: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_23[3];  dsv4_modelwide_state_c1_default_23 = None
            dsv4_modelwide_state_c1_default_24 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_92, l_positions_, getitem_93, getitem_94, getitem_95, l_self_modules_layers_modules_24_experimental_full_layer_dummy, 'layers.24');  getitem_92 = getitem_93 = getitem_94 = getitem_95 = l_self_modules_layers_modules_24_experimental_full_layer_dummy = None
            getitem_96: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_24[0]
            getitem_97: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_24[1]
            getitem_98: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_24[2]
            getitem_99: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_24[3];  dsv4_modelwide_state_c1_default_24 = None
            dsv4_modelwide_state_c1_default_25 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_96, l_positions_, getitem_97, getitem_98, getitem_99, l_self_modules_layers_modules_25_experimental_full_layer_dummy, 'layers.25');  getitem_96 = getitem_97 = getitem_98 = getitem_99 = l_self_modules_layers_modules_25_experimental_full_layer_dummy = None
            getitem_100: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_25[0]
            getitem_101: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_25[1]
            getitem_102: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_25[2]
            getitem_103: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_25[3];  dsv4_modelwide_state_c1_default_25 = None
            dsv4_modelwide_state_c1_default_26 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_100, l_positions_, getitem_101, getitem_102, getitem_103, l_self_modules_layers_modules_26_experimental_full_layer_dummy, 'layers.26');  getitem_100 = getitem_101 = getitem_102 = getitem_103 = l_self_modules_layers_modules_26_experimental_full_layer_dummy = None
            getitem_104: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_26[0]
            getitem_105: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_26[1]
            getitem_106: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_26[2]
            getitem_107: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_26[3];  dsv4_modelwide_state_c1_default_26 = None
            dsv4_modelwide_state_c1_default_27 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_104, l_positions_, getitem_105, getitem_106, getitem_107, l_self_modules_layers_modules_27_experimental_full_layer_dummy, 'layers.27');  getitem_104 = getitem_105 = getitem_106 = getitem_107 = l_self_modules_layers_modules_27_experimental_full_layer_dummy = None
            getitem_108: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_27[0]
            getitem_109: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_27[1]
            getitem_110: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_27[2]
            getitem_111: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_27[3];  dsv4_modelwide_state_c1_default_27 = None
            dsv4_modelwide_state_c1_default_28 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_108, l_positions_, getitem_109, getitem_110, getitem_111, l_self_modules_layers_modules_28_experimental_full_layer_dummy, 'layers.28');  getitem_108 = getitem_109 = getitem_110 = getitem_111 = l_self_modules_layers_modules_28_experimental_full_layer_dummy = None
            getitem_112: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_28[0]
            getitem_113: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_28[1]
            getitem_114: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_28[2]
            getitem_115: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_28[3];  dsv4_modelwide_state_c1_default_28 = None
            dsv4_modelwide_state_c1_default_29 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_112, l_positions_, getitem_113, getitem_114, getitem_115, l_self_modules_layers_modules_29_experimental_full_layer_dummy, 'layers.29');  getitem_112 = getitem_113 = getitem_114 = getitem_115 = l_self_modules_layers_modules_29_experimental_full_layer_dummy = None
            getitem_116: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_29[0]
            getitem_117: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_29[1]
            getitem_118: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_29[2]
            getitem_119: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_29[3];  dsv4_modelwide_state_c1_default_29 = None
            dsv4_modelwide_state_c1_default_30 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_116, l_positions_, getitem_117, getitem_118, getitem_119, l_self_modules_layers_modules_30_experimental_full_layer_dummy, 'layers.30');  getitem_116 = getitem_117 = getitem_118 = getitem_119 = l_self_modules_layers_modules_30_experimental_full_layer_dummy = None
            getitem_120: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_30[0]
            getitem_121: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_30[1]
            getitem_122: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_30[2]
            getitem_123: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_30[3];  dsv4_modelwide_state_c1_default_30 = None
            dsv4_modelwide_state_c1_default_31 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_120, l_positions_, getitem_121, getitem_122, getitem_123, l_self_modules_layers_modules_31_experimental_full_layer_dummy, 'layers.31');  getitem_120 = getitem_121 = getitem_122 = getitem_123 = l_self_modules_layers_modules_31_experimental_full_layer_dummy = None
            getitem_124: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_31[0]
            getitem_125: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_31[1]
            getitem_126: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_31[2]
            getitem_127: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_31[3];  dsv4_modelwide_state_c1_default_31 = None
            dsv4_modelwide_state_c1_default_32 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_124, l_positions_, getitem_125, getitem_126, getitem_127, l_self_modules_layers_modules_32_experimental_full_layer_dummy, 'layers.32');  getitem_124 = getitem_125 = getitem_126 = getitem_127 = l_self_modules_layers_modules_32_experimental_full_layer_dummy = None
            getitem_128: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_32[0]
            getitem_129: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_32[1]
            getitem_130: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_32[2]
            getitem_131: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_32[3];  dsv4_modelwide_state_c1_default_32 = None
            dsv4_modelwide_state_c1_default_33 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_128, l_positions_, getitem_129, getitem_130, getitem_131, l_self_modules_layers_modules_33_experimental_full_layer_dummy, 'layers.33');  getitem_128 = getitem_129 = getitem_130 = getitem_131 = l_self_modules_layers_modules_33_experimental_full_layer_dummy = None
            getitem_132: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_33[0]
            getitem_133: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_33[1]
            getitem_134: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_33[2]
            getitem_135: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_33[3];  dsv4_modelwide_state_c1_default_33 = None
            dsv4_modelwide_state_c1_default_34 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_132, l_positions_, getitem_133, getitem_134, getitem_135, l_self_modules_layers_modules_34_experimental_full_layer_dummy, 'layers.34');  getitem_132 = getitem_133 = getitem_134 = getitem_135 = l_self_modules_layers_modules_34_experimental_full_layer_dummy = None
            getitem_136: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_34[0]
            getitem_137: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_34[1]
            getitem_138: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_34[2]
            getitem_139: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_34[3];  dsv4_modelwide_state_c1_default_34 = None
            dsv4_modelwide_state_c1_default_35 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_136, l_positions_, getitem_137, getitem_138, getitem_139, l_self_modules_layers_modules_35_experimental_full_layer_dummy, 'layers.35');  getitem_136 = getitem_137 = getitem_138 = getitem_139 = l_self_modules_layers_modules_35_experimental_full_layer_dummy = None
            getitem_140: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_35[0]
            getitem_141: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_35[1]
            getitem_142: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_35[2]
            getitem_143: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_35[3];  dsv4_modelwide_state_c1_default_35 = None
            dsv4_modelwide_state_c1_default_36 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_140, l_positions_, getitem_141, getitem_142, getitem_143, l_self_modules_layers_modules_36_experimental_full_layer_dummy, 'layers.36');  getitem_140 = getitem_141 = getitem_142 = getitem_143 = l_self_modules_layers_modules_36_experimental_full_layer_dummy = None
            getitem_144: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_36[0]
            getitem_145: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_36[1]
            getitem_146: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_36[2]
            getitem_147: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_36[3];  dsv4_modelwide_state_c1_default_36 = None
            dsv4_modelwide_state_c1_default_37 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_144, l_positions_, getitem_145, getitem_146, getitem_147, l_self_modules_layers_modules_37_experimental_full_layer_dummy, 'layers.37');  getitem_144 = getitem_145 = getitem_146 = getitem_147 = l_self_modules_layers_modules_37_experimental_full_layer_dummy = None
            getitem_148: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_37[0]
            getitem_149: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_37[1]
            getitem_150: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_37[2]
            getitem_151: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_37[3];  dsv4_modelwide_state_c1_default_37 = None
            dsv4_modelwide_state_c1_default_38 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_148, l_positions_, getitem_149, getitem_150, getitem_151, l_self_modules_layers_modules_38_experimental_full_layer_dummy, 'layers.38');  getitem_148 = getitem_149 = getitem_150 = getitem_151 = l_self_modules_layers_modules_38_experimental_full_layer_dummy = None
            getitem_152: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_38[0]
            getitem_153: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_38[1]
            getitem_154: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_38[2]
            getitem_155: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_38[3];  dsv4_modelwide_state_c1_default_38 = None
            dsv4_modelwide_state_c1_default_39 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_152, l_positions_, getitem_153, getitem_154, getitem_155, l_self_modules_layers_modules_39_experimental_full_layer_dummy, 'layers.39');  getitem_152 = getitem_153 = getitem_154 = getitem_155 = l_self_modules_layers_modules_39_experimental_full_layer_dummy = None
            getitem_156: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_39[0]
            getitem_157: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_39[1]
            getitem_158: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_39[2]
            getitem_159: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_39[3];  dsv4_modelwide_state_c1_default_39 = None
            dsv4_modelwide_state_c1_default_40 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_156, l_positions_, getitem_157, getitem_158, getitem_159, l_self_modules_layers_modules_40_experimental_full_layer_dummy, 'layers.40');  getitem_156 = getitem_157 = getitem_158 = getitem_159 = l_self_modules_layers_modules_40_experimental_full_layer_dummy = None
            getitem_160: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_40[0]
            getitem_161: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_40[1]
            getitem_162: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_40[2]
            getitem_163: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_40[3];  dsv4_modelwide_state_c1_default_40 = None
            dsv4_modelwide_state_c1_default_41 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_160, l_positions_, getitem_161, getitem_162, getitem_163, l_self_modules_layers_modules_41_experimental_full_layer_dummy, 'layers.41');  getitem_160 = getitem_161 = getitem_162 = getitem_163 = l_self_modules_layers_modules_41_experimental_full_layer_dummy = None
            getitem_164: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_41[0]
            getitem_165: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_41[1]
            getitem_166: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_41[2]
            getitem_167: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_41[3];  dsv4_modelwide_state_c1_default_41 = None
            dsv4_modelwide_state_c1_default_42 = torch.ops.atom.dsv4_modelwide_state_c1.default(getitem_164, l_positions_, getitem_165, getitem_166, getitem_167, l_self_modules_layers_modules_42_experimental_full_layer_dummy, 'layers.42');  getitem_164 = l_positions_ = getitem_165 = getitem_166 = getitem_167 = l_self_modules_layers_modules_42_experimental_full_layer_dummy = None
            getitem_168: "bf16[s72, 4, 4096]" = dsv4_modelwide_state_c1_default_42[0]
            getitem_169: "f32[s72, 4]" = dsv4_modelwide_state_c1_default_42[1]
            getitem_170: "f32[s72, 4, 4]" = dsv4_modelwide_state_c1_default_42[2]
            getitem_171: "bf16[s72, 4096]" = dsv4_modelwide_state_c1_default_42[3];  dsv4_modelwide_state_c1_default_42 = None
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(getitem_168)
            unsqueeze: "f32[s72, 4, 1]" = getitem_169.unsqueeze(-1);  getitem_169 = None
            mhc_post = torch.ops.aiter.mhc_post(empty_like, getitem_171, getitem_168, unsqueeze, getitem_170, res_preshuffle = True);  getitem_171 = getitem_168 = unsqueeze = getitem_170 = mhc_post = None
            return empty_like
