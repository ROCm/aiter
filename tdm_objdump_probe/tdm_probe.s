        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  tdm_objdump_probe
        .p2align        8
        .type   tdm_objdump_probe,@function
tdm_objdump_probe:
        s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
        tensor_load_to_lds s[12:15], s[4:11]
        s_endpgm
.Lfunc_end0:
        .size   tdm_objdump_probe, .Lfunc_end0-tdm_objdump_probe
