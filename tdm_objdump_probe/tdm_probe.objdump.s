
tdm_objdump_probe/tdm_probe.co:	file format elf64-amdgpu

Disassembly of section .text:

0000000000000000 <tdm_objdump_probe>:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000000000: B9800641 00000001
	tensor_load_to_lds s[12:15], s[4:11]                       // 000000000008: D0710001 7C000000 7C7C040C
	s_endpgm                                                   // 000000000014: BFB00000
