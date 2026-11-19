// Selected actual ISA; omitted instructions are not shown.
// SHA256 of full assembly: 4f595292977afd6076cc2537e5d137b13228bcc788368b4adc1eb50704e888ac
.LBB0_2:
	s_addk_co_i32 s104, 0x100
	s_wait_dscnt 0x0
	s_cmp_lt_i32 s104, s80
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc0 .LBB0_310
.LBB0_3:
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
// Last store and the branch back to the loop:
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vdst(0)
	buffer_store_b64 v[2:3], v1, s[0:3], null offen scope:SCOPE_SYS
	s_branch .LBB0_2
.LBB0_310:
	s_wait_storecnt 0x0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
