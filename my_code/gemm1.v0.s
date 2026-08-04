	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
	.p2align	8
	.type	gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1,@function
gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1:
	// ============================================================================
	// S_SETREG_IMM32_B32 分类 1：开启有限专家调度模式（DEP_MODE=2）。
	//
	// s_setreg_imm32_b32 用一个随指令携带的 32-bit literal 修改 wave 硬件
	// 寄存器的指定字段。hwreg(REG, OFFSET, WIDTH) 给出寄存器、最低 bit 和字段
	// 宽度；literal 的低 WIDTH 位写入该字段，寄存器其他 bit 保持不变。
	//
	// 此处向 HW_REG_WAVE_SCHED_MODE[1:0] 写入二进制 10，即 DEP_MODE=2。
	// 正常模式下 SQ 会在发射前自动检查各类数据依赖；模式 2 只关闭 VA_VDST
	// 和 VM_VSRC 两项硬件检查，其余依赖检查仍保留。这样独立的 VALU 与
	// VMEM/DS 可以更紧密地交错发射，避免硬件过度保守造成的无谓停顿。
	//
	// 代价是软件必须用 s_wait_alu 或由其他 wait 指令提供的等价保证，显式处理
	// VALU 目的 VGPR 写回和 VMEM 源 VGPR 读取相关。漏掉必要等待会暴露
	// RAW/WAR/WAW hazard，并可能直接产生错误结果，而不只是性能下降。
	// ============================================================================
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	// ============================================================================
	// S_SETREG_IMM32_B32 分类 2：开启 multi-VMEM-group XNACK replay 模式。
	//
	// 此处只把 HW_REG_WAVE_MODE[25]（REPLAY_MODE）写成 1。值 0 是
	// single-VMEM-group：在 memory group 边界，硬件会通过 XCNT 自动施加较
	// 保守的等待；值 1 是 multi-VMEM-group，允许多个 SMEM/VMEM group 更自由
	// 地重叠，不再由硬件在这些边界自动等待 XCNT。
	//
	// 因此模式 1 下软件必须在真正需要 translation/XNACK 状态收敛的位置插入
	// s_wait_xcnt。该字段只能在 wave 开始附近、尚未发射任何 SMEM/VMEM 前
	// 修改，或者紧跟 s_wait_idle 后修改；本指令位于最前面的 s_load/global
	// load 之前，满足这一限制。若系统禁用 XNACK retry，本模式选择本身不参与
	// replay 调度，但 XCNT 仍可能用于跟踪迟到的 XNACK error。
	// ============================================================================
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x70 nv
	s_load_b96 s[52:54], s[0:1], 0xa4 nv
	v_mov_b32_e32 v16, 0
	s_mov_b32 s44, 1
	// ============================================================================
	// S_SETREG_IMM32_B32 分类 3：关闭 WMMA 后的 SQ 多周期仲裁停顿。
	//
	// 此处把 HW_REG_WAVE_SCHED_MODE[4] 写成 1。该字段文档中也称
	// DISABLE_XDL_ARB_STALL、DISABLE_VALU_ARB_STALL 或 DISABLE_VALU_STALL。
	// 因为只写 bit[4]，前面已经设置的 DEP_MODE[1:0] 不会被覆盖。
	//
	// 默认情况下，wave 发射一条多周期 XDL/WMMA 后，SQ arbiter 会让该 wave
	// 停到本次运算经过其多周期执行窗口，再给其他 wave 提供 co-execution
	// 机会。置 1 后取消这项“每条 WMMA 后都停”的策略，同一 wave 可连续发射
	// 多条 WMMA（文档预期约 5 条）后再去调度其他独立指令；本 kernel 后面的
	// 成组 v_wmma_scale 指令正是这种用法。
	//
	// 该 bit 只改变仲裁/发射节奏，并不会消除 WMMA 的执行延迟或数据 hazard；
	// 输入、累加器和结果仍须按 XDL 规则正确调度。连续占用 XDL 也会压缩同一
	// SIMD 上其他 wave 的 co-execution 机会，因此通常在每 SIMD 活跃 wave
	// 较少时收益更明显。
	// ============================================================================
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 4, 1), 1
	s_wait_kmcnt 0x0
	// ============================================================================
	// S_WAIT_ALU 分类 1：VA_VDST 全清空等待（depctr_va_vdst(0)）。
	//
	// 本 kernel 在开头把 DEP_MODE 设为 2，硬件不再自动检查 VA_VDST 与
	// VM_VSRC，因此这些 S_WAIT_ALU 是软件显式维护跨流水线 VGPR 依赖所必需的。
	// VA_VDST 在每条 VALU 发射时增加，并在该 VALU 完成 VGPR 写回后减少；
	// depctr_va_vdst(0) 要求计数器降到 0，即所有更早的 VALU 目的寄存器都已
	// 写回，之后才允许继续发射。
	//
	// 它主要保护“VALU 写 VGPR -> VMEM/DS/其他流水线读或写同一 VGPR”的
	// RAW/WAW 相关；它只等待 VALU 写回，不等待任何内存访问返回。此处前面的
	// v_mov 写 v16，而下一条 global_load 把 v16 当地址，所以必须先确认 v16
	// 已可供 VMEM 读取。若计数器本来就是 0，该等待可以不产生实际停顿。
	// ============================================================================
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v16, s[12:13] offset:768
	s_add_co_i32 s2, s53, 0xff
	v_mov_b32_e32 v4, 0xc0
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s22, 0
	s_lshr_b32 s3, s3, 24
	s_mov_b32 s20, 4
	s_add_co_i32 s3, s2, s3
	// ============================================================================
	// S_DELAY_ALU 分类 1：纯 SALU 周期依赖（SALU_CYCLE_N）。
	//
	// SALU_CYCLE_1/2/3 要求目标指令相对之前的 SALU 工作至少等待 1/2/3 个
	// 周期，用来覆盖 SGPR、SCC 或 carry/borrow 等标量结果尚未准备好的情况。
	// 它不是“倒数第 N 条 SALU”的编号；N 表示需要保证的 SALU 周期间隔。
	//
	// 只有 instid0 时，依赖作用于紧随 S_DELAY_ALU 的下一条指令。若同时编码
	// instid1，则一条 S_DELAY_ALU 可描述第二个目标：省略 instskip 时两个依赖
	// 都作用于下一条指令；NEXT 使 instid1 作用于再下一条指令；SKIP_k 表示在
	// 第一个目标之后跳过 k 条任意类型的真实指令，再把 instid1 用于第二目标。
	// 注释和标签不计入 SKIP。若所需间隔已经自然满足，本指令可以零周期完成。
	//
	// 此处下一条 s_and_b32 读取刚由 s_add_co_i32 写入的 s3，因此请求至少一个
	// SALU 周期，避免消费者过早读取标量结果。
	// ============================================================================
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s4, s3, 0xffffff00
	s_ashr_i32 s3, s3, 8
	s_cmp_lg_u32 s2, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, s4
	s_sub_co_ci_u32 s2, s3, 0
	s_add_co_i32 s3, s52, 15
	s_ashr_i32 s4, s3, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s4, s4, 28
	s_add_co_i32 s4, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s5, s4, -16
	s_ashr_i32 s4, s4, 4
	s_cmp_lg_u32 s3, s5
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_i32 s3, 0
	// ============================================================================
	// S_GETREG_B32 分类 1：读取当前 wave 的 workgroup-cluster 状态。
	//
	// s_getreg_b32 从 wave 硬件寄存器抽取一个 bit field，并将其右对齐、零扩展
	// 后写入目标 SGPR。hwreg(REG, OFFSET, WIDTH) 指定寄存器、最低 bit 和宽度；
	// 本指令读取只读的 IB_STS2[9:6]，因此 s3 最终得到 4-bit CLUSTER_ID。
	// CLUSTER_ID=0 表示当前 dispatch 未使用 workgroup cluster；1..15 表示 wave
	// 属于相应的物理 cluster。该物理 ID 在 context switch 前后不保证不变，
	// 但这里仅检查它是否为 0，所以只用它判断“是否处于 cluster 模式”。
	//
	// 后面的 s_cmp_eq_u32 根据该值选择 workgroup-X 的计算方法：非 cluster
	// 情况直接采用 ttmp9；cluster 情况则结合 ttmp9、TTMP6 中的 cluster 尺寸
	// 和本地 wg_x，得到展开后的 workgroup 位置。S_GETREG_B32 不修改 SCC，
	// 所以紧随其后的 s_cselect 仍使用前一条 s_cmp_lt_i32 留下的 SCC。
	//
	// S_GETREG_B32 本身不会隐式等待 wave idle；IB_STS2 的 cluster 字段在 wave
	// 启动时初始化且可直接读取，因此此处不需要额外的 s_wait_idle。
	// ============================================================================
	s_getreg_b32 s3, hwreg(HW_REG_IB_STS2, 6, 4)
	s_cselect_b32 s6, -1, 0
	s_bfe_u32 s7, ttmp6, 0x4000c
	s_and_b32 s9, ttmp6, 15
	s_add_co_i32 s7, s7, 1
	s_lshl_b32 s8, s2, 4
	s_mul_i32 s7, ttmp9, s7
	s_and_b32 s5, s6, s5
	s_add_co_i32 s9, s9, s7
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s3, ttmp9, s9
	s_abs_i32 s6, s8
	s_abs_i32 s10, s3
	s_cvt_f32_u32 s7, s6
	s_sub_co_i32 s9, 0, s6
	// ============================================================================
	// S_DELAY_ALU 分类 2：SALU 与 TRANS32 的跨流水线、双目标依赖。
	//
	// 这类编码把两种不同的等待打包在一条指令中。SALU_CYCLE_N 检查标量流水线
	// 的周期间隔；TRANS32_DEP_N 则引用倒数第 N 条 16/32-bit transcendental
	// VALU 指令（如 RCP、EXP、LOG 等）的 scoreboard 项，并等待其结果可供目标
	// 指令使用。两部分分别作用于 instskip 所选择的两个后续目标，并非要求同一
	// 条指令同时依赖两条生产者。
	//
	// 此例中 instid0(SALU_CYCLE_2) 作用于紧随其后的 v_rcp_iflag_f32，保证它
	// 读取的标量源 s7 已由 s_cvt_f32_u32 准备好；SKIP_1 跳过中间的 v_nop，
	// 使 instid1(TRANS32_DEP_1) 作用于 v_readfirstlane_b32，等待前面的 RCP
	// transcendental 结果写回 v2。若两项依赖均已满足，整条 delay 仍可零周期。
	// ============================================================================
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v2, s7
	v_nop
	v_readfirstlane_b32 s7, v2
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s7, s7
	s_mul_i32 s9, s9, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s9, s7, s9
	s_add_co_i32 s7, s7, s9
	s_xor_b32 s9, s3, s8
	s_mul_hi_u32 s7, s10, s7
	s_ashr_i32 s9, s9, 31
	s_mul_i32 s11, s7, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s10, s10, s11
	s_add_co_i32 s11, s7, 1
	s_sub_co_i32 s16, s10, s6
	s_cmp_ge_u32 s10, s6
	s_cselect_b32 s7, s11, s7
	s_cselect_b32 s10, s16, s10
	s_add_co_i32 s11, s7, 1
	s_cmp_ge_u32 s10, s6
	s_cselect_b32 s6, s11, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s6, s6, s9
	s_sub_co_i32 s7, s6, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s7, s7, s8
	s_cmp_lg_u32 s3, s7
	s_cselect_b32 s7, -1, 0
	s_xor_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s7
	s_sub_co_ci_u32 s2, s6, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s6, s2, s8
	s_lshl_b32 s7, s2, 4
	s_sub_co_i32 s3, s3, s6
	s_cmp_lg_u32 s5, 0
	s_sub_co_ci_u32 s2, s4, s7
	s_abs_i32 s9, s3
	s_min_i32 s4, s2, 16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_abs_i32 s5, s4
	s_cvt_f32_u32 s6, s5
	s_sub_co_i32 s8, 0, s5
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v2, s6
	v_nop
	v_readfirstlane_b32 s6, v2
	s_mul_f32 s6, s6, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s6, s6
	s_mul_i32 s8, s8, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s8, s6, s8
	s_add_co_i32 s6, s6, s8
	s_xor_b32 s8, s3, s4
	s_mul_hi_u32 s6, s9, s6
	s_ashr_i32 s8, s8, 31
	s_mul_i32 s10, s6, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s9, s9, s10
	s_add_co_i32 s10, s6, 1
	s_sub_co_i32 s11, s9, s5
	s_cmp_ge_u32 s9, s5
	s_cselect_b32 s6, s10, s6
	s_cselect_b32 s9, s11, s9
	s_add_co_i32 s10, s6, 1
	s_cmp_ge_u32 s9, s5
	s_cselect_b32 s5, s10, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s5, s5, s8
	s_sub_co_i32 s6, s5, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s6, s4, s6
	s_cmp_lg_u32 s3, s6
	s_cselect_b32 s6, -1, 0
	s_xor_b32 s2, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s6
	s_sub_co_ci_u32 s2, s5, s8
	s_add_co_i32 s3, s3, s7
	s_mul_i32 s4, s2, s4
	s_mov_b32 s5, s22
	s_sub_co_i32 s76, s3, s4
	s_mov_b32 s4, s22
	s_lshl_b32 s34, s76, 4
	s_mov_b32 s6, s22
	s_mov_b32 s7, s22
	v_readfirstlane_b32 s3, v0
	s_mov_b32 s8, 2
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v1
	v_cndmask_b32_e64 v1, 0xc1, 0, vcc_lo
	v_mov_b32_e32 v2, 0x60
	v_cndmask_b32_e32 v4, 0x180, v4, vcc_lo
	// ============================================================================
	// S_DELAY_ALU 分类 3：纯普通 VALU 数据依赖（VALU_DEP_N）。
	//
	// VALU_DEP_1..4 指向当前 wave 已发射的普通 VALU scoreboard 中倒数第
	// 1..4 条指令；这里的 N 按“已发射的 VALU 指令”倒数，而不是按汇编文本
	// 行数倒数。硬件只在对应生产者尚未达到可安全转发/写回的时刻插入必要周期，
	// 因此它描述的是精确的生产者-消费者关系，而不是固定长度的 s_nop。
	//
	// instid0 默认约束下一条指令；可用 instid1 与 instskip 再约束另一个后续
	// 目标。被跳过的 SALU、VALU、内存等真实指令都计入 SKIP（架构明确排除的
	// 特殊指令除外），但它们不会改变 VALU_DEP_N 所引用的既有 scoreboard 项。
	//
	// 此处下一条 v_cndmask 使用先前比较产生的条件及相关向量结果，编译器指定
	// 它依赖倒数第 2 条普通 VALU，硬件据此仅补足仍缺少的等待周期。
	// ============================================================================
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v2, 0x120, v2, vcc_lo
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v3, v2, s[12:13] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v3
	v_or_b32_e32 v5, 1, v2
	// ============================================================================
	// S_WAIT_ALU 分类 2：VM_VSRC 全清空等待（depctr_vm_vsrc(0)）。
	//
	// VM_VSRC 在 global/buffer/image/flat/scratch/LDS 等 VMEM 指令发射时增加，
	// 在该指令已经读取完全部源 VGPR（地址、store data 等）后减少。
	// depctr_vm_vsrc(0) 等待所有较早 VMEM 都完成“源操作数读取”，从而允许后续
	// VALU 安全覆盖那些源 VGPR，解决 VMEM 先读、VALU 后写同一 VGPR 的 WAR
	// hazard。它并不表示内存请求或 load 数据已经完成；后者仍由 LOADcnt、
	// STOREcnt、DScnt 等相应计数器管理。
	//
	// 此例中前面的 global_load 使用 v2 作为地址，后面的 v_dual_cndmask 将
	// 改写 v2。等待 VM_VSRC==0 可确保 global_load 已经锁存旧 v2，之后覆盖
	// v2 不会改变尚未取走的地址。
	// ============================================================================
	s_wait_alu depctr_vm_vsrc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v2, v4, v2 :: v_dual_cndmask_b32 v1, v5, v1
	v_add_nc_u32_e32 v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v3, 1, v3
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v4, v3, s[12:13] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v4
	v_dual_cndmask_b32 v2, v2, v3, vcc_lo :: v_dual_bitop2_b32 v5, 1, v3 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, v5, v1, vcc_lo
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v6, 1, v3
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v3, v6, s[12:13] scale_offset
	v_add_nc_u32_e32 v4, 1, v6
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v3
	v_dual_mov_b32 v3, v16 :: v_dual_mov_b32 v5, v16
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v2, v2, v6, vcc_lo :: v_dual_cndmask_b32 v4, v4, v1, vcc_lo
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u64_e32 v[6:7], v[4:5], v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b64 v[6:7], 1, v[6:7]
	v_lshlrev_b64_e32 v[8:9], 2, v[6:7]
	v_add_nc_u32_e32 v3, 1, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[8:9], s[12:13], v[8:9]
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v[8:9], off
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v1
	v_dual_cndmask_b32 v1, v3, v4 :: v_dual_cndmask_b32 v2, v2, v6
	v_lshl_add_u32 v6, v0, 4, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v3, v1, v2
	v_add_nc_u32_e32 v7, 0x10000, v6
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v8, 0x10800, v6
	v_add_nc_u32_e32 v9, 0x11000, v6
	v_add_nc_u32_e32 v10, 0x11800, v6
	v_lshrrev_b32_e32 v3, 1, v3
	v_add_nc_u32_e32 v11, 0x12000, v6
	v_add_nc_u32_e32 v12, 0x12800, v6
	v_add_nc_u32_e32 v13, 0x13000, v6
	// ============================================================================
	// S_WAIT_ALU 分类 3：VA_VDST 部分等待（depctr_va_vdst(K)，K > 0）。
	//
	// 该形式等待 VA_VDST <= K，而不是清空整个计数器。编译器知道真正相关的
	// VALU 生产者在未完成队列中的相对位置，因此可保留最多 K 条更年轻且与
	// 当前消费者无关的 VALU 在途，只等待相关生产者及其之前的项完成。K 是
	// “允许剩余的未完成项数量”，不是 cycle 数，也不是某条指令的编号。
	//
	// 此处 v_lshrrev 已生成 global_load 所需的地址 v3；它后面还有三条独立的
	// v_add（写 v11/v12/v13）。在这段同类普通 VALU 的有序链中，等待
	// VA_VDST<=3 即可保证较早的 v3 已写回，同时无需等待三条独立 v_add。
	// 后文的 va_vdst(1) 同理，只允许保留一项。混合 TRANS/XDL/普通 VALU、
	// EXEC 可能改变发射数量时，非零阈值必须按实际 ordering 谨慎计算。
	// ============================================================================
	s_wait_alu depctr_va_vdst(3)
	global_load_b32 v4, v3, s[12:13] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v4
	v_add_nc_u32_e32 v5, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v1, v5, v1 :: v_dual_cndmask_b32 v2, v2, v3
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v3, 1, v3
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v4, v3, s[12:13] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v4
	v_add_nc_u32_e32 v5, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v1, v5, v1 :: v_dual_cndmask_b32 v2, v2, v3
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v3, 1, v3
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v4, v3, s[12:13] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v4
	v_add_nc_u32_e32 v5, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v1, v5, v1 :: v_dual_cndmask_b32 v2, v2, v3
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v3, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v3, 1, v3
	v_min_u32_e32 v4, 0x17f, v3
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v4, v4, s[12:13] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v4
	v_add_nc_u32_e32 v5, 1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v1, v5, v1 :: v_dual_cndmask_b32 v2, v2, v3
	v_add_nc_u32_e32 v2, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v2, 1, v2
	v_min_u32_e32 v3, 0x17f, v2
	v_add_nc_u32_e32 v2, 1, v2
	s_wait_alu depctr_va_vdst(1)
	global_load_b32 v3, v3, s[12:13] scale_offset
	s_wait_loadcnt 0x0
	v_cmp_lt_i32_e32 vcc_lo, s34, v3
	v_cndmask_b32_e32 v48, v2, v1, vcc_lo
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b64_e32 v[2:3], s[4:5]
	v_mov_b64_e32 v[4:5], s[6:7]
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 0x180, v48
	v_cndmask_b32_e32 v1, 0x17f, v48, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 0x17f, v48
	s_wait_alu depctr_va_vdst(1)
	global_load_b32 v1, v1, s[12:13] scale_offset
	ds_store_b128 v6, v[2:5]
	ds_store_b128 v6, v[2:5] offset:2048
	ds_store_b128 v6, v[2:5] offset:4096
	ds_store_b128 v6, v[2:5] offset:6144
	ds_store_b128 v6, v[2:5] offset:8192
	ds_store_b128 v6, v[2:5] offset:10240
	ds_store_b128 v6, v[2:5] offset:12288
	ds_store_b128 v6, v[2:5] offset:14336
	ds_store_b128 v6, v[2:5] offset:16384
	ds_store_b128 v6, v[2:5] offset:18432
	ds_store_b128 v6, v[2:5] offset:20480
	ds_store_b128 v6, v[2:5] offset:22528
	ds_store_b128 v6, v[2:5] offset:24576
	ds_store_b128 v6, v[2:5] offset:26624
	ds_store_b128 v6, v[2:5] offset:28672
	ds_store_b128 v6, v[2:5] offset:30720
	ds_store_b128 v6, v[2:5] offset:32768
	ds_store_b128 v6, v[2:5] offset:34816
	ds_store_b128 v6, v[2:5] offset:36864
	ds_store_b128 v6, v[2:5] offset:38912
	ds_store_b128 v6, v[2:5] offset:40960
	ds_store_b128 v6, v[2:5] offset:43008
	ds_store_b128 v6, v[2:5] offset:45056
	ds_store_b128 v6, v[2:5] offset:47104
	ds_store_b128 v6, v[2:5] offset:49152
	ds_store_b128 v6, v[2:5] offset:51200
	ds_store_b128 v6, v[2:5] offset:53248
	ds_store_b128 v6, v[2:5] offset:55296
	ds_store_b128 v6, v[2:5] offset:57344
	ds_store_b128 v6, v[2:5] offset:59392
	ds_store_b128 v6, v[2:5] offset:61440
	ds_store_b128 v6, v[2:5] offset:63488
	ds_store_b128 v7, v[2:5]
	ds_store_b128 v8, v[2:5]
	ds_store_b128 v9, v[2:5]
	ds_store_b128 v10, v[2:5]
	ds_store_b128 v11, v[2:5]
	ds_store_b128 v12, v[2:5]
	ds_store_b128 v13, v[2:5]
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s4, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_4
	s_ashr_i32 s7, s53, 31
	s_mov_b32 s6, s53
	s_mov_b32 s13, s22
	s_add_nc_u64 s[10:11], s[6:7], 31
	s_mov_b64 s[16:17], 0xffffffffffffffe0
	s_lshr_b32 s12, s11, 27
	s_lshl_b32 s56, s2, 8
	s_add_nc_u64 s[12:13], s[10:11], s[12:13]
	s_lshl_b32 s2, s3, 1
	s_and_b64 s[16:17], s[12:13], s[16:17]
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s57, s56, 31
	s_and_b32 s71, s2, 0xc0
	s_lshr_b32 s77, s3, 3
	s_ashr_i64 s[2:3], s[12:13], 5
	s_cmp_lg_u64 s[10:11], s[16:17]
	s_clause 0x1
	s_load_b128 s[48:51], s[0:1], 0x28 nv
	s_load_b64 s[58:59], s[0:1], 0x38 nv
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_i32 s53, 0xffffffe1
	s_mul_u64 s[60:61], s[34:35], 0x1c00
	s_cselect_b32 s9, -1, 0
	s_lshr_b32 s10, s7, 28
	s_and_b32 s78, s9, s5
	s_add_co_i32 s10, s53, s10
	s_ashr_i64 s[12:13], s[56:57], 5
	s_ashr_i32 s10, s10, 4
	s_mov_b32 s63, s22
	s_ashr_i32 s11, s10, 31
	s_movk_i32 s21, 0x1c00
	s_lshl_b64 s[16:17], s[10:11], 4
	s_mov_b32 s23, s22
	s_cmp_lg_u64 s[16:17], s[6:7]
	s_mov_b32 s17, 0xffff0000
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_i32 s53, 0
	s_mov_b32 s16, 0x7500000
	s_cselect_b32 s72, -1, 0
	s_bfe_u32 s6, ttmp8, 0x50019
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v3, s22 :: v_dual_mov_b32 v49, v16
	s_and_b32 s33, s6, 3
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[6:7], s[48:49], s[60:61]
	s_lshl_b32 s52, s33, 2
	s_mul_i32 s62, s33, 0x7000
	s_mul_i32 s55, s33, 0x440
	s_add_nc_u64 s[46:47], s[62:63], s[6:7]
	s_or_b32 s6, s52, s34
	s_add_co_i32 s55, s55, 0
	s_sub_co_i32 s4, s4, s6
	s_bitset1_b32 s47, 31
	s_max_i32 s4, s4, 0
	s_mov_b32 s45, s55
	s_lshl_b32 s6, s4, 16
	s_lshr_b32 s70, s4, 16
	s_or_b32 s18, s6, 0x7fff
	s_or_b32 s19, s70, 0x1000000
	s_and_b32 s4, s72, s5
	tensor_load_to_lds s[44:47], s[16:23]
	v_cndmask_b32_e64 v2, 0, 1, s4
	s_lshr_b64 s[4:5], s[56:57], 4
	s_mul_i32 s64, s33, 0x38000
	s_mov_b32 s65, s22
	s_mov_b32 s29, 0xe000
	v_sub_nc_u64_e32 v[2:3], s[10:11], v[2:3]
	s_mov_b32 s27, 0x8007fff
	s_mov_b32 s24, s22
	s_mov_b32 s25, s17
	s_mov_b32 s28, s20
	s_mov_b32 s30, s22
	v_mul_u64_e32 v[2:3], v[2:3], v[48:49]
	s_mov_b32 s31, s22
	s_mul_u64 s[66:67], s[34:35], 0xe0
	s_mul_i32 s68, s33, 0x380
	s_lshl_b32 s74, s33, 5
	s_mov_b32 s69, s22
	s_mov_b32 s41, 56
	s_mov_b32 s39, 0x27fff
	s_mov_b32 s36, 0x20000
	s_mov_b32 s37, s17
	s_mov_b32 s40, s20
	s_mov_b32 s42, s22
	s_mov_b32 s43, s22
	s_mul_u64 s[84:85], s[12:13], 0x1c00
	s_mul_i32 s86, s33, 0x3800
	s_mov_b32 s87, s22
	s_movk_i32 s9, 0x700
	s_mov_b32 s10, s22
	s_mov_b32 s11, s22
	v_bfe_u32 v54, v0, 4, 1
	s_mulk_i32 s12, 0x1c00
	v_dual_mov_b32 v21, v16 :: v_dual_mov_b32 v22, v16
	v_dual_mov_b32 v23, v16 :: v_dual_mov_b32 v24, v16
	v_dual_mov_b32 v25, v16 :: v_dual_mov_b32 v26, v16
	v_dual_mov_b32 v27, v16 :: v_dual_mov_b32 v28, v16
	v_dual_mov_b32 v29, v16 :: v_dual_mov_b32 v30, v16
	v_dual_mov_b32 v31, v16 :: v_dual_mov_b32 v32, v16
	v_dual_mov_b32 v33, v16 :: v_dual_mov_b32 v34, v16
	v_dual_mov_b32 v35, v16 :: v_dual_mov_b32 v36, v16
	v_add_nc_u64_e32 v[2:3], s[4:5], v[2:3]
	s_lshl_b32 s4, s33, 13
	v_dual_mov_b32 v37, v16 :: v_dual_mov_b32 v38, v16
	s_add_co_i32 s73, s4, 0
	s_lshl_b32 s4, s33, 18
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u64_e32 v[6:7], 0xe000, v[2:3]
	s_addk_co_i32 s73, 0x1100
	s_xor_b32 s26, s4, 0xffff7fff
	s_mov_b32 s45, s73
	s_mov_b32 s38, s26
	v_dual_mov_b32 v39, v16 :: v_dual_mov_b32 v40, v16
	v_dual_mov_b32 v41, v16 :: v_dual_mov_b32 v42, v16
	v_dual_mov_b32 v43, v16 :: v_dual_mov_b32 v44, v16
	v_dual_mov_b32 v45, v16 :: v_dual_mov_b32 v46, v16
	v_dual_mov_b32 v47, v16 :: v_dual_bitop2_b32 v55, 15, v0 bitop3:0x40
	v_dual_mov_b32 v18, v16 :: v_dual_mov_b32 v17, v16
	v_dual_mov_b32 v19, v16 :: v_dual_mov_b32 v20, v16
	v_add_nc_u64_e32 v[2:3], s[50:51], v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_nc_u64_e32 v[8:9], s[64:65], v[2:3]
	v_mov_b64_e32 v[2:3], s[44:45]
	v_mov_b64_e32 v[4:5], s[46:47]
	v_or_b32_e32 v1, 0x80000000, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v4, v8
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_cndmask_b32_e64 v2, 0, 1, s78
	v_mov_b32_e32 v5, v1
	v_readfirstlane_b32 s6, v4
	v_mov_b32_e32 v3, s22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s7, v5
	v_sub_nc_u64_e32 v[2:3], s[2:3], v[2:3]
	s_lshl_b32 s3, s33, 9
	// ============================================================================
	// S_DELAY_ALU 分类 4：SALU 与普通 VALU 的跨流水线、双目标依赖。
	//
	// 此类指令在一个 SIMM16 中同时携带 SALU_CYCLE_N 和 VALU_DEP_N，让两个
	// 不同位置的消费者分别等待标量流水线和普通向量流水线。字段出现于 instid0
	// 或 instid1 并不决定流水线类型；字段里的 SALU_CYCLE/VALU_DEP 才决定检查
	// 哪类状态，而 instskip 只决定第二项依赖绑定到哪条后续指令。
	//
	// 此例的 SALU_CYCLE_1 约束下一条 s_add_co_i32，保证前一条 s_lshl_b32
	// 写入的 s3 可读。SKIP_2 跳过随后两条 SALU，使 VALU_DEP_1 绑定到
	// v_mul_u64，并等待最近一条普通 VALU 生产者。后文把 VALU_DEP 放在
	// instid0、SALU_CYCLE 放在 instid1 的写法语义相同，只是两个目标顺序相反。
	// ============================================================================
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_add_co_i32 s75, s3, 0
	s_lshl_b32 s3, s33, 17
	s_add_co_i32 s75, s75, 0x9180
	v_mul_u64_e32 v[2:3], v[2:3], v[48:49]
	v_and_or_b32 v49, 0x1ffffff0, s77, v55
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u64_e32 v[8:9], 0x1c00, v[2:3]
	tensor_load_to_lds s[4:7], s[24:31]
	s_add_nc_u64 s[4:5], s[58:59], s[66:67]
	s_add_co_i32 s6, s74, 0
	s_add_nc_u64 s[46:47], s[68:69], s[4:5]
	s_add_co_i32 s45, s6, 0x9100
	s_bitset1_b32 s47, 31
	s_xor_b32 s6, s3, 0xffff7fff
	s_mov_b32 s7, 0x407fff
	s_lshl_b32 s3, s71, 7
	s_cmp_lg_u32 s78, 0
	s_sub_co_ci_u32 s2, s2, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mul_lo_u32 v0, v48, s2
	tensor_load_to_lds s[44:47], s[36:43]
	s_load_b64 s[46:47], s[0:1], 0x60 nv
	s_mov_b32 s45, s75
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[46:47], s[84:85]
	s_delay_alu instid0(VALU_DEP_4) | instid1(SALU_CYCLE_1)
	v_add_nc_u64_e32 v[2:3], s[4:5], v[8:9]
	s_mov_b32 s4, s36
	s_mov_b32 s5, s17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_nc_u64_e32 v[10:11], s[86:87], v[2:3]
	v_mov_b64_e32 v[2:3], s[44:45]
	v_mov_b64_e32 v[4:5], s[46:47]
	s_mul_i32 s45, s76, 0x1c000
	s_add_co_i32 s45, s45, s62
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_or_b32_e32 v1, 0x80000000, v11
	v_mov_b32_e32 v4, v10
	v_readfirstlane_b32 s80, v2
	v_readfirstlane_b32 s81, v3
	v_dual_mov_b32 v5, v1 :: v_dual_lshlrev_b32 v2, 4, v55
	s_delay_alu instid0(VALU_DEP_4)
	v_readfirstlane_b32 s82, v4
	v_lshlrev_b32_e32 v1, 8, v54
	v_mul_lo_u32 v4, 0x1c00, v0
	v_lshlrev_b32_e32 v3, 2, v55
	v_readfirstlane_b32 s83, v5
	v_lshlrev_b32_e32 v5, 4, v54
	v_or3_b32 v2, s3, v1, v2
	v_add_nc_u64_e32 v[0:1], s[46:47], v[8:9]
	v_lshl_or_b32 v56, s71, 3, v3
	v_lshlrev_b32_e32 v3, 2, v54
	s_add_nc_u64 s[2:3], s[50:51], s[64:65]
	v_add_nc_u32_e32 v59, 0x1100, v2
	v_add_nc_u32_e32 v2, s86, v4
	v_add_nc_u64_e32 v[0:1], s[86:87], v[0:1]
	v_lshl_or_b32 v57, v49, 3, v3
	v_mad_u32 v58, 0x110, v49, v5
	v_or_b32_e32 v60, 0x9180, v56
	v_add_nc_u32_e32 v4, s12, v2
	v_add_nc_u64_e32 v[2:3], s[2:3], v[6:7]
	v_add_nc_u64_e32 v[0:1], s[84:85], v[0:1]
	s_add_nc_u64 s[12:13], s[58:59], s[68:69]
	s_add_co_i32 s45, s45, s48
	v_add_nc_u32_e32 v4, s46, v4
	s_add_nc_u64 s[2:3], s[12:13], s[66:67]
	v_add_nc_u64_e32 v[50:51], 0x800, v[2:3]
	v_add_nc_u64_e32 v[52:53], 0x100, v[0:1]
	s_add_nc_u64 s[12:13], s[48:49], s[62:63]
	v_add_nc_u32_e32 v61, 0x100, v4
	s_add_nc_u64 s[12:13], s[12:13], s[60:61]
	s_add_nc_u64 s[2:3], s[2:3], 8
	s_add_nc_u64 s[12:13], s[12:13], 0x100
	s_add_co_i32 s50, s45, 0x100
	s_mov_b64 s[48:49], 0
	s_mov_b32 s51, 1
	tensor_load_to_lds s[80:83], s[4:11]
.LBB0_2:
	s_bitcmp1_b32 s51, 0
	s_wait_tensorcnt 0x0
	s_cselect_b32 s45, 0, 0x9a00
	s_cselect_b32 s58, 0x9a00, 0
	s_add_co_i32 s59, s45, 0
	// ============================================================================
	// S_WAIT_ALU 分类 4：VM_VSRC 部分等待（depctr_vm_vsrc(K)，0 < K < 7）。
	//
	// 该形式等待 VM_VSRC <= K，允许最多 K 条较年轻、无关的 VMEM 源读取继续
	// 在途，只排空会与后续 VGPR 写入冲突的较老项。VM_VSRC 硬件计数器为
	// 4 bit，但指令只编码 3 bit 阈值：0..6 是实际等待值，7 表示不等待。
	// 非零 K 仍只表示“源 VGPR 是否已被读取”，不表示内存操作已经完成。
	//
	// 这里位于软件流水化循环入口：上一轮的 DS 指令曾以 v95/v100/v101 作为
	// 地址，下一轮即将重写这些寄存器。vm_vsrc(5) 在重写 v95/v100 前排到
	// <=5，随后 vm_vsrc(4) 在重写 v101 前再排到 <=4；后文的值 6 也是相同
	// 的部分排空机制。非零阈值依赖 VMEM ordering group；若依赖跨多个 group，
	// 必须采用能覆盖全部相关项的最小阈值。
	// ============================================================================
	s_wait_alu depctr_vm_vsrc(5)
	v_dual_add_nc_u32 v95, s59, v59 :: v_dual_add_nc_u32 v100, s59, v60
	s_wait_alu depctr_vm_vsrc(4)
	v_add_nc_u32_e32 v101, s59, v57
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[62:65], v95
	ds_load_b128 v[66:69], v95 offset:512
	ds_load_b128 v[70:73], v95 offset:2048
	ds_load_b128 v[74:77], v95 offset:2560
	ds_load_b128 v[78:81], v95 offset:4096
	ds_load_b128 v[82:85], v95 offset:4608
	ds_load_b128 v[86:89], v95 offset:6144
	ds_load_b128 v[90:93], v95 offset:6656
	ds_load_2addr_b32 v[96:97], v100 offset1:16
	ds_load_2addr_b32 v[98:99], v100 offset0:64 offset1:80
	ds_load_b32 v94, v101 offset:37120
	s_add_nc_u64 s[46:47], s[12:13], s[48:49]
	s_add_co_i32 s45, s55, s58
	s_add_co_i32 s46, s50, s48
	s_bitset1_b32 s47, 31
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v4, 0x80000000, v51
	tensor_load_to_lds s[44:47], s[16:23]
	s_add_co_i32 s45, s73, s58
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mov_b64_e32 v[0:1], s[44:45]
	v_mov_b64_e32 v[2:3], s[46:47]
	v_dual_mov_b32 v2, v50 :: v_dual_mov_b32 v3, v4
	s_add_co_i32 s45, s58, 0
	s_or_b32 s47, s3, 0x80000000
	v_readfirstlane_b32 s60, v0
	v_readfirstlane_b32 s61, v1
	v_readfirstlane_b32 s62, v2
	v_readfirstlane_b32 s63, v3
	s_add_co_i32 s45, s45, s74
	s_mov_b32 s46, s2
	s_add_co_i32 s45, s45, 0x9100
	v_add_nc_u64_e32 v[0:1], s[48:49], v[52:53]
	v_add_nc_u32_e32 v4, s48, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_or_b32_e32 v5, 0x80000000, v1
	tensor_load_to_lds s[60:63], s[24:31]
	tensor_load_to_lds s[44:47], s[36:43]
	s_add_co_i32 s45, s75, s58
	v_mov_b64_e32 v[0:1], s[44:45]
	v_mov_b64_e32 v[2:3], s[46:47]
	v_dual_mov_b32 v2, v4 :: v_dual_mov_b32 v3, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s60, v0
	v_readfirstlane_b32 s61, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s62, v2
	v_readfirstlane_b32 s63, v3
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[60:63], s[4:11]
	v_add_nc_u32_e32 v102, s59, v58
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v102
	ds_load_b128 v[4:7], v102 offset:32
	ds_load_b128 v[8:11], v102 offset:64
	ds_load_b128 v[12:15], v102 offset:96
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[62:69], v[0:15], v[16:23], v96, v94 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[70:77], v[0:15], v[24:31], v97, v94 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[78:85], v[0:15], v[32:39], v98, v94 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[86:93], v[0:15], v[40:47], v99, v94 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[62:65], v95 offset:1024
	ds_load_b128 v[66:69], v95 offset:1536
	ds_load_b128 v[70:73], v95 offset:3072
	ds_load_b128 v[74:77], v95 offset:3584
	ds_load_b128 v[78:81], v95 offset:5120
	ds_load_b128 v[82:85], v95 offset:5632
	ds_load_b128 v[86:89], v95 offset:7168
	ds_load_b128 v[90:93], v95 offset:7680
	ds_load_2addr_b32 v[96:97], v100 offset0:32 offset1:48
	ds_load_2addr_b32 v[98:99], v100 offset0:96 offset1:112
	ds_load_b32 v94, v101 offset:37124
	ds_load_b128 v[0:3], v102 offset:128
	ds_load_b128 v[4:7], v102 offset:160
	ds_load_b128 v[8:11], v102 offset:192
	ds_load_b128 v[12:15], v102 offset:224
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[62:69], v[0:15], v[16:23], v96, v94 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[70:77], v[0:15], v[24:31], v97, v94 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[78:85], v[0:15], v[32:39], v98, v94 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[86:93], v[0:15], v[40:47], v99, v94 matrix_a_fmt:MATRIX_FMT_FP4
	v_add_nc_u64_e32 v[50:51], 0x800, v[50:51]
	s_add_nc_u64 s[48:49], s[48:49], 0x100
	s_add_co_i32 s51, s51, 1
	s_cmp_lg_u32 s48, 0x1b00
	s_add_nc_u64 s[2:3], s[2:3], 8
	s_cbranch_scc1 .LBB0_2
	s_add_co_i32 s2, 0, 0x9a00
	s_add_co_i32 s3, 0, 0x12b80
	v_dual_add_nc_u32 v53, s2, v59 :: v_dual_add_nc_u32 v97, s2, v57
	v_add_nc_u32_e32 v98, s2, v58
	s_lshl_b32 s2, s71, 1
	v_add_nc_u32_e32 v96, s3, v56
	s_or_b32 s2, s2, 64
	s_wait_tensorcnt 0x0
	v_and_or_b32 v51, 0x1c0, s2, v55
	s_barrier_signal -1
	s_barrier_wait -1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v52, v51, 2, s3
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[56:59], v53
	ds_load_b128 v[60:63], v53 offset:512
	ds_load_2addr_b32 v[92:93], v96 offset1:16
	ds_load_b32 v50, v97 offset:37120
	ds_load_b128 v[0:3], v98
	ds_load_b128 v[4:7], v98 offset:32
	ds_load_b128 v[8:11], v98 offset:64
	ds_load_b128 v[12:15], v98 offset:96
	ds_load_b128 v[64:67], v53 offset:2048
	ds_load_b128 v[68:71], v53 offset:2560
	ds_load_b128 v[72:75], v53 offset:4096
	ds_load_b128 v[76:79], v53 offset:4608
	ds_load_b128 v[80:83], v53 offset:6144
	ds_load_b128 v[84:87], v53 offset:6656
	ds_load_b32 v51, v96 offset:256
	ds_load_b32 v52, v52 offset:64
	s_wait_dscnt 0x0
	ds_load_b128 v[88:91], v53 offset:1024
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[56:63], v[0:15], v[16:23], v92, v50 matrix_a_fmt:MATRIX_FMT_FP4
	s_load_b64 s[26:27], s[0:1], 0x0 nv
	s_mov_b32 s23, 0
	s_mov_b32 s24, 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[64:71], v[0:15], v[24:31], v93, v50 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[72:79], v[0:15], v[32:39], v51, v50 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[80:87], v[0:15], v[40:47], v52, v50 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_alu depctr_vm_vsrc(6)
	ds_load_b128 v[92:95], v53 offset:1536
	ds_load_b128 v[56:59], v53 offset:3072
	ds_load_b128 v[60:63], v53 offset:3584
	ds_load_b128 v[64:67], v53 offset:5120
	ds_load_b128 v[68:71], v53 offset:5632
	ds_load_b128 v[72:75], v53 offset:7168
	ds_load_b128 v[76:79], v53 offset:7680
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_2addr_b32 v[52:53], v96 offset0:32 offset1:48
	ds_load_2addr_b32 v[80:81], v96 offset0:96 offset1:112
	ds_load_b32 v50, v97 offset:37124
	ds_load_b128 v[0:3], v98 offset:128
	ds_load_b128 v[4:7], v98 offset:160
	ds_load_b128 v[8:11], v98 offset:192
	ds_load_b128 v[12:15], v98 offset:224
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[88:95], v[0:15], v[16:23], v52, v50 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[56:63], v[0:15], v[24:31], v53, v50 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[64:71], v[0:15], v[32:39], v80, v50 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[72:79], v[0:15], v[40:47], v81, v50 matrix_a_fmt:MATRIX_FMT_FP4
	v_lshlrev_b32_e32 v52, 3, v54
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v53, s71, v52
	s_wait_xcnt 0x0
	s_sub_f32 s0, 0, s54
	s_lshr_b32 s1, s53, 31
	s_mov_b32 s17, 0xffff0000
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_mad_u32 v0, v48, s53, v53
	v_lshlrev_b32_e32 v48, 8, v49
	s_add_co_i32 s1, s53, s1
	s_mov_b32 s20, 4
	s_mov_b32 s16, 0x10000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_nc_u32 v49, 0, v48 :: v_dual_add_nc_u32 v64, v53, v48
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v65, v53, v49, 16
	v_add3_u32 v66, v52, s71, v49
	v_lshl_add_u64 v[50:51], v[0:1], 1, s[14:15]
	s_ashr_i32 s15, s1, 1
	s_and_b32 s1, s1, -2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s53, s1
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x3
	global_load_b128 v[0:3], v[50:51], off
	global_load_b128 v[4:7], v[50:51], off offset:32
	global_load_b128 v[8:11], v[50:51], off offset:64
	global_load_b128 v[12:15], v[50:51], off offset:96
	s_cselect_b32 s1, -1, 0
	s_lshl_b32 s22, s52, 1
	s_and_b32 s1, s72, s1
	s_or_b32 s19, s70, 0x800000
	v_cndmask_b32_e64 v67, 0, 1, s1
	s_wait_loadcnt 0x3
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_and_b32_e32 v51, 0xffff0000, v1
	v_lshlrev_b32_e32 v50, 16, v1
	v_and_b32_e32 v49, 0xffff0000, v0
	v_lshlrev_b32_e32 v48, 16, v0
	v_and_b32_e32 v1, 0xffff0000, v3
	v_lshlrev_b32_e32 v0, 16, v3
	v_pk_add_f32 v[18:19], v[18:19], v[50:51]
	v_and_b32_e32 v3, 0xffff0000, v2
	v_lshlrev_b32_e32 v2, 16, v2
	s_wait_loadcnt 0x2
	v_and_b32_e32 v55, 0xffff0000, v5
	v_lshlrev_b32_e32 v54, 16, v5
	s_wait_loadcnt 0x1
	v_and_b32_e32 v57, 0xffff0000, v8
	v_lshlrev_b32_e32 v56, 16, v8
	v_pk_add_f32 v[16:17], v[16:17], v[48:49]
	v_cmp_gt_f32_e32 vcc_lo, s54, v18
	v_pk_add_f32 v[2:3], v[20:21], v[2:3]
	v_pk_add_f32 v[20:21], v[26:27], v[54:55]
	v_pk_add_f32 v[26:27], v[32:33], v[56:57]
	v_pk_add_f32 v[0:1], v[22:23], v[0:1]
	v_cndmask_b32_e32 v33, s54, v18, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v16
	v_and_b32_e32 v53, 0xffff0000, v4
	v_lshlrev_b32_e32 v52, 16, v4
	v_and_b32_e32 v59, 0xffff0000, v9
	v_dual_lshlrev_b32 v58, 16, v9 :: v_dual_cndmask_b32 v32, s54, v16, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v19, -s54
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[22:23], v[24:25], v[52:53]
	v_and_b32_e32 v5, 0xffff0000, v7
	v_pk_add_f32 v[24:25], v[34:35], v[58:59]
	v_lshlrev_b32_e32 v4, 16, v7
	v_cndmask_b32_e32 v18, s0, v19, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v17, -s54
	v_and_b32_e32 v7, 0xffff0000, v6
	v_lshlrev_b32_e32 v6, 16, v6
	v_pk_add_f32 v[4:5], v[30:31], v[4:5]
	v_and_b32_e32 v9, 0xffff0000, v11
	v_cndmask_b32_e32 v19, s0, v17, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v0
	v_pk_add_f32 v[6:7], v[28:29], v[6:7]
	v_lshlrev_b32_e32 v8, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v10
	v_dual_cndmask_b32 v17, s54, v0 :: v_dual_lshlrev_b32 v10, 16, v10
	v_cmp_gt_f32_e32 vcc_lo, s54, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f32 v[8:9], v[38:39], v[8:9]
	s_wait_loadcnt 0x0
	v_and_b32_e32 v63, 0xffff0000, v13
	v_pk_add_f32 v[10:11], v[36:37], v[10:11]
	v_dual_lshlrev_b32 v62, 16, v13 :: v_dual_cndmask_b32 v16, s54, v2, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v1, -s54
	v_and_b32_e32 v61, 0xffff0000, v12
	v_lshlrev_b32_e32 v60, 16, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f32 v[28:29], v[42:43], v[62:63]
	v_and_b32_e32 v13, 0xffff0000, v15
	v_cndmask_b32_e32 v34, s0, v1, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v3, -s54
	v_pk_add_f32 v[30:31], v[40:41], v[60:61]
	v_lshlrev_b32_e32 v12, 16, v15
	v_and_b32_e32 v15, 0xffff0000, v14
	v_dual_cndmask_b32 v35, s0, v3 :: v_dual_lshlrev_b32 v14, 16, v14
	v_cmp_gt_f32_e32 vcc_lo, s54, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[12:13], v[46:47], v[12:13]
	v_mul_f32_e32 v36, 0xbfb8aa3b, v32
	v_pk_add_f32 v[14:15], v[44:45], v[14:15]
	v_mul_f32_e32 v37, 0xbfb8aa3b, v33
	v_cndmask_b32_e32 v1, s54, v20, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v22
	v_dual_mul_f32 v38, 0xbfb8aa3b, v16 :: v_dual_mul_f32 v39, 0xbfb8aa3b, v17
	v_cndmask_b32_e32 v0, s54, v22, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v21, -s54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_gt_f32_e64 s1, 0xc2fc0000, v38
	v_cmp_gt_f32_e64 s2, 0xc2fc0000, v39
	v_cndmask_b32_e32 v20, s0, v21, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v23, -s54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v50, 0, 0x42800000, s1
	v_cndmask_b32_e64 v51, 0, 0x42800000, s2
	v_cndmask_b32_e32 v22, s0, v23, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v4
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v38, v38, v50 :: v_dual_add_f32 v39, v39, v51
	v_cndmask_b32_e64 v50, 0, 0xffffffc0, s1
	v_cndmask_b32_e64 v51, 0, 0xffffffc0, s2
	v_cndmask_b32_e32 v3, s54, v4, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v6
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v41, 0xbfb8aa3b, v3
	v_cndmask_b32_e32 v2, s54, v6, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v5, -s54
	v_cmp_gt_f32_e64 s6, 0xc2fc0000, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v40, 0xbfb8aa3b, v2
	v_cndmask_b32_e32 v21, s0, v5, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v7, -s54
	v_cndmask_b32_e64 v55, 0, 0x42800000, s6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_cmp_gt_f32_e64 s5, 0xc2fc0000, v40
	v_cndmask_b32_e32 v23, s0, v7, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v24
	v_add_f32_e32 v41, v41, v55
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v54, 0, 0x42800000, s5
	v_cndmask_b32_e64 v55, 0, 0xffffffc0, s6
	v_cndmask_b32_e32 v5, s54, v24, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v26
	v_add_f32_e32 v40, v40, v54
	v_exp_f32_e32 v41, v41
	v_cndmask_b32_e64 v54, 0, 0xffffffc0, s5
	v_mul_f32_e32 v43, 0xbfb8aa3b, v5
	v_cndmask_b32_e32 v4, s54, v26, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v25, -s54
	v_exp_f32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s8, 0xc2fc0000, v43
	v_mul_f32_e32 v42, 0xbfb8aa3b, v4
	v_cndmask_b32_e32 v24, s0, v25, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v27, -s54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v57, 0, 0x42800000, s8
	v_cmp_gt_f32_e64 s7, 0xc2fc0000, v42
	v_cndmask_b32_e32 v26, s0, v27, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v57
	v_cndmask_b32_e64 v56, 0, 0x42800000, s7
	v_cndmask_b32_e64 v57, 0, 0xffffffc0, s8
	v_cndmask_b32_e32 v7, s54, v8, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v42, v42, v56
	v_exp_f32_e32 v43, v43
	v_cndmask_b32_e64 v56, 0, 0xffffffc0, s7
	v_mul_f32_e32 v45, 0xbfb8aa3b, v7
	v_cndmask_b32_e32 v6, s54, v10, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v9, -s54
	v_exp_f32_e32 v42, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s10, 0xc2fc0000, v45
	v_mul_f32_e32 v44, 0xbfb8aa3b, v6
	v_cndmask_b32_e32 v25, s0, v9, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v11, -s54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v59, 0, 0x42800000, s10
	v_cmp_gt_f32_e64 s9, 0xc2fc0000, v44
	v_cndmask_b32_e32 v27, s0, v11, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v45, v45, v59
	v_cndmask_b32_e64 v58, 0, 0x42800000, s9
	v_cndmask_b32_e64 v59, 0, 0xffffffc0, s10
	v_cndmask_b32_e32 v9, s54, v28, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v44, v44, v58
	v_exp_f32_e32 v45, v45
	v_cndmask_b32_e64 v58, 0, 0xffffffc0, s9
	v_mul_f32_e32 v47, 0xbfb8aa3b, v9
	v_cndmask_b32_e32 v8, s54, v30, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v29, -s54
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s12, 0xc2fc0000, v47
	v_mul_f32_e32 v46, 0xbfb8aa3b, v8
	v_cndmask_b32_e32 v28, s0, v29, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v31, -s54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v61, 0, 0x42800000, s12
	v_cmp_gt_f32_e64 s11, 0xc2fc0000, v46
	v_cndmask_b32_e32 v30, s0, v31, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v61
	v_cndmask_b32_e64 v60, 0, 0x42800000, s11
	v_cndmask_b32_e64 v61, 0, 0xffffffc0, s12
	v_cndmask_b32_e32 v11, s54, v12, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v46, v46, v60
	v_exp_f32_e32 v47, v47
	v_cndmask_b32_e64 v60, 0, 0xffffffc0, s11
	v_mul_f32_e32 v49, 0xbfb8aa3b, v11
	v_cndmask_b32_e32 v10, s54, v14, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v13, -s54
	v_exp_f32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s14, 0xc2fc0000, v49
	v_mul_f32_e32 v48, 0xbfb8aa3b, v10
	v_cndmask_b32_e32 v29, s0, v13, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v15, -s54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v63, 0, 0x42800000, s14
	v_cmp_gt_f32_e64 s13, 0xc2fc0000, v48
	v_cndmask_b32_e32 v31, s0, v15, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v18
	v_cmp_gt_f32_e64 s0, 0xc2fc0000, v37
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v62, 0, 0x42800000, s13
	v_add_f32_e32 v49, v49, v63
	v_cndmask_b32_e64 v63, 0, 0xffffffc0, s14
	v_cndmask_b32_e32 v13, s54, v18, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v34
	v_add_f32_e32 v48, v48, v62
	v_exp_f32_e32 v49, v49
	v_cndmask_b32_e64 v62, 0, 0xffffffc0, s13
	v_cndmask_b32_e32 v15, s54, v34, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v35
	v_mul_f32_e32 v34, 0xbfb8aa3b, v0
	v_exp_f32_e32 v48, v48
	v_cndmask_b32_e32 v14, s54, v35, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v19
	v_mul_f32_e32 v35, 0xbfb8aa3b, v1
	v_cmp_gt_f32_e64 s3, 0xc2fc0000, v34
	v_cndmask_b32_e32 v12, s54, v19, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_gt_f32_e64 s4, 0xc2fc0000, v35
	v_cndmask_b32_e64 v52, 0, 0x42800000, s3
	v_cndmask_b32_e32 v19, s54, v20, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v53, 0, 0x42800000, s4
	v_add_f32_e32 v34, v34, v52
	v_cndmask_b32_e64 v52, 0, 0xffffffc0, s3
	v_cndmask_b32_e32 v21, s54, v21, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v23
	v_add_f32_e32 v35, v35, v53
	v_exp_f32_e32 v34, v34
	v_cndmask_b32_e64 v53, 0, 0xffffffc0, s4
	s_lshl_b32 s4, s33, 6
	v_cndmask_b32_e32 v20, s54, v23, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v22
	v_exp_f32_e32 v35, v35
	s_sub_co_i32 s25, s55, s4
	// ============================================================================
	// S_DELAY_ALU 分类 5：纯 TRANS32 数据依赖（TRANS32_DEP_N）。
	//
	// TRANS32_DEP_1..3 从独立的 transcendental/XDL scoreboard 中倒数引用
	// 16/32-bit SIN、COS、SQRT、RCP、RSQ、TANH、EXP、LOG 或 XDL WMMA。
	// 它不会把中间的普通 VALU 算入 N，因此可在普通 VALU 与长延迟 trans/XDL
	// 交错发射时，精确等待需要的生产者。gfx1250 还把组合的 LD_SCALE+WMMA
	// 视为一个 TRANS 项；其中 LD_SCALE 自身不另占 scoreboard 项。
	//
	// 此例中最近两条 TRANS 是 v_exp_f32 v35 与更早的 v_exp_f32 v34。
	// instid0(TRANS32_DEP_2) 让紧随其后的 v_ldexp_f32 v34 等待 v34 的 EXP；
	// SKIP_2 跳过两条中间指令，再由 instid1(TRANS32_DEP_1) 让
	// v_ldexp_f32 v35 等待最近的 v35 EXP。这样一条 delay 同时覆盖两条链。
	// ============================================================================
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v34, v34, v52
	v_cndmask_b32_e32 v18, s54, v22, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v24
	v_ldexp_f32 v35, v35, v53
	v_cndmask_b32_e32 v23, s54, v24, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v25
	v_cndmask_b32_e32 v25, s54, v25, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v27
	v_cndmask_b32_e32 v24, s54, v27, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v26
	v_cndmask_b32_e32 v22, s54, v26, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v28
	v_cndmask_b32_e32 v27, s54, v28, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v29
	v_cndmask_b32_e32 v29, s54, v29, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v31
	v_cndmask_b32_e32 v28, s54, v31, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s54, v30
	v_cndmask_b32_e64 v31, 0, 0x42800000, s0
	v_cndmask_b32_e32 v26, s54, v30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, 0xc2fc0000, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_f32_e32 v31, v37, v31
	v_cndmask_b32_e64 v37, 0, 0xffffffc0, s0
	v_readfirstlane_b32 s0, v67
	v_cndmask_b32_e64 v30, 0, 0x42800000, vcc_lo
	v_exp_f32_e32 v31, v31
	s_sub_co_i32 s0, s15, s0
	// ============================================================================
	// S_DELAY_ALU 分类 6：普通 VALU 与 TRANS32 的跨流水线、双目标依赖。
	//
	// 该类别同时引用普通 VALU scoreboard 和 transcendental/XDL scoreboard，
	// 用于一段交错调度代码中两个不同消费者。VALU_DEP_N 只在普通 VALU 历史中
	// 倒数，TRANS32_DEP_N 只在 16/32-bit trans/XDL 历史中倒数；两者的 N
	// 互不影响。instskip 再把第二项依赖准确绑定到稍后的目标。
	//
	// 此例中 VALU_DEP_1 约束下一条 v_add_f32。SKIP_3 跳过随后的三条指令，
	// 将 TRANS32_DEP_1 绑定到 v_ldexp_f32 v31，使它等待最近的
	// v_exp_f32 v31，而不必阻塞夹在两者之间的独立 SALU/VALU 工作。
	// ============================================================================
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v30, v36, v30
	v_cndmask_b32_e64 v36, 0, 0xffffffc0, vcc_lo
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s21, s0
	v_ldexp_f32 v31, v31, v37
	v_exp_f32_e32 v30, v30
	v_ldexp_f32 v37, v39, v51
	v_ldexp_f32 v39, v41, v55
	v_ldexp_f32 v41, v43, v57
	v_ldexp_f32 v43, v45, v59
	v_ldexp_f32 v45, v47, v61
	v_ldexp_f32 v47, v49, v63
	v_add_f32_e32 v31, 1.0, v31
	v_ldexp_f32 v30, v30, v36
	v_ldexp_f32 v36, v38, v50
	v_ldexp_f32 v38, v40, v54
	v_ldexp_f32 v40, v42, v56
	v_ldexp_f32 v42, v44, v58
	v_ldexp_f32 v44, v46, v60
	v_ldexp_f32 v46, v48, v62
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v36, 1.0, v36
	v_dual_add_f32 v37, 1.0, v37 :: v_dual_add_f32 v48, 1.0, v34
	v_dual_add_f32 v49, 1.0, v35 :: v_dual_add_f32 v38, 1.0, v38
	v_dual_add_f32 v39, 1.0, v39 :: v_dual_add_f32 v40, 1.0, v40
	v_dual_add_f32 v41, 1.0, v41 :: v_dual_add_f32 v42, 1.0, v42
	v_dual_add_f32 v43, 1.0, v43 :: v_dual_add_f32 v44, 1.0, v44
	v_dual_add_f32 v45, 1.0, v45 :: v_dual_add_f32 v46, 1.0, v46
	v_add_f32_e32 v47, 1.0, v47
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v34, v36
	v_rcp_f32_e32 v35, v37
	v_rcp_f32_e32 v36, v48
	v_rcp_f32_e32 v37, v49
	v_rcp_f32_e32 v38, v38
	v_rcp_f32_e32 v39, v39
	v_rcp_f32_e32 v40, v40
	v_rcp_f32_e32 v41, v41
	v_rcp_f32_e32 v42, v42
	v_rcp_f32_e32 v43, v43
	v_rcp_f32_e32 v44, v44
	v_rcp_f32_e32 v45, v45
	v_rcp_f32_e32 v46, v46
	v_rcp_f32_e32 v47, v47
	v_pk_mul_f32 v[30:31], v[32:33], v[30:31]
	v_pk_mul_f32 v[16:17], v[16:17], v[34:35]
	v_pk_mul_f32 v[0:1], v[0:1], v[36:37]
	v_pk_mul_f32 v[2:3], v[2:3], v[38:39]
	v_pk_mul_f32 v[4:5], v[4:5], v[40:41]
	v_pk_mul_f32 v[6:7], v[6:7], v[42:43]
	v_pk_mul_f32 v[8:9], v[8:9], v[44:45]
	v_pk_mul_f32 v[10:11], v[10:11], v[46:47]
	s_mul_u64 s[2:3], s[34:35], s[0:1]
	v_pk_mul_f32 v[14:15], v[14:15], v[16:17]
	v_pk_mul_f32 v[12:13], v[12:13], v[30:31]
	v_pk_mul_f32 v[2:3], v[20:21], v[2:3]
	v_pk_mul_f32 v[0:1], v[18:19], v[0:1]
	v_pk_mul_f32 v[6:7], v[24:25], v[6:7]
	v_pk_mul_f32 v[4:5], v[22:23], v[4:5]
	v_pk_mul_f32 v[10:11], v[28:29], v[10:11]
	v_pk_mul_f32 v[8:9], v[26:27], v[8:9]
	s_lshl_b64 s[2:3], s[2:3], 1
	s_mul_u64 s[4:5], s[22:23], s[0:1]
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[26:27], s[2:3]
	v_cvt_pk_bf16_f32 v15, v14, v15
	s_add_nc_u64 s[2:3], s[2:3], s[56:57]
	v_cvt_pk_bf16_f32 v14, v12, v13
	v_cvt_pk_bf16_f32 v3, v2, v3
	v_cvt_pk_bf16_f32 v2, v0, v1
	v_cvt_pk_bf16_f32 v1, v6, v7
	v_cvt_pk_bf16_f32 v0, v4, v5
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_cvt_pk_bf16_f32 v4, v8, v9
	s_add_nc_u64 s[26:27], s[4:5], s[2:3]
	s_and_b32 s22, s1, 0xffff
	s_bitset1_b32 s27, 31
	s_wait_alu depctr_va_vdst(0)
	ds_store_b64 v64, v[14:15]
	ds_store_b64 v65, v[2:3]
	ds_store_b64 v66, v[0:1] offset:32
	ds_store_b64 v64, v[4:5] offset:48
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_store_from_lds s[24:27], s[16:23]
	s_wait_tensorcnt 0x0
.LBB0_4:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 176
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 103
		.amdhsa_next_free_sgpr 88
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 53
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1, .Lfunc_end0-gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1

	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.num_vgpr, 103
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.num_agpr, 0
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.numbered_sgpr, 88
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.num_named_barrier, 0
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.private_seg_size, 0
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.uses_vcc, 1
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.uses_flat_scratch, 0
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.has_dyn_sized_stack, 0
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.has_recursion, 0
	.set gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     global_buffer
      - .offset:         136
        .size:           28
        .value_kind:     by_value
      - .offset:         164
        .size:           4
        .value_kind:     by_value
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .offset:         172
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 176
    .max_flat_workgroup_size: 128
    .name:           gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     90
    .sgpr_spill_count: 0
    .symbol:         gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     103
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
