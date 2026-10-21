import sys,re
r=sys.argv[1]
def rd(p):
    t=open(p,encoding='utf-8',newline='').read(); nl='\r\n' if '\r\n' in t else '\n'; return t.replace('\r\n','\n'),nl
def wr(p,t,nl): open(p,'w',encoding='utf-8',newline='').write(t.replace('\n',nl))
def rep(t,old,new,cnt=1):
    assert t.count(old)==cnt,('count %d != %d: '%(t.count(old),cnt))+old[:110]
    return t.replace(old,new)
# ---------------- traits ----------------
p=r+'/device/layout/sk_traits.hpp'; t,nl=rd(p)
t=rep(t,'constexpr int BM = 32, BN = 64, BK = 128;','constexpr int BM = static_cast<int>(SM_MP), BN = 64, BK = 128;      // r20: BM64 (BM == SM_MP): 4 row groups of 16, i.e. MT = BM/32 = 2 accumulator tiles per consumer wave')
t=rep(t,'constexpr int EM = BM / (WM * TM), EN = BN / (WN * TN), EK = BK / (WK * TK);   // 1,4,1','constexpr int EM = 1, EN = BN / (WN * TN), EK = BK / (WK * TK);   // the r18 tiled MMA is unchanged (1,4,1): one 16x16x128 MFMA per 32-row tile and K block\nconstexpr int MT = BM / 32;                          // 32-row accumulator tiles per consumer wave (2 at BM64): wave cw owns rows t*32 + cw*16 .. +15 of tile t')
t=rep(t,'constexpr int A_STAGE_BYTES = BM * BK;               // 4096  row-major [32][128]','constexpr int A_STAGE_BYTES = BM * BK;               // 8192  row-major [64][128]')
t=rep(t,'constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;   // 12288','constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;   // 16384')
t=rep(t,'constexpr int PFK = 6;                               // ring depth; PFK-1 = 5 K-blocks in flight per WG','constexpr int PFK = 4;                               // r20 ring depth (R13 card: PFK4 = 64 KiB; PFK5 = 80 KiB is exactly at the 2 WG/CU limit, A/B only); PFK-1 = 3 K-blocks in flight per WG')
t=rep(t,'constexpr int LDS_BYTES = PFK * STAGE_BYTES;         // 73728; x WG_PER_CU=2 = 147456 <= 163840','constexpr int LDS_BYTES = PFK * STAGE_BYTES;         // 65536; x WG_PER_CU=2 = 131072 <= 163840')
t=rep(t,'static_assert(EM == 1 && EN == 4 && EK == 1);','static_assert(EM == 1 && EN == 4 && EK == 1 && BM % 32 == 0 && BM >= 32);')
t=rep(t,'static_assert(PFK >= 3);','static_assert(PFK >= 3 && PFK - 2 <= 4, "wait_newer handles newer in [0,4]");')
t=rep(t,'constexpr int A_INSTR_M32 = 2, A_INSTR_M16 = 1;      // per producer wave: 1 KiB (8 rows x 128 B) each; total A rows 32 / 16','constexpr int AQ_MAX = BM / 16;                      // A instructions per producer wave at full BM (1 KiB = 8 rows x 128 B each, 2 producer waves): 4 at BM64')
t=rep(t,'OPUS_D auto a_frag_view() { return opus::make_layout<-1>(opus::make_tuple(32, 2, 4), opus::make_tuple(BK, 64, 16)); }','OPUS_D auto a_frag_view() { return opus::make_layout<-1>(opus::make_tuple(BM, 2, 4), opus::make_tuple(BK, 64, 16)); }')
wr(p,t,nl)
# ---------------- GEMM ----------------
p=r+'/device/sk_gemm_fp8.cu'; t,nl=rd(p)
t=rep(t,'// vmcnt(newer * MB), newer in [0, PFK-2]; PFK-2 == 4\ntemplate<int MB> OPUS_D void wait_newer(int newer) {','// vmcnt(newer * MB), newer in [0, PFK-2] (r20: PFK 4 -> newer <= 2; the default case is the 4 * MB maximum)\ntemplate<int MB> OPUS_D void wait_newer(int newer) {')
t=rep(t,'static_assert(sks::PFK == 6, "wait_newer is written for PFK-2 == 4");','static_assert(sks::PFK >= 3 && sks::PFK - 2 <= 4, "wait_newer handles newer in [0, 4]");')
t=rep(t,'    constexpr int MB = 4 + AQ;                         // vmem instructions per producer wave per K block','    constexpr int MB = 4 + AQ;                         // vmem instructions per producer wave per K block (AQ = ceil(M/16) <= 4 at BM64)\n    constexpr int MT = (AQ + 1) / 2;                   // 32-row accumulator tiles per consumer wave that can hold live rows')
t=rep(t,'    typename MMA::vtype_c accum{0};','    typename MMA::vtype_c accum[MT];\n    #pragma unroll\n    for (int t = 0; t < MT; ++t) accum[t] = typename MMA::vtype_c{0};')
i=t.index('    } else {\n        const int cw = wave - sks::PRODUCER_WAVES;'); j=t.index('    if (tid == 0) SM_STAMP(rec, 4);')
new='''    } else {
        const int cw = wave - sks::PRODUCER_WAVES;                                // 0: rows 0-15 of every 32-row tile, 1: rows 16-31
        const auto wc = sks::consumer_wave_coord(cw);
        const auto lc = sks::lane_coord(base.p_shape_c(), lane);
        const auto pc = make_tuple(get<0>(wc), get<0>(lc), get<1>(wc), get<1>(lc));
        const auto u_cm = partition_layout_c(mma, make_tuple(1_I, 0_I), pc);
        const auto u_cn = partition_layout_c(mma, make_tuple(0_I, 1_I), pc);
        const auto u_acc = sks::accumulator_register_view(mma);
        // LDS fragment offsets (bytes inside a stage). A/B lane decode: l16 = lane%16 (row / n), g = lane/16 (K group 0..3).
        // r20: tile t (rows t*32 + cw*16 .. +15) is LIVE iff its lowest 16-row group index 2*t + cw < AQ (AQ = ceil(M/16)); the r18 `live_wave` is the AQ <= 2 special case. A dead tile does no LDS read and no MFMA (its outputs are masked).
        const int l16 = lane & 15, g = lane >> 4;
        const auto u_af = sks::a_frag_view();                                     // (row, seg, g), BM rows
        const auto u_bf = sks::b_frag_view();                                     // (nb, seg, g/2, g%2, n16)
        int a_off0[MT], a_off1[MT];
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int a_row = t * 32 + cw * 16 + l16;
            a_off0[t] = static_cast<int>(u_af(a_row, 0, g)); a_off1[t] = static_cast<int>(u_af(a_row, 1, g));
        }
        int b_off0[4], b_off1[4];
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            b_off0[nb] = sks::A_STAGE_BYTES + static_cast<int>(u_bf(nb, 0, g >> 1, g & 1, l16));
            b_off1[nb] = sks::A_STAGE_BYTES + static_cast<int>(u_bf(nb, 1, g >> 1, g & 1, l16));
        }
        auto lds = smem<fp8_t>(stage);
        for (int i = 0; i < L; ++i) {
            bar();                                                                // R_i
            if (i == 0 && tid == 128) SM_STAMP(rec, 5);
            if (cw < AQ) {                                                        // tile 0 of this wave is live (2*0 + cw < AQ); higher tiles are tested below
                const int sb = (i % sks::PFK) * sks::STAGE_BYTES;
                typename MMA::vtype_a va[MT]; typename MMA::vtype_b vb;
                #pragma unroll
                for (int t = 0; t < MT; ++t) {
                    if (2 * t + cw < AQ) {
                        auto s0 = lds.template load<16>(sb + a_off0[t]); auto s1 = lds.template load<16>(sb + a_off1[t]);
                        U32x8 tt; __builtin_memcpy(&tt.w[0], &s0, 16); __builtin_memcpy(&tt.w[4], &s1, 16); __builtin_memcpy(&va[t], &tt, 32);
                    }
                }
                {
                    U32x8 tt[4];
                    #pragma unroll
                    for (int nb = 0; nb < 4; ++nb) {
                        auto s0 = lds.template load<16>(sb + b_off0[nb]); auto s1 = lds.template load<16>(sb + b_off1[nb]);
                        __builtin_memcpy(&tt[nb].w[0], &s0, 16); __builtin_memcpy(&tt[nb].w[4], &s1, 16);
                    }
                    __builtin_memcpy(&vb, &tt[0], 128);
                }
                s_waitcnt_lgkmcnt(number<0>{});
                #pragma unroll
                for (int t = 0; t < MT; ++t)
                    if (2 * t + cw < AQ) accum[t] = mma(va[t], vb, accum[t], 127, 127, number<0>{}, number<0>{});
            }
        }
        if (tid == 128) SM_STAMP(rec, 6);
        const auto u_p = sks::partial_view(N);
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            static_ford(mma.y_shape_c(), [&](auto im, auto rept, auto item, auto jn) {
                const auto y = make_tuple(im, rept, item, jn);
                const int m = t * 32 + u_cm(y), n = col_base + u_cn(y);
                if (m < M && n < N) {                                            // only live rows are stored
                    const int acc_idx = static_cast<int>(u_acc(y));
#if SM_NT_STORES
                    __builtin_nontemporal_store(accum[t][acc_idx], &args.P[u_p(split, m, n)]);          // r15 Q2: non-temporal partial
#else
                    args.P[u_p(split, m, n)] = accum[t][acc_idx];                // raw FP32 partial; no scale, no BF16
#endif
                }
            });
        }
        s_waitcnt_vmcnt(number<0>{});
        if (tid == 128) SM_STAMP(rec, 3);
    }
'''
t=t[:i]+new+t[j:]
# wrapper: refusal and dispatch
t=rep(t,'args.M == 0u || args.M > 32u) return;','args.M == 0u || args.M > SM_MP) return;')
t=rep(t,'    if (args.M <= 16u) smg::gemm_body<1>(args, stage);\n    else               smg::gemm_body<2>(args, stage);','''    switch ((args.M + 15u) / 16u) {                                                       // WG-uniform: AQ = ceil(M/16) in 1..4 (M <= SM_MP = 64)
        case 1u: smg::gemm_body<1>(args, stage); break;
        case 2u: smg::gemm_body<2>(args, stage); break;
        case 3u: smg::gemm_body<3>(args, stage); break;
        default: smg::gemm_body<4>(args, stage); break;
    }''')
wr(p,t,nl); print('gemm ok')