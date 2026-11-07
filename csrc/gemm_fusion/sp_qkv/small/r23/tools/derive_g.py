import sys

p = sys.argv[1] + "/tests/check_r23_static.py"
with open(p, encoding="utf-8") as _f:
    L = _f.read().split("\n")
out = []
for l in L:
    if l.startswith('need("constexpr bool rxord'):
        pass
    if "C-2: rx_view [2][RX_SLOTS]" in l:
        l = 'need("opus::make_tuple(2, kRxS, kMP, kF)" in lay and "SM_RX_BYTES == 2ull * SM_RX_SLOTS * SM_MP * SM_F * 2ull" in lay, "C-2: rx_view [2][RX_SLOTS][MP][F] and SM_RX_BYTES follow the slot count (MP256 values asserted below)")'
    elif "GEMM BM128 x BN128 / PFK4" in l:
        l = 'need("BN = static_cast<int>(SM_GEMM_BN)" in tr and "WG_PER_CU = SM_GEMM_BN128 ? 1 : 2" in tr and "B_INSTR_PER_WAVE = BN / 16" in tr and "EN == BN / 16" in tr, "GEMM BN128 traits (B_INSTR_PER_WAVE = BN/16, 1 WG per CU)")'
    elif "GEMM: refusal at SM_MP, AQ 1..8 dispatch" in l:
        l = 'need("args.M == 0u || args.M > SM_MP" in g and "default: smg::gemm_body<16>" in g and "SM_STAMP(rec, 5)" in g, "GEMM: refusal at SM_MP, AQ 1..16 dispatch, stamps")'
    out.append(l)
with open(p, "w", encoding="utf-8", newline="") as _f:
    _f.write("\n".join(out))
print("ok")
