from pathlib import Path
root=Path(__file__).parent
out=root/'candidate';out.mkdir(exist_ok=True)
s=(root/'originals/mhc_kernel.py').read_text()
s=s.replace('"FUSE_RMSNORM", "RES_SHUFFLED"],','"FUSE_RMSNORM", "RES_SHUFFLED", "FUSE_QUANT"],')
s=s.replace('    RES_KS: gl.constexpr\n','    RES_KS: gl.constexpr\n    FUSE_QUANT: gl.constexpr\n',1)
s=s.replace('        RES_KS=32,\n','        RES_KS=32,\n        FUSE_QUANT=False,\n',1)
s=s.replace('        self.RES_KS = gl.constexpr(RES_KS)','        self.RES_KS = gl.constexpr(RES_KS)\n        self.FUSE_QUANT = gl.constexpr(FUSE_QUANT)',1)
start=s.index('def _reduce_apply_main(');end=s.index('\n\n@gluon.jit\ndef _post_sinkhorn_ws',start)
part=s[start:end]
part=part.replace('    norm_w_ptr,  # (C,) bf16, read only with FUSE_RMSNORM\n','    norm_w_ptr,  # (C,) bf16, read only with FUSE_RMSNORM\n    quant_ptr,\n    quant_scale_ptr,\n',1)
needle='            gl.store(li_ptr + li_rows * C + cc + li_c, out.to(dt), mask=ok)'
replacement=needle+'''
            if K.FUSE_QUANT:
                # Match HIP per-128 RoundUp scales after the BF16 rounding point.
                v = gl.reshape(gl.where(ok, out.to(dt).to(gl.float32), 0.0),
                               [ROWS, CHUNK // 128, 128])
                amax = gl.max(gl.abs(v), 2)
                bits = (amax * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
                exponent = gl.minimum(((bits & 0x7fffffff) + 0x7fffff) >> 23, 254)
                scales = gl.broadcast_to(exponent[:, :, None], [ROWS, CHUNK // 128, 4])
                scales = gl.reshape(scales, [ROWS, CHUNK // 32]).to(gl.uint8)
                q = gl.amd.cdna5.scaled_downcast(
                    gl.reshape(v, [ROWS, CHUNK]), scales, "e4m3", axis=1)
                q = gl.reshape(q, [ROWS, CHUNK // 32, 32])
                gl.store(quant_ptr + li_rows * C + cc + li_c, q, mask=ok)
                EL: gl.constexpr = exponent.type.layout
                qr = row0 + gl.arange(0, ROWS, layout=gl.SliceLayout(1, EL))
                qg = cc // 128 + gl.arange(0, CHUNK // 128, layout=gl.SliceLayout(0, EL))
                gl.store(quant_scale_ptr + qg[None, :] * M + qr[:, None],
                         exponent.to(gl.uint8), (qr[:, None] < M) & (qg[None, :] < C // 128))'''
assert needle in part;part=part.replace(needle,replacement,1);s=s[:start]+part+s[end:]
start=s.index('def _mhc_pre_reduce_apply_gfx1250_kernel(');part=s[start:]
part=part.replace('    RES_KS: gl.constexpr = 32,\n','    RES_KS: gl.constexpr = 32,\n    quant_ptr=None,\n    quant_scale_ptr=None,\n    FUSE_QUANT: gl.constexpr = False,\n',1)
part=part.replace('        norm_w_ptr,\n','        norm_w_ptr,\n        quant_ptr if FUSE_QUANT else li_ptr,\n        quant_scale_ptr if FUSE_QUANT else li_ptr,\n',1)
part=part.replace('        RES_KS,\n','        RES_KS,\n        FUSE_QUANT,\n',1)
s=s[:start]+part;(out/'mhc_kernel.py').write_text(s)
s=(root/'originals/mhc_wrapper.py').read_text()
start=s.index('def _mhc_post_pre_gluon_gfx1250(');end=s.index('\n\ndef mhc_post_pre(',start)
part=s[start:end]
part=part.replace('    residual_out: torch.Tensor | None = None,\n','    residual_out: torch.Tensor | None = None,\n    quant_out: torch.Tensor | None = None,\n    quant_scale: torch.Tensor | None = None,\n',1)
part=part.replace('        FUSE_RMSNORM=fuse_norm,\n','        FUSE_RMSNORM=fuse_norm,\n        quant_ptr=quant_out,\n        quant_scale_ptr=quant_scale,\n        FUSE_QUANT=quant_out is not None,\n',1)
s=s[:start]+part+s[end:]
s+='''

def mhc_post_pre_quant_gfx1250(
    layer_input, residual_in, post_mix, comb_mix, fn, alphas, bias, norm_weight,
    eps=1e-6, hc_pre_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_iters=20,
    hc_sinkhorn_eps=1e-6, norm_eps=1e-6, res_preshuffle=True,
):
    """Run mHC post/pre and RMSNorm, and quantize its BF16 result for an FP8 GEMM.

    Inputs follow mhc_post_pre, except fn uses the (24, 4*C) AITER layout.
    Return its four outputs plus FP8 data and per-128 E8M0 scales. The scale
    tensor has shape (M, C/128), with column-major bytes as in per_group_quant_hip.
    Requires gfx1250, four HC streams, a supported hidden size, and RMSNorm.
    """
    if get_arch() != "gfx1250" or norm_weight is None:
        raise ValueError("mHC quant requires gfx1250 and a norm weight")
    if not mhc_post_pre_gfx1250_supported(
        layer_input, residual_in, fn.T, alphas, bias, 4, sinkhorn_iters,
        True, norm_weight, fn.dtype == torch.int32,
    ):
        raise ValueError("unsupported mHC quant inputs")
    m, c = layer_input.shape
    q = torch.empty((m, c), device=layer_input.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, c // 128), device=layer_input.device, dtype=torch.float8_e8m0fnu)
    result = _mhc_post_pre_gluon_gfx1250(
        layer_input, residual_in, post_mix, comb_mix, fn.T, alphas, bias, 4,
        eps, hc_pre_eps, hc_post_mult_value, sinkhorn_iters, hc_sinkhorn_eps,
        norm_weight, norm_eps, res_preshuffle,
        quant_out=q, quant_scale=scales.view(torch.uint8),
    )
    return (*result, q, scales)
'''
(out/'mhc_wrapper.py').write_text(s)
