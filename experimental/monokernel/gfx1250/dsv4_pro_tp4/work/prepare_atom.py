from pathlib import Path
r=Path(__file__).parent;s=(r/'deepseek_v4.original.py').read_text()
s=s.replace('from atom.model_ops.layernorm import RMSNorm, rmsnorm2d_fwd_', 'from atom.model_ops.dsv4_hybrid import dsv4_hybrid_hc\nfrom atom.model_ops.layernorm import RMSNorm, rmsnorm2d_fwd_',1)
for name in ('_v4_attention_fake','v4_attention_with_output'):
 start=s.index('def '+name+'(');end=s.index('\n\n',start)
 p=s[start:end].replace('    layer_name: str,\n','    layer_name: str,\n    x_quant: torch.Tensor | None = None,\n    x_scale: torch.Tensor | None = None,\n',1)
 p=p.replace('return self.forward_impl(x, positions)','return self.forward_impl(x, positions, x_quant, x_scale)')
 s=s[:start]+p+s[end:]
start=s.index('class DeepseekV4Attention(');end=s.index('\nclass Expert(',start);part=s[start:end]
startf=part.index('    def forward(\n');endf=part.index('    def _forward_piecewise_attention(',startf)
p=part[startf:endf].replace('        positions: torch.Tensor,\n','        positions: torch.Tensor,\n        x_quant: torch.Tensor | None = None,\n        x_scale: torch.Tensor | None = None,\n',1)
p=p.replace('torch.ops.aiter.v4_attention_with_output(x, positions, self.layer_name)','torch.ops.aiter.v4_attention_with_output(x, positions, self.layer_name, x_quant, x_scale)')
part=part[:startf]+p+part[endf:]
part=part.replace('        run_indexer_proj: bool = True,\n','        run_indexer_proj: bool = True,\n        x_quant: torch.Tensor | None = None,\n        x_scale: torch.Tensor | None = None,\n',1)
part=part.replace('        qkv_a = self.wqkv_a(x)','        qkv_a = self.wqkv_a(x) if x_quant is None else self.wqkv_a(x_quant, x_scale=x_scale)',1)
startf=part.index('    def forward_impl(\n');endf=part.index('    def _attn_compress(',startf)
p=part[startf:endf].replace('        positions: torch.Tensor,  # [num_tokens] int  absolute token positions\n','        positions: torch.Tensor,  # [num_tokens] int  absolute token positions\n        x_quant: torch.Tensor | None = None,\n        x_scale: torch.Tensor | None = None,\n',1)
p=p.replace('x, positions, run_indexer_proj=False','x, positions, run_indexer_proj=False, x_quant=x_quant, x_scale=x_scale',1)
part=part[:startf]+p+part[endf:];s=s[:start]+part+s[end:]
needle='''        self.enable_fused_hc = (
            self._mhc_fused_post_pre is not None and self.layer_id != 0
        )'''
assert needle in s
s=s.replace(needle,needle+'''
        cg = get_current_atom_config().compilation_config.cudagraph_mode
        projection = self.attn.wqkv_a
        self.hybrid_hc = (
            os.environ.get("ATOM_DSV4_HYBRID", "0") == "1"
            and get_gfx_runtime() == "gfx1250"
            and args.dim in (4096, 7168) and args.hc_mult == 4
            and not _V4_FORCE_UE8M0_QUANT
            and cg is not None and not cg.requires_piecewise_compilation()
            and projection.params_dtype == torch.float8_e4m3fn
            and projection.blockscale_e8m0_scale
            and projection.native_a8_group_rows is None
            and not projection.use_mxscale_shuffle
            and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        )
        self.hybrid_max_tokens = int(os.environ.get("ATOM_DSV4_HYBRID_MAX_TOKENS", "128"))
        if self.layer_id == 0:
            logger.info("DSV4 hybrid mHC quant enabled=%s max_tokens=%s", self.hybrid_hc, self.hybrid_max_tokens)
''',1)
needle='''        # ----- Attention sub-layer with mHC mixing -----
        hc_state = self.fuse_hc(
            hc_state,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm.weight,
            self.norm_eps,
        )
        x = hc_state.x_prev
        x = self.attn(x, positions)  # [num_tokens, dim]'''
assert needle in s
s=s.replace(needle,'''        # ----- Attention sub-layer with mHC mixing -----
        x_quant = x_scale = None
        if (self.hybrid_hc and hc_state.x_prev is not None
                and 0 < hc_state.x_prev.shape[0] <= self.hybrid_max_tokens):
            post, comb, x, res, x_quant, x_scale = dsv4_hybrid_hc(
                hc_state.x_prev, hc_state.residual, hc_state.post_mix,
                hc_state.comb_mix, self.hc_attn_fn, self.hc_attn_scale,
                self.hc_attn_base, self.attn_norm.weight,
                self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters,
                hc_state.res_preshuffle,
            )
            hc_state = HCState(res, post.squeeze(-1), comb, x, hc_state.res_preshuffle)
        else:
            hc_state = self.fuse_hc(
                hc_state, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                self.attn_norm.weight, self.norm_eps,
            )
        x = hc_state.x_prev
        x = self.attn(x, positions, x_quant, x_scale)  # [num_tokens, dim]''',1)
(r/'candidate/deepseek_v4.py').write_text(s)
