from pathlib import Path
p=Path('/app/ATOM/atom/models/deepseek_v4.py');s=p.read_text();root=Path('/work/prefill-dispatch-fix')
if 'ATOM_DSV4_DECODE_ONLY_DISPATCH' not in s:
 root.joinpath('deepseek_v4.before.py').write_text(s)
 old='''        full_layer = getattr(self, "_experimental_full_layer", None)
        if full_layer is not None'''
 new='''        full_layer = getattr(self, "_experimental_full_layer", None)
        # Decode-only integration enters outside the compiled backbone. Keep
        # every backbone execution (including tiny/chunked prefill) native.
        if os.environ.get("ATOM_DSV4_DECODE_ONLY_DISPATCH", "0") == "1":
            full_layer = None
        if full_layer is not None'''
 assert s.count(old)==1;s=s.replace(old,new,1)
 old='''        h = self.model(input_ids, positions)

        # ----- PCP: all-gather'''
 new='''        # Select the decode implementation outside torch.compile. In
        # particular, token count alone must never identify a tiny prefill.
        md = ctx.attn_metadata
        decode_only = (
            os.environ.get("ATOM_DSV4_DECODE_ONLY_DISPATCH", "0") == "1"
            and not ctx.context.is_dummy_run
            and not ctx.context.is_prefill
            and md is not None
            and md.state is AttnState.DECODE
            and input_ids.numel() in (1, 2, 4, 8)
            and not use_pcp
            and getattr(self.model.layers[0], "_experimental_full_layer", None) is not None
        )
        if decode_only:
            from decode_forward import decode_forward
            h = decode_forward(self.model, input_ids, positions, md)
        else:
            h = self.model(input_ids, positions)

        # ----- PCP: all-gather'''
 assert s.count(old)==1;s=s.replace(old,new,1);p.write_text(s)
root.joinpath('deepseek_v4.after.py').write_text(p.read_text())
print('Installed opt-in decode dispatch outside compiled backbone')
