from pathlib import Path
p=Path('/app/ATOM/atom/models/deepseek_v4.py');s=p.read_text();old='''        if self.layer_id == 0 and os.environ.get("ATOM_DSV4_FULL_LAYER_EXPERIMENTAL", "0") == "1":''';new='''        if (
            self.layer_id == 0
            and os.environ.get("ATOM_DSV4_FULL_LAYER_EXPERIMENTAL", "0") == "1"
            and get_gfx_runtime() == "gfx1250"
            and self.attn_norm.weight.numel() == 4096
            and self.hc_mult == 4
            and self.attn.n_local_heads == 64
            and self.attn.head_dim == 512
            and self.attn.compress_ratio == 0
            and self.norm_eps == 1e-6
        ):''';assert old in s;s=s.replace(old,new);p.write_text(s)
p=Path('/work/full-layer-native/install.py');s=p.read_text().replace(old,new);p.write_text(s)
