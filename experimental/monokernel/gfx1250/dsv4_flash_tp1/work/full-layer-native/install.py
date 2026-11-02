from pathlib import Path
import shutil
root=Path('/work/full-layer-native');p=Path('/app/ATOM/atom/models/deepseek_v4.py')
backup=root/'deepseek_v4.before_full_layer.py'
if not backup.exists():shutil.copy2(p,backup)
s=p.read_text()
if '_experimental_full_layer.bind(self)' in s:raise SystemExit('Full-layer integration already installed')
needle='            fn.data = aiter.mhc_shuffle_fn(fn.data.contiguous().float())\n'
patch='''
        # Experimental C1 uncompressed full-layer kernel; native remains default.
        if (
            self.layer_id == 0
            and os.environ.get("ATOM_DSV4_FULL_LAYER_EXPERIMENTAL", "0") == "1"
            and get_gfx_runtime() == "gfx1250"
            and self.attn_norm.weight.numel() == 4096
            and self.hc_mult == 4
            and self.attn.n_local_heads == 64
            and self.attn.head_dim == 512
            and self.attn.compress_ratio == 0
            and self.norm_eps == 1e-6
        ):
            from atom.model_ops.dsv4_full_layer import prepare
            self._experimental_full_layer = prepare()
            self._experimental_full_layer.bind(self)
'''
assert s.count(needle)==1
s=s.replace(needle,needle+patch)
needle='''        # ----- Attention sub-layer with mHC mixing -----
'''
patch='''        full_layer = getattr(self, "_experimental_full_layer", None)
        if full_layer is not None and hc_state.x_prev is None and hc_state.res_preshuffle:
            return full_layer(self, hc_state, positions)
'''
assert s.count(needle)==1;s=s.replace(needle,patch+needle);p.write_text(s)
shim='''"""Experimental gfx1250 Flash layer-0 monokernel integration (default off)."""
import os
import sys
from pathlib import Path

def prepare():
    root = Path(os.environ.get("ATOM_DSV4_FULL_LAYER_ROOT", "/work/full-layer-native"))
    if not (root / "dsv4_full_layer_runtime.py").is_file():
        raise RuntimeError(f"Missing full-layer experimental artifacts: {root}")
    sys.path.insert(0, str(root))
    from dsv4_full_layer_runtime import ServingLayer
    return ServingLayer()
'''
Path('/app/ATOM/atom/model_ops/dsv4_full_layer.py').write_text(shim)
(root/'dsv4_full_layer.py').write_text(shim)

shutil.copy2(root/"run_dsv4_flash_full_layer_c1.sh","/app/scripts/run_dsv4_flash_full_layer_c1.sh")
Path("/app/scripts/run_dsv4_flash_full_layer_c1.sh").chmod(0o755)
