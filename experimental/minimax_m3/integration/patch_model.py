#!/usr/bin/env python3
"""Patch vLLM's MiniMax-M3 AMD model to call the mono hook (idempotent).

Keeps a pristine copy as model.py.orig next to it; re-running restores from it first.
Edits: MiniMaxM3Model.__init__ keeps vllm_config; its forward's decoder loop goes
through ``mono.hook.run_layers`` (which runs vLLM's own layers when mono is off).
"""
import pathlib
import shutil
import sys

import vllm

f = pathlib.Path(vllm.__file__).parent / "models/minimax_m3/amd/model.py"
orig = f.with_suffix(".py.orig")
if not orig.exists():
    shutil.copy(f, orig)
s = orig.read_text()

old_init = '''class MiniMaxM3Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
'''
new_init = old_init + "        self._m3_vllm_config = vllm_config  # mono hook\n"
assert s.count(old_init) == 1, "init anchor"
s = s.replace(old_init, new_init)

old_loop = '''        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )
'''
new_loop = '''        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        # MiniMax-M3 mono decode (vllm.models.minimax_m3.amd.mono.hook); EAGLE3's aux
        # hidden states collected on either path
        from vllm.models.minimax_m3.amd.mono.hook import run_layers

        hidden_states, residual = run_layers(
            self, positions, hidden_states, residual, self._m3_vllm_config,
            aux_hidden_states if getattr(self, "aux_hidden_state_layers", None) else None,
        )
'''
assert s.count(old_loop) == 1, "loop anchor"
s = s.replace(old_loop, new_loop)
f.write_text(s)
print("patched", f, file=sys.stderr)
