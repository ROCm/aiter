from pathlib import Path
p=Path('/app/ATOM/atom/models/deepseek_v4.py');s=p.read_text();s=s.replace('self._experimental_full_layer = prepare()','self._experimental_full_layer = prepare()\n            self._experimental_full_layer.bind(self)')
s=s.replace('''        if full_layer is not None:
            fc = get_forward_context()
            if full_layer.eligible(self, hc_state, positions, fc):
                return full_layer(self, hc_state, positions, fc)
''','''        if full_layer is not None and hc_state.x_prev is None and hc_state.res_preshuffle:
            return full_layer(self, hc_state, positions)
''');p.write_text(s)
