"""Retry the interrupted final pair while preserving its failed log."""

import json

import tune


_, rows = tune.read_csv(tune.ARTIFACTS / 'main.csv')
tokens = [int(row['token']) for row in rows]
for config in ('main', 'selected'):
    name = f'finalverify_r3_{config}'
    actual_name = name+'_retry1' if config == 'main' else name
    values = tune.run(config, tokens, actual_name)
    if actual_name != name:
        (tune.ARTIFACTS / f'{name}_results.json').write_text(
            json.dumps(values, indent=2)+'\n')
