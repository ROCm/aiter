"""Verify selected rows against current main with alternating native runs."""

import tune


_, rows = tune.read_csv(tune.ARTIFACTS / 'main.csv')
tokens = [int(row['token']) for row in rows]
for repeat in (1, 2, 3):
    order = ('main', 'selected') if repeat != 2 else ('selected', 'main')
    for config in order:
        tune.run(config, tokens, f'finalverify_r{repeat}_{config}')
