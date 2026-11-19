"""Select measured N-width candidates and report stage rates against main."""

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import statistics

import tune


ARTIFACTS = Path(__file__).resolve().parent


def results(name):
    """Index a native benchmark result by token count."""
    rows = json.loads((ARTIFACTS / f'{name}_results.json').read_text())
    return {row['token']: row for row in rows}


def screen():
    """Select full rows, preserving main whenever no measurable gain appears."""
    tables = {name: results('screen_'+name)
              for name in ('main', 'n256', 'n512')}
    fields, baseline = tune.read_csv(ARTIFACTS / 'main.csv')
    configs = {name: {int(row['token']): row for row in
                     tune.read_csv(ARTIFACTS / f'{name}.csv')[1]}
               for name in tables}
    selected = []
    decisions = []
    for row in baseline:
        token = int(row['token'])
        for name in tables:
            assert all(tables[name][token][stage+'_us'] is not None
                       and tables[name][token][stage+'_us'] > 0
                       for stage in ('gemm1', 'gemm2')), (name, token)
        total = {name: sum(tables[name][token][stage+'_us']
                           for stage in ('gemm1', 'gemm2')) for name in tables}
        winner = min(total, key=total.get)
        if total[winner] > total['main'] * 0.99:
            winner = 'main'
        selected.append(configs[winner][token].copy())
        decision = {'token': token, 'selected': winner, 'totals_us': total,
                    'screen_speedup_pct': (total['main']/total[winner]-1)*100}
        decisions.append(decision)
        print(token, winner,
              {name: [tables[name][token]['gemm1_us'],
                      tables[name][token]['gemm2_us']] for name in tables},
              f"{decision['screen_speedup_pct']:.2f}%")
    tune.write_csv(ARTIFACTS / 'selected.csv', fields, selected)
    (ARTIFACTS / 'selection.json').write_text(
        json.dumps(decisions, indent=2)+'\n')


def main():
    """Generate a screening selection from complete native timing results."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', action='store_true')
    args = parser.parse_args()
    if args.screen:
        screen()


if __name__ == '__main__':
    main()
