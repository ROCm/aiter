"""Apply confirmed TP8 CSV gains and export a full stage-throughput table."""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import analyze
import tune


ARTIFACTS = Path(__file__).resolve().parent


def median_stages(samples):
    """Aggregate native stage timings without dropping positive outliers."""
    values = {}
    for stage in ('gemm1', 'gemm2'):
        times = [sample[stage+'_us'] for sample in samples]
        assert all(value is not None and math.isfinite(value) and value > 0
                   for value in times), times
        values[stage] = statistics.median(times)
    return values


def work_function():
    """Read the benchmark's CPU-pure logical FLOP/byte model."""
    source = tune.ROOT / 'op_tests/flydsl_tests/test_flydsl_grouped_gemm.py'
    node = next(item for item in ast.parse(source.read_text()).body
                if isinstance(item, ast.FunctionDef)
                and item.name == '_gemm_work_metrics')
    namespace = {'SCALE_BLOCK': 32}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'),
         namespace)
    return namespace['_gemm_work_metrics']


def main():
    """Retain only confirmed improvements and show all original CSV cases."""
    fields, baseline = tune.read_csv(ARTIFACTS / 'main.csv')
    _, selected = tune.read_csv(ARTIFACTS / 'selected.csv')
    selections = json.loads((ARTIFACTS / 'selection.json').read_text())
    runs = {config: [analyze.results(f'finalverify_r{repeat}_{config}')
                     for repeat in (1, 2, 3)]
            for config in ('main', 'selected')}
    work_metrics = work_function()
    decisions = []
    rows = []
    final = []
    for old, candidate, selection in zip(baseline, selected, selections):
        token = int(old['token'])
        samples = {config: [run[token] for run in runs[config]]
                   for config in runs}
        medians = {config: median_stages(values)
                   for config, values in samples.items()}
        speedup = (sum(medians['main'].values()) /
                   sum(medians['selected'].values()) - 1) * 100
        pairs = [(sum(samples['main'][index][stage+'_us']
                      for stage in ('gemm1', 'gemm2')) /
                  sum(samples['selected'][index][stage+'_us']
                      for stage in ('gemm1', 'gemm2')) - 1) * 100
                 for index in range(3)]
        changed = selection['selected'] != 'main'
        retained = changed and speedup >= 1 and all(gain > 0 for gain in pairs)
        decision = {'token': token, 'candidate': selection['selected'],
                    'retained': retained, 'paired_gain_pct': pairs,
                    'median_gain_pct': speedup,
                    'main_samples': samples['main'],
                    'selected_samples': samples['selected']}
        decisions.append(decision)
        final.append(candidate.copy() if retained else old.copy())
        if not changed:
            common = median_stages(samples['main'] + samples['selected'])
            medians = {'main': common, 'selected': common}
        elif not retained:
            medians['selected'] = medians['main']
        work = work_metrics(experts=896, tokens=token, topk=16,
                            model_dim=3584, inter_dim=384, data_format='a4w4')
        row = {'tokens': token, 'tile_m': int(old['tile_m']),
               'changed': retained, 'tile_n1': int(final[-1]['tile_n']),
               'tile_n2': int(final[-1]['tile_n2'])}
        for stage in ('gemm1', 'gemm2'):
            for config in ('main', 'selected'):
                elapsed = medians[config][stage]
                row[f'{stage}_{config}_tflops'] = work[stage][0]/elapsed/1e6
                row[f'{stage}_{config}_bw_tb_s'] = work[stage][1]/elapsed/1e6
            row[f'{stage}_gain_pct'] = (
                medians['main'][stage]/medians['selected'][stage]-1)*100
        row['combined_gain_pct'] = (
            sum(medians['main'].values()) /
            sum(medians['selected'].values())-1)*100
        rows.append(row)
    (ARTIFACTS / 'verification.json').write_text(
        json.dumps(decisions, indent=2)+'\n')
    tune.write_csv(ARTIFACTS / 'final.csv', fields, final)
    # Preserve unchanged lines and existing trailing whitespace in production.
    original_lines = (ARTIFACTS / 'main.csv').read_text().splitlines(keepends=True)
    final_lines = (ARTIFACTS / 'final.csv').read_text().splitlines()
    for index, (old, new) in enumerate(zip(baseline, final), 1):
        if old != new:
            original_lines[index] = final_lines[index]+'\n'
    tune.CONFIG.write_text(''.join(original_lines))
    (ARTIFACTS / 'final.csv').write_bytes(tune.CONFIG.read_bytes())
    (ARTIFACTS / 'final.sha256').write_text(
        hashlib.sha256(tune.CONFIG.read_bytes()).hexdigest()+'\n')
    with (ARTIFACTS / 'PERFORMANCE.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Kimi3 TP8: copied non-EP schedules with N256/N512 comparison', '',
        'Branch: dev/kimi3_tp8. Main baseline: '
        '956fd46d879582fec765f614429638b21c6200cc.', '',
        'A4W4, hidden=3584, intermediate=384, experts=896, topk=16, '
        'SiTUv2 (beta=4, linear_beta=25), gfx1250, FlyDSL 0.5.0. '
        'Uniform random inputs and scales, balanced routing, seed=0. '
        'Native isolated GEMM timings, 5 warmups, 101 timing iterations. '
        'All correctness checks passed.', '',
        'Copy DSV4.1 Flash non-EP expert96/topk6 schedules with matching '
        'tile_m and tile_m2, choosing the closest routed rows per expert. '
        'Keep all copied scheduling fields fixed. GEMM1 uses N256 because '
        'its N=768 does not admit N512; GEMM2 compares N256/N512. '
        'main.csv, donor.csv, mapping.json and n256/n512.csv preserve inputs.', '',
        'Three fresh paired verification runs, alternating execution order. '
        'Changed rows use independent stage medians over those three runs. '
        'Unchanged rows pool all six identical-configuration samples, shown '
        'equally in both columns to avoid reporting noise as a code gain. '
        'Retain a row only when combined stage throughput improves at least '
        '1% and all three paired comparisons improve. Raw samples and '
        'decisions are in verification.json; screening results are separate.', '',
        'The token=8192 copied N512 configuration is rejected despite '
        'its initial speedup: it produced GPU memory-access faults in '
        'verify_r2_selected.log and fault_check_selected.log. The original '
        'main row passed fault_check_main.log and is retained. The tuning '
        'does not change kernels or attempt to repair this separate issue.', '',
        'Each rate cell is **TFLOPS / effective BW (TB/s), per GPU**. '
        'BW uses the native benchmark logical byte model, including active '
        'expert weights/scales and stage inputs/outputs. It is not measured '
        'physical HBM traffic. Timings exclude TP communication, routing, '
        'auxiliary kernels and end-to-end MoE overhead.', '',
        '| Tokens | G1 main | G1 tuned | G1 gain | G2 main | G2 tuned | G2 gain | G1+G2 gain |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        cells = [str(row['tokens'])]
        for stage in ('gemm1', 'gemm2'):
            for config in ('main', 'selected'):
                cells.append(f"{row[f'{stage}_{config}_tflops']:.1f} / "
                             f"{row[f'{stage}_{config}_bw_tb_s']:.2f}")
            cells.append(f"{row[f'{stage}_gain_pct']:+.2f}%"
                         if row['changed'] else 'unchanged')
        cells.append(f"{row['combined_gain_pct']:+.2f}%"
                     if row['changed'] else 'unchanged')
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Retained scheduling changes', '',
              '| Tokens | Field | Main | Selected |',
              '|---:|---|---:|---:|']
    for old, new in zip(baseline, final):
        for field in fields:
            if old[field] != new[field]:
                lines.append(f"| {old['token']} | {field} | "
                             f"{old[field] or '(blank)'} | {new[field]} |")
    (ARTIFACTS / 'PERFORMANCE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    changed_decisions = [row for row in decisions if row['candidate']!='main']
    print('Candidate verification:', json.dumps(changed_decisions, indent=2))


if __name__ == '__main__':
    main()
