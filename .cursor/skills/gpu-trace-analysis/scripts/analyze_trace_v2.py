#!/usr/bin/env python3
"""
Comprehensive GPU kernel trace analysis for ML serving frameworks.
Supports: Atom (DeepSeek MLA), vLLM/SGLang (annotation-based), and similar.
Analyzes prefill vs decode sections, disambiguates kernels, and measures GPU idle time.

Usage: python3 analyze_trace_v2.py <trace_file.json.gz>
"""
import json
import gzip
import sys
import re
import bisect
from collections import defaultdict
from pathlib import Path

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def load_trace(path):
    p = Path(path)
    opener = gzip.open if p.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        return json.load(f)

def fmt_us(us):
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f} s"
    if us >= 1000:
        return f"{us/1000:.2f} ms"
    return f"{us:.1f} us"

# ────────────────────────────────────────────────────────────
# Framework detection
# ────────────────────────────────────────────────────────────
EXEC_PATTERN = re.compile(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)')

def detect_framework(events):
    has_atom_forward = False
    has_exec_annotation = False
    for e in events:
        cat = e.get('cat', '')
        name = e.get('name', '')
        if cat == 'python_function' and 'model_runner' in name and 'forward' in name:
            has_atom_forward = True
        if cat == 'user_annotation' and EXEC_PATTERN.match(name):
            has_exec_annotation = True
        if has_atom_forward or has_exec_annotation:
            break
    if has_exec_annotation:
        return 'annotation'
    if has_atom_forward:
        return 'atom'
    return 'unknown'

# ────────────────────────────────────────────────────────────
# Section identification
# ────────────────────────────────────────────────────────────

def classify_annotation(name):
    m = EXEC_PATTERN.match(name)
    if not m:
        return 'unknown', 0, 0
    n_ctx, ctx_tok, n_gen, gen_tok = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if n_ctx > 0 and n_gen > 0:
        return 'mixed', ctx_tok, gen_tok
    elif n_ctx > 0:
        return 'prefill', ctx_tok, 0
    elif n_gen > 0:
        return 'decode', 0, gen_tok
    return 'unknown', 0, 0

def find_sections_annotation(events):
    """Find sections from user_annotation events with execute_context pattern."""
    sections = []
    for e in events:
        if e.get('cat') != 'user_annotation':
            continue
        name = e.get('name', '')
        stype, pf_tok, dc_tok = classify_annotation(name)
        if stype == 'unknown':
            continue
        sections.append({
            'ts': e['ts'], 'dur': e['dur'], 'end': e['ts'] + e['dur'],
            'type': stype, 'name': name,
            'prefill_tokens': pf_tok, 'decode_tokens': dc_tok,
        })
    sections.sort(key=lambda x: x['ts'])
    for i, s in enumerate(sections):
        s['idx'] = i
    return sections

def find_sections_atom(events):
    """Find sections from Atom framework python_function events."""
    forwards = []
    for e in events:
        if (e.get('cat') == 'python_function' and
                'model_runner' in e.get('name', '') and
                e.get('name', '').endswith(': forward')):
            forwards.append({'ts': e['ts'], 'dur': e['dur'], 'end': e['ts'] + e['dur']})
    forwards.sort(key=lambda x: x['ts'])

    prefill_ts = []
    decode_ts = []
    for e in events:
        if e.get('cat') != 'python_function':
            continue
        name = e.get('name', '')
        if '_forward_prefill' in name:
            prefill_ts.append(e['ts'])
        if 'prepare_decode' in name:
            decode_ts.append(e['ts'])

    sections = []
    for i, f in enumerate(forwards):
        has_pf = any(f['ts'] <= t <= f['end'] for t in prefill_ts)
        has_dc = any(f['ts'] <= t <= f['end'] for t in decode_ts)
        if has_pf and has_dc:
            stype = 'mixed'
        elif has_pf:
            stype = 'prefill'
        elif has_dc:
            stype = 'decode'
        else:
            stype = 'unknown'
        f['type'] = stype
        f['idx'] = i
        f['name'] = f'Forward_{i}'
        f['prefill_tokens'] = 0
        f['decode_tokens'] = 0
        sections.append(f)
    return sections

# ────────────────────────────────────────────────────────────
# Kernel-to-section mapping
# ────────────────────────────────────────────────────────────

def map_kernels_annotation(events, sections):
    """Map kernels to sections using gpu_user_annotation GPU-side time ranges."""
    gpu_sections = []
    for e in events:
        if e.get('cat') != 'gpu_user_annotation':
            continue
        stype, pf_tok, dc_tok = classify_annotation(e.get('name', ''))
        if stype == 'unknown':
            continue
        gpu_sections.append({
            'ts': e['ts'], 'end': e['ts'] + e['dur'], 'name': e['name'],
            'type': stype,
        })
    gpu_sections.sort(key=lambda x: x['ts'])

    # Match gpu_sections to cpu sections by name + order
    # Build a fast lookup: for each kernel ts, binary search into gpu_sections
    gpu_starts = [g['ts'] for g in gpu_sections]

    gpu_kernels = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]

    # Also match gpu_sections to cpu sections by matching order of same-name annotations
    # We match gpu_section[i] to section[j] where names match in order
    cpu_by_name = defaultdict(list)
    for s in sections:
        cpu_by_name[s['name']].append(s)
    gpu_to_cpu = {}
    gpu_name_idx = defaultdict(int)
    for gi, gs in enumerate(gpu_sections):
        name = gs['name']
        idx = gpu_name_idx[name]
        if idx < len(cpu_by_name[name]):
            gpu_to_cpu[gi] = cpu_by_name[name][idx]['idx']
        gpu_name_idx[name] += 1

    fwd_kernels = defaultdict(list)
    unassigned = []
    for k in gpu_kernels:
        kts = k['ts']
        # Binary search for containing gpu_section
        idx = bisect.bisect_right(gpu_starts, kts) - 1
        assigned = False
        for di in (0, -1, 1):
            ci = idx + di
            if 0 <= ci < len(gpu_sections):
                gs = gpu_sections[ci]
                if gs['ts'] <= kts <= gs['end']:
                    if ci in gpu_to_cpu:
                        fwd_kernels[gpu_to_cpu[ci]].append(k)
                        assigned = True
                        break
        if not assigned:
            unassigned.append(k)

    return fwd_kernels, unassigned

def map_kernels_atom(events, sections):
    """Map kernels to sections using External id and hipGraphLaunch correlation."""
    cpu_ops = {}
    for e in events:
        if e.get('cat') in ('cpu_op', 'user_annotation'):
            eid = e.get('args', {}).get('External id')
            if eid is not None:
                cpu_ops[eid] = e

    sec_starts = [s['ts'] for s in sections]

    def find_section_for_ts(ts):
        idx = bisect.bisect_right(sec_starts, ts) - 1
        if 0 <= idx < len(sections) and sections[idx]['ts'] <= ts <= sections[idx]['end']:
            return sections[idx]['idx']
        return None

    # Build correlation → hipGraphLaunch CPU timestamp map for graph-replayed kernels
    graph_launch_by_corr = {}
    for e in events:
        if (e.get('cat') == 'cuda_runtime' and
                'GraphLaunch' in e.get('name', '')):
            corr = e.get('args', {}).get('correlation')
            if corr is not None:
                graph_launch_by_corr[corr] = e['ts']

    gpu_events = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]

    fwd_kernels = defaultdict(list)
    unassigned = []
    for k in gpu_events:
        sec_idx = None
        # Method 1: External id → cpu_op timestamp → section
        eid = k.get('args', {}).get('External id')
        if eid is not None and eid in cpu_ops:
            sec_idx = find_section_for_ts(cpu_ops[eid]['ts'])
        # Method 2: correlation → hipGraphLaunch timestamp → section
        if sec_idx is None:
            corr = k.get('args', {}).get('correlation')
            if corr is not None and corr in graph_launch_by_corr:
                sec_idx = find_section_for_ts(graph_launch_by_corr[corr])
        # Method 3 (last resort): GPU timestamp → section
        if sec_idx is None:
            sec_idx = find_section_for_ts(k['ts'])
        if sec_idx is not None:
            fwd_kernels[sec_idx].append(k)
        else:
            unassigned.append(k)

    return fwd_kernels, unassigned

# ────────────────────────────────────────────────────────────
# Kernel name disambiguation
# ────────────────────────────────────────────────────────────

def short_kernel_name(full_name):
    n = full_name
    # Tensile GEMMs (AMD)
    if 'Cijk_' in n:
        parts = n.split('_')
        mt = [p for p in parts if p.startswith('MT')]
        return f"tensile_gemm_{'_'.join(mt)}" if mt else "tensile_gemm"
    # CK blockscale GEMMs
    if 'kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale' in n:
        if 'Sequence<1, 32, 1, 8>' in n:
            return "ck_blockscale_gemm (large tile)"
        elif 'Sequence<1, 16, 1, 16>' in n:
            return "ck_blockscale_gemm (small tile)"
        return "ck_blockscale_gemm"
    # Batched gemm a8w8
    if 'batched_gemm_a8w8' in n:
        parts = n.split('_')
        bi = []
        for j, p in enumerate(parts):
            if p in ('M', 'N', 'K') and j > 0 and parts[j-1] == 'SIZE' and j+1 < len(parts):
                bi.append(f"{p}{parts[j+1]}")
        return f"batched_gemm_a8w8_{'_'.join(bi)}" if bi else "batched_gemm_a8w8"
    # Triton gemm a16w16
    if '_gemm_a16_w16_kernel' in n:
        parts = n.split('_')
        bi = []
        for j, p in enumerate(parts):
            if p in ('M', 'N', 'K') and j > 0 and parts[j-1] == 'SIZE' and j+1 < len(parts):
                bi.append(f"{p}{parts[j+1]}")
        return f"triton_gemm_a16w16_{'_'.join(bi)}" if bi else "triton_gemm_a16w16"
    # MoE gemm a8w4
    if '_moe_gemm_a8w4' in n:
        return "moe_gemm_a8w4"
    # RMSNorm
    if 'Rmsnorm2dFwd' in n:
        return "rmsnorm2d_fwd"
    if '_fused_add_rmsnorm_pad' in n:
        return "fused_add_rmsnorm_pad"
    # Quantization
    if 'dynamic_per_group_scaled_quant' in n:
        return "fp8_quantize"
    if '_downcast_to_static_fp8' in n:
        return "downcast_to_fp8"
    # Activation
    if 'act_and_mul_kernel' in n:
        return "act_and_mul_silu"
    # All-reduce
    if 'cross_device_reduce_1stage' in n:
        return "allreduce_1stage"
    if 'cross_device_reduce_2stage' in n:
        return "allreduce_2stage"
    # MoE routing/sorting
    if '_combined_routing_fused' in n:
        return "moe_routing_fused"
    if '_combined_routing' in n:
        return "moe_routing"
    if 'grouped_topk_opt_sort' in n:
        return "moe_topk_sort"
    if 'MoeSortingClearWorkspace' in n:
        return "moe_sort_clear"
    if 'MoeSortingKernel<' in n and 'MultiPhase' not in n:
        return "moe_sorting"
    if 'MoeSortingMultiPhaseKernel_P0' in n:
        return "moe_sort_phase0"
    if 'MoeSortingMultiPhaseKernel_P1' in n:
        return "moe_sort_phase1"
    if 'MoeSortingMultiPhaseKernel_P23' in n:
        return "moe_sort_phase23"
    if '_topk' in n:
        return "topk"
    if '_sum_bitmatrix_rows' in n:
        return "sum_bitmatrix_rows"
    if '_reduce_grouped' in n:
        return "reduce_grouped"
    # Attention
    if 'fmha_fwd' in n:
        return "flash_attention_fwd"
    if '_fwd_kernel' in n:
        return "triton_attention_fwd"
    if 'paged_attention' in n:
        return "paged_attention"
    if 'fmoe_bf16_blockscaleFp8' in n:
        return "fused_moe_gemm"
    if 'mla_a16w16' in n:
        return "mla_decode_attention"
    if 'kn_mla_reduce' in n:
        return "mla_reduce"
    if 'concat_and_cache_mla' in n:
        return "mla_kv_cache_update"
    if 'fused_qk_rope_cat_and_cache_mla' in n:
        return "fused_qk_rope_kv_cache"
    if 'reshape_and_cache_kernel' in n:
        return "kv_cache_update"
    # KV cache / attention indirect
    if 'kn_entry_2c_sbhd_cached_indirect' in n:
        return "cached_attn_indirect"
    if 'kn_get_mla_metadata' in n:
        return "mla_metadata_compute"
    # wvSplitK (AMD attention)
    if 'wvSplitK' in n:
        return "wv_splitk_attn"
    # Sampling
    if 'mixed_sample_outer_exponential' in n:
        return "sampling"
    # NCCL
    if 'ncclDevKernel' in n:
        return "nccl_collective"
    # Triton fused ops
    if 'triton_poi_fused' in n:
        parts = n.replace('triton_poi_fused_', '').split('_')
        return f"triton_fused_{n.split('triton_poi_fused_')[1][:40]}" if 'triton_poi_fused_' in n else "triton_fused"
    # Standard PyTorch ops
    if 'CatArrayBatchedCopy' in n:
        return "tensor_cat"
    if 'distribution_elementwise' in n:
        return "random_sampling"
    if 'direct_copy_kernel_cuda' in n or 'elementwise_kernel_manual_unroll' in n:
        return "copy_kernel"
    if 'vectorized_gather_kernel' in n:
        return "gather_kernel"
    if 'unrolled_elementwise_kernel' in n and 'add<int>' in n:
        return "elementwise_add_int"
    if 'unrolled_elementwise_kernel' in n and 'FillFunctor' in n:
        return "fill_kernel"
    if 'vectorized_elementwise_kernel' in n and 'FillFunctor' in n:
        return "fill_kernel"
    if 'vectorized_elementwise_kernel' in n and 'bfloat16tofloat32' in n:
        return "bf16_to_fp32_copy"
    if 'vectorized_elementwise_kernel' in n and 'BinaryFunctor' in n:
        return "binary_elementwise"
    if 'SoftMaxForward' in n:
        return "softmax_forward"
    if 'reduce_kernel' in n and 'ArgMax' in n:
        return "argmax_reduce"
    if 'index_elementwise_kernel' in n:
        return "index_elementwise"
    if 'scatter_gather' in n:
        return "scatter_gather"
    if '__amd_rocclr_copyBuffer' in n:
        return "rocclr_copy_buffer"
    # GPU memcpy/memset
    if 'MemcpyDtoD' in n or 'Memcpy' in n:
        return "gpu_memcpy"
    return n[:60]

def disambig_key(k):
    sname = short_kernel_name(k['name'])
    grid = tuple(k.get('args', {}).get('grid', []))
    return (sname, grid)

# ────────────────────────────────────────────────────────────
# Kernel invocation variant detection
# ────────────────────────────────────────────────────────────

def detect_kernel_variants(fwd_kernels, slist, min_ratio=1.10, min_cluster_size=3):
    """Identify distinct invocation variants for each kernel name.

    Within each iteration, kernels of the same name are numbered by GPU-timestamp
    order (ordinal 0, 1, 2, ...).  Across iterations the same ordinal represents
    the same "role" (e.g. Q-proj GEMM vs K-proj GEMM).  Ordinal positions are
    then clustered by their mean runtime so that positions with similar runtimes
    are grouped into one "variant".

    Uses recursive divisive clustering: repeatedly finds the largest gap in sorted
    ordinal means that creates sub-clusters whose means differ by >= min_ratio.

    Returns: dict  kernel_name -> list of variant dicts sorted by avg duration desc.
    Each variant dict has: label, calls_per_iter, total_count, durs, avg, min, max, total.
    """
    if not slist:
        return {}

    n_iters = len(slist)

    # ordinal_durs[kernel_name][ordinal] = [dur across iterations]
    ordinal_durs = defaultdict(lambda: defaultdict(list))
    for s in slist:
        kernels = sorted(fwd_kernels.get(s['idx'], []), key=lambda k: k['ts'])
        name_counter = defaultdict(int)
        for k in kernels:
            sname = short_kernel_name(k['name'])
            ordinal = name_counter[sname]
            name_counter[sname] += 1
            ordinal_durs[sname][ordinal].append(k['dur'])

    min_samples = max(1, n_iters // 2)

    result = {}
    for kname, ord_map in ordinal_durs.items():
        ord_stats = [(o, sum(d) / len(d), d)
                     for o, d in sorted(ord_map.items())
                     if len(d) >= min_samples]
        if not ord_stats:
            continue

        if len(ord_stats) == 1:
            result[kname] = [_make_variant(ord_stats, None, n_iters)]
            continue

        sorted_by_mean = sorted(ord_stats, key=lambda x: x[1])
        clusters = _cluster_ordinals_recursive(sorted_by_mean, min_ratio, min_cluster_size)

        variants = []
        for i, cluster in enumerate(clusters):
            label = chr(65 + i) if len(clusters) > 1 else None
            variants.append(_make_variant(cluster, label, n_iters))
        variants.sort(key=lambda v: -v['total'])
        result[kname] = variants

    return result


def _cluster_ordinals_recursive(sorted_by_mean, min_ratio, min_size):
    """Recursively split ordinal clusters at the largest gap where resulting
    sub-cluster means differ by at least min_ratio.

    Tries gaps in descending order.  A split is accepted only if both resulting
    sub-clusters have >= min_size members, their mean ratio >= min_ratio, AND
    the mean difference exceeds the measurement noise (2-sigma of within-ordinal
    variance, pooled across all ordinals).
    """
    if len(sorted_by_mean) <= 1:
        return [sorted_by_mean]

    means = [m for _, m, _ in sorted_by_mean]

    # Estimate noise: pooled within-ordinal variance / n_iters gives the
    # variance of each ordinal mean.  Splits where the cluster-mean gap is
    # < 2 * sqrt(pooled_var / n_iters) are indistinguishable from noise.
    noise_threshold = 0.0
    noise_vars = []
    for _, _, durs in sorted_by_mean:
        if len(durs) > 1:
            m = sum(durs) / len(durs)
            v = sum((d - m) ** 2 for d in durs) / (len(durs) - 1)
            noise_vars.append((v, len(durs)))
    if noise_vars:
        total_weight = sum(n for _, n in noise_vars)
        pooled_var = sum(v * n for v, n in noise_vars) / total_weight
        avg_n = total_weight / len(noise_vars)
        noise_threshold = 3.0 * (pooled_var / avg_n) ** 0.5

    gaps = []
    for i in range(1, len(means)):
        if means[i - 1] > 0:
            gaps.append((i, (means[i] - means[i - 1]) / means[i - 1]))
        else:
            gaps.append((i, float('inf')))

    gaps.sort(key=lambda x: -x[1])

    for split_idx, gap_val in gaps:
        if gap_val < 0.005:
            break

        left = sorted_by_mean[:split_idx]
        right = sorted_by_mean[split_idx:]

        if len(left) < min_size or len(right) < min_size:
            continue

        left_mean = sum(m for _, m, _ in left) / len(left)
        right_mean = sum(m for _, m, _ in right) / len(right)

        if right_mean / left_mean < min_ratio:
            continue

        if noise_threshold > 0 and (right_mean - left_mean) < noise_threshold:
            continue

        return (_cluster_ordinals_recursive(left, min_ratio, min_size) +
                _cluster_ordinals_recursive(right, min_ratio, min_size))

    return [sorted_by_mean]


def _make_variant(ord_stats, label, n_iters):
    all_durs = []
    for _, _, durs in ord_stats:
        all_durs.extend(durs)
    return {
        'label': label,
        'calls_per_iter': len(ord_stats),
        'total_count': len(all_durs),
        'durs': all_durs,
        'avg': sum(all_durs) / len(all_durs) if all_durs else 0,
        'min': min(all_durs) if all_durs else 0,
        'max': max(all_durs) if all_durs else 0,
        'total': sum(all_durs),
    }


# ────────────────────────────────────────────────────────────
# GPU idle time analysis — works on the global GPU timeline
# ────────────────────────────────────────────────────────────

def compute_idle_global(fwd_kernels, sections, events):
    """Compute GPU idle on the global timeline, correctly attributing gaps to sections.

    Instead of computing idle per-section (which breaks when sections overlap on GPU),
    we sort ALL GPU events by timestamp and compute gaps on the single-stream timeline.
    Each gap is attributed to the section of the PRECEDING kernel.
    """
    # Build kernel → section index mapping
    kernel_to_section = {}
    for sec_idx, kerns in fwd_kernels.items():
        for k in kerns:
            kernel_to_section[id(k)] = sec_idx

    # Collect all GPU events and sort by GPU timestamp
    all_gpu = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    all_gpu.sort(key=lambda e: e['ts'])

    if not all_gpu:
        return {}, [], []

    # Compute all gaps on the global timeline
    all_gaps = []
    for i in range(1, len(all_gpu)):
        prev_end = all_gpu[i-1]['ts'] + all_gpu[i-1]['dur']
        cur_start = all_gpu[i]['ts']
        gap = cur_start - prev_end
        if gap > 0.01:  # skip sub-microsecond rounding
            prev_sec = kernel_to_section.get(id(all_gpu[i-1]))
            next_sec = kernel_to_section.get(id(all_gpu[i]))
            all_gaps.append({
                'dur': gap,
                'after': short_kernel_name(all_gpu[i-1]['name']),
                'before': short_kernel_name(all_gpu[i]['name']),
                'prev_section': prev_sec,
                'next_section': next_sec,
            })

    # Build section type lookup
    sec_type = {s['idx']: s['type'] for s in sections}

    # Aggregate idle per section type
    idle_by_type = defaultdict(lambda: {'intra': 0, 'inter': 0, 'intra_count': 0, 'inter_count': 0})
    for g in all_gaps:
        ps, ns = g['prev_section'], g['next_section']
        if ps is not None and ns is not None and ps == ns:
            # Intra-section gap (both neighbors in same section)
            t = sec_type.get(ps, 'unknown')
            idle_by_type[t]['intra'] += g['dur']
            idle_by_type[t]['intra_count'] += 1
        elif ps is not None and ns is not None and ps != ns:
            # Inter-section gap (transition between sections)
            idle_by_type['_inter_']['inter'] += g['dur']
            idle_by_type['_inter_']['inter_count'] += 1
        else:
            # Gap adjacent to unassigned kernels
            idle_by_type['_unassigned_']['intra'] += g['dur']
            idle_by_type['_unassigned_']['intra_count'] += 1

    # Also compute per-section idle
    per_section_idle = defaultdict(lambda: {'intra': 0, 'count': 0})
    for g in all_gaps:
        ps, ns = g['prev_section'], g['next_section']
        if ps is not None and ns is not None and ps == ns:
            per_section_idle[ps]['intra'] += g['dur']
            per_section_idle[ps]['count'] += 1

    # Global stats
    first_start = all_gpu[0]['ts']
    last_end = all_gpu[-1]['ts'] + all_gpu[-1]['dur']
    total_span = last_end - first_start
    total_active = sum(e['dur'] for e in all_gpu)
    total_idle = sum(g['dur'] for g in all_gaps)

    global_stats = {
        'total_span': total_span,
        'total_active': total_active,
        'total_idle': total_idle,
        'utilization': total_active / total_span * 100 if total_span > 0 else 0,
    }

    return global_stats, idle_by_type, all_gaps, per_section_idle

# ────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────

def analyze(trace_path):
    print(f"Loading trace: {trace_path}")
    data = load_trace(trace_path)
    events = data['traceEvents']
    print(f"Total events: {len(events)}")

    framework = detect_framework(events)
    print(f"Detected framework: {framework}")

    # Get sections
    if framework == 'annotation':
        sections = find_sections_annotation(events)
    elif framework == 'atom':
        sections = find_sections_atom(events)
    else:
        print("ERROR: Could not detect framework type. Trying annotation-based...")
        sections = find_sections_annotation(events)
        if not sections:
            print("No sections found. Exiting.")
            return

    # Map kernels
    if framework == 'annotation':
        fwd_kernels, unassigned = map_kernels_annotation(events, sections)
    else:
        fwd_kernels, unassigned = map_kernels_atom(events, sections)

    total_kernels = sum(1 for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset'))
    assigned_total = sum(len(v) for v in fwd_kernels.values())

    # Group sections by type
    decode_secs = [s for s in sections if s['type'] == 'decode']
    prefill_secs = [s for s in sections if s['type'] == 'prefill']
    mixed_secs = [s for s in sections if s['type'] == 'mixed']
    non_decode_secs = [s for s in sections if s['type'] != 'decode']

    # Detect warmup: for Atom, the first decode with very few kernels
    warmup_idxs = set()
    if framework == 'atom':
        for s in decode_secs:
            nk = len(fwd_kernels.get(s['idx'], []))
            if nk < 100:
                warmup_idxs.add(s['idx'])

    steady_decode_secs = [s for s in decode_secs if s['idx'] not in warmup_idxs]

    # ═══════════════ OUTPUT ═══════════════
    W = 110
    print("\n" + "=" * W)
    print("GPU KERNEL TRACE ANALYSIS")
    print("=" * W)
    print(f"  Trace: {Path(trace_path).name}")
    print(f"  Framework: {framework}")
    print(f"  Total sections: {len(sections)}")
    print(f"    Prefill: {len(prefill_secs)}  |  Decode: {len(decode_secs)}  |  Mixed: {len(mixed_secs)}")
    print(f"  Total GPU events: {total_kernels}  (assigned: {assigned_total}, unassigned: {len(unassigned)})")

    # ── Section length summary ──
    print("\n" + "─" * W)
    print("SECTION LENGTH SUMMARY")
    print("─" * W)

    for label, slist in [("Non-decode-only (Prefill + Mixed)", non_decode_secs),
                          ("Decode-only", decode_secs),
                          ("Decode steady-state", steady_decode_secs)]:
        if not slist:
            continue
        durs = [s['dur'] for s in slist]
        gpu_times = [sum(k['dur'] for k in fwd_kernels.get(s['idx'], [])) for s in slist]
        n = len(slist)
        print(f"\n  {label}: {n} sections")
        print(f"    CPU duration — Total: {fmt_us(sum(durs))}  Avg: {fmt_us(sum(durs)/n)}  "
              f"Min: {fmt_us(min(durs))}  Max: {fmt_us(max(durs))}")
        print(f"    GPU kernel time — Total: {fmt_us(sum(gpu_times))}  Avg: {fmt_us(sum(gpu_times)/n)}")

    # If not too many sections, show each one
    if len(sections) <= 30:
        print(f"\n  Per-iteration detail:")
        print(f"  {'Idx':<5s} {'Type':<8s} {'Name':<55s} {'CPU dur':>10s} {'GPU krnls':>9s} {'GPU time':>10s}")
        print(f"  {'─'*100}")
        for s in sections:
            nk = len(fwd_kernels.get(s['idx'], []))
            gt = sum(k['dur'] for k in fwd_kernels.get(s['idx'], []))
            note = " (warmup)" if s['idx'] in warmup_idxs else ""
            print(f"  {s['idx']:<5d} {s['type']:<8s} {s.get('name','')[:55]:<55s} "
                  f"{fmt_us(s['dur']):>10s} {nk:>9d} {fmt_us(gt):>10s}{note}")

    # ── Kernel breakdown per section type ──
    section_groups = [
        ("NON-DECODE-ONLY (Prefill + Mixed)", non_decode_secs),
        ("DECODE-ONLY (steady-state)", steady_decode_secs),
    ]
    if mixed_secs:
        section_groups.insert(1, ("MIXED (Prefill+Decode)", mixed_secs))
    if len(decode_secs) != len(steady_decode_secs):
        section_groups.append(("DECODE-ONLY (all incl. warmup)", decode_secs))

    for section_label, slist in section_groups:
        if not slist:
            continue
        all_k = []
        for s in slist:
            all_k.extend(fwd_kernels.get(s['idx'], []))
        if not all_k:
            continue

        n_iters = len(slist)
        total_gpu = sum(k['dur'] for k in all_k)

        print("\n" + "=" * W)
        print(f"KERNEL BREAKDOWN: {section_label}")
        print(f"  Iterations: {n_iters}  |  Kernel invocations: {len(all_k)}  |  Total GPU time: {fmt_us(total_gpu)}")
        print("=" * W)

        # Variant-aware breakdown
        variants = detect_kernel_variants(fwd_kernels, slist)

        # Flatten variants into rows for sorted display
        rows = []
        for kname, var_list in variants.items():
            for v in var_list:
                rows.append((kname, v))
        rows.sort(key=lambda r: -r[1]['total'])

        print(f"\n  {'Kernel':<45s} {'Var':>3s} {'#/It':>5s} {'Cnt':>6s} {'Total':>10s} "
              f"{'Avg':>9s} {'Min':>8s} {'Max':>8s} {'%GPU':>6s}")
        print(f"  {'─'*100}")

        for kname, v in rows:
            pct = v['total'] / total_gpu * 100
            if pct < 0.05 and len(rows) > 30:
                continue
            var_str = v['label'] if v['label'] else '—'
            print(f"  {kname:<45s} {var_str:>3s} {v['calls_per_iter']:>5d} {v['total_count']:>6d} "
                  f"{fmt_us(v['total']):>10s} {fmt_us(v['avg']):>9s} {fmt_us(v['min']):>8s} "
                  f"{fmt_us(v['max']):>8s} {pct:>5.1f}%")

        # Aggregated summary (all variants of same kernel merged)
        print(f"\n  ── Aggregated by kernel type ──")
        name_groups = defaultdict(list)
        for kname, v in rows:
            name_groups[kname].extend(v['durs'])
        sorted_names = sorted(name_groups.items(), key=lambda x: -sum(x[1]))
        print(f"  {'Kernel Type':<50s} {'Cnt':>6s} {'Total':>10s} {'Avg':>10s} {'%GPU':>6s}")
        print(f"  {'─'*88}")
        for sname, durs in sorted_names:
            total = sum(durs)
            pct = total / total_gpu * 100
            if pct < 0.05:
                continue
            print(f"  {sname:<50s} {len(durs):>6d} {fmt_us(total):>10s} {fmt_us(total/len(durs)):>10s} {pct:>5.1f}%")

    # ── GEMM disambiguation ──
    gemm_keywords = ['gemm', 'cijk', 'fmoe', 'splitk', 'moe_gemm']
    has_gemms = False
    for section_label, slist in section_groups:
        if not slist:
            continue
        variants = detect_kernel_variants(fwd_kernels, slist)
        gemm_variants = {k: vl for k, vl in variants.items()
                         if any(g in k.lower() for g in gemm_keywords)}
        if not gemm_variants:
            continue

        if not has_gemms:
            print("\n" + "=" * W)
            print("GEMM KERNEL DISAMBIGUATION")
            print("=" * W)
            has_gemms = True

        total_gemm = sum(v['total'] for vl in gemm_variants.values() for v in vl)
        n_iters = len(slist)

        print(f"\n  ── {section_label} GEMMs ({fmt_us(total_gemm)} total) ──")
        rows = []
        for kname, vl in gemm_variants.items():
            for v in vl:
                rows.append((kname, v))
        rows.sort(key=lambda r: -r[1]['total'])

        print(f"  {'GEMM Kernel':<45s} {'Var':>3s} {'#/It':>5s} {'Cnt':>6s} "
              f"{'Total':>10s} {'Avg':>9s} {'Avg/It':>9s} {'%GEMM':>6s}")
        print(f"  {'─'*100}")
        for kname, v in rows:
            pct = v['total'] / total_gemm * 100
            var_str = v['label'] if v['label'] else '—'
            per_iter = v['total'] / n_iters
            print(f"  {kname:<45s} {var_str:>3s} {v['calls_per_iter']:>5d} {v['total_count']:>6d} "
                  f"{fmt_us(v['total']):>10s} {fmt_us(v['avg']):>9s} {fmt_us(per_iter):>9s} {pct:>5.1f}%")

    # ── Prefill vs Decode comparison ──
    if non_decode_secs and steady_decode_secs:
        print("\n" + "=" * W)
        print("KEY DIFFERENCES: PREFILL vs DECODE")
        print("=" * W)

        pf_k, dc_k = [], []
        for s in non_decode_secs:
            pf_k.extend(fwd_kernels.get(s['idx'], []))
        for s in steady_decode_secs:
            dc_k.extend(fwd_kernels.get(s['idx'], []))

        pf_names = defaultdict(list)
        dc_names = defaultdict(list)
        for k in pf_k:
            pf_names[short_kernel_name(k['name'])].append(k['dur'])
        for k in dc_k:
            dc_names[short_kernel_name(k['name'])].append(k['dur'])

        all_names = sorted(set(list(pf_names.keys()) + list(dc_names.keys())))
        print(f"\n  {'Kernel Type':<50s} {'Prefill Avg':>12s} {'Decode Avg':>12s} {'Ratio P/D':>10s}")
        print(f"  {'─'*90}")
        for name in all_names:
            pf_avg = sum(pf_names[name]) / len(pf_names[name]) if pf_names[name] else 0
            dc_avg = sum(dc_names[name]) / len(dc_names[name]) if dc_names[name] else 0
            if pf_avg == 0 and dc_avg == 0:
                continue
            pf_str = fmt_us(pf_avg) if pf_names[name] else "—"
            dc_str = fmt_us(dc_avg) if dc_names[name] else "—"
            ratio = pf_avg / dc_avg if dc_avg > 0 and pf_avg > 0 else 0
            ratio_str = f"{ratio:.1f}x" if ratio > 0 else "—"
            print(f"  {name:<50s} {pf_str:>12s} {dc_str:>12s} {ratio_str:>10s}")

    # ════════════════════════════════════════════════════════
    # GPU IDLE TIME ANALYSIS (global timeline approach)
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * W)
    print("GPU IDLE TIME ANALYSIS")
    print("=" * W)

    global_stats, idle_by_type, all_gaps, per_section_idle = \
        compute_idle_global(fwd_kernels, sections, events)

    sec_type = {s['idx']: s['type'] for s in sections}

    # Overall stats
    print(f"\n  ── Overall GPU Timeline ──")
    print(f"    Wall clock (first→last kernel): {fmt_us(global_stats['total_span'])}")
    print(f"    GPU active time:                {fmt_us(global_stats['total_active'])}  "
          f"({global_stats['utilization']:.1f}% utilization)")
    print(f"    GPU idle time:                  {fmt_us(global_stats['total_idle'])}  "
          f"({global_stats['total_idle']/global_stats['total_span']*100:.1f}%)")

    # Idle broken down by section type
    print(f"\n  ── Idle Breakdown by Section Type ──")
    print(f"    (Intra = gaps between consecutive kernels within the same section)")
    print(f"    (Inter = gaps at section boundaries, between last kernel of one section and first of next)")
    for t in ['prefill', 'mixed', 'decode', '_inter_', '_unassigned_']:
        info = idle_by_type.get(t)
        if not info or (info['intra'] == 0 and info['inter'] == 0):
            continue
        label = {'_inter_': 'Inter-section transitions',
                 '_unassigned_': 'Adjacent to unassigned kernels'}.get(t, t.capitalize())
        total_idle_t = info['intra'] + info['inter']
        pct = total_idle_t / global_stats['total_idle'] * 100 if global_stats['total_idle'] > 0 else 0
        if info['intra'] > 0:
            print(f"    {label + ' (intra):':<45s} {fmt_us(info['intra']):>10s}  "
                  f"({info['intra_count']:>5d} gaps, avg {fmt_us(info['intra']/info['intra_count']):>8s})  "
                  f"{pct:>5.1f}% of all idle")
        if info['inter'] > 0:
            print(f"    {label + ' (inter):':<45s} {fmt_us(info['inter']):>10s}  "
                  f"({info['inter_count']:>5d} gaps, avg {fmt_us(info['inter']/info['inter_count']):>8s})")

    # Per-section type: active time and idle
    print(f"\n  ── Per-Section-Type GPU Utilization ──")
    for label, slist in section_groups:
        if not slist:
            continue
        total_active = sum(sum(k['dur'] for k in fwd_kernels.get(s['idx'], [])) for s in slist)
        total_idle_s = sum(per_section_idle[s['idx']]['intra'] for s in slist if s['idx'] in per_section_idle)
        total_combined = total_active + total_idle_s
        util = total_active / total_combined * 100 if total_combined > 0 else 0
        n = len(slist)
        print(f"    {label}:")
        print(f"      Active: {fmt_us(total_active)}  Intra-idle: {fmt_us(total_idle_s)}  "
              f"Utilization: {util:.1f}%  ({n} iters, avg active/iter: {fmt_us(total_active/n)})")

    # Top 15 largest gaps
    top_gaps = sorted(all_gaps, key=lambda g: -g['dur'])[:15]
    print(f"\n  ── Top 15 Largest GPU Idle Gaps ──")
    print(f"  {'Gap':>10s}  {'Type':<12s} {'After Kernel':<40s} {'Before Kernel':<40s}")
    print(f"  {'─'*108}")
    for g in top_gaps:
        ps, ns = g['prev_section'], g['next_section']
        if ps is not None and ns is not None and ps == ns:
            gtype = sec_type.get(ps, '?') + ' intra'
        elif ps is not None and ns is not None:
            gtype = f"{sec_type.get(ps,'?')}→{sec_type.get(ns,'?')}"
        else:
            gtype = 'unassigned'
        print(f"  {fmt_us(g['dur']):>10s}  {gtype:<12s} {g['after']:<40s} {g['before']:<40s}")

    # Idle hotspots by kernel pair
    print(f"\n  ── Idle Hotspots: Which Kernel Transitions Cause the Most Idle? ──")
    pair_idle = defaultdict(lambda: {'total': 0, 'count': 0, 'max': 0})
    for g in all_gaps:
        key = (g['after'], g['before'])
        pair_idle[key]['total'] += g['dur']
        pair_idle[key]['count'] += 1
        pair_idle[key]['max'] = max(pair_idle[key]['max'], g['dur'])

    sorted_pairs = sorted(pair_idle.items(), key=lambda x: -x[1]['total'])
    total_all_idle = global_stats['total_idle']

    print(f"  {'After Kernel':<40s} {'Before Kernel':<40s} {'Tot Idle':>10s} {'Cnt':>6s} {'Avg':>8s} {'Max':>8s} {'%Idle':>6s}")
    print(f"  {'─'*115}")
    for (after, before), info in sorted_pairs[:20]:
        pct = info['total'] / total_all_idle * 100 if total_all_idle > 0 else 0
        avg_gap = info['total'] / info['count']
        print(f"  {after:<40s} {before:<40s} {fmt_us(info['total']):>10s} {info['count']:>6d} "
              f"{fmt_us(avg_gap):>8s} {fmt_us(info['max']):>8s} {pct:>5.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_trace_v2.py <trace_file.json.gz>")
        sys.exit(1)
    analyze(sys.argv[1])
