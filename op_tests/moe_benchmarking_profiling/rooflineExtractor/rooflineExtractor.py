import pandas as pd
import numpy as np
import pdb
import shutil
import argparse
import requests
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D


sigRuntime = 10  # Default value, can be changed from command line

# Bandwidths (GB/s) - Updated from benchmark measurements
caches = {
    "MI250X":  {"HBM": 1340,     "L2": 5019,     "vL1d": 9217,     "LDS": 20816},
    "MI300A":  {"HBM": 3688,     "L2": 20343,    "vL1d": 26092,    "LDS": 57574},
    "MI300X":  {"HBM": 4219.40,  "L2": 25092.59, "vL1d": 32941.97, "LDS": 72747.51},  # From rocm-amdgpu-bench measurements
    "MI355X":  {"HBM": 6198,     "L2": 34800,    "vL1d": 38263,    "LDS": 67368},
}

# Roofline numbers (peak compute in GFLOPs) - Updated from benchmark measurements
compute_peaks = {
    "MI250X":  36111,
    "MI300A":  78716,
    "MI300X":  508959.62,  # MFMA BF16 peak from rocm-amdgpu-bench measurements
    "MI355X":  126857,
}


# Function to convert columns with type mismatches to integers
def convert_columns_to_int(df):
    counters = ['SQ_INSTS_VALU_ADD_F16', 'SQ_INSTS_VALU_MUL_F16',       'SQ_INSTS_VALU_FMA_F16', 'SQ_INSTS_VALU_TRANS_F16',       'SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32',       'SQ_INSTS_VALU_FMA_F32', 'SQ_INSTS_VALU_TRANS_F32',       'SQ_INSTS_VALU_ADD_F64', 'SQ_INSTS_VALU_MUL_F64',       'SQ_INSTS_VALU_FMA_F64', 'SQ_INSTS_VALU_TRANS_F64',       'SQ_INSTS_VALU_MFMA_MOPS_F16', 'SQ_INSTS_VALU_MFMA_MOPS_BF16',       'SQ_INSTS_VALU_MFMA_MOPS_F32', 'SQ_INSTS_VALU_MFMA_MOPS_F64', 'SQ_LDS_IDX_ACTIVE', 'SQ_LDS_BANK_CONFLICT',       'TCP_TCC_READ_REQ_sum', 'TCP_TCC_WRITE_REQ_sum',       'TCP_TCC_ATOMIC_WITH_RET_REQ_sum', 'TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum',       'TCP_TOTAL_CACHE_ACCESSES_sum', 'SQ_INSTS_VALU_INT32',       'SQ_INSTS_VALU_INT64', 'SQ_INSTS_VALU_CVT', 'SQ_INSTS_SALU']

    # Checks for counters that were added in later versions of rooflineExtractor (to stay compatible with earlier counter files)
    if 'TCC_REQ_sum' in df.columns:
        counters.append('TCC_REQ_sum')
    if 'SQ_INSTS_VMEM_WR' in df.columns:
        counters.append('SQ_INSTS_VMEM_WR')
    if 'SQ_INSTS_VMEM_RD' in df.columns:
        counters.append('SQ_INSTS_VMEM_RD')
    if 'SQ_INSTS_VALU' in df.columns:
        counters.append('SQ_INSTS_VALU')
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        counters.append('SQ_INSTS_VALU_MFMA_MOPS_I8')
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        counters.append('SQ_INSTS_VALU_MFMA_MOPS_F8')
    # Check for CDNA2 vs. CDNA3-4 counters
    if 'TCC_BUBBLE_sum' in df.columns:
        counters.append('TCC_BUBBLE_sum')
        counters.append('TCC_EA0_RDREQ_sum')
        counters.append('TCC_EA0_RDREQ_32B_sum')
        counters.append('TCC_EA0_WRREQ_sum')
        counters.append('TCC_EA0_WRREQ_64B_sum')
    else:
        counters.append('TCC_EA_RDREQ_sum')
        counters.append('TCC_EA_RDREQ_32B_sum')
        counters.append('TCC_EA_WRREQ_sum')
        counters.append('TCC_EA_WRREQ_64B_sum')

    for counter in counters:
        df[counter] = pd.to_numeric(df[counter], errors='coerce').astype(int)
    return df




# Compute total flops, AI's
def compute_flops(df, arch, proj_arch=None):
    # Create a dictionary to store all new columns
    new_columns = {}

    # Compute total achieved FLOPs for each datatype (FP16, FP32, FP64)
    ## Scalar Ops
    new_columns['TOTAL_SALU'] = df['SQ_INSTS_SALU']

    ## Vector Ops
    new_columns['TOTAL_VALU_F16'] = 64 * (df['SQ_INSTS_VALU_ADD_F16'] + df['SQ_INSTS_VALU_MUL_F16'] + df['SQ_INSTS_VALU_TRANS_F16'] + 2 * df['SQ_INSTS_VALU_FMA_F16'])
    new_columns['TOTAL_VALU_F32'] = 64 * (df['SQ_INSTS_VALU_ADD_F32'] + df['SQ_INSTS_VALU_MUL_F32'] + df['SQ_INSTS_VALU_TRANS_F32'] + 2 * df['SQ_INSTS_VALU_FMA_F32'])
    new_columns['TOTAL_VALU_F64'] = 64 * (df['SQ_INSTS_VALU_ADD_F64'] + df['SQ_INSTS_VALU_MUL_F64'] + df['SQ_INSTS_VALU_TRANS_F64'] + 2 * df['SQ_INSTS_VALU_FMA_F64'])
    new_columns['TOTAL_VALU_I32'] = 64 * df['SQ_INSTS_VALU_INT32']
    new_columns['TOTAL_VALU_I64'] = 64 * df['SQ_INSTS_VALU_INT64']

    ## Matrix Ops
    new_columns['TOTAL_MOPS_F16'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F16']
    new_columns['TOTAL_MOPS_BF16'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_BF16']
    new_columns['TOTAL_MOPS_F32'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F32']
    new_columns['TOTAL_MOPS_F64'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F64']
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        new_columns['TOTAL_MOPS_F8'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F8']
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        new_columns['TOTAL_MOPS_I8'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_I8']

    # Other VALU Ops (e.g. Int16, Int8)
    if 'SQ_INSTS_VALU' in df.columns:
        new_columns['TOTAL_VALU_OTHER'] = 64 * (df['SQ_INSTS_VALU'] - (df['SQ_INSTS_VALU_ADD_F16'] + df['SQ_INSTS_VALU_MUL_F16'] + df['SQ_INSTS_VALU_TRANS_F16'] + df['SQ_INSTS_VALU_FMA_F16'] + df['SQ_INSTS_VALU_ADD_F32'] + df['SQ_INSTS_VALU_MUL_F32'] + df['SQ_INSTS_VALU_TRANS_F32'] + df['SQ_INSTS_VALU_FMA_F32'] + df['SQ_INSTS_VALU_ADD_F64'] + df['SQ_INSTS_VALU_MUL_F64'] + df['SQ_INSTS_VALU_TRANS_F64'] + df['SQ_INSTS_VALU_FMA_F64'] + df['SQ_INSTS_VALU_INT32'] + df['SQ_INSTS_VALU_INT64']))

    # Concat first batch of columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    ## Total
    TOTAL_OPS = df['TOTAL_SALU'] + df['TOTAL_VALU_F16'] + df['TOTAL_VALU_F32'] + df['TOTAL_VALU_F64'] + df['TOTAL_VALU_I32'] + df['TOTAL_VALU_I64'] + df['TOTAL_MOPS_F16'] + df['TOTAL_MOPS_BF16'] + df['TOTAL_MOPS_F32'] + df['TOTAL_MOPS_F64']
    if 'SQ_INSTS_VALU' in df.columns:
        TOTAL_OPS = TOTAL_OPS + df['TOTAL_VALU_OTHER']
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        TOTAL_OPS = TOTAL_OPS + df['TOTAL_MOPS_F8']
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        TOTAL_OPS = TOTAL_OPS + df['TOTAL_MOPS_I8']

    # Compute Bandwidths
    new_columns = {}
    new_columns['TOTAL_OPS'] = TOTAL_OPS

    ## LDS
    new_columns['BW_LDS'] = 32 * 4 * (df['SQ_LDS_IDX_ACTIVE'] - df['SQ_LDS_BANK_CONFLICT'])
    new_columns['BW_LDS_ATOMICS'] = 64 * (df['TCP_TCC_ATOMIC_WITH_RET_REQ_sum'] + df['TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum'])

    ## L2
    if 'TCC_REQ_sum' in df.columns:
        new_columns['BW_L2'] = 128 * df['TCC_REQ_sum']
    else:
        # Less reliable calculation kept to be backwards compatible with earlier rooflineExtractor versions
        new_columns['BW_L2'] = 64 * df['TCP_TCC_READ_REQ_sum'] + 64 * df['TCP_TCC_WRITE_REQ_sum'] + new_columns['BW_LDS_ATOMICS']

    ## vL1D
    if 'SQ_INSTS_VMEM_WR' in df.columns:
        new_columns['BW_vL1d'] = 256 * (df['SQ_INSTS_VMEM_WR'] + df['SQ_INSTS_VMEM_RD'])
    else:
        # Less reliable calculation kept to be backwards compatible with earlier rooflineExtractor versions
        new_columns['BW_vL1d'] = 128 * df['TCP_TOTAL_CACHE_ACCESSES_sum']

    ## HBM
    ### Check architecture
    if df.keys().str.contains('TCC_BUBBLE').sum() > 0:
        # We have a gfx942 or gfx950 arch counter file
        new_columns['BW_HBM'] = 128 * df['TCC_BUBBLE_sum'] + 32 * df['TCC_EA0_RDREQ_32B_sum'] + 64 * (df['TCC_EA0_RDREQ_sum'] - df['TCC_BUBBLE_sum'] - df['TCC_EA0_RDREQ_32B_sum']) + 32 * (df['TCC_EA0_WRREQ_sum'] - df['TCC_EA0_WRREQ_64B_sum']) + 64 * df['TCC_EA0_WRREQ_64B_sum']
    else:
        # Assuming gfx90a
        new_columns['BW_HBM'] = 32 * df['TCC_EA_RDREQ_32B_sum'] + 64 * (df['TCC_EA_RDREQ_sum'] - df['TCC_EA_RDREQ_32B_sum']) + 32 * (df['TCC_EA_WRREQ_sum'] - df['TCC_EA_WRREQ_64B_sum']) + 64 * df['TCC_EA_WRREQ_64B_sum']

    # Concat bandwidth columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Compute AI for each part of memory hierarchy (HBM, L2, L1)
    new_columns = {}

    ## LDS
    new_columns['AI_LDS_TOT'] = df['TOTAL_OPS'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_SALU'] = df['TOTAL_SALU'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        new_columns['AI_LDS_MOPS_F8'] = df['TOTAL_MOPS_F8'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        new_columns['AI_LDS_MOPS_I8'] = df['TOTAL_MOPS_I8'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_LDS_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    ## L2
    new_columns['AI_L2_TOT'] = df['TOTAL_OPS'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_SALU'] = df['TOTAL_SALU'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        new_columns['AI_L2_MOPS_F8'] = df['TOTAL_MOPS_F8'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        new_columns['AI_L2_MOPS_I8'] = df['TOTAL_MOPS_I8'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_L2_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    ## vL1D
    new_columns['AI_vL1d_TOT'] = df['TOTAL_OPS'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_SALU'] = df['TOTAL_SALU'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        new_columns['AI_vL1d_MOPS_F8'] = df['TOTAL_MOPS_F8'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        new_columns['AI_vL1d_MOPS_I8'] = df['TOTAL_MOPS_I8'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_vL1d_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    ## HBM
    new_columns['AI_HBM_TOT'] = df['TOTAL_OPS'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_SALU'] = df['TOTAL_SALU'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        new_columns['AI_HBM_MOPS_F8'] = df['TOTAL_MOPS_F8'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        new_columns['AI_HBM_MOPS_I8'] = df['TOTAL_MOPS_I8'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    new_columns['AI_HBM_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)

    # Concat AI columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Add columns for peaks
    new_columns = {}
    new_columns['HBM_BW_PEAK'] = df['AI_HBM_TOT'] * caches[arch]['HBM']
    new_columns['L2_BW_PEAK'] = df['AI_L2_TOT'] * caches[arch]['L2']
    new_columns['vL1d_BW_PEAK'] = df['AI_vL1d_TOT'] * caches[arch]['vL1d']
    new_columns['LDS_BW_PEAK'] = df['AI_LDS_TOT'] * caches[arch]['LDS']

    # Concat peak columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Load benchWarmer.csv relative to this script's location to avoid path errors
    bw_path = Path(__file__).parent / "benchWarmer.csv"
    df_peaks = pd.read_csv(bw_path)

    fp16_add_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' Add'), arch].values[0]
    fp16_mul_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' Mul'), arch].values[0]
    fp16_muladd_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' MulAdd'), arch].values[0]
    fp16_trans_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' Rsqrt'), arch].values[0]
    fp32_add_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' Add'), arch].values[0]
    fp32_mul_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' Mul'), arch].values[0]
    fp32_muladd_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' MulAdd'), arch].values[0]
    fp32_trans_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' Rsqrt'), arch].values[0]
    fp64_add_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' Add'), arch].values[0]
    fp64_mul_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' Mul'), arch].values[0]
    fp64_muladd_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' MulAdd'), arch].values[0]
    fp64_trans_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' Rsqrt'), arch].values[0]

    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        fp8_mfma_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp8') & (df_peaks['Operation'] == ' mfma'), arch].values[0]
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        i8_mfma_peak = df_peaks.loc[(df_peaks['Datatype'] == 'int8') & (df_peaks['Operation'] == ' mfma'), arch].values[0]
    fp16_mfma_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' mfma'), arch].values[0]
    bf16_mfma_peak = df_peaks.loc[(df_peaks['Datatype'] == 'bf16') & (df_peaks['Operation'] == ' mfma'), arch].values[0]
    fp32_mfma_peak = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' mfma'), arch].values[0]
    # Handle case where fp64 mfma is not present (MI100)
    fp64_mfma_row = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' mfma'), arch]
    fp64_mfma_peak = fp64_mfma_row.values[0] if not fp64_mfma_row.empty else 0

    # We'll do instruction-level analysis instead of operation-level analysis for integers, since we don't know the distribution
    # of instruction types
    # For INT32, Mul is the best performing test that hardware counters recognize as INT32
    int32_peak = df_peaks.loc[(df_peaks['Datatype'] == 'int32') & (df_peaks['Operation'] == ' Mul'), arch].values[0]
    # For INT64, INT32 MulAdd is the best performing test that hardware counters recognize as INT64
    # We divide by 2 to convert back to GInsts/s
    int64_peak = df_peaks.loc[(df_peaks['Datatype'] == 'int32') & (df_peaks['Operation'] == ' MulAdd'), arch].values[0] / 2

    # Assume 'other' operations are int8/16 muladds (big assumption, but better than nothing)
    other_peak = df_peaks.loc[(df_peaks['Datatype'] == 'int8') & (df_peaks['Operation'] == ' MulAdd'), arch].values[0] / 2

    # Calculate kernel-specific compute peak
    KERNEL_COMPUTE_PEAK = (
        64 * df['SQ_INSTS_VALU_ADD_F16'] / fp16_add_peak +
        64 * df['SQ_INSTS_VALU_MUL_F16'] / fp16_mul_peak +
        64 * 2 * df['SQ_INSTS_VALU_FMA_F16'] / fp16_muladd_peak +
        64 * df['SQ_INSTS_VALU_TRANS_F16'] / fp16_trans_peak +
        64 * df['SQ_INSTS_VALU_ADD_F32'] / fp32_add_peak +
        64 * df['SQ_INSTS_VALU_MUL_F32'] / fp32_mul_peak +
        64 * 2 * df['SQ_INSTS_VALU_FMA_F32'] / fp32_muladd_peak +
        64 * df['SQ_INSTS_VALU_TRANS_F32'] / fp32_trans_peak +
        64 * df['SQ_INSTS_VALU_ADD_F64'] / fp64_add_peak +
        64 * df['SQ_INSTS_VALU_MUL_F64'] / fp64_mul_peak +
        64 * 2 * df['SQ_INSTS_VALU_FMA_F64'] / fp64_muladd_peak +
        64 * df['SQ_INSTS_VALU_TRANS_F64'] / fp64_trans_peak +
        64 * df['SQ_INSTS_VALU_INT32'] / int32_peak +
        64 * df['SQ_INSTS_VALU_INT64'] / int64_peak +
        df['TOTAL_MOPS_F16'] / fp16_mfma_peak +
        df['TOTAL_MOPS_BF16'] / bf16_mfma_peak +
        df['TOTAL_MOPS_F32'] / fp32_mfma_peak +
        df['TOTAL_MOPS_F64'] / fp64_mfma_peak
    )

    if 'TOTAL_VALU_OTHER' in df.columns:
        KERNEL_COMPUTE_PEAK += df['TOTAL_VALU_OTHER'] / other_peak
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        KERNEL_COMPUTE_PEAK += df['TOTAL_MOPS_F8'] / fp8_mfma_peak
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        KERNEL_COMPUTE_PEAK += df['TOTAL_MOPS_I8'] / i8_mfma_peak

    KERNEL_COMPUTE_PEAK = (df['TOTAL_OPS'] / KERNEL_COMPUTE_PEAK).replace(np.inf, 0)

    new_columns = {}
    new_columns['KERNEL_COMPUTE_PEAK'] = KERNEL_COMPUTE_PEAK
    new_columns['COMPUTE_PEAK'] = compute_peaks[arch]

    # Concat compute peak columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Determine performance peak/limiter
    df_peaks = df[['HBM_BW_PEAK','L2_BW_PEAK','vL1d_BW_PEAK','LDS_BW_PEAK','KERNEL_COMPUTE_PEAK']]
    peaks = df_peaks.where(df_peaks > 0).min(axis=1)
    limiters = df_peaks.where(df_peaks > 0).idxmin(axis=1).str[:-5] + f" ({arch})"

    new_columns = {}
    new_columns['PEAK'] = peaks
    new_columns['LIMITER'] = limiters

    # Concat peak/limiter columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Calculate peaks for second architecture if provided
    if proj_arch is not None:
        new_columns = {}
        
        # Calculate bandwidth peaks for proj_arch
        new_columns[f'HBM_BW_PEAK_{proj_arch}'] = df['AI_HBM_TOT'] * caches[proj_arch]['HBM']
        new_columns[f'L2_BW_PEAK_{proj_arch}'] = df['AI_L2_TOT'] * caches[proj_arch]['L2']
        new_columns[f'vL1d_BW_PEAK_{proj_arch}'] = df['AI_vL1d_TOT'] * caches[proj_arch]['vL1d']
        new_columns[f'LDS_BW_PEAK_{proj_arch}'] = df['AI_LDS_TOT'] * caches[proj_arch]['LDS']
        
        # Calculate KERNEL_COMPUTE_PEAK for proj_arch
        df_peaks = pd.read_csv(bw_path)
        fp16_add_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' Add'), proj_arch].values[0]
        fp16_mul_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' Mul'), proj_arch].values[0]
        fp16_muladd_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' MulAdd'), proj_arch].values[0]
        fp16_trans_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' Rsqrt'), proj_arch].values[0]
        fp32_add_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' Add'), proj_arch].values[0]
        fp32_mul_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' Mul'), proj_arch].values[0]
        fp32_muladd_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' MulAdd'), proj_arch].values[0]
        fp32_trans_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' Rsqrt'), proj_arch].values[0]
        fp64_add_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' Add'), proj_arch].values[0]
        fp64_mul_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' Mul'), proj_arch].values[0]
        fp64_muladd_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' MulAdd'), proj_arch].values[0]
        fp64_trans_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' Rsqrt'), proj_arch].values[0]
        
        if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
            fp8_mfma_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp8') & (df_peaks['Operation'] == ' mfma'), proj_arch].values[0]
        if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
            i8_mfma_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'int8') & (df_peaks['Operation'] == ' mfma'), proj_arch].values[0]
        fp16_mfma_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp16') & (df_peaks['Operation'] == ' mfma'), proj_arch].values[0]
        bf16_mfma_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'bf16') & (df_peaks['Operation'] == ' mfma'), proj_arch].values[0]
        fp32_mfma_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp32') & (df_peaks['Operation'] == ' mfma'), proj_arch].values[0]
        fp64_mfma_row_2 = df_peaks.loc[(df_peaks['Datatype'] == 'fp64') & (df_peaks['Operation'] == ' mfma'), proj_arch]
        fp64_mfma_peak_2 = fp64_mfma_row_2.values[0] if not fp64_mfma_row_2.empty else 0
        
        int32_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'int32') & (df_peaks['Operation'] == ' Mul'), proj_arch].values[0]
        int64_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'int32') & (df_peaks['Operation'] == ' MulAdd'), proj_arch].values[0] / 2
        other_peak_2 = df_peaks.loc[(df_peaks['Datatype'] == 'int8') & (df_peaks['Operation'] == ' MulAdd'), proj_arch].values[0] / 2
        
        KERNEL_COMPUTE_PEAK_2 = (
            64 * df['SQ_INSTS_VALU_ADD_F16'] / fp16_add_peak_2 +
            64 * df['SQ_INSTS_VALU_MUL_F16'] / fp16_mul_peak_2 +
            64 * 2 * df['SQ_INSTS_VALU_FMA_F16'] / fp16_muladd_peak_2 +
            64 * df['SQ_INSTS_VALU_TRANS_F16'] / fp16_trans_peak_2 +
            64 * df['SQ_INSTS_VALU_ADD_F32'] / fp32_add_peak_2 +
            64 * df['SQ_INSTS_VALU_MUL_F32'] / fp32_mul_peak_2 +
            64 * 2 * df['SQ_INSTS_VALU_FMA_F32'] / fp32_muladd_peak_2 +
            64 * df['SQ_INSTS_VALU_TRANS_F32'] / fp32_trans_peak_2 +
            64 * df['SQ_INSTS_VALU_ADD_F64'] / fp64_add_peak_2 +
            64 * df['SQ_INSTS_VALU_MUL_F64'] / fp64_mul_peak_2 +
            64 * 2 * df['SQ_INSTS_VALU_FMA_F64'] / fp64_muladd_peak_2 +
            64 * df['SQ_INSTS_VALU_TRANS_F64'] / fp64_trans_peak_2 +
            64 * df['SQ_INSTS_VALU_INT32'] / int32_peak_2 +
            64 * df['SQ_INSTS_VALU_INT64'] / int64_peak_2 +
            df['TOTAL_MOPS_F16'] / fp16_mfma_peak_2 +
            df['TOTAL_MOPS_BF16'] / bf16_mfma_peak_2 +
            df['TOTAL_MOPS_F32'] / fp32_mfma_peak_2 +
            df['TOTAL_MOPS_F64'] / fp64_mfma_peak_2
        )
        
        if 'TOTAL_VALU_OTHER' in df.columns:
            KERNEL_COMPUTE_PEAK_2 += df['TOTAL_VALU_OTHER'] / other_peak_2
        if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
            KERNEL_COMPUTE_PEAK_2 += df['TOTAL_MOPS_F8'] / fp8_mfma_peak_2
        if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
            KERNEL_COMPUTE_PEAK_2 += df['TOTAL_MOPS_I8'] / i8_mfma_peak_2
        
        KERNEL_COMPUTE_PEAK_2 = (df['TOTAL_OPS'] / KERNEL_COMPUTE_PEAK_2).replace(np.inf, 0)
        
        new_columns[f'KERNEL_COMPUTE_PEAK_{proj_arch}'] = KERNEL_COMPUTE_PEAK_2
        new_columns[f'COMPUTE_PEAK_{proj_arch}'] = compute_peaks[proj_arch]
        
        # Concat proj_arch bandwidth and compute peak columns
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        
        # Determine performance peak/limiter for proj_arch
        df_peaks_2 = df[[f'HBM_BW_PEAK_{proj_arch}', f'L2_BW_PEAK_{proj_arch}', f'vL1d_BW_PEAK_{proj_arch}', f'LDS_BW_PEAK_{proj_arch}', f'KERNEL_COMPUTE_PEAK_{proj_arch}']]
        peaks_2 = df_peaks_2.where(df_peaks_2 > 0).min(axis=1)
        limiters_2 = df_peaks_2.where(df_peaks_2 > 0).idxmin(axis=1).str[:-len(proj_arch)-1] + f" ({proj_arch})"
        
        new_columns = {}
        new_columns[f'PEAK_{proj_arch}'] = peaks_2
        new_columns[f'LIMITER_{proj_arch}'] = limiters_2
        
        # Concat proj_arch peak/limiter columns
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df


def extract(counter, results, sig_runtime, plot, send, dump, arch = None, proj_arch = None):
    roofCountFilename = ""
    htmlFilename = send


    last_dot_index = counter.rfind('.')
    roofCountFilename = counter[:last_dot_index] # everything before .csv
    last_dot_index = results.rfind('.')

    sigRuntime = sig_runtime

    # Load roof counters into pandas df
    df_roof = pd.read_csv(counter)


    # Check if empty
    if df_roof.empty:
        print('Input roof counters file is empty')
        quit()
    # Check for wrong file
    if "CompleteNs" in df_roof.columns:
        print('Error: "results.csv" log file submitted with "-c" flag, which is for "roof-counters.csv" log file')
        quit()

    # Check for 'None' values and remove them
    total_none_kernels = max(df_roof.isnull().any(axis=1).sum(), (df_roof == 'None').any(axis=1).sum())

    if total_none_kernels > 0:
        print(f'{total_none_kernels} kernels had None values for some of the counters, which is a sign that runs of the application are non-deterministic. Removing these kernels and attempting to continue')
        df_roof = df_roof[~(df_roof == 'None').any(axis=1)]
        df_roof = df_roof[~(df_roof.isnull()).any(axis=1)]

    # If no arch specified, guess
    if (arch == None) & (df_roof.keys().str.contains('TCC_BUBBLE').sum() > 0):
        arch = 'MI300X'
    elif arch == None:
        arch = 'MI250X'

    df_roof = convert_columns_to_int(df_roof)
    df_roof = compute_flops(df_roof, arch, proj_arch)  # Compute AI's for each kernel dispatch

    # Check for rocprofv3 and convert to v1 format
    if 'Agent_Id' in df_roof.columns:
        df_roof = df_roof.drop(columns=['Agent_Id'])
        df_roof = df_roof.rename(columns={'Dispatch_Id':'Index'})
        df_roof = df_roof.rename(columns={'Kernel_Name':'KernelName'})

    # Aggregate kernels
    df = df_roof.groupby('KernelName', sort=False).mean(numeric_only=True).reset_index()


    # Get runtime stats
    df_runtime = pd.read_csv(results)

    if 'Kernel_Name' in df_runtime.columns:
        df_runtime = df_runtime.rename(columns={'Kernel_Name':'KernelName'})
        df_runtime['DurationNs'] = df_runtime['End_Timestamp'] - df_runtime['Start_Timestamp']
        df_runtime = df_runtime.rename(columns={'Dispatch_Id':'Index'})

    totalRuntimes = df_runtime.groupby('KernelName')['DurationNs'].sum()
    averageRuntimes = df_runtime.groupby('KernelName')['DurationNs'].mean()
    percentRuntimes = df_runtime.groupby('KernelName').sum()['DurationNs']/df_runtime['DurationNs'].sum()*100

    df = pd.merge(df, totalRuntimes, on='KernelName').rename(columns={"DurationNs":"RuntimeNs"})
    df = pd.merge(df, averageRuntimes, on='KernelName').rename(columns={"DurationNs":"AverageNs"})
    df = pd.merge(df, percentRuntimes, on='KernelName').rename(columns={"DurationNs":"Percentage"})
    # Add column for number of kernels
    df = df.merge(df_roof.groupby('KernelName', sort=False).size().reset_index(name='Count'), on='KernelName')

    # Calculate projected runtime for second architecture if provided
    if proj_arch is not None:
        df['PROJECTED_RUNTIME'] = df['RuntimeNs'] * (df['PEAK'] / df[f'PEAK_{proj_arch}'])

    insigKernels = len(df[df['Percentage'] < sigRuntime])  # save this value for guided analysis

    df_roof = df_roof.merge(df_runtime, on='Index')

    df.set_index('KernelName', inplace=True)

    # Prepare for analysis
    df = df.sort_values(by='Percentage', ascending=False)
    df_peaks = df[['HBM_BW_PEAK','L2_BW_PEAK','vL1d_BW_PEAK','LDS_BW_PEAK','KERNEL_COMPUTE_PEAK']]

    df['LIMITER'] = df_peaks.where(df_peaks > 0).idxmin(axis=1).str[:-5] + " (" + arch + ")"
    df['Throughput'] = df['TOTAL_OPS'] / df['AverageNs']
    df['PercentAchieved'] = df['Throughput'] / df['PEAK'] * 100

    # Guided analysis
    print(f"Total unique kernels: {len(df)}")
    print(f"Total kernel dispatches: {len(df_roof)}")
    for index, row in df.iterrows():  # loop through each kernel
        if row['Percentage'] < sigRuntime:
            continue
        print("\n" + index)
        print(f"  Total operations per dispatch: ", "{:e}".format(row['TOTAL_OPS']), "ops")
        print(f"  Average runtime per dispatch:  ", "{:e}".format(row['AverageNs']), "ns")
        print(f"  Total runtime:                 ", "{:e}".format(row['RuntimeNs']), "ns")
        if proj_arch is not None:
            print(f"  Projected runtime ({proj_arch}):    ", "{:e}".format(row['PROJECTED_RUNTIME']), "ns")
        print(f"  Total contribution to runtime:  {round(row['Percentage'], 3)} %")
        print(f"  Total dispatches:               {int(row['Count'])}")

        print(f"\n  Arithmetic intensity (HBM):  ", round(row['AI_HBM_TOT'], 4))
        print(f"  Arithmetic intensity (L2):   ", round(row['AI_L2_TOT'], 4))
        print(f"  Arithmetic intensity (L1):   ", round(row['AI_vL1d_TOT'], 4))
        print(f"  Arithmetic intensity (LDS):  ", round(row['AI_LDS_TOT'], 4))

        print(f"\n  Performance limiter:         ", row['LIMITER'])
        print(f"  Peak achievable throughput:  ", round(row['PEAK'], 2), "GFLOPS/s")
        print(f"  Distance from roofline:      ", round(row['PercentAchieved'], 4), "%")

        # Calculate achieved
        achieved = row['Throughput']
        print(f"\n  Achieved throughput:         ", round(achieved, 2), "GFLOPS/s")

    print(f"\n{insigKernels} kernels omitted for having less than {sigRuntime} percent runtime (use --sig-runtime to change threshold)")
    print(f"\nTotal application runtime on {arch}: {sum(totalRuntimes):.2e} ns ({sum(totalRuntimes)/1e9:.6f} s)")
    if proj_arch is not None:
        total_projected = df['PROJECTED_RUNTIME'].sum()
        speedup = sum(totalRuntimes) / total_projected
        print(f"Total projected runtime on {proj_arch}: {total_projected:.2e} ns ({total_projected/1e9:.6f} s)")
        print(f"Projected speedup: {speedup:.3f}x")

    df_roof['Percentage'] = df_roof['DurationNs']/sum(totalRuntimes) * 100
    df_roof['Throughput'] = df_roof['TOTAL_OPS'] / df_roof['DurationNs']
    df_roof['PercentAchieved'] = df_roof['Throughput'] / df_roof['PEAK'] * 100
    df_roof = df_roof.rename(columns={'KernelName_x':'KernelName'}).merge(df['Percentage'],on='KernelName')
    df_roof = df_roof.rename(columns={'Percentage_x':'Percentage'})
    df_roof = df_roof.rename(columns={'Percentage_y':'PercentageAggregate'})
    roofline_distance = (df_roof['Percentage'] * df_roof['PercentAchieved']).sum()/100
    print(f"Average distance from roofline: {roofline_distance:.3f} %")
    print()

    # Interactive roofline plot
    if plot:
        df_plot = df_roof
        df_plot['TotalKernels'] = len(df_plot)  # Saving in this format to pass to tooltip

        # Maximum number of kernel dispatches to plot (browser slows down with more)
        n_samples = 50000

        if len(df_plot) > n_samples:
            df_plot.sort_values('Index')
            df_plot = df_plot.iloc[::len(df_plot) // n_samples]
            n_samples = len(df_plot)  # Adjust for remainder

        # Sort the data by significance
        df_plot = df_plot.sort_values(by='PercentageAggregate', ascending=False)

        # Assign a unique color to each name
        color_map = px.colors.qualitative.Plotly
        name_to_color = {name: color_map[i % len(color_map)] for i, name in enumerate(df.index)}
        df_plot['color'] = [name_to_color[name] for name in df_plot['KernelName']]
        df['color'] = [name_to_color[name] for name in df.index]

        # Compute the range for the plot
        x_min = 0.001
        x_max = 100000
        y_min = 0.5
        y_max = 200000

        # Get x-values for lines
        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        # Add a point for each spot a bandwidth line intersects the compute line
        x_vals = np.sort(np.append(x_vals, [compute_peaks[arch] / caches[arch][cache] for cache in caches[arch] for arch in caches]))

        rooflines = [
            go.Scatter(
                x=x_vals,
                y=np.minimum(caches[arch][key] * x_vals, compute_peaks[arch]),
                visible=True,
                mode='lines',
                name=f'{arch} {key} Achievable Peak',
                line=dict(color=color_map[0], dash='solid')
            )
            for key in caches[arch].keys()
        ]

        # Truncate long kernel names (looking at you, rocblas)
        df['short_name'] = np.where(
            df.index.str.len() > 50,
            df.index.str.slice(0, 47) + '…',  # 47 + 1 char ellipsis = 48 visible chars
            df.index
        )
        # Truncate long kernel names (looking at you, rocblas)
        df_plot['short_name'] = np.where(
            df_plot['KernelName'].str.len() > 50,
            df_plot['KernelName'].str.slice(0, 47) + '…',  # 47 + 1 char ellipsis = 48 visible chars
            df_plot['KernelName']
        )

        # Layout with log-log axes and slider
        layout = go.Layout(
            title=f'Roofline Plot for kernels in {roofCountFilename}',
            xaxis=dict(
                type='log',
                title=f'Arithmetic Intensity (Flops per Byte Accessed)',
                range=[np.log10(x_min), np.log10(x_max)],
                autorange=False
            ),
            yaxis=dict(
                type='log',
                title='Throughput (GFLOPs/s)',
                range=[np.log10(y_min), np.log10(y_max)],
                autorange=False
            )
        )

        def interactive_plot(cache):

            # Add scatter plot
            scatter_items = []
            for name in df_plot['KernelName'].unique():
                kernels = df_plot[df_plot['KernelName'] == name]
                short_name = kernels.iloc[0]['short_name']
                short_name += f'\t{round(kernels.iloc[0]["PercentageAggregate"], 3)}% runtime'
                # Pass name, percent for tooltip
                customdata=np.stack([kernels['short_name'], kernels['Percentage'], kernels['Index'], kernels['TotalKernels'], kernels['PercentageAggregate'], kernels['PEAK'], kernels['LIMITER']], axis=-1)
                scatter_items.append(go.Scatter(
                    x=kernels[f'AI_{cache}_TOT'],
                    y=kernels['Throughput'],
                    mode='markers',
                    name=short_name,
                    marker=dict(color=kernels['color']),
                    visible=False,
                    customdata = customdata,
                    hovertemplate=
                        'Name: %{customdata[0]}<br>' +
                        'Index: %{customdata[2]} / %{customdata[3]}<br>' +
                        'AI: %{x}<br>' +
                        'Achieved throughput: %{y:.3f} GFLOPs/s<br>' +
                        'Peak throughput: %{customdata[5]:.3f} GFLOPs/s<br>' +
                        'Performance limiter: %{customdata[6]}<br>' +
                        'Aggregate percent runtime: %{customdata[4]:.5f} %<br>' +
                        'Individual percent runtime: %{customdata[1]:.5f} %<extra></extra>'
                ))

            return scatter_items

        def interactive_plot_agg(cache):

            # Add scatter plot
            scatter_items = []
            for index, kernel in df.iterrows():
                short_name = kernel['short_name']
                short_name += f'\t{round(kernel["Percentage"], 3)}% runtime'
                # Pass name, percent for tooltip
                customdata=np.stack([[kernel['short_name']], [kernel['Percentage']], [kernel['PEAK']], [kernel['LIMITER']], [kernel['Count']]], axis=-1)
                scatter_items.append(go.Scatter(
                    x=[kernel[f'AI_{cache}_TOT']],
                    y=[kernel['Throughput']],
                    mode='markers',
                    name=short_name,
                    marker=dict(color=kernel['color']),
                    visible=False,
                    customdata = customdata,
                    hovertemplate=
                        'Name: %{customdata[0]}<br>' +
                        'AI: %{x}<br>' +
                        'Achieved throughput: %{y:.3f} GFLOPs/s<br>' +
                        'Peak throughput: %{customdata[2]:.3f} GFLOPs/s<br>' +
                        'Performance limiter: %{customdata[3]}<br>' +
                        'Total dispatches: %{customdata[4]}<br>' +
                        'Aggregate percent runtime: %{customdata[1]:.5f} %<br>' +
                        '<extra></extra>'
                ))

            return scatter_items

        scatter_hbm = interactive_plot('HBM')
        scatter_l2 = interactive_plot('L2')
        scatter_l1 = interactive_plot('vL1d')
        scatter_lds = interactive_plot('LDS')
        scatter_hbm_agg = interactive_plot_agg('HBM')
        scatter_l2_agg = interactive_plot_agg('L2')
        scatter_l1_agg = interactive_plot_agg('vL1d')
        scatter_lds_agg = interactive_plot_agg('LDS')

        # Set HBM aggregate as default visibility
        for scatter in scatter_hbm_agg:
            scatter.visible = True
        for roofline in rooflines:
            roofline.line.width = 1
        for roofline in rooflines[::4]:
            roofline.line.width = 3

        # Add a new column which is the percentage runtime of the kernel and all kernels with a higher percentage runtime
        df['CumulativePercentageAbove'] = df['Percentage'].apply(lambda x: df[df['Percentage'] >= x]['Percentage'].sum())
        df_plot = df_plot.merge(df['CumulativePercentageAbove'], on='KernelName')
        # Create slider steps based on percentage runtime thresholds
        thresholds = df['CumulativePercentageAbove'].tolist()
        thresholds.sort(reverse=True)
        # Set max number of thresholds at 40
        # Ensure there is a 100% visible option
        thresholds = [100] + thresholds[-40:]

        # Create separate slider for each of the four cache levels, and another four for the aggregates
        sliders = []
        # Individual kernel dispatches
        for c in range(4):
            steps = []
            # Iterate over each threshold (point on slider)
            for threshold in thresholds:
                # Make the rooflines visible for every cache level
                visible = [True] * 4
                visible = visible * int(len(rooflines)/4)

                # Filter the scatter plots to display only the selected cache level
                visible = visible + [False] * (c * len(df_plot.groupby('KernelName')))
                visible = visible + (pd.Series(df_plot.groupby('KernelName')['CumulativePercentageAbove'].first().tolist()).sort_values() <= threshold).tolist()
                visible = visible + [False] * ((3 - c) * len(df_plot.groupby('KernelName')))
                # Add invisible aggregates for every cache level
                visible = visible + [False] * (4 * len(df))
                steps.append(dict(
                    method="update",
                    args=[{"visible": visible}],
                    label=f"{threshold:.3f}%"
                ))
            sliders.append(dict(
                active=0,
                currentvalue={"prefix": "Total Percent Runtime Displayed: "},
                pad={"t": 50},
                steps=steps
            ))
        # Aggregate kernel dispatches
        for c in range(4):
            steps = []
            for threshold in thresholds:
                # Make the rooflines visible for every cache level
                visible = [True] * 4
                visible = visible * int(len(rooflines)/4)

                # Make individual kernel dispatches invisible for every cache level
                visible = visible + [False] * (4 * len(df_plot.groupby('KernelName')))
                # Aggregates
                # Filter the scatter plots to display only the selected cache level
                visible = visible + [False] * (c * len(df))
                visible = visible + (pd.Series(df['CumulativePercentageAbove'].tolist()).sort_values() <= threshold).tolist()
                visible = visible + [False] * ((3 - c) * len(df))
                steps.append(dict(
                    method="update",
                    args=[{"visible": visible}],
                    label=f"{threshold:.3f}%"
                ))
            sliders.append(dict(
                active=0,
                currentvalue={"prefix": "Total Percent Runtime Displayed: "},
                pad={"t": 50},
                steps=steps
            ))

        # Add disclaimer if kernel dispatches need to be filtered
        disclaimer = None
        if df_plot.iloc[0]['TotalKernels'] > n_samples:
            disclaimer = dict(
                text=f"Depicting {n_samples} kernel dispatches out of {df_plot.iloc[0]['TotalKernels']}",
                xref="paper", yref="paper",
                x=1, y=0.05,
                showarrow=False,
                font=dict(size=12),
                xanchor='left',
                yanchor='top'
            )

        # Combine scatters and rooflines
        fig = go.Figure(data=rooflines + scatter_hbm + scatter_l2 + scatter_l1 + scatter_lds + scatter_hbm_agg + scatter_l2_agg + scatter_l1_agg + scatter_lds_agg, layout=layout)
        fig.update_layout(
            sliders=[sliders[4]],
            plot_bgcolor="white",
            xaxis=dict(
                gridcolor="lightgray",
                zerolinecolor="lightgray",
                linecolor="black",
            ),
            yaxis=dict(
                gridcolor="lightgray",
                zerolinecolor="lightgray",
                linecolor="black",
            ),
            updatemenus=[
                # Add dark mode toggle
                dict(
                    type="buttons",
                    direction="right",
                    buttons=list([
                        dict(label="Light Mode",
                            method="relayout",
                            args=[{
                                "plot_bgcolor": "white",
                                "paper_bgcolor": "white",
                                "font.color": "black",
                                "xaxis.gridcolor": "lightgray",
                                "xaxis.zerolinecolor": "lightgray",
                                "xaxis.linecolor": "black",
                                "yaxis.gridcolor": "lightgray",
                                "yaxis.zerolinecolor": "lightgray",
                                "yaxis.linecolor": "black"
                            }]),
                        dict(label="Dark Mode",
                            method="relayout",
                            args=[{
                                "plot_bgcolor": "black",
                                "paper_bgcolor": "black",
                                "font.color": "white",
                                "xaxis.gridcolor": "gray",
                                "xaxis.zerolinecolor": "gray",
                                "xaxis.linecolor": "white",
                                "yaxis.gridcolor": "gray",
                                "yaxis.zerolinecolor": "gray",
                                "yaxis.linecolor": "white"
                            }])
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1,
                    xanchor="right",
                    y=1.1,
                    yanchor="top"
                ),
                # Add memory hierarchy toggle
                dict(
                    type="buttons",
                    direction="down",
                    buttons=list([
                        dict(label="HBM Agg",
                            method="update",
                            args=[
                                {
                                    "line": [
                                        {"width": 3 if j == 0 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [False] * len(scatter_hbm) * 4 +
                                        [True] * len(scatter_hbm_agg) +
                                        [False] * len(scatter_hbm_agg) * 3
                                    ),
                                }, {
                                    "sliders": [sliders[4]],
                                    "annotations": [None]
                                }
                            ]),
                        dict(label="HBM",
                            method="update",
                            args=[
                                {
                                    "line": [
                                        {"width": 3 if j == 0 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [True] * len(scatter_hbm) +
                                        [False] * len(scatter_hbm) * 3 +
                                        [False] * len(scatter_hbm_agg) * 4
                                    ),
                                }, {
                                    "sliders": [sliders[0]],
                                    "annotations": [disclaimer]
                                }
                            ]),
                        dict(label="L2 Agg",
                            method="update",
                            args=[{
                                    "line": [
                                        {"width": 3 if j == 1 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [False] * len(scatter_l2) * 4 +
                                        [False] * len(scatter_l2_agg) +
                                        [True] * len(scatter_l2_agg) +
                                        [False] * len(scatter_l2_agg) * 2
                                    ),
                                }, {
                                    "sliders": [sliders[5]],
                                    "annotations": [None]
                                }
                            ]),
                        dict(label="L2",
                            method="update",
                            args=[{
                                    "line": [
                                        {"width": 3 if j == 1 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [False] * len(scatter_l2) +
                                        [True] * len(scatter_l2) +
                                        [False] * len(scatter_l2) * 2 +
                                        [False] * len(scatter_l2_agg) * 4
                                    ),
                                }, {
                                    "sliders": [sliders[1]],
                                    "annotations": [disclaimer]
                                }
                            ]),
                        dict(label="vL1d Agg",
                            method="update",
                            args=[{
                                    "line": [
                                        {"width": 3 if j == 2 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [False] * len(scatter_l1) * 4 +
                                        [False] * len(scatter_l1_agg) * 2 +
                                        [True] * len(scatter_l1_agg) +
                                        [False] * len(scatter_l1_agg)
                                    ),
                                }, {
                                    "sliders": [sliders[6]],
                                    "annotations": [None]
                                }
                            ]),
                        dict(label="vL1d",
                            method="update",
                            args=[{
                                    "line": [
                                        {"width": 3 if j == 2 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [False] * len(scatter_l1) * 2 +
                                        [True] * len(scatter_l1) +
                                        [False] * len(scatter_l1) +
                                        [False] * len(scatter_l1_agg) * 4
                                    ),
                                }, {
                                    "sliders": [sliders[2]],
                                    "annotations": [disclaimer]
                                }
                            ]),
                        dict(label="LDS Agg",
                            method="update",
                            args=[{
                                    "line": [
                                        {"width": 3 if j == 3 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [False] * len(scatter_lds) * 4 +
                                        [False] * len(scatter_lds_agg) * 3 +
                                        [True] * len(scatter_lds_agg)
                                    ),
                                }, {
                                    "sliders": [sliders[7]],
                                    "annotations": [None]
                                }
                            ]),
                        dict(label="LDS",
                            method="update",
                            args=[{
                                    "line": [
                                        {"width": 3 if j == 3 else 1, "color": color_map[i]}
                                        for i in range(int(len(rooflines) / 4))
                                        for j in range(4)
                                    ],
                                    "visible": (
                                        [True] * len(rooflines) +
                                        [False] * len(scatter_lds) * 3 +
                                        [True] * len(scatter_lds) +
                                        [False] * len(scatter_lds_agg) * 4
                                    ),
                                }, {
                                    "sliders": [sliders[3]],
                                    "annotations": [disclaimer]
                                }
                            ]),
                    ]),
                    showactive=True,
                ),

            ]
        )

        fig.write_html(f'{roofCountFilename}.html')
        print(f"Roofline plot saved to                  {roofCountFilename}.html")

    # Send html file to webserver
    if send:
      if not Path(f'{roofCountFilename}.html').exists():
        print(f"Error: {roofCountFilename}.html does not exist. Please use --plot when using --send.")
        return
      try:
        shutil.move(f'{roofCountFilename}.html',f'{htmlFilename}.html')
      except OSError as e:
        print(f"Error moving file {roofCountFilename}.html to {htmlFilename}.html: {e}")
        return
      # URL of the upload endpoint
      url = "http://canofcorn.amd.com:5003/upload"

      # Open the file in binary mode and send via POST
      try:
        with open(f'{htmlFilename}.html', "rb") as f:
          files = {"file": f}
          response = requests.post(url, files=files)
        # Print server response
        print(response.text)
        print(f"Roofline plot saved to                  {htmlFilename}.html")
      except (OSError, requests.RequestException) as e:
        print(f"Error sending file {htmlFilename}.html to server: {e}")
    # Output Results to CSV
    if dump:
        df.to_csv(roofCountFilename + '_EXTRACTED_AGG.csv')
        df_roof.to_csv(roofCountFilename + '_EXTRACTED.csv')
        print(f"Full dataframe dumped to                {roofCountFilename}_EXTRACTED.csv")
        print(f"Aggregate kernels dataframe dumped to   {roofCountFilename}_EXTRACTED_AGG.csv")

def main():

    # Get filenames from input args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--counter', required=True, help='Provide roof_counters.csv filename from rocprof -i')
    parser.add_argument('-r', '--results', required=True, help='Provide results.csv filename from rocprof --stats')
    parser.add_argument('--sig-runtime', required=False, default=10, type=float, help='Provide the percentage of runtime considered "significant" for analysis (kernels under that percentage will be omitted)')
    parser.add_argument('-p', '--plot', action='store_true', required=False, help='Generate plots')
    parser.add_argument('-s', '--send', required=False, help='Provide a string to name the html file generated from --plot. Ex: --send lammps_mi300x')
    parser.add_argument('-d', '--dump', action='store_true', required=False, help='Dump DataFrame to csv')
    parser.add_argument('--arch', required=False, help='Supply architecture (to aid in guided analysis). Options: MI250X, MI300A, MI300X, MI355X')
    parser.add_argument('--proj-arch', required=False, help='Supply second architecture for runtime projection. Options: MI250X, MI300A, MI300X, MI355X')

    args = parser.parse_args()
    if args.arch != None:
        args.arch = args.arch.upper()
    if args.proj_arch != None:
        args.proj_arch = args.proj_arch.upper()
    extract(args.counter, args.results, args.sig_runtime, args.plot, args.send, args.dump, args.arch, args.proj_arch)

if __name__ == "__main__":
    main()
