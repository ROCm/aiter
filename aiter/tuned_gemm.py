'''
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import os
from pathlib import Path
import functools
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import hipb_create_extension, hipb_mm, getHipblasltKernelName
from aiter import rocb_create_extension, rocb_mm
from aiter import logger

this_dir = os.path.dirname(os.path.abspath(__file__))


class TunedGemm:

    def __init__(self):
        self.extensions_created = False
        self.save_gemm = int(os.environ.get('VLLM_TUNE_GEMM', 0))
        self.untune_path = f'{this_dir}/configs/untuned_gemm.csv'
        self.tune_path = f'{this_dir}/configs/tuned_gemm.csv'
        self.bestsols = {}
        self.solMap = ['torch', 'hipblaslt', 'rocblas', 'skinny']
        self.cu_count = torch.cuda.get_device_properties(
            device='cuda').multi_processor_count

        # self.use_skinny = is_hip() and VLLM_USE_ROCM_SKINNY_GEMM and \
        #     "gfx1" not in torch.cuda.get_device_properties('cuda').gcnArchName
        self.use_skinny = True

        if (self.save_gemm == 1):
            self.tuned_df = pd.DataFrame(
                columns=['M', 'N', 'K', 'bias', 'dtype', 'outdtype', 'scaleAB'])
        else:
            self.tuned_df = None

    def load_best_sols(self):
        if self.tune_path is not None and Path(self.tune_path).is_file():
            self.bestsols = pd.read_csv(self.tune_path)
            if len(self.bestsols) > 0 and 'kernelName' in self.bestsols.columns:
                hipblasltKernelNames = self.bestsols.apply(
                    lambda s: getHipblasltKernelName(s.solidx) if s.libtype == 'hipblaslt' else "", axis=1)
                pd.set_option('display.max_colwidth', 100)
                assert hipblasltKernelNames.equals(
                    self.bestsols['kernelName'].fillna('')), "error: gradlib tune gemm not match the current environment, need re-tune!!!\n" + \
                    f"differece:\n{pd.concat([self.bestsols[['solidx','kernelName']], hipblasltKernelNames], axis=1)[hipblasltKernelNames != self.bestsols['kernelName'].fillna('')]}"

    def create_ds(self):
        df: pd.DataFrame = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            key = (ds['M'], ds['N'], ds['K'], ds['bias'], ds['dtype'], ds['outdtype'], ds['scaleAB'])
            if ds['libtype'] == 'hipblaslt':
                soltype = self.solMap.index(ds['libtype'])
            elif ds['libtype'] == 'rocblas':
                soltype = self.solMap.index(ds['libtype'])
            solds[key] = (soltype, int(ds['solidx']))
        self.solids = solds
        self.solfuncs = [
            self.apply_torch_mm,
            self.apply_hipb_mm,
            self.apply_rocb_mm,
            self.apply_skinny,
        ]

    @functools.lru_cache(maxsize=1024)
    def query_sol(self, m, n, k, bias, dtype, otype, scaleAB=False):
        if dtype == torch.float16 and k % 8 == 0:
            if n > 8 and 0 < m <= 4:
                return 3, 0
            elif n % 4 == 0 and m == 1 and k <= 8192:
                return 3, 1
        soltype, solidx = self.solids.get((m, n, k, bias, str(dtype), str(otype), scaleAB), (0, 0))
        logger.info(
            f'using {soltype=}, {solidx=} for {m=} {n=} {k=} {dtype=} {bias=}, {scaleAB=}')
        return soltype, solidx

    def apply_skinny(self, inp, weights, solidx, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
        import aiter as ops
        if solidx == 0:
            out = torch.empty(inp.shape[0],
                              weights.shape[0],
                              dtype=inp.dtype,
                              device='cuda')
            ops.wvSpltK(weights, inp, out, inp.shape[0], self.cu_count)
        elif solidx == 1:
            out = torch.empty(inp.shape[0],
                              weights.shape[0],
                              dtype=inp.dtype,
                              device='cuda')
            ops.LLMM1(weights, inp, out, 4)
        if bias is not None:
            return out + bias

    def apply_hipb_mm(self, inp, weights, solidx, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
        if otype is None:
            otype = inp.dtype
        return hipb_mm(inp, weights.t(), solidx,bias, otype, scale_a, scale_b, scale_c)

    def apply_rocb_mm(self, inp, weights, solidx, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
        assert scale_a != None and scale_b != None and scale_c != None, "scale_a, scale_b, scale_c must be None for rocblas"
        out = rocb_mm(inp, weights.t(), solidx)
        if bias is not None:
            out = out + bias
        return out

    def apply_torch_mm(self, inp, weights, solidx, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
        if (self.save_gemm == 1):
            m, k = inp.shape
            n = weights.shape[0]
            self.tuned_df = pd.concat([
                self.tuned_df,
                pd.DataFrame({
                    'M': [m],
                    'N': [n],
                    'K': [k],
                    'bias': [bias is not None],
                    'dtype': [inp.dtype],
                    'outdtype': [otype],
                    'scaleAB':[scale_a is not None or scale_b is not None],
                })
            ]).drop_duplicates()
            self.tuned_df.to_csv(self.untune_path, index=False)
        if inp.dtype == torch.float8_e4m3fnuz:
            if scale_a is None:
                scale_a = torch.ones(1, dtype=torch.float, device = inp.device)
            if scale_b is None:
                scale_b = torch.ones(1, dtype=torch.float, device = inp.device)
            
            return torch._scaled_mm(inp,
                                    weights.t(),
                                    out_dtype=otype,
                                    scale_a=scale_a,
                                    scale_b=scale_b,
                                    bias=bias)
        out = F.linear(inp, weights, bias)
        if otype is not None:
            out = out.to(otype)
        return out

    def mm(self, inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
        # F.Linear can take a 3 dimensional input. vllm
        # uses this for linear units. However, sampler
        # will use torch.matmul with 2 dimensions only
        if self.extensions_created == False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
            self.load_best_sols()
            self.create_ds()
        if inp.dim() == 3:
            inp_view = inp.view(-1, inp.size(-1))
            batched = True
        else:
            inp_view = inp
            batched = False
        m, k = inp_view.shape
        n = weights.shape[0]
        use_bias = bias is not None
        soltype, solidx = self.query_sol(m=m,
                                         n=n,
                                         k=k,
                                         bias=use_bias,
                                         dtype=inp.dtype,
                                         otype=otype,
                                         scaleAB=scale_a is not None or scale_b is not None)
        out = self.solfuncs[soltype](
            inp_view, weights, solidx, bias, otype, scale_a, scale_b, scale_c)
        if batched:
            out = out.view(inp.shape[0], inp.shape[1], weights.shape[0])
        return out


tgemm = TunedGemm()
