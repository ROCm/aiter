import os
from pathlib import Path
import functools
import pandas as pd
import torch
import torch.nn.functional as F
from ater import hipb_create_extension, hipb_mm, getHipblasltKernelName
from ater import rocb_create_extension, rocb_mm
from ater import logger

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
                columns=['M', 'N', 'K', 'bias', 'dtype'])
        else:
            self.tuned_df = None

    def load_best_sols(self):
        if self.tune_path is not None and Path(self.tune_path).is_file():
            self.bestsols = pd.read_csv(self.tune_path)
            if len(self.bestsols) > 0 and 'kernelName' in self.bestsols.columns:
                hipblasltKernelNames = self.bestsols.apply(
                    lambda s: getHipblasltKernelName(s.solidx) if s.libtype == 'hipblaslt' else "", axis=1)
                assert hipblasltKernelNames.equals(
                    self.bestsols['kernelName'].fillna("")), "error: gradlib tune gemm not match the current environment, need re-tune!!!"

    def create_ds(self):
        df: pd.DataFrame = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            key = (ds['M'], ds['N'], ds['K'], ds['bias'], ds['dtype'])
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
    def query_sol(self, m, n, k, bias, dtype):
        if dtype == torch.float16 and k % 8 == 0:
            if n > 8 and 0 < m <= 4:
                return 3, 0
            elif n % 4 == 0 and m == 1 and k <= 8192:
                return 3, 1
        soltype, solidx = self.solids.get((m, n, k, bias, str(dtype)), (0, 0))
        logger.info(
            f'using {soltype=}, {solidx=} for {m=} {n=} {k=} {dtype=} {bias=}')
        return soltype, solidx

    def apply_skinny(self, inp, weights, solidx, bias=None, otype=None):
        import ater as ops
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

    def apply_hipb_mm(self, inp, weights, solidx, bias=None, otype=None):
        if otype is None:
            otype = inp.dtype
        return hipb_mm(inp, weights.t(), solidx, bias=bias, out_dtype=otype)

    def apply_rocb_mm(self, inp, weights, solidx, bias=None, otype=None):
        out = rocb_mm(inp, weights.t(), solidx)
        if bias is not None:
            out = out + bias
        return out

    def apply_torch_mm(self, inp, weights, solidx, bias=None, otype=None):
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
                })
            ]).drop_duplicates()
            self.tuned_df.to_csv(self.untune_path, index=False)
        return F.linear(inp, weights, bias)

    def mm(self, inp, weights, bias=None, otype=None):
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
                                         dtype=inp.dtype)
        out = self.solfuncs[soltype](
            inp_view, weights, solidx, bias=bias, otype=otype)
        if batched:
            out = out.view(inp.shape[0], inp.shape[1], weights.shape[0])
        return out


tgemm = TunedGemm()
