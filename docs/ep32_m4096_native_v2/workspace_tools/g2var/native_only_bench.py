"""Diagnostic only: original amd bench with early import and a pure forwarder."""
from functools import wraps
import importlib
import sys
sys.path.insert(0,'/root/workspace/workflow_34/moonshot_ep32_moe_v2/amd_moe_bench_share')
import bench_moe

def main():
    args=bench_moe.build_parser().parse_args()
    if args.command=='run':
        module=importlib.import_module('aiter.fused_moe')
        original=module.fused_moe
        @wraps(original)
        def forward(*p,**kw):
            return original(*p,**kw)
        module.fused_moe=forward
        bench_moe.run(args)
        return
    from trace_utils import parse_traces
    parse_traces(args)

if __name__=='__main__': main()
