import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aiter.ops.attention
import run_asm_pa as m
import torch
from torch.profiler import profile, ProfilerActivity, schedule

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['benchmark', 'trace'], default='trace')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--num-kernels', type=int, default=450)
    p.add_argument('--num-replays', type=int, default=5)
    p.add_argument('--output', default='./libtorch_pa_trace.json')
    args = p.parse_args()

    bench = m.PABenchmark(batch=args.batch, num_kernels=args.num_kernels, num_replays=args.num_replays)

    if args.mode == 'benchmark':
        bench.benchmark()
    else:
        bench.warmup()
        bench.capture_graph()
        for _ in range(5): bench.replay()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=2, warmup=2, active=args.num_replays, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(args.output),
            record_shapes=True, with_stack=True,
        ) as prof:
            for _ in range(4 + args.num_replays):
                bench.replay()
                prof.step()
        print(f"\n Trace: {args.output}")
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == '__main__':
    main()
