AITER_HOME=$(pip show aiter | grep Location | awk '{ print $2 }')

echo $AITER_HOME
#model=$1
# bf16
for M in 128 256 512 1024 2048 4096 8192 10240; do
    python3 ${AITER_HOME}/op_tests/op_benchmarks/triton/bench_moe.py --model "$model" -M $M -dtype bf16
done


