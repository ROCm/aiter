export ENABLE_CK=0
export AITER_MOE_EXPERT_BALANCE=true
export AITER_LOG_MORE=1
export AITER_FORCE_GFX1250=1

# Overridable from the caller: HIP_VISIBLE_DEVICES=1 FLYDSL_DUMP_IR=0 bash my_code/run_gemm.sh
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export FLYDSL_DUMP_IR="${FLYDSL_DUMP_IR:-1}"
export FLYDSL_DUMP_DIR="${FLYDSL_DUMP_DIR:-./my_code/flydsl_dump}"


# python op_tests/test_flydsl_grouped_gemm_gfx1250.py   --scenario bench --data-format a8w4 --layout gugu   --experts 384 --tokens  4096 4096 4096 4096 4096 4096 4096 4096 4096 4096 --topk 6 --iters 100   --model-dim 7168 --inter-dim 768 --act silu --real-gemm --no-check-aot
python op_tests/test_flydsl_grouped_gemm_gfx1250.py   --scenario bench --data-format a8w4 --layout gugu   --experts 384 --tokens  4096 --topk 6 --iters 100   --model-dim 7168 --inter-dim 768 --act silu --real-gemm --no-check-aot

