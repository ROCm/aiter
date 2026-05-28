# source $GFX1250_MODEL_PATH/ffmlite_env.sh
export AITER_LOG_MORE=1
# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py -t 64 -dim 7168,256 -e 256 -k 8 -q 7 --no-flydsl-csv -hip 0,0 --grouped-gemm
# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py -t 64 -dim 7168,512 -e 256 -k 8 -q 7 --no-flydsl-csv -hip 0,0 --grouped-gemm
ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 64 -dim 7168,2048 -e 256 -k 8 -q 7 --no-flydsl-csv -hip 0,0 --grouped-gemm --no-legacy
#ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py -t 64 -dim 7168,256 -e 256 -k 8 -q 4 --no-flydsl-csv -hip 0,0 --grouped-gemm
#ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py -t 64 -dim 7168,512 -e 256 -k 8 -q 4 --no-flydsl-csv -hip 0,0 --grouped-gemm
#ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py -t 64 -dim 7168,2048 -e 256 -k 8 -q 4 --no-flydsl-csv -hip 0,0 --grouped-gemm
