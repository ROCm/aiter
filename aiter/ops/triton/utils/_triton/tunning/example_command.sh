
python3 screen.py \
    64 8192 3584 7 \
    ut_a16w16_gemm.py \
    --m-range 4 8 16 32 \
    --n-range 16 32 \
    --k-range 128 \
    --overwrite

python3 screen.py \
    32 2112 7168 7 \
    ut_a8w8_gemm_blockscale.py \
    --m-range 4 8 16 32 \
    --n-range 16 32 \
    --k-range 128 \
    --overwrite

python3 screen.py \
    32 2112 7168 7 \
    ut_afp4wfp4_gemm_preshuffle.py \
    --m-range 4 32 \
    --n-range 16 32 \
    --k-range 256 \
    --overwrite