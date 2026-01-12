
# example 1:
#   Tunning for A16W16 GEMM using default BLOCK_SIZE ranges using GPU 0
#   (see screen.py for deafult range)
python3 screen.py \
    64 8192 3584 0 \
    ut_a16w16_gemm.py \
    --overwrite
# run view-screen.py to view results and generate JSON files
python view-screen.py ut_a16w16_gemm.py --n-list 8192 --k-list 3584

# example 2:
#   Background tunning for A8W8 GEMM blockscale using specific BLOCK_SIZE_K ranges using GPU 0 ~ 6
#   because A8W8 blockscale gemm requires only BLOCK_SIZE_K=128
N=2112
K=7168
for M_G in "8 0" "16 1" "32 2" "64 3" "128 4" "256 5" "8192 6"; do
    set -- $M_G
    M=$1
    G=$2
    nohup python3 screen.py $M 2112 7168 0 ut_a8w8_gemm_blockscale.py --k-range 128 --overwrite > output-M=$M-N=$N-K=$K-G=$G.out &
done
# run view-screen.py to view results and generate JSON files
python view-screen.py ut_a8w8_gemm_blockscale.py --n-list 2112 --k-list 7168

# example 3:
#   Tunning for AFP4WFP4 GEMM using specific BLOCK_SIZE_M ranges to meet BLOCK_SIZE_M requirements:
#       BLOCK_SIZE_M < 32 for M < 32
#       BLOCK_SIZE_M >= 32 for M >= 32
python3 screen.py \
    64 2112 7168 0 \
    ut_afp4wfp4_gemm_preshuffle.py \
    --m-range 32 64 \
    --n-range 16 32 64 128 \
    --k-range 256 512 1024 \
    --overwrite
# run view-screen.py to view results and generate JSON files
python view-screen.py ut_afp4wfp4_gemm_preshuffle.py --n-list 2112 --k-list 7168
