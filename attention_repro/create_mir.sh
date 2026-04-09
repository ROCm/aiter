folder=v1_irs
mkdir -p $folder
bash run.sh
mv *.txt $folder
llc -mtriple=amdgcn -mcpu=gfx1250 $folder/unified_attention_2d_buf_2_remove_indirect_1_gluon_wpeu_1_num_warps_4_block_m_128_tile_size_128_block_size_128_head_size_128_sfl_0_llir.txt --print-before=machine-scheduler &> $folder/MIR.txt
python3 mir_cfg.py $folder/MIR.txt
# v1 -> no early exit
# v2 -> no early exit + loop assume
# v3 -> no early exit + loop assume + no pred
# v4 -> no early exit + loop constexpr
# v5 -> no early exit + loop constexpr + no BS
# v6 -> no early exit + loop constexpr + no sink
# v7 -> no early exit + loop constexpr + b. load sink
# v8 -> no early exit + b. load sink