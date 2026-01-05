#!/bin/bash


################################## Usage
# bash aiter/op_tests/sagev1_tests/test_WAN.sh


################################## Git relted
# cd /home/waqahmed/codes/aiter
# git checkout main && git fetch && git pull
# git submodule sync && git submodule update --init --recursive
# git checkout anguyenh/sageattn
# git pull && git submodule sync && git submodule update --init --recursive


################################## setup docker
# docker pull amdsiloai/pytorch-xdit:v25.13.1
# docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host -v /home/waqahmed/codes:/workspace -w /workspace amdsiloai/pytorch-xdit:v25.13.1 /bin/bash
# pip install moviepy
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local-dir /workspace/wan_models/Wan2.1-I2V-14B-720P-Diffusers --local-dir-use-symlinks False
# huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir /workspace/wan_models/Wan2.2-I2V-A14B-Diffusers --local-dir-use-symlinks False
# cd /workspace
# cp -r /app /workspace/
# we will now follow /app/Wan/RUN.md below


################################## copy required files
cd aiter/op_tests/sagev1_tests/wan_files
cp run.py /app/Wan/run.py
cp wan_utils.py /app/Wan/wan_utils.py
cp usp.py /app/external/xDiT/xfuser/model_executor/layers/usp.py
cp runtime_state.py /app/external/xDiT/xfuser/core/distributed/runtime_state.py


################################## run exp
model_name="Wan2.2-I2V-A14B-Diffusers"
res_dir_base=results/$model_name; model_path=/workspace/wan_models/$model_name
cd /workspace
k_list="default_fp8 sagev1 fav3_sage fav3_fp8"
for k in $k_list; do
    res_dir=${res_dir_base}_${k}
    rm -rf "$res_dir" # remove previous results (optional)
    mkdir -p "$res_dir"
    torchrun --nproc_per_node=8 /app/Wan/run.py \
        --use_fp8_attn \
        --kernel "$k" \
        --task i2v \
        --height 720 \
        --width 1280 \
        --model $model_path \
        --img_file_path /app/Wan/i2v_input.JPG \
        --ulysses_degree 8 \
        --seed 42 \
        --num_frames 81 \
        --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
        --num_repetitions 1 \
        --num_inference_steps 40 \
        --use_torch_compile \
        --benchmark_output_directory "$res_dir" 2>&1 | tee "$res_dir/logs.txt"
done


################################## run exp without fp8
model_name="Wan2.2-I2V-A14B-Diffusers"
res_dir=results/${model_name}_BF16; model_path=/workspace/wan_models/$model_name
cd /workspace
rm -rf "$res_dir" # remove previous results (optional)
mkdir -p "$res_dir"
torchrun --nproc_per_node=8 /app/Wan/run.py \
    --task i2v \
    --height 720 \
    --width 1280 \
    --model $model_path \
    --img_file_path /app/Wan/i2v_input.JPG \
    --ulysses_degree 8 \
    --seed 42 \
    --num_frames 81 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_repetitions 1 \
    --num_inference_steps 40 \
    --use_torch_compile \
    --benchmark_output_directory "$res_dir" 2>&1 | tee "$res_dir/logs.txt"


################################## run rendering script
# bash aiter/op_tests/sagev1_tests/wan_files/render_all_wan.sh --output Wan22_FP8_compare.mp4