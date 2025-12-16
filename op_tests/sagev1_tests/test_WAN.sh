#!/bin/bash


################################## setup docker
# docker pull amdsiloai/pytorch-xdit:v25.13.1
# docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host -v /home/waqahmed/codes:/workspace -w /workspace amdsiloai/pytorch-xdit:v25.13.1 /bin/bash
# pip install moviepy
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
# huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers
# cd /workspace
# we will now follow /app/Wan/RUN.md below


################################## run exp: Wan2.2_default
# The "if use_fp8_attn:" condirion in usp.py will be False in this case, so aiter/op_tests/sagev1_tests/usp.py does not required to be copied
res_dir=results_Wan22_default; model_type=Wan-AI/Wan2.2-I2V-A14B-Diffusers; mkdir -p $res_dir
torchrun --nproc_per_node=8 /app/Wan/run.py \
    --task i2v \
    --height 720 \
    --width 1280 \
    --model $model_type \
    --img_file_path /app/Wan/i2v_input.JPG \
    --ulysses_degree 8 \
    --seed 42 \
    --num_frames 81 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_repetitions 1 \
    --num_inference_steps 40 \
    --use_torch_compile \
    --benchmark_output_directory $res_dir 2>&1 | tee $res_dir/logs.txt


################################## run exp: Wan2.2_fp8_attn
# make manual code changes in _attention of usp.py to enable _aiter_fp8_attn_call
res_dir=results_Wan22_fp8_attn; model_type=Wan-AI/Wan2.2-I2V-A14B-Diffusers; mkdir -p $res_dir
cp aiter/op_tests/sagev1_tests/usp.py /app/external/xDiT/xfuser/model_executor/layers/usp.py 
torchrun --nproc_per_node=8 /app/Wan/run.py \
    --use_fp8_attn \
    --task i2v \
    --height 720 \
    --width 1280 \
    --model $model_type \
    --img_file_path /app/Wan/i2v_input.JPG \
    --ulysses_degree 8 \
    --seed 42 \
    --num_frames 81 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_repetitions 1 \
    --num_inference_steps 40 \
    --use_torch_compile \
    --benchmark_output_directory $res_dir 2>&1 | tee $res_dir/logs.txt


################################## run exp: Wan2.2_fp8_attn_v1
# make manual code changes in _attention of usp.py to enable _aiter_fp8_attn_call_v1
res_dir=results_Wan22_fp8_attn_v1; model_type=Wan-AI/Wan2.2-I2V-A14B-Diffusers; mkdir -p $res_dir
cp aiter/op_tests/sagev1_tests/usp.py /app/external/xDiT/xfuser/model_executor/layers/usp.py 
torchrun --nproc_per_node=8 /app/Wan/run.py \
    --use_fp8_attn \
    --task i2v \
    --height 720 \
    --width 1280 \
    --model $model_type \
    --img_file_path /app/Wan/i2v_input.JPG \
    --ulysses_degree 8 \
    --seed 42 \
    --num_frames 81 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_repetitions 1 \
    --num_inference_steps 40 \
    --use_torch_compile \
    --benchmark_output_directory $res_dir 2>&1 | tee $res_dir/logs.txt


################################## run exp: Wan2.2_fp8_attn_v2
# make manual code changes in _attention of usp.py to enable _aiter_fp8_attn_call_v2 
res_dir=results_Wan22_fp8_attn_v2; model_type=Wan-AI/Wan2.2-I2V-A14B-Diffusers; mkdir -p $res_dir
cp aiter/op_tests/sagev1_tests/usp.py /app/external/xDiT/xfuser/model_executor/layers/usp.py 
torchrun --nproc_per_node=8 /app/Wan/run.py \
    --use_fp8_attn \
    --task i2v \
    --height 720 \
    --width 1280 \
    --model $model_type \
    --img_file_path /app/Wan/i2v_input.JPG \
    --ulysses_degree 8 \
    --seed 42 \
    --num_frames 81 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_repetitions 1 \
    --num_inference_steps 40 \
    --use_torch_compile \
    --benchmark_output_directory $res_dir 2>&1 | tee $res_dir/logs.txt


################################## run rendering script
bash aiter/op_tests/sagev1_tests/render_all_WAN.sh --output Wan22_FP8_compare.mp4