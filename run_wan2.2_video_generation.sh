mkdir -p results_Wan22_fav3_sage
torchrun --nproc_per_node=8 op_tests/sagev1_tests/run_wan.py \
    --task i2v \
    --height 720 \
    --width 1280 \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --img_file_path /app/Wan/i2v_input.JPG \
    --ulysses_degree 8 \
    --seed 42 \
    --num_frames 81 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_repetitions 1 \
    --num_inference_steps 40 \
    --use_torch_compile \
    --benchmark_output_directory results_Wan22_fav3_sage \
    --attention_type fav3_sage \
    # --save_inputs \
    # --max_captures 10