import os
import time
import json
import torch
import numpy as np
from diffusers import WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image
from torch.profiler import profile, ProfilerActivity, record_function

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    initialize_runtime_state,
    is_dp_last_group,
)
from wan_utils import (
    time_pipe_func,
    get_pipe_kwargs,
    print_model_args,
    create_arg_parser,
    trace_handler,
    parallelize_transformer,
    set_timestep_embedding_dtype,
    quantize_linear_layers_to_fp8,
    TASK_FLOW_SHIFT,
    TASK_FPS,
)


def setup_attention(args):
    """
    Configure attention based on --attention_type.
    Returns True if monkey-patch was applied, False otherwise.
    """
    if args.attention_type == 'default':
        return False  # Use internal attention, no monkey-patch
    
    import xfuser.model_executor.layers.usp as usp_mod
    from utils import InputCaptureWrapper
    
    # Select base attention function with appropriate layout for BHSD input
    if args.attention_type == 'sagev1':
        from op_tests.sagev1_tests.core import sageattn
        # sageattn uses tensor_layout="HND" (BHSD) by default
        attn_fn = sageattn
        needs_permute = False
        
    elif args.attention_type == 'fav3_sage':
        from aiter.ops.triton.fav3_sage import fav3_sage_wrapper_func
        from functools import partial
        # Pass layout="bhsd" to avoid permutation
        attn_fn = partial(fav3_sage_wrapper_func, layout="bhsd")
        needs_permute = False
        
    elif args.attention_type == 'fav3_fp8':
        from aiter.ops.triton.mha_v3 import flash_attn_fp8_func
        # fav3_fp8 expects BSHD format, needs permutation
        attn_fn = flash_attn_fp8_func
        needs_permute = True
    
    # Optionally wrap with input capture
    if args.save_inputs:
        import os
        os.makedirs(args.benchmark_output_directory, exist_ok=True)
        max_caps = args.max_captures if args.max_captures > 0 else None
        attn_fn = InputCaptureWrapper(
            attn_fn, args.benchmark_output_directory, 
            name=args.attention_type, max_captures=max_caps
        )
    
    # Create wrapper matching usp._attention signature: (query, key, value, dropout_p, is_causal)
    # Returns just the output tensor (not a tuple)
    if needs_permute:
        # BHSD -> BSHD permutation for fav3_fp8
        def usp_attention_wrapper(query, key, value, dropout_p, is_causal):
            q = query.permute(0, 2, 1, 3).contiguous()
            k = key.permute(0, 2, 1, 3).contiguous()
            v = value.permute(0, 2, 1, 3).contiguous()
            output = attn_fn(q, k, v)
            return output.permute(0, 2, 1, 3)
    else:
        # No permute needed for sagev1/fav3_sage
        def usp_attention_wrapper(query, key, value, dropout_p, is_causal):
            return attn_fn(query, key, value)
    
    # Monkey-patch
    usp_mod._attention = usp_attention_wrapper
    return True


def main():
    args = create_arg_parser()
    setup_attention(args)
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    rank = os.environ.get("RANK")
    if rank == "0":
        print_model_args(engine_config, input_config, __file__)

    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    is_i2v_task = args.task == "i2v" or (args.task == "ti2v" and args.img_file_path is not None)

    if is_i2v_task:
        if not args.img_file_path: # Checking here prior to loading in the model
            raise ValueError("Please provide an input image path via --img_file_path. This may be a local path or a URL.")

    task_pipeline = WanImageToVideoPipeline if is_i2v_task else WanPipeline
    pipe = task_pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
    )
    initialize_runtime_state(pipe, engine_config)
    parallelize_transformer(pipe)
    is_last_process = is_dp_last_group()
    pipe = pipe.to(f"cuda:{local_rank}")
    pipe.scheduler.config.flow_shift = TASK_FLOW_SHIFT[args.task]

    if is_last_process:
        print(f"Running {args.task} task")

    if is_i2v_task:
        image = load_image(args.img_file_path)

        if args.force_output_size:
            # New logic: Force output to match defined dimensions by resizing and cropping
            # First, align target dimensions to model requirements
            mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            target_height = input_config.height // mod_value * mod_value
            target_width = input_config.width // mod_value * mod_value

            if is_last_process:
                print("Force output size mode enabled.")
                print(f"Input image resolution: {image.height}x{image.width}")
                print(f"Requested output resolution: {input_config.height}x{input_config.width}")
                print(f"Aligned output resolution (multiple of {mod_value}): {target_height}x{target_width}")

            # Step 1: Resize image maintaining aspect ratio so both dimensions >= target
            img_width, img_height = image.size

            # Calculate scale factor to ensure both dimensions are at least target size
            scale_width = target_width / img_width
            scale_height = target_height / img_height
            scale = max(scale_width, scale_height)  # Use max to ensure both dims are >= target

            # Resize with aspect ratio preserved
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height))

            if is_last_process:
                print(f"Resized image to: {new_height}x{new_width} (maintaining aspect ratio)")

            # Step 2: Crop from center to get exact target dimensions
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            image = image.crop((left, top, left + target_width, top + target_height))

            if is_last_process:
                print(f"Cropped from center to: {target_height}x{target_width}")
            
            # Step 3: Use the aligned dimensions directly (no additional resizing needed)
            height = target_height
            width = target_width
        else:
            # Original logic: Calculate dimensions based on input image aspect ratio
            # Calculate actual height and width based on input image and max area
            max_area = input_config.height * input_config.width
            aspect_ratio = image.height / image.width
            mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

            if is_last_process:
                print("Input width and height are used to calculate the max area for the video and the output video's aspect ratio is retained from the input image.")
                print(f"Input image resolution: {image.height}x{image.width}")
                print(f"Generating a video with resolution: {height}x{width}")
            image = image.resize((width, height))
    else:
        # Calculate actual height and width based on allowed multiples
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = input_config.height // mod_value * mod_value
        width = input_config.width // mod_value * mod_value
        if height != input_config.height or width != input_config.width:
            if is_last_process:
                print(f"Adjusting height and width to be multiples of {mod_value}. New dimensions: {height}x{width}")
            input_config.height = height
            input_config.width = width
        image = None

    if args.use_bf16_te_gemms:
        set_timestep_embedding_dtype(pipe, torch.bfloat16)
    if args.use_fp8_gemms:
        quantize_linear_layers_to_fp8(pipe.transformer.blocks)
        if pipe.transformer_2 is not None:
            quantize_linear_layers_to_fp8(pipe.transformer_2.blocks)

    if args.use_hybrid_fp8_attn:
        get_runtime_state().check_fp8_availability(use_fp8_attn=args.use_fp8_attn, use_hybrid_fp8_attn=args.use_hybrid_fp8_attn)
        guidance_scale = input_config.guidance_scale
        multiplier = 2 if guidance_scale > 1.0 else 1 # CFG is switched on in this case and double the transformers are called
        fp8_steps_threshold = 10 * multiplier # Number of initial and final steps to use bf16 attention for stability
        total_steps = input_config.num_inference_steps * multiplier # Total number of transformer calls during the denoising process
        # Create a boolean vector indicating which steps should use fp8 attention
        fp8_decision_vector = torch.tensor(
        [i >= fp8_steps_threshold and i < (total_steps - fp8_steps_threshold)
            for i in range(total_steps)], dtype=torch.bool)
        get_runtime_state().set_hybrid_attn_parameters(fp8_decision_vector)
    elif args.use_fp8_attn:
        get_runtime_state().check_fp8_availability(use_fp8_attn=args.use_fp8_attn, use_hybrid_fp8_attn=args.use_hybrid_fp8_attn)
        get_runtime_state().use_fp8_attn = True

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
        if pipe.transformer_2 is not None:
            pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="max-autotune-no-cudagraphs")

        # one step to warmup the torch compiler
        pipe(**get_pipe_kwargs(input_config, image))

    if args.profile_output != "":
        all_experiments_data = None # will not save timings from profiling runs
        # Ensure output directory exists
        destination = os.path.join(args.benchmark_output_directory, f"traces/{args.profile_output}/")
        if is_last_process:
            os.makedirs(destination, exist_ok=True)

        schedule = torch.profiler.schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
        )

        n_reps = args.profile_wait + args.profile_warmup + args.profile_active

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            schedule=schedule,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            on_trace_ready=trace_handler,
        ) as prof:
            prof.destination = destination
            for i in range(n_reps):
                with record_function("model_inference"):
                    output = pipe(**get_pipe_kwargs(input_config, image))
                torch.cuda.synchronize()
                prof.step()
    else:
        all_experiments_data = []
        for i in range(1, args.num_repetitions + 1):
            pipe_kwargs = get_pipe_kwargs(input_config, image)
            output, events = time_pipe_func(pipe=pipe, pipe_kwargs=pipe_kwargs)
            if is_last_process:
                pipe_elapsed_time = events["start"].elapsed_time(events["end"]) / 1000
                print(f"Iteration {i}")
                print(f"Pipe epoch time: {pipe_elapsed_time:.2f} sec")

                data = {"pipe_time": pipe_elapsed_time}
                last_inference_step_idx = pipe_kwargs['num_inference_steps'] - 1
                vae_elapsed_time = events[f"inference_step_end_{last_inference_step_idx}"].elapsed_time(events["end"]) / 1000
                print(f"VAE decode epoch time: {vae_elapsed_time:.2f} sec")
                data["vae_decode_time"] = vae_elapsed_time
                all_experiments_data.append(data)

    parallel_info = f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}"
    compile_info = f"compile{engine_args.use_torch_compile}"
    attention_info = f"attn_{args.attention_type}"
    if is_last_process:
        video_name = f"wan_result_{args.task}_{parallel_info}_{compile_info}_{attention_info}_{input_config.height}x{input_config.width}.mp4"
        destination = os.path.join(args.benchmark_output_directory, "results")
        os.makedirs(destination, exist_ok=True)
        video_filename = os.path.abspath(os.path.join(destination, video_name))
        export_to_video(output.frames[0], video_filename, fps=TASK_FPS[args.task])
        print(f"video saved to {video_filename}")
        print("#" * 10 + " Timings: " + "#" * 10)
        print(all_experiments_data)

        # Save data from runs if they exist
        if all_experiments_data is not None:
            timing_filename = os.path.abspath(os.path.join(destination, "timing.json"))
            with open(timing_filename, "w") as f:
                json.dump(all_experiments_data, f, indent=4)

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
