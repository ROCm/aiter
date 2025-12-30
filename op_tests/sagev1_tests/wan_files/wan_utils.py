import math
import torchao
from torchao.quantization.quantize_.common import KernelPreference
import os
import torch
import functools
import logging
from typing import Any, Dict, Optional, Union, Callable
import numpy as np

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanAttnProcessor
from xfuser.core.distributed import (
    get_world_group,
    get_sp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_runtime_state,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_FPS = {
    "i2v": 16,
    "t2v": 16,
    "ti2v": 24,
}

TASK_FLOW_SHIFT = {
    "i2v": 5,
    "t2v": 12,
    "ti2v": 5,
}

def maybe_transformer_2(transformer_2):
    if transformer_2 is not None:
        return functools.wraps(transformer_2.__class__.forward)
    else:
        return (lambda f:f)

def parallelize_transformer(pipe):
    transformer = pipe.transformer
    transformer_2 = pipe.transformer_2
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    @maybe_transformer_2(transformer_2)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        
        get_runtime_state().increment_step_counter()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            # We only reach this for Wan2.1, when doing cross attention with image embeddings
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        else:
            # Wan2.1 fails if we chunk encoder_hidden_states when cross attention is used. Should cross attention really be sharded?
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        # Part of sequence parallel: given the resolution, we may need to pad the sequence length to match this prior to chunking
        max_chunked_sequence_length = int(math.ceil(hidden_states.shape[1] / get_sequence_parallel_world_size())) * get_sequence_parallel_world_size()
        sequence_pad_amount = max_chunked_sequence_length - hidden_states.shape[1]
        hidden_states = torch.cat([
            hidden_states,
            torch.zeros(batch_size, sequence_pad_amount, hidden_states.shape[2], device=hidden_states.device, dtype=hidden_states.dtype)
        ], dim=1)
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        if ts_seq_len is not None: # (wan2.2 ti2v)
            temb = torch.cat([
                temb,
                torch.zeros(batch_size, sequence_pad_amount, temb.shape[2], device=temb.device, dtype=temb.dtype)
            ], dim=1)
            timestep_proj = torch.cat([
                timestep_proj,
                torch.zeros(batch_size, sequence_pad_amount, timestep_proj.shape[2], timestep_proj.shape[3], device=timestep_proj.device, dtype=timestep_proj.dtype)
            ], dim=1)
            temb = torch.chunk(temb, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            timestep_proj = torch.chunk(timestep_proj, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs, sequence_pad_amount):
            freqs = torch.cat([
                freqs,
                torch.zeros(1, sequence_pad_amount, freqs.shape[2], freqs.shape[3], device=freqs.device, dtype=freqs.dtype)
            ], dim=1)
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos, sequence_pad_amount)
        freqs_sin = get_rotary_emb_chunk(freqs_sin, sequence_pad_amount)
        rotary_emb = (freqs_cos, freqs_sin)


        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)

        # Removing excess padding to get back to original sequence length
        hidden_states = hidden_states[:, :math.prod([post_patch_num_frames, post_patch_height, post_patch_width]), :]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    new_forward_1 = new_forward.__get__(transformer)
    transformer.forward = new_forward_1

    for block in transformer.blocks:
        block.attn1.processor = xFuserWanAttnProcessor()
        block.attn2.processor = xFuserWanAttnProcessor()

    if transformer_2 is not None:
        new_forward_2 = new_forward.__get__(transformer_2)
        transformer_2.forward = new_forward_2

        for block in transformer_2.blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor()

def set_timestep_embedding_dtype(pipe, dtype: torch.dtype):
    logger.info(f"Setting timestep embedding dtype to {dtype}")
    pipe.transformer.condition_embedder.time_embedder = pipe.transformer.condition_embedder.time_embedder.to(dtype)
    if pipe.transformer_2 is not None:
        pipe.transformer_2.condition_embedder.time_embedder = pipe.transformer_2.condition_embedder.time_embedder.to(dtype)


def quantize_linear_layers_to_fp8(modules_to_quantize: list[torch.nn.Module]):
    """ Quantize all linear layers in the given modules """
    for module in modules_to_quantize:
        _quantize_module_linear_layers_to_fp8(module)

def _quantize_module_linear_layers_to_fp8(module: torch.nn.Module|list[torch.nn.Module], parent=None, name=None, link=None):
    """ Quantize all linear layers in the given module to FP8  """
    for child_name, child in module.named_children():
        _quantize_module_linear_layers_to_fp8(child, module, child_name)

    if isinstance(module, torch.nn.Linear):
        setattr(parent, name, module.to(torch.bfloat16))
        torchao.quantization.quantize_(
              module,
              config=torchao.quantization.Float8DynamicActivationFloat8WeightConfig(
                  granularity=torchao.quantization.PerTensor(),
                  set_inductor_config=False,
                  kernel_preference=KernelPreference.AUTO
            )
        )

def time_pipe_func(pipe: Callable, pipe_kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.cuda.Event], dict[str, torch.cuda.Event]]:
        """Run pipe_func and record timing events. Pipe events are recorded before and after running pipe_func.
        Inputs:
            pipe: pipeline to call
            pipe_kwargs: dictionary of keyword arguments to pass to the pipeline.

        Returns:
            A tuple of (output tensor, pipe events dictionary, task events dictionary).
        """
        events = {
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        for step_index in range(pipe_kwargs["num_inference_steps"]):
            events[f"inference_step_end_{step_index}"] = torch.cuda.Event(enable_timing=True)

        # Set up a callback function that records event timing after each inference/denoising step.
        # This may be used to infer VAE decode time as time between end of last inference step and end of pipe.
        def record_inference_step_end_time_cb(pipeline, step_index: int, timestep: int, callback_kwargs: dict):
            events[f"inference_step_end_{step_index}"].record()
            return callback_kwargs

        pipe_kwargs["callback_on_step_end"] = record_inference_step_end_time_cb

        events["start"].record()
        output = pipe(**pipe_kwargs)
        events["end"].record()
        torch.cuda.synchronize()

        return output, events

def get_pipe_kwargs(input_config, image = None) -> dict[str, Any]:
        kwargs = {
            "prompt": input_config.prompt,
            "num_inference_steps": input_config.num_inference_steps,
            "num_frames": input_config.num_frames,
            "guidance_scale": input_config.guidance_scale,
            "generator": torch.Generator(device="cuda").manual_seed(input_config.seed),
        }

        if image is not None:
            kwargs["image"] = image
            kwargs["height"] = image.height
            kwargs["width"] = image.width
        else:
            kwargs["height"] = input_config.height
            kwargs["width"] = input_config.width

        return kwargs

def print_model_args(engine_config, input_config, filename: str):
        """
        Prints all model arguments for logging.
        """

        def print_args(args):
            for key, val in vars(args).items():
                print(f"{key:>35} = {val}", flush=True)

        print(f"{'-'*20}PARAMETERS{'-'*20}")
        print_args(input_config)
        print_args(engine_config.model_config)
        print_args(engine_config.runtime_config)
        print_args(engine_config.parallel_config.dp_config)
        print_args(engine_config.parallel_config.sp_config)
        print_args(engine_config.parallel_config.pp_config)
        print_args(engine_config.parallel_config.tp_config)
        print(f"{'filename':>35} = {filename}")
        print(f"{'-'*20}PARAMETERS{'-'*20}")

def create_arg_parser():
        parser = FlexibleArgumentParser(description="xFuser Arguments")
        parser.add_argument(
            "--num_repetitions",
            type=int,
            default=5,
            help="The number of benchmark repetitions.",
        )
        parser.add_argument(
            "--rep_sleep_duration",
            type=int,
            default=None,
            help="The duration to sleep in between different pipe calls in seconds.",
        )
        parser.add_argument(
            "--profile_output",
            type=str,
            default="",
            help="When provided, run Pytorch profiler. "
            + "See --profile_wait, --profile_warmup and --profile_active for profiler specific warmup.",
        )
        parser.add_argument(
            "--profile_wait",
            type=int,
            default=2,
            help="wait argument for torch.profiler.schedule. Only used with --profile_output.",
        )
        parser.add_argument(
            "--profile_warmup",
            type=int,
            default=2,
            help="warmup argument for torch.profiler.schedule. Only used with --profile_output.",
        )
        parser.add_argument(
            "--profile_active",
            type=int,
            default=1,
            help="active argument for torch.profiler.schedule. Only used with --profile_output.",
        )
        parser.add_argument(
            "--warmup_calls",
            help="The number of full pipe calls to warmup the model.",
            type=int,
        )
        parser.add_argument(
            "--benchmark_output_directory",
            type=str,
            default=".",
            help="Benchmark output directory"
        )
        parser.add_argument(
             "--use_bf16_te_gemms",
             default=False,
             action="store_true",
             help="Use bfloat16 for time embedding gemms.",
        )
        parser.add_argument(
            "--use_fp8_gemms",
             default=False,
             action="store_true",
             help="Use FP8 for linear layer gemms.",
        )
        parser.add_argument(
            "--use_fp8_attn",
            action="store_true",
            help="Enable FP8 attention for faster inference."
        )
        parser.add_argument(
            "--kernel",
            type=str,
            default="default_fp8",
            choices=["default_fp8", "fav3_fp8", "fav3_sage", "sagev1"],
            help="The kernel to run.")
        parser.add_argument(
            "--use_hybrid_fp8_attn",
            action="store_true",
            help="Enable hybrid FP8 attention for faster inference and improved quality."
        )
        parser.add_argument(
            "--task",
            type=str,
            default="i2v",
            choices=["i2v", "t2v", "ti2v"],
            help="The task to run.")
        parser.add_argument(
            "--force_output_size",
            default=False,
            action="store_true",
            help="Force the output to match the defined dimensions by resizing and cropping the input image.",
        )
        return xFuserArgs.add_cli_args(parser).parse_args()


def trace_handler(prof):
    destination = prof.destination
    trace_output_file = os.path.join(destination, f"wan_traces_rank{get_world_group().rank}.json.gz")
    print(prof.key_averages(group_by_stack_n=25).table(sort_by="cuda_time_total", row_limit=15))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(trace_output_file)
