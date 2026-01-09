#!/usr/bin/env python3
"""
Generate reference tensors from PyTorch for Qwen-Image substitution testing.

This script uses the OFFICIAL QwenImagePipeline and hooks into it to extract
intermediate tensors, guaranteeing exact compatibility.

Usage:
    python generate_reference_tensors.py --prompt "A serene mountain landscape" --seed 42

The tensors are saved to debug_tensors/pytorch/ in NumPy format.
"""

import argparse
import os
import sys
from pathlib import Path

# Enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch

# Add diffusers to path if needed
diffusers_path = Path(__file__).parent.parent.parent.parent / "diffusers" / "src"
if diffusers_path.exists():
    sys.path.insert(0, str(diffusers_path))

transformers_path = Path(__file__).parent.parent.parent.parent / "transformers" / "src"
if transformers_path.exists():
    sys.path.insert(0, str(transformers_path))


def save_tensor(name: str, tensor: torch.Tensor, output_dir: str):
    """Save a PyTorch tensor as NumPy .npy file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npy")
    arr = tensor.detach().cpu().float().numpy()
    np.save(path, arr)
    print(f"  Saved {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")


def tensor_stats(name: str, tensor: torch.Tensor):
    """Print tensor statistics."""
    t = tensor.detach().float()
    print(f"  {name}: shape={list(t.shape)}, mean={t.mean():.6f}, std={t.std():.6f}, "
          f"min={t.min():.6f}, max={t.max():.6f}")


# Global counter for debugging
_debug_attn_step = {"count": 0}


class DebugQwenDoubleStreamAttnProcessor:
    """Debug version that logs all intermediate values."""

    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        import torch.nn.functional as F
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        if encoder_hidden_states is None:
            raise ValueError("requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if _debug_attn_step["count"] == 0:
            print(f"  [DEBUG.ATTN] img_q_after_norm: shape={list(img_query.shape)}, "
                  f"mean={img_query.float().mean():.6f}, std={img_query.float().std():.6f}")
            print(f"  [DEBUG.ATTN] img_k_after_norm: shape={list(img_key.shape)}, "
                  f"mean={img_key.float().mean():.6f}, std={img_key.float().std():.6f}")
            print(f"  [DEBUG.ATTN] txt_q_after_norm: shape={list(txt_query.shape)}, "
                  f"mean={txt_query.float().mean():.6f}, std={txt_query.float().std():.6f}")
            print(f"  [DEBUG.ATTN] txt_k_after_norm: shape={list(txt_key.shape)}, "
                  f"mean={txt_key.float().mean():.6f}, std={txt_key.float().std():.6f}")

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        if _debug_attn_step["count"] == 0:
            print(f"  [DEBUG.ATTN] img_q_after_rope: shape={list(img_query.shape)}, "
                  f"mean={img_query.float().mean():.6f}, std={img_query.float().std():.6f}")
            print(f"  [DEBUG.ATTN] img_k_after_rope: shape={list(img_key.shape)}, "
                  f"mean={img_key.float().mean():.6f}, std={img_key.float().std():.6f}")
            print(f"  [DEBUG.ATTN] txt_q_after_rope: shape={list(txt_query.shape)}, "
                  f"mean={txt_query.float().mean():.6f}, std={txt_query.float().std():.6f}")
            print(f"  [DEBUG.ATTN] txt_k_after_rope: shape={list(txt_key.shape)}, "
                  f"mean={txt_key.float().mean():.6f}, std={txt_key.float().std():.6f}")

        # Concatenate for joint attention - Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        if _debug_attn_step["count"] == 0:
            print(f"  [DEBUG.ATTN] joint_q: shape={list(joint_query.shape)}, "
                  f"mean={joint_query.float().mean():.6f}, std={joint_query.float().std():.6f}")
            print(f"  [DEBUG.ATTN] joint_k: shape={list(joint_key.shape)}, "
                  f"mean={joint_key.float().mean():.6f}, std={joint_key.float().std():.6f}")
            print(f"  [DEBUG.ATTN] joint_v: shape={list(joint_value.shape)}, "
                  f"mean={joint_value.float().mean():.6f}, std={joint_value.float().std():.6f}")

        # DEBUG: Compute logits manually to compare with Rust
        if _debug_attn_step["count"] == 0:
            # Transpose for SDPA format: [B, S, H, D] -> [B, H, S, D]
            q_t = joint_query.permute(0, 2, 1, 3)
            k_t = joint_key.permute(0, 2, 1, 3)
            # Compute Q @ K^T
            logits = torch.matmul(q_t, k_t.transpose(-2, -1))
            scale = 1.0 / (128 ** 0.5)
            scaled_logits = logits * scale
            probs = torch.softmax(scaled_logits, dim=-1)
            # Entropy
            log_probs = torch.log(probs.clamp(min=1e-10))
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            print(f"  [DEBUG.SDPA] logits: mean={logits.float().mean():.4f}, std={logits.float().std():.4f}")
            print(f"  [DEBUG.SDPA] entropy={entropy:.4f}")

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        if _debug_attn_step["count"] == 0:
            print(f"  [DEBUG.ATTN] joint_attn_output: shape={list(joint_hidden_states.shape)}, "
                  f"mean={joint_hidden_states.float().mean():.6f}, std={joint_hidden_states.float().std():.6f}")

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        if _debug_attn_step["count"] == 0:
            print(f"  [DEBUG.ATTN] img_attn_pre_proj: shape={list(img_attn_output.shape)}, "
                  f"mean={img_attn_output.float().mean():.6f}, std={img_attn_output.float().std():.6f}")

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        _debug_attn_step["count"] += 1

        return img_attn_output, txt_attn_output


def main():
    parser = argparse.ArgumentParser(description="Generate reference tensors for Qwen-Image")
    parser.add_argument("--prompt", type=str, default="A serene mountain landscape",
                        help="Text prompt for generation")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt for CFG")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0,
                        help="True CFG scale (>1 enables classifier-free guidance)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="debug_tensors/pytorch",
                        help="Output directory for tensors")
    parser.add_argument("--save-image", type=str, default="pytorch_output.png",
                        help="Path to save generated image")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu, NOT mps - has issues)")
    parser.add_argument("--use-f32", action="store_true",
                        help="Use full F32 precision (default is BF16 mixed precision)")
    parser.add_argument("--use-f16", action="store_true",
                        help="Use F16 precision for lower memory usage")
    args = parser.parse_args()

    # Force CPU if MPS was selected - MPS has numerical issues with this model
    if args.device == "mps":
        print("  Note: MPS has numerical issues, using CPU instead")
        args.device = "cpu"

    device = torch.device(args.device)

    # Determine dtype based on flags and device capabilities
    # CPU: default to F32 (BF16 is emulated and very slow without AVX-512 BF16)
    # GPU: default to BF16 (fast and good quality)
    if args.use_f32:
        dtype = torch.float32
    elif args.use_f16:
        # F16 causes NaN in attention due to limited dynamic range, fall back to BF16
        print("  Warning: F16 causes NaN in attention, using BF16 instead")
        dtype = torch.bfloat16
    elif args.device == "cpu":
        dtype = torch.float32  # BF16 on CPU is emulated and slow
    else:
        dtype = torch.bfloat16  # GPU: use BF16 mixed precision

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Prompt: {args.prompt}")
    print(f"Negative prompt: {args.negative_prompt or '(none)'}")
    print(f"True CFG scale: {args.true_cfg_scale}")
    print(f"Output dir: {args.output_dir}")
    print()

    # =========================================================================
    # Load the OFFICIAL pipeline
    # =========================================================================
    print("[1/2] Loading official QwenImagePipeline...")

    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    # Match Candle's mixed precision: text encoder always runs in F32
    if dtype in (torch.bfloat16, torch.float16):
        pipe.text_encoder = pipe.text_encoder.to(torch.float32)
        print(f"  Text encoder set to F32 (mixed precision with {dtype})")

    print("  Pipeline loaded!")
    print()

    # =========================================================================
    # Storage for intermediate tensors
    # =========================================================================
    saved_tensors = {}

    # =========================================================================
    # Hook into the pipeline using callback
    # =========================================================================
    def save_step_tensors(pipe, step_idx, timestep, callback_kwargs):
        """Callback to save tensors at each denoising step.

        NOTE: callback_on_step_end fires AFTER the scheduler step, so latents
        here are the OUTPUT of the step, not the input.
        """
        latents = callback_kwargs["latents"]

        if step_idx == 0:
            # This is latents AFTER step 0 (scheduler has already run)
            save_tensor("packed_latents_after_step0", latents, args.output_dir)
            tensor_stats("packed_latents_after_step0", latents)

            # Also save unpacked version for Rust comparison
            latents_unpacked = pipe._unpack_latents(latents, args.height, args.width, pipe.vae_scale_factor)
            save_tensor("latents_after_step0", latents_unpacked, args.output_dir)
            tensor_stats("latents_after_step0", latents_unpacked)

        # Save for later
        saved_tensors["latents"] = latents
        saved_tensors["step"] = step_idx

        return callback_kwargs

    # =========================================================================
    # Run the pipeline
    # =========================================================================
    print("[2/2] Running pipeline with tensor extraction...")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # We need to manually extract some tensors before and during the call
    # First, let's encode the prompt to save those tensors
    print("\n  Encoding prompt...")

    # First, let's tokenize to save the input_ids for comparison
    # IMPORTANT: Use the same template as encode_prompt() to get accurate comparison!
    print("\n  Tokenizing prompt (with template)...")
    templated_prompt = pipe.prompt_template_encode.format(args.prompt)
    drop_idx = pipe.prompt_template_encode_start_idx
    print(f"  Template: '{pipe.prompt_template_encode[:50]}...'")
    print(f"  Drop tokens: {drop_idx}")

    text_inputs = pipe.tokenizer(
        templated_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024 + drop_idx,
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask_raw = text_inputs.attention_mask.to(device)

    save_tensor("input_ids", input_ids.float(), args.output_dir)  # Save as float for .npy
    save_tensor("attention_mask_raw", attention_mask_raw.float(), args.output_dir)
    print(f"  Templated input_ids: shape={list(input_ids.shape)}, num_tokens={input_ids.shape[1]}")
    print(f"  First 5 tokens: {input_ids[0, :5].tolist()}")
    print(f"  Last 5 tokens: {input_ids[0, -5:].tolist()}")
    print(f"  After dropping {drop_idx}: {input_ids.shape[1] - drop_idx} tokens remain")
    print(f"  attention_mask_raw: shape={list(attention_mask_raw.shape)}, sum={attention_mask_raw.sum().item()}")

    # Hook into text encoder to get intermediate values
    text_encoder_hooks = []
    captured_values = {}

    def capture_embed_tokens(module, args, output):
        """Capture token embeddings before transformer layers."""
        captured_values["embed_tokens_output"] = output.detach().clone()
        return output

    def capture_layer0(module, args, kwargs, output):
        """Capture output of first transformer layer."""
        captured_values["layer0_output"] = output[0].detach().clone()
        return output

    def capture_layer1(module, args, kwargs, output):
        """Capture output of second transformer layer."""
        captured_values["layer1_output"] = output[0].detach().clone()
        return output

    def capture_layer2(module, args, kwargs, output):
        """Capture output of third transformer layer."""
        captured_values["layer2_output"] = output[0].detach().clone()
        return output

    # Register hooks on text encoder
    if hasattr(pipe.text_encoder.model, 'embed_tokens'):
        hook = pipe.text_encoder.model.embed_tokens.register_forward_hook(capture_embed_tokens)
        text_encoder_hooks.append(hook)

    if hasattr(pipe.text_encoder.model, 'layers') and len(pipe.text_encoder.model.layers) >= 3:
        hook0 = pipe.text_encoder.model.layers[0].register_forward_hook(capture_layer0, with_kwargs=True)
        hook1 = pipe.text_encoder.model.layers[1].register_forward_hook(capture_layer1, with_kwargs=True)
        hook2 = pipe.text_encoder.model.layers[2].register_forward_hook(capture_layer2, with_kwargs=True)
        text_encoder_hooks.extend([hook0, hook1, hook2])

    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=1024,
    )

    # Remove hooks
    for hook in text_encoder_hooks:
        hook.remove()

    save_tensor("prompt_embeds", prompt_embeds, args.output_dir)
    save_tensor("prompt_mask", prompt_embeds_mask, args.output_dir)
    tensor_stats("prompt_embeds", prompt_embeds)

    # Save captured intermediate values
    if "embed_tokens_output" in captured_values:
        save_tensor("text_embed_tokens", captured_values["embed_tokens_output"], args.output_dir)
        tensor_stats("text_embed_tokens", captured_values["embed_tokens_output"])

    if "layer0_output" in captured_values:
        save_tensor("text_layer0_output", captured_values["layer0_output"], args.output_dir)
        tensor_stats("text_layer0_output", captured_values["layer0_output"])

    if "layer1_output" in captured_values:
        save_tensor("text_layer1_output", captured_values["layer1_output"], args.output_dir)
        tensor_stats("text_layer1_output", captured_values["layer1_output"])

    if "layer2_output" in captured_values:
        save_tensor("text_layer2_output", captured_values["layer2_output"], args.output_dir)
        tensor_stats("text_layer2_output", captured_values["layer2_output"])

    # Prepare latents manually so we can save them
    print("\n  Preparing latents...")
    latent_height = args.height // 8
    latent_width = args.width // 8
    num_channels = pipe.transformer.config.in_channels // 4

    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels,
        height=args.height,
        width=args.width,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    # Save initial packed latents
    save_tensor("initial_latents_packed", latents, args.output_dir)
    tensor_stats("initial_latents_packed", latents)

    # Unpack to save in spatial format too
    latents_unpacked = pipe._unpack_latents(latents, args.height, args.width, pipe.vae_scale_factor)
    save_tensor("initial_latents", latents_unpacked, args.output_dir)
    tensor_stats("initial_latents", latents_unpacked)

    # Now run the full pipeline
    print("\n  Running denoising loop...")

    # Reset generator for the actual run
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Hook transformer to capture noise_pred at step 0 AND internal states
    transformer_step_counter = {"count": 0}
    output_dir_for_hook = args.output_dir  # Capture in closure
    transformer_internal_captures = {}

    def capture_noise_pred(module, hook_args, kwargs, output):
        """Capture transformer output (noise_pred) at step 0."""
        if transformer_step_counter["count"] == 0:
            noise_pred = output[0] if isinstance(output, tuple) else output
            save_tensor("noise_pred_step0", noise_pred, output_dir_for_hook)
            tensor_stats("noise_pred_step0", noise_pred)
        transformer_step_counter["count"] += 1
        return output

    # ==========================================================================
    # Add detailed internal transformer hooks (matching Rust debug output)
    # ==========================================================================
    transformer_hooks = []

    def make_block_hook(block_idx):
        """Create a hook for a specific transformer block."""
        def hook(module, args, kwargs, output):
            # Only capture on first denoising step (transformer_step_counter tracks this)
            if transformer_step_counter["count"] == 0:
                # output is (encoder_hidden_states, hidden_states)
                txt_out, img_out = output
                key = f"after_block_{block_idx}"
                transformer_internal_captures[key] = img_out.detach().clone()
                print(f"  [TRANSFORMER] {key}: shape={list(img_out.shape)}, "
                      f"mean={img_out.float().mean():.6f}, std={img_out.float().std():.6f}, "
                      f"min={img_out.float().min():.6f}, max={img_out.float().max():.6f}")
            return output
        return hook

    def capture_img_in(module, args, output):
        """Capture output of img_in projection."""
        if transformer_step_counter["count"] == 0:
            transformer_internal_captures["after_img_in"] = output.detach().clone()
            print(f"  [TRANSFORMER] after_img_in: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_txt_norm(module, args, output):
        """Capture output of txt_norm (RMSNorm)."""
        if transformer_step_counter["count"] == 0:
            transformer_internal_captures["after_txt_norm"] = output.detach().clone()
            print(f"  [TRANSFORMER] after_txt_norm: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_txt_in(module, args, output):
        """Capture output of txt_in projection."""
        if transformer_step_counter["count"] == 0:
            transformer_internal_captures["after_txt_in"] = output.detach().clone()
            print(f"  [TRANSFORMER] after_txt_in: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_time_text_embed(module, args, output):
        """Capture temb (timestep embedding)."""
        if transformer_step_counter["count"] == 0:
            transformer_internal_captures["temb"] = output.detach().clone()
            print(f"  [TRANSFORMER] temb: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_norm_out(module, args, kwargs, output):
        """Capture output of norm_out."""
        if transformer_step_counter["count"] == 0:
            transformer_internal_captures["after_norm_out"] = output.detach().clone()
            print(f"  [TRANSFORMER] after_norm_out: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_proj_out(module, args, output):
        """Capture output of proj_out (final projection)."""
        if transformer_step_counter["count"] == 0:
            transformer_internal_captures["after_proj_out"] = output.detach().clone()
            print(f"  [TRANSFORMER] after_proj_out: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    # We need to hook into the transformer's forward to capture pos_embed output
    original_transformer_forward = pipe.transformer.forward

    def hooked_transformer_forward(
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        timestep=None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance=None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict=True,
    ):
        """Wrapper to capture pos_embed output and input tensors."""
        if transformer_step_counter["count"] == 0:
            # Capture inputs to transformer
            transformer_internal_captures["input_hidden_states"] = hidden_states.detach().clone()
            transformer_internal_captures["input_encoder_hidden_states"] = encoder_hidden_states.detach().clone()
            transformer_internal_captures["input_timestep"] = timestep.detach().clone()

            print(f"  [TRANSFORMER] input_hidden_states: shape={list(hidden_states.shape)}, "
                  f"mean={hidden_states.float().mean():.6f}, std={hidden_states.float().std():.6f}, "
                  f"min={hidden_states.float().min():.6f}, max={hidden_states.float().max():.6f}")
            print(f"  [TRANSFORMER] input_encoder_hidden_states: shape={list(encoder_hidden_states.shape)}, "
                  f"mean={encoder_hidden_states.float().mean():.6f}, std={encoder_hidden_states.float().std():.6f}, "
                  f"min={encoder_hidden_states.float().min():.6f}, max={encoder_hidden_states.float().max():.6f}")
            print(f"  [TRANSFORMER] input_timestep: shape={list(timestep.shape)}, "
                  f"mean={timestep.float().mean():.6f}, std={timestep.float().std():.6f}, "
                  f"min={timestep.float().min():.6f}, max={timestep.float().max():.6f}")
            # Capture the rotary embeddings
            image_rotary_emb = pipe.transformer.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
            vid_freqs, txt_freqs = image_rotary_emb

            # Convert complex to real for stats (freqs are complex numbers)
            vid_real = torch.view_as_real(vid_freqs)
            txt_real = torch.view_as_real(txt_freqs)

            transformer_internal_captures["rope_img_freqs"] = vid_real.detach().clone()
            transformer_internal_captures["rope_txt_freqs"] = txt_real.detach().clone()

            print(f"  [TRANSFORMER] rope_img_freqs: shape={list(vid_real.shape)}, "
                  f"mean={vid_real.float().mean():.6f}, std={vid_real.float().std():.6f}, "
                  f"min={vid_real.float().min():.6f}, max={vid_real.float().max():.6f}")
            print(f"  [TRANSFORMER] rope_txt_freqs: shape={list(txt_real.shape)}, "
                  f"mean={txt_real.float().mean():.6f}, std={txt_real.float().std():.6f}, "
                  f"min={txt_real.float().min():.6f}, max={txt_real.float().max():.6f}")

        # Call original forward
        return original_transformer_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            additional_t_cond=additional_t_cond,
            return_dict=return_dict,
        )

    pipe.transformer.forward = hooked_transformer_forward

    # Register hooks on transformer internals
    transformer_hooks.append(pipe.transformer.img_in.register_forward_hook(capture_img_in))
    transformer_hooks.append(pipe.transformer.txt_norm.register_forward_hook(capture_txt_norm))
    transformer_hooks.append(pipe.transformer.txt_in.register_forward_hook(capture_txt_in))
    transformer_hooks.append(pipe.transformer.time_text_embed.register_forward_hook(capture_time_text_embed))
    transformer_hooks.append(pipe.transformer.norm_out.register_forward_hook(capture_norm_out, with_kwargs=True))
    transformer_hooks.append(pipe.transformer.proj_out.register_forward_hook(capture_proj_out))

    # Hook specific blocks: 0, 1, 10, 30, 59 (matching Rust output)
    blocks_to_hook = [0, 1, 10, 30, 59]
    for block_idx in blocks_to_hook:
        if block_idx < len(pipe.transformer.transformer_blocks):
            hook = pipe.transformer.transformer_blocks[block_idx].register_forward_hook(
                make_block_hook(block_idx), with_kwargs=True
            )
            transformer_hooks.append(hook)

    # ==========================================================================
    # DETAILED BLOCK 0 HOOKS - to pinpoint where values explode
    # ==========================================================================
    block0 = pipe.transformer.transformer_blocks[0]
    block0_captures = {}

    # Replace block0's attention processor with debug version
    block0.attn.set_processor(DebugQwenDoubleStreamAttnProcessor())

    def capture_block0_img_mod(module, args, output):
        """Capture img_mod output (modulation params for image stream)."""
        if transformer_step_counter["count"] == 0:
            block0_captures["img_mod_output"] = output.detach().clone()
            print(f"  [BLOCK0] img_mod output: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_txt_mod(module, args, output):
        """Capture txt_mod output (modulation params for text stream)."""
        if transformer_step_counter["count"] == 0:
            block0_captures["txt_mod_output"] = output.detach().clone()
            print(f"  [BLOCK0] txt_mod output: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_img_norm1(module, args, output):
        """Capture img_norm1 output (LayerNorm before attention)."""
        if transformer_step_counter["count"] == 0:
            block0_captures["img_norm1_output"] = output.detach().clone()
            print(f"  [BLOCK0] img_norm1 output: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_txt_norm1(module, args, output):
        """Capture txt_norm1 output (LayerNorm before attention)."""
        if transformer_step_counter["count"] == 0:
            block0_captures["txt_norm1_output"] = output.detach().clone()
            print(f"  [BLOCK0] txt_norm1 output: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn(module, args, kwargs, output):
        """Capture attention output."""
        if transformer_step_counter["count"] == 0:
            img_attn_out, txt_attn_out = output
            block0_captures["img_attn_output"] = img_attn_out.detach().clone()
            block0_captures["txt_attn_output"] = txt_attn_out.detach().clone()
            print(f"  [BLOCK0] img_attn output: shape={list(img_attn_out.shape)}, "
                  f"mean={img_attn_out.float().mean():.6f}, std={img_attn_out.float().std():.6f}, "
                  f"min={img_attn_out.float().min():.6f}, max={img_attn_out.float().max():.6f}")
            print(f"  [BLOCK0] txt_attn output: shape={list(txt_attn_out.shape)}, "
                  f"mean={txt_attn_out.float().mean():.6f}, std={txt_attn_out.float().std():.6f}, "
                  f"min={txt_attn_out.float().min():.6f}, max={txt_attn_out.float().max():.6f}")
        return output

    def capture_block0_img_mlp(module, args, output):
        """Capture img_mlp output."""
        if transformer_step_counter["count"] == 0:
            block0_captures["img_mlp_output"] = output.detach().clone()
            print(f"  [BLOCK0] img_mlp output: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_txt_mlp(module, args, output):
        """Capture txt_mlp output."""
        if transformer_step_counter["count"] == 0:
            block0_captures["txt_mlp_output"] = output.detach().clone()
            print(f"  [BLOCK0] txt_mlp output: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    # Register block 0 internal hooks
    transformer_hooks.append(block0.img_mod.register_forward_hook(capture_block0_img_mod))
    transformer_hooks.append(block0.txt_mod.register_forward_hook(capture_block0_txt_mod))
    transformer_hooks.append(block0.img_norm1.register_forward_hook(capture_block0_img_norm1))
    transformer_hooks.append(block0.txt_norm1.register_forward_hook(capture_block0_txt_norm1))
    transformer_hooks.append(block0.attn.register_forward_hook(capture_block0_attn, with_kwargs=True))
    transformer_hooks.append(block0.img_mlp.register_forward_hook(capture_block0_img_mlp))
    transformer_hooks.append(block0.txt_mlp.register_forward_hook(capture_block0_txt_mlp))

    # Also hook into attention internals (QKV projections)
    def capture_block0_attn_to_q(module, args, output):
        """Capture Q projection for image stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["img_q_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] img Q proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn_to_k(module, args, output):
        """Capture K projection for image stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["img_k_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] img K proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn_to_v(module, args, output):
        """Capture V projection for image stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["img_v_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] img V proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn_add_q(module, args, output):
        """Capture Q projection for text stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["txt_q_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] txt Q proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn_add_k(module, args, output):
        """Capture K projection for text stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["txt_k_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] txt K proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn_add_v(module, args, output):
        """Capture V projection for text stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["txt_v_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] txt V proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn_to_out(module, args, output):
        """Capture output projection for image stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["img_out_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] img out proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    def capture_block0_attn_to_add_out(module, args, output):
        """Capture output projection for text stream."""
        if transformer_step_counter["count"] == 0:
            block0_captures["txt_out_proj"] = output.detach().clone()
            print(f"  [BLOCK0.ATTN] txt out proj: shape={list(output.shape)}, "
                  f"mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
                  f"min={output.float().min():.6f}, max={output.float().max():.6f}")
        return output

    # Register attention internal hooks
    transformer_hooks.append(block0.attn.to_q.register_forward_hook(capture_block0_attn_to_q))
    transformer_hooks.append(block0.attn.to_k.register_forward_hook(capture_block0_attn_to_k))
    transformer_hooks.append(block0.attn.to_v.register_forward_hook(capture_block0_attn_to_v))
    transformer_hooks.append(block0.attn.add_q_proj.register_forward_hook(capture_block0_attn_add_q))
    transformer_hooks.append(block0.attn.add_k_proj.register_forward_hook(capture_block0_attn_add_k))
    transformer_hooks.append(block0.attn.add_v_proj.register_forward_hook(capture_block0_attn_add_v))
    transformer_hooks.append(block0.attn.to_out[0].register_forward_hook(capture_block0_attn_to_out))
    transformer_hooks.append(block0.attn.to_add_out.register_forward_hook(capture_block0_attn_to_add_out))

    transformer_hook = pipe.transformer.register_forward_hook(capture_noise_pred, with_kwargs=True)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
        true_cfg_scale=args.true_cfg_scale,
        callback_on_step_end=save_step_tensors,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="latent",  # Get latents instead of image
    )

    # Clean up all hooks
    transformer_hook.remove()
    for hook in transformer_hooks:
        hook.remove()
    # Restore original forward
    pipe.transformer.forward = original_transformer_forward

    # Save captured internal tensors
    print("\n  Saving transformer internal tensors...")
    for name, tensor in transformer_internal_captures.items():
        save_tensor(f"transformer_{name}", tensor, args.output_dir)

    # Save block 0 internal tensors
    print("\n  Saving block 0 internal tensors...")
    for name, tensor in block0_captures.items():
        save_tensor(f"block0_{name}", tensor, args.output_dir)

    # Save final packed latents
    final_latents_packed = result.images
    save_tensor("final_latents_packed", final_latents_packed, args.output_dir)
    tensor_stats("final_latents_packed", final_latents_packed)

    # Unpack final latents
    final_latents = pipe._unpack_latents(final_latents_packed, args.height, args.width, pipe.vae_scale_factor)
    save_tensor("final_latents", final_latents, args.output_dir)
    tensor_stats("final_latents", final_latents)

    # Denormalize
    print("\n  Denormalizing latents...")
    final_latents = final_latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(final_latents.device, final_latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(final_latents.device, final_latents.dtype)
    )
    denormalized = final_latents / latents_std + latents_mean
    save_tensor("denormalized_latents", denormalized, args.output_dir)
    tensor_stats("denormalized_latents", denormalized)

    # VAE decode
    print("\n  Decoding with VAE...")
    with torch.no_grad():
        image = pipe.vae.decode(denormalized, return_dict=False)[0][:, :, 0]
    save_tensor("decoded_image", image, args.output_dir)
    tensor_stats("decoded_image", image)

    # Post-process and save
    image_out = image.squeeze(0)
    image_out = (image_out / 2 + 0.5).clamp(0, 1)
    image_out = image_out.permute(1, 2, 0).cpu().float().numpy()
    image_out = (image_out * 255).astype(np.uint8)

    from PIL import Image
    Image.fromarray(image_out).save(args.save_image)
    print(f"\n  Image saved to: {args.save_image}")

    print()
    print("=" * 60)
    print("Reference tensors saved to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
