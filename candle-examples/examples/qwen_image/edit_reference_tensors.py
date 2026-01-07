#!/usr/bin/env python3
"""
Generate reference tensors from PyTorch for Qwen-Image EDIT pipeline debugging.

This script uses the OFFICIAL QwenImageEditPipeline and hooks into it to extract
intermediate tensors, guaranteeing exact compatibility with the Rust implementation.

Usage:
    uv run candle-examples/examples/qwen_image/edit_reference_tensors.py \
        --input-image input.png \
        --prompt "Change the sky to sunset colors" \
        --seed 42

The tensors are saved to debug_tensors/pytorch_edit/ in NumPy format.
"""

import argparse
import os
import sys
from pathlib import Path

# Enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from PIL import Image

# Add diffusers and transformers to path if needed
diffusers_path = Path(__file__).parent.parent.parent.parent / "diffusers" / "src"
if diffusers_path.exists():
    sys.path.insert(0, str(diffusers_path))

transformers_path = Path(__file__).parent.parent.parent.parent / "transformers" / "src"
if transformers_path.exists():
    sys.path.insert(0, str(transformers_path))


def save_tensor(name: str, tensor: torch.Tensor, output_dir: str):
    """Save a PyTorch tensor as NumPy .npy file.

    Saves tensors in their native format without transposing.
    - Noise latents: [B, T, C, H, W] (diffusers native format)
    - VAE outputs: [B, C, T, H, W] (VAE native format)
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npy")
    arr = tensor.detach().cpu().float().numpy()
    print(f"  Saved {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")
    np.save(path, arr)


def tensor_stats(name: str, tensor: torch.Tensor):
    """Print tensor statistics."""
    t = tensor.detach().float()
    print(f"  {name}: shape={list(t.shape)}, mean={t.mean():.6f}, std={t.std():.6f}, "
          f"min={t.min():.6f}, max={t.max():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Generate reference tensors for Qwen-Image Edit")
    parser.add_argument("--input-image", type=str, required=True,
                        help="Input image to edit")
    parser.add_argument("--prompt", type=str, default="Make the sky more colorful",
                        help="Edit instruction")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality",
                        help="Negative prompt for CFG")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0,
                        help="True CFG scale (>1 enables classifier-free guidance)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="debug_tensors/pytorch_edit",
                        help="Output directory for tensors")
    parser.add_argument("--save-image", type=str, default="pytorch_edit_output.png",
                        help="Path to save generated image")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu, NOT mps - has issues)")
    parser.add_argument("--max-resolution", type=int, default=512,
                        help="Maximum resolution for the longer side")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen-Image-Edit-2511",
                        help="HuggingFace model ID for the edit pipeline")
    args = parser.parse_args()

    # Force CPU if MPS was selected - MPS has numerical issues with this model
    if args.device == "mps":
        print("  Note: MPS has numerical issues, using CPU instead")
        args.device = "cpu"

    device = torch.device(args.device)
    dtype = torch.float32  # Use float32 for consistency

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Model: {args.model_id}")
    print(f"Input image: {args.input_image}")
    print(f"Prompt: {args.prompt}")
    print(f"Negative prompt: {args.negative_prompt or '(none)'}")
    print(f"True CFG scale: {args.true_cfg_scale}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Load input image
    input_image = Image.open(args.input_image).convert("RGB")
    orig_width, orig_height = input_image.size
    print(f"Original image size: {orig_width}x{orig_height}")

    # =========================================================================
    # Load the OFFICIAL edit pipeline
    # =========================================================================
    print("\n[1/3] Loading official QwenImageEditPipeline...")

    from diffusers import QwenImageEditPipeline

    pipe = QwenImageEditPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    print("  Pipeline loaded!")
    print(f"  VAE scale factor: {pipe.vae_scale_factor}")
    print(f"  Latent channels: {pipe.latent_channels}")
    print()

    # =========================================================================
    # Storage for intermediate tensors
    # =========================================================================
    saved_tensors = {}
    transformer_step_counter = {"count": 0}

    # =========================================================================
    # Calculate dimensions (matching pipeline behavior)
    # =========================================================================
    import math
    target_area = 512 * 512
    ratio = orig_width / orig_height
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    width, height = int(width), int(height)

    print(f"Target dimensions: {width}x{height}")

    # Align to VAE scale factor * 2
    multiple_of = pipe.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    print(f"Aligned dimensions: {width}x{height}")

    # =========================================================================
    # Step 1: Preprocess image and encode with VAE
    # =========================================================================
    print("\n[2/3] Preprocessing and encoding image...")

    # Resize image for VAE
    resized_image = input_image.resize((width, height), Image.LANCZOS)

    # Preprocess for VAE (same as pipeline does)
    image_tensor = pipe.image_processor.preprocess(resized_image, height, width)
    image_tensor = image_tensor.unsqueeze(2)  # Add temporal dim: [B, C, 1, H, W]
    image_tensor = image_tensor.to(device=device, dtype=dtype)

    save_tensor("vae_input", image_tensor, args.output_dir)
    tensor_stats("vae_input", image_tensor)

    # Encode with VAE - with intermediate tensor capture
    print("\n  === VAE Encoder Debug (intermediate tensors) ===")

    def debug_vae_encoder(vae, x):
        """Run encoder with intermediate tensor capture."""
        encoder = vae.encoder

        tensor_stats("encoder_input", x)
        save_tensor("vae_encoder_input", x, args.output_dir)

        # conv_in
        x = encoder.conv_in(x)
        tensor_stats("after_conv_in", x)
        save_tensor("vae_after_conv_in", x, args.output_dir)

        # down_blocks
        for i, block in enumerate(encoder.down_blocks):
            x = block(x)
            tensor_stats(f"after_down_block_{i}", x)
            save_tensor(f"vae_after_down_block_{i}", x, args.output_dir)

        # mid_block
        x = encoder.mid_block(x)
        tensor_stats("after_mid_block", x)
        save_tensor("vae_after_mid_block", x, args.output_dir)

        # norm_out + activation + conv_out
        x = encoder.norm_out(x)
        tensor_stats("after_norm_out", x)
        save_tensor("vae_after_norm_out", x, args.output_dir)

        x = torch.nn.functional.silu(x)
        tensor_stats("after_silu", x)
        save_tensor("vae_after_silu", x, args.output_dir)

        x = encoder.conv_out(x)
        tensor_stats("after_conv_out (encoder output)", x)
        save_tensor("vae_after_conv_out", x, args.output_dir)

        return x

    with torch.no_grad():
        # Run encoder with debug
        encoder_output = debug_vae_encoder(pipe.vae, image_tensor)

        # Run quant_conv
        h = pipe.vae.quant_conv(encoder_output)
        tensor_stats("after_quant_conv", h)
        save_tensor("vae_after_quant_conv", h, args.output_dir)

        # Get latent distribution
        mean, logvar = torch.chunk(h, 2, dim=1)
        image_latents_raw = mean  # mode() returns mean

    save_tensor("image_latents_raw", image_latents_raw, args.output_dir)
    tensor_stats("image_latents_raw", image_latents_raw)

    # Normalize latents (same as pipeline's _encode_vae_image)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.latent_channels, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.latent_channels, 1, 1, 1)
        .to(device, dtype)
    )
    image_latents_normalized = (image_latents_raw - latents_mean) / latents_std

    save_tensor("image_latents_normalized", image_latents_normalized, args.output_dir)
    tensor_stats("image_latents_normalized", image_latents_normalized)

    # Pack image latents
    batch_size = 1
    num_channels = pipe.latent_channels
    latent_height = image_latents_normalized.shape[3]
    latent_width = image_latents_normalized.shape[4]

    packed_image_latents = pipe._pack_latents(
        image_latents_normalized, batch_size, num_channels, latent_height, latent_width
    )

    save_tensor("packed_image_latents", packed_image_latents, args.output_dir)
    tensor_stats("packed_image_latents", packed_image_latents)

    # =========================================================================
    # Step 2: Encode prompt with vision
    # =========================================================================
    print("\n  Encoding prompt with vision...")

    # Debug: Check merge configuration and tokenization
    print("\n  [DEBUG] Analyzing vision/text tokenization...")
    print(f"  [DEBUG] image_processor.merge_size = {pipe.processor.image_processor.merge_size}")

    # After encoding, check the input shape
    model_inputs = pipe.processor(text=[args.prompt], images=resized_image, return_tensors="pt")
    print(f"  [DEBUG] input_ids shape: {model_inputs.input_ids.shape}")
    print(f"  [DEBUG] image_grid_thw: {model_inputs.image_grid_thw}")

    # Count image tokens (151655 is the <|image_pad|> token ID)
    num_image_tokens_in_ids = (model_inputs.input_ids == 151655).sum().item()
    print(f"  [DEBUG] Number of image tokens (151655) in input_ids: {num_image_tokens_in_ids}")

    # Calculate expected tokens
    if hasattr(model_inputs, 'image_grid_thw') and model_inputs.image_grid_thw is not None:
        grid = model_inputs.image_grid_thw[0]  # First image
        total_patches = grid.prod().item()
        merge_length = pipe.processor.image_processor.merge_size ** 2
        expected_tokens = total_patches // merge_length
        print(f"  [DEBUG] Grid: T={grid[0].item()}, H={grid[1].item()}, W={grid[2].item()}")
        print(f"  [DEBUG] Total patches: {total_patches}, merge_length: {merge_length}")
        print(f"  [DEBUG] Expected image tokens after merge: {expected_tokens}")

    # =========================================================================
    # Extract vision embeddings separately (before encode_prompt merges them)
    # =========================================================================
    print("\n  Extracting vision embeddings...")

    vision_captures = {}

    def capture_vision_output(module, args, output):
        """Capture the vision encoder output."""
        vision_captures["vision_embeds"] = output.detach().clone()
        return output

    # The vision model is at pipe.text_encoder.model.visual
    vision_hook = pipe.text_encoder.model.visual.register_forward_hook(capture_vision_output)

    # Run a dummy encode_prompt to trigger vision encoding (we'll capture the embeddings)
    # Use the same image that will be used for the actual prompt encoding
    _ = pipe.encode_prompt(
        prompt=args.prompt,
        image=resized_image,
        device=device,
        num_images_per_prompt=1,
    )

    vision_hook.remove()

    if "vision_embeds" in vision_captures:
        vision_embeds = vision_captures["vision_embeds"]
        # Vision embeds shape is [num_tokens, hidden_dim] after processing
        save_tensor("vision_embeds", vision_embeds, args.output_dir)
        tensor_stats("vision_embeds", vision_embeds)
        print(f"  Vision embeddings captured: {list(vision_embeds.shape)}")
    else:
        print("  WARNING: Failed to capture vision embeddings!")

    # Hook to capture intermediate text encoder outputs
    text_encoder_captures = {}

    def capture_hidden_states(module, args, kwargs, output):
        """Capture the final hidden states."""
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            text_encoder_captures["hidden_states_last"] = output.hidden_states[-1].detach().clone()
        return output

    hook = pipe.text_encoder.register_forward_hook(capture_hidden_states, with_kwargs=True)

    # Encode positive prompt
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=args.prompt,
        image=resized_image,
        device=device,
        num_images_per_prompt=1,
    )

    hook.remove()

    save_tensor("prompt_embeds", prompt_embeds, args.output_dir)
    save_tensor("prompt_mask", prompt_mask.float(), args.output_dir)
    tensor_stats("prompt_embeds", prompt_embeds)
    print(f"  Prompt sequence length: {prompt_embeds.shape[1]}")

    if "hidden_states_last" in text_encoder_captures:
        save_tensor("text_hidden_states_last", text_encoder_captures["hidden_states_last"], args.output_dir)
        tensor_stats("text_hidden_states_last", text_encoder_captures["hidden_states_last"])

    # Encode negative prompt (always encode for debugging, even if CFG not used)
    do_true_cfg = args.true_cfg_scale > 1 and args.negative_prompt
    if args.negative_prompt:
        neg_prompt_embeds, neg_prompt_mask = pipe.encode_prompt(
            prompt=args.negative_prompt,
            image=resized_image,
            device=device,
            num_images_per_prompt=1,
        )
        save_tensor("negative_prompt_embeds", neg_prompt_embeds, args.output_dir)
        save_tensor("negative_prompt_mask", neg_prompt_mask.float(), args.output_dir)
        tensor_stats("negative_prompt_embeds", neg_prompt_embeds)
        tensor_stats("negative_prompt_mask", neg_prompt_mask.float())
        print(f"  Negative prompt sequence length: {neg_prompt_embeds.shape[1]}")

    # =========================================================================
    # Step 3: Prepare noise latents
    # =========================================================================
    print("\n  Preparing noise latents...")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Create noise latents (same shape as image latents)
    noise_shape = (batch_size, 1, num_channels, latent_height, latent_width)
    noise_latents = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)

    save_tensor("noise_latents_unpacked", noise_latents, args.output_dir)
    tensor_stats("noise_latents_unpacked", noise_latents)

    packed_noise = pipe._pack_latents(noise_latents, batch_size, num_channels, latent_height, latent_width)

    save_tensor("packed_noise_latents", packed_noise, args.output_dir)
    tensor_stats("packed_noise_latents", packed_noise)

    # =========================================================================
    # Step 4: Run the full pipeline with hooks
    # =========================================================================
    print("\n[3/3] Running pipeline with tensor extraction...")

    # =========================================================================
    # MONKEY-PATCH: Force pipeline to use 512x512 for source image encoding
    # By default, the pipeline uses calculate_dimensions(1024*1024, ...) which
    # gives 1024x1024 for square images. We override this to match Rust.
    # =========================================================================
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit as edit_pipeline_module

    def _patched_calculate_dimensions(target_area, ratio):
        """Force 512x512 encoding dimensions (matching Rust implementation)."""
        # Use 512*512 instead of 1024*1024
        patched_area = 512 * 512
        width = math.sqrt(patched_area * ratio)
        height_val = width / ratio
        width = round(width / 32) * 32
        height_val = round(height_val / 32) * 32
        return int(width), int(height_val), None

    edit_pipeline_module.calculate_dimensions = _patched_calculate_dimensions
    print("  [PATCH] Overriding calculate_dimensions to use 512x512 (matching Rust)")

    # Reset generator
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Hook for transformer inputs/outputs
    # With true CFG, transformer is called TWICE per step: positive then negative
    transformer_captures = {}
    transformer_call_counter = {"count": 0}  # Counts calls within current step

    def capture_transformer_io(module, hook_args, kwargs, output):
        """Capture transformer input and output at step 0."""
        step = transformer_step_counter["count"]
        call_in_step = transformer_call_counter["count"]

        # Only capture at step 0
        if step == 0:
            hidden_states = kwargs.get("hidden_states", hook_args[0] if hook_args else None)
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            timestep = kwargs.get("timestep")
            noise_pred = output[0] if isinstance(output, tuple) else output

            # Extract only the noise prediction part (first half of sequence)
            noise_seq_len = noise_pred.shape[1] // 2  # noise + image -> just noise
            noise_pred_extracted = noise_pred[:, :noise_seq_len, :]

            if call_in_step == 0:
                # First call = positive/conditional prediction
                if hidden_states is not None:
                    transformer_captures["input_hidden_states"] = hidden_states.detach().clone()
                    print(f"  [TRANSFORMER] input_hidden_states: shape={list(hidden_states.shape)}, "
                          f"mean={hidden_states.float().mean():.6f}, std={hidden_states.float().std():.6f}")
                if encoder_hidden_states is not None:
                    transformer_captures["input_encoder_hidden_states"] = encoder_hidden_states.detach().clone()
                    print(f"  [TRANSFORMER] input_encoder_hidden_states: shape={list(encoder_hidden_states.shape)}, "
                          f"mean={encoder_hidden_states.float().mean():.6f}, std={encoder_hidden_states.float().std():.6f}")
                if timestep is not None:
                    transformer_captures["input_timestep"] = timestep.detach().clone()
                    print(f"  [TRANSFORMER] input_timestep: {timestep.item():.6f}")

                transformer_captures["noise_pred_full"] = noise_pred.detach().clone()
                print(f"  [TRANSFORMER] noise_pred_full: shape={list(noise_pred.shape)}, "
                      f"mean={noise_pred.float().mean():.6f}, std={noise_pred.float().std():.6f}")

                # Save positive prediction (extracted noise part only)
                transformer_captures["noise_pred_pos_step0"] = noise_pred_extracted.detach().clone()
                print(f"  [TRANSFORMER] noise_pred_pos_step0: shape={list(noise_pred_extracted.shape)}, "
                      f"mean={noise_pred_extracted.float().mean():.6f}, std={noise_pred_extracted.float().std():.6f}")

            elif call_in_step == 1 and do_true_cfg:
                # Second call = negative/unconditional prediction
                transformer_captures["noise_pred_neg_step0"] = noise_pred_extracted.detach().clone()
                print(f"  [TRANSFORMER] noise_pred_neg_step0: shape={list(noise_pred_extracted.shape)}, "
                      f"mean={noise_pred_extracted.float().mean():.6f}, std={noise_pred_extracted.float().std():.6f}")

                # Compute guided prediction (CFG)
                pos_pred = transformer_captures["noise_pred_pos_step0"]
                neg_pred = noise_pred_extracted
                # CFG formula: neg + scale * (pos - neg)
                guided = neg_pred + args.true_cfg_scale * (pos_pred - neg_pred)

                # Apply norm rescaling (matching pipeline)
                cond_norm = torch.norm(pos_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(guided, dim=-1, keepdim=True)
                guided = guided * (cond_norm / noise_norm)

                transformer_captures["guided_pred_step0"] = guided.detach().clone()
                print(f"  [TRANSFORMER] guided_pred_step0: shape={list(guided.shape)}, "
                      f"mean={guided.float().mean():.6f}, std={guided.float().std():.6f}")

        transformer_call_counter["count"] += 1
        return output

    transformer_hook = pipe.transformer.register_forward_hook(capture_transformer_io, with_kwargs=True)

    # Callback to save tensors at each step
    def save_step_tensors(pipe_obj, step_idx, timestep, callback_kwargs):
        """Callback to save tensors at each denoising step."""
        latents = callback_kwargs["latents"]

        # Reset call counter for next step, increment step counter
        transformer_call_counter["count"] = 0
        transformer_step_counter["count"] += 1

        if step_idx == 0:
            save_tensor("packed_latents_after_step0", latents, args.output_dir)
            tensor_stats("packed_latents_after_step0", latents)

            # Unpack for comparison
            latents_unpacked = pipe_obj._unpack_latents(latents, height, width, pipe_obj.vae_scale_factor)
            save_tensor("latents_after_step0", latents_unpacked, args.output_dir)

        saved_tensors["latents"] = latents
        saved_tensors["step"] = step_idx

        return callback_kwargs

    # Run pipeline
    result = pipe(
        image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if do_true_cfg else None,
        true_cfg_scale=args.true_cfg_scale,
        height=height,
        width=width,
        num_inference_steps=args.steps,
        generator=generator,
        callback_on_step_end=save_step_tensors,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="latent",  # Get latents instead of decoded image
    )

    transformer_hook.remove()

    # Save transformer captures
    print("\n  Saving transformer captures...")
    for name, tensor in transformer_captures.items():
        save_tensor(f"transformer_{name}", tensor, args.output_dir)

    # Save final latents
    final_latents_packed = result.images
    save_tensor("final_latents_packed", final_latents_packed, args.output_dir)
    tensor_stats("final_latents_packed", final_latents_packed)

    # Unpack final latents
    final_latents = pipe._unpack_latents(final_latents_packed, height, width, pipe.vae_scale_factor)
    save_tensor("final_latents", final_latents, args.output_dir)
    tensor_stats("final_latents", final_latents)

    # =========================================================================
    # Decode and save image
    # =========================================================================
    print("\n  Decoding final latents...")

    # Denormalize
    final_latents = final_latents.to(pipe.vae.dtype)
    denormalized = final_latents / (1.0 / latents_std) + latents_mean
    save_tensor("denormalized_latents", denormalized, args.output_dir)
    tensor_stats("denormalized_latents", denormalized)

    # VAE decode
    with torch.no_grad():
        decoded = pipe.vae.decode(denormalized, return_dict=False)[0][:, :, 0]
    save_tensor("decoded_image", decoded, args.output_dir)
    tensor_stats("decoded_image", decoded)

    # Post-process and save
    image_out = decoded.squeeze(0)
    image_out = (image_out / 2 + 0.5).clamp(0, 1)
    image_out = image_out.permute(1, 2, 0).cpu().float().numpy()
    image_out = (image_out * 255).astype(np.uint8)

    Image.fromarray(image_out).save(args.save_image)
    print(f"\n  Edited image saved to: {args.save_image}")

    print()
    print("=" * 60)
    print("Reference tensors saved to:", args.output_dir)
    print("=" * 60)
    print("\nTensors saved:")
    print("  - vae_input: Input image preprocessed for VAE")
    print("  - image_latents_raw: VAE encoder output")
    print("  - image_latents_normalized: Normalized image latents")
    print("  - packed_image_latents: Packed image latents for transformer")
    print("  - vision_embeds: Vision encoder output (before text encoder)")
    print("  - prompt_embeds: Text+vision prompt embeddings")
    print("  - prompt_mask: Attention mask for prompt")
    print("  - noise_latents_unpacked: Initial noise (5D)")
    print("  - packed_noise_latents: Packed noise for transformer")
    print("  - transformer_input_hidden_states: Concatenated [noise, image] latents")
    print("  - transformer_noise_pred_full: Full transformer output")
    print("  - packed_latents_after_step0: Latents after first denoising step")
    print("  - final_latents_packed/final_latents: Final denoised latents")
    print("  - denormalized_latents: Latents ready for VAE decode")
    print("  - decoded_image: VAE decoded output")


if __name__ == "__main__":
    main()