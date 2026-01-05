# Qwen-Image: Text-to-Image Generation

This example demonstrates text-to-image generation using the [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) diffusion model, a 20B parameter Multimodal Diffusion Transformer (MMDiT).

## Architecture

Qwen-Image consists of three main components:

1. **Text Encoder**: Qwen2.5-VL-7B-Instruct (~15GB) for prompt encoding
2. **Transformer**: 20B parameter dual-stream MMDiT (~40GB) for denoising
3. **VAE**: 3D Causal VAE (~300MB) for latent decoding

## Requirements

- **GPU Memory**: ~55GB for full inference (sequential loading reduces peak usage)
- **Storage**: ~55GB for model weights

For CPU inference, you'll need significant RAM and patience.

## Usage

### Basic Usage

```bash
cargo run --release --example qwen_image -- \
    --prompt "A serene mountain landscape at sunset" \
    --output landscape.png
```

### With Options

```bash
cargo run --release --example qwen_image -- \
    --prompt "A futuristic city at night with neon lights" \
    --negative-prompt "blurry, low quality" \
    --height 1024 \
    --width 1024 \
    --num-inference-steps 50 \
    --true-cfg-scale 4.0 \
    --seed 42 \
    --output city.png
```

### GPU Features

For CUDA:
```bash
cargo run --release --features cuda --example qwen_image -- --prompt "..."
```

For Metal (macOS):
```bash
cargo run --release --features metal --example qwen_image -- --prompt "..."
```

### CPU Mode

```bash
cargo run --release --example qwen_image -- --cpu --use-f32 --prompt "..."
```

Note: CPU inference is very slow and may require significant RAM.

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | "A serene mountain..." | Text description of the image |
| `--negative-prompt` | "" | What to avoid in the image |
| `--height` | 1024 | Image height (must be divisible by 16) |
| `--width` | 1024 | Image width (must be divisible by 16) |
| `--num-inference-steps` | 50 | Number of denoising steps |
| `--true-cfg-scale` | 4.0 | Guidance scale for True CFG |
| `--seed` | random | Random seed for reproducibility |
| `--output` | "qwen_image_output.png" | Output filename |
| `--cpu` | false | Force CPU inference |
| `--use-f32` | false | Use F32 instead of BF16 |
| `--tracing` | false | Enable Chrome tracing profiler |
| `--transformer-path` | HuggingFace | Local path to transformer weights |
| `--vae-path` | HuggingFace | Local path to VAE weights |
| `--text-encoder-path` | HuggingFace | Local path to text encoder |
| `--tokenizer-path` | HuggingFace | Local path to tokenizer |

## Model Files

Models are automatically downloaded from HuggingFace on first run:

| Component | Repository | Size |
|-----------|------------|------|
| Text Encoder | `Qwen/Qwen2.5-VL-7B-Instruct` | ~15GB |
| Transformer | `Qwen/Qwen-Image` | ~40GB |
| VAE | `Qwen/Qwen-Image` | ~300MB |

## Pipeline Details

1. **Prompt Encoding**: Text is tokenized and passed through Qwen2.5-VL to get embeddings
2. **Dynamic Shift**: Scheduler computes resolution-dependent shift parameter
3. **Latent Initialization**: Random noise latents are created in VAE latent space
4. **Denoising Loop**: Iterative refinement using True CFG (positive + negative prompts)
5. **VAE Decoding**: Final latents are decoded to RGB image

## Tips

- Use BF16 (default) for faster inference on supported GPUs
- Increase `--num-inference-steps` for higher quality (at the cost of speed)
- Use `--seed` for reproducible results
- Adjust `--true-cfg-scale` to control prompt adherence (higher = more strict)

## References

- [Qwen-Image Model Card](https://huggingface.co/Qwen/Qwen-Image)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
