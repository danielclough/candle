# Qwen-Image: Text-to-Image Generation

This example demonstrates text-to-image generation using the [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) diffusion model, a 20B parameter Multimodal Diffusion Transformer (MMDiT).

## Pipelines

The CLI supports multiple generation modes:

- **generate**: Text-to-image generation (with optional img2img)
- **edit**: Image editing with text instructions
- **inpaint**: Fill in masked regions of an image
- **layered**: Decompose an image into transparent layers
- **controlnet**: ControlNet-guided generation

## Usage

### Basic Text-to-Image

```bash
cargo run --release --example qwen_image -- generate \
    --prompt "A serene mountain landscape at sunset" \
    --output landscape.png
```

### With Options

```bash
cargo run --release --example qwen_image -- generate \
    --prompt "A futuristic city at night with neon lights" \
    --negative-prompt "blurry, low quality" \
    --height 1024 \
    --width 1024 \
    --steps 50 \
    --true-cfg-scale 4.0 \
    --seed 42 \
    --output city.png
```

### Image Editing

```bash
cargo run --release --example qwen_image -- edit \
    --input-image photo.jpg \
    --prompt "Make the sky purple" \
    --output edited.png
```

### Device Features

CUDA:
```bash
cargo run --release --features cuda --example qwen_image -- generate --prompt "..."
```

Metal (macOS):
```bash
cargo run --release --features metal --example qwen_image -- generate --prompt "..."
```

CPU:
```bash
cargo run --release --example qwen_image -- --cpu --use-f32 generate --prompt "..."
```

## Memory Optimization

The following flags help reduce GPU memory usage for large images or constrained hardware:

### VAE Slicing

Process batch dimension one sample at a time, reducing peak memory for batch operations:

```bash
cargo run --release --example qwen_image -- --enable-vae-slicing generate --prompt "..."
```

### VAE Tiling

Split large images into overlapping tiles for encode/decode, enabling generation of images larger than GPU memory:

```bash
cargo run --release --example qwen_image -- \
    --enable-vae-tiling \
    --vae-tile-size 256 \
    --vae-tile-stride 192 \
    generate --prompt "..." --height 2048 --width 2048
```

Options:
- `--vae-tile-size`: Tile size in pixels (default: 256)
- `--vae-tile-stride`: Stride between tiles (default: 192, giving 64px overlap)

### Text Caching for CFG

Cache text Q/K/V projections during Classifier-Free Guidance (CFG), reduce transformer compute on the negative prompt pass:

```bash
cargo run --release --example qwen_image -- --enable-text-cache generate \
    --prompt "A beautiful sunset" \
    --negative-prompt "blurry, low quality" \
    --true-cfg-scale 4.0
```

**Note**: Text caching is only effective when:
- Using True CFG with a negative prompt (`--true-cfg-scale > 1.0`)
- Using FP16 transformer (not quantized GGUF)

### Combined Example

For large image generation with all optimizations:

```bash
cargo run --release --features metal --example qwen_image -- \
    --enable-vae-tiling \
    --enable-text-cache \
    generate \
    --prompt "A detailed fantasy landscape with mountains and rivers" \
    --negative-prompt "blurry, artifacts" \
    --height 2048 \
    --width 2048 \
    --true-cfg-scale 3.5 \
    --output large_landscape.png
```

## Quantized Models

Use GGUF quantized models for reduced memory footprint:

```bash
cargo run --release --example qwen_image -- \
    --gguf-transformer \
    --gguf-text-encoder \
    generate --prompt "..."
```

Available quantization levels (from `city96/Qwen-Image-gguf`):
- Q2_K (7GB) - Fastest, lowest quality
- Q3_K_M (9.7GB)
- Q4_K_M (13GB) - Good balance (default)
- Q5_K_M (15GB)
- Q8_0 (22GB)
- BF16 (41GB) - Full precision
