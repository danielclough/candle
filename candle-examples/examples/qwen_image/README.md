# Qwen-Image: Text-to-Image Generation

This example demonstrates text-to-image generation using the [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) diffusion model, a 20B parameter Multimodal Diffusion Transformer (MMDiT).

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

### Device Features

CUDA:
```bash
cargo run --release --features cuda --example qwen_image -- --prompt "..."
```

Metal (macOS):
```bash
cargo run --release --features metal --example qwen_image -- --prompt "..."
```

CPU
```bash
cargo run --release --example qwen_image -- --cpu --use-f32 --prompt "..."
```
