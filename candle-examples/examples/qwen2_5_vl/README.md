# Qwen2.5-VL: Vision-Language Model

Qwen2.5-VL is a multimodal model that combines a vision encoder with the Qwen2.5 language model for understanding images and videos.

## Architecture

Qwen2.5-VL combines:

- **Vision Encoder**: ViT with 2D RoPE and window attention (32 layers)
- **Text Decoder**: Qwen2.5 with M-RoPE for multimodal position encoding (28 layers for 7B)
- **Patch Merger**: 2x2 spatial merge to reduce vision token count

### Key Features

- **M-RoPE (Multimodal RoPE)**: 3D position encoding (temporal, height, width)
- **Dynamic Resolution**: Smart resizing maintains aspect ratio
- **Grouped Query Attention**: 7:1 ratio (28 Q heads, 4 KV heads for 7B)
- **No QK-normalization**: Unlike Qwen3-VL

## Model Sizes

| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| 3B | ~3 billion | `Qwen/Qwen2.5-VL-3B-Instruct` |
| 7B | ~7 billion | `Qwen/Qwen2.5-VL-7B-Instruct` (default) |
| 72B | ~72 billion | `Qwen/Qwen2.5-VL-72B-Instruct` |

## CLI Reference

### Input Options

| Flag | Description | Default |
|------|-------------|---------|
| `--image <PATH>` | Path to image file(s). Can specify multiple times. | Required (or --video) |
| `--video <PATH>` | Path to video file (requires ffmpeg) | - |
| `--prompt <TEXT>` | Question or instruction about the image/video | "Describe this image in detail." |

### Model Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-id <ID>` | HuggingFace model repository | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `--revision <REV>` | Model revision/branch | `main` |
| `--cpu` | Run on CPU instead of GPU | false |
| `--bf16` | Use bfloat16 precision | false |

### Generation Options

| Flag | Description | Default |
|------|-------------|---------|
| `--max-length <N>` | Maximum tokens to generate | 512 |
| `--temperature <F>` | Sampling temperature (0 = greedy) | None (greedy) |
| `--top-p <F>` | Nucleus sampling threshold (0.0-1.0) | None |
| `--top-k <N>` | Top-k sampling | None |
| `--repeat-penalty <F>` | Penalty for repeated tokens | 1.0 |
| `--repeat-last-n <N>` | Tokens to apply repeat penalty to | 64 |
| `--seed <N>` | Random seed for reproducibility | 299792458 |
| `--stream` | Print tokens as they're generated | false |

### Attention Options

| Flag | Description | Default |
|------|-------------|---------|
| `--flash-attn` | Use Flash Attention 2 (requires CUDA + feature) | false |
| `--sliding-window` | Enable sliding window attention | false |
| `--sliding-window-size <N>` | Sliding window size | 4096 |
| `--max-window-layers <N>` | Layers using sliding window | 0 (all) |

### Video Options

| Flag | Description | Default |
|------|-------------|---------|
| `--video-fps <F>` | Frame extraction rate | 2.0 |
| `--max-frames <N>` | Maximum frames to extract | 32 |

## Examples

### Basic Image Understanding

```bash
# Describe an image
cargo run --example qwen2_5_vl --release -- \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image in detail."

# Ask a specific question
cargo run --example qwen2_5_vl --release -- \
    --image chart.png \
    --prompt "What trends does this chart show?"
```

### Multi-Image Comparison

```bash
# With placeholders (explicit positioning)
cargo run --example qwen2_5_vl --release -- \
    --image before.jpg --image after.jpg \
    --prompt "Compare {image1} and {image2}. What changed?"

# Without placeholders (images placed before prompt)
cargo run --example qwen2_5_vl --release -- \
    --image page1.png --image page2.png \
    --prompt "Compare these two pages"
```

### Video Understanding

```bash
# Requires ffmpeg installed
cargo run --example qwen2_5_vl --release -- \
    --video /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/bunny_10s.mp4 \
    --prompt "Describe what happens in this video"

# With custom frame rate
cargo run --example qwen2_5_vl --release -- \
    --video /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/bunny_10s.mp4 --video-fps 4.0 --max-frames 64 \
    --prompt "What actions are shown?"
```

### Creative Generation with Sampling

```bash
# Temperature sampling for creative responses
cargo run --example qwen2_5_vl --release -- \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Write a creative story about this image" \
    --temperature 0.8

# Nucleus (top-p) sampling
cargo run --example qwen2_5_vl --release -- \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image poetically" \
    --temperature 0.9 --top-p 0.95

# Combined top-k + top-p
cargo run --example qwen2_5_vl --release -- \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "What might happen next?" \
    --temperature 0.7 --top-k 50 --top-p 0.9
```

### Streaming Output

```bash
# Watch tokens appear in real-time
cargo run --example qwen2_5_vl --release -- \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image in detail" \
    --stream --temperature 0.7
```

### Reducing Repetition

```bash
# Apply repeat penalty
cargo run --example qwen2_5_vl --release -- \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image" \
    --repeat-penalty 1.1 --repeat-last-n 64
```

### Performance Optimization

```bash
# Flash Attention (CUDA only, requires feature flag)
cargo run --example qwen2_5_vl --release --features flash-attn -- \
    --flash-attn \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image"

# Sliding window for long sequences
cargo run --example qwen2_5_vl --release -- \
    --sliding-window --sliding-window-size 4096 \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image"

# Use smaller model for faster inference
cargo run --example qwen2_5_vl --release -- \
    --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "What is this?"
```

### Precision Options

```bash
# BFloat16 (faster on supported hardware)
cargo run --example qwen2_5_vl --release -- \
    --bf16 \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image"

# CPU inference (slower but works without GPU)
cargo run --example qwen2_5_vl --release -- \
    --cpu \
    --image /Users/daniel/git/candle/candle-examples/examples/qwen2_5_vl/dice.png \
    --prompt "Describe this image"
```
