# Qwen-Image Example

## Overview
This example implements the Qwen-Image diffusion transformer for image generation and editing.

## Usage

### Generate Images
```bash
cargo run --release --features cuda --example qwen_image -- generate \
    --prompt "A serene mountain landscape" \
    --height 512 --width 512
```

### Edit Images
```bash
cargo run --release --features cuda --example qwen_image -- edit \
    --input-image input.png \
    --prompt "Add a sunset sky"
```

### ControlNet
```bash
cargo run --release --features cuda --example qwen_image -- controlnet \
    --input-image control.png \
    --prompt "A beautiful landscape"
```

## Quantized Models
Use GGUF quantized models for reduced memory:
```bash
cargo run --release --features cuda --example qwen_image -- generate \
    --gguf path/to/model.gguf \
    --prompt "..."
```

## Python Environment
- **Always use `uv`** for Python scripts
- Run scripts: `uv run script.py`
- Reference tensor scripts are available for validation against PyTorch
