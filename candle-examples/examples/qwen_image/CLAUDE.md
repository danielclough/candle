# Qwen-Image Debug Guide

## Problem
The Candle implementation produces noise instead of valid images. We need to find where it diverges from PyTorch.

## Architecture

### Debug Module (`debug.rs`)
A **utility module** that provides:
- `DebugContext` - holds state for tensor comparison and substitution
- `checkpoint()` - saves tensors, compares with PyTorch, optionally substitutes
- `save_npy()` / `load_npy()` - NumPy format I/O for cross-language comparison

### Integration with Pipelines
Debug is a **global flag** (`--debug`) that works with any pipeline:
```bash
cargo run --example qwen_image -- --debug generate --prompt "..."
cargo run --example qwen_image -- --debug edit --input-image ...
```

Each pipeline's `run()` function accepts `Option<&mut DebugContext>`:
```rust
pub fn run(args, paths, device, dtype, mut debug_ctx: Option<&mut DebugContext>) -> Result<()>
```

Checkpoints are inserted at key stages:
```rust
let tensor = if let Some(ctx) = debug_ctx.as_mut() {
    ctx.checkpoint("tensor_name", tensor)?
} else {
    tensor
};
```

## Directory Structure
```
candle/
├── debug_tensors/
│   ├── pytorch/     # Reference tensors from official pipeline
│   │   ├── prompt_embeds.npy
│   │   ├── initial_latents_packed.npy
│   │   ├── packed_latents_step0.npy
│   │   ├── noise_pred_step0.npy
│   │   └── ...
│   └── rust/        # Tensors saved by Candle for comparison
│       └── ...
├── candle-examples/examples/qwen_image/
│   ├── generate_reference_tensors.py  # Uses official QwenImagePipeline
│   ├── test_official_pipeline.py      # Minimal official pipeline test
│   ├── debug.rs                       # Debug utilities module
│   ├── generate.rs                    # Generate pipeline (with debug hooks)
│   └── main.rs                        # CLI with --debug flag
```

## Workflow

### Step 1: Generate PyTorch Reference Tensors
Uses the **official QwenImagePipeline** to guarantee correctness:
```bash
uv run candle-examples/examples/qwen_image/generate_reference_tensors.py \
    --prompt "A serene mountain landscape" \
    --height 512 --width 512 \
    --seed 42
```

This saves tensors to `debug_tensors/pytorch/` and an image to `pytorch_output.png`.

### Step 2: Run Candle with Debug Flag
```bash
cargo run --release --features metal,accelerate --example qwen_image -- \
    --debug generate \
    --prompt "A serene mountain landscape" \
    --height 512 --width 512
```

## Python Environment
- **Always use `uv`** for Python
- Run scripts: `uv run script.py`
- The scripts use local `diffusers/` and `transformers/` forks