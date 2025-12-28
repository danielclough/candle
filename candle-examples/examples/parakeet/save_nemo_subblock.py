#!/usr/bin/env python3
"""
Save sub-block intermediate tensors from NeMo for comparison with Candle.
Focuses on Block 0 and Block 1 to identify divergence point.
"""

import torch
import nemo.collections.asr as nemo_asr
from pathlib import Path

# Load CTC-1B model
print("Loading CTC-1B model...")
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('nvidia/parakeet-ctc-1.1b')
model.eval()

# Get encoder
encoder = model.encoder

# Find test audio
audio_path = Path(__file__).parent / "text" / "en-Carter_man.wav"
print(f"Processing audio: {audio_path}")

# Storage for captured tensors
captured = {}
hooks = []

# Hook to capture mel spectrogram
def capture_mel(module, input, output):
    if isinstance(output, tuple):
        mel, mel_len = output
    else:
        mel = output
    captured['mel'] = mel.clone()
    print(f"Captured mel: shape={mel.shape}, rms={mel.pow(2).mean().sqrt().item():.4f}")

# Hook for pre_encode (subsampling)
def capture_subsampling(module, input, output):
    if isinstance(output, tuple):
        out = output[0]
    else:
        out = output
    captured['subsampling'] = out.clone()
    print(f"Captured subsampling: shape={out.shape}, rms={out.pow(2).mean().sqrt().item():.4f}")

# Hook for conformer layers with detailed capture
def make_layer_hook(layer_idx, layer):
    captured_layer = {}

    # Capture norm outputs and intermediate values
    def capture_norm_ff1(module, input, output):
        captured_layer['after_norm_ff1'] = output.clone()
        print(f"  Layer {layer_idx} norm_ff1: rms={output.pow(2).mean().sqrt().item():.4f}")

    def capture_ff1(module, input, output):
        captured_layer['after_ff1'] = output.clone()
        print(f"  Layer {layer_idx} ff1: rms={output.pow(2).mean().sqrt().item():.4f}")

    def capture_norm_att(module, input, output):
        captured_layer['after_norm_att'] = output.clone()
        print(f"  Layer {layer_idx} norm_att: rms={output.pow(2).mean().sqrt().item():.4f}")

    def capture_attn(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        captured_layer['after_attn'] = out.clone()
        print(f"  Layer {layer_idx} attn: rms={out.pow(2).mean().sqrt().item():.4f}")

    def capture_norm_conv(module, input, output):
        captured_layer['after_norm_conv'] = output.clone()
        print(f"  Layer {layer_idx} norm_conv: rms={output.pow(2).mean().sqrt().item():.4f}")

    def capture_conv(module, input, output):
        captured_layer['after_conv'] = output.clone()
        print(f"  Layer {layer_idx} conv: rms={output.pow(2).mean().sqrt().item():.4f}")

    def capture_norm_ff2(module, input, output):
        captured_layer['after_norm_ff2'] = output.clone()
        print(f"  Layer {layer_idx} norm_ff2: rms={output.pow(2).mean().sqrt().item():.4f}")

    def capture_ff2(module, input, output):
        captured_layer['after_ff2'] = output.clone()
        print(f"  Layer {layer_idx} ff2: rms={output.pow(2).mean().sqrt().item():.4f}")

    def capture_layer_output(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        captured_layer['output'] = out.clone()
        captured[f'layer_{layer_idx}'] = captured_layer.copy()
        print(f"  Layer {layer_idx} output: rms={out.pow(2).mean().sqrt().item():.4f}")

    # Register hooks
    h = []
    h.append(layer.norm_feed_forward1.register_forward_hook(capture_norm_ff1))
    h.append(layer.feed_forward1.register_forward_hook(capture_ff1))
    h.append(layer.norm_self_att.register_forward_hook(capture_norm_att))
    h.append(layer.self_attn.register_forward_hook(capture_attn))
    h.append(layer.norm_conv.register_forward_hook(capture_norm_conv))
    h.append(layer.conv.register_forward_hook(capture_conv))
    h.append(layer.norm_feed_forward2.register_forward_hook(capture_norm_ff2))
    h.append(layer.feed_forward2.register_forward_hook(capture_ff2))
    h.append(layer.register_forward_hook(capture_layer_output))
    return h

# Register hooks
hooks.append(model.preprocessor.register_forward_hook(capture_mel))
hooks.append(encoder.pre_encode.register_forward_hook(capture_subsampling))

# Only hook first 3 layers for detailed analysis
for i, layer in enumerate(encoder.layers[:3]):
    print(f"Registering hooks for layer {i}")
    hooks.extend(make_layer_hook(i, layer))

# Run transcription
print("\nRunning forward pass...")
try:
    result = model.transcribe([str(audio_path)])
    print(f"\nTranscription: {result}")
except Exception as e:
    print(f"Error: {e}")
finally:
    for h in hooks:
        h.remove()

# Print summary
print("\n=== Summary ===")
print(f"Mel shape: {captured.get('mel', torch.zeros(1)).shape}")
print(f"Subsampling RMS: {captured.get('subsampling', torch.zeros(1)).pow(2).mean().sqrt().item():.4f}")

for layer_idx in range(3):
    layer_data = captured.get(f'layer_{layer_idx}', {})
    if 'output' in layer_data:
        print(f"\nLayer {layer_idx}:")
        for key in ['after_norm_ff1', 'after_ff1', 'after_norm_att', 'after_attn',
                    'after_norm_conv', 'after_conv', 'after_norm_ff2', 'after_ff2', 'output']:
            if key in layer_data:
                t = layer_data[key]
                print(f"  {key}: rms={t.pow(2).mean().sqrt().item():.4f}")
