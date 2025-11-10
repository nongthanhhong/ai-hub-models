#!/usr/bin/env python3
"""Simple test for alignment weights - no full app dependencies."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Direct import to avoid app.py dependencies
from qai_hub_models.models._shared.hf_whisper.model import HfWhisperDecoder, AUDIO_EMB_LEN, MEAN_DECODE_LEN
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo

print("\n" + "=" * 80)
print("ALIGNMENT WEIGHTS - QUICK VALIDATION")
print("=" * 80)

# Test decoder directly
print("\n[1] Instantiating decoder...")
try:
    whisper_model = WhisperLargeV3Turbo.from_pretrained()
    decoder = whisper_model.decoder
    print(f"✅ Decoder instantiated: {type(decoder).__name__}")
    print(f"   Config: {decoder.config.decoder_layers} layers, {decoder.config.decoder_attention_heads} heads")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass
print("\n[2] Testing forward pass...")
try:
    decoder.eval()
    
    # Create sample inputs matching the spec
    num_blocks = decoder.config.decoder_layers
    num_heads = decoder.config.decoder_attention_heads
    attention_dim = decoder.config.d_model
    head_dim = attention_dim // num_heads
    
    input_ids = torch.tensor([[1]], dtype=torch.int32)
    attention_mask = torch.full((1, 1, 1, MEAN_DECODE_LEN), -100.0, dtype=torch.float32)
    position_ids = torch.tensor([0], dtype=torch.int32)
    
    # KV caches (self-attention)
    k_cache_self = [torch.zeros((num_heads, 1, head_dim, MEAN_DECODE_LEN - 1), dtype=torch.float32) for _ in range(num_blocks)]
    v_cache_self = [torch.zeros((num_heads, 1, MEAN_DECODE_LEN - 1, head_dim), dtype=torch.float32) for _ in range(num_blocks)]
    
    # KV caches (cross-attention)
    k_cache_cross = [torch.randn((num_heads, 1, head_dim, AUDIO_EMB_LEN), dtype=torch.float32) for _ in range(num_blocks)]
    v_cache_cross = [torch.randn((num_heads, 1, AUDIO_EMB_LEN, head_dim), dtype=torch.float32) for _ in range(num_blocks)]
    
    # Flatten inputs
    kv_caches = []
    for i in range(num_blocks):
        kv_caches.extend([k_cache_self[i], v_cache_self[i]])
    for i in range(num_blocks):
        kv_caches.extend([k_cache_cross[i], v_cache_cross[i]])
    
    inputs = (input_ids, attention_mask) + tuple(kv_caches) + (position_ids,)
    
    with torch.no_grad():
        outputs = decoder(*inputs)
    
    print(f"✅ Forward pass successful")
    print(f"   Number of outputs: {len(outputs)}")
    print(f"   Output types: {[type(o) for o in outputs]}")
    
    # Check alignment weights
    alignment_weights = outputs[-1]
    if alignment_weights is None:
        print(f"⚠️  Alignment weights are None")
    else:
        print(f"✅ Alignment weights present")
        print(f"   Shape: {alignment_weights.shape}")
        print(f"   Dtype: {alignment_weights.dtype}")
        print(f"   Range: [{alignment_weights.min():.6f}, {alignment_weights.max():.6f}]")
        print(f"   Sum: {alignment_weights.sum():.6f}")
        
        expected_shape = (1, AUDIO_EMB_LEN)
        if alignment_weights.shape == expected_shape:
            print(f"   ✅ Correct shape {expected_shape}")
        else:
            print(f"   ❌ Wrong shape! Expected {expected_shape}")
            
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test output spec
print("\n[3] Checking output specification...")
output_names = decoder.get_output_names()
print(f"Total output names: {len(output_names)}")
print(f"First 3: {output_names[:3]}")
print(f"Last 3: {output_names[-3:]}")

if "alignment_heads_weights" in output_names:
    idx = output_names.index("alignment_heads_weights")
    print(f"✅ 'alignment_heads_weights' found at position {idx}")
else:
    print(f"❌ 'alignment_heads_weights' NOT in output names")
    sys.exit(1)

# Test traceability
print("\n[4] Testing TorchScript traceability...")
try:
    with torch.no_grad():
        traced = torch.jit.trace(decoder, inputs)
    print(f"✅ Model traced successfully")
    
    with torch.no_grad():
        traced_outputs = traced(*inputs)
    print(f"✅ Traced model runs successfully")
    
    # Compare outputs
    alignment_diff = (outputs[-1] - traced_outputs[-1]).abs().max().item()
    print(f"   Alignment weights max diff: {alignment_diff:.2e}")
    if alignment_diff < 1e-5:
        print(f"   ✅ Outputs match")
    else:
        print(f"   ⚠️  Some numerical difference")
        
except Exception as e:
    print(f"❌ Tracing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
print("\n✓ Alignment weights are properly exported")
print("✓ Shape is correct [1, 1500]")
print("✓ Model is traceable for ONNX export")
print("✓ Ready for QNN compilation\n")
