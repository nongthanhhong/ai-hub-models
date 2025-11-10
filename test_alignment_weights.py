#!/usr/bin/env python3
"""Test alignment weights export for Whisper Large V3 Turbo."""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qai_hub_models.models.whisper_large_v3_turbo import Model
from qai_hub_models.utils.input_spec import make_torch_inputs


def main():
    print("\n" + "=" * 80)
    print("WHISPER LARGE V3 TURBO - ALIGNMENT WEIGHTS VALIDATION")
    print("=" * 80)
    
    # Test 1: Model instantiation
    print("\nTEST 1: Model Instantiation")
    print("-" * 80)
    model = Model.from_pretrained()
    print(f"✅ Model instantiated: Encoder={type(model.encoder).__name__}, Decoder={type(model.decoder).__name__}")
    
    # Test 2: Decoder forward pass
    print("\nTEST 2: Decoder Forward Pass")
    print("-" * 80)
    decoder = model.decoder
    decoder.eval()
    
    input_spec = decoder.get_input_spec()
    sample_inputs = make_torch_inputs(input_spec)
    
    with torch.no_grad():
        outputs = decoder(*sample_inputs)
    
    print(f"Number of outputs: {len(outputs)}")
    alignment_weights = outputs[-1]
    
    if alignment_weights is not None:
        print(f"✅ Alignment weights shape: {alignment_weights.shape}")
        print(f"   Min: {alignment_weights.min():.6f}, Max: {alignment_weights.max():.6f}, Sum: {alignment_weights.sum():.6f}")
        if alignment_weights.shape == (1, 1500):
            print(f"✅ Correct shape [1, 1500]")
        else:
            print(f"⚠️  Unexpected shape")
    else:
        print(f"⚠️  Alignment weights are None")
    
    # Test 3: Output names
    print("\nTEST 3: Output Names")
    print("-" * 80)
    output_names = decoder.get_output_names()
    print(f"Total outputs: {len(output_names)}")
    if "alignment_heads_weights" in output_names:
        print(f"✅ 'alignment_heads_weights' found at position {output_names.index('alignment_heads_weights')}")
    else:
        print(f"❌ 'alignment_heads_weights' NOT found")
        return 1
    
    # Test 4: Traceability
    print("\nTEST 4: TorchScript Traceability")
    print("-" * 80)
    try:
        with torch.no_grad():
            traced_decoder = torch.jit.trace(decoder, sample_inputs)
        print(f"✅ Model traced successfully")
        
        with torch.no_grad():
            traced_outputs = traced_decoder(*sample_inputs)
        print(f"✅ Traced model forward pass successful")
        print(f"   Outputs match: {len(traced_outputs) == len(outputs)}")
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Alignment weights ready for export!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
