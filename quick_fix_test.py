#!/usr/bin/env python3
"""
Quick test to verify the dtype fix works.
Run this before running the full test again.
"""

import torch
import torch.nn as nn

def test_dtype_fix():
    print("🔧 Testing Dtype Fix")
    print("=" * 30)
    
    # First, test just the neural network part
    print("Step 1: Testing neural network creation...")
    feature_projector = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 768)
    ).float()  # Ensure float32
    
    print(f"✅ Network weights dtype: {next(feature_projector.parameters()).dtype}")
    
    # Test with double tensor
    print("Step 2: Testing with Double tensor...")
    feature_double = torch.randn(1, 512, dtype=torch.float64)
    print(f"✅ Input tensor dtype: {feature_double.dtype}")
    
    try:
        # Convert to float before passing
        feature_float = feature_double.float()
        output = feature_projector(feature_float)
        print(f"✅ Output tensor dtype: {output.dtype}")
        print("✅ Neural network test passed!")
    except Exception as e:
        print(f"❌ Neural network error: {e}")
        return False
    
    # Now test the full mapper
    print("Step 3: Testing FeatureToTextMapper...")
    try:
        from utils.feature_to_text import FeatureToTextMapper
        
        mapper = FeatureToTextMapper(feature_dim=512, device='cpu')
        print("✅ Mapper created")
        
        # Test with float64 (Double) tensor - this was causing the error
        feature_double = torch.randn(512, dtype=torch.float64)
        print(f"✅ Created Double tensor: {feature_double.dtype}")
        
        # Test prompt generation
        prompt = mapper.generate_text_prompt(feature_double, "test_class")
        print(f"✅ Generated prompt: '{prompt[:50]}...'")
        print("🎉 Full mapper test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Mapper error: {e}")
        
        # Print more debug info
        print(f"Error details: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dtype_fix()