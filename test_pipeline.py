#!/usr/bin/env python3
"""
Quick test script to verify the pipeline works without external model dependencies.
Run this after installing the dependencies.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that all required packages can be imported."""
    print("=== Testing Basic Imports ===")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers available")
    except ImportError:
        print("❌ Diffusers not available")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers not available")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow available")
    except ImportError:
        print("❌ Pillow not available")
        return False
    
    print("✅ All basic imports successful!\n")
    return True

def test_feature_to_text_mapper():
    """Test the FeatureToTextMapper without external models."""
    print("=== Testing Feature-to-Text Mapper ===")
    
    try:
        from utils.feature_to_text import FeatureToTextMapper
        
        # Create mapper (should work without external models)
        mapper = FeatureToTextMapper(
            feature_dim=2048,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("✅ FeatureToTextMapper initialized successfully")
        
        # Test prompt generation with dummy feature
        dummy_feature = torch.randn(2048)
        prompt = mapper.generate_text_prompt(dummy_feature, "ruby_throated_hummingbird")
        
        print(f"✅ Generated test prompt: '{prompt}'")
        
        # Test quality computation (should use fallback)
        quality = mapper.compute_prompt_quality(prompt, dummy_feature)
        print(f"✅ Computed quality score: {quality:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FeatureToTextMapper test failed: {e}")
        return False

def test_memory_bank_integration():
    """Test integration with existing memory bank system."""
    print("=== Testing Memory Bank Integration ===")
    
    try:
        from utils.memory_bank import MemoryBank
        from utils.memory_manager import MemoryManager
        
        # Create a dummy model class for testing
        class DummyModel:
            def get_feature_dim(self):
                return 2048
            
            def forward(self, x, return_features=False):
                batch_size = x.shape[0]
                features = torch.randn(batch_size, 2048)
                logits = torch.randn(batch_size, 10)
                
                if return_features:
                    return logits, features
                return logits
        
        # Test MemoryBank
        memory_bank = MemoryBank(
            num_classes=10,
            feature_dim=2048,
            capacity_per_class=32,
            device='cpu'  # Use CPU for testing
        )
        
        # Add some dummy features
        for class_id in range(5):  # Add to first 5 classes
            for _ in range(10 if class_id < 2 else 2):  # Classes 0,1 get more samples (head), others get fewer (tail)
                dummy_feature = torch.randn(2048)
                memory_bank.update(class_id, dummy_feature)
        
        print(f"✅ MemoryBank test: {memory_bank.total_samples} samples stored")
        
        # Test MemoryManager
        dummy_model = DummyModel()
        memory_manager = MemoryManager(
            model=dummy_model,
            num_classes=10,
            capacity_per_class=32,
            device='cpu'
        )
        
        # Test getting tail classes
        tail_classes = memory_bank.get_tail_classes()
        print(f"✅ Identified tail classes: {tail_classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory bank integration test failed: {e}")
        return False

def test_image_generation_setup():
    """Test that image generation pipeline can be initialized."""
    print("=== Testing Image Generation Setup ===")
    
    try:
        from utils.image_diffusion_pipeline import TailClassImageGenerator
        
        # Test initialization (should work even if CLIP is not available)
        generator = TailClassImageGenerator(
            device='cpu',  # Use CPU for testing
            seed=42
        )
        
        print("✅ TailClassImageGenerator initialized (Note: First run will download Stable Diffusion ~5GB)")
        
        # Test prompt processing
        test_prompts = {
            0: ["A detailed photograph of a ruby-throated hummingbird in flight"],
            1: ["A beautiful rare bird species with distinctive coloring"]
        }
        
        print("✅ Image generation pipeline setup complete")
        print("⚠️  Note: Actual image generation requires ~6GB VRAM and will download models")
        
        return True
        
    except Exception as e:
        print(f"❌ Image generation setup failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Memory-Conditioned Diffusion Pipeline Setup\n")
    
    tests = [
        test_basic_imports,
        test_feature_to_text_mapper,
        test_memory_bank_integration,
        test_image_generation_setup
    ]
    
    passed = 0
    for i, test in enumerate(tests, 1):
        print(f"Test {i}/{len(tests)}: {test.__name__}")
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}\n")
    
    print(f"=== Test Results: {passed}/{len(tests)} passed ===")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your pipeline is ready.")
        print("\n📋 Next steps:")
        print("1. Train your model with memory bank enabled: python main.py --use_memory_bank")
        print("2. Generate semantic prompts from tail classes")
        print("3. Generate synthetic images for tail classes") 
        print("4. Integrate synthetic images into training")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
        print("Most likely issues:")
        print("- Missing dependencies (run pip install commands)")
        print("- File path issues (ensure utils/ directory exists)")
        print("- Import path issues (run from project root)")

if __name__ == "__main__":
    main()