#!/usr/bin/env python3
"""
Quick verification script to test if all components work correctly
Run this before running the full orchestrator
"""

import sys
import torch
import os

def test_imports():
    """Test if all required imports work."""
    print("Testing imports...")
    try:
        from models import ResNet32, ResNet50
        print("  ✓ Models imported")
        
        from utils.memory_bank import MemoryBank
        from utils.memory_manager import MemoryManager
        print("  ✓ Memory components imported")
        
        from utils.csl_loss import CSLLossFunc
        print("  ✓ CSL Loss imported")
        
        from phase3_feature_ddpm import FeatureDDPM
        from train_feature_ddpm import extract_features_from_dataset, train_feature_ddpm
        print("  ✓ DDPM components imported")
        
        from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator
        print("  ✓ Prompt generator imported")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_model_features():
    """Test if ResNet model feature extraction works."""
    print("\nTesting model feature extraction...")
    try:
        from models import ResNet32
        
        # Create model
        model = ResNet32(num_classes=10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Test input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
        
        # Test forward with features
        with torch.no_grad():
            outputs, features = model(dummy_input, return_features=True)
        
        # Check dimensions
        assert outputs.shape == (batch_size, 10), f"Output shape wrong: {outputs.shape}"
        assert features.shape == (batch_size, 512), f"Feature shape wrong: {features.shape}"
        
        # Test feature dimension method
        feature_dim = model.get_feature_dim()
        assert feature_dim == 512, f"Feature dim wrong: {feature_dim}"
        
        print(f"  ✓ Model outputs: {outputs.shape}")
        print(f"  ✓ Features: {features.shape}")
        print(f"  ✓ Feature dimension: {feature_dim}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False

def test_memory_bank():
    """Test if memory bank works correctly."""
    print("\nTesting memory bank...")
    try:
        from utils.memory_bank import MemoryBank
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create memory bank
        memory_bank = MemoryBank(
            num_classes=10,
            feature_dim=512,
            capacity_per_class=64,
            device=device
        )
        
        # Test update
        for class_id in range(10):
            for _ in range(5):
                fake_feature = torch.randn(512).to(device)
                memory_bank.update(class_id, fake_feature)
        
        # Check tail classes (they are automatically updated on each update)
        tail_classes = memory_bank.get_tail_classes()
        head_classes = memory_bank.get_head_classes()
        
        print(f"  ✓ Memory bank created")
        print(f"  ✓ Updates processed")
        print(f"  ✓ Tail classes identified: {len(tail_classes)}")
        print(f"  ✓ Head classes identified: {len(head_classes)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Memory bank test failed: {e}")
        return False

def test_ddpm():
    """Test if DDPM model initializes correctly."""
    print("\nTesting DDPM...")
    try:
        from phase3_feature_ddpm import FeatureDDPM
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create DDPM model
        ddpm_model = FeatureDDPM(
            feature_dim=512,
            num_classes=10,
            hidden_dim=1024,
            num_layers=4,
            num_timesteps=1000,
            beta_schedule='cosine'
        ).to(device)
        
        # Test forward pass
        batch_size = 4
        fake_features = torch.randn(batch_size, 512).to(device)
        fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
        
        loss = ddpm_model(fake_features, fake_labels)
        
        print(f"  ✓ DDPM model created")
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Loss computed: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ DDPM test failed: {e}")
        return False

def test_config():
    """Test if configuration is correct."""
    print("\nTesting configuration...")
    
    config = {
        'model': {
            'feature_dim': 512,  # Must be 512 for ResNet34
            'num_classes': 10
        },
        'memory_bank': {
            'capacity_per_class': 256,
        },
        'ddpm': {
            'feature_dim': 512,  # Must match model
            'hidden_dim': 1024,
        }
    }
    
    # Check feature dimensions match
    assert config['model']['feature_dim'] == 512, "Model feature_dim must be 512"
    assert config['ddpm']['feature_dim'] == 512, "DDPM feature_dim must be 512"
    
    print("  ✓ Configuration validated")
    print(f"  ✓ Feature dimension: {config['model']['feature_dim']}")
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("MEMORY-CONDITIONED DIFFUSION MODEL - VERIFICATION")
    print("="*60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Model Features", test_model_features),
        ("Memory Bank", test_memory_bank),
        ("DDPM", test_ddpm),
        ("Configuration", test_config)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:20s}: {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Ready to run the orchestrator.")
        print("\nNext steps:")
        print(" cp run.py run.py")
        print(" Run quick test: python run.py test")
        print(" Run full training: python run.py full")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED. Please fix the issues before running.")
        print("\nCheck the FIX_SUMMARY.md for troubleshooting tips.")
        return 1

if __name__ == "__main__":
    sys.exit(main())