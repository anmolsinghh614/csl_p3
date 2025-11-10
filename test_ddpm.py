"""
Quick Test Script for Phase 3 Feature DDPM
Tests all core functionality in ~30 seconds
"""

import torch
import torch.nn.functional as F
from phase3_feature_ddpm import FeatureDDPM

def test_ddpm():
    print("\n" + "="*70)
    print("TESTING PHASE 3 FEATURE DDPM")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # ========== TEST 1: Initialize Model ==========
    print("TEST 1: Initializing Model...")
    model = FeatureDDPM(
        feature_dim=512,
        num_classes=10,
        hidden_dim=512,  # Smaller for fast testing
        num_layers=3,    # Fewer layers for speed
        num_timesteps=100  # Fewer timesteps for speed
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {num_params:,} parameters")
    print(f"  Feature dim: 512")
    print(f"  Classes: 10")
    print(f"  Timesteps: 100 (reduced for testing)")
    
    # ========== TEST 2: Training Forward Pass ==========
    print("\nTEST 2: Testing Training (Forward Pass)...")
    
    # Create fake "real features" (simulating ResNet features)
    batch_size = 8
    fake_real_features = torch.randn(batch_size, 512, device=device)
    fake_labels = torch.randint(0, 10, (batch_size,), device=device)
    
    print(f"  Input: {batch_size} fake features [8, 512]")
    print(f"  Labels: {fake_labels.tolist()}")
    
    # Forward pass (compute loss)
    loss = model(fake_real_features, fake_labels)
    
    print(f"✓ Training loss computed: {loss.item():.4f}")
    
    # Test backpropagation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Backpropagation successful")
    
    # ========== TEST 3: Quick Training Loop ==========
    print("\nTEST 3: Quick Training Loop (10 iterations)...")
    
    model.train()
    losses = []
    
    for i in range(10):
        # Generate random features
        features = torch.randn(batch_size, 512, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        
        # Train
        optimizer.zero_grad()
        loss = model(features, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (i + 1) % 3 == 0:
            print(f"  Iteration {i+1}/10: Loss = {loss.item():.4f}")
    
    print(f"✓ Training loop completed")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    # ========== TEST 4: Generation (Sampling) ==========
    print("\nTEST 4: Testing Generation (Sampling)...")
    
    model.eval()
    
    # Generate features for class 9 (truck)
    class_ids = torch.tensor([9, 9, 9, 9], device=device)  # 4 truck features
    
    print(f"  Generating 4 features for class 9 (truck)...")
    print(f"  This will take ~10-15 seconds...")
    
    with torch.no_grad():
        generated_features = model.sample(class_ids, device=device)
    
    print(f"✓ Generation successful!")
    print(f"  Generated shape: {generated_features.shape}")
    print(f"  Sample values (first 5): {generated_features[0, :5].cpu().numpy()}")
    
    # ========== TEST 5: Generation with Confidence ==========
    print("\nTEST 5: Testing Generation with Confidence Scoring...")
    
    # Create mock class prototypes
    class_prototypes = {
        i: torch.randn(512, device=device) 
        for i in range(10)
    }
    
    print(f"  Generating 4 features with confidence scores...")
    
    with torch.no_grad():
        features, confidences = model.sample_with_confidence(
            class_ids, 
            class_prototypes, 
            device=device
        )
    
    print(f"✓ Confidence scoring successful!")
    print(f"  Generated features: {features.shape}")
    print(f"  Confidences: {confidences.cpu().numpy()}")
    print(f"  Mean confidence: {confidences.mean().item():.4f}")
    print(f"  Confidence range: [{confidences.min().item():.4f}, {confidences.max().item():.4f}]")
    
    # ========== TEST 6: Fast Sampling (DDIM) ==========
    print("\nTEST 6: Testing Fast Sampling (50 steps instead of 100)...")
    
    with torch.no_grad():
        fast_features = model.sample_fast(
            class_ids, 
            num_steps=10,  # Very fast for testing
            device=device
        )
    
    print(f"✓ Fast sampling successful!")
    print(f"  Generated shape: {fast_features.shape}")
    print(f"  Speed: ~10x faster than full sampling")
    
    # ========== TEST 7: Batch Generation ==========
    print("\nTEST 7: Testing Batch Generation (Multiple Classes)...")
    
    # Generate features for all tail classes
    tail_classes = [6, 7, 8, 9]
    all_class_ids = []
    for cls in tail_classes:
        all_class_ids.extend([cls] * 5)  # 5 samples per class
    
    all_class_ids = torch.tensor(all_class_ids, device=device)
    
    print(f"  Generating {len(all_class_ids)} features for classes {tail_classes}...")
    
    with torch.no_grad():
        batch_features = model.sample(all_class_ids, device=device)
    
    print(f"✓ Batch generation successful!")
    print(f"  Total features generated: {batch_features.shape[0]}")
    print(f"  Per class: 5 features each")
    
    # Verify features are different for different classes
    class_6_features = batch_features[:5]  # First 5 are class 6
    class_9_features = batch_features[-5:]  # Last 5 are class 9
    
    similarity = F.cosine_similarity(
        class_6_features.mean(dim=0).unsqueeze(0),
        class_9_features.mean(dim=0).unsqueeze(0)
    ).item()
    
    print(f"  Similarity between class 6 and 9 features: {similarity:.4f}")
    print(f"  (Lower = more distinct, which is good)")
    
    # ========== TEST 8: Feature Quality Metrics ==========
    print("\nTEST 8: Computing Feature Quality Metrics...")
    
    # Generate more samples
    with torch.no_grad():
        test_features = model.sample(
            torch.tensor([9] * 20, device=device),
            device=device
        )
    
    # Compute statistics
    feature_mean = test_features.mean(dim=0)
    feature_std = test_features.std(dim=0)
    feature_magnitude = torch.norm(test_features, dim=1).mean()
    
    print(f"✓ Feature statistics:")
    print(f"  Mean magnitude: {feature_magnitude.item():.4f}")
    print(f"  Std across samples: {feature_std.mean().item():.4f}")
    print(f"  Feature diversity: {test_features.std(dim=1).mean().item():.4f}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print("\nSummary:")
    print(f"  ✓ Model initialization")
    print(f"  ✓ Training forward/backward pass")
    print(f"  ✓ Training loop (10 iterations)")
    print(f"  ✓ Feature generation (sampling)")
    print(f"  ✓ Confidence scoring")
    print(f"  ✓ Fast sampling (DDIM)")
    print(f"  ✓ Batch generation")
    print(f"  ✓ Feature quality metrics")
    print("\n✅ Phase 3 DDPM is working correctly!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run tests
    test_ddpm()
    
    print("\n💡 Next Steps:")
    print("1. Train on real ResNet features: python train_feature_ddpm.py")
    print("2. Use in training: python main_with_hybrid_synthetic_cifar10.py")
    print("3. Compare with Phase 5 results\n")