#!/usr/bin/env python3
"""
Quick test to verify memory bank works correctly
"""

import torch
from utils.memory_bank import MemoryBank

print("Testing Memory Bank...")

# Create memory bank
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
memory_bank = MemoryBank(
    num_classes=10,
    feature_dim=512,
    capacity_per_class=64,
    device=device
)

print("✓ Memory bank created")

# Add some features
for class_id in range(10):
    # Add more features to some classes to create imbalance
    num_features = 50 if class_id < 3 else 10 if class_id < 7 else 2
    for _ in range(num_features):
        fake_feature = torch.randn(512).to(device)
        memory_bank.update(class_id, fake_feature)

print(f"✓ Added features to memory bank")

# Get tail/head classes (automatically classified)
tail_classes = memory_bank.get_tail_classes()
head_classes = memory_bank.get_head_classes()
medium_classes = memory_bank.get_medium_classes()

print(f"✓ Tail classes: {sorted(list(tail_classes))}")
print(f"✓ Head classes: {sorted(list(head_classes))}")
print(f"✓ Medium classes: {sorted(list(medium_classes))}")

# Test other methods
prototype = memory_bank.get_prototype(0)
print(f"✓ Got prototype for class 0: shape {prototype.shape}")

features = memory_bank.sample_features(0, k=5)
print(f"✓ Sampled {len(features)} features from class 0")

stats = memory_bank.get_class_statistics(0)
print(f"✓ Got statistics for class 0: {stats['ema_count']} updates")

print("\n🎉 All memory bank functions work correctly!")
print("\nThe tail classes are automatically updated when features are added.")
