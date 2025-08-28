#!/usr/bin/env python3
"""
Test script for the Memory Bank implementation.
Demonstrates EMA + Reservoir sampling and tail class identification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.memory_bank import MemoryBank
from utils.memory_manager import MemoryManager

def create_synthetic_model(feature_dim=512, num_classes=100):
    """Create a synthetic model for testing."""
    class SyntheticModel:
        def __init__(self, feature_dim, num_classes):
            self.feature_dim = feature_dim
            self.num_classes = num_classes
        
        def get_feature_dim(self):
            return self.feature_dim
        
        def __call__(self, x, return_features=False):
            batch_size = x.shape[0]
            # Generate random features
            features = torch.randn(batch_size, self.feature_dim)
            # Generate random logits
            logits = torch.randn(batch_size, self.num_classes)
            
            if return_features:
                return logits, features
            return logits
    
    return SyntheticModel(feature_dim, num_classes)

def test_memory_bank_basic():
    """Test basic memory bank functionality."""
    print("="*60)
    print("TESTING BASIC MEMORY BANK FUNCTIONALITY")
    print("="*60)
    
    # Initialize memory bank
    num_classes = 10
    feature_dim = 128
    capacity_per_class = 50
    
    memory_bank = MemoryBank(
        num_classes=num_classes,
        feature_dim=feature_dim,
        capacity_per_class=capacity_per_class,
        alpha_base=0.1,
        tail_threshold_percentile=20.0,
        device='cpu'
    )
    
    print(f"Memory bank initialized with {num_classes} classes, {feature_dim} features")
    print(f"Capacity per class: {capacity_per_class}")
    
    # Simulate imbalanced data distribution
    class_frequencies = [1000, 800, 600, 400, 200, 150, 100, 80, 60, 40]  # Imbalanced
    
    # Update memory bank with synthetic data
    for class_id in range(num_classes):
        num_samples = class_frequencies[class_id]
        print(f"\nUpdating class {class_id} with {num_samples} samples...")
        
        for i in range(num_samples):
            # Generate synthetic feature with some class-specific characteristics
            feature = torch.randn(feature_dim)
            # Add class-specific bias
            feature += class_id * 0.1
            
            memory_bank.update(class_id, feature)
            
            if i % 200 == 0:
                print(f"  Processed {i}/{num_samples} samples")
    
    # Print summary
    print("\n" + "="*60)
    print("MEMORY BANK SUMMARY AFTER UPDATES")
    print("="*60)
    
    for class_id in range(num_classes):
        stats = memory_bank.get_class_statistics(class_id)
        print(f"Class {class_id}:")
        print(f"  Frequency: {stats['class_frequency']}")
        print(f"  Percentage: {stats['class_percentage']:.2f}%")
        print(f"  Is tail: {stats['is_tail']}")
        print(f"  Reservoir count: {stats['reservoir_count']}")
        print(f"  EMA count: {stats['ema_count']}")
        print()
    
    # Show class distribution
    print("Class Distribution:")
    distribution = memory_bank.get_class_distribution()
    for category, classes in distribution.items():
        print(f"  {category.capitalize()}: {len(classes)} classes")
        if classes:
            print(f"    IDs: {classes[:5]}{'...' if len(classes) > 5 else ''}")
    
    return memory_bank

def test_memory_manager():
    """Test memory manager integration."""
    print("\n" + "="*60)
    print("TESTING MEMORY MANAGER INTEGRATION")
    print("="*60)
    
    # Create synthetic model
    model = create_synthetic_model(feature_dim=256, num_classes=20)
    
    # Initialize memory manager
    memory_manager = MemoryManager(
        model=model,
        num_classes=20,
        capacity_per_class=100,
        alpha_base=0.15,
        tail_threshold_percentile=25.0,
        device='cpu',
        save_dir='./test_memory_checkpoints'
    )
    
    print("Memory manager initialized")
    
    # Simulate training batches
    batch_size = 32
    num_batches = 50
    
    print(f"\nSimulating {num_batches} training batches...")
    
    for batch_idx in range(num_batches):
        # Generate synthetic batch
        inputs = torch.randn(batch_size, 3, 224, 224)  # Synthetic images
        labels = torch.randint(0, 20, (batch_size,))  # Random class labels
        
        # Update memory
        memory_manager.update_memory(inputs, labels)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: {memory_manager.update_stats['total_updates']} total updates")
    
    # Print summary
    memory_manager.print_summary()
    
    # Test semantic prompt data extraction
    print("\n" + "="*60)
    print("TESTING SEMANTIC PROMPT DATA EXTRACTION")
    print("="*60)
    
    tail_classes = memory_manager.get_tail_classes()
    if tail_classes:
        print(f"Found {len(tail_classes)} tail classes")
        
        # Get prompt data for first few tail classes
        for class_id in tail_classes[:3]:
            prompt_data = memory_manager.get_semantic_prompt_data(class_id, k_features=5)
            print(f"\nClass {class_id} prompt data:")
            print(f"  Is tail: {prompt_data['is_tail']}")
            print(f"  Class frequency: {prompt_data['class_frequency']}")
            print(f"  Class percentage: {prompt_data['class_percentage']:.2f}%")
            print(f"  Feature count: {prompt_data['feature_count']}")
            print(f"  Prototype norm: {torch.norm(prompt_data['prototype']):.4f}")
    
    # Test saving and loading
    print("\n" + "="*60)
    print("TESTING SAVE/LOAD FUNCTIONALITY")
    print("="*60)
    
    save_path = memory_manager.save_memory(100, "test")
    print(f"Memory saved to: {save_path}")
    
    # Create new memory manager and load
    new_memory_manager = MemoryManager(
        model=model,
        num_classes=20,
        device='cpu'
    )
    
    new_memory_manager.load_memory(save_path)
    print("Memory loaded successfully")
    
    # Verify data integrity
    original_tail = set(memory_manager.get_tail_classes())
    loaded_tail = set(new_memory_manager.get_tail_classes())
    
    print(f"Original tail classes: {len(original_tail)}")
    print(f"Loaded tail classes: {len(loaded_tail)}")
    print(f"Tail classes match: {original_tail == loaded_tail}")
    
    return memory_manager

def test_visualization():
    """Test memory bank visualization."""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION")
    print("="*60)
    
    # Create a memory bank with more realistic data
    num_classes = 50
    feature_dim = 256
    capacity_per_class = 100
    
    memory_bank = MemoryBank(
        num_classes=num_classes,
        feature_dim=feature_dim,
        capacity_per_class=capacity_per_class,
        alpha_base=0.1,
        tail_threshold_percentile=20.0,
        device='cpu'
    )
    
    # Generate imbalanced data
    np.random.seed(42)
    class_frequencies = np.random.power(3, num_classes) * 1000  # Power law distribution
    
    print("Generating imbalanced synthetic data...")
    
    for class_id in range(num_classes):
        num_samples = int(class_frequencies[class_id])
        
        for i in range(num_samples):
            feature = torch.randn(feature_dim)
            feature += class_id * 0.05  # Slight class bias
            memory_bank.update(class_id, feature)
    
    print("Data generation complete")
    
    # Test visualization
    print("Generating visualization...")
    try:
        memory_bank.visualize_class_distribution("./test_visualization.png")
        print("Visualization saved to ./test_visualization.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Print final statistics
    print(f"\nFinal statistics:")
    print(f"Total samples: {memory_bank.total_samples}")
    print(f"Tail classes: {len(memory_bank.get_tail_classes())}")
    print(f"Head classes: {len(memory_bank.get_head_classes())}")
    print(f"Medium classes: {len(memory_bank.get_medium_classes())}")
    
    return memory_bank

def main():
    """Run all tests."""
    print("Starting Memory Bank Tests")
    print("="*60)
    
    try:
        # Test 1: Basic memory bank
        memory_bank = test_memory_bank_basic()
        
        # Test 2: Memory manager
        memory_manager = test_memory_manager()
        
        # Test 3: Visualization
        viz_memory_bank = test_visualization()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Final summary
        print("\nFinal Memory Usage Summary:")
        usage = memory_bank.get_memory_usage()
        for key, value in usage.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
