#!/usr/bin/env python3
"""
Demonstration script for the Memory Bank implementation.
Shows how to integrate it with the training pipeline and extract tail class information.
"""

import torch
import torch.nn as nn
from models import ResNet50
from utils.memory_manager import MemoryManager
import matplotlib.pyplot as plt

def create_demo_model(num_classes=1000):
    """Create a demo ResNet50 model."""
    model = ResNet50(num_classes=num_classes)
    print(f"Model created with {num_classes} classes")
    print(f"Feature dimension: {model.get_feature_dim()}")
    return model

def simulate_training_data(num_classes=1000, samples_per_class=100):
    """Simulate imbalanced training data."""
    # Create imbalanced distribution (power law)
    import numpy as np
    np.random.seed(42)
    
    # Generate class frequencies following power law distribution
    class_frequencies = np.random.power(3, num_classes) * samples_per_class
    class_frequencies = np.maximum(class_frequencies, 10)  # Minimum 10 samples per class
    
    print(f"Generated imbalanced data distribution:")
    print(f"  Most frequent class: {int(np.max(class_frequencies))} samples")
    print(f"  Least frequent class: {int(np.min(class_frequencies))} samples")
    print(f"  Total samples: {int(np.sum(class_frequencies))}")
    
    return class_frequencies

def demo_memory_bank_integration():
    """Demonstrate memory bank integration with training."""
    print("="*70)
    print("MEMORY BANK INTEGRATION DEMONSTRATION")
    print("="*70)
    
    # Configuration
    num_classes = 100
    device = 'cpu'  # Use CPU for demo
    
    # Create model
    print("\n1. Creating model...")
    model = create_demo_model(num_classes)
    
    # Initialize memory manager
    print("\n2. Initializing memory manager...")
    memory_manager = MemoryManager(
        model=model,
        num_classes=num_classes,
        capacity_per_class=128,  # Store 128 features per class
        alpha_base=0.1,          # EMA learning rate
        tail_threshold_percentile=20.0,  # 20% of classes are "tail"
        device=device,
        save_dir='./demo_memory_checkpoints'
    )
    
    # Simulate training data distribution
    print("\n3. Simulating training data...")
    class_frequencies = simulate_training_data(num_classes, 100)
    
    # Simulate training loop
    print("\n4. Simulating training loop...")
    batch_size = 32
    total_batches = 0
    
    for class_id in range(num_classes):
        num_samples = int(class_frequencies[class_id])
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        print(f"  Class {class_id}: {num_samples} samples ({num_batches} batches)")
        
        for batch_idx in range(num_batches):
            # Generate synthetic batch
            actual_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            inputs = torch.randn(actual_batch_size, 3, 224, 224)
            labels = torch.full((actual_batch_size,), class_id, dtype=torch.long)
            
            # Update memory bank
            memory_manager.update_memory(inputs, labels)
            total_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx}/{num_batches} - Total updates: {memory_manager.update_stats['total_updates']}")
    
    # Print final summary
    print("\n5. Training complete! Final memory bank status:")
    memory_manager.print_summary()
    
    # Analyze tail classes
    print("\n6. Tail Class Analysis:")
    tail_classes = memory_manager.get_tail_classes()
    print(f"  Found {len(tail_classes)} tail classes")
    
    if tail_classes:
        print("  First 10 tail classes:", tail_classes[:10])
        
        # Get detailed information for first few tail classes
        print("\n  Detailed analysis of first 3 tail classes:")
        for i, class_id in enumerate(tail_classes[:3]):
            stats = memory_manager.get_class_statistics(class_id)
            print(f"    Class {class_id}:")
            print(f"      Frequency: {stats['class_frequency']}")
            print(f"      Percentage: {stats['class_percentage']:.2f}%")
            print(f"      Reservoir count: {stats['reservoir_count']}")
            print(f"      Prototype norm: {stats['prototype_norm']:.4f}")
    
    # Get semantic prompt data
    print("\n7. Semantic Prompt Data Extraction:")
    tail_prompt_data = memory_manager.get_all_tail_prompt_data(k_features_per_class=5)
    print(f"  Extracted prompt data for {len(tail_prompt_data)} tail classes")
    
    if tail_prompt_data:
        first_class = list(tail_prompt_data.keys())[0]
        data = tail_prompt_data[first_class]
        print(f"  Example - Class {first_class}:")
        print(f"    Prototype shape: {data['prototype'].shape}")
        print(f"    Feature count: {data['feature_count']}")
        print(f"    Is tail: {data['is_tail']}")
    
    # Save memory bank
    print("\n8. Saving memory bank...")
    save_path = memory_manager.save_memory(total_batches, "demo_training")
    print(f"  Memory bank saved to: {save_path}")
    
    # Export tail class analysis
    print("\n9. Exporting tail class analysis...")
    export_path = "./demo_tail_class_analysis.json"
    memory_manager.export_tail_class_analysis(export_path)
    print(f"  Analysis exported to: {export_path}")
    
    # Visualization
    print("\n10. Generating visualizations...")
    try:
        memory_manager.visualize_memory("./demo_memory_visualization.png")
        print("  Visualization saved to: ./demo_memory_visualization.png")
    except Exception as e:
        print(f"  Visualization failed: {e}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return memory_manager

def demo_semantic_prompt_generation(memory_manager):
    """Demonstrate how to use the memory bank for semantic prompt generation."""
    print("\n" + "="*70)
    print("SEMANTIC PROMPT GENERATION DEMONSTRATION")
    print("="*70)
    
    # Get tail class prototypes
    tail_prototypes = memory_manager.get_tail_class_prototypes()
    print(f"Retrieved {len(tail_prototypes)} tail class prototypes")
    
    # Example: Generate semantic prompts for tail classes
    print("\nExample semantic prompts for tail classes:")
    for class_id, prototype in list(tail_prototypes.items())[:5]:
        # Calculate prototype characteristics
        prototype_norm = torch.norm(prototype).item()
        prototype_std = torch.std(prototype).item()
        
        print(f"\nClass {class_id}:")
        print(f"  Prototype norm: {prototype_norm:.4f}")
        print(f"  Prototype std: {prototype_std:.4f}")
        
        # Example semantic prompt generation strategies
        print("  Semantic prompt strategies:")
        print(f"    1. Direct: 'a photo of class_{class_id}'")
        print(f"    2. Feature-based: 'an object with features: norm={prototype_norm:.3f}, std={prototype_std:.3f}'")
        
        # Get sample features for diversity analysis
        features = memory_manager.memory_bank.sample_features(class_id, k=3)
        if features:
            feature_norms = [torch.norm(f).item() for f in features]
            print(f"    3. Diversity: sample feature norms: {[f'{n:.3f}' for n in feature_norms]}")
    
    print("\n" + "="*70)
    print("This demonstrates how the memory bank provides:")
    print("1. Tail class identification")
    print("2. Class prototypes for semantic guidance")
    print("3. Feature diversity for generation")
    print("4. Data for training diffusion models")
    print("="*70)

def main():
    """Run the complete demonstration."""
    print("Starting Memory Bank Demonstration")
    print("="*70)
    
    try:
        # Run main demonstration
        memory_manager = demo_memory_bank_integration()
        
        # Demonstrate semantic prompt generation
        demo_semantic_prompt_generation(memory_manager)
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Use the memory bank with real training data")
        print("2. Integrate with diffusion models for feature generation")
        print("3. Generate semantic prompts for image-level synthesis")
        print("4. Use tail class analysis for curriculum learning")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
