#!/usr/bin/env python3
"""
Create CIFAR-10-LT dataset for testing the memory bank pipeline.
This is much smaller and downloads automatically.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
from pathlib import Path
import json

class CIFAR10LT(Dataset):
    """
    CIFAR-10 Long-Tail dataset.
    Creates imbalanced version of CIFAR-10 for testing.
    """
    
    def __init__(self, root='./data', train=True, download=True, transform=None, imbalance_factor=0.1):
        """
        Args:
            imbalance_factor: Controls tail length (0.1 = very imbalanced, 1.0 = balanced)
        """
        
        # Download regular CIFAR-10
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        
        self.transform = transform
        self.num_classes = 10
        self.imbalance_factor = imbalance_factor
        
        # Create long-tail distribution
        self.indices = self._create_long_tail_indices()
        
        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        print(f"Created CIFAR-10-LT with {len(self.indices)} samples")
        self._print_class_distribution()
    
    def _create_long_tail_indices(self):
        """Create indices for long-tail distribution."""
        
        # Get all labels
        targets = np.array(self.cifar10.targets)
        
        # Calculate samples per class (exponential decay)
        samples_per_class = []
        for i in range(self.num_classes):
            # Exponential decay: head classes get many samples, tail get few
            num_samples = int(len(targets) // self.num_classes * (self.imbalance_factor ** (i / (self.num_classes - 1))))
            samples_per_class.append(max(num_samples, 10))  # Minimum 10 samples per class
        
        print(f"Samples per class: {samples_per_class}")
        
        # Sample indices according to this distribution
        selected_indices = []
        
        for class_id in range(self.num_classes):
            class_indices = np.where(targets == class_id)[0]
            num_samples = min(samples_per_class[class_id], len(class_indices))
            
            # Randomly sample from this class
            selected = np.random.choice(class_indices, size=num_samples, replace=False)
            selected_indices.extend(selected)
        
        return selected_indices
    
    def _print_class_distribution(self):
        """Print distribution statistics."""
        targets = np.array(self.cifar10.targets)
        selected_targets = targets[self.indices]
        
        print("\n📊 CIFAR-10-LT Class Distribution:")
        print("Class ID | Class Name    | Samples | Type")
        print("-" * 45)
        
        class_counts = []
        for i in range(self.num_classes):
            count = np.sum(selected_targets == i)
            class_counts.append(count)
            
            # Determine if head, medium, or tail
            if i < 3:
                class_type = "HEAD"
            elif i < 7:
                class_type = "MEDIUM" 
            else:
                class_type = "TAIL"
            
            print(f"{i:8d} | {self.class_names[i]:12s} | {count:7d} | {class_type}")
        
        total = len(selected_targets)
        tail_samples = sum(class_counts[7:])  # Last 3 classes
        print(f"\nTotal samples: {total}")
        print(f"Tail classes (7-9): {tail_samples} samples ({tail_samples/total*100:.1f}%)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.cifar10[real_idx]
    
    def get_unique_labels(self):
        """Get unique labels (for compatibility with your main.py)."""
        return list(range(self.num_classes))


def create_cifar10_lt_loaders(batch_size=256, imbalance_factor=0.01):
    """
    Create CIFAR-10-LT data loaders.
    
    Args:
        batch_size: Batch size
        imbalance_factor: How imbalanced (0.01 = very imbalanced)
    """
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Create datasets
    train_dataset = CIFAR10LT(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train,
        imbalance_factor=imbalance_factor
    )
    
    # Use regular CIFAR-10 for validation (balanced)
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # For compatibility, create same val for test
    test_loader = val_loader
    
    return train_loader, val_loader, test_loader


def test_cifar10_lt():
    """Test CIFAR-10-LT creation."""
    print("🧪 Testing CIFAR-10-LT Creation")
    print("=" * 40)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_cifar10_lt_loaders(
        batch_size=64, 
        imbalance_factor=0.01  # Very imbalanced
    )
    
    print(f"✅ Train loader: {len(train_loader)} batches")
    print(f"✅ Val loader: {len(val_loader)} batches") 
    print(f"✅ Test loader: {len(test_loader)} batches")
    
    # Test a batch
    for inputs, labels in train_loader:
        print(f"✅ Batch shape: {inputs.shape}")
        print(f"✅ Label range: {labels.min()}-{labels.max()}")
        break
    
    print("\n🎉 CIFAR-10-LT ready for memory bank testing!")


if __name__ == "__main__":
    test_cifar10_lt()