"""
Main training script for CIFAR-10-LT with Phase 5 synthetic feature augmentation
Integrates synthetic features to balance tail classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
from collections import defaultdict

from models import ResNet32, ResNet50
from utils import CSLLossFunc, plot_loss_curve, plot_accuracy_curve
from utils.memory_manager import MemoryManager
from utils.synthetic_training_integration import ConfidenceAdaptiveCSL


def create_cifar10_lt(train_dataset, imbalance_ratio=100):
    """Create long-tail version of CIFAR-10."""
    print("\n" + "="*70)
    print("CREATING CIFAR-10-LT (LONG-TAIL) DATASET")
    print("="*70)
    
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    img_max = len(class_indices[0])
    img_num_per_cls = []
    
    print(f"Original samples per class: {img_max}")
    print(f"Imbalance ratio: {imbalance_ratio}:1 (head:tail)")
    print(f"\nClass distribution after imbalancing:")
    print("-"*70)
    
    for cls_idx in range(10):
        num = img_max * (imbalance_ratio ** (-cls_idx / (10 - 1.0)))
        img_num_per_cls.append(int(num))
    
    selected_indices = []
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for cls_idx, num_samples in enumerate(img_num_per_cls):
        indices = class_indices[cls_idx]
        np.random.seed(42)
        np.random.shuffle(indices)
        selected_indices.extend(indices[:num_samples])
        
        if num_samples > 1000:
            class_type = "HEAD "
        elif num_samples < 200:
            class_type = "TAIL "
        else:
            class_type = "MEDIUM"
        
        print(f"Class {cls_idx} ({class_names[cls_idx]:10s}): {num_samples:4d} samples [{class_type}]")
    
    total_samples = sum(img_num_per_cls)
    print("-"*70)
    print(f"Total training samples: {total_samples} (original: {len(train_dataset)})")
    print(f"Reduction: {(1 - total_samples/len(train_dataset))*100:.1f}%")
    print("="*70 + "\n")
    
    return Subset(train_dataset, selected_indices), img_num_per_cls


def load_synthetic_features(synthetic_features_path, confidence_threshold=0.5):
    """Load and filter synthetic features by confidence."""
    print("\n" + "="*70)
    print("LOADING SYNTHETIC FEATURES")
    print("="*70)
    
    if not os.path.exists(synthetic_features_path):
        print(f"✗ Synthetic features not found: {synthetic_features_path}")
        return None
    
    data = torch.load(synthetic_features_path)
    
    synthetic_data = {}
    total_features = 0
    filtered_features = 0
    
    for class_id_str, class_data in data['classes'].items():
        class_id = int(class_id_str)
        features = class_data['features']
        confidences = class_data['confidences']
        
        # Normalize features
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        
        # Filter by confidence
        high_conf_mask = confidences >= confidence_threshold
        filtered_features_cls = features[high_conf_mask]
        filtered_confidences = confidences[high_conf_mask]
        
        total_features += len(features)
        filtered_features += len(filtered_features_cls)
        
        if len(filtered_features_cls) > 0:
            synthetic_data[class_id] = {
                'features': filtered_features_cls,
                'confidences': filtered_confidences,
                'mean_confidence': class_data['mean_confidence']
            }
            
            print(f"Class {class_id}: {len(filtered_features_cls)}/{len(features)} features "
                  f"(confidence: {class_data['mean_confidence']:.4f})")
    
    print("-"*70)
    print(f"Total synthetic features: {filtered_features}/{total_features}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Classes with synthetic data: {len(synthetic_data)}")
    print("="*70)
    
    return synthetic_data


def train_epoch_with_synthetic(model, criterion, optimizer, train_loader, 
                                device, epoch, memory_manager, synthetic_data,
                                synthetic_ratio=0.3):
    """
    Training epoch with synthetic feature augmentation.
    
    FIXED VERSION - No more bugs!
    
    Args:
        synthetic_ratio: Total fraction of synthetic samples relative to real samples
                        Example: 0.5 means add 50% synthetic (128 real + 64 synthetic)
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    real_samples = 0
    synthetic_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Extract features from real images
        outputs, real_features = model(inputs, return_features=True)
        
        # Update memory bank with real samples
        if memory_manager is not None:
            memory_manager.update_memory(inputs, labels)
        
        real_samples += len(inputs)
        
        # Prepare for mixed batch with synthetic features
        all_outputs = [outputs]  # Start with real outputs
        all_labels = [labels]
        real_confidences = torch.ones(len(labels), device=device)
        confidences_list = [real_confidences]
        is_synthetic_list = [torch.zeros(len(labels), dtype=torch.bool, device=device)]
        
        # ========== FIXED: Add synthetic features ==========
        if synthetic_data is not None and len(synthetic_data) > 0:
            # Total synthetic budget for this batch
            total_synthetic_budget = int(len(inputs) * synthetic_ratio)
            
            # Distribute evenly across tail classes with synthetic data
            num_syn_per_class = max(1, total_synthetic_budget // len(synthetic_data))
            
            for class_id, syn_data in synthetic_data.items():
                if len(syn_data['features']) == 0:
                    continue
                
                # Sample from available synthetic features
                # Use replacement if we need more than available
                num_available = len(syn_data['features'])
                
                if num_syn_per_class <= num_available:
                    # Sample without replacement
                    indices = torch.randperm(num_available)[:num_syn_per_class]
                else:
                    # Sample with replacement (repeat features if needed)
                    indices = torch.randint(0, num_available, (num_syn_per_class,))
                
                syn_features = syn_data['features'][indices].to(device)
                syn_confidences = syn_data['confidences'][indices].to(device)
                syn_labels = torch.full((len(indices),), class_id, dtype=torch.long, device=device)
                
                # FIXED: Pass synthetic features through classifier head only
                # No torch.no_grad() - we want gradients!
                syn_outputs = model.fc(syn_features)
                
                all_outputs.append(syn_outputs)
                all_labels.append(syn_labels)
                confidences_list.append(syn_confidences)
                is_synthetic_list.append(torch.ones(len(indices), dtype=torch.bool, device=device))
                
                synthetic_samples += len(indices)
        # ========== END FIX ==========
        
        # Combine real and synthetic
        combined_outputs = torch.cat(all_outputs, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        combined_confidences = torch.cat(confidences_list, dim=0)
        combined_is_synthetic = torch.cat(is_synthetic_list, dim=0)
        
        # Forward pass through classifier head - now batch sizes match!
        optimizer.zero_grad()
        
        # Compute loss with confidence weighting
        if isinstance(criterion, ConfidenceAdaptiveCSL):
            loss = criterion(
                combined_labels, combined_outputs, epoch,
                sample_confidences=combined_confidences,
                is_synthetic=combined_is_synthetic
            )
        else:
            loss = criterion(combined_labels, combined_outputs, epoch)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item() * len(combined_labels)
        _, predicted = combined_outputs.max(1)
        total += len(combined_labels)
        correct += predicted.eq(combined_labels).sum().item()
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f} "
                  f"[Real: {real_samples}, Syn: {synthetic_samples}]")
    
    train_loss /= total
    train_accuracy = 100. * correct / total
    
    print(f"\nEpoch {epoch} Training Summary:")
    print(f"  Real samples processed: {real_samples}")
    print(f"  Synthetic samples used: {synthetic_samples}")
    if real_samples + synthetic_samples > 0:
        print(f"  Synthetic ratio: {synthetic_samples / (real_samples + synthetic_samples) * 100:.1f}%")
    
    return train_loss, train_accuracy


def validate(model, criterion, val_loader, device, epoch):
    """Standard validation."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(labels, outputs, epoch)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class stats
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1
    
    val_loss /= total
    val_accuracy = 100. * correct / total
    
    # Print per-class accuracy for tail classes
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    tail_classes = [6, 7, 8, 9]  # frog, horse, ship, truck
    
    print(f"\n  Tail Class Accuracy:")
    for cls_id in tail_classes:
        if class_total[cls_id] > 0:
            cls_acc = 100. * class_correct[cls_id] / class_total[cls_id]
            print(f"    Class {cls_id} ({class_names[cls_id]:10s}): {cls_acc:.2f}% ({class_correct[cls_id]}/{class_total[cls_id]})")
    
    return val_loss, val_accuracy


def main(model_name='resnet32', batch_size=128, num_epochs=50, learning_rate=0.01,
         use_memory_bank=True, memory_capacity=256, memory_alpha=0.1, 
         memory_tail_threshold=20.0, use_long_tail=True, imbalance_ratio=100,
         use_synthetic_features=False, synthetic_features_path='./synthetic_features_phase5.pth',
         synthetic_ratio=0.3, confidence_threshold=0.5, synthetic_weight=0.6):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10 Data Loading
    print("\nLoading CIFAR-10 dataset...")
    
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
    
    train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True, 
                                          transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, 
                                   transform=transform_test)
    
    # Create long-tail version
    if use_long_tail:
        train_dataset, class_distribution = create_cifar10_lt(train_dataset_full, imbalance_ratio)
        dataset_suffix = f"_lt{imbalance_ratio}_synthetic"
    else:
        train_dataset = train_dataset_full
        dataset_suffix = "_synthetic"
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    num_classes = 10
    target_class_index = list(range(10))
    
    print(f"\nDataset Configuration:")
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Long-tail: {use_long_tail}")
    print(f"  Synthetic augmentation: {use_synthetic_features}")

    # Load synthetic features
    synthetic_data = None
    if use_synthetic_features:
        synthetic_data = load_synthetic_features(synthetic_features_path, confidence_threshold)
        if synthetic_data is None:
            print("\n⚠ Warning: Continuing without synthetic features")
            use_synthetic_features = False

    # Initialize model
    if model_name == 'resnet32':
        model = ResNet32(num_classes=num_classes).to(device)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    print(f"\nModel initialized: {model_name}")

    # Initialize memory manager
    memory_manager = None
    if use_memory_bank:
        print("\nInitializing memory bank...")
        save_dir = f'./memory_checkpoints/cifar10{dataset_suffix}_{model_name}'
        memory_manager = MemoryManager(
            model=model,
            num_classes=num_classes,
            capacity_per_class=memory_capacity,
            alpha_base=memory_alpha,
            tail_threshold_percentile=memory_tail_threshold,
            device=device,
            save_dir=save_dir
        )
        print(f"  Memory bank directory: {save_dir}")

    # Initialize loss function
    if use_synthetic_features:
        print(f"\nUsing ConfidenceAdaptiveCSL (synthetic_weight={synthetic_weight})")
        criterion = ConfidenceAdaptiveCSL(
            target_class_index=target_class_index,
            num_classes=num_classes,
            synthetic_weight=synthetic_weight,
            confidence_threshold=confidence_threshold
        ).to(device)
    else:
        print("\nUsing standard CSL")
        criterion = CSLLossFunc(
            target_class_index=target_class_index,
            num_classes=num_classes
        ).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\nOptimizer: SGD (lr={learning_rate}, momentum=0.9)")
    print(f"Scheduler: CosineAnnealingLR (T_max={num_epochs})")

    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING WITH SYNTHETIC AUGMENTATION")
    print("="*70)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Training with synthetic augmentation
        if use_synthetic_features:
            train_loss, train_accuracy = train_epoch_with_synthetic(
                model, criterion, optimizer, train_loader, device, epoch,
                memory_manager, synthetic_data, synthetic_ratio
            )
        else:
            # Standard training without synthetic features
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                if memory_manager is not None:
                    outputs, features = model(inputs, return_features=True)
                    memory_manager.update_memory(inputs, labels)
                else:
                    outputs = model(inputs)
                
                loss = criterion(labels, outputs, epoch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
            
            train_loss /= total
            train_accuracy = 100. * correct / total
        
        # Validation
        val_loss, val_accuracy = validate(model, criterion, val_loader, device, epoch)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"\n  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Print synthetic stats if available
        if use_synthetic_features and isinstance(criterion, ConfidenceAdaptiveCSL):
            stats = criterion.get_synthetic_stats()
            print(f"  Synthetic stats: {stats['used_samples']} used, "
                  f"{stats['confidence_filtered']} filtered")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_checkpoint = f'./checkpoints/cifar10{dataset_suffix}_{model_name}_best.pth'
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'num_classes': num_classes
            }, best_checkpoint)
            print(f"  ✓ New best model saved: {val_accuracy:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'./checkpoints/cifar10{dataset_suffix}_{model_name}_epoch_{epoch+1}.pth'
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        if memory_manager is not None:
            memory_manager.save_memory(epoch + 1, f"cifar10{dataset_suffix}_{model_name}")
    
    # Final save
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    final_checkpoint = f'./checkpoints/cifar10{dataset_suffix}_{model_name}_final.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'best_val_acc': best_val_acc
    }, final_checkpoint)
    print(f"\n✓ Final checkpoint: {final_checkpoint}")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    
    if memory_manager is not None:
        final_memory = memory_manager.save_memory(num_epochs, f"cifar10{dataset_suffix}_{model_name}_final")
        print(f"✓ Final memory bank: {final_memory}")

    # Plot curves
    try:
        plot_loss_curve(train_losses, val_losses)
        plot_accuracy_curve(train_accuracies, val_accuracies)
        print("✓ Training curves plotted")
    except Exception as e:
        print(f"Could not generate plots: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CIFAR-10-LT with Synthetic Feature Augmentation')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='resnet32', 
                       choices=['resnet32', 'resnet50'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    
    # Memory bank arguments
    parser.add_argument('--use_memory_bank', action='store_true')
    parser.add_argument('--memory_capacity', type=int, default=256)
    parser.add_argument('--memory_alpha', type=float, default=0.1)
    parser.add_argument('--memory_tail_threshold', type=float, default=20.0)
    
    # Long-tail dataset arguments
    parser.add_argument('--use_long_tail', action='store_true')
    parser.add_argument('--imbalance_ratio', type=int, default=100)
    
    # Synthetic augmentation arguments
    parser.add_argument('--use_synthetic_features', action='store_true',
                       help='Enable Phase 5 synthetic feature augmentation')
    parser.add_argument('--synthetic_features_path', type=str,
                       default='./synthetic_features_phase5.pth')
    parser.add_argument('--synthetic_ratio', type=float, default=0.3,
                       help='Ratio of synthetic to real samples (0.3 = 30%% synthetic)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--synthetic_weight', type=float, default=0.6,
                       help='Weight factor for synthetic samples in loss')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CIFAR-10-LT TRAINING WITH SYNTHETIC AUGMENTATION")
    print("="*70)
    print(f"Dataset: {'CIFAR-10-LT' if args.use_long_tail else 'CIFAR-10'}")
    if args.use_long_tail:
        print(f"Imbalance Ratio: {args.imbalance_ratio}:1")
    print(f"Model: {args.model_name}")
    print(f"Synthetic Features: {'Enabled' if args.use_synthetic_features else 'Disabled'}")
    if args.use_synthetic_features:
        print(f"  Synthetic Ratio: {args.synthetic_ratio}")
        print(f"  Confidence Thresh: {args.confidence_threshold}")
        print(f"  Synthetic Weight: {args.synthetic_weight}")
    print("="*70 + "\n")
    
    main(args.model_name, args.batch_size, args.num_epochs, args.learning_rate,
         args.use_memory_bank, args.memory_capacity, args.memory_alpha, 
         args.memory_tail_threshold, args.use_long_tail, args.imbalance_ratio,
         args.use_synthetic_features, args.synthetic_features_path,
         args.synthetic_ratio, args.confidence_threshold, args.synthetic_weight)