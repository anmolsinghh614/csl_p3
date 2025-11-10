"""
Main training script with HYBRID Phase 3 + Phase 5 support
Drop-in replacement for main_with_synthetic_cifar10.py
Integrates with your existing memory bank, prompt generation, and image generation
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

# Import hybrid pipeline
from hybrid_synthetic_pipeline import (
    HybridSyntheticFeatureGenerator,
    train_epoch_with_hybrid_features
)

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


def validate(model, criterion, val_loader, device, epoch):
    """Standard validation."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
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
            
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1
    
    val_loss /= total
    val_accuracy = 100. * correct / total
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    tail_classes = [6, 7, 8, 9]
    
    print(f"\n  Tail Class Accuracy:")
    for cls_id in tail_classes:
        if class_total[cls_id] > 0:
            cls_acc = 100. * class_correct[cls_id] / class_total[cls_id]
            print(f"    Class {cls_id} ({class_names[cls_id]:10s}): {cls_acc:.2f}% ({class_correct[cls_id]}/{class_total[cls_id]})")
    
    return val_loss, val_accuracy


def main(model_name='resnet32', batch_size=128, num_epochs=50, learning_rate=0.01,
         use_memory_bank=True, memory_capacity=256, memory_alpha=0.1, 
         memory_tail_threshold=20.0, use_long_tail=True, imbalance_ratio=100,
         # Hybrid synthetic arguments
         use_synthetic_features=False,
         generation_method='hybrid',  # 'phase3', 'phase5', or 'hybrid'
         phase3_model_path=None,
         phase5_features_path=None,
         synthetic_ratio=0.3,
         phase3_ratio=0.5,
         confidence_threshold=0.5,
         synthetic_weight=0.6):
    
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
    
    if use_long_tail:
        train_dataset, class_distribution = create_cifar10_lt(train_dataset_full, imbalance_ratio)
        dataset_suffix = f"_lt{imbalance_ratio}_hybrid"
    else:
        train_dataset = train_dataset_full
        dataset_suffix = "_hybrid"
    
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
    if use_synthetic_features:
        print(f"  Generation method: {generation_method}")

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

    # ========== INITIALIZE HYBRID GENERATOR ==========
    hybrid_generator = None
    if use_synthetic_features:
        print("\n" + "="*70)
        print("INITIALIZING HYBRID SYNTHETIC PIPELINE")
        print("="*70)
        
        hybrid_generator = HybridSyntheticFeatureGenerator(
            memory_manager=memory_manager,
            phase3_model_path=phase3_model_path,
            phase5_features_path=phase5_features_path,
            device=device
        )
        
        # Print statistics
        stats = hybrid_generator.get_statistics()
        print(f"\nHybrid Pipeline Configuration:")
        print(f"  Generation method: {generation_method}")
        print(f"  Synthetic ratio: {synthetic_ratio}")
        if generation_method == 'hybrid':
            print(f"  Phase 3 ratio: {phase3_ratio}")
            print(f"  Phase 5 ratio: {1 - phase3_ratio}")
        print("="*70)
    # ========== END INITIALIZATION ==========

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
    print("STARTING TRAINING WITH HYBRID SYNTHETIC PIPELINE")
    print("="*70)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # ========== TRAINING WITH HYBRID FEATURES ==========
        if use_synthetic_features and hybrid_generator is not None:
            train_loss, train_accuracy = train_epoch_with_hybrid_features(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                epoch=epoch,
                hybrid_generator=hybrid_generator,
                tail_classes=[6, 7, 8, 9],
                synthetic_ratio=synthetic_ratio,
                generation_method=generation_method,
                phase3_ratio=phase3_ratio
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
        # ========== END TRAINING ==========
        
        # Validation
        val_loss, val_accuracy = validate(model, criterion, val_loader, device, epoch)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"\n  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
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
    
    parser = argparse.ArgumentParser(description='Train CIFAR-10-LT with Hybrid Synthetic Features')
    
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
    
    # Hybrid synthetic augmentation arguments
    parser.add_argument('--use_synthetic_features', action='store_true',
                       help='Enable hybrid synthetic feature augmentation')
    parser.add_argument('--generation_method', type=str, default='hybrid',
                       choices=['phase3', 'phase5', 'hybrid'],
                       help='Which method to use: phase3 (DDPM), phase5 (SD), or hybrid (both)')
    parser.add_argument('--phase3_model_path', type=str, default=None,
                       help='Path to trained Phase 3 DDPM model')
    parser.add_argument('--phase5_features_path', type=str, default=None,
                       help='Path to Phase 5 extracted features')
    parser.add_argument('--synthetic_ratio', type=float, default=0.3,
                       help='Ratio of synthetic to real samples')
    parser.add_argument('--phase3_ratio', type=float, default=0.5,
                       help='If hybrid, fraction from Phase 3 (rest from Phase 5)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--synthetic_weight', type=float, default=0.6,
                       help='Weight factor for synthetic samples in loss')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CIFAR-10-LT TRAINING WITH HYBRID SYNTHETIC PIPELINE")
    print("="*70)
    print(f"Dataset: {'CIFAR-10-LT' if args.use_long_tail else 'CIFAR-10'}")
    if args.use_long_tail:
        print(f"Imbalance Ratio: {args.imbalance_ratio}:1")
    print(f"Model: {args.model_name}")
    print(f"Synthetic Features: {'Enabled' if args.use_synthetic_features else 'Disabled'}")
    if args.use_synthetic_features:
        print(f"  Generation Method: {args.generation_method}")
        print(f"  Synthetic Ratio: {args.synthetic_ratio}")
        if args.generation_method == 'hybrid':
            print(f"  Phase 3 (DDPM) Ratio: {args.phase3_ratio}")
            print(f"  Phase 5 (SD) Ratio: {1 - args.phase3_ratio}")
        print(f"  Confidence Threshold: {args.confidence_threshold}")
        print(f"  Synthetic Weight: {args.synthetic_weight}")
    print("="*70 + "\n")
    
    main(args.model_name, args.batch_size, args.num_epochs, args.learning_rate,
         args.use_memory_bank, args.memory_capacity, args.memory_alpha, 
         args.memory_tail_threshold, args.use_long_tail, args.imbalance_ratio,
         args.use_synthetic_features, args.generation_method,
         args.phase3_model_path, args.phase5_features_path,
         args.synthetic_ratio, args.phase3_ratio,
         args.confidence_threshold, args.synthetic_weight)