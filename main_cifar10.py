#!/usr/bin/env python3
"""
Modified main.py that works with CIFAR-10-LT for immediate testing.
Use this to test your memory bank pipeline without waiting for ImageNet download.
"""

import torch
import torch.optim as optim
from models import ResNet32  # Use ResNet32 for CIFAR-10
from create_cifar10_lt import create_cifar10_lt_loaders
from utils import CSLLossFunc, plot_loss_curve, plot_accuracy_curve, plot_validation_accuracy
from utils.memory_manager import MemoryManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(model, criterion, optimizer, scheduler, train_loader, val_loader, device, epoch, memory_manager=None):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass with feature extraction
        if memory_manager is not None:
            outputs, features = model(inputs, return_features=True)
            # Update memory bank with features
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
        
        # Print memory bank status every 50 batches
        if memory_manager is not None and batch_idx % 50 == 0:
            print(f"Batch {batch_idx}: Memory updates: {memory_manager.update_stats['total_updates']}")

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(labels, outputs, epoch) 

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / total
    scheduler.step()

    return train_loss, train_accuracy, val_loss, val_accuracy

def main_cifar10(batch_size=256, num_epochs=50, learning_rate=0.01, 
                 use_memory_bank=True, memory_capacity=64, memory_alpha=0.1, memory_tail_threshold=20.0):
    """
    Main function for CIFAR-10-LT training with memory bank.
    """
    print("🚀 CIFAR-10-LT Training with Memory Bank")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CIFAR-10-LT data
    print("📥 Creating CIFAR-10-LT dataset...")
    train_loader, val_loader, test_loader = create_cifar10_lt_loaders(
        batch_size=batch_size,
        imbalance_factor=0.01  # Very imbalanced for testing
    )
    
    num_classes = 10
    target_class_index = list(range(num_classes))  # All classes are targets
    print(f"Number of classes: {num_classes}")

    # Initialize model (ResNet32 for CIFAR-10)
    print("🤖 Initializing model...")
    model = ResNet32(num_classes=num_classes).to(device)
    print(f"Model feature dim: {model.get_feature_dim()}")

    # Initialize memory manager if requested
    memory_manager = None
    if use_memory_bank:
        print("🧠 Initializing memory bank...")
        memory_manager = MemoryManager(
            model=model,
            num_classes=num_classes,
            capacity_per_class=memory_capacity,
            alpha_base=memory_alpha,
            tail_threshold_percentile=memory_tail_threshold,
            device=device,
            save_dir=f'./memory_checkpoints/cifar10_resnet32'
        )
        print("✅ Memory bank initialized.")
        print(f"Feature dimension: {model.get_feature_dim()}")
        print(f"Memory capacity per class: {memory_capacity}")
        print(f"EMA alpha base: {memory_alpha}")
        print(f"Tail threshold percentile: {memory_tail_threshold}%")

    # Initialize training components
    criterion = CSLLossFunc(target_class_index=target_class_index, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print("✅ Loss function, optimizer, and scheduler initialized.")

    # Training loop
    print(f"\n🏃 Starting training for {num_epochs} epochs...")
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        train_loss, train_accuracy, val_loss, val_accuracy = train(
            model, criterion, optimizer, scheduler, train_loader, val_loader, device, epoch, memory_manager
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Memory bank analysis every 10 epochs
        if memory_manager is not None and (epoch + 1) % 10 == 0:
            print(f"\n📊 Memory Bank Status (Epoch {epoch+1}):")
            
            # Save memory bank
            save_path = memory_manager.save_memory(epoch + 1, f"cifar10_resnet32")
            print(f"💾 Memory bank saved: {save_path}")
            
            # Print summary
            memory_manager.print_summary()
            
            # Show tail class information
            tail_classes = memory_manager.get_tail_classes()
            head_classes = memory_manager.get_head_classes()
            
            print(f"🎯 Tail classes: {tail_classes} (should be [7,8,9] for CIFAR-10)")
            print(f"📈 Head classes: {head_classes} (should be [0,1,2] for CIFAR-10)")
            
            if tail_classes:
                # Export tail class analysis
                export_path = f"./tail_class_analysis_cifar10_epoch_{epoch+1}.json"
                memory_manager.export_tail_class_analysis(export_path)
                print(f"📋 Tail analysis exported: {export_path}")

    # Final memory bank save and visualization  
    if memory_manager is not None:
        print(f"\n🏁 Final Training Summary:")
        final_save_path = memory_manager.save_memory(num_epochs, "cifar10_resnet32_final")
        print(f"💾 Final memory bank: {final_save_path}")
        
        memory_manager.print_summary()
        memory_manager.visualize_memory(f"./memory_visualization_cifar10_final.png")
        
        # Final tail class analysis
        final_export_path = "./tail_class_analysis_cifar10_final.json"
        memory_manager.export_tail_class_analysis(final_export_path)
        print(f"📊 Final analysis: {final_export_path}")

    # Plot training curves
    print("\n📈 Generating training plots...")
    plot_loss_curve(train_losses, val_losses)
    plot_accuracy_curve(train_accuracies, val_accuracies)
    plot_validation_accuracy(val_accuracies, num_epochs)
    
    print("\n🎉 CIFAR-10-LT training completed!")
    print("\n🚀 Next steps:")
    print("1. Check tail class analysis files")
    print("2. Test prompt generation: python test_prompt_generation_cifar10.py")
    print("3. Generate synthetic images for CIFAR-10 tail classes")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CIFAR-10-LT training with memory bank')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')  
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--use_memory_bank', action='store_true', default=True, help='Enable memory bank')
    parser.add_argument('--memory_capacity', type=int, default=64, help='Memory per class')
    parser.add_argument('--memory_alpha', type=float, default=0.1, help='EMA alpha')
    parser.add_argument('--memory_tail_threshold', type=float, default=20.0, help='Tail threshold %')

    args = parser.parse_args()
    main_cifar10(args.batch_size, args.num_epochs, args.learning_rate,
                 args.use_memory_bank, args.memory_capacity, args.memory_alpha, args.memory_tail_threshold)