"""
Main training script for CIFAR-10 with memory bank support
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import ResNet32, ResNet50  # Your models
from utils import CSLLossFunc, plot_loss_curve, plot_accuracy_curve
from utils.memory_manager import MemoryManager

def main(model_name='resnet18', batch_size=128, num_epochs=50, learning_rate=0.01,
         use_memory_bank=True, memory_capacity=256, memory_alpha=0.1, memory_tail_threshold=20.0):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10 Data Loading
    print("Loading CIFAR-10 dataset...")
    
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
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    num_classes = 10
    target_class_index = list(range(10))
    
    print(f"Number of classes: {num_classes}")

    # Initialize model
    if model_name == 'resnet32':
        model = ResNet32(num_classes=num_classes).to(device)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    print("Model initialized.")

    # Initialize memory manager
    memory_manager = None
    if use_memory_bank:
        print("Initializing memory bank...")
        memory_manager = MemoryManager(
            model=model,
            num_classes=num_classes,
            capacity_per_class=memory_capacity,
            alpha_base=memory_alpha,
            tail_threshold_percentile=memory_tail_threshold,
            device=device,
            save_dir=f'./memory_checkpoints/cifar10_{model_name}'
        )
        print("Memory bank initialized.")

    criterion = CSLLossFunc(target_class_index=target_class_index, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print("Loss function, optimizer, and scheduler initialized.")

    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
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
                print(f"Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
        
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
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }
            checkpoint_path = f'./checkpoints/cifar10_{model_name}_epoch_{epoch+1}.pth'
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            if memory_manager is not None:
                memory_manager.save_memory(epoch + 1, f"cifar10_{model_name}")
                memory_manager.print_summary()
    
    # Final save
    final_checkpoint = f'./checkpoints/cifar10_{model_name}_final.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes
    }, final_checkpoint)
    print(f"\n✓ Final checkpoint saved: {final_checkpoint}")
    
    if memory_manager is not None:
        final_memory = memory_manager.save_memory(num_epochs, f"cifar10_{model_name}_final")
        print(f"✓ Final memory bank saved: {final_memory}")

    # Plot curves
    try:
        plot_loss_curve(train_losses, val_losses)
        plot_accuracy_curve(train_accuracies, val_accuracies)
    except:
        print("Could not generate plots")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet32', 'resnet50'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--use_memory_bank', action='store_true')
    parser.add_argument('--memory_capacity', type=int, default=256)
    parser.add_argument('--memory_alpha', type=float, default=0.1)
    parser.add_argument('--memory_tail_threshold', type=float, default=20.0)
    
    args = parser.parse_args()
    
    main(args.model_name, args.batch_size, args.num_epochs, args.learning_rate,
         args.use_memory_bank, args.memory_capacity, args.memory_alpha, args.memory_tail_threshold)