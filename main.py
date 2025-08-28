import torch
import torch.optim as optim
from models import ResNet32, ResNet50, ResNeXt50, ResNeXt101
from dataloaders import get_imagenet_lt_loaders, get_inaturalist_loaders
from utils import CSLLossFunc, plot_loss_curve, plot_accuracy_curve ,plot_validation_accuracy
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
        
        # Print memory bank status every 100 batches
        if memory_manager is not None and batch_idx % 100 == 0:
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

def main(dataset_name, model_name, batch_size, num_epochs, learning_rate, data_path='./datasets', 
         use_memory_bank=True, memory_capacity=256, memory_alpha=0.1, memory_tail_threshold=20.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    if dataset_name == 'imagenet':
        train_loader, val_loader, test_loader = get_imagenet_lt_loaders(
            train_txt='dataloaders/ImageNet_LT/ImageNet_LT_train.txt',
            val_txt='dataloaders/ImageNet_LT/ImageNet_LT_val.txt',
            test_txt='dataloaders/ImageNet_LT/ImageNet_LT_test.txt',
            train_dir=f'{data_path}/ImageNet_LT',
            val_dir=f'{data_path}/ImageNet_LT',
            test_dir=f'{data_path}/ImageNet_LT'
        )
        # Retrieve num_classes from validation set
        num_classes = len(val_loader.dataset.get_unique_labels())
        target_class_index = list(range(0, num_classes))

    elif dataset_name == 'inaturalist':
        train_loader, val_loader, test_loader = get_inaturalist_loaders(
            train_txt='dataloaders/Inaturalist18/iNaturalist18_train.txt',
            val_txt='dataloaders/Inaturalist18/iNaturalist18_val.txt',
            test_txt='dataloaders/Inaturalist18/iNaturalist18_test.txt',
            train_dir=f'{data_path}/INaturalist/',
            val_dir=f'{data_path}/INaturalist/',
            test_dir=f'{data_path}/INaturalist/',
            batch_size=batch_size
        )
        # Retrieve num_classes from validation set
        num_classes = len(val_loader.dataset.get_unique_labels())
        target_class_index = list(range(num_classes)) 
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    print(f"Number of classes in dataset: {num_classes}")

    # Initialize model
    if model_name == 'resnet32':
      model = ResNet32(num_classes=num_classes).to(device)
    elif model_name == 'resnet50':
      model = ResNet50(num_classes=num_classes).to(device)
    elif model_name == 'resnext50':
      model = ResNeXt50(num_classes=num_classes).to(device)  
    elif model_name == 'resnext101':
      model = ResNeXt101(num_classes=num_classes).to(device) 
    else:
      raise ValueError(f"Model {model_name} not supported")
    print("Model initialized.")

    # Initialize memory manager if requested
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
            save_dir=f'./memory_checkpoints/{dataset_name}_{model_name}'
        )
        print("Memory bank initialized.")
        print(f"Feature dimension: {model.get_feature_dim()}")
        print(f"Memory capacity per class: {memory_capacity}")
        print(f"EMA alpha base: {memory_alpha}")
        print(f"Tail threshold percentile: {memory_tail_threshold}%")

    criterion = CSLLossFunc(target_class_index=target_class_index, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print("Loss function, optimizer, and scheduler initialized.")

    # Training and evaluation
    start_epoch = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy, val_loss, val_accuracy = train(
            model, criterion, optimizer, scheduler, train_loader, val_loader, device, epoch, memory_manager
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save memory bank every 10 epochs
        if memory_manager is not None and (epoch + 1) % 10 == 0:
            save_path = memory_manager.save_memory(epoch + 1, f"{dataset_name}_{model_name}")
            print(f"Memory bank saved to: {save_path}")
            
            # Print memory summary
            memory_manager.print_summary()
            
            # Show tail class information
            tail_classes = memory_manager.get_tail_classes()
            if tail_classes:
                print(f"\nTail classes identified: {len(tail_classes)}")
                print(f"First 10 tail classes: {tail_classes[:10]}")
                
                # Export tail class analysis for semantic prompt generation
                export_path = f"./tail_class_analysis_{dataset_name}_{model_name}_epoch_{epoch+1}.json"
                memory_manager.export_tail_class_analysis(export_path)

    # Final memory bank save and visualization
    if memory_manager is not None:
        final_save_path = memory_manager.save_memory(num_epochs, f"{dataset_name}_{model_name}_final")
        print(f"Final memory bank saved to: {final_save_path}")
        
        # Final summary and visualization
        memory_manager.print_summary()
        memory_manager.visualize_memory(f"./memory_visualization_{dataset_name}_{model_name}_final.png")
        
        # Export final tail class analysis
        final_export_path = f"./tail_class_analysis_{dataset_name}_{model_name}_final.json"
        memory_manager.export_tail_class_analysis(final_export_path)

    # Plot curves
    plot_loss_curve(train_losses, val_losses)
    plot_accuracy_curve(train_accuracies, val_accuracies)
    plot_validation_accuracy(val_accuracies, num_epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='./datasets', help='Path to the datasets')
    
    # Memory bank arguments
    parser.add_argument('--use_memory_bank', action='store_true', help='Enable memory bank for feature storage')
    parser.add_argument('--memory_capacity', type=int, default=256, help='Features per class in memory bank')
    parser.add_argument('--memory_alpha', type=float, default=0.1, help='EMA learning rate for memory bank')
    parser.add_argument('--memory_tail_threshold', type=float, default=20.0, help='Percentile threshold for tail classes')

    args = parser.parse_args()
    main(args.dataset_name, args.model_name, args.batch_size, args.num_epochs, args.learning_rate, args.data_path,
         args.use_memory_bank, args.memory_capacity, args.memory_alpha, args.memory_tail_threshold)