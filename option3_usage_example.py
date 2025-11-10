"""
Example usage of Option 3: Visual Exemplar + BLIP Caption generation
Integrates with existing CSL memory bank system.
"""

import torch
from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator
from utils.memory_manager import MemoryManager # Your existing implementation
# Import your existing model and dataloaders

def main():
    # Initialize your existing model and memory manager
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load your trained model (with CSL memory bank)
    model = YourModel()  # Replace with your model
    model.load_state_dict(torch.load('your_trained_model.pth'))
    
    # Initialize memory manager with your trained memory bank
    memory_manager = MemoryManager(
        model=model,
        num_classes=100,  # Your dataset's number of classes
        device=device
    )
    
    # Load the trained memory bank
    memory_manager.load_latest_memory()  # or load_memory('specific_checkpoint.json')
    
    # Create class name mapping (replace with your dataset's class names)
    class_names = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        # ... add all your class names
    }
    
    # Initialize Option 3 generator
    option3_generator = VisualExemplarPromptGenerator(
        memory_manager=memory_manager,
        clip_model_name="ViT-B/32",  # or "ViT-L/14" for better quality
        device=device,
        cache_dir="./clip_cache"
    )
    
    # Create your training dataloader
    # Make sure it returns (images, labels, image_paths) or (images, labels)
    train_dataloader = create_your_dataloader(
        dataset='cifar100',  # or your dataset
        batch_size=32,
        shuffle=False,  # Don't shuffle for consistent caching
        include_paths=True  # If possible, include image paths
    )
    
    print("Starting Option 3 pipeline...")
    
    # Run the complete pipeline
    semantic_prompts = option3_generator.run_full_pipeline(
        dataloader=train_dataloader,
        class_names=class_names,
        k_exemplars=5,  # Number of nearest exemplars per tail class
        max_images_per_class=1000,  # Max images to process per class
        output_dir="./option3_results"
    )
    
    # Print results summary
    print(f"\nGenerated prompts for {len(semantic_prompts)} tail classes:")
    for class_id, prompts in semantic_prompts.items():
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"\nClass {class_id} ({class_name}):")
        for i, prompt in enumerate(prompts[:3]):  # Show first 3 prompts
            print(f"  {i+1}: {prompt}")
        if len(prompts) > 3:
            print(f"  ... and {len(prompts)-3} more prompts")


def create_your_dataloader(dataset, batch_size, shuffle=False, include_paths=False):
    """
    Create your dataloader. Modify this based on your existing dataloader setup.
    The dataloader should return:
    - (images, labels, image_paths) if include_paths=True
    - (images, labels) if include_paths=False
    """
    if dataset == 'cifar100':
        # Example for CIFAR-100
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        if include_paths:
            # Custom dataset that includes paths
            dataset = CustomCIFAR100WithPaths(
                root='./data', 
                train=True, 
                transform=transform, 
                download=True
            )
        else:
            dataset = datasets.CIFAR100(
                root='./data', 
                train=True, 
                transform=transform, 
                download=True
            )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    else:
        # Replace with your dataset loading logic
        raise NotImplementedError(f"Dataset {dataset} not implemented")


class CustomCIFAR100WithPaths(datasets.CIFAR100):
    """Custom CIFAR-100 dataset that also returns image paths/indices."""
    
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # Use index as path since CIFAR-100 doesn't have file paths
        path = f"cifar100_train_{index}"
        return image, label, path


def run_step_by_step_example():
    """Example showing step-by-step usage for more control."""
    
    # Setup (same as above)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    memory_manager = setup_memory_manager()  # Your setup function
    
    option3_generator = VisualExemplarPromptGenerator(
        memory_manager=memory_manager,
        device=device
    )
    
    # Step 1: Build image database
    train_dataloader = create_your_dataloader('cifar100', batch_size=32)
    
    print("Building CLIP image database...")
    option3_generator.build_image_database(
        dataloader=train_dataloader,
        max_images_per_class=500
    )
    
    # Step 2: Find nearest exemplars for tail classes
    print("Finding nearest exemplars...")
    nearest_exemplars = option3_generator.find_nearest_exemplars_for_tail_classes(
        k_exemplars=3,
        use_memory_prototypes=True  # Use your CSL memory bank prototypes
    )
    
    # Step 3: Generate captions
    print("Generating BLIP captions...")
    class_captions = option3_generator.generate_captions_for_exemplars(
        nearest_exemplars, 
        train_dataloader
    )
    
    # Step 4: Create semantic prompts
    class_names = get_your_class_names()  # Your function
    semantic_prompts = option3_generator.create_semantic_prompts(
        class_captions, 
        class_names
    )
    
    # Step 5: Save results
    option3_generator.save_prompts_json(
        semantic_prompts,
        class_names,
        "option3_prompts.json"
    )
    
    return semantic_prompts


def analyze_memory_bank_first():
    """Analyze your memory bank before running Option 3."""
    
    memory_manager = setup_memory_manager()
    memory_manager.load_latest_memory()
    
    # Print memory bank summary
    memory_manager.print_summary()
    
    # Get tail class information
    tail_classes = memory_manager.get_tail_classes()
    print(f"\nTail classes: {tail_classes}")
    
    # Export detailed tail class analysis
    memory_manager.export_tail_class_analysis("tail_class_analysis.json")
    
    # Visualize memory bank
    memory_manager.visualize_memory("memory_bank_visualization.png")


if __name__ == "__main__":
    # First analyze your memory bank
    print("Analyzing existing memory bank...")
    analyze_memory_bank_first()
    
    # Then run Option 3
    print("\nRunning Option 3 pipeline...")
    main()
    
    # Or run step by step for more control
    # run_step_by_step_example()