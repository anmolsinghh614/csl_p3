"""
Run the FULL Option 3 pipeline with actual BLIP captions.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator
from utils.memory_manager import MemoryManager


class CIFAR10WithPaths(datasets.CIFAR10):
    """CIFAR-10 that returns (image, label, path) tuples."""
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = f"cifar10_train_{index}"
        return image, label, path


def create_memory_manager_with_real_data():
    """Create memory manager and populate with some real data."""
    
    # Create a simple model for testing
    class SimpleModel:
        def get_feature_dim(self):
            return 2048
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel()
    
    memory_manager = MemoryManager(
        model=model,
        num_classes=10,
        device=device,
        save_dir='./memory_checkpoints'
    )
    
    # Try to load existing memory bank
    latest_path = memory_manager.load_latest_memory()
    if latest_path:
        print(f"Loaded existing memory bank: {latest_path}")
        return memory_manager
    
    # If no existing memory bank, create one with realistic data
    print("Creating memory bank with realistic class imbalance...")
    
    # Simulate class imbalance: some classes have fewer samples
    class_sample_counts = {
        0: 100, 1: 100, 2: 100, 3: 100, 4: 100,  # Head classes
        5: 100, 6: 100,                           # Medium classes  
        7: 30, 8: 25, 9: 20                       # Tail classes (horse, ship, truck)
    }
    
    for class_id, count in class_sample_counts.items():
        for _ in range(count):
            # Generate features that vary slightly per class
            base_feature = torch.randn(2048, device=device)
            # Add class-specific bias to make classes distinguishable
            class_bias = torch.randn(2048, device=device) * 0.1
            feature = base_feature + class_bias * class_id
            feature = feature / torch.norm(feature)  # Normalize
            
            memory_manager.memory_bank.update(class_id, feature)
    
    # Save this memory bank
    memory_manager.save_memory(step=0, prefix="realistic_test")
    return memory_manager


def run_full_option3_pipeline():
    """Run the complete Option 3 pipeline with BLIP captions."""
    
    print("="*60)
    print("RUNNING FULL OPTION 3 PIPELINE")
    print("="*60)
    
    # Step 1: Setup memory manager
    print("Step 1: Setting up memory manager...")
    memory_manager = create_memory_manager_with_real_data()
    memory_manager.print_summary()
    
    # Verify we have tail classes
    tail_classes = memory_manager.get_tail_classes()
    if not tail_classes:
        print("No tail classes found! Adjusting threshold...")
        # Manually set some classes as tail for testing
        memory_manager.memory_bank.tail_classes = {7, 8, 9}
        tail_classes = [7, 8, 9]
    
    print(f"Tail classes: {tail_classes}")
    
    # Step 2: Create dataloader
    print("\nStep 2: Creating dataloader...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    dataset = CIFAR10WithPaths(
        root='./data',
        train=True, 
        transform=transform,
        download=True
    )
    
    # Use smaller batch size and subset for testing
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Step 3: Initialize Option 3 generator
    print("\nStep 3: Initializing Option 3 generator...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    option3_generator = VisualExemplarPromptGenerator(
        memory_manager=memory_manager,
        clip_model_name="ViT-B/32",
        blip_model_name="Salesforce/blip-image-captioning-base",
        device=device,
        cache_dir="./clip_cache"
    )
    
    # Step 4: Run the FULL pipeline
    print("\nStep 4: Running FULL Option 3 pipeline...")
    
    class_names = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
    }
    
    # This will do the REAL Option 3 process:
    # 1. Build CLIP database of images
    # 2. Find nearest exemplars to prototypes  
    # 3. Use BLIP to caption the exemplars
    # 4. Create semantic prompts from captions
    semantic_prompts = option3_generator.run_full_pipeline(
        dataloader=dataloader,
        class_names=class_names,
        k_exemplars=3,                # Number of exemplars per tail class
        max_images_per_class=200,     # Process subset for testing
        output_dir="./full_option3_results"
    )
    
    # Step 5: Show results
    print("\n" + "="*60)
    print("REAL OPTION 3 RESULTS")
    print("="*60)
    
    for class_id, prompts in semantic_prompts.items():
        class_name = class_names[class_id]
        print(f"\nTail Class {class_id} ({class_name}):")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}: {prompt}")
    
    print(f"\nFull results saved to: ./full_option3_results/option3_semantic_prompts.json")
    print(f"Detailed analysis in: ./full_option3_results/option3_exemplar_analysis.json")
    
    return semantic_prompts


def quick_test_single_class():
    """Quick test focusing on one tail class to see the difference."""
    
    print("QUICK TEST: Single Class Option 3")
    print("="*40)
    
    # Setup (same as above but minimal)
    memory_manager = create_memory_manager_with_real_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    option3_generator = VisualExemplarPromptGenerator(
        memory_manager=memory_manager,
        device=device
    )
    
    # Create minimal dataloader - just first few batches
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    dataset = CIFAR10WithPaths(root='./data', train=True, transform=transform, download=True)
    
    # Process only first 5 batches (80 images)
    limited_data = []
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        limited_data.append(batch)
        if i >= 4:  # 5 batches total
            break
    
    print("Building CLIP database (limited)...")
    # Manually build database with limited data
    for batch in limited_data:
        images, labels, paths = batch
        for image, label, path in zip(images, labels, paths):
            class_id = label.item()
            
            if class_id not in option3_generator.image_database:
                option3_generator.image_database[class_id] = []
            
            # Compute CLIP embedding
            clip_embedding = option3_generator._compute_clip_embedding(image, path)
            if clip_embedding is not None:
                from utils.visual_exemplar_prompt_generator import ImageExemplar
                exemplar = ImageExemplar(
                    image_path=path,
                    clip_embedding=clip_embedding,
                    class_id=class_id,
                    distance_to_prototype=0.0
                )
                option3_generator.image_database[class_id].append(exemplar)
    
    print("Finding nearest exemplars...")
    nearest_exemplars = option3_generator.find_nearest_exemplars_for_tail_classes(k_exemplars=2)
    
    print("Generating BLIP captions...")
    class_captions = option3_generator.generate_captions_for_exemplars(nearest_exemplars, limited_data)
    
    print("Creating semantic prompts...")
    class_names = {8: "ship", 9: "truck", 7: "horse"}
    semantic_prompts = option3_generator.create_semantic_prompts(class_captions, class_names)
    
    print("\nRESULTS:")
    for class_id, prompts in semantic_prompts.items():
        print(f"\nClass {class_id} ({class_names[class_id]}):")
        for prompt in prompts:
            print(f"  - {prompt}")
    
    return semantic_prompts


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test with limited data
        quick_test_single_class()
    else:
        # Full pipeline
        run_full_option3_pipeline()