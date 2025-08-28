#!/usr/bin/env python3
"""
First real test with your actual trained model and memory bank.
Run this after training your model with memory bank enabled.
"""

import torch
from utils.memory_manager import MemoryManager
from utils.feature_to_text import FeatureToTextMapper
from models import ResNet50  # Import your model

def main():
    # Replace these with your actual values
    dataset_name = "imagenet"  # or "inaturalist"
    model_name = "resnet50"
    num_classes = 1000  # Your actual number of classes
    
    # Example class names dictionary (create this for your dataset)
    class_names = {
        0: "ruby_throated_hummingbird",
        1: "yellow_warbler", 
        2: "house_finch",
        # ... add all your class names
        999: "some_tail_class"
    }
    
    print("=== Loading Trained Model and Memory Bank ===")
    
    # Initialize your trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(num_classes=num_classes).to(device)
    
    # Load model weights if you have them
    # model.load_state_dict(torch.load("path/to/your/model.pth"))
    # model.eval()
    
    # Initialize memory manager (this will try to load latest checkpoint)
    memory_manager = MemoryManager(
        model=model,
        num_classes=num_classes,
        capacity_per_class=256,
        alpha_base=0.1,
        tail_threshold_percentile=20.0,
        device=device,
        save_dir=f'./memory_checkpoints/{dataset_name}_{model_name}'
    )
    
    # Try to load latest memory checkpoint
    loaded_path = memory_manager.load_latest_memory(f"{dataset_name}_{model_name}")
    if loaded_path:
        print(f"✅ Loaded memory bank from: {loaded_path}")
    else:
        print("⚠️  No memory bank checkpoint found. Train with --use_memory_bank first.")
        print("Example: python main.py --dataset_name imagenet --model_name resnet50 --use_memory_bank")
        return
    
    # Print memory bank summary
    memory_manager.print_summary()
    
    print("\n=== Generating Semantic Prompts ===")
    
    # Initialize feature-to-text mapper
    mapper = FeatureToTextMapper(
        feature_dim=model.get_feature_dim(),
        device=device
    )
    
    # Generate prompts for tail classes
    tail_prompts = mapper.generate_prompts_for_tail_classes(
        memory_manager=memory_manager,
        class_names=class_names,
        num_prompts_per_class=3
    )
    
    print(f"Generated prompts for {len(tail_prompts)} tail classes:")
    for class_id, prompts in list(tail_prompts.items())[:5]:  # Show first 5
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"\nClass {class_id} ({class_name}):")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. {prompt}")
    
    # Save prompts for later use
    prompts_file = f"./tail_class_prompts_{dataset_name}_{model_name}.json"
    mapper.save_prompts_to_file(tail_prompts, class_names, prompts_file)
    
    print(f"\n✅ Success! Next steps:")
    print(f"1. Check the generated prompts in: {prompts_file}")
    print(f"2. Run image generation: python generate_images.py")
    print(f"3. Integrate synthetic images into training")

if __name__ == "__main__":
    main()