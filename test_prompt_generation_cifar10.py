#!/usr/bin/env python3
"""
Test prompt generation with CIFAR-10-LT trained memory bank.
Run this after training with main_cifar10.py
"""

import torch
from utils.memory_manager import MemoryManager
from utils.feature_to_text import FeatureToTextMapper
from models import ResNet32

def test_cifar10_prompt_generation():
    """Test prompt generation with CIFAR-10-LT memory bank."""
    
    print("🧪 Testing Prompt Generation with CIFAR-10-LT")
    print("=" * 50)
    
    # CIFAR-10 class names
    cifar10_class_names = {
        0: "airplane",
        1: "automobile", 
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship", 
        9: "truck"
    }
    
    print("📋 CIFAR-10 Classes:")
    for i, name in cifar10_class_names.items():
        class_type = "HEAD" if i < 3 else "MEDIUM" if i < 7 else "TAIL" 
        print(f"  {i}: {name:10s} ({class_type})")
    
    # Configuration
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n🖥️  Using device: {device}")
    
    # Initialize model
    print("🤖 Loading model...")
    model = ResNet32(num_classes=num_classes).to(device)
    
    # Initialize memory manager
    print("🧠 Loading memory bank...")
    memory_manager = MemoryManager(
        model=model,
        num_classes=num_classes,
        capacity_per_class=64,
        alpha_base=0.1,
        tail_threshold_percentile=20.0,
        device=device,
        save_dir='./memory_checkpoints/cifar10_resnet32'
    )
    
    # Try to load latest memory checkpoint
    loaded_path = memory_manager.load_latest_memory("cifar10_resnet32")
    if not loaded_path:
        print("❌ No memory bank checkpoint found!")
        print("💡 First run: python main_cifar10.py")
        return
    
    print(f"✅ Loaded memory bank from: {loaded_path}")
    
    # Print memory bank summary
    print("\n📊 Memory Bank Summary:")
    memory_manager.print_summary()
    
    # Get class distribution
    tail_classes = memory_manager.get_tail_classes()
    head_classes = memory_manager.get_head_classes()
    medium_classes = memory_manager.get_medium_classes()
    
    print(f"\n🎯 Class Distribution:")
    print(f"  Head classes: {head_classes}")
    print(f"  Medium classes: {medium_classes}")
    print(f"  Tail classes: {tail_classes}")
    
    # Initialize feature-to-text mapper
    print("\n🔤 Initializing prompt generator...")
    mapper = FeatureToTextMapper(
        feature_dim=model.get_feature_dim(),
        device=device
    )
    print("✅ Prompt generator ready")
    
    # Generate prompts for all classes (focusing on tail)
    print("\n📝 Generating prompts...")
    all_prompts = mapper.generate_prompts_for_tail_classes(
        memory_manager=memory_manager,
        class_names=cifar10_class_names,
        num_prompts_per_class=3
    )
    
    print(f"✅ Generated prompts for {len(all_prompts)} classes")
    
    # Display prompts by category
    print("\n" + "="*60)
    print("🎨 GENERATED PROMPTS BY CLASS TYPE")
    print("="*60)
    
    # Show all classes, highlighting tail classes
    for class_id in range(num_classes):
        class_name = cifar10_class_names[class_id]
        
        if class_id in head_classes:
            class_type = "📈 HEAD"
        elif class_id in medium_classes:
            class_type = "📊 MEDIUM"
        else:
            class_type = "🎯 TAIL"
        
        print(f"\n{class_type} Class {class_id}: {class_name}")
        
        if class_id in all_prompts:
            prompts = all_prompts[class_id]
            for i, prompt in enumerate(prompts, 1):
                print(f"  {i}. {prompt}")
        else:
            # Generate one prompt even for non-tail classes to show difference
            if class_id in [class_id for class_id in range(num_classes)]:  # For all classes
                try:
                    prototype = memory_manager.get_semantic_prompt_data(class_id, k_features=1)
                    if prototype and prototype.get('prototype') is not None:
                        single_prompt = mapper.generate_text_prompt(
                            prototype['prototype'], class_name
                        )
                        print(f"  1. {single_prompt}")
                    else:
                        print(f"  No features stored for this class yet")
                except:
                    print(f"  Unable to generate prompt (insufficient data)")
    
    # Save prompts
    prompts_file = "./cifar10_tail_class_prompts.json"
    mapper.save_prompts_to_file(all_prompts, cifar10_class_names, prompts_file)
    print(f"\n💾 Prompts saved to: {prompts_file}")
    
    # Quality assessment
    print(f"\n📊 Prompt Quality Analysis:")
    total_prompts = sum(len(prompts) for prompts in all_prompts.values())
    print(f"  Total prompts generated: {total_prompts}")
    print(f"  Classes with prompts: {len(all_prompts)}")
    print(f"  Average prompts per class: {total_prompts/len(all_prompts):.1f}")
    
    # Test quality scoring for a few prompts
    print(f"\n🧪 Testing prompt quality scoring...")
    for class_id, prompts in list(all_prompts.items())[:3]:  # First 3 classes
        class_name = cifar10_class_names[class_id]
        prompt = prompts[0] if prompts else "default prompt"
        
        # Get prototype for quality assessment
        try:
            prompt_data = memory_manager.get_semantic_prompt_data(class_id, k_features=1)
            if prompt_data and prompt_data.get('prototype') is not None:
                prototype = prompt_data['prototype']
                quality_score = mapper.compute_prompt_quality(prompt, prototype)
                print(f"  Class {class_id} ({class_name}): Quality = {quality_score:.3f}")
            else:
                print(f"  Class {class_id} ({class_name}): No prototype available")
        except Exception as e:
            print(f"  Class {class_id} ({class_name}): Error computing quality")
    
    print(f"\n🎉 Prompt generation test completed!")
    print(f"\n🚀 Next steps:")
    print(f"1. Review prompts in {prompts_file}")
    print(f"2. Test image generation with these prompts")
    print(f"3. Scale up to larger dataset when ready")
    
    # Show what typical tail class prompts look like
    if tail_classes:
        print(f"\n💡 Example Tail Class Prompts:")
        for class_id in tail_classes[:2]:  # Show first 2 tail classes
            class_name = cifar10_class_names[class_id]
            if class_id in all_prompts:
                print(f"\n🎯 {class_name} (Class {class_id}) prompts:")
                for prompt in all_prompts[class_id]:
                    print(f"   '{prompt}'")


if __name__ == "__main__":
    test_cifar10_prompt_generation()