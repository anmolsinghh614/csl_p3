#!/usr/bin/env python3
"""
Option 1 Implementation: Direct Class Label Prompts
Simple baseline approach using just class names for prompt generation.
"""

import torch
import json
from typing import Dict, List, Optional
from utils.memory_manager import MemoryManager
from models import ResNet32


class SimpleClassLabelMapper:
    """
    Simple prompt generator using only class names (Option 1).
    This serves as a baseline comparison to your feature-driven approach.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        print("Using simple class label prompts (Option 1 baseline)")
    
    def generate_text_prompt(self, feature: torch.Tensor, class_name: str = "") -> str:
        """
        Generate simple prompt using only class name.
        Note: feature parameter is ignored in this baseline approach.
        """
        if not class_name or class_name.startswith('class_'):
            return "a photo of an object"
        
        # Simple templates for variety
        templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a photograph of a {}",
            "a clear photo of a {}"
        ]
        
        # Rotate through templates for minimal variety
        template_idx = hash(class_name) % len(templates)
        return templates[template_idx].format(class_name)
    
    def generate_prompts_for_tail_classes(self, 
                                        memory_manager,
                                        class_names: Dict[int, str],
                                        num_prompts_per_class: int = 3) -> Dict[int, List[str]]:
        """
        Generate simple prompts for tail classes using only class names.
        """
        tail_classes = memory_manager.get_tail_classes()
        tail_prompts = {}
        
        print(f"Generating simple prompts for {len(tail_classes)} tail classes...")
        
        for class_id in tail_classes:
            class_name = class_names.get(class_id, f"class_{class_id}")
            prompts = []
            
            # Generate multiple variants using different templates
            base_templates = [
                "a photo of a {}",
                "a high-quality photo of a {}",
                "a detailed photo of a {}",
                "a clear image of a {}",
                "a picture of a {}"
            ]
            
            # Use as many templates as requested, cycling if necessary
            for i in range(num_prompts_per_class):
                template = base_templates[i % len(base_templates)]
                prompt = template.format(class_name)
                prompts.append(prompt)
            
            tail_prompts[class_id] = prompts
            print(f"  Class {class_id} ({class_name}): {len(prompts)} prompts")
        
        return tail_prompts
    
    def compute_prompt_quality(self, prompt: str, reference_features: torch.Tensor) -> float:
        """
        Simple quality score based on prompt length.
        Note: reference_features ignored since this approach doesn't use features.
        """
        # Simple heuristic: non-generic prompts get higher scores
        if "object" in prompt:
            return 0.3  # Generic fallback
        elif len(prompt.split()) >= 5:
            return 0.8  # Detailed prompt
        else:
            return 0.6  # Basic prompt
    
    def save_prompts_to_file(self, prompts: Dict[int, List[str]], 
                           class_names: Dict[int, str], 
                           filepath: str):
        """Save prompts to JSON file."""
        prompt_data = {
            'metadata': {
                'method': 'Option 1: Direct Class Labels',
                'num_classes': len(prompts),
                'total_prompts': sum(len(p) for p in prompts.values()),
                'model_info': 'SimpleClassLabelMapper'
            },
            'prompts': {}
        }
        
        for class_id, class_prompts in prompts.items():
            prompt_data['prompts'][str(class_id)] = {
                'class_name': class_names.get(class_id, f"class_{class_id}"),
                'prompts': class_prompts,
                'count': len(class_prompts)
            }
        
        with open(filepath, 'w') as f:
            json.dump(prompt_data, f, indent=2)
        
        print(f"Simple prompts saved to: {filepath}")


def test_option1_prompts():
    """
    Test Option 1 (simple class labels) with the same memory bank.
    Compare directly against your feature-driven approach.
    """
    
    print("Testing Option 1: Simple Class Label Prompts")
    print("=" * 50)
    
    # CIFAR-10 class names (same as your existing tests)
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
    
    print("CIFAR-10 Classes:")
    for i, name in cifar10_class_names.items():
        class_type = "HEAD" if i < 3 else "MEDIUM" if i < 7 else "TAIL" 
        print(f"  {i}: {name:10s} ({class_type})")
    
    # Configuration
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nUsing device: {device}")
    
    # Initialize model and memory manager (same as your existing setup)
    model = ResNet32(num_classes=num_classes).to(device)
    
    memory_manager = MemoryManager(
        model=model,
        num_classes=num_classes,
        capacity_per_class=64,
        alpha_base=0.1,
        tail_threshold_percentile=20.0,
        device=device,
        save_dir='./memory_checkpoints/cifar10_resnet32'
    )
    
    # Load existing memory bank
    loaded_path = memory_manager.load_latest_memory("cifar10_resnet32")
    if not loaded_path:
        print("Error: No memory bank checkpoint found!")
        print("Run 'python main_cifar10.py' first to train the memory bank")
        return
    
    print(f"Loaded memory bank from: {loaded_path}")
    
    # Print memory bank summary
    print("\nMemory Bank Summary:")
    memory_manager.print_summary()
    
    # Get class distribution (same tail classes as your feature approach)
    tail_classes = memory_manager.get_tail_classes()
    print(f"\nTail classes identified: {tail_classes}")
    
    # Initialize simple prompt generator
    print("\nInitializing Option 1 prompt generator...")
    simple_mapper = SimpleClassLabelMapper(device=device)
    
    # Generate simple prompts
    print("\nGenerating simple prompts...")
    simple_prompts = simple_mapper.generate_prompts_for_tail_classes(
        memory_manager=memory_manager,
        class_names=cifar10_class_names,
        num_prompts_per_class=3
    )
    
    print(f"\nGenerated prompts for {len(simple_prompts)} tail classes")
    
    # Display prompts
    print("\n" + "=" * 60)
    print("OPTION 1 GENERATED PROMPTS (Simple Class Labels)")
    print("=" * 60)
    
    for class_id in tail_classes:
        class_name = cifar10_class_names[class_id]
        print(f"\nTAIL Class {class_id}: {class_name}")
        
        if class_id in simple_prompts:
            prompts = simple_prompts[class_id]
            for i, prompt in enumerate(prompts, 1):
                print(f"  {i}. {prompt}")
    
    # Save prompts
    prompts_file = "./cifar10_option1_prompts.json"
    simple_mapper.save_prompts_to_file(simple_prompts, cifar10_class_names, prompts_file)
    
    # Quality analysis
    print(f"\nPrompt Analysis:")
    total_prompts = sum(len(prompts) for prompts in simple_prompts.values())
    print(f"  Total prompts: {total_prompts}")
    print(f"  Classes with prompts: {len(simple_prompts)}")
    print(f"  Average prompts per class: {total_prompts/len(simple_prompts):.1f}")
    
    # Compare with your feature-driven approach
    print(f"\n" + "=" * 60)
    print("COMPARISON: Option 1 vs Your Feature-Driven Approach")
    print("=" * 60)
    
    print("Option 1 (Simple) prompts:")
    for class_id in list(simple_prompts.keys())[:2]:
        class_name = cifar10_class_names[class_id]
        print(f"  {class_name}: {simple_prompts[class_id][0]}")
    
    # Try to load your feature-driven prompts for comparison
    try:
        with open("./cifar10_tail_class_prompts.json", 'r') as f:
            feature_prompts = json.load(f)
        
        print("\nYour Feature-Driven prompts:")
        for class_id_str in list(feature_prompts['prompts'].keys())[:2]:
            class_info = feature_prompts['prompts'][class_id_str]
            print(f"  {class_info['class_name']}: {class_info['prompts'][0]}")
            
    except FileNotFoundError:
        print("\nFeature-driven prompts not found. Run test_prompt_generation_cifar10.py first.")
    
    print(f"\nOption 1 Implementation Complete!")
    print(f"\nNext steps:")
    print(f"1. Use {prompts_file} to generate images with Option 1")
    print(f"2. Compare image quality: Option 1 vs Feature-driven")
    print(f"3. Measure classification performance with both approaches")
    
    return simple_prompts


def generate_images_option1():
    """
    Generate images using Option 1 prompts (for comparison with your feature approach).
    """
    print("Generating images using Option 1 prompts...")
    
    # Check if Option 1 prompts exist
    prompts_file = "./cifar10_option1_prompts.json"
    if not os.path.exists(prompts_file):
        print(f"Error: {prompts_file} not found!")
        print("Run test_option1_prompts() first")
        return
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        prompt_data = json.load(f)
    
    print(f"Loaded Option 1 prompts for {len(prompt_data['prompts'])} classes")
    
    # Use the same image generation pipeline as your feature approach
    # (You can adapt your generate_cifar10_images.py to use these prompts)
    
    print("To generate images:")
    print("1. Modify generate_cifar10_images.py to load from cifar10_option1_prompts.json")
    print("2. Run generation with Option 1 prompts")
    print("3. Compare results with feature-driven approach")


if __name__ == "__main__":
    import os
    
    # Test Option 1 prompt generation
    test_option1_prompts()