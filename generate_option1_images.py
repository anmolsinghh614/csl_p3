#!/usr/bin/env python3
"""
Generate images using Option 1 (simple class label) prompts.
This allows direct comparison with your feature-driven approach.
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import json
import os
from pathlib import Path
import time

def generate_images_option1():
    """Generate images using Option 1 simple prompts for comparison."""
    
    print("Generating Images Using Option 1 (Simple Class Labels)")
    print("=" * 55)
    
    # Check if Option 1 prompts exist
    prompts_file = "./cifar10_option1_prompts.json"
    if not Path(prompts_file).exists():
        print(f"Error: {prompts_file} not found!")
        print("Run 'python option1_simple_prompts.py' first")
        return
    
    # Load Option 1 prompts
    with open(prompts_file, 'r') as f:
        prompt_data = json.load(f)
    
    print(f"Loaded Option 1 prompts for {len(prompt_data['prompts'])} classes")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Stable Diffusion (same as your feature approach)
    print("Loading Stable Diffusion pipeline...")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("Stable Diffusion loaded successfully")
        
    except Exception as e:
        print(f"Error loading Stable Diffusion: {e}")
        return
    
    # Generation parameters (same as feature approach for fair comparison)
    num_images_per_prompt = 4
    num_inference_steps = 25
    guidance_scale = 7.5
    
    # Create output directory
    output_dir = Path("./option1_cifar10_images")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Generate images for each class
    total_generated = 0
    generation_stats = {}
    
    for class_id_str, class_info in prompt_data['prompts'].items():
        class_id = int(class_id_str)
        class_name = class_info['class_name']
        prompts = class_info['prompts']
        
        print(f"\nGenerating Option 1 images for Class {class_id}: {class_name}")
        print(f"Using {len(prompts)} simple prompts")
        
        # Create class directory
        class_dir = output_dir / f"class_{class_id}_{class_name}"
        class_dir.mkdir(exist_ok=True)
        
        class_images = []
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"  Prompt {prompt_idx+1}: '{prompt}'")
            
            start_time = time.time()
            
            try:
                # Generate images
                images = pipe(
                    prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=512,
                    width=512,
                    generator=torch.Generator(device=device).manual_seed(42 + prompt_idx)
                ).images
                
                # Save images
                for img_idx, image in enumerate(images):
                    image_filename = f"option1_{class_id}_{prompt_idx}_{img_idx}.jpg"
                    image_path = class_dir / image_filename
                    image.save(image_path, quality=95)
                    class_images.append(str(image_path))
                
                generation_time = time.time() - start_time
                print(f"    Generated {len(images)} images in {generation_time:.1f}s")
                
            except Exception as e:
                print(f"    Error generating images: {e}")
                continue
        
        total_generated += len(class_images)
        generation_stats[class_id] = {
            'class_name': class_name,
            'images_generated': len(class_images),
            'image_paths': class_images
        }
        
        print(f"  Total Option 1 images for {class_name}: {len(class_images)}")
    
    # Save metadata
    metadata = {
        'method': 'Option 1: Simple Class Labels',
        'generation_settings': {
            'model': 'runwayml/stable-diffusion-v1-5',
            'num_images_per_prompt': num_images_per_prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'device': device
        },
        'classes': generation_stats,
        'total_images': total_generated
    }
    
    metadata_file = output_dir / "option1_generation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create training dataset file
    dataset_file = output_dir / "option1_training_dataset.txt"
    with open(dataset_file, 'w') as f:
        for class_id, stats in generation_stats.items():
            for image_path in stats['image_paths']:
                f.write(f"{image_path} {class_id}\n")
    
    print(f"\nOption 1 Generation Complete!")
    print(f"=" * 40)
    print(f"Total images generated: {total_generated}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset file: {dataset_file}")
    
    # Compare with feature approach
    print(f"\n📊 COMPARISON READY")
    print(f"Option 1 images: {output_dir}")
    print(f"Feature images: ./synthetic_cifar10_images")
    print(f"\nNext steps:")
    print(f"1. Visually compare image quality between approaches")
    print(f"2. Train models with both datasets")
    print(f"3. Measure tail class performance improvements")
    
    return str(output_dir), str(dataset_file)

if __name__ == "__main__":
    generate_images_option1()