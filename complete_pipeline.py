"""
Complete Workflow: End-to-End Pipeline
Option 3 Prompt Generation → Image Synthesis → Phase 5 Re-embedding → Phase 6 Training
"""

import os
import sys
import argparse
from pathlib import Path
import json


def check_prerequisites():
    """Check if all required components are available."""
    print("\n" + "="*70)
    print("CHECKING PREREQUISITES")
    print("="*70)
    
    required_files = {
        'Memory Bank': ['./memory_checkpoints/*/memory_bank_*.json'],
        'Option3 Generator': ['./utils/visual_exemplar_prompt_generator.py'],
        'Image Generator': ['./option3_image_generator.py'],
        'Phase 5 Module': ['./phase5_feature_reembedding.py']
    }
    
    checks = []
    for name, paths in required_files.items():
        exists = any(Path('.').glob(p) for p in paths)
        status = "✓" if exists else "✗"
        print(f"{status} {name}")
        checks.append(exists)
    
    if not all(checks):
        print("\n⚠ Some prerequisites are missing.")
        print("Make sure you have:")
        print("  1. Trained model with memory bank")
        print("  2. Option 3 prompt generation code")
        print("  3. Image generation pipeline")
    
    return all(checks)


def step1_generate_prompts(memory_bank_path: str, output_dir: str = "./option3_prompts"):
    """Step 1: Generate semantic prompts using Option 3."""
    print("\n" + "="*70)
    print("STEP 1: GENERATING SEMANTIC PROMPTS (OPTION 3)")
    print("="*70)
    
    # Import your Option 3 generator
    from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator
    from utils.memory_manager import MemoryManager
    from models import ResNet50  # Adjust to your model
    
    # Load memory manager
    print("Loading memory bank...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy model for memory manager
    class DummyModel:
        def get_feature_dim(self):
            return 2048
    
    memory_manager = MemoryManager(
        model=DummyModel(),
        num_classes=1000,  # Adjust to your dataset
        device=device
    )
    memory_manager.load_memory(memory_bank_path)
    
    # Initialize Option 3 generator
    print("Initializing Option 3 generator...")
    generator = VisualExemplarPromptGenerator(
        memory_manager=memory_manager,
        device=device
    )
    
    # Generate prompts (you need to provide your dataloader here)
    print("Generating prompts... (Note: Provide your dataloader)")
    print("This step requires running full_run_option3.py with your data")
    print(f"Output will be saved to: {output_dir}")
    
    # For now, return placeholder
    return f"{output_dir}/option3_semantic_prompts.json"


def step2_generate_images(prompts_file: str, output_dir: str = "./option3_generated_images"):
    """Step 2: Generate synthetic images using Stable Diffusion."""
    print("\n" + "="*70)
    print("STEP 2: GENERATING SYNTHETIC IMAGES")
    print("="*70)
    
    if not os.path.exists(prompts_file):
        print(f"✗ Prompts file not found: {prompts_file}")
        print("Run Step 1 first or provide prompts file")
        return None
    
    print(f"Loading prompts from: {prompts_file}")
    print(f"Output directory: {output_dir}")
    
    # Run image generation
    cmd = f"python option3_image_generator.py test"
    print(f"\nRunning: {cmd}")
    os.system(cmd)
    
    # Check if generation was successful
    summary_path = os.path.join(output_dir, 'generation_summary.json')
    if os.path.exists(summary_path):
        print(f"✓ Image generation complete")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print(f"  Total images: {summary['total_images']}")
        print(f"  Total classes: {summary['total_classes']}")
        return summary_path
    else:
        print(f"✗ Generation failed - summary not found")
        return None


def step3_extract_features(model_checkpoint: str,
                          memory_bank_path: str,
                          generation_dir: str = "./option3_generated_images",
                          output_path: str = "./synthetic_features_phase5.pth"):
    """Step 3: Extract features from synthetic images (Phase 5)."""
    print("\n" + "="*70)
    print("STEP 3: EXTRACTING FEATURES (PHASE 5)")
    print("="*70)
    
    # Check inputs
    if not os.path.exists(model_checkpoint):
        print(f"✗ Model checkpoint not found: {model_checkpoint}")
        return None
    
    if not os.path.exists(memory_bank_path):
        print(f"✗ Memory bank not found: {memory_bank_path}")
        return None
    
    if not os.path.exists(generation_dir):
        print(f"✗ Generated images not found: {generation_dir}")
        return None
    
    print(f"Model: {model_checkpoint}")
    print(f"Memory bank: {memory_bank_path}")
    print(f"Images: {generation_dir}")
    print(f"Output: {output_path}")
    
    # Run Phase 5 extraction
    from phase5_feature_reembedding import run_phase5_extraction
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    features = run_phase5_extraction(
        model_checkpoint_path=model_checkpoint,
        memory_bank_path=memory_bank_path,
        generation_dir=generation_dir,
        output_path=output_path,
        device=device
    )
    
    if os.path.exists(output_path):
        print(f"✓ Feature extraction complete: {output_path}")
        return output_path
    else:
        print(f"✗ Feature extraction failed")
        return None


def step4_train_with_synthetic(dataset_name: str,
                              model_name: str,
                              synthetic_features_path: str,
                              batch_size: int = 256,
                              num_epochs: int = 100,
                              learning_rate: float = 0.01,
                              synthetic_ratio: float = 0.2,
                              confidence_threshold: float = 0.5):
    """Step 4: Train with synthetic features (Phase 6)."""
    print("\n" + "="*70)
    print("STEP 4: TRAINING WITH SYNTHETIC FEATURES (PHASE 6)")
    print("="*70)
    
    if not os.path.exists(synthetic_features_path):
        print(f"✗ Synthetic features not found: {synthetic_features_path}")
        return None
    
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Synthetic features: {synthetic_features_path}")
    print(f"Synthetic ratio: {synthetic_ratio}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Build command
    cmd = f"""python main.py \
        --dataset_name {dataset_name} \
        --model_name {model_name} \
        --batch_size {batch_size} \
        --num_epochs {num_epochs} \
        --learning_rate {learning_rate} \
        --use_memory_bank \
        --use_synthetic_features \
        --synthetic_features_path {synthetic_features_path} \
        --synthetic_ratio {synthetic_ratio} \
        --confidence_threshold {confidence_threshold}
    """
    
    print(f"\nRunning: {cmd}")
    os.system(cmd)
    
    print("✓ Training complete")


def run_complete_pipeline(args):
    """Run the complete end-to-end pipeline."""
    print("\n" + "="*70)
    print("COMPLETE PIPELINE: OPTION 3 → PHASE 5 → PHASE 6")
    print("="*70)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠ Fix prerequisites before continuing")
        if not args.skip_checks:
            return
    
    # Step 1: Generate prompts (if needed)
    if args.prompts_file and os.path.exists(args.prompts_file):
        print(f"\n✓ Using existing prompts: {args.prompts_file}")
        prompts_file = args.prompts_file
    else:
        print("\n⚠ No prompts file provided")
        print("Run: python full_run_option3.py")
        print("Or provide --prompts_file argument")
        return
    
    # Step 2: Generate images
    if args.skip_generation and os.path.exists(args.generation_dir):
        print(f"\n✓ Using existing images: {args.generation_dir}")
        generation_summary = os.path.join(args.generation_dir, 'generation_summary.json')
    else:
        generation_summary = step2_generate_images(
            prompts_file,
            args.generation_dir
        )
        if generation_summary is None:
            print("\n✗ Image generation failed")
            return
    
    # Step 3: Extract features (Phase 5)
    if args.skip_extraction and os.path.exists(args.synthetic_features_path):
        print(f"\n✓ Using existing features: {args.synthetic_features_path}")
        features_path = args.synthetic_features_path
    else:
        features_path = step3_extract_features(
            model_checkpoint=args.model_checkpoint,
            memory_bank_path=args.memory_bank_path,
            generation_dir=args.generation_dir,
            output_path=args.synthetic_features_path
        )
        if features_path is None:
            print("\n✗ Feature extraction failed")
            return
    
    # Step 4: Train with synthetic features (Phase 6)
    if not args.skip_training:
        step4_train_with_synthetic(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            synthetic_features_path=features_path,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            synthetic_ratio=args.synthetic_ratio,
            confidence_threshold=args.confidence_threshold
        )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print(f"  • Prompts: {prompts_file}")
    print(f"  • Images: {args.generation_dir}")
    print(f"  • Features: {features_path}")
    print(f"  • Model checkpoints: ./checkpoints/")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Pipeline: Option 3 → Phase 5 → Phase 6',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python complete_pipeline.py \\
    --dataset_name imagenet \\
    --model_name resnet50 \\
    --model_checkpoint ./checkpoints/model_best.pth \\
    --memory_bank_path ./memory_checkpoints/latest.json \\
    --prompts_file ./option3_prompts/prompts.json

  # Skip steps if already done
  python complete_pipeline.py \\
    --skip_generation \\
    --skip_extraction \\
    --dataset_name imagenet \\
    --model_name resnet50
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                       choices=['imagenet', 'inaturalist', 'cifar10'],
                       help='Dataset name')
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['resnet32', 'resnet50', 'resnext50', 'resnext101'],
                       help='Model architecture')
    
    # Checkpoint paths
    parser.add_argument('--model_checkpoint', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--memory_bank_path', type=str,
                       help='Path to memory bank checkpoint')
    
    # Pipeline control
    parser.add_argument('--skip_checks', action='store_true',
                       help='Skip prerequisite checks')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip image generation (use existing)')
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip feature extraction (use existing)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training (only generate features)')
    
    # File paths
    parser.add_argument('--prompts_file', type=str,
                       default='./option3_test_results/test_prompts.json',
                       help='Path to Option 3 prompts file')
    parser.add_argument('--generation_dir', type=str,
                       default='./option3_generated_images',
                       help='Directory for generated images')
    parser.add_argument('--synthetic_features_path', type=str,
                       default='./synthetic_features_phase5.pth',
                       help='Path to save/load synthetic features')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    
    # Synthetic augmentation parameters
    parser.add_argument('--synthetic_ratio', type=float, default=0.2,
                       help='Ratio of synthetic to real samples')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence for synthetic features')
    
    args = parser.parse_args()
    
    # Validate required paths
    if not args.skip_extraction and not args.model_checkpoint:
        parser.error("--model_checkpoint required unless --skip_extraction")
    
    if not args.skip_extraction and not args.memory_bank_path:
        parser.error("--memory_bank_path required unless --skip_extraction")
    
    # Run pipeline
    run_complete_pipeline(args)


if __name__ == "__main__":
    import torch  # Import here to avoid issues if torch not available
    main()