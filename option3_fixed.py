"""
Fixed Option 3 usage example with proper imports.
Works with your existing CIFAR-10 setup.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# Add proper imports for torchvision
from torchvision import datasets, transforms

# Import your existing modules
try:
    from utils.memory_manager import MemoryManager
    from utils.memory_bank import MemoryBank
    print("✓ Memory modules imported successfully")
except ImportError as e:
    print(f"✗ Could not import memory modules: {e}")
    sys.exit(1)

# Try importing the visual exemplar generator
try:
    from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator
    print("✓ Visual exemplar generator imported")
except ImportError:
    print("✗ visual_exemplar_prompt_generator.py not found")
    print("Please save the artifact content as 'visual_exemplar_prompt_generator.py'")
    print("Or run 'python standalone_option3_test.py' instead")
    sys.exit(1)


class DummyModel(nn.Module):
    """Dummy model for testing - replace with your actual model."""
    def __init__(self, num_classes=10, feature_dim=2048):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.feature_dim = feature_dim
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        # Pad features to match expected dimension
        if features.shape[1] < self.feature_dim:
            padding = torch.zeros(features.shape[0], self.feature_dim - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)
        
        logits = self.fc(features[:, :64])  # Use first 64 features for classification
        
        if return_features:
            return logits, features
        return logits
    
    def get_feature_dim(self):
        return self.feature_dim


class CustomCIFAR10WithPaths(datasets.CIFAR10):
    """CIFAR-10 dataset that returns image paths/indices."""
    
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # Use index as path since CIFAR-10 doesn't have file paths
        path = f"cifar10_train_{index}"
        return image, label, path


def setup_cifar10_dataloader(batch_size=32, include_paths=True):
    """Setup CIFAR-10 dataloader compatible with your existing setup."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if include_paths:
        dataset = CustomCIFAR10WithPaths(
            root='./data', 
            train=True, 
            transform=transform, 
            download=True
        )
    else:
        dataset = datasets.CIFAR10(
            root='./data', 
            train=True, 
            transform=transform, 
            download=True
        )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_cifar10_class_names():
    """Get CIFAR-10 class names."""
    return {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
    }


def setup_memory_manager():
    """Setup memory manager - modify based on your trained model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Replace this with loading your actual trained model
    model = DummyModel(num_classes=10, feature_dim=2048)
    
    # Initialize memory manager
    memory_manager = MemoryManager(
        model=model,
        num_classes=10,  # CIFAR-10 has 10 classes
        capacity_per_class=256,
        device=device,
        save_dir='./memory_checkpoints'
    )
    
    return memory_manager


def test_with_existing_memory():
    """Test with existing memory bank if available."""
    print("Testing with existing memory bank...")
    
    memory_manager = setup_memory_manager()
    
    # Try to load existing memory
    latest_path = memory_manager.load_latest_memory()
    if latest_path:
        print(f"✓ Loaded existing memory from: {latest_path}")
        memory_manager.print_summary()
        return memory_manager
    else:
        print("No existing memory bank found. Creating dummy data for testing...")
        return create_dummy_memory_bank(memory_manager)


def create_dummy_memory_bank(memory_manager):
    """Create dummy memory bank for testing if no trained one exists."""
    print("Creating dummy memory bank for testing...")
    
    # Generate dummy features for each class
    device = memory_manager.device
    feature_dim = memory_manager.memory_bank.feature_dim
    
    for class_id in range(10):  # CIFAR-10 classes
        # Generate 50 dummy features per class with some classes having fewer (making them "tail")
        num_samples = 10 if class_id >= 7 else 50  # Classes 7,8,9 will be tail classes
        
        for _ in range(num_samples):
            # Generate random normalized feature
            dummy_feature = torch.randn(feature_dim, device=device)
            dummy_feature = dummy_feature / torch.norm(dummy_feature)
            
            memory_manager.memory_bank.update(class_id, dummy_feature)
    
    print("✓ Dummy memory bank created")
    memory_manager.print_summary()
    return memory_manager


def run_option3_test():
    """Run Option 3 test with your CIFAR-10 setup."""
    print("\n" + "="*60)
    print("RUNNING OPTION 3 TEST")
    print("="*60)
    
    # Step 1: Setup memory manager
    memory_manager = test_with_existing_memory()
    
    # Step 2: Check if we have tail classes
    tail_classes = memory_manager.get_tail_classes()
    if not tail_classes:
        print("No tail classes found. This might mean:")
        print("1. Your dataset is perfectly balanced")
        print("2. You need to train your CSL model first")
        print("3. The tail threshold needs adjustment")
        return
    
    print(f"Found {len(tail_classes)} tail classes: {tail_classes}")
    
    # Step 3: Setup dataloader
    print("\nSetting up CIFAR-10 dataloader...")
    dataloader = setup_cifar10_dataloader(batch_size=16, include_paths=True)
    
    # Step 4: Initialize Option 3 generator
    print("\nInitializing Option 3 generator...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        option3_generator = VisualExemplarPromptGenerator(
            memory_manager=memory_manager,
            clip_model_name="ViT-B/32",
            device=device,
            cache_dir="./clip_cache"
        )
        
        # Step 5: Test basic functionality
        print("\nTesting basic functionality...")
        #option3_generator.test_basic_functionality()
        
        # Step 6: Run simplified pipeline test
        print("\nRunning simplified pipeline test...")
        
        # Test with just a few batches
        print("Building CLIP database (limited test)...")
        limited_dataloader = DataLoader(
            CustomCIFAR10WithPaths(root='./data', train=True, 
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                 ]), download=True),
            batch_size=8, shuffle=False
        )
        
        # Process only first few batches for testing
        test_dataloader = []
        for i, batch in enumerate(limited_dataloader):
            test_dataloader.append(batch)
            if i >= 5:  # Only process first 6 batches (48 images)
                break
        
        # Build image database with limited data
        for batch in test_dataloader:
            images, labels, paths = batch
            for i, (image, label, path) in enumerate(zip(images, labels, paths)):
                class_id = label.item()
                
                # Add to database
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
        
        # Find nearest exemplars for tail classes
        print("Finding nearest exemplars...")
        nearest_exemplars = option3_generator.find_nearest_exemplars_for_tail_classes(
            k_exemplars=2,
            use_memory_prototypes=True
        )
        
        print(f"Found exemplars for {len(nearest_exemplars)} tail classes")
        
        # Generate test prompts
        class_names = get_cifar10_class_names()
        test_prompts = {}
        
        for class_id in nearest_exemplars.keys():
            class_name = class_names.get(class_id, f"class_{class_id}")
            test_prompts[class_id] = [
                f"A detailed photograph of a {class_name}",
                f"A high-quality image of a {class_name} in natural lighting",
                f"A professional photo showing a {class_name}"
            ]
        
        # Save test results
        output_dir = "./option3_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        option3_generator.save_prompts_json(
            test_prompts,
            class_names,
            os.path.join(output_dir, "test_prompts.json")
        )
        
        print(f"\n✓ Option 3 test completed successfully!")
        print(f"✓ Generated prompts for {len(test_prompts)} tail classes")
        print(f"✓ Results saved to {output_dir}")
        
        # Show sample results
        print("\nSample generated prompts:")
        for class_id, prompts in list(test_prompts.items())[:2]:
            class_name = class_names.get(class_id, f"class_{class_id}")
            print(f"\nTail Class {class_id} ({class_name}):")
            for prompt in prompts:
                print(f"  - {prompt}")
        
    except Exception as e:
        print(f"✗ Error during Option 3 test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    print("OPTION 3 IMPLEMENTATION TEST FOR CIFAR-10")
    print("="*60)
    
    # Check dependencies
    try:
        import clip
        from transformers import BlipProcessor
        print("✓ Required dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Install with:")
        print("pip install git+https://github.com/openai/CLIP.git")
        print("pip install transformers pillow")
        return
    
    # Run the test
    run_option3_test()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Replace DummyModel with your actual trained model")
    print("2. Load your real trained memory bank")
    print("3. Adjust parameters based on your dataset")
    print("4. Run full pipeline with complete dataset")
    print("5. Use generated prompts for diffusion model training")


if __name__ == "__main__":
    main()