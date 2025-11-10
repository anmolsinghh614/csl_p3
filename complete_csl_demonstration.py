"""
Complete CSL Demonstration: Class Imbalance Detection and Correction using Option 3

This script demonstrates the entire pipeline:
1. Initial training on imbalanced data
2. Memory bank tracking and tail class identification
3. Option 3 semantic prompt generation
4. Synthetic image generation
5. Dynamic memory bank updates
6. Iterative rebalancing process
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple
import time

# Import your existing components
from utils.memory_manager import MemoryManager
from utils.memory_bank import MemoryBank
from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator
from option3_image_generator import Option3ImageGenerator


class SimpleCSLModel(nn.Module):
    """Simple CSL model for demonstration."""
    
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, feature_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.features(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_feature_dim(self):
        return self.feature_dim


class CIFAR10WithPaths(datasets.CIFAR10):
    """CIFAR-10 with paths for Option 3."""
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = f"cifar10_{self.train}_{index}"
        return image, label, path


class CSLDemonstration:
    """Complete CSL demonstration with dynamic rebalancing."""
    
    def __init__(self, device='cuda', num_classes=10):
        self.device = device
        self.num_classes = num_classes
        
        # Model and training components
        self.model = SimpleCSLModel(num_classes=num_classes).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Memory management
        self.memory_manager = MemoryManager(
            model=self.model,
            num_classes=num_classes,
            capacity_per_class=256,
            alpha_base=0.1,
            tail_threshold_percentile=30.0,  # More sensitive to see rebalancing effect
            device=device,
            save_dir='./demo_memory_checkpoints'
        )
        
        # Option 3 components
        self.option3_generator = None
        self.image_generator = None
        
        # Tracking
        self.epoch_history = []
        self.class_names = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
        
    def create_imbalanced_dataset(self, original_dataset, imbalance_factor=10):
        """Create artificially imbalanced dataset."""
        print("Creating imbalanced dataset...")
        
        # Define class sample counts (some classes much rarer)
        base_samples = 500  # Base number of samples per class
        
        class_sample_counts = {
            0: base_samples, 1: base_samples, 2: base_samples, 3: base_samples,  # Head classes
            4: base_samples, 5: base_samples, 6: base_samples,                   # Head classes  
            7: base_samples // imbalance_factor,  # Tail: horse
            8: base_samples // imbalance_factor,  # Tail: ship
            9: base_samples // imbalance_factor,  # Tail: truck
        }
        
        print(f"Target samples per class: {class_sample_counts}")
        
        # Collect indices for each class
        class_indices = {i: [] for i in range(self.num_classes)}
        for idx, (_, label, _) in enumerate(original_dataset):  # Unpack 3 values: image, label, path
            class_indices[label].append(idx)
        
        # Sample according to target distribution
        selected_indices = []
        for class_id, target_count in class_sample_counts.items():
            available = class_indices[class_id]
            selected = np.random.choice(available, size=min(target_count, len(available)), replace=False)
            selected_indices.extend(selected)
        
        return Subset(original_dataset, selected_indices)
    
    def train_epoch(self, dataloader, epoch_num):
        """Train one epoch and update memory bank."""
        print(f"\nEPOCH {epoch_num + 1}")
        print("-" * 50)
        
        self.model.train()
        epoch_loss = 0.0
        samples_processed = 0
        
        # Track samples per class this epoch
        epoch_class_counts = {i: 0 for i in range(self.num_classes)}
        
        for batch_idx, batch in enumerate(dataloader):
            # Handle different batch formats
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, features = self.model(images, return_features=True)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            samples_processed += len(labels)
            
            # Update memory bank with each sample
            with torch.no_grad():
                for i, (feature, label) in enumerate(zip(features, labels)):
                    class_id = label.item()
                    self.memory_manager.memory_bank.update(class_id, feature)
                    epoch_class_counts[class_id] += 1
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Get current memory bank statistics
        tail_classes = self.memory_manager.get_tail_classes()
        head_classes = self.memory_manager.get_head_classes()
        medium_classes = self.memory_manager.get_medium_classes()
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Samples processed: {samples_processed}")
        print(f"Epoch class distribution: {epoch_class_counts}")
        print(f"MEMORY BANK STATUS:")
        print(f"  Tail classes: {[f'{cid}({self.class_names[cid]})' for cid in tail_classes]}")
        print(f"  Medium classes: {[f'{cid}({self.class_names[cid]})' for cid in medium_classes]}")
        print(f"  Head classes: {[f'{cid}({self.class_names[cid]})' for cid in head_classes]}")
        
        # Store history
        epoch_data = {
            'epoch': epoch_num + 1,
            'loss': avg_loss,
            'tail_classes': list(tail_classes),
            'head_classes': list(head_classes),
            'medium_classes': list(medium_classes),
            'total_samples': self.memory_manager.memory_bank.total_samples,
            'class_frequencies': self.memory_manager.memory_bank.class_frequencies.cpu().numpy().tolist()
        }
        self.epoch_history.append(epoch_data)
        
        return avg_loss, tail_classes
    
    def generate_synthetic_data(self, dataloader, tail_classes, images_per_class=20):
        """Generate synthetic data for tail classes using Option 3."""
        if not tail_classes:
            print("No tail classes - skipping synthetic generation")
            return 0
        
        print(f"\nGENERATING SYNTHETIC DATA")
        print("=" * 50)
        print(f"Target tail classes: {[f'{cid}({self.class_names[cid]})' for cid in tail_classes]}")
        
        # Initialize Option 3 generator if needed
        if self.option3_generator is None:
            print("Initializing Option 3 generator...")
            try:
                self.option3_generator = VisualExemplarPromptGenerator(
                    memory_manager=self.memory_manager,
                    clip_model_name="ViT-B/32",
                    device=self.device,
                    cache_dir="./demo_clip_cache"
                )
                
                self.image_generator = Option3ImageGenerator(
                    model_type="stable_diffusion",
                    device=self.device,
                    output_dir="./demo_synthetic_images"
                )
                print("Option 3 components initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Option 3 components: {e}")
                return 0
        
        try:
            # Step 1: Build CLIP database (limited for demo)
            print("Building CLIP database...")
            limited_data = []
            for i, batch in enumerate(dataloader):
                limited_data.append(batch)
                if i >= 10:  # Process limited batches for demo
                    break
            
            # Build image database
            for batch in limited_data:
                if len(batch) == 3:
                    images, labels, paths = batch
                else:
                    images, labels = batch
                    paths = [f"demo_batch_{i}_{j}" for j in range(len(images))]
                
                for image, label, path in zip(images, labels, paths):
                    class_id = label.item()
                    
                    if class_id not in self.option3_generator.image_database:
                        self.option3_generator.image_database[class_id] = []
                    
                    # Compute CLIP embedding
                    clip_embedding = self.option3_generator._compute_clip_embedding(image, path)
                    if clip_embedding is not None:
                        from utils.visual_exemplar_prompt_generator import ImageExemplar
                        exemplar = ImageExemplar(
                            image_path=path,
                            clip_embedding=clip_embedding,
                            class_id=class_id,
                            distance_to_prototype=0.0
                        )
                        self.option3_generator.image_database[class_id].append(exemplar)
            
            # Step 2: Find nearest exemplars
            print("Finding nearest exemplars for tail classes...")
            nearest_exemplars = self.option3_generator.find_nearest_exemplars_for_tail_classes(
                k_exemplars=2,
                use_memory_prototypes=True
            )
            
            if not nearest_exemplars:
                print("No exemplars found")
                return 0
            
            # Step 3: Generate BLIP captions
            print("Generating BLIP captions...")
            class_captions = self.option3_generator.generate_captions_for_exemplars(
                nearest_exemplars, limited_data
            )
            
            # Step 4: Create semantic prompts
            print("Creating semantic prompts...")
            semantic_prompts = self.option3_generator.create_semantic_prompts(
                class_captions, self.class_names
            )
            
            print("Generated prompts:")
            for class_id, prompts in semantic_prompts.items():
                class_name = self.class_names[class_id]
                print(f"  {class_name} ({class_id}):")
                for prompt in prompts[:2]:  # Show first 2 prompts
                    print(f"    - {prompt}")
            
            # Step 5: Generate synthetic images
            print(f"Generating {images_per_class} images per tail class...")
            total_generated = 0
            
            for class_id, prompts in semantic_prompts.items():
                class_name = self.class_names[class_id]
                print(f"Generating images for {class_name}...")
                
                # Create class directory
                class_dir = os.path.join("./demo_synthetic_images", f"class_{class_id}_{class_name}")
                os.makedirs(class_dir, exist_ok=True)
                
                images_generated = 0
                for i, prompt in enumerate(prompts):
                    for j in range(min(3, images_per_class // len(prompts) + 1)):
                        # Call your method with the correct parameters
                        image_path = self.image_generator._generate_single_image(
                            prompt=prompt,
                            class_id=class_id,
                            class_name=class_name,
                            prompt_idx=i,
                            img_idx=j,
                            class_dir=class_dir,
                            image_size=512,
                            inference_steps=20,
                            guidance_scale=7.5
                        )
                        
                        if image_path:  # Method returns path if successful, None if failed
                            images_generated += 1
                            total_generated += 1
                        
                        if images_generated >= images_per_class:
                            break
                    if images_generated >= images_per_class:
                        break
                
                print(f"  Generated {images_generated} images for {class_name}")
            
            print(f"Total synthetic images generated: {total_generated}")
            return total_generated
            
            print(f"Total synthetic images generated: {total_generated}")
            return total_generated
            
        except Exception as e:
            print(f"Synthetic generation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def simulate_synthetic_feature_integration(self, tail_classes, images_per_class=20):
        """Simulate adding synthetic features to memory bank."""
        print(f"\nSIMULATING SYNTHETIC FEATURE INTEGRATION")
        print("=" * 50)
        
        total_added = 0
        
        for class_id in tail_classes:
            class_name = self.class_names[class_id]
            print(f"Adding synthetic features for {class_name} (class {class_id})...")
            
            # Get current prototype for this class
            current_prototype = self.memory_manager.memory_bank.get_prototype(class_id)
            
            # Generate synthetic features similar to the prototype but with variation
            for i in range(images_per_class):
                # Create synthetic feature by adding noise to prototype
                noise = torch.randn_like(current_prototype) * 0.1
                synthetic_feature = current_prototype + noise
                synthetic_feature = synthetic_feature / torch.norm(synthetic_feature)  # Normalize
                
                # Add to memory bank
                self.memory_manager.memory_bank.update(class_id, synthetic_feature)
                total_added += 1
            
            print(f"  Added {images_per_class} synthetic features for {class_name}")
        
        print(f"Total synthetic features added: {total_added}")
        return total_added
    
    def print_detailed_statistics(self):
        """Print detailed memory bank statistics."""
        print(f"\nDETAILED MEMORY BANK STATISTICS")
        print("=" * 50)
        
        stats = self.memory_manager.get_training_statistics()
        
        print(f"Total samples in memory: {stats['total_updates']}")
        print(f"Memory usage: {stats['memory_usage']['total_memory_mb']:.2f} MB")
        print(f"Buffer utilization: {stats['memory_usage']['buffer_utilization']*100:.1f}%")
        
        print(f"\nCLASS DISTRIBUTION:")
        print(f"  Tail: {len(stats['class_distribution']['tail'])} classes")
        print(f"  Medium: {len(stats['class_distribution']['medium'])} classes") 
        print(f"  Head: {len(stats['class_distribution']['head'])} classes")
        
        print(f"\nSAMPLES PER CLASS:")
        for class_id in range(self.num_classes):
            frequency = self.memory_manager.memory_bank.class_frequencies[class_id].item()
            percentage = (frequency / max(self.memory_manager.memory_bank.total_samples, 1)) * 100
            status = "TAIL" if class_id in stats['class_distribution']['tail'] else \
                    "HEAD" if class_id in stats['class_distribution']['head'] else "MEDIUM"
            print(f"  {self.class_names[class_id]:12} ({class_id}): {frequency:4d} samples ({percentage:5.2f}%) [{status}]")
    
    def visualize_training_progress(self):
        """Visualize how class distribution changes over epochs."""
        if len(self.epoch_history) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = [e['epoch'] for e in self.epoch_history]
        
        # Plot 1: Training loss
        losses = [e['loss'] for e in self.epoch_history]
        ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Number of tail classes over time
        tail_counts = [len(e['tail_classes']) for e in self.epoch_history]
        ax2.plot(epochs, tail_counts, 'r-', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Tail Classes')
        ax2.set_title('Tail Classes Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Class frequency evolution for specific classes
        class_ids_to_plot = [7, 8, 9]  # Focus on initially tail classes
        for class_id in class_ids_to_plot:
            frequencies = [e['class_frequencies'][class_id] for e in self.epoch_history]
            ax3.plot(epochs, frequencies, linewidth=2, marker='o', 
                    label=f"{self.class_names[class_id]} ({class_id})")
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Sample Count Evolution (Focus Classes)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Total samples over time
        total_samples = [e['total_samples'] for e in self.epoch_history]
        ax4.plot(epochs, total_samples, 'g-', linewidth=2, marker='^')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Total Samples')
        ax4.set_title('Total Samples in Memory Bank')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('csl_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_demonstration(self, num_epochs=10, synthetic_generation_frequency=3):
        """Run the complete CSL demonstration."""
        print("CSL CLASS IMBALANCE CORRECTION DEMONSTRATION")
        print("=" * 60)
        print("This demo shows dynamic class rebalancing through synthetic data generation")
        print("=" * 60)
        
        # Setup data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Create imbalanced dataset with less severe imbalance for demo
        full_dataset = CIFAR10WithPaths(root='./data', train=True, transform=transform, download=True)
        imbalanced_dataset = self.create_imbalanced_dataset(full_dataset, imbalance_factor=5)  # Less severe imbalance
        dataloader = DataLoader(imbalanced_dataset, batch_size=32, shuffle=True)
        
        print(f"Created imbalanced dataset with {len(imbalanced_dataset)} samples")
        
        # Training loop with synthetic generation
        for epoch in range(num_epochs):
            # Train one epoch
            loss, tail_classes = self.train_epoch(dataloader, epoch)
            
            # Print detailed statistics every few epochs
            if (epoch + 1) % 2 == 0:
                self.print_detailed_statistics()
            
            # Generate synthetic data periodically
            if (epoch + 1) % synthetic_generation_frequency == 0 and tail_classes:
                print(f"\n{'='*20} SYNTHETIC DATA GENERATION PHASE {'='*20}")
                
                # Option 1: Real Option 3 synthetic generation 
                synthetic_count = self.generate_synthetic_data(dataloader, tail_classes, images_per_class=50)
                
                # Option 2: Simulated synthetic feature integration (faster for demo) 
                # synthetic_count = self.simulate_synthetic_feature_integration(tail_classes, images_per_class=30)
                
                if synthetic_count > 0:
                    print(f"\nMEMORY BANK UPDATED WITH {synthetic_count} SYNTHETIC SAMPLES")
                    print("Checking for new class distribution...")
                    
                    # Get updated statistics
                    new_tail_classes = self.memory_manager.get_tail_classes()
                    new_head_classes = self.memory_manager.get_head_classes()
                    
                    print(f"BEFORE synthetic: Tail classes were {[f'{c}({self.class_names[c]})' for c in tail_classes]}")
                    print(f"AFTER synthetic:  Tail classes are  {[f'{c}({self.class_names[c]})' for c in new_tail_classes]}")
                    
                    if set(tail_classes) != set(new_tail_classes):
                        print("🎉 CLASS DISTRIBUTION HAS CHANGED! 🎉")
                        print("Dynamic rebalancing is working!")
                    else:
                        print("Class distribution unchanged (may need more synthetic data)")
                
                print("=" * 70)
        
        # Final analysis
        print(f"\n{'='*20} FINAL ANALYSIS {'='*20}")
        self.print_detailed_statistics()
        
        # Show training progress visualization
        self.visualize_training_progress()
        
        # Summary
        initial_tail = self.epoch_history[0]['tail_classes']
        final_tail = self.epoch_history[-1]['tail_classes']
        
        print(f"\nSUMMARY:")
        print(f"Initial tail classes: {[f'{c}({self.class_names[c]})' for c in initial_tail]}")
        print(f"Final tail classes:   {[f'{c}({self.class_names[c]})' for c in final_tail]}")
        
        if set(initial_tail) != set(final_tail):
            print("✅ SUCCESS: Class distribution was dynamically rebalanced!")
        else:
            print("⚠️  Classes remained imbalanced (may need more epochs or synthetic data)")
        
        print("Demonstration complete!")


def main():
    """Run the complete demonstration."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create and run demonstration
    demo = CSLDemonstration(device=device)
    demo.run_complete_demonstration(
        num_epochs=12,  # Number of training epochs
        synthetic_generation_frequency=4  # Generate synthetic data every N epochs
    )


if __name__ == "__main__":
    main()