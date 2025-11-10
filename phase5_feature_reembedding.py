"""
Phase 5: Feature Re-embedding for Synthetic Images
Integrated with existing CSL Extended Memory Bank framework
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict

# Import from your existing codebase
from utils.memory_bank import MemoryBank


class SyntheticFeatureExtractor:
    """
    Extract features from synthetic images using the same model architecture
    as your main training pipeline. Computes confidence scores relative to
    memory bank prototypes.
    """
    
    def __init__(self,
                 model: nn.Module,
                 memory_bank: MemoryBank,
                 device: str = 'cuda',
                 batch_size: int = 32):
        """
        Initialize feature extractor.
        
        Args:
            model: Your trained model (ResNet, ResNeXt, etc.)
            memory_bank: Your existing memory bank with class prototypes
            device: Device for inference
            batch_size: Batch size for processing
        """
        self.model = model
        self.model.eval()
        self.model = self.model.to(device)
        
        self.memory_bank = memory_bank
        self.device = device
        self.batch_size = batch_size
        
        # Get feature dimension from model
        self.feature_dim = model.get_feature_dim()
        
        # Image preprocessing (same as your training pipeline)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"SyntheticFeatureExtractor initialized")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
    
    def extract_features_from_directory(self,
                                       generation_dir: str = "./option3_generated_images",
                                       output_path: str = "./synthetic_features_phase5.pth") -> Dict:
        """
        Extract features from all synthetic images in generation directory.
        
        Args:
            generation_dir: Directory with generated images
            output_path: Where to save extracted features
            
        Returns:
            Dictionary with extracted features and confidence scores
        """
        print("\n" + "="*70)
        print("PHASE 5: EXTRACTING FEATURES FROM SYNTHETIC IMAGES")
        print("="*70)
        
        # Load generation summary
        summary_path = os.path.join(generation_dir, 'generation_summary.json')
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Generation summary not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            generation_summary = json.load(f)
        
        # Process each tail class
        all_features = {}
        
        for class_id_str, class_data in generation_summary['classes'].items():
            class_id = int(class_id_str)
            image_paths = class_data['image_paths']
            
            if not image_paths:
                print(f"\nClass {class_id}: No images found, skipping...")
                continue
            
            print(f"\nProcessing Class {class_id}: {len(image_paths)} images")
            
            # Extract features for this class
            features, confidences = self._extract_class_features(
                image_paths, class_id
            )
            
            all_features[class_id] = {
                'features': features,
                'image_paths': image_paths,
                'confidences': confidences,
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'num_images': len(image_paths)
            }
            
            print(f"  ✓ Features: {features.shape}")
            print(f"  ✓ Mean confidence: {np.mean(confidences):.4f} ± {np.std(confidences):.4f}")
            print(f"  ✓ High quality (>0.7): {np.sum(confidences > 0.7)}")
        
        # Save features
        self._save_features(all_features, output_path)
        
        # Generate report
        self._generate_report(all_features, generation_dir)
        
        return all_features
    
    def _extract_class_features(self,
                                image_paths: List[str],
                                class_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for a specific class.
        
        Args:
            image_paths: List of paths to synthetic images
            class_id: Class ID
            
        Returns:
            (features, confidences) as numpy arrays
        """
        # Create dataset and loader
        dataset = SimpleSyntheticDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        features_list = []
        
        with torch.no_grad():
            for images in tqdm(loader, desc=f"Class {class_id}", leave=False):
                images = images.to(self.device)
                
                # Extract features using your model's feature extraction
                _, features = self.model(images, return_features=True)
                features_list.append(features.cpu().numpy())
        
        # Concatenate all features
        features = np.concatenate(features_list, axis=0)
        
        # Compute confidence scores
        confidences = self._compute_confidence_scores(features, class_id)
        
        return features, confidences
    
    def _compute_confidence_scores(self,
                                   features: np.ndarray,
                                   class_id: int) -> np.ndarray:
        """
        Compute confidence scores using memory bank prototypes.
        
        Method:
        1. Cosine similarity to real class prototype from memory bank
        2. Intra-class consistency among synthetic features
        3. Weighted combination
        
        Args:
            features: Synthetic features [N, feature_dim]
            class_id: Target class ID
            
        Returns:
            Confidence scores [N] in range [0, 1]
        """
        N = features.shape[0]
        
        # Get real prototype from memory bank
        prototype = self.memory_bank.get_prototype(class_id).cpu().numpy()
        
        # Method 1: Cosine similarity to real prototype
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        prototype_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
        
        cosine_sim = np.dot(features_norm, prototype_norm)
        confidence_prototype = (cosine_sim + 1) / 2  # Map [-1, 1] to [0, 1]
        
        # Method 2: Intra-class consistency
        synthetic_centroid = features.mean(axis=0)
        centroid_norm = synthetic_centroid / (np.linalg.norm(synthetic_centroid) + 1e-8)
        
        intra_sim = np.dot(features_norm, centroid_norm)
        confidence_intra = (intra_sim + 1) / 2
        
        # Weighted combination (favor prototype similarity)
        confidence = 0.8 * confidence_prototype + 0.2 * confidence_intra
        
        return confidence
    
    def _save_features(self, all_features: Dict, output_path: str):
        """Save extracted features."""
        # Prepare data for saving
        save_data = {
            'feature_dim': self.feature_dim,
            'model_name': self.model.__class__.__name__,
            'classes': {}
        }
        
        for class_id, data in all_features.items():
            save_data['classes'][str(class_id)] = {
                'features': torch.from_numpy(data['features']),
                'image_paths': data['image_paths'],
                'confidences': torch.from_numpy(data['confidences']),
                'mean_confidence': data['mean_confidence'],
                'std_confidence': data['std_confidence'],
                'num_images': data['num_images']
            }
        
        torch.save(save_data, output_path)
        print(f"\n✓ Features saved to: {output_path}")
        
        # Save JSON summary
        summary_path = output_path.replace('.pth', '_summary.json')
        summary = {
            'feature_dim': self.feature_dim,
            'model_name': self.model.__class__.__name__,
            'total_classes': len(all_features),
            'total_images': sum(d['num_images'] for d in all_features.values()),
            'classes': {
                str(cid): {
                    'num_images': d['num_images'],
                    'mean_confidence': d['mean_confidence'],
                    'std_confidence': d['std_confidence']
                }
                for cid, d in all_features.items()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_path}")
    
    def _generate_report(self, all_features: Dict, generation_dir: str):
        """Generate detailed analysis report."""
        report = {
            'phase': 'Phase 5: Feature Re-embedding',
            'total_classes': len(all_features),
            'total_images': sum(d['num_images'] for d in all_features.values()),
            'classes': {}
        }
        
        for class_id, data in all_features.items():
            confidences = data['confidences']
            report['classes'][class_id] = {
                'num_images': data['num_images'],
                'confidence_stats': {
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences)),
                    'median': float(np.median(confidences))
                },
                'quality_distribution': {
                    'high_quality (>0.7)': int(np.sum(confidences > 0.7)),
                    'medium_quality (0.4-0.7)': int(np.sum((confidences >= 0.4) & (confidences <= 0.7))),
                    'low_quality (<0.4)': int(np.sum(confidences < 0.4))
                }
            }
        
        report_path = os.path.join(generation_dir, 'phase5_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to: {report_path}")


class SimpleSyntheticDataset(Dataset):
    """Simple dataset for loading synthetic images."""
    
    def __init__(self, image_paths: List[str], transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image
            return torch.zeros(3, 224, 224)


class SyntheticFeatureDataset(Dataset):
    """
    Dataset that loads synthetic features extracted in Phase 5.
    Compatible with your existing training pipeline.
    """
    
    def __init__(self,
                 features_path: str,
                 confidence_threshold: float = 0.3):
        """
        Args:
            features_path: Path to synthetic_features_phase5.pth
            confidence_threshold: Minimum confidence to include
        """
        # Load features
        data = torch.load(features_path)
        
        self.features = []
        self.labels = []
        self.confidences = []
        
        # Filter by confidence threshold
        for class_id_str, class_data in data['classes'].items():
            class_id = int(class_id_str)
            features = class_data['features']
            confidences = class_data['confidences']
            
            # Filter by confidence
            high_conf_mask = confidences >= confidence_threshold
            filtered_features = features[high_conf_mask]
            filtered_confidences = confidences[high_conf_mask]
            
            if len(filtered_features) > 0:
                self.features.append(filtered_features)
                self.labels.extend([class_id] * len(filtered_features))
                self.confidences.extend(filtered_confidences.tolist())
        
        # Concatenate
        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.confidences = torch.tensor(self.confidences, dtype=torch.float32)
        
        print(f"SyntheticFeatureDataset loaded:")
        print(f"  Total features: {len(self.features)}")
        print(f"  Mean confidence: {self.confidences.mean():.4f}")
        print(f"  Confidence threshold: {confidence_threshold}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.confidences[idx]


def run_phase5_extraction(
    model_checkpoint_path: str,
    memory_bank_path: str,
    generation_dir: str = "./option3_generated_images",
    output_path: str = "./synthetic_features_phase5.pth",
    device: str = 'cuda'
) -> Dict:
    """
    Complete Phase 5 workflow for your project.
    
    Args:
        model_checkpoint_path: Path to your trained model checkpoint
        memory_bank_path: Path to your memory bank checkpoint
        generation_dir: Directory with generated images
        output_path: Where to save extracted features
        device: Device to use
        
    Returns:
        Dictionary with extracted features
    """
    print("\n" + "="*70)
    print("PHASE 5: FEATURE RE-EMBEDDING WORKFLOW")
    print("="*70)
    
    # Load model
    print("\n[1/4] Loading trained model...")
    from models import ResNet50  # Adjust based on your model
    
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    num_classes = checkpoint['num_classes'] if 'num_classes' in checkpoint else 1000
    
    model = ResNet50(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"  ✓ Model loaded: {model.__class__.__name__}")
    
    # Load memory bank
    print("\n[2/4] Loading memory bank...")
    from utils.memory_manager import MemoryManager
    
    memory_manager = MemoryManager(
        model=model,
        num_classes=num_classes,
        device=device
    )
    memory_manager.load_memory(memory_bank_path)
    print(f"  ✓ Memory bank loaded")
    print(f"  ✓ Tail classes: {len(memory_manager.get_tail_classes())}")
    
    # Extract features
    print("\n[3/4] Extracting features from synthetic images...")
    extractor = SyntheticFeatureExtractor(
        model=model,
        memory_bank=memory_manager.memory_bank,
        device=device,
        batch_size=32
    )
    
    features = extractor.extract_features_from_directory(
        generation_dir=generation_dir,
        output_path=output_path
    )
    
    # Summary
    print("\n[4/4] Phase 5 Complete!")
    print("="*70)
    print(f"✓ Extracted features for {len(features)} tail classes")
    print(f"✓ Total synthetic images processed: {sum(d['num_images'] for d in features.values())}")
    print(f"✓ Output: {output_path}")
    print(f"✓ Ready for Phase 6: CSL Training Integration")
    
    return features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 5: Feature Re-embedding')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--memory_bank', type=str, required=True,
                       help='Path to memory bank checkpoint')
    parser.add_argument('--generation_dir', type=str, 
                       default='./option3_generated_images',
                       help='Directory with generated images')
    parser.add_argument('--output', type=str,
                       default='./synthetic_features_phase5.pth',
                       help='Output path for extracted features')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Run Phase 5
    features = run_phase5_extraction(
        model_checkpoint_path=args.model_checkpoint,
        memory_bank_path=args.memory_bank,
        generation_dir=args.generation_dir,
        output_path=args.output,
        device=args.device
    )