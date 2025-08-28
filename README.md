# Memory Bank Implementation for Tail Class Feature Storage

This implementation provides a memory bank system that uses **EMA (Exponential Moving Average) + Reservoir sampling** to store features per class and automatically identify tail classes for later semantic prompt generation.

## 🏗️ Architecture Overview

The memory bank system consists of three main components:

1. **MemoryBank** (`utils/memory_bank.py`): Core implementation of EMA + Reservoir sampling
2. **MemoryManager** (`utils/memory_manager.py`): High-level interface for training integration
3. **Modified Models**: Updated ResNet/ResNeXt models that can extract features

## 🔧 Key Features

### EMA (Exponential Moving Average) Prototypes
- **Formula**: `M_c ← (1 - α) * M_c + α * feature`
- **Adaptive α**: Lower for head classes (stability), higher for tail classes (adaptation)
- **Bias correction**: Handles initialization bias automatically

### Reservoir Sampling
- **Bounded memory**: Fixed capacity per class (configurable)
- **Uniform sampling**: Each seen feature has equal probability of being retained
- **Formula**: Replace with probability `K/n` where K=capacity, n=total seen

### Tail Class Identification
- **Automatic detection**: Based on percentile thresholds (configurable)
- **Dynamic updates**: Reclassifies as training progresses
- **Three categories**: Tail, Medium, Head classes

## 🚀 Quick Start

### 1. Basic Usage

```python
from utils.memory_bank import MemoryBank

# Initialize memory bank
memory_bank = MemoryBank(
    num_classes=1000,
    feature_dim=2048,
    capacity_per_class=256,
    alpha_base=0.1,
    tail_threshold_percentile=20.0,
    device='cuda'
)

# Update with features
memory_bank.update(class_id=5, feature=torch.randn(2048))

# Get tail classes
tail_classes = memory_bank.get_tail_classes()
```

### 2. Training Integration

```python
from utils.memory_manager import MemoryManager

# Initialize memory manager
memory_manager = MemoryManager(
    model=your_model,
    num_classes=1000,
    capacity_per_class=256,
    alpha_base=0.1,
    tail_threshold_percentile=20.0,
    device='cuda'
)

# During training loop
for inputs, labels in train_loader:
    # Update memory bank automatically
    memory_manager.update_memory(inputs, labels)
    
    # Continue with normal training...
```

### 3. Command Line Training

```bash
# Enable memory bank during training
python main.py \
    --dataset_name imagenet \
    --model_name resnet50 \
    --use_memory_bank \
    --memory_capacity 256 \
    --memory_alpha 0.1 \
    --memory_tail_threshold 20.0
```

## 📊 Memory Bank Operations

### Feature Storage
```python
# Update memory with new feature
memory_bank.update(class_id, feature)

# Get EMA prototype for a class
prototype = memory_bank.get_prototype(class_id)

# Sample features from reservoir
features = memory_bank.sample_features(class_id, k=10)
```

### Class Analysis
```python
# Get comprehensive statistics
stats = memory_bank.get_class_statistics(class_id)

# Get class distribution
distribution = memory_bank.get_class_distribution()

# Check if class is tail/head/medium
is_tail = class_id in memory_bank.get_tail_classes()
```

### Semantic Prompt Data
```python
# Get data needed for semantic prompt generation
prompt_data = memory_manager.get_semantic_prompt_data(class_id, k_features=5)

# Get all tail class prompt data
tail_prompt_data = memory_manager.get_all_tail_prompt_data(k_features_per_class=10)

# Export detailed analysis
memory_manager.export_tail_class_analysis("./tail_analysis.json")
```

## 🎯 Tail Class Identification

### How It Works
1. **Frequency calculation**: Tracks samples per class during training
2. **Percentile thresholds**: Configurable thresholds (default: 20th percentile)
3. **Dynamic updates**: Reclassifies after each memory update
4. **Three categories**:
   - **Tail**: Classes below tail threshold (rare)
   - **Head**: Classes above head threshold (frequent)
   - **Medium**: Classes in between

### Configuration
```python
memory_bank = MemoryBank(
    tail_threshold_percentile=20.0,  # Classes below 20th percentile are "tail"
    # ... other parameters
)
```

## 💾 Persistence and Checkpoints

### Automatic Saving
```python
# Save every N epochs (configured in main.py)
if (epoch + 1) % 10 == 0:
    save_path = memory_manager.save_memory(epoch + 1, "checkpoint_name")
```

### Manual Save/Load
```python
# Save memory bank
memory_bank.save("./memory_checkpoint.json")

# Load memory bank
memory_bank.load("./memory_checkpoint.json")

# Load latest checkpoint
latest_path = memory_manager.load_latest_memory("checkpoint_prefix")
```

### File Structure
```
memory_checkpoints/
├── imagenet_resnet50/
│   ├── imagenet_resnet50_step_10.json
│   ├── imagenet_resnet50_step_10_stats.json
│   ├── imagenet_resnet50_step_20.json
│   └── imagenet_resnet50_step_20_stats.json
└── imagenet_resnet50_final.json
```

## 📈 Visualization and Analysis

### Built-in Visualizations
```python
# Generate comprehensive plots
memory_bank.visualize_class_distribution("./memory_analysis.png")

# Shows:
# - Class frequency distribution
# - Tail/Head/Medium class counts
# - Buffer utilization
# - Prototype norms
```

### Statistics Export
```python
# Get training statistics
stats = memory_manager.get_training_statistics()

# Get memory usage
usage = memory_bank.get_memory_usage()

# Export tail class analysis for external tools
memory_manager.export_tail_class_analysis("./tail_analysis.json")
```

## 🔍 Testing and Validation

### Run Test Suite
```bash
python test_memory_bank.py
```

### Test Components Individually
```python
# Test basic functionality
from test_memory_bank import test_memory_bank_basic
test_memory_bank_basic()

# Test memory manager
from test_memory_bank import test_memory_manager
test_memory_manager()

# Test visualization
from test_memory_bank import test_visualization
test_visualization()
```

## ⚙️ Configuration Parameters

### Memory Bank Parameters
- `num_classes`: Number of classes in dataset
- `feature_dim`: Dimension of feature vectors
- `capacity_per_class`: Maximum features per class in reservoir
- `alpha_base`: Base learning rate for EMA updates
- `tail_threshold_percentile`: Percentile for tail class identification
- `device`: Device for tensor operations

### Training Integration Parameters
- `use_memory_bank`: Enable/disable memory bank (default: True)
- `memory_capacity`: Features per class (default: 256)
- `memory_alpha`: EMA learning rate (default: 0.1)
- `memory_tail_threshold`: Tail threshold percentile (default: 20.0)

## 🧮 Mathematical Details

### EMA Update
```
M_c(t) = (1 - α_c) * M_c(t-1) + α_c * feature(t)
```

Where `α_c` is adaptive:
```
α_c = clamp(α_base / freq_ratio, α_min, α_max)
freq_ratio = class_frequency / total_samples
```

### Reservoir Sampling
For the i-th element in a stream of size n:
- **Selection probability**: `K/n` where K = capacity
- **Replacement**: Random position if selected
- **Uniformity**: Each element has equal probability of being retained

### Tail Classification
```
tail_threshold = percentile(class_frequencies, tail_threshold_percentile)
head_threshold = percentile(class_frequencies, 100 - tail_threshold_percentile)

tail_classes = {c | class_frequency[c] ≤ tail_threshold}
head_classes = {c | class_frequency[c] ≥ head_threshold}
medium_classes = {c | tail_threshold < class_frequency[c] < head_threshold}
```

## 🚨 Important Notes

### Memory Usage
- **Feature storage**: `num_classes × capacity_per_class × feature_dim × 4 bytes`
- **Example**: 1000 classes × 256 features × 2048 dim × 4 bytes = ~2 GB
- **Optimization**: Use `float16` for features if precision allows

### Performance Considerations
- **Update frequency**: Memory updates happen every batch
- **GPU memory**: Features are moved to CPU after extraction
- **Batch processing**: Consider updating every N batches if performance is critical

### Feature Extraction
- **Model modification**: Models now support `return_features=True`
- **Feature location**: Extracted from penultimate layer (before final classifier)
- **Normalization**: Features are L2-normalized before storage

## 🔮 Next Steps

This memory bank implementation provides the foundation for:

1. **Feature-level diffusion**: Train DDPM on stored features
2. **Semantic prompt generation**: Use prototypes and features for text prompts
3. **Image-level generation**: Guide diffusion models with semantic information
4. **Curriculum learning**: Gradually introduce synthetic data

## 📚 References

- **EMA**: Exponential Moving Average for stable prototypes
- **Reservoir Sampling**: Vitter's algorithm for uniform sampling
- **Long-tail learning**: Addressing class imbalance in deep learning
- **Feature hallucination**: Generating synthetic features for rare classes

## 🤝 Contributing

The memory bank is designed to be modular and extensible. Key extension points:

- **Custom sampling strategies**: Override `_update_reservoir` method
- **Alternative prototypes**: Modify EMA update logic
- **Additional metrics**: Extend statistics tracking
- **Visualization**: Add custom plotting functions

## 📞 Support

For questions or issues:
1. Check the test suite for usage examples
2. Review the inline documentation in the code
3. Examine the generated visualizations for insights
4. Use the export functions to analyze data externally
