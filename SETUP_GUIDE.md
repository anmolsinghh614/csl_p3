# CIFAR-10 Extended Memory Bank - Setup Guide

This guide will help you clone and set up the repository on a new machine to run CIFAR-10 image generation.

## Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended for faster generation)
- At least 8GB RAM
- At least 10GB free disk space

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_NAME>

# Or if you have the repository locally, copy it to the new machine
```

## Step 2: Set Up Python Environment

### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n csl_memory python=3.9
conda activate csl_memory

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

### Option B: Using Virtual Environment
```bash
# Create virtual environment
python -m venv csl_memory
source csl_memory/bin/activate  # On Windows: csl_memory\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

## Step 3: Download CIFAR-10 Dataset

The CIFAR-10 dataset will be automatically downloaded when you run the training script. It's approximately 170MB.

## Step 4: Run the Complete Pipeline

### Step 4.1: Train the Memory Bank
```bash
# Train the model and build memory bank
python main_cifar10.py --epochs 20 --batch_size 128 --lr 0.1
```

This will:
- Download CIFAR-10 dataset
- Train ResNet32 on CIFAR-10-LT (long-tail distribution)
- Build and save memory bank checkpoints

### Step 4.2: Generate Prompts
```bash
# Generate text prompts for tail classes
python test_prompt_generation_cifar10.py
```

This will:
- Load the trained memory bank
- Generate text prompts for each CIFAR-10 class
- Save prompts to `cifar10_tail_class_prompts.json`

### Step 4.3: Generate Synthetic Images
```bash
# Generate synthetic images using Stable Diffusion
python generate_cifar10_images.py
```

This will:
- Download Stable Diffusion model (~5GB, first run only)
- Generate synthetic images for each class
- Save images to `./synthetic_cifar10_images/`

## Step 5: Verify Results

Check the following outputs:
- `./memory_checkpoints/cifar10_resnet32/` - Memory bank checkpoints
- `cifar10_tail_class_prompts.json` - Generated text prompts
- `./synthetic_cifar10_images/` - Generated synthetic images
- `./synthetic_cifar10_images/generation_metadata.json` - Generation statistics

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   - Reduce batch size in training
   - Use CPU mode for generation (slower but works)
   - Close other GPU applications

2. **Missing Dependencies**
   - Ensure you're in the correct virtual environment
   - Run `pip install -r requirements.txt` again
   - Check Python version compatibility

3. **Slow Generation**
   - First run downloads ~5GB models (one-time)
   - Use GPU if available
   - Reduce `num_inference_steps` in generation script

4. **Dataset Download Issues**
   - Check internet connection
   - Verify write permissions in project directory

### Performance Tips:

- Use GPU for both training and generation
- Adjust `num_inference_steps` (lower = faster, lower quality)
- Use `num_images_per_prompt` based on your needs
- Monitor GPU memory usage

## File Structure

```
├── main_cifar10.py              # Main training script
├── test_prompt_generation_cifar10.py  # Prompt generation
├── generate_cifar10_images.py   # Image generation
├── requirements.txt              # Dependencies
├── models/                      # Neural network architectures
├── utils/                       # Utility functions
├── dataloaders/                 # Data loading utilities
└── data/                        # Dataset storage
```

## Expected Output

After successful completion, you should have:
- Trained model with memory bank
- Text prompts for each CIFAR-10 class
- Synthetic images for training augmentation
- Complete pipeline for memory-conditioned image generation

## Next Steps

- Experiment with different prompt generation strategies
- Adjust generation parameters for better image quality
- Integrate synthetic images into training pipeline
- Extend to other datasets (CIFAR-100, ImageNet-LT)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure sufficient system resources
4. Check console output for specific error messages
