#!/usr/bin/env python3
"""
Quick setup script for CIFAR-10 Extended Memory Bank
Automates environment setup and dependency installation
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_git():
    """Check if Git is installed."""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Git is not installed")
    print("   Please install Git from: https://git-scm.com/")
    return False

def create_requirements_file():
    """Create a comprehensive requirements.txt file."""
    requirements = """torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=8.3.0
diffusers>=0.21.0
transformers>=4.21.0
accelerate>=0.20.0
scikit-learn>=1.1.0
tqdm>=4.64.0
pathlib2>=2.3.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Created comprehensive requirements.txt")

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    # Try to install PyTorch with CUDA support first
    pytorch_commands = [
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116",
        "pip install torch torchvision torchaudio"  # CPU-only fallback
    ]
    
    pytorch_installed = False
    for command in pytorch_commands:
        if run_command(command, "Installing PyTorch"):
            pytorch_installed = True
            break
    
    if not pytorch_installed:
        print("❌ Failed to install PyTorch")
        return False
    
    # Install other requirements
    if not run_command("pip install -r requirements.txt", "Installing other dependencies"):
        print("❌ Failed to install other dependencies")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "memory_checkpoints",
        "memory_checkpoints/cifar10_resnet32",
        "synthetic_cifar10_images",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created necessary directories")

def run_basic_tests():
    """Run basic import tests to verify installation."""
    print("\n🧪 Running basic tests...")
    
    test_script = """
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
import sklearn
import tqdm

print("✅ All basic imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
"""
    
    with open('test_imports.py', 'w') as f:
        f.write(test_script)
    
    if run_command("python test_imports.py", "Testing imports"):
        os.remove('test_imports.py')
        return True
    else:
        print("❌ Import tests failed")
        return False

def main():
    """Main setup function."""
    print("🚀 CIFAR-10 Extended Memory Bank - Quick Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_git():
        return False
    
    # Create requirements file
    create_requirements_file()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    create_directories()
    
    # Run tests
    if not run_basic_tests():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Train the memory bank: python main_cifar10.py --epochs 20")
    print("2. Generate prompts: python test_prompt_generation_cifar10.py")
    print("3. Generate images: python generate_cifar10_images.py")
    print("\n📖 For detailed instructions, see SETUP_GUIDE.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
