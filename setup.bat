@echo off
echo ========================================
echo CIFAR-10 Extended Memory Bank Setup
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv csl_memory
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call csl_memory\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo WARNING: CUDA PyTorch installation failed, trying CPU version...
    pip install torch torchvision torchaudio
)

echo.
echo Installing other dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Creating necessary directories...
if not exist "memory_checkpoints" mkdir memory_checkpoints
if not exist "memory_checkpoints\cifar10_resnet32" mkdir memory_checkpoints\cifar10_resnet32
if not exist "synthetic_cifar10_images" mkdir synthetic_cifar10_images
if not exist "logs" mkdir logs

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Activate environment: csl_memory\Scripts\activate.bat
echo 2. Train model: python main_cifar10.py --epochs 20
echo 3. Generate prompts: python test_prompt_generation_cifar10.py
echo 4. Generate images: python generate_cifar10_images.py
echo.
echo For detailed instructions, see SETUP_GUIDE.md
echo.
pause
