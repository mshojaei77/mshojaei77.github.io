---
layout: post
title: "Setting Up Your LLM Development Environment: A Complete Guide"
date: 2025-01-07
author: "Mohammad Shojaei"
tags: ["development", "environment", "setup", "pytorch", "transformers"]
excerpt: "Learn how to set up a complete development environment for working with Large Language Models, including Python, PyTorch, Transformers, and essential tools."
image: "/assets/img/blog/dev-environment.png"
---

# Setting Up Your LLM Development Environment

Before diving into the exciting world of Large Language Models, you need a robust development environment. This guide will walk you through setting up everything you need to start building, training, and deploying LLMs.

## Prerequisites

Before we begin, ensure you have:
- A computer with at least 16GB RAM (32GB+ recommended)
- NVIDIA GPU with 8GB+ VRAM (for training/fine-tuning)
- Basic familiarity with Python and command line

## Step 1: Python Environment Setup

### Install Python 3.9+

We recommend Python 3.9 or later for optimal compatibility:

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv

# On macOS (using Homebrew)
brew install python@3.9

# On Windows
# Download from python.org or use Windows Store
```

### Create a Virtual Environment

```bash
# Create virtual environment
python3.9 -m venv llm-env

# Activate it
# On Linux/macOS:
source llm-env/bin/activate

# On Windows:
llm-env\Scripts\activate
```

## Step 2: Core Libraries Installation

### PyTorch Installation

Install PyTorch with CUDA support (if you have an NVIDIA GPU):

```bash
# For CUDA 11.8 (check your CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (not recommended for training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Hugging Face Ecosystem

```bash
# Core transformers library
pip install transformers

# Datasets for easy data loading
pip install datasets

# Accelerate for distributed training
pip install accelerate

# Evaluate for model evaluation
pip install evaluate

# PEFT for parameter-efficient fine-tuning
pip install peft
```

### Additional Essential Libraries

```bash
# Data manipulation and analysis
pip install pandas numpy matplotlib seaborn

# Jupyter for interactive development
pip install jupyter jupyterlab

# Weights & Biases for experiment tracking
pip install wandb

# TensorBoard for visualization
pip install tensorboard

# Gradio for quick demos
pip install gradio
```

## Step 3: Development Tools

### Code Editor Setup

We recommend **Visual Studio Code** with these extensions:

- Python
- Jupyter
- GitLens
- Python Docstring Generator
- autoDocstring

### Git Configuration

```bash
# Configure Git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Install Git LFS for large files
git lfs install
```

## Step 4: GPU Setup (NVIDIA)

### CUDA Installation

1. Check your GPU compatibility:
```bash
nvidia-smi
```

2. Install CUDA Toolkit (version 11.8 recommended):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Follow installation instructions for your OS

3. Verify installation:
```bash
nvcc --version
```

### Memory Optimization

For working with large models, configure GPU memory:

```python
# In your Python scripts
import torch

# Enable memory fraction (use 90% of GPU memory)
torch.cuda.set_per_process_memory_fraction(0.9)

# Enable memory growth (allocate as needed)
torch.backends.cudnn.benchmark = True
```

## Step 5: Project Structure

Create a standardized project structure:

```
llm-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── checkpoints/
│   └── final/
├── notebooks/
│   ├── exploration/
│   └── experiments/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── configs/
├── scripts/
├── requirements.txt
├── README.md
└── .gitignore
```

## Step 6: Configuration Files

### requirements.txt

Create a `requirements.txt` file:

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
acceleerate>=0.20.0
evaluate>=0.4.0
peft>=0.4.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
wandb>=0.15.0
tensorboard>=2.13.0
gradio>=3.35.0
```

### .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/

# Jupyter
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
llm-env/

# Model files
*.bin
*.safetensors
models/checkpoints/
models/final/

# Data
data/raw/
data/processed/
*.csv
*.json
*.parquet

# Logs
logs/
*.log
wandb/
runs/

# IDE
.vscode/
.idea/
```

## Step 7: Verification Script

Create a verification script to test your setup:

```python
# test_setup.py
import torch
import transformers
import datasets
import sys

def test_setup():
    print("Testing LLM Development Environment...\n")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Transformers
    print(f"Transformers version: {transformers.__version__}")
    
    # Datasets
    print(f"Datasets version: {datasets.__version__}")
    
    # Test model loading
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        print("\n✅ Model loading test: PASSED")
    except Exception as e:
        print(f"\n❌ Model loading test: FAILED - {e}")
    
    print("\n🎉 Setup verification complete!")

if __name__ == "__main__":
    test_setup()
```

Run the verification:

```bash
python test_setup.py
```

## Step 8: Optional Enhancements

### Docker Setup (Advanced)

For reproducible environments:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
```

### Cloud Setup

For cloud development:
- **Google Colab**: Free GPU access
- **Kaggle Notebooks**: Free GPU/TPU
- **AWS SageMaker**: Professional cloud ML
- **Google Cloud AI Platform**: Scalable training

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```python
   # Reduce batch size or use gradient accumulation
   torch.cuda.empty_cache()
   ```

2. **Slow model loading**:
   ```python
   # Use local cache
   from transformers import AutoModel
   model = AutoModel.from_pretrained("model-name", cache_dir="./cache")
   ```

3. **Import errors**:
   ```bash
   # Reinstall in correct order
   pip uninstall torch transformers
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install transformers
   ```

## Next Steps

With your environment ready, you can:

1. **Start with tutorials**: Follow our [beginner tutorials](/tutorials/)
2. **Explore models**: Browse [Hugging Face Model Hub](https://huggingface.co/models)
3. **Join the community**: Connect with other developers in our [Telegram group](https://t.me/AI_LLMs)
4. **Read the book**: Dive deeper with our [comprehensive guide](/book/)

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Git LFS Documentation](https://git-lfs.github.io/)

---

*Having trouble with setup? Join our [community](https://t.me/AI_LLMs) for help, or check out our [troubleshooting guide](/resources/#troubleshooting).*

*This post was originally published on [mshojaei77.github.io](https://mshojaei77.github.io/blog/). For more tutorials and updates, visit our main site.*