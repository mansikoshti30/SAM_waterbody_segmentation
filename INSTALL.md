# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Conda or pip package manager
- CUDA-capable GPU (optional but recommended for faster processing)
- At least 8GB RAM (16GB+ recommended)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SAM_project.git
cd SAM_project
```

### 2. Create Virtual Environment

#### Using Conda (Recommended)

```bash
conda create -n sam_env python=3.9
conda activate sam_env
```

#### Using venv

```bash
python -m venv sam_env
# On Windows:
sam_env\Scripts\activate
# On Linux/Mac:
source sam_env/bin/activate
```

### 3. Install PyTorch

Choose the appropriate command for your system from [PyTorch's official website](https://pytorch.org/get-started/locally/).

#### For CUDA 11.8 (GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU only:
```bash
pip install torch torchvision
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download SAM Model

Download the SAM ViT-B model checkpoint:

```bash
# Using wget (Linux/Mac)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth

# Using curl (Mac)
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o sam_vit_b.pth

# Or download manually from:
# https://github.com/facebookresearch/segment-anything#model-checkpoints
```

Place the downloaded `sam_vit_b.pth` file in the project root directory.

### 6. Verify Installation

```bash
python -c "import torch; import cv2; import rasterio; import geopandas; from segment_anything import sam_model_registry; print('All imports successful!')"
```

## Troubleshooting

### GDAL/Rasterio Installation Issues

If you encounter issues installing `rasterio`, try using conda:

```bash
conda install -c conda-forge rasterio
```

### GeoPandas Installation Issues

```bash
conda install -c conda-forge geopandas
```

### CUDA/GPU Issues

Check if CUDA is available:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not detected but you have a GPU, reinstall PyTorch with the correct CUDA version.

### Memory Issues

If you encounter out-of-memory errors:
- Use CPU instead of GPU by setting `DEVICE = "cpu"` in the script
- Process smaller batches
- Reduce image resolution if possible

## Alternative Installation with Docker

Coming soon...

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Deactivate environment
conda deactivate  # or deactivate for venv

# Remove environment
conda env remove -n sam_env  # or rm -rf sam_env for venv
```

## Next Steps

After installation, see the [README.md](README.md) for usage instructions.
