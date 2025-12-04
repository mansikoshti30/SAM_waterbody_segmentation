# Quick Reference Guide

Quick reference for common tasks in the SAM Water Body Segmentation project.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run latest version (recommended)
python test_sam_5_overall.py

# Run with different versions
python test_sam_1_overall.py  # Basic version
python test_sam_3_overall.py  # Advanced morphology
```

## File Overview

| File | Purpose | Use When |
|------|---------|----------|
| `test_sam_5_overall.py` | Production-ready processor | General use, best results |
| `test_sam_3_overall.py` | Advanced morphology | Complex water bodies |
| `test_sam_1_overall.py` | Simple processor | Quick testing |
| `test_sam_2.py` | Enhanced indices | SWIR band available |
| `test_sam_4.py` | Experimental | Testing new features |

## Key Parameters

### Detection Sensitivity

```python
# More sensitive (detect smaller water bodies)
MIN_REGION_AREA = 200
SMALL_ISLAND_MIN = 50

# Less sensitive (fewer false positives)
MIN_REGION_AREA = 1000
SMALL_ISLAND_MIN = 200
```

### Processing Quality

```python
# High quality (slower)
PAD = 20
MORPH_KERNEL = (7, 7)
FINAL_CLOSE_KERNEL = (11, 11)

# Fast processing (lower quality)
PAD = 8
MORPH_KERNEL = (3, 3)
FINAL_CLOSE_KERNEL = (5, 5)
```

### Cloud Handling

```python
# Strict cloud removal
bright_th = 220
whiteness_diff = 30

# Lenient cloud removal
bright_th = 240
whiteness_diff = 40
```

## Common Issues & Quick Fixes

| Issue | Solution |
|-------|----------|
| No water detected | Lower `MIN_REGION_AREA` to 200-300 |
| Too many false positives | Increase `MIN_REGION_AREA` to 800-1000 |
| Out of memory | Change `DEVICE = "cpu"` |
| Slow processing | Use GPU: `DEVICE = "cuda"` |
| Missing NIR band | Script auto-falls back to RGB |
| Cloud interference | Increase `bright_th` to 240 |

## Input Requirements

**Required:**
- Multi-band GeoTIFF format (`.tif`)
- At least RGB bands (3 bands)

**Recommended:**
- RGB + NIR (4 bands) for NDWI
- RGB + NIR + SWIR (6 bands) for MNDWI
- Georeferenced with CRS

**Band Order (Sentinel-2):**
1. Red
2. Green
3. Blue
4. NIR (Near-Infrared)
5. (optional)
6. SWIR (Short-Wave Infrared)

## Output Files

```
batch_outputs_5/
├── scene_001_overlay.png      # Visual overlay
└── scene_001_polygons.geojson # Vector polygons
```

## Quick Python API

```python
from test_sam_5_overall import *

# Load SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# Process one file
process_one_tif("image.tif", sam, predictor, "output_folder")

# Compute indices
ndwi = compute_ndwi(green_band, nir_band)
mndwi = compute_mndwi(green_band, swir_band)

# Apply threshold
mask, threshold = otsu_thresh_index(ndwi)
```

## Performance Tips

✅ **DO:**
- Use GPU for batch processing
- Pre-filter images (remove cloudy scenes)
- Process similar images with same parameters
- Use appropriate `MIN_REGION_AREA` for your resolution

❌ **DON'T:**
- Process images with >90% cloud cover
- Use extremely small `MIN_REGION_AREA` (<50)
- Mix different sensor types without parameter adjustment

## Accuracy Checklist

- [ ] Visual inspection of overlay images
- [ ] Compare with known water bodies
- [ ] Check for false positives (clouds, shadows, buildings)
- [ ] Verify GeoJSON areas are reasonable
- [ ] Test on diverse scenes (urban, rural, coastal)

## Directory Structure

```
SAM_project/
├── test_sam_*.py          # Processing scripts
├── sam_vit_b.pth          # Model (download separately)
├── requirements.txt       # Dependencies
├── README.md              # Main documentation
├── data_set/              # Input data (gitignored)
└── batch_outputs_*/       # Results (gitignored)
```

## Conda Commands

```bash
# Create environment
conda create -n sam_env python=3.9
conda activate sam_env

# Install with conda
conda install -c conda-forge gdal rasterio geopandas

# Deactivate
conda deactivate

# Remove environment
conda env remove -n sam_env
```

## Git Commands

```bash
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/user/repo.git
git push -u origin main

# Regular updates
git add .
git commit -m "Update message"
git push

# Check status
git status
```

## Useful Links

- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [PyTorch Install](https://pytorch.org/get-started/locally/)
- [Rasterio Docs](https://rasterio.readthedocs.io/)
- [GeoPandas Docs](https://geopandas.org/)

## Support

- Documentation: See `README.md`
- Examples: See `USAGE.md`
- Installation: See `INSTALL.md`
- Issues: GitHub Issues page

---

*Last updated: December 2025*
