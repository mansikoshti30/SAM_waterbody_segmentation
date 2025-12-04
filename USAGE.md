# Example Usage Guide

This guide provides examples of how to use the SAM Water Body Segmentation scripts.

## Quick Start

### Example 1: Process a Single Folder

The simplest way to process your satellite images:

```bash
python test_sam_5_overall.py
```

This will process all TIFF files in the `data_set/dset-s2/tra_scene` folder and save outputs to `batch_outputs_5/`.

### Example 2: Custom Input/Output Folders

Edit `test_sam_5_overall.py` to change the input and output paths:

```python
INPUT_FOLDER = "path/to/your/satellite/images"
OUTPUT_FOLDER = "path/to/output/folder"
```

### Example 3: Adjust Detection Sensitivity

For better detection of small water bodies:

```python
MIN_REGION_AREA = 200        # Lower threshold (default: 500)
SMALL_ISLAND_MIN = 50        # Keep smaller islands (default: 100)
```

For more conservative detection (fewer false positives):

```python
MIN_REGION_AREA = 1000       # Higher threshold
MIN_IOU_ACCEPT = 0.1         # Stricter IoU requirement
```

## Configuration Examples

### Example 4: High-Resolution Processing

For high-resolution imagery:

```python
PAD = 20                     # Larger padding
MORPH_KERNEL = (7, 7)        # Larger morphological kernel
FINAL_CLOSE_KERNEL = (11, 11)
```

### Example 5: Low-Resolution Processing

For lower resolution or coarser detection:

```python
PAD = 8
MORPH_KERNEL = (3, 3)
FINAL_CLOSE_KERNEL = (5, 5)
MIN_REGION_AREA = 100
```

### Example 6: Cloud-Heavy Scenes

Adjust cloud masking parameters:

```python
# In the quick_cloud_mask function
cloud_mask = quick_cloud_mask(R, G, B, bright_th=220, whiteness_diff=30)
```

## Output Examples

After processing, you'll get:

### 1. Overlay Images (`*_overlay.png`)
RGB visualization with detected water bodies highlighted in red

### 2. GeoJSON Files (`*_polygons.geojson`)
Vector polygons with attributes:
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {
      "area_m2": 15000.5,
      "area_ha": 1.5
    },
    "geometry": {
      "type": "Polygon",
      "coordinates": [...]
    }
  }]
}
```

## Advanced Usage

### Example 7: Process Specific Files

Create a custom script to process only specific files:

```python
import os
from test_sam_5_overall import process_one_tif, sam_model_registry, SamPredictor

# Load SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# Process specific files
files = ["scene_001.tif", "scene_005.tif"]
for fname in files:
    tif_path = os.path.join("data_set/dset-s2/tra_scene", fname)
    process_one_tif(tif_path, sam, predictor, "custom_output")
```

### Example 8: Programmatic Access

Use the functions in your own scripts:

```python
import rasterio
import numpy as np
from test_sam_5_overall import compute_ndwi, compute_mndwi, otsu_thresh_index

# Open your image
with rasterio.open("your_image.tif") as ds:
    G = ds.read(2)   # Green band
    NIR = ds.read(4) # NIR band
    
    # Compute NDWI
    ndwi = compute_ndwi(G, NIR)
    
    # Apply Otsu threshold
    water_mask, threshold = otsu_thresh_index(ndwi)
    
    print(f"Otsu threshold: {threshold}")
    print(f"Water pixels: {water_mask.sum()}")
```

## Expected Results

### Processing Times (approximate)
- Small image (1000x1000 px): ~5-10 seconds
- Medium image (2000x2000 px): ~15-30 seconds  
- Large image (5000x5000 px): ~1-3 minutes

Times vary based on:
- GPU/CPU
- Number of water bodies
- Image complexity

### Accuracy
Results depend on:
- Image quality and resolution
- Cloud coverage
- Water body characteristics
- Parameter tuning

## Viewing Results

### GeoJSON in QGIS
1. Open QGIS
2. Drag and drop the `.geojson` file
3. Style the layer and view attributes

### GeoJSON in Python
```python
import geopandas as gpd

gdf = gpd.read_file("output_polygons.geojson")
print(gdf.head())
print(f"Total water bodies: {len(gdf)}")
print(f"Total area: {gdf['area_ha'].sum():.2f} hectares")
```

## Tips for Best Results

1. **Choose the right script**: Use `test_sam_5_overall.py` for production work
2. **Pre-process your data**: Ensure TIFFs are properly georeferenced
3. **Tune parameters**: Start with defaults, then adjust based on results
4. **Check outputs**: Review overlay images to validate detection quality
5. **Use GPU**: Much faster than CPU for large batches

## Troubleshooting Common Issues

### No water detected
- Lower `MIN_REGION_AREA`
- Check if image has NIR/SWIR bands
- Try different scripts (test_sam_3_overall.py)

### Too many false positives
- Increase `MIN_REGION_AREA`
- Adjust `MIN_IOU_ACCEPT` higher
- Check cloud masking parameters

### Memory errors
- Process fewer files at once
- Use CPU instead of GPU
- Reduce padding and kernel sizes

## Next Steps

- Experiment with different parameters
- Try different versions (test_sam_1 through test_sam_5)
- Integrate results into your GIS workflow
- Contribute improvements back to the project!
