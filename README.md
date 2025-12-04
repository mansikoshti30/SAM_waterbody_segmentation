# SAM Water Body Segmentation

Automated water body detection and segmentation from satellite imagery using Meta's **Segment Anything Model (SAM)**. This project combines spectral indices (NDWI/MNDWI) with deep learning to accurately identify and delineate water bodies in multi-spectral satellite images.

## Results Preview

<div align="center">
  <img src="images/result_comparison.png" alt="Water Body Segmentation Results" width="100%">
  <p><em>Left: Original Sentinel-2A satellite imagery | Right: Detected water bodies highlighted in red</em></p>
</div>

## Features

- **Multi-spectral water detection** using NDWI (Normalized Difference Water Index) and MNDWI (Modified NDWI)
- **SAM-based refinement** for precise water body segmentation
- **Batch processing** of multiple satellite images
- **Cloud masking** to automatically remove cloud interference
- **GeoJSON export** with area calculations (hectares and m²)
- **Visualization overlays** with detected water bodies highlighted
- **Configurable parameters** for different scenarios and sensors

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- SAM model checkpoint file (`sam_vit_b.pth`)

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mansikoshti30/SAM_waterbody_segmentation.git
cd SAM_waterbody_segmentation
```

2. **Create a virtual environment:**
```bash
conda create -n sam_env python=3.9
conda activate sam_env
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download SAM model checkpoint:**
   
   The SAM model file (~375 MB) is not included in this repository. Download it from the official source:
   
   ```bash
   # Linux/Mac
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth
   
   # Or download manually from:
   # https://github.com/facebookresearch/segment-anything#model-checkpoints
   ```
   
   Place `sam_vit_b.pth` in the project root directory.

5. **Download sample dataset (optional):**
   
   Test dataset for water body detection is available on Zenodo:
   
   [Download Sentinel-2 Water Bodies Dataset](https://zenodo.org/records/5205674)
   
   Extract the dataset to `data_set/dset-s2/` directory.

6. **Run the processor:**
```bash
python test_sam_5_overall.py
```

> **Note:** For detailed installation instructions, see [INSTALL.md](INSTALL.md)

## Project Structure

```
SAM_project/
├── test_sam_1.py              # Basic SAM water detection
├── test_sam_1_overall.py      # Batch processing version 1
├── test_sam_2.py              # Enhanced detection with NDWI
├── test_sam_3.py              # Advanced segmentation
├── test_sam_3_overall.py      # Batch processing version 3
├── test_sam_4.py              # Experimental version
├── test_sam_5_overall.py      # Latest batch processor with cloud masking
├── sam_vit_b.pth              # SAM model weights (download separately)
├── sam_polygons.geojson       # Sample output
├── data_set/
│   └── dset-s2/
│       ├── tra_scene/         # Training scene images
│       ├── tra_truth/         # Training ground truth
│       ├── val_scene/         # Validation scene images
│       └── val_truth/         # Validation ground truth
├── batch_outputs_3/           # Output directory for version 3
├── batch_outputs_5/           # Output directory for version 5
└── batch_overlays/            # Visualization overlays
```

## Usage

### Basic Usage

Process all TIFF files in the input folder:

```bash
python test_sam_5_overall.py
```

Results will be saved in `batch_outputs_5/` with overlays and GeoJSON polygons.

### Custom Configuration

Edit the configuration parameters in `test_sam_5_overall.py`:

```python
INPUT_FOLDER = "data_set/dset-s2/tra_scene"   # Input TIFF folder
OUTPUT_FOLDER = "batch_outputs_5"             # Output folder
SAM_CHECKPOINT = "sam_vit_b.pth"              # SAM model path
MODEL_TYPE = "vit_b"                          # Model architecture

# Tunable parameters
MIN_REGION_AREA = 500        # Minimum water body area (pixels)
PAD = 12                     # Padding around candidate boxes
MORPH_KERNEL = (5, 5)        # Morphological operation kernel
FINAL_CLOSE_KERNEL = (9, 9)  # Final closing kernel
SMALL_ISLAND_MIN = 100       # Remove tiny islands threshold
```

### Input Data Format

The project expects multi-band GeoTIFF files with the following band structure:
- Band 1: Red
- Band 2: Green
- Band 3: Blue
- Band 4: NIR (Near-Infrared) - optional but recommended
- Band 6: SWIR (Short-Wave Infrared) - optional, used for MNDWI

### Output Files

For each input image, the script generates:
- **`{filename}_overlay.png`** - RGB visualization with water bodies highlighted in red
- **`{filename}_polygons.geojson`** - Vector polygons with area attributes (hectares, m²)

Example GeoJSON output:
```json
{
  "type": "Feature",
  "properties": {
    "area_m2": 15000.5,
    "area_ha": 1.5
  },
  "geometry": {...}
}
```

> **Note:** For more examples and advanced usage, see [USAGE.md](USAGE.md)

## Methodology

1. **Spectral Index Calculation**:
   - MNDWI (if SWIR available): `(Green - SWIR) / (Green + SWIR)`
   - NDWI (if NIR available): `(Green - NIR) / (Green + NIR)`
   - Fallback: Blue-Red ratio

2. **Coarse Segmentation**:
   - Otsu thresholding on spectral index
   - Morphological operations for noise removal
   - Cloud masking using RGB brightness/whiteness

3. **SAM Refinement**:
   - Region-based processing with bounding boxes
   - Point and box prompts for SAM
   - IoU-based mask selection and validation
   - Controlled expansion to prevent over-segmentation

4. **Post-processing**:
   - Morphological closing for gap filling
   - Small island removal
   - Vector polygon generation with area calculation

## Script Versions

| Script | Description | Best For |
|--------|-------------|----------|
| `test_sam_5_overall.py` | **Recommended** - Production-ready with cloud masking | General use, best accuracy |
| `test_sam_3_overall.py` | Advanced morphological refinement | Complex water bodies |
| `test_sam_1_overall.py` | Basic NDWI + SAM implementation | Quick testing |
| `test_sam_2.py` | Enhanced spectral indices | Images with SWIR band |
| `test_sam_4.py` | Experimental features | Development/testing |

## Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [**Segment Anything Model (SAM)**](https://github.com/facebookresearch/segment-anything) by Meta AI Research
- **Sentinel-2** satellite imagery by ESA (European Space Agency)
- Open source geospatial community

## Documentation

- [Installation Guide](INSTALL.md) - Detailed setup instructions
- [Usage Examples](USAGE.md) - Tutorials and examples
- [Quick Reference](QUICKREF.md) - Command and parameter reference
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Changelog](CHANGELOG.md) - Version history

## Support

- **Issues**: [GitHub Issues](https://github.com/mansikoshti30/SAM_waterbody_segmentation/issues)
- **Documentation**: See guides above
- **Star this repo** if you find it useful!

## References

- [SAM Paper (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643)
- [Sentinel-2 Mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [Water Bodies Dataset on Zenodo](https://zenodo.org/records/5205674) - Test dataset for validation
- [NDWI Index (McFeeters, 1996)](https://en.wikipedia.org/wiki/Normalized_difference_water_index)
- [MNDWI Index (Xu, 2006)](https://www.tandfonline.com/doi/abs/10.1080/01431160600589179)

---

**Made for the remote sensing community**


