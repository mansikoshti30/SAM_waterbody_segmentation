# SAM Water Body Segmentation Project

This project uses Meta's **Segment Anything Model (SAM)** to detect and segment water bodies from satellite imagery. It processes multi-spectral satellite images (Sentinel-2) and generates accurate water masks using spectral indices (NDWI/MNDWI) combined with SAM's powerful segmentation capabilities.

## 🌊 Features

- **Multi-spectral water detection** using NDWI (Normalized Difference Water Index) and MNDWI (Modified NDWI)
- **SAM-based refinement** for precise water body segmentation
- **Batch processing** of satellite imagery
- **Cloud masking** to remove cloud interference
- **GeoJSON export** with area calculations (in hectares and square meters)
- **Visualization overlays** showing detected water bodies on RGB imagery
- **Multiple processing strategies** with configurable parameters

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- SAM model checkpoint file (`sam_vit_b.pth`)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SAM_project.git
cd SAM_project
```

2. Create a virtual environment (recommended):
```bash
conda create -n sam_env python=3.9
conda activate sam_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the SAM model checkpoint:
   - Download `sam_vit_b.pth` from [SAM GitHub repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
   - Place it in the project root directory

## 📂 Project Structure

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

## 🎯 Usage

### Basic Usage

Run the latest batch processor on your satellite imagery:

```bash
python test_sam_5_overall.py
```

### Configuration

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

For each input TIFF file, the script generates:
- `{filename}_overlay.png` - RGB visualization with water bodies highlighted in red
- `{filename}_polygons.geojson` - Vector polygons of detected water bodies with area attributes

## 🔬 Methodology

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

## 📊 Different Versions

- **test_sam_1**: Basic implementation with NDWI + SAM
- **test_sam_2**: Enhanced with multiple spectral indices
- **test_sam_3**: Advanced with morphological refinement
- **test_sam_4**: Experimental optimizations
- **test_sam_5**: Production-ready with cloud masking and robust error handling

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- Sentinel-2 satellite imagery

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🔗 References

- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [Sentinel-2 Documentation](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [NDWI Index Reference](https://en.wikipedia.org/wiki/Normalized_difference_water_index)
