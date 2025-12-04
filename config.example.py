# SAM Water Body Segmentation - Configuration File
# Copy this file to config.py and customize as needed

# Model Configuration
SAM_CHECKPOINT = "sam_vit_b.pth"
MODEL_TYPE = "vit_b"  # Options: vit_b, vit_l, vit_h

# Input/Output Paths
INPUT_FOLDER = "data_set/dset-s2/tra_scene"
OUTPUT_FOLDER = "batch_outputs_5"

# Device Configuration
# Options: "cuda", "cpu", or "auto" (automatic detection)
DEVICE = "auto"

# Water Detection Parameters
MIN_REGION_AREA = 500          # Minimum area of water bodies to detect (pixels)
SMALL_ISLAND_MIN = 100         # Remove tiny isolated regions smaller than this

# SAM Processing Parameters
PAD = 12                       # Padding around candidate boxes (pixels)
MIN_IOU_ACCEPT = 0.06          # Minimum IoU threshold for accepting SAM predictions
INTERSECT_MIN_RATIO = 0.2      # Minimum intersection ratio with coarse mask

# Morphological Operation Parameters
MORPH_KERNEL = (5, 5)          # Kernel size for initial morphological operations
FINAL_CLOSE_KERNEL = (9, 9)    # Kernel size for final closing operation
DILATE_ALLOWED_KSIZE = (7, 7)  # Maximum allowed expansion of SAM predictions

# Cloud Masking Parameters
CLOUD_BRIGHT_THRESHOLD = 230   # Brightness threshold for cloud detection
CLOUD_WHITENESS_DIFF = 35      # Maximum color difference for cloud detection

# Output Options
SAVE_OVERLAY = True            # Save RGB overlay images with detected water bodies
SAVE_GEOJSON = True            # Save vector polygons as GeoJSON
SAVE_MASK_PNG = False          # Save binary mask as PNG (set to True if needed)
SAVE_MASK_TIF = False          # Save georeferenced mask as GeoTIFF (set to True if needed)

# Visualization Parameters
OVERLAY_ALPHA = 0.4            # Transparency of water overlay (0.0-1.0)
WATER_COLOR = [255, 0, 0]      # Color for water bodies in overlay (RGB)

# Advanced Options
MULTIMASK_OUTPUT = True        # Use SAM's multimask output for better results
USE_CLOUD_MASKING = True       # Apply cloud masking to remove cloud interference
VERBOSE = True                 # Print detailed processing information
