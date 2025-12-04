import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from segment_anything import sam_model_registry, SamPredictor

# ---------------- CONFIG ----------------
IMAGE_TIF = "test_data.tif"          
SAM_CHECKPOINT = "sam_vit_b.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NDWI_THRESHOLD = 0.0             # good default for water
RGB_BLUE_THRESHOLD = 30          # fallback if no NIR band
MIN_REGION_AREA = 300
PAD = 10                         # extra padding around detected water

OUTPUT_PNG = "binary_water_mask.png"
OUTPUT_TIF = "binary_water_mask.tif"
OUTPUT_OVERLAY = "overlay_water.png"
# ----------------------------------------

def compute_ndwi(green, nir):
    g = green.astype(float)
    n = nir.astype(float)
    return (g - n) / (g + n + 1e-6)

def get_candidate_mask(dataset):
    bands = dataset.count
    
    if bands >= 4:
        print("Using NDWI (Green & NIR) for water detection.")
        R = dataset.read(1)
        G = dataset.read(2)
        B = dataset.read(3)
        NIR = dataset.read(4)

        ndwi = compute_ndwi(G, NIR)
        cand = (ndwi > NDWI_THRESHOLD).astype(np.uint8)

        # Normalize RGB for SAM (stretch for visibility)
        rgb = np.stack([R, G, B], axis=-1)
        rgb = (255 * (rgb / np.percentile(rgb, 98))).clip(0,255).astype(np.uint8)
        return cand, rgb

    else:
        print("No NIR band → using RGB blue dominance.")
        R = dataset.read(1)
        G = dataset.read(2)
        B = dataset.read(3)

        Bc = B.astype(float)
        Rc = R.astype(float)
        cand = ((Bc - Rc) > RGB_BLUE_THRESHOLD).astype(np.uint8)

        rgb = np.stack([R, G, B], axis=-1)
        rgb = (255 * (rgb / np.percentile(rgb, 98))).clip(0,255).astype(np.uint8)
        return cand, rgb

def expand_box(box, pad, W, H):
    x0,y0,x1,y1 = box
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(W-1, x1 + pad),
        min(H-1, y1 + pad)
    ]

def main():
    assert os.path.exists(IMAGE_TIF), f"File not found: {IMAGE_TIF}"
    assert os.path.exists(SAM_CHECKPOINT), "Missing SAM checkpoint."

    ds = rasterio.open(IMAGE_TIF)
    H, W = ds.height, ds.width

    # Step 1: Candidate water regions
    candidate_mask, rgb_image = get_candidate_mask(ds)
    candidate_mask = cv2.medianBlur(candidate_mask, 5)

    # Step 2: Connected components
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
    boxes = []
    for i in range(1, nlab):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_REGION_AREA: continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append([x, y, x+w_box, y+h_box])

    print(f"Found {len(boxes)} water candidate region(s).")

    # Step 3: SAM Predictor
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb_image)

    final_mask = np.zeros((H,W), dtype=np.uint8)

    for box in boxes:
        bx = expand_box(box, PAD, W, H)

        masks, scores, _ = predictor.predict(
            box=np.array(bx),
            multimask_output=True
        )

        best = np.argmax(scores)
        final_mask = np.logical_or(final_mask, masks[best]).astype(np.uint8)

    # Step 4: Save PNG binary mask
    binary = (final_mask * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_PNG, binary)
    print("Saved PNG mask:", OUTPUT_PNG)

    # Step 5: Save GeoTIFF binary mask
    tif_profile = ds.profile.copy()
    tif_profile.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw"
    })

    with rasterio.open(OUTPUT_TIF, "w", **tif_profile) as dst:
        dst.write(binary, 1)

    print("Saved GeoTIFF mask:", OUTPUT_TIF)

    # Step 6: Save overlay
    overlay = rgb_image.copy()
    overlay[final_mask==1] = [255,0,0]
    blend = cv2.addWeighted(rgb_image, 0.6, overlay, 0.4, 0)
    cv2.imwrite(OUTPUT_OVERLAY, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print("Saved overlay:", OUTPUT_OVERLAY)

if __name__ == "__main__":
    main()
