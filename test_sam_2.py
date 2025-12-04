import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from segment_anything import sam_model_registry, SamPredictor

# ---------------- CONFIG ----------------
IMAGE_TIF = "test_data.tif"
SAM_CHECKPOINT = "sam_vit_b.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_REGION_AREA = 300      # good default
PAD = 10                   # expand candidate box for SAM
MORPH_KERNEL = (7, 7)      # for cleaning mask

OUTPUT_TIF = "binary_water_mask_2.tif"
OUTPUT_PNG = "binary_water_mask_2.png"
OUTPUT_OVERLAY = "overlay_water_2.png"
# ----------------------------------------

def normalize_rgb(arr):
    arr = (255 * (arr / np.percentile(arr, 98))).clip(0, 255)
    return arr.astype(np.uint8)

def compute_ndwi(G, NIR):
    g = G.astype(float); n = NIR.astype(float)
    return (g - n) / (g + n + 1e-6)

def compute_mndwi(G, SWIR):
    g = G.astype(float); s = SWIR.astype(float)
    return (g - s) / (g + s + 1e-6)

def best_water_index(ds):
    bands = ds.count
    R = ds.read(1)
    G = ds.read(2)
    B = ds.read(3)

    has_NIR = (bands >= 4)
    has_SWIR = (bands >= 6)

    if has_SWIR:
        print("Using **MNDWI** (best accuracy) …")
        SWIR = ds.read(6)  # assuming band-6 is SWIR1
        index = compute_mndwi(G, SWIR)
    elif has_NIR:
        print("Using NDWI (Green - NIR) …")
        NIR = ds.read(4)
        index = compute_ndwi(G, NIR)
    else:
        print("No NIR/SWIR available → fallback RGB water detection …")
        index = (B.astype(float) - R.astype(float)) / (B + R + 1e-6)

    return index, normalize_rgb(np.stack([R, G, B], axis=-1))

def otsu_threshold(index):
    index_norm = ((index - index.min()) / (index.max() - index.min() + 1e-6))
    index_8 = (index_norm * 255).astype(np.uint8)
    thr, _ = cv2.threshold(index_8, 0, 255, cv2.THRESH_OTSU)
    print(f"Otsu threshold = {thr}")
    return (index_8 > thr).astype(np.uint8)

def expand_box(box, pad, W, H):
    x0,y0,x1,y1 = box
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(W-1, x1 + pad),
        min(H-1, y1 + pad)
    ]

def main():
    assert os.path.exists(IMAGE_TIF), "TIFF file missing"
    assert os.path.exists(SAM_CHECKPOINT), "SAM checkpoint missing"

    ds = rasterio.open(IMAGE_TIF)
    H, W = ds.height, ds.width

    # 1. WATER INDEX (NDWI / MNDWI / RGB fallback)
    index, rgb_image = best_water_index(ds)

    # 2. ADAPTIVE THRESHOLD (Otsu)
    candidate_mask = otsu_threshold(index)

    # 3. MORPH CLEAN
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    # 4. CONNECTED REGIONS
    nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, 8)
    boxes = []
    for i in range(1, nlab):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_REGION_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]

        boxes.append([x, y, x+w_box, y+h_box])

    print(f"Water candidates found: {len(boxes)}")

    # 5. LOAD SAM
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # FINAL MASK
    final_mask = np.zeros((H, W), dtype=np.uint8)

    # 6. SAM REFINEMENT WITH box + point prompt
    for box in boxes:
        bx = expand_box(box, PAD, W, H)

        # crop region for sharper SAM
        x0,y0,x1,y1 = bx
        crop = rgb_image[y0:y1, x0:x1]

        predictor.set_image(crop)

        # point prompt (center of candidate box)
        cx = (x1 - x0)//2
        cy = (y1 - y0)//2
        pt = np.array([[cx, cy]])

        masks, scores, logits = predictor.predict(
            point_coords=pt,
            point_labels=np.array([1]),
            box=np.array([0,0,x1-x0,y1-y0]),
            multimask_output=True
        )

        best = np.argmax(scores)
        mask = masks[best].astype(np.uint8)

        final_mask[y0:y1, x0:x1] |= mask

    # 7. POST-PROCESSING
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # 8. SAVE PNG
    binary_png = (final_mask * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_PNG, binary_png)
    print("Saved PNG:", OUTPUT_PNG)

    # 9. SAVE GEOTIFF
    profile = ds.profile.copy()
    profile.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw"
    })

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(binary_png, 1)

    print("Saved GeoTIFF:", OUTPUT_TIF)

    # 10. SAVE OVERLAY
    overlay = rgb_image.copy()
    overlay[final_mask == 1] = [255, 0, 0]
    blend = cv2.addWeighted(rgb_image, 0.6, overlay, 0.4, 0)
    cv2.imwrite(OUTPUT_OVERLAY, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print("Saved Overlay:", OUTPUT_OVERLAY)


if __name__ == "__main__":
    main()
