import os
import cv2
import numpy as np
import torch
import rasterio
from segment_anything import sam_model_registry, SamPredictor

# ---------------- PATH SETTINGS ----------------
INPUT_FOLDER = "data_set/dset-s2/tra_scene"
OUTPUT_FOLDER = "batch_overlays"
SAM_CHECKPOINT = "sam_vit_b.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NDWI_THRESHOLD = 0.0
RGB_BLUE_THRESHOLD = 30
MIN_REGION_AREA = 300
PAD = 10
# ------------------------------------------------


# ----------- SAME FUNCTIONS FROM OLD CODE -----------

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

        rgb = np.stack([R, G, B], axis=-1)
        rgb = (255 * (rgb / np.percentile(rgb, 98))).clip(0,255).astype(np.uint8)
        return cand, rgb

    else:
        print("No NIR band → using RGB blue-dominance.")
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

# ----------------------------------------------------

def process_single_file(tif_path, sam, predictor):
    print(f"\nProcessing: {tif_path}")

    ds = rasterio.open(tif_path)
    H, W = ds.height, ds.width

    # Step 1: NDWI candidate mask
    candidate_mask, rgb_image = get_candidate_mask(ds)
    candidate_mask = cv2.medianBlur(candidate_mask, 5)

    # Step 2: Connected components
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
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

    print(f"Found {len(boxes)} candidate water regions")

    # Step 3: SAM refine
    predictor.set_image(rgb_image)
    final_mask = np.zeros((H, W), dtype=np.uint8)

    for box in boxes:
        bx = expand_box(box, PAD, W, H)
        masks, scores, _ = predictor.predict(
            box=np.array(bx),
            multimask_output=True
        )
        best = np.argmax(scores)
        final_mask = np.logical_or(final_mask, masks[best]).astype(np.uint8)

    # Step 4: Create overlay
    overlay = rgb_image.copy()
    overlay[final_mask==1] = [255,0,0]
    blend = cv2.addWeighted(rgb_image, 0.6, overlay, 0.4, 0)

    # save output
    base_name = os.path.basename(tif_path).replace(".tif", "_overlay.png")
    out_path = os.path.join(OUTPUT_FOLDER, base_name)
    cv2.imwrite(out_path, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print("Saved overlay:", out_path)


def main_batch():
    # Prepare output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load SAM model once
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # Loop through all TIFFs
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(".tif"):
            tif_path = os.path.join(INPUT_FOLDER, file)
            process_single_file(tif_path, sam, predictor)


if __name__ == "__main__":
    main_batch()
