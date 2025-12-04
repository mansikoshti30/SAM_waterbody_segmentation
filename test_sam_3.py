# sam_water_tiff_pointbox.py
import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape, mapping
import geopandas as gpd
from segment_anything import sam_model_registry, SamPredictor

# --------------- CONFIG ---------------
IMAGE_TIF = "test_data.tif"        # path to your input TIFF/GeoTIFF
SAM_CHECKPOINT = "sam_vit_b.pth"   # path to SAM checkpoint
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_REGION_AREA = 500              # ignore tiny candidates (pixels)
PAD = 12                           # pad around boxes before cropping
MORPH_KERNEL = (5, 5)              # morphological kernel for cleaning
FINAL_CLOSE_KERNEL = (9, 9)        # final post-processing closing kernel

OUT_TIF = "binary_water_mask_3.tif"
OUT_PNG = "binary_water_mask_3.png"
OUT_OVERLAY = "overlay_water_3.png"
OUT_GEOJSON = "water_polygons_3.geojson"
# --------------------------------------

def normalize_rgb(arr):
    a = arr.astype(float)
    a = (255.0 * (a / (np.percentile(a, 98) + 1e-9))).clip(0, 255)
    return a.astype(np.uint8)

def compute_ndwi(G, NIR):
    g = G.astype(float); n = NIR.astype(float)
    return (g - n) / (g + n + 1e-6)

def compute_mndwi(G, SWIR):
    g = G.astype(float); s = SWIR.astype(float)
    return (g - s) / (g + s + 1e-6)

def otsu_thresh_index(index):
    # map index to 0..255 and apply Otsu
    norm = (index - np.nanmin(index))
    denom = (np.nanmax(index) - np.nanmin(index)) + 1e-9
    idx8 = (255.0 * (norm / denom)).astype(np.uint8)
    thr, _ = cv2.threshold(idx8, 0, 255, cv2.THRESH_OTSU)
    print(f"Otsu threshold (8-bit) = {thr}")
    return (idx8 > thr).astype(np.uint8)

def area_filter_and_boxes(mask, min_area=MIN_REGION_AREA):
    nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))
    return boxes

def compute_centroid(mask):
    # mask is binary (uint8) for a single component
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        # fallback to bounding-box center
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        cx = int(np.mean(xs)); cy = int(np.mean(ys))
    else:
        cx = int(M["m10"] / (M["m00"] + 1e-9))
        cy = int(M["m01"] / (M["m00"] + 1e-9))
    return cx, cy

def iou_mask(a, b):
    a_bool = (a > 0)
    b_bool = (b > 0)
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def polyize_and_save(mask_arr, ds, out_geojson):
    # mask_arr: 0/255 uint8
    transform = ds.transform
    geoms = []
    for geom, val in shapes(mask_arr.astype(np.uint8), mask=mask_arr.astype(np.uint8), transform=transform):
        if val == 1:
            shap = shape(geom)
            if shap.is_valid and shap.area > 0:
                geoms.append(shap)
    if len(geoms) == 0:
        print("No polygons to save.")
        return
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=ds.crs)
    gdf.to_file(out_geojson, driver="GeoJSON")
    print("Saved polygons:", out_geojson)

def main():
    assert os.path.exists(IMAGE_TIF), f"Input TIFF not found: {IMAGE_TIF}"
    assert os.path.exists(SAM_CHECKPOINT), f"SAM checkpoint not found: {SAM_CHECKPOINT}"

    ds = rasterio.open(IMAGE_TIF)
    H, W = ds.height, ds.width
    bands = ds.count
    print(f"Opened {IMAGE_TIF}: width={W}, height={H}, bands={bands}, crs={ds.crs}")

    # pick best water index (MNDWI -> NDWI -> RGB fallback)
    R = ds.read(1)
    G = ds.read(2)
    B = ds.read(3)
    if bands >= 6:
        print("Using MNDWI (Green - SWIR).")
        SWIR = ds.read(6)  # sentinel style; change if your SWIR is at different band index
        index = compute_mndwi(G, SWIR)
    elif bands >= 4:
        print("Using NDWI (Green - NIR).")
        NIR = ds.read(4)
        index = compute_ndwi(G, NIR)
    else:
        print("No NIR/SWIR found: using RGB blue-red difference fallback.")
        index = (B.astype(float) - R.astype(float)) / (B.astype(float) + R.astype(float) + 1e-6)

    # normalize RGB preview for SAM (uint8)
    rgb_preview = normalize_rgb(np.stack([R, G, B], axis=-1))

    # threshold with Otsu
    candidate_mask = otsu_thresh_index(index)

    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    candidate_mask = cv2.morphologyEx(candidate_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    # connected components -> boxes
    boxes = area_filter_and_boxes(candidate_mask, MIN_REGION_AREA)
    print("Candidate boxes found:", len(boxes))

    # prepare SAM
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    final_mask = np.zeros((H, W), dtype=np.uint8)

    # iterate candidates
    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        # pad box
        x0p = max(0, x0 - PAD)
        y0p = max(0, y0 - PAD)
        x1p = min(W, x1 + PAD)
        y1p = min(H, y1 + PAD)

        w = x1p - x0p
        h = y1p - y0p
        if w <= 4 or h <= 4:
            continue

        # crop rgb preview and candidate mask for this component
        crop_rgb = rgb_preview[y0p:y1p, x0p:x1p]
        comp_crop = candidate_mask[y0p:y1p, x0p:x1p].astype(np.uint8)

        # compute point inside the detected component (centroid of the component area)
        # we compute centroid on the component crop (only pixels >0)
        comp_only_mask = (comp_crop > 0).astype(np.uint8)
        if comp_only_mask.sum() == 0:
            # fallback center
            cx, cy = w // 2, h // 2
        else:
            centroid = compute_centroid(comp_only_mask)
            if centroid is None:
                cx, cy = w // 2, h // 2
            else:
                cx, cy = centroid

        # tell predictor about the crop
        predictor.set_image(crop_rgb)

        # create point and box in crop coordinates
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])  # 1 => foreground
        box_for_predict = np.array([0, 0, w, h])  # box covering full crop

        # predict multimask
        try:
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_for_predict,
                multimask_output=True
            )
        except Exception as e:
            print(f"SAM predict failed for box {idx} ({e}), trying box-only fallback.")
            try:
                masks, scores, logits = predictor.predict(box=box_for_predict, multimask_output=True)
            except Exception:
                continue

        # pick best mask by IoU with the candidate crop
        best_iou = -1.0
        best_mask = None
        for mi in range(masks.shape[0]):
            m = masks[mi].astype(np.uint8)
            iou_val = iou_mask(m, comp_only_mask)
            if iou_val > best_iou:
                best_iou = iou_val
                best_mask = m

        # if IoU is very low, optionally fallback to highest SAM score
        if best_mask is None:
            continue

        if best_iou < 0.05:
            # pick highest score mask instead
            best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
            best_mask = masks[best_idx].astype(np.uint8)
            best_iou = iou_mask(best_mask, comp_only_mask)

        # place mask into final_mask (global coords)
        final_mask[y0p:y1p, x0p:x1p] = np.logical_or(final_mask[y0p:y1p, x0p:x1p], best_mask).astype(np.uint8)

        print(f"Processed candidate {idx}/{len(boxes)}  box=({x0p},{y0p},{x1p},{y1p})  iou={best_iou:.3f}")

    # final post-processing: close small holes, optionally remove tiny islands
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FINAL_CLOSE_KERNEL))

    # remove tiny islands after final pass
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8)
    cleaned_mask = np.zeros_like(final_mask)
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= MIN_REGION_AREA // 4:  # allow smaller tolerance now
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cleaned_mask[ labels == i ] = 1
    final_mask = cleaned_mask

    # write PNG (white=water)
    out_png = (final_mask * 255).astype(np.uint8)
    cv2.imwrite(OUT_PNG, out_png)
    print("Saved PNG mask:", OUT_PNG)

    # write geotiff with same profile
    profile = ds.profile.copy()
    profile.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw"
    })
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(out_png, 1)
    print("Saved GeoTIFF mask:", OUT_TIF)

    # write overlay for quick visual check
    overlay = rgb_preview.copy()
    overlay[final_mask == 1] = [255, 0, 0]
    blend = cv2.addWeighted(rgb_preview, 0.6, overlay, 0.4, 0)
    cv2.imwrite(OUT_OVERLAY, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print("Saved overlay:", OUT_OVERLAY)

    # polygonize and save geojson
    try:
        polyize_and_save(out_png, ds, OUT_GEOJSON)
    except Exception as e:
        print("Failed to save GeoJSON:", e)

    ds.close()
    print("Done.")

if __name__ == "__main__":
    main()
