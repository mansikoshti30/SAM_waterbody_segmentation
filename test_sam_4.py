"""
sam_water_fusion.py

Final combined water extraction pipeline:
 - NDWI / MNDWI (auto select) -> Otsu threshold -> coarse mask
 - SAM (SamPredictor) point+box refinement per candidate
 - Fuse coarse_mask OR sam_refined_mask
 - Post-process and save GeoTIFF + PNG + overlay + GeoJSON
"""

import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from segment_anything import sam_model_registry, SamPredictor

# ---------------- CONFIG ----------------
IMAGE_TIF = "test_data.tif"         # input TIFF/GeoTIFF
SAM_CHECKPOINT = "sam_vit_b.pth"    # sam checkpoint
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_REGION_AREA = 500       # min area (px) to keep candidate regions
PAD = 12                    # pad in px around candidate when cropping for SAM
MORPH_KERNEL = (5,5)        # cleaning kernel for candidate mask
FINAL_KERNEL_CLOSE = (9,9)  # final closing kernel
SMALL_ISLAND_MIN = 100      # remove islands smaller than this (px) after fusion

OUT_TIF = "binary_water_fused_4.tif"
OUT_PNG = "binary_water_fused_4.png"
OUT_OVERLAY = "overlay_water_fused_4.png"
OUT_GEOJSON = "water_polygons_fused_4.geojson"
# ----------------------------------------

def normalize_rgb(arr):
    arr = arr.astype(float)
    arr = (255.0 * (arr / (np.percentile(arr, 98) + 1e-9))).clip(0, 255)
    return arr.astype(np.uint8)

def compute_ndwi(G, NIR):
    g = G.astype(float); n = NIR.astype(float)
    return (g - n) / (g + n + 1e-6)

def compute_mndwi(G, SWIR):
    g = G.astype(float); s = SWIR.astype(float)
    return (g - s) / (g + s + 1e-6)

def otsu_threshold(index):
    # scale to 0..255 and Otsu
    i = index.copy()
    i[np.isfinite(i) == False] = np.nanmin(i)
    norm = (i - np.nanmin(i))
    denom = (np.nanmax(i) - np.nanmin(i)) + 1e-9
    i8 = (255.0 * (norm / denom)).astype(np.uint8)
    thr, _ = cv2.threshold(i8, 0, 255, cv2.THRESH_OTSU)
    print(f"Otsu threshold (8-bit): {thr}")
    return (i8 > thr).astype(np.uint8)

def component_boxes(mask, min_area=MIN_REGION_AREA):
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    boxes = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area: 
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))
    return boxes

def centroid_from_mask(mask):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(np.mean(xs)), int(np.mean(ys))
    cx = int(M["m10"] / (M["m00"] + 1e-9))
    cy = int(M["m01"] / (M["m00"] + 1e-9))
    return cx, cy

def polyize_mask(mask_uint8, dataset, out_geojson):
    transform = dataset.transform
    geoms = []
    for geom, val in shapes(mask_uint8, mask=mask_uint8, transform=transform):
        if val == 1:
            g = shape(geom)
            if g.is_valid and g.area > 0:
                geoms.append(g)
    if not geoms:
        print("No polygons produced.")
        return None
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=dataset.crs)
    # add area in m2 and hectares
    # pixel area:
    px_area = abs(dataset.transform.a * dataset.transform.e)
    gdf["area_m2"] = gdf.geometry.area  # shapely area in units of CRS (should be meters if CRS is meters)
    # if dataset.crs is geographic degrees then pixel area approach differs; warn user
    if dataset.crs is None or dataset.crs.is_geographic:
        # fallback compute from pixel_area * pixel_count
        print("Warning: CRS is geographic or missing. area_m2 column may be invalid. Consider reprojecting to metric CRS.")
    else:
        # OK
        pass
    gdf.to_file(out_geojson, driver="GeoJSON")
    print("Saved polygons:", out_geojson)
    return gdf

def main():
    assert os.path.exists(IMAGE_TIF), f"Input TIFF not found: {IMAGE_TIF}"
    assert os.path.exists(SAM_CHECKPOINT), f"SAM checkpoint not found: {SAM_CHECKPOINT}"

    ds = rasterio.open(IMAGE_TIF)
    H, W = ds.height, ds.width
    bands = ds.count
    print(f"Opened {IMAGE_TIF} (W={W}, H={H}, bands={bands}, crs={ds.crs})")

    # choose best index: MNDWI > NDWI > RGB fallback
    R = ds.read(1); G = ds.read(2); B = ds.read(3)
    if bands >= 6:
        print("Using MNDWI (Green - SWIR)")
        SWIR = ds.read(6)
        index = compute_mndwi(G, SWIR)
    elif bands >= 4:
        print("Using NDWI (Green - NIR)")
        NIR = ds.read(4)
        index = compute_ndwi(G, NIR)
    else:
        print("No NIR/SWIR found: using (B-R) / (B+R) fallback")
        index = (B.astype(float) - R.astype(float)) / (B.astype(float) + R.astype(float) + 1e-6)

    # preview RGB for SAM (uint8)
    rgb_preview = normalize_rgb(np.stack([R, G, B], axis=-1))

    # coarse: Otsu threshold on index
    coarse_mask = otsu_threshold(index)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    coarse_mask = cv2.morphologyEx(coarse_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    coarse_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_CLOSE, kernel)

    # get candidate boxes from coarse mask
    boxes = component_boxes(coarse_mask, min_area=MIN_REGION_AREA)
    print("Candidate boxes from index:", len(boxes))

    # load SAM predictor
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    sam_mask = np.zeros((H,W), dtype=np.uint8)

    # iterate candidate boxes and refine using SAM (point + box)
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        # pad
        x0p = max(0, x0 - PAD); y0p = max(0, y0 - PAD)
        x1p = min(W, x1 + PAD); y1p = min(H, y1 + PAD)
        w = x1p - x0p; h = y1p - y0p
        if w <= 4 or h <= 4: 
            continue

        crop_rgb = rgb_preview[y0p:y1p, x0p:x1p]
        crop_coarse = coarse_mask[y0p:y1p, x0p:x1p].astype(np.uint8)

        if crop_coarse.sum() == 0:
            # fallback point center
            cx, cy = w//2, h//2
        else:
            # centroid relative to crop
            centroid = centroid_from_mask(crop_coarse)
            if centroid is None:
                cx, cy = w//2, h//2
            else:
                cx, cy = centroid

        predictor.set_image(crop_rgb)
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])

        try:
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.array([0,0,w,h]),
                multimask_output=True
            )
        except Exception as e:
            # fallback to box-only predict
            print(f"SAM predict failed for candidate {i}, fallback to box-only: {e}")
            try:
                masks, scores, logits = predictor.predict(box=np.array([0,0,w,h]), multimask_output=True)
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                continue

        # pick best mask by IoU with crop_coarse (if crop_coarse has signal)
        best_mask = None
        best_iou = -1.0
        for mi in range(masks.shape[0]):
            m = masks[mi].astype(np.uint8)
            # ensure mask shape matches crop
            if m.shape != crop_coarse.shape:
                # SamPredictor should return crop-sized masks when set_image was crop
                # but guard anyway: resize if mismatch
                m = cv2.resize(m.astype(np.uint8), (crop_coarse.shape[1], crop_coarse.shape[0]), interpolation=cv2.INTER_NEAREST)
            inter = np.logical_and(m>0, crop_coarse>0).sum()
            union = np.logical_or(m>0, crop_coarse>0).sum()
            iou_val = float(inter) / union if union>0 else 0.0
            if iou_val > best_iou:
                best_iou = iou_val
                best_mask = m

        # if IoU is tiny, pick highest SAM score
        if best_mask is None:
            continue
        if best_iou < 0.03 and len(scores)>0:
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(np.uint8)

        # paste into global sam_mask
        sam_mask[y0p:y1p, x0p:x1p] = np.logical_or(sam_mask[y0p:y1p, x0p:x1p], best_mask).astype(np.uint8)
        print(f"Refined candidate {i}/{len(boxes)}  bbox=({x0p},{y0p},{x1p},{y1p}) iou={best_iou:.3f}")

    # Fusion: OR coarse and SAM masks
    fused = np.logical_or(coarse_mask>0, sam_mask>0).astype(np.uint8)

    # final cleanup: morphological close then remove tiny islands
    fused = cv2.morphologyEx(fused.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FINAL_KERNEL_CLOSE))
    # remove tiny islands
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(fused, 8)
    cleaned = np.zeros_like(fused)
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= SMALL_ISLAND_MIN:
            cleaned[ labels == i ] = 1
    fused = cleaned

    # write PNG
    out_png = (fused * 255).astype(np.uint8)
    cv2.imwrite(OUT_PNG, out_png)
    print("Saved PNG:", OUT_PNG)

    # write GeoTIFF (preserve profile)
    profile = ds.profile.copy()
    profile.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(out_png, 1)
    print("Saved GeoTIFF:", OUT_TIF)

    # overlay visualization
    overlay = rgb_preview.copy()
    overlay[fused==1] = [255,0,0]
    blend = cv2.addWeighted(rgb_preview, 0.6, overlay, 0.4, 0)
    cv2.imwrite(OUT_OVERLAY, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print("Saved overlay:", OUT_OVERLAY)

    # polygonize and save geojson and compute area (if CRS metric)
    try:
        gdf = polyize_mask(out_png, ds, OUT_GEOJSON)
        if gdf is not None:
            # attempt area in m2 if CRS metric
            try:
                # ensure metric CRS
                if ds.crs and not ds.crs.is_geographic:
                    gdf["area_m2"] = gdf.geometry.area
                    gdf["area_ha"] = gdf["area_m2"] / 10000.0
                else:
                    gdf["area_m2"] = None
                    gdf["area_ha"] = None
                gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
                print("Saved GeoJSON with area attributes:", OUT_GEOJSON)
            except Exception as e:
                print("Failed to add area attributes:", e)
    except Exception as e:
        print("Polygonize failed:", e)

    # final stats
    pix_count = int(fused.sum())
    pixel_area = abs(ds.transform.a * ds.transform.e)  # map units per pixel (may be deg^2 if geographic)
    total_area = pix_count * pixel_area
    print(f"Pixels labelled water: {pix_count}")
    print(f"Pixel area (map units^2): {pixel_area:.6f}; total water area (map units^2): {total_area:.3f}")

    ds.close()
    print("Done.")

if __name__ == "__main__":
    main()
