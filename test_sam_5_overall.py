# batch_sam_pointbox_fixed.py
import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from segment_anything import sam_model_registry, SamPredictor

# --------------- USER CONFIG ---------------
INPUT_FOLDER = "data_set/dset-s2/tra_scene"   # folder with input TIFFs
OUTPUT_FOLDER = "batch_outputs_5"         # folder where outputs will be saved
SAM_CHECKPOINT = "sam_vit_b.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_REGION_AREA = 500        # candidate filter (pixels)
PAD = 12                     # pad around candidate boxes for SAM crop
MORPH_KERNEL = (5, 5)
FINAL_CLOSE_KERNEL = (9, 9)
SMALL_ISLAND_MIN = 100       # remove tiny islands after final mask

# parameters for restricting SAM expansion
DILATE_ALLOWED_KSIZE = (7, 7)   # how much SAM can expand beyond coarse mask
MIN_IOU_ACCEPT = 0.06           # if IoU < this, use safer fallback
INTERSECT_MIN_RATIO = 0.2       # accept intersection if it covers >=20% of coarse area
# -------------------------------------------

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
    i = index.copy()
    i[np.isfinite(i) == False] = np.nanmin(i)
    norm = (i - np.nanmin(i))
    denom = (np.nanmax(i) - np.nanmin(i)) + 1e-9
    idx8 = (255.0 * (norm / denom)).astype(np.uint8)
    thr, _ = cv2.threshold(idx8, 0, 255, cv2.THRESH_OTSU)
    return (idx8 > thr).astype(np.uint8), thr

def area_filter_boxes(mask, min_area=MIN_REGION_AREA):
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
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

def iou_mask(a, b):
    a_bool = (a > 0); b_bool = (b > 0)
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def polyize_and_save(mask_uint8, ds, out_geojson):
    transform = ds.transform
    geoms = []
    for geom, val in shapes(mask_uint8, mask=mask_uint8, transform=transform):
        if val == 1:
            g = shape(geom)
            if g.is_valid and g.area > 0:
                geoms.append(g)
    if not geoms:
        return None
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=ds.crs)
    # try add area in m2 if metric CRS
    try:
        if ds.crs and not ds.crs.is_geographic:
            gdf["area_m2"] = gdf.geometry.area
            gdf["area_ha"] = gdf["area_m2"] / 10000.0
        else:
            gdf["area_m2"] = None
            gdf["area_ha"] = None
    except Exception:
        pass
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf

def quick_cloud_mask(R, G, B, bright_th=230, whiteness_diff=35):
    # simple RGB cloud heuristic (fast & dirty)
    rgb = normalize_rgb(np.stack([R, G, B], axis=-1)).astype(np.int32)
    brightness = rgb.mean(axis=2)
    diff_rg = np.abs(rgb[:,:,0] - rgb[:,:,1])
    diff_rb = np.abs(rgb[:,:,0] - rgb[:,:,2])
    whiteness = (diff_rg < whiteness_diff) & (diff_rb < whiteness_diff)
    cloud_mask = (brightness > bright_th) & whiteness
    return cloud_mask.astype(np.uint8)

def process_one_tif(tif_path, sam, predictor, out_folder):
    base = os.path.splitext(os.path.basename(tif_path))[0]
    out_mask_tif = os.path.join(out_folder, f"{base}_mask.tif")
    out_mask_png = os.path.join(out_folder, f"{base}_mask.png")
    out_overlay = os.path.join(out_folder, f"{base}_overlay.png")
    out_geojson = os.path.join(out_folder, f"{base}_polygons.geojson")

    print(f"\n--- Processing: {base} ---")
    ds = rasterio.open(tif_path)
    H, W = ds.height, ds.width
    bands = ds.count

    # read bands
    R = ds.read(1); G = ds.read(2); B = ds.read(3)

    # pick index
    if bands >= 6:
        SWIR = ds.read(6)
        index = compute_mndwi(G, SWIR); used_idx = "MNDWI"
    elif bands >= 4:
        NIR = ds.read(4)
        index = compute_ndwi(G, NIR); used_idx = "NDWI"
    else:
        index = (B.astype(float) - R.astype(float)) / (B.astype(float) + R.astype(float) + 1e-6)
        used_idx = "B-R fallback"

    print(f"Index used: {used_idx}  shape: {index.shape}")

    rgb_preview = normalize_rgb(np.stack([R, G, B], axis=-1))

    # coarse candidate via Otsu
    candidate_mask, thr = otsu_thresh_index(index)
    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    candidate_mask = cv2.morphologyEx(candidate_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    # remove cloud pixels from candidate mask
    cloud_mask = quick_cloud_mask(R, G, B, bright_th=230, whiteness_diff=35)
    if cloud_mask.sum() > 0:
        candidate_mask[cloud_mask == 1] = 0
        print(f"Cloud pixels removed: {int(cloud_mask.sum())}")

    boxes = area_filter_boxes(candidate_mask, min_area=MIN_REGION_AREA)
    print("Candidates:", len(boxes))

    final_mask = np.zeros((H, W), dtype=np.uint8)

    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        x0p = max(0, x0 - PAD); y0p = max(0, y0 - PAD)
        x1p = min(W, x1 + PAD); y1p = min(H, y1 + PAD)
        w = x1p - x0p; h = y1p - y0p
        if w <= 4 or h <= 4:
            continue

        crop_rgb = rgb_preview[y0p:y1p, x0p:x1p]
        crop_coarse = candidate_mask[y0p:y1p, x0p:x1p].astype(np.uint8)
        comp_mask = (crop_coarse > 0).astype(np.uint8)

        # centroid or fallback center
        if comp_mask.sum() == 0:
            cx, cy = w//2, h//2
        else:
            c = centroid_from_mask(comp_mask)
            cx, cy = (c if c is not None else (w//2, h//2))

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
        except Exception:
            # fallback to box-only predict
            try:
                masks, scores, logits = predictor.predict(box=np.array([0,0,w,h]), multimask_output=True)
            except Exception:
                print(f"Skipping candidate {idx} due to SAM failure.")
                continue

        # choose best mask using IoU with coarse component
        best_mask = None; best_iou = -1.0
        for mi in range(masks.shape[0]):
            m = masks[mi].astype(np.uint8)
            if m.shape != comp_mask.shape:
                m = cv2.resize(m, (comp_mask.shape[1], comp_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            iouv = iou_mask(m, comp_mask)
            if iouv > best_iou:
                best_iou = iouv; best_mask = m

        if best_mask is None:
            continue

        # safer fallback: if IoU is tiny, prefer intersection or coarse mask instead of full SAM mask
        if best_iou < MIN_IOU_ACCEPT:
            intersect = np.logical_and(best_mask, comp_mask).astype(np.uint8)
            if comp_mask.sum() > 0 and intersect.sum() >= max(10, int(INTERSECT_MIN_RATIO * comp_mask.sum())):
                chosen = intersect
            else:
                chosen = comp_mask  # safer fallback
        else:
            chosen = best_mask

        # restrict SAM expansion: allow only small dilation of coarse mask
        if comp_mask.sum() > 0:
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATE_ALLOWED_KSIZE)
            allowed_region = cv2.dilate(comp_mask, dilate_k, iterations=1)
            final_crop_mask = np.logical_and(chosen, allowed_region).astype(np.uint8)
            # if restricting removes too much, fallback to chosen ∩ coarse or coarse
            if final_crop_mask.sum() < max(5, int(0.05 * comp_mask.sum())):
                # try intersection
                final_crop_mask = np.logical_and(chosen, comp_mask).astype(np.uint8)
                if final_crop_mask.sum() < 5:
                    final_crop_mask = comp_mask
        else:
            # no coarse signal, be conservative: only accept SAM mask if it's reasonably small (not full crop)
            if best_mask.sum() >= 0.95 * (w * h):
                # likely full-box -> skip or accept small region
                final_crop_mask = np.zeros_like(best_mask)
            else:
                final_crop_mask = best_mask

        # paste into global mask
        final_mask[y0p:y1p, x0p:x1p] = np.logical_or(final_mask[y0p:y1p, x0p:x1p], final_crop_mask).astype(np.uint8)
        print(f"Candidate {idx}/{len(boxes)} processed  iou={best_iou:.3f}  paste_px={int(final_crop_mask.sum())}")

    # final post-processing
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FINAL_CLOSE_KERNEL))

    # remove tiny islands
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8)
    cleaned = np.zeros_like(final_mask)
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= max(1, MIN_REGION_AREA // 4, SMALL_ISLAND_MIN):
            cleaned[ labels == i ] = 1
    final_mask = cleaned

    # save PNG mask and overlay only
    out_png = (final_mask * 255).astype(np.uint8)
    # cv2.imwrite(out_mask_png, out_png)
    # print("Saved PNG:", out_mask_png)

    # # write geotiff mask
    # profile = ds.profile.copy()
    # profile.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
    # with rasterio.open(out_mask_tif, "w", **profile) as dst:
    #     dst.write(out_png, 1)
    # print("Saved TIF:", out_mask_tif)

    # overlay visualization
    overlay = rgb_preview.copy()
    overlay[final_mask == 1] = [255, 0, 0]
    blend = cv2.addWeighted(rgb_preview, 0.6, overlay, 0.4, 0)
    cv2.imwrite(out_overlay, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print("Saved OVERLAY:", out_overlay)

    # polygons
    try:
        gdf = polyize_and_save(out_png, ds, out_geojson)
        if gdf is not None:
            print("Saved GeoJSON:", out_geojson)
    except Exception as e:
        print("Poly save failed:", e)

    ds.close()
    print(f"Finished {base}")
    return True

def main_batch():
    if not os.path.isdir(INPUT_FOLDER):
        raise SystemExit(f"Input folder not found: {INPUT_FOLDER}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # load SAM once
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    tif_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".tif")]
    tif_files.sort()
    if len(tif_files) == 0:
        print("No tif files found in input folder.")
        return

    for fname in tif_files:
        tif_path = os.path.join(INPUT_FOLDER, fname)
        try:
            process_one_tif(tif_path, sam, predictor, OUTPUT_FOLDER)
        except Exception as e:
            print(f"Failed for {fname}: {e}")

if __name__ == "__main__":
    main_batch()
