import os
import ants
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from shapely.geometry import shape
from shapely import speedups
from skimage.draw import polygon
from tifffile import imwrite

# Speed optimization for shapely polygons
if speedups.available:
    speedups.enable()

# ---------------------------
# Paths
# ---------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "HE+MALDI")
anno_dir = os.path.join(os.path.dirname(__file__), "..", "data", "annotations")
he_dir = os.path.join(base_dir, "lowres_he")
maldi_dir = os.path.join(base_dir, "maldi_mask")

output_dir_images = os.path.join(os.path.dirname(__file__), "..", "results_ants_images")
output_dir_masks = os.path.join(os.path.dirname(__file__), "..", "results_masks")
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)

# ---------------------------
# Pair files by prefix
# ---------------------------
he_files = sorted([f for f in os.listdir(he_dir) if f.endswith(".tif")])
maldi_files = sorted([f for f in os.listdir(maldi_dir) if f.endswith(".tif")])
anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".geojson")])

paired_files = []
for he_file in he_files:
    prefix = he_file.split("_")[0]  # e.g., A1, B22
    match = next((m for m in maldi_files if m.split("_")[0] == prefix), None)
    anno_match = next((a for a in anno_files if a.split("_")[0] == prefix), None)

    if match and anno_match:
        paired_files.append((he_file, match, anno_match))
    else:
        print(f"⚠️ Missing match for prefix {prefix}")

print(f"✅ Found {len(paired_files)} valid triplets (H&E + MALDI + annotation)")

# ---------------------------
# Constants
# ---------------------------
DOWNSAMPLE_FACTOR = 128

# ---------------------------
# Process each pair
# ---------------------------
for idx, (he_file, maldi_file, anno_file) in enumerate(tqdm(paired_files, desc="Processing pairs")):
    # Load images
    he_img_orig = cv2.imread(os.path.join(he_dir, he_file), cv2.IMREAD_GRAYSCALE)
    maldi_img = cv2.imread(os.path.join(maldi_dir, maldi_file), cv2.IMREAD_GRAYSCALE)

    if he_img_orig is None or maldi_img is None:
        print(f"⚠️ Skipping due to missing image for prefix {he_file}")
        continue

    # Resize H&E → match MALDI resolution (do NOT resize MALDI)
    target_shape = (maldi_img.shape[1], maldi_img.shape[0])
    he_img_resized = cv2.resize(he_img_orig, target_shape).astype(np.float32)

    # Normalize [0,1]
    he_img_resized = (he_img_resized - he_img_resized.min()) / (he_img_resized.max() - he_img_resized.min() + 1e-8)
    maldi_img = (maldi_img - maldi_img.min()) / (maldi_img.max() - maldi_img.min() + 1e-8)

    # Convert to ANTs format
    fixed = ants.from_numpy(maldi_img)   # MALDI = Fixed
    moving = ants.from_numpy(he_img_resized)  # H&E = Moving

    # Nonlinear registration (SyN)
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")
    warped = reg["warpedmovout"].numpy()

    # ---------------------------
    # Generate binary mask from GeoJSON
    # ---------------------------
    anno_path = os.path.join(anno_dir, anno_file)
    if not os.path.exists(anno_path) or os.path.getsize(anno_path) == 0:
        print(f"⚠️ Skipping empty or missing annotation file: {anno_file}")
        continue

    try:
        with open(anno_path, "r") as f:
            anno_data = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON in {anno_file}, skipping.")
        continue

    mask = np.zeros_like(maldi_img, dtype=np.uint8)

    for feat in anno_data.get("features", []):
        geom = shape(feat["geometry"])
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            coords = np.array(poly.exterior.coords)

            # Scale from full-resolution coordinates → MALDI resolution
            coords[:, 0] = coords[:, 0] / DOWNSAMPLE_FACTOR * (maldi_img.shape[1] / he_img_orig.shape[1])
            coords[:, 1] = coords[:, 1] / DOWNSAMPLE_FACTOR * (maldi_img.shape[0] / he_img_orig.shape[0])

            rr, cc = polygon(coords[:, 1], coords[:, 0], mask.shape)
            mask[rr, cc] = 1

    # ---------------------------
    # Warp mask using registration transform
    # ---------------------------
    mask_ants = ants.from_numpy(mask.astype(np.float32))
    warped_mask = ants.apply_transforms(
        fixed=fixed,
        moving=mask_ants,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor"
    ).numpy()

    # ---------------------------
    # Visualization (6 panels)
    # ---------------------------
    warped_norm = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)
    overlay_on_he = np.dstack([warped_norm, warped_norm * 0.5 + warped_mask * 0.5, warped_norm * 0.5])
    overlay_on_maldi = np.dstack([maldi_img, maldi_img * 0.5 + warped_mask * 0.5, maldi_img * 0.5])

    plt.figure(figsize=(28, 4))
    titles = [
        "MALDI (Fixed)", "H&E (Resized)", "Warped H&E",
        "Binary Mask", "Mask on Warped H&E", "Mask on MALDI (Fixed)"
    ]
    images = [maldi_img, he_img_resized, warped_norm, mask, overlay_on_he, overlay_on_maldi]

    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, 6, i)
        plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_images, f"alignment_{idx}.png"), bbox_inches="tight")
    plt.close()

    # ---------------------------
    # Save binary warped mask
    # ---------------------------
    imwrite(os.path.join(output_dir_masks, f"mask_{idx}.tif"), (warped_mask * 255).astype(np.uint8))

print(f"✅ Done! Results saved in {output_dir_images} and {output_dir_masks}")
