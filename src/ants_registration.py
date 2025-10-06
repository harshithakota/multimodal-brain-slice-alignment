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
    match = next((m for m in maldi_files if m.startswith(prefix)), None)
    anno_match = next((a for a in anno_files if a.startswith(prefix)), None)
    if match and anno_match:
        paired_files.append((he_file, match, anno_match))
    else:
        print(f"⚠️ Missing match for prefix {prefix}")

print(f"✅ Found {len(paired_files)} pairs.")

# ---------------------------
# Constants
# ---------------------------
DOWNSAMPLE_FACTOR = 128
RESIZE_SHAPE = (256, 256)

# ---------------------------
# Process each pair
# ---------------------------
for idx, (he_file, maldi_file, anno_file) in enumerate(tqdm(paired_files, desc="Processing pairs")):
    # Load images
    he_img_orig = cv2.imread(os.path.join(he_dir, he_file), cv2.IMREAD_GRAYSCALE)
    maldi_img_orig = cv2.imread(os.path.join(maldi_dir, maldi_file), cv2.IMREAD_GRAYSCALE)

    # print(f"{he_file} original size: {he_img_orig.shape}, {maldi_file} size: {maldi_img_orig.shape}")

    # Resize both to a common shape
    he_img = cv2.resize(he_img_orig, RESIZE_SHAPE).astype(np.float32)
    maldi_img = cv2.resize(maldi_img_orig, RESIZE_SHAPE).astype(np.float32)

    # Normalize [0,1]
    he_img = (he_img - he_img.min()) / (he_img.max() - he_img.min() + 1e-8)
    maldi_img = (maldi_img - maldi_img.min()) / (maldi_img.max() - maldi_img.min() + 1e-8)

    # Convert to ANTs format
    fixed = ants.from_numpy(maldi_img)   # MALDI → Fixed
    moving = ants.from_numpy(he_img)     # H&E → Moving

    # Nonlinear registration (SyN)
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")
    warped = reg["warpedmovout"].numpy()

    # ---------------------------
    # Generate binary mask from GeoJSON annotations
    # ---------------------------
    anno_path = os.path.join(anno_dir, anno_file)
    with open(anno_path, "r") as f:
        anno_data = json.load(f)

    mask = np.zeros(RESIZE_SHAPE, dtype=np.uint8)

    for feat in anno_data["features"]:
        geom = shape(feat["geometry"])
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            coords = np.array(poly.exterior.coords)

            # Scale from full-resolution coordinates to low-res resized coordinates
            coords[:, 0] = (coords[:, 0] / DOWNSAMPLE_FACTOR) * (RESIZE_SHAPE[0] / he_img_orig.shape[1])
            coords[:, 1] = (coords[:, 1] / DOWNSAMPLE_FACTOR) * (RESIZE_SHAPE[1] / he_img_orig.shape[0])

            rr, cc = polygon(coords[:, 1], coords[:, 0], mask.shape)
            mask[rr, cc] = 1

    # ---------------------------
    # Warp the mask using registration transform
    # ---------------------------
    mask_ants = ants.from_numpy(mask.astype(np.float32))
    warped_mask = ants.apply_transforms(
        fixed=fixed,
        moving=mask_ants,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor"
    ).numpy()

    # ---------------------------
    # Prepare overlays
    # ---------------------------
    warped_norm = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)
    maldi_norm = (maldi_img - maldi_img.min()) / (maldi_img.max() - maldi_img.min() + 1e-8)

    # Overlay mask on warped H&E
    overlay_on_he = np.dstack([
        warped_norm, 
        warped_norm * 0.5 + warped_mask * 0.5,
        warped_norm * 0.5
    ])

    # Overlay mask on MALDI
    overlay_on_maldi = np.dstack([
        maldi_norm,
        maldi_norm * 0.5 + warped_mask * 0.5,
        maldi_norm * 0.5
    ])

    # Aligned overlay (H&E vs MALDI)
    aligned_overlay = np.dstack([
        warped_norm, 
        maldi_norm, 
        np.zeros_like(warped_norm)
    ])

    # ---------------------------
    # Visualization
    # ---------------------------
    plt.figure(figsize=(24, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(maldi_norm, cmap="Greens")
    plt.title("MALDI (Fixed)")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.imshow(he_img, cmap="Reds")
    plt.title("H&E (Moving)")
    plt.axis("off")

    plt.subplot(1, 6, 3)
    plt.imshow(warped_norm, cmap="gray")
    plt.title("Warped H&E")
    plt.axis("off")

    plt.subplot(1, 6, 4)
    plt.imshow(mask, cmap="gray")
    plt.title("Binary Mask")
    plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.imshow(overlay_on_he)
    plt.title("Mask on Warped H&E")
    plt.axis("off")

    plt.subplot(1, 6, 6)
    plt.imshow(overlay_on_maldi)
    plt.title("Mask on MALDI (Fixed)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_images, f"alignment_{idx}.png"), bbox_inches="tight")
    plt.close()

    # Save binary warped mask (both .npy and .tif)
    # np.save(os.path.join(output_dir_masks, f"mask_{idx}.npy"), warped_mask)
    imwrite(os.path.join(output_dir_masks, f"mask_{idx}.tif"), (warped_mask * 255).astype(np.uint8))

print(f"Done! Results saved in {output_dir_images} and {output_dir_masks}")
