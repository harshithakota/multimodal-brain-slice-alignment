import os
import ants
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# Paths
# ---------------------------
base_dir = "data/HE+MALDI"
he_dir = os.path.join(base_dir, "lowres_he")
maldi_dir = os.path.join(base_dir, "maldi_mask")
output_dir = "results_ants_images"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Pair files by prefix
# ---------------------------
he_files = sorted([f for f in os.listdir(he_dir) if f.endswith(".tif")])
maldi_files = sorted([f for f in os.listdir(maldi_dir) if f.endswith(".tif")])

paired_files = []
for he_file in he_files:
    prefix = he_file.split("_")[0]  # match prefix like A1, B14
    match = next((m for m in maldi_files if m.startswith(prefix)), None)
    if match:
        paired_files.append((he_file, match))
    else:
        print(f"⚠️ No MALDI match found for {he_file}")

print(f"✅ Found {len(paired_files)} pairs.")

# ---------------------------
# Process each pair
# ---------------------------
for idx, (he_file, maldi_file) in enumerate(tqdm(paired_files, desc="Processing pairs")):
    # Load images in grayscale
    he_img = cv2.imread(os.path.join(he_dir, he_file), cv2.IMREAD_GRAYSCALE)
    maldi_img = cv2.imread(os.path.join(maldi_dir, maldi_file), cv2.IMREAD_GRAYSCALE)

    # Resize to same size
    resize_shape = (256, 256)
    he_img = cv2.resize(he_img, resize_shape).astype(np.float32)
    maldi_img = cv2.resize(maldi_img, resize_shape).astype(np.float32)

    # Normalize [0,1]
    he_img = (he_img - he_img.min()) / (he_img.max() - he_img.min() + 1e-8)
    maldi_img = (maldi_img - maldi_img.min()) / (maldi_img.max() - maldi_img.min() + 1e-8)

    # Convert to ANTs
    fixed = ants.from_numpy(maldi_img)   # MALDI (fixed)
    moving = ants.from_numpy(he_img)     # H&E (moving)

    # Run registration
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN"  # deformable registration
    )
    warped = reg["warpedmovout"].numpy()

    # Overlay (Red = warped H&E, Green = MALDI)
    # Normalize before overlay
    warped_norm = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)
    maldi_norm  = (maldi_img - maldi_img.min()) / (maldi_img.max() - maldi_img.min() + 1e-8)

    overlay_rgb = np.stack([warped_norm, maldi_norm, np.zeros_like(maldi_norm)], axis=-1)


    # ---------------------------
    # Plot and Save
    # ---------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(maldi_img, cmap="Greens")
    plt.title(f"MALDI (Fixed) #{idx}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(he_img, cmap="Reds")
    plt.title(f"H&E (Moving) #{idx}")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(warped, cmap="gray")
    plt.title("Warped H&E")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay_rgb)
    plt.title("Overlay (RGB)")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"alignment_{idx}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

print(f"✅ Done! Results saved in {output_dir}")
