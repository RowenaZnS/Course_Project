import os
import random
from sklearn.model_selection import KFold
from shutil import copyfile

# Path to the directory containing GeoTIFF files
PathToGeoTiffFile = "Potsdam-GeoTif"

# Output folder for 5-fold splits
output_folder = "Potsdam-5Fold-Splits"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Read all GeoTIFF file paths
all_files = [os.path.join(PathToGeoTiffFile, f) for f in os.listdir(PathToGeoTiffFile) if f.endswith(".tif")]

# Step 2: Randomly sample at least 10,000 images
random.seed(42)  # Ensure reproducibility
sampled_files = random.sample(all_files, 10000)  # Extract 10,000 samples

# Step 3: Split into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (_, fold_idx) in enumerate(kf.split(sampled_files)):
    fold_folder = os.path.join(output_folder, f"fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)

    # Copy files for this fold
    for idx in fold_idx:
        src = sampled_files[idx]
        dst = os.path.join(fold_folder, os.path.basename(src))
        copyfile(src, dst)

    print(f"Fold {fold} prepared: {len(fold_idx)} files")

print(f"5-Fold splits saved in: {output_folder}")