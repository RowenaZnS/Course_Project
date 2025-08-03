import os
import random
import rasterio
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile

# Path to the directory containing GeoTIFF files
PathToGeoTiffFile = "Potsdam-GeoTif"

# Output folder for train-test split
output_folder = "Potsdam-dataset"
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Step 1: Read all GeoTIFF file paths
all_files = [os.path.join(PathToGeoTiffFile, f) for f in os.listdir(PathToGeoTiffFile) if f.endswith(".tif")]

# Step 2: Extract geographic bounding box (latitude and longitude) for each GeoTIFF file
geo_info = []  # To store file paths and their bounding boxes
for file in all_files:
    with rasterio.open(file) as src:
        bounds = src.bounds  # Get the geographic bounds (left, bottom, right, top)
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)  # Create a shapely box
        centroid = bbox.centroid  # Get the centroid of the bounding box
        geo_info.append({
            "file": file,
            "centroid": (centroid.x, centroid.y)  # (longitude, latitude)
        })

# Step 3: Create a chessboard-style split based on latitude and longitude
# Define grid size for the chessboard
grid_size = 8  # Divide the area into a 4x4 grid (you can adjust this)

# Get the min and max longitude/latitude
min_lon = min(info["centroid"][0] for info in geo_info)
max_lon = max(info["centroid"][0] for info in geo_info)
min_lat = min(info["centroid"][1] for info in geo_info)
max_lat = max(info["centroid"][1] for info in geo_info)

# Calculate grid cell size
lon_step = (max_lon - min_lon) / grid_size
lat_step = (max_lat - min_lat) / grid_size

# Assign each file to train or test based on the grid cell it falls into
train_files = []
test_files = []

for info in geo_info:
    lon, lat = info["centroid"]
    # Calculate grid cell indices
    lon_idx = int((lon - min_lon) / lon_step)
    lat_idx = int((lat - min_lat) / lat_step)
    # Chessboard pattern: Use (lon_idx + lat_idx) % 2 to alternate between train and test
    if (lon_idx + lat_idx) % 2 == 0:
        train_files.append(info["file"])
    else:
        test_files.append(info["file"])

# Step 4: Copy files into train and test folders
for file in train_files:
    dst = os.path.join(train_folder, os.path.basename(file))
    copyfile(file, dst)

for file in test_files:
    dst = os.path.join(test_folder, os.path.basename(file))
    copyfile(file, dst)

print(f"Train dataset prepared: {len(train_files)} files")
print(f"Test dataset prepared: {len(test_files)} files")

# Step 5: Visualize train and test split
train_coords = [(info["centroid"][0], info["centroid"][1]) for info in geo_info if info["file"] in train_files]
test_coords = [(info["centroid"][0], info["centroid"][1]) for info in geo_info if info["file"] in test_files]

train_coords = np.array(train_coords)
test_coords = np.array(test_coords)

plt.figure(figsize=(10, 10))
plt.scatter(train_coords[:, 0], train_coords[:, 1], c="grey", label="Train", alpha=0.5)
plt.scatter(test_coords[:, 0], test_coords[:, 1], c="black", label="Test", alpha=0.5)

# Draw grid lines for the chessboard
for i in range(grid_size + 1):
    # Longitude grid lines
    plt.axvline(min_lon + i * lon_step, color="black", linestyle="--", linewidth=0.5)
    # Latitude grid lines
    plt.axhline(min_lat + i * lat_step, color="black", linestyle="--", linewidth=0.5)

plt.title("Train-Test Split")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.show()