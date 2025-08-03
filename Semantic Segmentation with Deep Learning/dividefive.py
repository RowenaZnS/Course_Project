import os
import rasterio
from rasterio.transform import xy
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile

# Path to the directory containing GeoTIFF files
PathToGeoTiffFile = "Potsdam-GeoTif"

# Output folder for 5-fold split
output_folder = "Potsdam-folds-step4"
os.makedirs(output_folder, exist_ok=True)

# Create folders for each fold (fold1 to fold5)
fold_folders = [os.path.join(output_folder, f"fold{i + 1}") for i in range(5)]
for folder in fold_folders:
    os.makedirs(folder, exist_ok=True)

# Step 1: Read all GeoTIFF file paths
all_files = [os.path.join(PathToGeoTiffFile, f) for f in os.listdir(PathToGeoTiffFile) if f.endswith(".tif")]

# Step 2: Extract geographic centroid (longitude and latitude) for each GeoTIFF file
geo_info = []  # To store file paths and their geographic centroids
for file in all_files:
    with rasterio.open(file) as src:
        # Get the raster dimensions
        width, height = src.width, src.height

        # Calculate the center pixel
        center_x = width // 2  # Center column
        center_y = height // 2  # Center row

        # Use rasterio's xy function to get the geographic coordinates of the center pixel
        lon, lat = xy(src.transform, center_y, center_x)

        geo_info.append({
            "file": file,
            "centroid": (lon, lat)  # (longitude, latitude)
        })

# Step 3: Create a chessboard-style split with 5 colors based on latitude and longitude
# Define grid size for the chessboard
grid_size = 8  # Divide the area into an 8x8 grid (you can adjust this)

# Get the min and max longitude/latitude
min_lon = min(info["centroid"][0] for info in geo_info)
max_lon = max(info["centroid"][0] for info in geo_info)
min_lat = min(info["centroid"][1] for info in geo_info)
max_lat = max(info["centroid"][1] for info in geo_info)

# Calculate grid cell size
lon_step = (max_lon - min_lon) / grid_size
lat_step = (max_lat - min_lat) / grid_size

# Assign each file to one of the 5 folds based on the grid cell it falls into
fold_files = [[] for _ in range(5)]  # List of lists to hold file paths for each fold

for info in geo_info:
    lon, lat = info["centroid"]
    # Calculate grid cell indices
    lon_idx = int((lon - min_lon) / lon_step)
    lat_idx = int((lat - min_lat) / lat_step)
    # Chessboard pattern with 5 colors: Use (lon_idx + lat_idx) % 5 to assign to a fold
    fold_index = (lon_idx + lat_idx) % 5
    fold_files[fold_index].append(info["file"])

# Step 4: Copy files into respective fold folders
for i, files in enumerate(fold_files):
    fold_folder = fold_folders[i]
    for file in files:
        dst = os.path.join(fold_folder, os.path.basename(file))
        copyfile(file, dst)

print(f"5-fold dataset prepared:")
for i, files in enumerate(fold_files):
    print(f"Fold {i + 1}: {len(files)} files")

# Step 5: Visualize the 5-fold split
colors = ["red", "blue", "green", "yellow", "purple"]  # Colors for visualization
plt.figure(figsize=(10, 10))

for i, files in enumerate(fold_files):
    coords = [(info["centroid"][0], info["centroid"][1]) for info in geo_info if info["file"] in files]
    coords = np.array(coords)
    plt.scatter(coords[:, 0], coords[:, 1], c=colors[i], label=f"Fold {i + 1}", alpha=0.5)

# Draw grid lines for the chessboard
for i in range(grid_size + 1):
    # Longitude grid lines
    plt.axvline(min_lon + i * lon_step, color="black", linestyle="--", linewidth=0.5)
    # Latitude grid lines
    plt.axhline(min_lat + i * lat_step, color="black", linestyle="--", linewidth=0.5)

plt.title("5-Fold Split")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.show()