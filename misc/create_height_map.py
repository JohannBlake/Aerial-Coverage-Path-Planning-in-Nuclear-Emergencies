import os
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from get_parameters import *

# Load API key from environment
API_KEY = "bba26e533a8ad6b2bf5cf83e98716462"  # os.getenv("OPENTOPOGRAPHY_TOKEN")
if not API_KEY:
    raise ValueError("Environment variable OPENTOPOGRAPHY_TOKEN not set.")

# center coordinates
anchor_point_lon, anchor_point_lat = anchor_point
half_side_deg = 1  # ~100 km

# Define bounding box
south = anchor_point_lat - half_side_deg
north = anchor_point_lat + half_side_deg
west = anchor_point_lon - half_side_deg
east = anchor_point_lon + half_side_deg

# Define file paths
base_dir = os.getcwd()
downloads_dir = os.path.join(base_dir, 'misc', 'geo_data', 'downloads_from_opentopography')
os.makedirs(downloads_dir, exist_ok=True)
filename = f"aw3d30_{south:.10f}_{north:.10f}_{west:.10f}_{east:.10f}.tif"
file_path = os.path.join(downloads_dir, filename)
npy_file_path = os.path.join(base_dir, 'height_data.npy')

# Check if the GeoTIFF file already exists
if not os.path.exists(file_path):
    # Request data from OpenTopography
    params = {
        "demtype": "AW3D30",
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": API_KEY
    }

    response = requests.get("https://portal.opentopography.org/API/globaldem", params=params)

    if response.ok:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded heightmap from OpenTopography")
    else:
        raise RuntimeError(f"❌ Error {response.status_code}: {response.text}")
else:
    print(f"Skip download from OpenTopography (Already exists).")

# Check if the .npy file already exists
if not os.path.exists(npy_file_path):
    # Open the GeoTIFF file
    with rasterio.open(file_path) as dataset:
        # Read the height data
        height_data = dataset.read(1)

        # Get the coordinates of the pixels
        rows, cols = np.indices(height_data.shape)
        xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)

        # Flatten the arrays
        lons = np.array(xs).flatten()
        lats = np.array(ys).flatten()
        heights = height_data.flatten()

        # Combine into an array of tuples
        data = np.array(list(zip(lons, lats, heights)))

    # Save the numpy array to file
    np.save(npy_file_path, data)
else:
    print(f"Skip heightmap creation (Already exists).")