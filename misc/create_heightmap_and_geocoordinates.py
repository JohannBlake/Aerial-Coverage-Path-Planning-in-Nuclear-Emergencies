import os
import sys
import importlib

# Add the parent directory of the script to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import importlib

import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
import numpy as np
import geopy.distance
from misc.helpful_geo_functions import create_grid, compute_geo_coordinates_from_grid, geotiff_to_np_heightmap

span_point = geopy.distance.distance(meters=distance_east_from_anchor_point).destination(geopy.distance.distance(meters=distance_north_from_anchor_point).destination(anchor_point, 0), 90)
geocoordinates_array_helper_tbd = create_grid(anchor_point, span_point, granularity)
gridsize = geocoordinates_array_helper_tbd.shape

geo_coordinate_array = compute_geo_coordinates_from_grid(gridsize, anchor_point, granularity)

file_path_heightmap = os.path.join(".\misc\geo_data", "files_npy", f"{name_of_heightmap}.npy")
file_path_geocoordinates = os.path.join(".\misc\geo_data", "files_npy", f"{name_of_geocoordinates_array}.npy")

if os.name == 'posix':  # This is for Linux
    file_path_heightmap = file_path_heightmap.replace('\\', '/')  # Adjust for Linux path format
elif os.name == 'nt':  # This is for Windows
    file_path_heightmap = file_path_heightmap  # No change needed for Windows
if os.name == 'posix':  # This is for Linux
    file_path_geocoordinates = file_path_geocoordinates.replace('\\', '/')  # Adjust for Linux path format
elif os.name == 'nt':  # This is for Windows
    file_path_geocoordinates = file_path_geocoordinates  # No change needed for Windows

height_map = geotiff_to_np_heightmap(geo_coordinate_array,gridsize)
np.save(file_path_heightmap, height_map) 
np.save(file_path_geocoordinates, geo_coordinate_array) 