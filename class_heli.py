# Imports
import os
import sys
import importlib
import matplotlib.pyplot as plt
from shapely.prepared import prep
import copy
# Add the parent directory of the script to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import cv2
import importlib
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
import numpy as np
from skimage.measure import block_reduce
import dill
import uuid
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from matplotlib.path import Path
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import geopandas as gpd
import os
from misc.helpful_geo_functions import compute_geo_coordinate_from_position, create_grid_on_earth
from pyproj import Geod
from scipy.ndimage import binary_erosion
import pandas as pd
import random
from geopy.distance import distance
from geopy.point import Point
from shapely.geometry import Point as ShapelyPoint, Polygon
from shapely.geometry.polygon import orient
from scipy.interpolate import interp1d
from shapely.ops import transform
from functools import partial
import pyproj

class heli:
    def __init__(self, env = None):
        self.distance_to_move_per_step = timestep * speed_in_meters_per_second
        self.geod = Geod(ellps='WGS84')
        self.lookup_table_for_cone_energy_2615 = dict({1: 8.7,
            10: 58.2,
            15: 77.8,
            20: 94.9,
            25: 110.1,
            30: 124.1,
            35: 137.1,
            40: 149.3,
            45: 160.8,
            50: 171.6,
            55: 182.0,
            60: 192.0,
            65: 201.6,
            70: 210.8,
            75: 219.8,
            80: 228.4,
            85: 236.8,
            90: 245.0,
            95: 253.0,
            100: 260.70000000000005,
            105: 268.3,
            110: 275.70000000000005,
            115: 283.00000000000006,
            120: 290.1,
            125: 297.00000000000006,
            130: 303.90000000000003,
            135: 310.6,
            140: 317.20000000000005,
            145: 323.70000000000005,
            150: 330.00000000000006,
            155: 336.30000000000007,
            160: 342.50000000000006,
            165: 348.6,
            170: 354.6,
            175: 360.50000000000006,
            180: 366.30000000000007,
            185: 372.1,
            190: 377.80000000000007,
            195: 383.4,
            200: 388.9,
            205: 394.4,
            210: 399.80000000000007,
            215: 405.2000000000001,
            220: 410.50000000000006,
            225: 415.7000000000001,
            230: 420.9,
            235: 426.00000000000006,
            240: 431.1,
            245: 436.1,
            250: 441.1,
            255: 446.00000000000006,
            260: 450.9,
            265: 455.80000000000007,
            270: 460.6,
            275: 465.30000000000007,
            280: 470.00000000000006,
            285: 474.7000000000001,
            290: 479.30000000000007,
            295: 483.9,
            300: 488.50000000000006,
            305: 493.00000000000006,
            310: 497.50000000000006,
            315: 502.00000000000006,
            320: 506.4,
            325: 510.80000000000007,
            330: 515.1,
            335: 519.4000000000001,
            340: 523.7,
            345: 528.0,
            350: 532.2})
        self.angles_for_cone = np.radians([0, 40, 80, 120, 160, 200, 240, 280, 320])
        # Interpolate radius values for each meter above ground
        heights = np.array(list(self.lookup_table_for_cone_energy_2615.keys()))
        radii = np.array(list(self.lookup_table_for_cone_energy_2615.values()))
        interpolation_function = interp1d(heights, radii, kind='linear', fill_value='extrapolate')
        self.lookup_table_for_cone_energy_2615_for_each_m = interpolation_function(np.arange(1, 350))
        self.current_bearing = 0
        self.env = env
        self.first_step = True
        self.observed_radiation_map_points = np.empty((0, 3))
        self.polygon_center = self.env.measuring_area_scenario['polygon_center']
        self.target_area = self.env.measuring_area_scenario['target_area']
        area, border_length = self.geod.geometry_area_perimeter(self.target_area)
        self.target_area_size = abs(area)
        self.target_area_size_start_of_episode = self.target_area_size
        self.target_area_border_length = abs(border_length)
        random_point_on_boundary = self.target_area.boundary.interpolate(random.uniform(0, self.target_area.boundary.length))
        self.env.anchor_point = np.array([random_point_on_boundary.x, random_point_on_boundary.y])
        distances, indices = self.env.tree_surface_point_cloud_.query(np.array([self.env.anchor_point]))
        if surface_grid_creation_type == 'no_height_map':
            self.height_of_surface_grid_at_start_position = 0
        else:
            self.height_of_surface_grid_at_start_position = self.env.surface_point_cloud_[indices[0]][2]
        self.position_as_geo_coordinate = np.array([self.polygon_center[0] - 1, self.polygon_center[1]])
        self.update_position(self.env.anchor_point)
        self.action_type = action_type
        self.heli_id = str(uuid.uuid4())[:4]
        self.x_range = np.arange(0, self.env.gridsize[0])
        self.y_range = np.arange(0, self.env.gridsize[1])
        self.x, self.y = np.meshgrid(self.x_range, self.y_range, indexing='ij')
        self.observed_radiation_map = np.zeros((self.env.gridsize[0], self.env.gridsize[1]))
        self.last_cone_bool_inverted = np.zeros((self.env.gridsize[0], self.env.gridsize[1]), dtype=bool)
        self.current_radiation_exposure = 0
        self.cumulative_radiation_exposure = 0
        self.cumulative_radiation_exposure_history = np.array([])
        self.cones = []
        self.measured_areas = []
        self.cone_polygon = Polygon()
        self.update_radiation_exposure_history_target_area()
        self.new_measured_points_in_between_actions = np.zeros((self.env.gridsize[0], self.env.gridsize[1]), dtype=bool)
        self.remeasured_points = np.zeros((self.env.gridsize[0], self.env.gridsize[1]), dtype=bool)

        self.current_direction_via_dot_product = np.array([0, 0])
        self.calculate_current_direction_via_dot_product()
        self.angle_between_current_direction_and_direction_to_polygon_center = self.calculate_angle(self.position_as_geo_coordinate, self.polygon_center, self.current_direction_via_dot_product)
    def reset(self):
        self.observed_radiation_map_points = np.empty((0, 3))
        self.polygon_center = self.env.measuring_area_scenario['polygon_center']
        self.target_area = self.env.measuring_area_scenario['target_area']
        area, border_length = self.geod.geometry_area_perimeter(self.target_area)
        self.target_area_size = abs(area)
        self.target_area_size_start_of_episode = self.target_area_size
        self.target_area_border_length = abs(border_length)
        self.env.anchor_point = self.target_area.boundary.interpolate(random.uniform(0, self.target_area.boundary.length))
        random_point_on_boundary = self.target_area.boundary.interpolate(random.uniform(0, self.target_area.boundary.length))
        self.env.anchor_point = np.array([random_point_on_boundary.x, random_point_on_boundary.y])
        distances, indices = self.env.tree_surface_point_cloud_.query(np.array([self.env.anchor_point]))
        if surface_grid_creation_type == 'no_height_map':
            self.height_of_surface_grid_at_start_position = 0
        else:
            self.height_of_surface_grid_at_start_position = self.env.surface_point_cloud_[indices[0]][2]
        self.update_position(self.env.anchor_point)
        self.observed_radiation_map = np.zeros((self.env.gridsize[0], self.env.gridsize[1]))
        self.last_cone_bool_inverted = np.zeros((self.env.gridsize[0], self.env.gridsize[1]), dtype=bool)
        self.current_radiation_exposure = 0
        self.cumulative_radiation_exposure = 0
        self.cumulative_radiation_exposure_history = np.array([])
        self.cones = []
        self.measured_areas = []
        self.cone_polygon = Polygon()
        self.update_radiation_exposure_history_target_area()
        self.new_measured_points_in_between_actions = np.zeros((self.env.gridsize[0], self.env.gridsize[1]), dtype=bool)
        self.remeasured_points = np.zeros((self.env.gridsize[0], self.env.gridsize[1]), dtype=bool)
        self.calculate_current_direction_via_dot_product()
        self.angle_between_current_direction_and_direction_to_polygon_center = self.calculate_angle(self.position_as_geo_coordinate, self.polygon_center, self.current_direction_via_dot_product)
    def calculate_angle(self, A, B, current_direction):
        if (current_direction != np.array([0, 0])).any():
            # Extract longitude and latitude values
            lon_A, lat_A = A
            lon_B, lat_B = B[1], B[0]  # B is a Shapely Point

            # Create vectors
            vector_AB = np.array([lon_B - lon_A, lat_B - lat_A])
            vector_current = current_direction

            # Calculate the cosine of the angle
            cosine_angle = np.dot(vector_AB, vector_current) / (np.linalg.norm(vector_AB) * np.linalg.norm(vector_current))
            
            return np.clip(cosine_angle, -1, 1)
        else:
            return 0
    def find_closest_point(self,point, grid):
        distance, index = self.env.tree.query(point)
        i, j = np.unravel_index(index, grid.shape[:2])
        return np.array([i, j])
    def calculate_odl_via_precalculated_grids(self):
        index_of_closest_radiation_grid_coordinate = self.env.tree_radiation_grid.query(self.position_as_geo_coordinate)[1]
        i, j = np.unravel_index(index_of_closest_radiation_grid_coordinate, self.env.radiation_grid.shape)
        odl = self.env.radiation_grid[i, j]
        return odl  # in mSv/h
    def random_number_between_x_y_with_close_to_y_more_likely(self, x, y):
        # Generate a uniform random number in [0, 1]
        u = np.random.uniform(0, 0.03)
        # Transform it using the inverse CDF
        z = ((u * (y**3 - x**3)) + x**3)**(1/3)
        return z

    def idw_interpolation(self,points, values, target, p=2):
        distances = np.linalg.norm(points - target, axis=1)
        weights = 1 / (distances ** p)
        return np.sum(weights * values) / np.sum(weights)
    def update_radiation_exposure_history_target_area(self, odl_from_live_data = None, live_mode = False):
        if live_mode:
            self.current_radiation_exposure = odl_from_live_data
        else:
            if how_to_calculate_odl == 'via simulated radiation grid':
                self.current_radiation_exposure = self.calculate_odl_via_precalculated_grids()
            elif how_to_calculate_odl == 'random_test_values':
                self.current_radiation_exposure = np.random.uniform(0, 1)
        if 50 <= self.position_z_above_ground <= 200:
            radius = self.lookup_table_for_cone_energy_2615_for_each_m[np.ceil(self.position_z_above_ground).astype(int)]
            # Calculate the points
            lon, lat = self.position_as_geo_coordinate
            lat_rad = np.radians(lat)
            delta_lon = radius * np.cos(self.angles_for_cone + self.current_bearing) / (111320 * np.cos(lat_rad))  # Adjusted conversion for longitude
            delta_lat = radius * np.sin(self.angles_for_cone + self.current_bearing) / 110540  # Conversion for latitude remains the same
            points = np.column_stack((lon + delta_lon, lat + delta_lat))
            # Create the 
            self.last_cone_polygon = self.cone_polygon
            self.cone_polygon = Polygon(points)
            # Compute the convex hull of last_cone_polygon and cone_polygon
            self.cone_union_last_cone_convex_hull = self.cone_polygon.union(self.last_cone_polygon).convex_hull
        else:
            self.cone_polygon = Polygon()
        self.observed_radiation_map_points = np.vstack((self.observed_radiation_map_points, np.array([[self.position_as_geo_coordinate[0], self.position_as_geo_coordinate[1], self.current_radiation_exposure]])))
        # Calculate the percentage of new area measured that is not already measured.
        cone_area, _ = self.geod.geometry_area_perimeter(self.cone_union_last_cone_convex_hull)
        self.cone_polygon_size = abs(cone_area)
        # If the heli is outside of the grid in any direction no points are measured
        self.target_area_last_step_border_length = self.target_area_border_length
        self.target_area_last_step_size = self.target_area_size
        self.target_area = self.target_area.difference(self.cone_union_last_cone_convex_hull)
        # Adjust target_area_border_length calculation to use geographic coordinates
        area, border_length = self.geod.geometry_area_perimeter(self.target_area)
        self.target_area_size = abs(area)
        self.target_area_border_length = abs(border_length)
        self.length_new_measured_minus_length_old_measured = self.target_area_last_step_border_length - self.target_area_border_length
        
        self.percentage_of_new_area_measured_ = (
            (self.target_area_last_step_size - self.target_area_size)
            / self.cone_polygon_size
            if self.cone_polygon_size != 0 else 0
        )
        self.percentage_of_target_area_left = self.target_area_size/self.target_area_size_start_of_episode
    def create_border(self, array, border_width):
        array = array.astype(bool)
        structure = np.ones((3, 3), dtype=bool)
        dilated_array = binary_dilation(array, structure=structure, iterations=border_width)
        border = dilated_array & ~array
        return border    


    def get_static_point_90m_above_surface(self):
        while True:
            # Select random x, y, z coordinates within the grid size
            x = 0
            y = 0
            z = self.env.surface_grid[x, y] + 90
            return np.array([x, y, z])
    def get_random_2d_point_in_one_of_the_4_corners(self):
        # Define the possible corner coordinates adjusted 5 grid steps towards the center
        grid_steps = 8
        corners = [
            (grid_steps, grid_steps),
            (grid_steps, self.env.surface_grid.shape[1] - grid_steps -1),
            (self.env.surface_grid.shape[0] - grid_steps - 1, grid_steps),
            (self.env.surface_grid.shape[0] - grid_steps - 1, self.env.surface_grid.shape[1] - grid_steps - 1 )
        ]
        
        # Randomly select one of the corners
        x, y = corners[np.random.randint(0, 4)]

        # Choose z to be the maximum-min height divided by 2
        return np.array([x, y, 348])
    def get_random_point_above_surface(self):
        while True:
            # Select random x, y, z coordinates within the grid size
            x = np.random.randint(0, self.env.gridsize[0])
            y = np.random.randint(0, self.env.gridsize[1])
            z = np.random.randint(0, self.env.gridsize[2])

            # Check if the z coordinate is above the surface
            if z > self.env.surface_grid[x, y] and z < self.env.gridsize[2]:
                return np.array([x, y, z])    

    def update_position(self, position_as_geo_coordinate):
        self.last_position_as_geo_coordinate = self.position_as_geo_coordinate
        self.position_as_geo_coordinate = position_as_geo_coordinate
        if surface_grid_creation_type == 'no_height_map':
            self.position_z_above_ground = 90
        else:
            distances, indices = self.env.tree_surface_point_cloud_.query(np.array([self.position_as_geo_coordinate]))
            self.height_of_surface_grid_at_position = self.env.surface_point_cloud_[indices[0]][2]
            print("Height change not implemented. define self.position_z_above_ground")
    def rotate_vector_around_axis(self, axis, theta, vector):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(np.pi - theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rotated_vector_not_adjusted_to_graniularity = np.dot(    np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
                    , vector)
        return np.array([rotated_vector_not_adjusted_to_graniularity[0],rotated_vector_not_adjusted_to_graniularity[1],rotated_vector_not_adjusted_to_graniularity[2]*self.env.granularity])
    def rotate_vector_around_z_axis_and_set_height_angle(self, vector, horizontal_angle, vertical_change_in_meters_per_timestep):

        # Convert degrees to radians
        horizontal_angle_rad = np.radians(horizontal_angle)

        vector[2] = vertical_change_in_meters_per_timestep
        
        # Rotation matrix around the z-axis
        Rz = np.array([
            [np.cos(horizontal_angle_rad), -np.sin(horizontal_angle_rad), 0],
            [np.sin(horizontal_angle_rad), np.cos(horizontal_angle_rad), 0],
            [0, 0, 1]
        ])
        
        # Apply the z-axis rotation
        rotated_vector = Rz @ vector
        return rotated_vector
    def calculate_current_direction_via_dot_product(self):
        self.current_direction_via_dot_product_for_test = np.array(self.position_as_geo_coordinate) - np.array(self.last_position_as_geo_coordinate)
        if (self.current_direction_via_dot_product_for_test != np.array([0, 0])).any():
            self.current_direction_via_dot_product = self.current_direction_via_dot_product_for_test / np.linalg.norm(self.current_direction_via_dot_product_for_test)
    def step(self, action_vector):
        self.current_bearing = (self.current_bearing + action_vector[0]) % 360
        destination = distance(meters=self.distance_to_move_per_step).destination(Point(self.position_as_geo_coordinate[1], self.position_as_geo_coordinate[0]), bearing=self.current_bearing)
        self.update_position(np.array([destination.longitude, destination.latitude]))
        self.angle_between_current_direction_and_direction_to_polygon_center = self.calculate_angle(self.position_as_geo_coordinate, self.polygon_center, self.current_direction_via_dot_product)
        self.update_radiation_exposure_history_target_area()
    def get_state(self):
        """
        Returns a dict containing all state needed for observation.
        """
        return {
            "position_as_geo_coordinate": np.copy(self.position_as_geo_coordinate),
            "position_z_above_ground": self.position_z_above_ground,
            "current_bearing": self.current_bearing,
            "observed_radiation_map_points": np.copy(self.observed_radiation_map_points),
            "angle_between_current_direction_and_direction_to_polygon_center": self.angle_between_current_direction_and_direction_to_polygon_center,
            "target_area": copy.deepcopy(self.target_area),  # deep copy for shapely Polygon
            "polygon_center": self.polygon_center,
        }

    def set_state(self, state):
        """
        Restores all state needed for observation.
        """
        self.position_as_geo_coordinate = np.copy(state["position_as_geo_coordinate"])
        self.position_z_above_ground = state["position_z_above_ground"]
        self.current_bearing = state["current_bearing"]
        self.observed_radiation_map_points = np.copy(state["observed_radiation_map_points"])
        self.angle_between_current_direction_and_direction_to_polygon_center = state["angle_between_current_direction_and_direction_to_polygon_center"]
        self.target_area = state["target_area"]
        self.polygon_center = state["polygon_center"]