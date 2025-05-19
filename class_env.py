# Imports
import os
import sys
import importlib
from scipy.spatial import KDTree

# Add the parent directory of the script to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import plotly.graph_objects as go
import geopy.distance
from skimage.draw import random_shapes
import random

from misc.helpful_geo_functions import create_grid
from scipy.spatial import cKDTree
import misc.create_height_map

class env:
    def __init__(self, height_data, measuring_area_scenarios, radiation_data, scenarios_geo_coords, test_env = False):
        self.test_env = test_env
        if self.test_env:
            self.scenario_number_for_measurement_scenario = 0
        else:
            self.scenario_number_for_measurement_scenario = np.random.randint(0, num_scenarios - 1)
        self.surface_point_cloud_ = height_data #np.load('height_data.npy')
        self.tree_surface_point_cloud_ = cKDTree(self.surface_point_cloud_[:, :2])
        self.measuring_area_scenarios = measuring_area_scenarios # np.load(measuring_area_scenarios_filename, allow_pickle=True)
        self.radiation_scenarios = radiation_data #np.load(scenarios_filename)
        self.geo_coordinates_of_radiation_grid_for_all_scenarios = scenarios_geo_coords # np.load(scenarios_filename + '_geo_coords' + '.npy' , allow_pickle=True)
        self.geo_coordinates_of_radiation_grid = self.geo_coordinates_of_radiation_grid_for_all_scenarios[self.scenario_number_for_measurement_scenario]
        self.tree_radiation_grid = cKDTree(self.geo_coordinates_of_radiation_grid.reshape(-1, 2))
        self.measuring_area_scenario = self.measuring_area_scenarios[self.scenario_number_for_measurement_scenario]
        self.anchor_point = anchor_point
        self.end_sim_reason = None
        self.distance_east_from_anchor_point = distance_east_from_anchor_point
        self.distance_north_from_anchor_point = distance_north_from_anchor_point
        # span_point goes distance_north meters north and distance_east meters east from the anchor point
        self.span_point = geopy.distance.distance(meters=distance_east_from_anchor_point).destination(geopy.distance.distance(meters=distance_north_from_anchor_point).destination(self.anchor_point, 0), 90)
        self.timestep = timestep
        self.start_time = start_time
        self.end_sim = False        
        self.time_abs = start_time
        self.time_rel = 0
        self.granularity = granularity
        self.grid_coordinates = create_grid(self.anchor_point, self.span_point, self.granularity)
        self.gridsize = np.array([self.grid_coordinates.shape[0], self.grid_coordinates.shape[1],max_grid_height])
        if surface_grid_creation_type == 'no_height_map':
            self.surface_grid = np.zeros(self.gridsize[:2], dtype=int) + 40
        self.source_position = np.array([np.random.randint(0, self.gridsize[0]), np.random.randint(0, self.gridsize[1])])
        self.radiation_grid = self.radiation_scenarios[self.scenario_number_for_measurement_scenario]
        self.percentage_of_episode_finished_by_time_passed = 0.0
        self.normalized_gridsize = self.gridsize/np.max(self.gridsize)
        self.total_amount_of_measurable_points = self.gridsize[0]*self.gridsize[1]
    def reset(self):
        if self.test_env:
            self.scenario_number_for_measurement_scenario += 1
            if self.scenario_number_for_measurement_scenario == num_scenarios:
                self.scenario_number_for_measurement_scenario = 0
        else:
            self.scenario_number_for_measurement_scenario = np.random.randint(0, num_scenarios - 1)
        self.geo_coordinates_of_radiation_grid = self.geo_coordinates_of_radiation_grid_for_all_scenarios[self.scenario_number_for_measurement_scenario]
        self.tree_radiation_grid = cKDTree(self.geo_coordinates_of_radiation_grid.reshape(-1, 2))
        self.measuring_area_scenario = self.measuring_area_scenarios[self.scenario_number_for_measurement_scenario]
        if surface_grid_creation_type == 'load_height_map':
            pass
        elif surface_grid_creation_type == 'perlin noise':
            self.surface_grid = self.generate_surface_grid_via_perlin_noise()
        self.end_sim_reason = None
        self.end_sim = False        
        self.time_abs = start_time
        self.time_rel = 0
        self.percentage_of_episode_finished_by_time_passed = 0.0
        self.radiation_grid = self.radiation_scenarios[self.scenario_number_for_measurement_scenario]
    def generate_surface_grid_200m_const_height(self):
        height_map = np.ones(self.gridsize[:2])*200
        return height_map
    def step(self):
        self.time_rel += self.timestep
        self.time_abs += self.timestep
        self.percentage_of_episode_finished_by_time_passed = self.time_rel / self.max_flight_time
    def create_heightmap(self):
            # Initialize heightmap
            heightmap = np.zeros(self.gridsize)

            # Generate Perlin noise
            scale = 400.0  # Adjust for different levels of detail
            octaves = 3
            persistence = 0.9
            lacunarity = 2.0
            max_height = 150  # Maximum height value

            # Use a random seed for each run
            base = random.randint(-1000, 0)

            # Create a grid of coordinates
            x = np.arange(self.gridsize[0])
            y = np.arange(self.gridsize[1])
            x, y = np.meshgrid(x, y)

            # Vectorized Perlin noise generation
            heightmap = np.vectorize(lambda i, j: pnoise2(i / scale, 
                                                        j / scale, 
                                                        octaves=octaves, 
                                                        persistence=persistence, 
                                                        lacunarity=lacunarity, 
                                                        repeatx=self.gridsize[0], 
                                                        repeaty=self.gridsize[1], 
                                                        base=base))(x, y)

            # Scale and clip the heightmap
            heightmap = np.clip((heightmap + 1) * (max_height / 2), 0, max_height)

            # Bring heightmap values to 0
            heightmap -= np.min(heightmap)

            return heightmap

    def get_state(self):
        """
        Returns a dict containing all state needed for observation.
        """
        return {
            "time_abs": self.time_abs,
            "surface_grid": np.copy(self.surface_grid),
            # Add more fields if needed by observation
        }

    def set_state(self, state):
        """
        Restores all state needed for observation.
        """
        self.time_abs = state["time_abs"]
        self.surface_grid = np.copy(state["surface_grid"])
        # Add more fields if needed