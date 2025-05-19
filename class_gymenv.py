import gymnasium as gym
from gymnasium import spaces
import numpy as np
import class_env
import class_heli
import class_sim
import importlib    
importlib.reload(class_env)
importlib.reload(class_heli)
importlib.reload(class_sim)
from class_env import env
from class_heli import heli
from class_sim import sim
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
import uuid
import time
from misc.helpful_geo_functions import create_grid_on_earth
from shapely import vectorized
from pyproj import Geod
import shapely
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import cKDTree
import cv2
from shapely.geometry import Polygon, Point
import numpy as np
from geopy.distance import distance
import random

#from omnisafe.envs.core import env_register

class GymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    def __init__(self, height_data, measuring_area_scenarios, radiation_data, scenarios_geo_coords, test_env = False, render_mode=None, camera_id=None,camera_name=None,width=None,height=None):
        self.geod = Geod(ellps="WGS84")
        super(GymnasiumEnv, self).__init__()
        self.count_radiation_exposure_exceeded = 0
        self.rew_little_action_per_episode = 0 
        self.rew_incremental_tv_per_episode = 0
        self.current_radiation_exposure_previous = 0
        self.channel_number = 3
        if include_distance_to_points:
            self.channel_number += 1
        self.rew_turning_angle_factor = 1
        self.rew_radiation = 0
        self.ter_reason_too_much_remeasured_count = 0
        self.env_sim = env(height_data, measuring_area_scenarios, radiation_data, scenarios_geo_coords, test_env = False)
        self.heli = heli(env=self.env_sim)
        self.sim = sim(env=self.env_sim, heli=self.heli)
        
        self.number_of_points_for_grids_on_earth = observed_image_size  # has to be odd
        self.list_of_grid_sizes = [2500, 8000, 40000]
        self.heli.position_as_geo_coordinate_init = self.heli.position_as_geo_coordinate
        self.grids_on_earth_for_observation = []  # Initialize an empty list to store grids
        self.grids_kdtrees = []  # Cache for KDTree objects for each grid
        if only_one_distorted_image_in_observation:
            grid = self.generate_distorted_grid()
            self.grids_on_earth_for_observation.append(grid)
            self.grids_kdtrees.append(cKDTree(grid.reshape(-1, 2)))
        else:
            for i, grid_size_in_m in enumerate(self.list_of_grid_sizes):
                # following function makes sim 2x slower
                grid = create_grid_on_earth(grid_size_in_m, self.number_of_points_for_grids_on_earth, self.heli.position_as_geo_coordinate, 0)
                self.grids_on_earth_for_observation.append(grid)
                self.grids_kdtrees.append(cKDTree(grid.reshape(-1, 2)))

        self.env_sim.max_flight_time = 999999 # will be adjusted automatically later by chekcking how much time the initial state needs minimally.
        self.ter_reason_cov_thresh_reached_count = 0
        self.ter_reason_oob_count = 0
        self.ter_reason_neg_rew_count = 0
        self.ter_reason_time_limit_count = 0
        self.ter_reason_manual_termination_count = 0
        self.last_time_html_logged = None
        self.last_time_html_saved = None
        self.gymenv_id = str(uuid.uuid4())[:4]
        self.make_deepcopy_of_gymenv_state_on_reset = False
        self.render_mode = render_mode
        self.reward = 0
        self.total_reward_per_episode = 0
        self.average_reward_per_episode = 0
        self.total_actions_per_episode = 0
        self.episode_count = 0
        self.info = {}
        self.info['data_at_end_of_preceding_episode'] = {  # logging will be displayed in wandb
            'total_reward': self.total_reward_per_episode,
            'mean_reward': self.total_reward_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
            'total_actions_per_episode': self.total_actions_per_episode,
            'count_radiation_exposure_exceeded': self.count_radiation_exposure_exceeded,
            'percentage_of_target_area_left': self.heli.percentage_of_target_area_left,
            'mean_rew_little_action_per_episode': self.rew_little_action_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
            'mean_incremental_tv_per_episode': self.rew_incremental_tv_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
        }
        self.info['cost'] = 0
        self.rewards = []
        self.max_measured_points_in_between_actions_at_90m = 1
        self.set_time_and_max_flight_time()
        if training_library == 'sb3':
            if action_type == 'ta':
                self.action_space = spaces.Box(low=np.array([-1, -1]), 
                                            high=np.array([1,  1]), 
                                            dtype=np.float64)
            elif action_type == 'ta_with_time_in_between_actions':
                self.action_space = spaces.Box(low=np.array([-1, -1, 0]), 
                                            high=np.array([1,  1, 1]), 
                                            dtype=np.float64)
            elif action_type == 'ta without height change':
                self.action_space = spaces.Box(low=np.array([-1]), 
                                            high=np.array([1]), 
                                            dtype=np.float64)
            elif action_type == 'ta without height change with termination option':
                self.action_space = spaces.Box(low=np.array([-1, -1]), 
                                            high=np.array([1,  1]), 
                                            dtype=np.float64)             
        elif training_library == 'omnisafe':
            pass # will be set in omnienv
        image_channels = 2
        image_size = observed_image_size
        self.image_observation_space = spaces.Box(low=0, high=255, shape=(image_size, image_size, image_channels), dtype=np.uint8)
        if surface_grid_creation_type != "no_height_map":
            self.scalar_observation_space = spaces.Box(
                                        low=np.concatenate([
                                            np.zeros(1),
                                            -1*np.ones(1),
                                        ]),
                                        high=np.concatenate([
                                            np.ones(1),
                                            np.ones(1),
                                        ]),
                                        dtype=np.float64
                                    )
        else:
            self.scalar_observation_space = spaces.Box(
                                        low=np.concatenate([
                                            -1*np.ones(1),
                                        ]),
                                        high=np.concatenate([
                                            np.ones(1),
                                        ]),
                                        dtype=np.float64
                                    )
        if only_one_distorted_image_in_observation:
            self.observation_space = spaces.Dict({
                "image": self.image_observation_space,
            })
        else:
            self.observation_space = spaces.Dict({
                "image_small": self.image_observation_space,
                "image_medium": self.image_observation_space,
                "image_large": self.image_observation_space,
            })
        self.difference_from_init_to_current_position_geo = np.array(self.heli.position_as_geo_coordinate) - np.array(self.heli.position_as_geo_coordinate_init)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.count_radiation_exposure_exceeded = 0
        self.rew_little_action_per_episode = 0
        self.rew_incremental_tv_per_episode = 0
        self.current_radiation_exposure_previous = 0
        self.rew_turning_angle_factor = 1
        self.rew_radiation = 0
        self.env_sim.reset()
        self.heli.reset()
        self.sim.reset()
        self.heli.position_as_geo_coordinate_init = self.heli.position_as_geo_coordinate
        self.grids_on_earth_for_observation = []  # Initialize an empty list to store grids
        self.grids_kdtrees = []  # Reset KDTree cache
        if only_one_distorted_image_in_observation:
            grid = self.generate_distorted_grid()
            self.grids_on_earth_for_observation.append(grid)
            self.grids_kdtrees.append(cKDTree(grid.reshape(-1, 2)))
        else:
            for i, grid_size_in_m in enumerate(self.list_of_grid_sizes):
                grid = create_grid_on_earth(grid_size_in_m, self.number_of_points_for_grids_on_earth, self.heli.position_as_geo_coordinate, 0)
                self.grids_on_earth_for_observation.append(grid)
                self.grids_kdtrees.append(cKDTree(grid.reshape(-1, 2)))
        obs = self._get_obs()
        self.observation = obs
        self.reward = 0
        self.reward_for_taking_or_ignoring_step = 0
        self.punishment_for_ignored_steps = 0
        self.punishment_for_ignored_steps = 0
        self.reward = 0
        self.total_reward_per_episode = 0
        self.average_reward_per_episode = 0
        self.total_actions_per_episode = 0
        self.rewards = []
        self.info['cost'] = 0
        self.set_time_and_max_flight_time()
        return self.observation, self.info
    def generate_distorted_grid(self, n=observed_image_size, range_m=18000, a=0.00008):
        """
        Generate a distorted grid with radial sine modulation centered around the helicopter's initial position.
        
        Parameters:
            n (int): Number of points along each dimension.
            range_m (float): Range for the grid in meters.
            a (float): Amplitude of the sine modulation.
        
        Returns:
            np.ndarray: Distorted grid of shape (n, n, 2) with latitude and longitude values.
        """
        center_lon, center_lat = self.heli.position_as_geo_coordinate  # Center of the grid

        # Create a 1D array of offset distances in meters
        distances = np.linspace(-range_m/2, range_m/2, n)

        # Compute points in the north-south direction with bearing 0 (positive distance = north, negative = south)
        # and east-west direction with bearing 90 (positive = east, negative = west).
        # Note: Geod.fwd expects (lon, lat, azimuth, distance)
        lat_points = np.array([self.geod.fwd(center_lon, center_lat, 0, d)[1] for d in distances])
        lon_points = np.array([self.geod.fwd(center_lon, center_lat, 90, d)[0] for d in distances])

        # Create 2D grid using meshgrid. Use 'ij' indexing so that rows correspond to latitude.
        lat_grid, lon_grid = np.meshgrid(lat_points, lon_points, indexing="ij")

        # Radial Sine Modulation function (vectorized)
        # For vectorization we first estimate the metric distance from the center for each grid point.
        # Here we use an approximate conversion: 1 degree lat ~ 111320 meters,
        # and for longitude the scale factor is cos(latitude).
        R = 6371000  # Earth's approximate radius in meters
        # Compute differences in radians
        dlat = np.deg2rad(lat_grid - center_lat)
        dlon = np.deg2rad(lon_grid - center_lon)
        # Approximate meters per radian in lat and lon (using center latitude for longitude)
        r = R * np.sqrt(dlat**2 + (np.cos(np.deg2rad(center_lat)) * dlon)**2)
        factor = np.sin(a * r)
        new_lat = center_lat + (lat_grid - center_lat) * factor
        new_lon = center_lon + (lon_grid - center_lon) * factor

        # Stack to form an array of shape (n, n, 2) with each element as a (lat, lon) pair
        grid = np.stack((new_lon,new_lat), axis=-1)
        return grid
    def rotate_coords(self, coords, origin_lon, origin_lat, angle):
        # Convert coords to a numpy array for vectorized computation: shape (N,2)
        coords_np = np.array(coords)
        lons = coords_np[:, 0]
        lats = coords_np[:, 1]
        
        # Compute azimuth and distance from origin to each point
        origin_lons = np.full_like(lons, origin_lon)
        origin_lats = np.full_like(lats, origin_lat)
        az, _, dist = self.geod.inv(origin_lons, origin_lats, lons, lats)
        
        # Add the rotation angle
        new_az = az + angle
        
        # Compute new coordinates using forward geodesic computation
        new_lon, new_lat, _ = self.geod.fwd(origin_lons, origin_lats, new_az, dist)
        
        return list(zip(new_lon, new_lat))

    def rotate_polygon_geodesic(self, polygon, angle, origin):
        """
        Rotate a shapely Polygon or MultiPolygon geodesically about a given origin using pyproj.Geod.

        Parameters:
            polygon (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The polygon to rotate.
            angle (float): Rotation angle in degrees.
            origin (shapely.geometry.Point): The center of rotation.

        Returns:
            shapely.geometry.Polygon or shapely.geometry.MultiPolygon: The rotated polygon.
        """
        origin_lon, origin_lat = origin.x, origin.y

        if polygon.geom_type == 'Polygon':
            return self._rotate_single_polygon(polygon, origin_lon, origin_lat, angle)
        elif polygon.geom_type == 'MultiPolygon':
            rotated_polygons = [self._rotate_single_polygon(poly, origin_lon, origin_lat, angle) for poly in polygon.geoms]
            return MultiPolygon(rotated_polygons)
        else:
            raise ValueError("Unsupported geometry type: {}".format(polygon.geom_type))

    def _rotate_single_polygon(self, poly, origin_lon, origin_lat, angle):
        exterior_rotated = self.rotate_coords(poly.exterior.coords, origin_lon, origin_lat, angle)
        interiors_rotated = [self.rotate_coords(interior.coords, origin_lon, origin_lat, angle) for interior in poly.interiors]
        return Polygon(exterior_rotated, interiors_rotated)
    def get_grid_from_polygons(self,grid_geo_coordinates, polygon):
         lon = grid_geo_coordinates[..., 0]
         lat = grid_geo_coordinates[..., 1]
         return vectorized.contains(polygon, lon, lat)
    def set_time_and_max_flight_time(self):
        self.env_sim.max_flight_time = max(timestep*3, episode_length_factor * self.env_sim.timestep * self.heli.target_area_size / area_measurable_per_step)
        # set time of episode to some time in the future depending on how much has been measured in the init measurement state and how much is radiated.
        self.env_sim.time_rel = 0.0
        self.env_sim.time_abs = self.env_sim.start_time
        self.env_sim.percentage_of_episode_finished_by_time_passed = 0.0
    def _get_obs(self):
        self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid = self.get_observation_as_images_()
        obs_images = []
        for i in range(len(self.boolean_maps_within_target_area)):
            obs_images.append(np.stack([
                (self.boolean_maps_within_target_area[i] * 255).astype(np.uint8),
                np.clip(
                    (255 * self.observed_radiation_mapped_to_grid[i] / (radiation_value_to_avoid * 1.5)),
                    0, 255
                ).astype(np.uint8)
            ], axis=-1))
        if surface_grid_creation_type != "no_height_map":
            pos_z = self.heli.position_z_above_ground / (self.env_sim.gridsize[2])
        if only_one_distorted_image_in_observation:
            obs = dict(image=obs_images[0])
        else:
            obs = dict(image_small=obs_images[0], image_medium=obs_images[1], image_large=obs_images[2])
        return obs
    def check_if_sim_needs_termination(self):
        if self.env_sim.end_sim_reason == 'Time limit reached':
            self.ter_reason_time_limit_count += 1
        elif self.env_sim.end_sim_reason == 'Manual termination':
            self.ter_reason_manual_termination_count += 1
            
    def step(self, action):
        self.current_radiation_exposure_previous = self.heli.current_radiation_exposure
        self.sim.step(action)
        if action_type == 'ta without height change with termination option':
            if action[1] < 0:
                self.env_sim.end_sim = True
                self.env_sim.end_sim_reason = 'Manual termination'
        self.difference_from_init_to_current_position_geo = np.array(self.heli.position_as_geo_coordinate) - np.array(self.heli.position_as_geo_coordinate_init)
        self.check_if_sim_needs_termination()
        self.reward = self._calculate_reward(action)
        self.rewards.append(self.reward)
        self.total_actions_per_episode += 1
        self.total_reward_per_episode += self.reward
        self.average_reward_per_episode = self.total_reward_per_episode/self.total_actions_per_episode
        obs = self._get_obs()
        self.observation = obs
        self.info['cost'] = - self.rew_radiation # probably cost wants to be minimized. so i use - here to make radiation reward positive
        if self.env_sim.end_sim:
            self.episode_count += 1
            self.info['data_at_end_of_preceding_episode'] = {  # logging will be displayed in wandb
                'total_reward': self.total_reward_per_episode,
                'mean_reward': self.total_reward_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
                'total_actions_per_episode': self.total_actions_per_episode,
                'count_radiation_exposure_exceeded': self.count_radiation_exposure_exceeded,
                'percentage_of_target_area_left': self.heli.percentage_of_target_area_left,
                'mean_rew_little_action_per_episode': self.rew_little_action_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
                'mean_incremental_tv_per_episode': self.rew_incremental_tv_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
            }
        return self.observation, self.reward, self.env_sim.end_sim, False, self.info
    def _calculate_reward(self, action):
        if self.heli.current_radiation_exposure > radiation_value_to_avoid:
            self.rew_radiation = constant_radiation_reward # * self.cumulative_radiation_exposure_in_between_actions_normalized
            reward = self.rew_radiation
            self.count_radiation_exposure_exceeded += 1
        else:
            self.scaling_factor_for_percentage_of_new_measured_points = self.scale_measurement_reward_weight_by_height(heli_height=self.heli.position_z_above_ground)
            self.rew_measurement_via_percentage_of_new_measured_points_scaled_by_height = weight_percentage_of_new_measured_points * self.heli.percentage_of_new_area_measured_ * self.scaling_factor_for_percentage_of_new_measured_points
            reward = self.rew_measurement_via_percentage_of_new_measured_points_scaled_by_height
            if include_reward_for_little_action_with_weight_but_added_instead_of_multiplied:
                self.rew_little_action_summand = reward_for_little_action_weight_of_summand * abs(action[0]) # np.minimum(2 * abs(action[0]), 1) * (abs(action[0])**0.5) / (abs(action[0])**0.5 + (1 - abs(action[0]))**5) #(-(abs(action[0]))**4+2*abs(action[0]) )
                reward -= self.rew_little_action_summand
                self.rew_little_action_per_episode += self.rew_little_action_summand
            self.rew_incremental_tv = weight_incremental_tv_reward * (self.heli.length_new_measured_minus_length_old_measured)
            reward += self.rew_incremental_tv
            self.rew_incremental_tv_per_episode += self.rew_incremental_tv
        reward += reward_for_time_passed
        return reward
    def scale_measurement_reward_weight_by_height(self, heli_height):
        heights = [-100, 0, 80, 85, 90, 95, 100, 349, 1000]
        weights = [0.01, 0.01, rew_weight_height_interpolation , 1, 1, 1, rew_weight_height_interpolation, 0.01, 0.01]
        return np.interp(heli_height, heights, weights)
    def compute_grid_averages_vectorized(self, radiation_points, grid, tree=None):
        """
        Map each radiation_point to its closest grid point and compute the average value for each grid cell.
        """
        # Flatten grid to (N, 2) for KDTree
        grid_flat = grid.reshape(-1, 2)
        # Query nearest grid point for each radiation point
        _, idxs = tree.query(radiation_points[:, :2])
        # Prepare array for sums and counts
        sums = np.zeros(grid_flat.shape[0], dtype=np.float64)
        counts = np.zeros(grid_flat.shape[0], dtype=np.int32)
        # Accumulate sums and counts
        np.add.at(sums, idxs, radiation_points[:, 2])
        np.add.at(counts, idxs, 1)
        # Avoid division by zero
        averages = np.zeros_like(sums)
        mask = counts > 0
        averages[mask] = sums[mask] / counts[mask]
        # Reshape to grid
        return averages.reshape(grid.shape[:2])
    def get_observation_as_images_(self):
        boolean_maps_within_target_area = []
        observed_radiation_mapped_to_grid = []
        center_of_grid_as_lon_lat = self.heli.position_as_geo_coordinate
        direction_given_by_bearing = self.heli.current_bearing
        number_of_points = observed_image_size  # has to be odd
        self.grids_on_earth_for_observation_with_offset = []
        for i in range(len(self.grids_on_earth_for_observation)):
            # following function makes sim 3x slower
            self.grids_on_earth_for_observation_with_offset.append(self.grids_on_earth_for_observation[i] + self.difference_from_init_to_current_position_geo)
            self.heli.target_area_rotated = self.rotate_polygon_geodesic(
                self.heli.target_area,
                -self.heli.current_bearing,
                Point(self.heli.position_as_geo_coordinate)
            )
            boolean_maps_within_target_area.append(
                self.get_grid_from_polygons(self.grids_on_earth_for_observation_with_offset[i], self.heli.target_area_rotated)
            )

            # Create radiation_map_grid for this iteration using the vectorized function.
            radiation_points = self.heli.observed_radiation_map_points  # shape: (n, 3): lon, lat, radiation_value
            if radiation_points.shape[0] == 0:
                radiation_map_grid = np.zeros((number_of_points, number_of_points))
            else:
                # Rotate the radiation points by the bearing
                radiation_points_rotated_coords = self.rotate_coords(radiation_points[:,0:2], center_of_grid_as_lon_lat[0], center_of_grid_as_lon_lat[1], -direction_given_by_bearing) - self.difference_from_init_to_current_position_geo
                radiation_points_rotated_with_buffer_to_signal_that_point_has_been_measured = np.concatenate([radiation_points_rotated_coords, radiation_points[:,2].reshape(-1, 1)+1], axis=1)
                # Use cached KDTree for this grid
                radiation_map_grid = self.compute_grid_averages_vectorized(
                    radiation_points_rotated_with_buffer_to_signal_that_point_has_been_measured,
                    self.grids_on_earth_for_observation[i],
                    tree=self.grids_kdtrees[i]
                )

            observed_radiation_mapped_to_grid.append(radiation_map_grid)

        return boolean_maps_within_target_area, observed_radiation_mapped_to_grid

    def set_seed(self, seed: int):
        random.seed(seed)

    def close(self):
        pass

    def render(self):
        pass

    def get_state(self):
        """
        Returns a dict containing all state needed for _get_obs().
        """
        state = {
            # GymnasiumEnv state
            "difference_from_init_to_current_position_geo": np.copy(self.difference_from_init_to_current_position_geo),
            "grids_on_earth_for_observation": [np.copy(grid) for grid in self.grids_on_earth_for_observation],
            # Heli state
            "heli": self.heli.get_state(),
            # Env state
            "env_sim": self.env_sim.get_state(),
        }
        return state

    def set_state(self, state):
        """
        Restores all state needed for _get_obs() from a dict.
        """
        self.difference_from_init_to_current_position_geo = np.copy(state["difference_from_init_to_current_position_geo"])
        self.grids_on_earth_for_observation = [np.copy(grid) for grid in state["grids_on_earth_for_observation"]]
        self.heli.set_state(state["heli"])
        self.env_sim.set_state(state["env_sim"])

'''
from omnisafe.envs.core import registry
from omnisafe.envs.core import CMDP
import torch
@env_register
class GymEnvOmniSafe(CMDP):
    """OmniSafe-compatible wrapper for GymnasiumEnv."""
    _support_envs = ["GymEnvOmniSafe-v0"]

    def __init__(self, env_id="GymEnvOmniSafe-v0", **kwargs):
        super().__init__(env_id=env_id, cost_limit=25.0, **kwargs)
        self._env = GymnasiumEnv()
        self._observation_space = self._env.image_observation_space
        self._action_space = self._env.action_space
        self._num_envs = 1

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def need_time_limit_wrapper(self):
        return False

    @property
    def need_auto_reset_wrapper(self):
        return False

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        # Convert observation dict to torch tensor if needed
        obs_tensor = torch.from_numpy(np.asarray(obs["image"])).float()
        return obs_tensor, info

    def step(self, action):
        # Convert action to numpy if it's a torch tensor
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, terminated, truncated, info = self._env.step(action)
        cost = float(self._env.heli.current_radiation_exposure)
        obs_tensor = torch.from_numpy(np.asarray(obs["image"])).float()
        reward = torch.tensor(reward, dtype=torch.float32)
        cost = torch.tensor(cost, dtype=torch.float32)
        terminated = torch.tensor(terminated, dtype=torch.bool)
        truncated = torch.tensor(truncated, dtype=torch.bool)
        return obs_tensor, reward, cost, terminated, truncated, info

    def close(self):
        self._env.close()

    def render(self, mode="human"):
        self._env.render(mode=mode)

    def set_seed(self, seed=None):
        self._env.set_seed(seed)
'''