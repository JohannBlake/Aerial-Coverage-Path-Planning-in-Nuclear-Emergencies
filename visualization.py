import numpy as np
import pandas as pd
from geopy.distance import geodesic
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from plotly import graph_objects as go
import plotly.io as pio
import os
import numpy as np
import plotly.graph_objects as go
import numpy as np
import importlib
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *

def append_to_metric_data(metric_data, key, value, precision=3):
    if key[:20] not in metric_data.keys():
        metric_data[key[:20]] = []
    if isinstance(value, (int, float, np.float32, np.float64)):
        metric_data[key[:20]].append(round(value, precision))
    elif isinstance(value, bool) or value == None:
        metric_data[key[:20]].append(value)
    else:
        metric_data[key[:20]].append(value)
def append_metric_data_if_exists(obj, attr_name, metric_name, metric_data):
    if hasattr(obj, attr_name):
        append_to_metric_data(metric_data, metric_name, getattr(obj, attr_name))
    else:
        append_to_metric_data(metric_data, metric_name, None)

def append_output_gymenv_values(metric_data, gymenv):
    append_metric_data_if_exists(gymenv, 'reward','Reward', metric_data)
    append_metric_data_if_exists(gymenv, 'rew_measurement_via_percentage_of_new_measured_points_scaled_by_height','rew_measurement_via_percentage_of_new_measured_points_scaled_by_height', metric_data)
    append_metric_data_if_exists(gymenv, 'rew_incremental_tv','rew_incremental_tv', metric_data)
    append_metric_data_if_exists(gymenv.env_sim, 'percentage_of_episode_finished_by_time_passed', "% episode finished", metric_data)
    append_metric_data_if_exists(gymenv, 'rew_radiation', 'Rew. radiation', metric_data)
    append_metric_data_if_exists(gymenv, 'rew_little_action_summand', 'rew_little_action_summand', metric_data)
    if hasattr(gymenv.heli, 'position'):
        append_to_metric_data(metric_data, 'Height rel. to ground', gymenv.heli.position_z_above_ground)
    else:
        append_to_metric_data(metric_data, 'Height rel. to ground', None)
    append_metric_data_if_exists(gymenv.heli, 'current_radiation_exposure', 'Curr rad exp', metric_data)
    append_metric_data_if_exists(gymenv.heli, 'target_area_size', 'target_area_size', metric_data)
    return metric_data
def visualize_heightmap(gridsize, heightmap):
    x = np.arange(0, gridsize[0])
    y = np.arange(0, gridsize[1])
    x, y = np.meshgrid(x, y)
    z = heightmap

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='3D Heightmap', autosize=True,
                        scene=dict(zaxis=dict(range=[0, 150]),
                                    aspectratio=dict(x=1, y=1, z=0.5)))
    fig.show()

def append_reward_bar_to_image(image, reward):
    # Ensure the image has 3 channels
    if image.shape[-1] == 2:  # If the image has 2 channels
        image = np.stack((image[:, :, 0], image[:, :, 1], np.zeros_like(image[:, :, 0])), axis=-1)
    elif len(image.shape) == 2:  # If the image is grayscale (2D)
        image = np.stack((image, image, image), axis=-1)

    image_height, image_width = image.shape[:2]

    # Create a black bar of 3 pixels height and the same width as the image
    reward_bar = np.zeros((3, image_width, 3), dtype=np.uint8)

    # Find the midpoint of the image width
    midpoint = image_width // 2

    # Calculate the length of the reward bar
    reward_length = int((image_width * abs(reward)) / 2)
    reward_length = min(reward_length, midpoint)  # Cap the reward length to the midpoint

    # Draw the reward bar
    if reward > 0:
        # Green bar for positive reward
        reward_bar[:, midpoint:midpoint + reward_length] = [0, 255, 0]  # Green color
    else:
        # Red bar for negative reward
        reward_bar[:, midpoint - reward_length:midpoint] = [255, 0, 0]  # Red color

    # Create a white line of 1 pixel height and the same width as the image
    white_line_horizontal = np.ones((1, image_width, 3), dtype=np.uint8) * 255

    # Use np.vstack to add the white line and the black bar with the reward visualization below the image
    image = np.vstack((image, white_line_horizontal, reward_bar))
    return image