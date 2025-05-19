# run this in z_test.ipynb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Polygon, mapping
import pandas as pd
import misc.simulate_and_precalculate_radiation_data

base_path = './misc/radiation_data/simulated/'
scenarios_filename = base_path + 'simulated_radiation_scenarios.npy'
measuring_area_filename = base_path + 'measuring_area_scenarios.npy'
geo_coords_filename = base_path + 'simulated_radiation_scenarios.npy_geo_coords.npy'

scenarios = np.load(scenarios_filename)
measuring_areas = np.load(measuring_area_filename, allow_pickle=True)
geo_coords = np.load(geo_coords_filename)

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

num_scenarios = len(scenarios)
for scenario_idx in range(num_scenarios):
    grid = geo_coords[scenario_idx]
    radiation = scenarios[scenario_idx]
    poly: Polygon = measuring_areas[scenario_idx]['target_area']
    x, y = poly.exterior.xy

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [grid[..., 0].min(), grid[..., 0].max(), grid[..., 1].min(), grid[..., 1].max()]
    im = ax.imshow(
        radiation.T,  # transpose if needed for correct orientation
        cmap='hot',
        origin='lower',
        extent=extent,
        alpha=0.7
    )
    ax.plot(x, y, color='cyan', linewidth=2, label='Polygon border')
    plt.tight_layout()
    plt.show()
# ...existing code...