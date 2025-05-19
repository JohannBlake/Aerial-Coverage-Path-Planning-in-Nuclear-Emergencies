import numpy as np
import math
import random
import os
from skimage.transform import rotate
from misc.helpful_geo_functions import get_grid_centroids_and_values, compute_geo_coordinates_from_grid
from get_parameters import *
from geopy.distance import distance
from shapely.geometry import Polygon
from shapely.ops import transform
import geopy.distance
from shapely.affinity import translate
from shapely.geometry import Point

# --- Geo helpers ---

def generate_polygon_center(max_distance_km=0):
    return distance(kilometers=random.uniform(0, max_distance_km)).destination((anchor_point[1],anchor_point[0]), random.uniform(0, 360))

def random_polygon(center_lon, center_lat):
    area_target = np.clip(20 + np.random.exponential(scale=45), 20, 150) # size in km². min 20, max 200, more likely 20
    num_points = np.random.randint(3, 8)
    while True:
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
        scale = np.sqrt(area_target) / 2  # ungefähre Skala in km
        points = []
        for a in angles:
            dist = np.random.uniform(0.5 * scale, 1.5 * scale)
            destination = geopy.distance.distance(kilometers=dist).destination((center_lat, center_lon), np.degrees(a))
            points.append((destination.longitude, destination.latitude))
        poly = Polygon(points)
        if poly.is_valid:
            centroid = poly.centroid
            dx = center_lon - centroid.x
            dy = center_lat - centroid.y
            poly_shifted = translate(poly, xoff=dx, yoff=dy)
            return poly_shifted

def vector_length(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def points_distance(point1, point2):
    return vector_length((point1[0] - point2[0], (point1[1] - point2[1])))

def clamp(value, minimum, maximum):
    return max(min(value, maximum), minimum)

# --- Image transformation helpers ---

def mirror_image(image):
    return np.fliplr(image)

def rotate_image(image):
    angle = random.uniform(0, 360)
    return rotate(image, angle, resize=False, mode='constant', cval=0)

def rescale_image(image):
    scale_factor = random.uniform(0.1, 5)
    return image * scale_factor

def shift_image(image):
    h, w = image.shape
    shift_x = random.randint(-int(0.25 * w), int(0.25 * w))
    shift_y = random.randint(-int(0.25 * h), int(0.25 * h))
    shifted_image = np.roll(image, shift_x, axis=1)
    shifted_image = np.roll(shifted_image, shift_y, axis=0)
    return shifted_image

def warp_array(values_grid):
    h, w = values_grid.shape
    result = np.zeros_like(values_grid)
    max_idx = np.unravel_index(np.argmax(values_grid), values_grid.shape)
    max_y, max_x = max_idx
    radius = 10
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, radius)
    px_ = int(np.clip(max_x + r * np.cos(angle), 0, w - 1))
    py_ = int(np.clip(max_y + r * np.sin(angle), 0, h - 1))
    dx = np.random.randint(-11, 11)
    dy = np.random.randint(-11, 11)
    points = [(px_, py_, dx, dy)]

    # Vectorized grid coordinates
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    offset_x = np.zeros((h, w), dtype=np.float32)
    offset_y = np.zeros((h, w), dtype=np.float32)

    for px_, py_, dx, dy in points:
        shift_vector = np.array([dx, dy], dtype=np.float32)
        sv_len = np.linalg.norm(shift_vector)
        if sv_len == 0:
            continue
        point_position = np.array([px_ + dx, py_ + dy], dtype=np.float32)
        # Compute distances
        dist = np.sqrt((x_grid - point_position[0]) ** 2 + (y_grid - point_position[1]) ** 2)
        helper = 1.0 / (3 * (dist / sv_len) ** 4 + 1)
        offset_x -= helper * shift_vector[0]
        offset_y -= helper * shift_vector[1]

    nx = np.clip(np.round(x_grid + offset_x).astype(int), 0, w - 1)
    ny = np.clip(np.round(y_grid + offset_y).astype(int), 0, h - 1)
    result = values_grid[ny, nx]
    return result

def transform_radiation_data(radiation_data):
    skewed_grid = warp_array(radiation_data)
    mirrored_grid = mirror_image(skewed_grid)
    rotated_grid = rotate_image(mirrored_grid)
    rescaled_grid = rescale_image(rotated_grid)
    shifted_grid = shift_image(rescaled_grid)
    return shifted_grid

# --- Main simulation logic ---

def process_scenario(args):
    i, radiation_data, granularity = args
    polygon_center = generate_polygon_center()
    target_area = random_polygon(polygon_center.longitude, polygon_center.latitude)
    geo_coordinates_of_radiation_grid = compute_geo_coordinates_from_grid(
        gridsize=(radiation_data.shape[0], radiation_data.shape[1]),
        anchor_point=np.array([polygon_center.longitude,polygon_center.latitude]),
        granularity=granularity
    )
    scenario = transform_radiation_data(radiation_data)

    # --- shift polygon so its centroid lands on the highest-radiation cell ---
    max_y, max_x = np.unravel_index(np.argmax(scenario), scenario.shape)
    high_rad_coord = geo_coordinates_of_radiation_grid[max_y, max_x]

    def random_point_in_polygon(polygon):
        minx, miny, maxx, maxy = polygon.bounds
        for _ in range(20):
            rand_x = np.random.uniform(minx, maxx)
            rand_y = np.random.uniform(miny, maxy)
            p = Point(rand_x, rand_y)
            if polygon.contains(p):
                return p
        return polygon.centroid

    random_point = random_point_in_polygon(target_area)
    dx = high_rad_coord[0] - random_point.x
    dy = high_rad_coord[1] - random_point.y
    target_area = translate(target_area, xoff=dx, yoff=dy)

    measuring_area_scenario = {
        "polygon_center": high_rad_coord,
        "target_area": target_area,
    }
    return scenario, geo_coordinates_of_radiation_grid, measuring_area_scenario

def precalculate_and_save_simulated_radiation_scenarios(
    radiation_data, filename, num_scenarios, granularity, 
    measuring_area_scenarios_filename
):
    scenarios = []
    scenarios_geo_coords = []
    measuring_area_scenarios = []

    args = [(i, radiation_data, granularity) for i in range(num_scenarios)]
    results = [process_scenario(arg) for arg in args]

    for scenario, geo_coords, measuring_area in results:
        scenarios.append(scenario)
        scenarios_geo_coords.append(geo_coords)
        measuring_area_scenarios.append(measuring_area)

    scenarios = np.array(scenarios)
    scenarios_geo_coords = np.array(scenarios_geo_coords)
    np.save(measuring_area_scenarios_filename, measuring_area_scenarios)
    np.save(filename, scenarios)
    np.save(filename + "_geo_coords.npy", scenarios_geo_coords)
def simulate_and_save_radiation_data(
    distance_east_from_anchor_point, distance_north_from_anchor_point,
    granularity, max_distance=max_distance, num_scenarios=num_scenarios
):
    save_dir = os.path.join('.', 'misc', 'radiation_data', 'simulated')
    os.makedirs(save_dir, exist_ok=True)
    measuring_area_scenarios_filename = os.path.join(save_dir, 'measuring_area_scenarios.npy')
    scenarios_filename = os.path.join(save_dir, 'simulated_radiation_scenarios.npy')
    print('Create radiation and measurement scenarios.')
    radiation_grid_from_odl_simulator_90m_above_ground = np.load(
        os.path.join('.', 'misc', 'radiation_data', 'odl_map_90m_above_ground_from_berlin_radiation_scenario_constant_2025_2026_23h_after_min.npy')
    )
    precalculate_and_save_simulated_radiation_scenarios(
        radiation_data=radiation_grid_from_odl_simulator_90m_above_ground,
        filename=scenarios_filename,
        num_scenarios=num_scenarios,
        granularity=granularity,
        measuring_area_scenarios_filename=measuring_area_scenarios_filename
    )

# Call the function
simulate_and_save_radiation_data(
    distance_east_from_anchor_point=distance_east_from_anchor_point,
    distance_north_from_anchor_point=distance_north_from_anchor_point,
    granularity=granularity,
    max_distance=500,
    num_scenarios=num_scenarios
)