# DANGER: ONLY IN MAIN BRANCH ONLY POSSIBLE TO USE ONE SWEEP at a time
num_runs_displayed = 6
sweep_ids = ["9lh37lqs"] #, 9hiajomq xxdjd7b6
# DANGER: ONLY IN MAIN BRANCH ONLY POSSIBLE TO USE ONE SWEEP at a time
upload_vis_to_enable_easy_sharing = False
run_ids_to_be_considered = []
cluster = 'lmu' # needs to be set manually. how to automate? sweep could get cluster as info. no. check on both clusters. introduces too much unneccesary complexity and slow.

# Retrieve the lates
# Start a local server and open the HTML file
import wandb
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import matplotlib.pyplot as plt
import os
import time
import git
import subprocess
from stable_baselines3 import PPO, A2C, TD3, DDPG, SAC, DQN
from sb3_contrib import TRPO
import numpy as np
import copy
import yaml
import importlib
import sys
import visualization
import json
import base64
import datetime
import shutil
from shapely.geometry import mapping
from tqdm import tqdm


# Define main folders
main_folder = os.getcwd()
git_clones_folder = os.path.join(main_folder, 'git_clones')
html_data_folder = os.path.join(main_folder, 'html_data')

importlib.reload(visualization)
from visualization import append_output_gymenv_values, append_to_metric_data

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Prepare lists to store the three different image sizes
images_small_for_3d_visualisation = []
images_medium_for_3d_visualisation = []
images_large_for_3d_visualisation = []

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def clone_repo(repo_url, folder_name, commit_id):
    clone_path = folder_name
    os.system('git config --global http.postBuffer 157286400')
    if os.path.exists(clone_path):
        print(f"Skip clone (Already exists).")
        return
    os.makedirs(clone_path, exist_ok=True)
    repo = git.Repo.clone_from(repo_url, clone_path)
    print(f"Repository cloned to {clone_path}")
    repo.git.checkout(commit_id)
    print(f"Checked out to commit {commit_id}")

repo_url = "https://github.com/JohannBlake/Simulation.git"

for sweep_id in sweep_ids:
    sweep_path = f"johanndavidblake-ludwig-maximilianuniversity-of-munich/Heli-Logs/{sweep_id}"
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    runs = sweep.runs
    run_ids = [run.id for run in runs]
    # Restrict run_ids to the ones specified in the run_ids list. If the list is empty, all runs are considered
    if run_ids_to_be_considered:
        run_ids = [run_id for run_id in run_ids if run_id in run_ids_to_be_considered]

    # Clone the repository only once per sweep
    first_run_id = run_ids[0]
    run = api.run(f"{sweep_path}/{first_run_id}")
    commit_id_fitting_to_model = run.config['commit_id']
    sweep_base_folder = os.path.join(git_clones_folder, sweep_id)
    sweep_base_folder_test_file_to_decide_to_delete = os.path.join(
        git_clones_folder, sweep_id, 'Simulation', 'parameters_default.yaml'
    )
    if os.path.exists(sweep_base_folder) and not os.path.exists(sweep_base_folder_test_file_to_decide_to_delete):
        shutil.rmtree(sweep_base_folder)
    base_folder = os.path.join(sweep_base_folder, 'Simulation')
    clone_repo(repo_url, base_folder, commit_id_fitting_to_model)

    for run_id in run_ids:
        run_path = f"{sweep_path}/{run_id}"
        run = api.run(run_path)
        run_name = run.name
        config = run.config

        # Update parameters
        os.chdir(base_folder)
        yaml_file_path = os.path.join(base_folder, 'parameters_default.yaml')
        with open(yaml_file_path, 'r') as file:
            parameters = yaml.safe_load(file)
        parameters.update(config)
        with open(yaml_file_path, 'w') as file:
            yaml.safe_dump(parameters, file)

        # SCP command setup
        if cluster == 'lmu':
            remote_user = 'blake@madeira.dbs.ifi.lmu.de'
            remote_base_path = f"/home/stud/blake/git_clones/Simulation_{sweep_id}/logs/{run_name}"
            remote_model_zip_path = f"{remote_base_path}/best_model.zip"
            destination_model_zip_path = os.path.join(base_folder, f"{run_name}-best_model.zip")
            scp_command = (
                f'scp {remote_user}:"{remote_model_zip_path}" "{destination_model_zip_path}"'
            )
        elif cluster == 'lrz':
            remote_user = 'di97sog@login.ai.lrz.de'
            remote_base_path = f"/dss/dsshome1/0C/di97sog/git_clones/Simulation_{sweep_id}/logs/{run_name}"
            ssh_key = "C:\\Users\\johan\\.ssh\\id_rsa_lrz"
            remote_model_zip_path = f"{remote_base_path}/best_model.zip"
            destination_model_zip_path = os.path.join(base_folder, f"{run_name}-best_model.zip")
            scp_command = (
                f'scp -i "{ssh_key}" {remote_user}:"{remote_model_zip_path}" "{destination_model_zip_path}"'
            )
        #print('attention: best model not copied. was turned off bevause cluster ont reachable')
        subprocess.run(scp_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')

        model_file_name = f"{run_name}-best_model.zip"
        model_path = os.path.join(base_folder, model_file_name)
        os.chdir(base_folder)
        sys.path.insert(0, base_folder)
        import class_gymenv
        importlib.reload(class_gymenv)
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
        import get_parameters
        importlib.reload(get_parameters)
        from get_parameters import *
        from misc.helpful_geo_functions import compute_geo_coordinate_from_position
        if 'gymenv' in globals():
            del gymenv
        os.chdir(base_folder)
        import misc.simulate_and_precalculate_radiation_data
        # Initialize with a single environment instead of 8
        height_data = np.load('height_data.npy')
        measuring_area_scenarios = np.load(os.path.join('.', 'misc', 'radiation_data', 'simulated', 'measuring_area_scenarios.npy'), allow_pickle=True) 
        scenarios_filename = os.path.join('.', 'misc', 'radiation_data', 'simulated', f'simulated_radiation_scenarios.npy')
        radiation_data = np.load(scenarios_filename)
        scenarios_geo_coords = np.load(scenarios_filename + '_geo_coords' + '.npy' , allow_pickle=True)

        envs = [lambda: Monitor(class_gymenv.GymnasiumEnv(height_data, measuring_area_scenarios, radiation_data, scenarios_geo_coords))]
        gymenv = DummyVecEnv(envs)
        gymenv = VecTransposeImage(gymenv)
        def load_model(model_path, env):
            model_classes = {
                'PPO': PPO,
                'A2C': A2C,
                'TD3': TD3,
                'DDPG': DDPG,
                'SAC': SAC,
                'DQN': DQN,
                'TRPO': TRPO
            }
            try:
                model_class = model_classes[parameters['sb3_model_type']]
                model = model_class.load(model_path, env=env, buffer_size=10)
                print(f"Loaded {parameters['sb3_model_type']} model")
                return model
            except Exception as e:
                print(f"Failed to load {parameters['sb3_model_type']} model: {e}")
                raise ValueError("Failed to load model")

        model = load_model(model_path, gymenv)
        sbenv = model.get_env()

        config_data = run.config
        config_data.pop('commit_id', None)
        config_data.pop('sweep_info', None)
        config_data['run_name'] = run.name

        obs = sbenv.reset()
        images_for_3d_visualisation = []
        paths_for_3d_visualisation = []
        metric_data = {}

        # Track the geo-coordinates at each step
        observed_images_geo_coordinates_history = []
        geo_json = {
            "type": "FeatureCollection",
            "features": []
        }
        current_step_in_animation = 0
        previous_polygon_with_height_target_area = None
        positions_per_step_list = []
        colors_per_step_list = []
        tree = sbenv.envs[0].unwrapped.env_sim.tree_surface_point_cloud_
        surface_point_cloud = np.array(sbenv.envs[0].unwrapped.env_sim.surface_point_cloud_)

        for episode_index in tqdm(range(num_runs_displayed), desc="Processing episodes"):  # Wrap loop with tqdm
            terminated = False
            path_of_episode = []
            lon = sbenv.envs[0].unwrapped.heli.position_as_geo_coordinate[0]
            lat = sbenv.envs[0].unwrapped.heli.position_as_geo_coordinate[1]
            if surface_grid_creation_type == 'no_height_map':
                # Get height from the surface point cloud using tree query logic
                _, index = tree.query(np.array([sbenv.envs[0].unwrapped.heli.position_as_geo_coordinate[0:2]]))  # Wrap position in 2D array
                surface_grid_height = surface_point_cloud[index[0], 2]  # Retrieve height (z-coordinate)
                path_of_episode.append([float(lon), float(lat), float(surface_grid_height + 90)])
            else:
                path_of_episode.append([float(lon), float(lat), float(sbenv.heli.position[2])])
            step_number = 0
            coords = np.array(sbenv.envs[0].unwrapped.env_sim.geo_coordinates_of_radiation_grid)  # shape: (h, w, 2)
            positions_array = np.dstack([coords[:, :, 0], coords[:, :, 1]])  # shape: (h, w, 3)

            while not terminated:
                action, _states = model.predict(obs, deterministic=True)
                output_vectorized_env = sbenv.step(action)
                obs = copy.deepcopy(output_vectorized_env[0])
                gymenv = sbenv.envs[0].unwrapped
                terminated = output_vectorized_env[2][0]

                # Log step number and episode progress
                tqdm.write(f"Episode {episode_index + 1}/{num_runs_displayed}, Step {step_number + 1}")

                # Radiation color using inferno
                radiation_grid = gymenv.env_sim.radiation_grid
                min_val = np.nanmin(radiation_grid)
                max_val = np.nanmax(radiation_grid)
                if max_val > min_val:
                    normalized_radiation = (radiation_grid - min_val) / (max_val - min_val)
                else:
                    normalized_radiation = np.zeros_like(radiation_grid)
                normalized_radiation = np.clip(normalized_radiation, 0, 1)
                inferno_cmap = plt.get_cmap('inferno')
                colored_array = (inferno_cmap(normalized_radiation) * 255).astype(np.uint8)  # shape: (h, w, 4)
                
                # Store arrays for this step
                positions_per_step_list.append(positions_array)
                colors_per_step_list.append(colored_array)

                lon = gymenv.heli.position_as_geo_coordinate[0]
                lat = gymenv.heli.position_as_geo_coordinate[1]
                if not terminated:
                    if surface_grid_creation_type == 'no_height_map':
                        # Get height from the surface point cloud using tree query logic
                        _, index = tree.query(np.array([gymenv.heli.position_as_geo_coordinate[0:2]]))  # Wrap position in 2D array
                        surface_grid_height = surface_point_cloud[index[0], 2]  # Retrieve height (z-coordinate)
                        path_of_episode.append([float(lon), float(lat), float(surface_grid_height + 90)])
                    else:
                        path_of_episode.append([float(lon), float(lat), float(sbenv.heli.position[2])])
                polygon_with_height_target_area = gymenv.heli.target_area


                # if previous_polygon_with_height_target_area:
                #     diff_polygon_with_height_target_area = polygon_with_height_target_area.difference(
                #         previous_polygon_with_height_target_area
                #     )
                # else:
                #     diff_polygon_with_height_target_area = polygon_with_height_target_area

                geo_json["features"].append({
                    "type": "Feature",
                    "geometry": mapping(polygon_with_height_target_area),
                    "properties": {
                        "category": "target_area",
                        "timestamp": gymenv.env_sim.time_rel,
                        "current_step_in_animation": current_step_in_animation
                    }
                })

                geo_json["features"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": copy.deepcopy(path_of_episode)
                    },
                    "properties": {
                        "category": "path_of_episode",
                        "timestamp": gymenv.env_sim.time_rel,
                        "current_step_in_animation": current_step_in_animation,
                        "run_id": run_id  # Added run identifier property to tag episode
                    }
                })
                previous_polygon_with_height_target_area = polygon_with_height_target_area
                append_output_gymenv_values(metric_data, gymenv)
                step_number += 1
                current_step_in_animation += 1
            tqdm.write(f"Episode {episode_index + 1}/{num_runs_displayed} completed with {step_number} steps")  # Log episode completion

        serializable_run_config = convert_to_serializable(config_data)
        os.chdir(main_folder)

        run_html_data_folder = os.path.join(html_data_folder, sweep_id, run_id)
        os.makedirs(run_html_data_folder, exist_ok=True)

        run_config_file_path = os.path.join(run_html_data_folder, 'run_config.json')
        with open(run_config_file_path, 'w') as f:
            json.dump(serializable_run_config, f)
        serializable_metric_data = convert_to_serializable(metric_data)
        all_point_clouds = []
        for step_index, (pos_array, col_array) in enumerate(zip(positions_per_step_list, colors_per_step_list)):
            # Flatten positions and colors
            flattened_positions = pos_array.reshape(-1, 2).tolist()         # shape: (h*w, 3)
            flattened_colors = col_array[:, :, :3].reshape(-1, 3).tolist()  # shape: (h*w, 3)
            
            # Filter out points where all color components are below 5
            filtered_positions_colors = [
                (pos, col) for pos, col in zip(flattened_positions, flattened_colors)
                if not all(c < 50 for c in col)
            ]
            filtered_positions, filtered_colors = zip(*filtered_positions_colors) if filtered_positions_colors else ([], [])

            # Bundle them into a dictionary for each step
            point_cloud_dict = {
                "step": step_index,
                "positions": filtered_positions,
                "colors": filtered_colors
            }
            all_point_clouds.append(point_cloud_dict)
        
        # Save to JSON
        point_cloud_file_path = os.path.join(run_html_data_folder, 'radiation_point_cloud.json')
        with open(point_cloud_file_path, 'w') as f:
            json.dump(all_point_clouds, f)
        center_coordinates = {
            "center_lat": anchor_point[1],
            "center_lon": anchor_point[0]
        }
        
        # Define the file path to save the JSON
        geo_json_file_path = os.path.join(run_html_data_folder, 'geo_json_data.json')

        # Save the geo_json dictionary to a file
        with open(geo_json_file_path, 'w') as file:
            json.dump(geo_json, file)

        metric_data_file_path = os.path.join(run_html_data_folder, 'metric_data.json')
        with open(metric_data_file_path, 'w') as f:
            json.dump(serializable_metric_data, f)

        center_coordinates_file_path = os.path.join(run_html_data_folder, 'center_coordinates.json')
        with open(center_coordinates_file_path, 'w') as f:
            json.dump(center_coordinates, f)

    run_ids_file_path = os.path.join(html_data_folder, sweep_id, 'run_ids.json')
    with open(run_ids_file_path, 'w') as f:
        json.dump(run_ids, f)

sweep_ids_file_path = os.path.join(html_data_folder, 'sweep_ids.json')
with open(sweep_ids_file_path, 'w') as f:
    json.dump(sweep_ids, f)

import os
import shutil
from datetime import datetime

folder_to_zip = html_data_folder
zip_file_path = os.path.join(main_folder, 'misc/html_data.zip')
shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', root_dir=folder_to_zip, base_dir='.')
shutil.rmtree(folder_to_zip)

import os
import shutil
import datetime
import subprocess

root_dir = os.getcwd()
archive_subdir = 'misc/3d_vis_archive' if upload_vis_to_enable_easy_sharing else 'misc/3d_vis_archive/offline'
archive_dir = os.path.join(root_dir, archive_subdir)

os.makedirs(archive_dir, exist_ok=True)

vis_timestamp_dir = os.path.join(archive_dir, f'vis_{timestamp}')
os.makedirs(vis_timestamp_dir, exist_ok=True)

original_js = os.path.join(root_dir, 'misc','vis_3d_main.js')
with open(original_js, 'r', encoding='utf-8') as f:
    content = f.read()
if upload_vis_to_enable_easy_sharing:
    content = content.replace(
        "fetch('html_data.zip')",
        f"fetch('https://JohannBlake.github.io/Simulation/misc/3d_vis_archive/vis_{timestamp}/html_data.zip')"
    )

new_js = os.path.join(vis_timestamp_dir, 'vis_3d_main.js')
with open(new_js, 'w', encoding='utf-8') as f:
    f.write(content)

index_file_path = os.path.join(root_dir, 'misc','vis_3d_index.html')
with open(index_file_path, 'r', encoding='utf-8') as f:
    index_content = f.read()
if upload_vis_to_enable_easy_sharing:
    index_content = index_content.replace(
        '<link rel="stylesheet" href="vis_3d_styles.css">',
        f'<link rel="stylesheet" href="https://JohannBlake.github.io/Simulation/misc/3d_vis_archive/vis_{timestamp}/vis_3d_styles.css">'
    )
    index_content = index_content.replace(
        '<script src="vis_3d_main.js"></script>',
        f'<script src="https://JohannBlake.github.io/Simulation/misc/3d_vis_archive/vis_{timestamp}/vis_3d_main.js"></script>'
    )

new_index_file_path = os.path.join(vis_timestamp_dir, 'vis_3d_index.html')
with open(new_index_file_path, 'w', encoding='utf-8') as f:
    f.write(index_content)

shutil.copy(os.path.join(main_folder, 'misc', 'vis_3d_styles.css'), vis_timestamp_dir)
shutil.copy(os.path.join(main_folder, 'misc', 'html_data.zip'    ), vis_timestamp_dir)

# start a local server and open html file
# Change directory to the folder containing the HTML file

if upload_vis_to_enable_easy_sharing:
    def run_git_command(command):
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(result.stderr)
        else:
            print(result.stdout)

    run_git_command("git add .")
    commit_message = "commit to upload vis data"
    run_git_command(f'git commit -m "{commit_message}"')
    run_git_command("git push origin main")

    time.sleep(120)

    html_file_folder = os.path.join(main_folder, 'misc', '3d_vis_archive', f'vis_{timestamp}')
    os.startfile(html_file_folder)
else:
    # Extract the directory and file name from index_file_path
    index_file_directory = os.path.dirname(new_index_file_path)
    index_file_name = os.path.basename(new_index_file_path)

    # Change to the directory containing the index file
    os.chdir(index_file_directory)

    # Start a local server
    PORT = 8000
    with TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
        # Suppress server outputs
        with open(os.devnull, 'w') as devnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                # Open the index file in the default web browser
                webbrowser.open(f"http://localhost:{PORT}/{index_file_name}")
                # Keep the server running
                httpd.serve_forever()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr