# Variable to determine how to get GPS and ODL data
gps_origin = 'msfs'      # options: 'msfs', 'serial', 'input'
odl_origin = 'yevhen'    # options: 'yevhen', 'gymenv', 'input'
how_many_steps_to_predict = 1
threshold = -0.5  # threshold for ODL in mSv/h. values below this are not considered radiation
radius_for_close_enough = 150  # meters
# Fixed selections
sweep_id = "11sq4hqw"
run_id = "7pu9yjwt"
cluster = 'lmu'

'''
Instructions:
1) Start MSFS
2) Start Yevhen simulator
3) Start Comport rooter -> not needed 
4) Start this script
6) Start live.html
'''
from geopy.point import Point
from shapely.geometry import Polygon
import os
import sys
import time
import copy
import yaml
import shutil
import serial
import importlib
import numpy as np
import random
import threading
import pickle
import atexit

import wandb
import git
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from misc.helpful_geo_functions import compute_position_from_geo_coordinate
from stable_baselines3 import PPO, A2C, TD3, DDPG, SAC, DQN
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from SimConnect import SimConnect, AircraftRequests
from socketio import Client
from geopy.distance import geodesic
from shapely.geometry import mapping
import subprocess

# path to the saved polygon
polygon_path = os.path.join("C:\\Users\\johan\\programming\\Simulation", "polygon.geojson")

@atexit.register
def delete_polygon_file():
    if os.path.exists(polygon_path):
        os.remove(polygon_path)


# --- Flask server definition and initialization ---
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
# Initialize in-app storage
app.path_data = []
app.predicted_path = []
app.target_area = None

@app.route("/data", methods=["GET"])
def get_data():
    return jsonify({
        "path": [{"path": app.path_data}],
        "predicted_path": [{"path": app.predicted_path}],
        "target_area": app.target_area,
        "how_many_steps_to_predict": how_many_steps_to_predict  # <-- add this line
    })

@app.route("/update_data", methods=["POST"])
def update_data():
    data = request.get_json()
    if 'path' in data:
        app.path_data = data['path']
        return jsonify({"status": "success"}), 200
    return jsonify({"status": "error", "message": "Invalid data format"}), 400

@app.route("/upload_polygon", methods=["POST"])
def upload_polygon():
    polygon_geojson = request.get_json()
    save_path = os.path.join("C:\\Users\\johan\\programming\\Simulation", "polygon.geojson")
    with open(save_path, "w", encoding="utf-8") as f:
        import json
        json.dump(polygon_geojson, f)
    return jsonify({"status": "success", "path": save_path})

@app.route('/polygon.geojson', methods=['GET'])
def get_polygon():
    directory = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(directory, 'polygon.geojson')

@socketio.on('update_data')
def handle_update_data(data):
    new_point = data.get('new_point')
    if new_point:
        app.path_data.append(new_point)
        socketio.emit('data_updated', {"path": [{"path": app.path_data}]})

@socketio.on('predicted_path_update')
def handle_predicted_path_update(data):
    new_predicted = data.get('predicted_path')
    if new_predicted is not None:
        app.predicted_path = new_predicted
    socketio.emit('predicted_path_updated', {"predicted_path": [{"path": app.predicted_path}]})

@socketio.on('target_area_update')
def handle_target_area_update(data):
    new_target = data.get('target_area')
    if new_target:
        app.target_area = new_target
        socketio.emit('target_area_updated', {"target_area": app.target_area})

def run_flask_server():
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
# --- End Flask server definition ---

def get_aircraft_position(sm, aq):
    max_retries = 30
    for attempt in range(max_retries):
        try:
            altitude = aq.find("PLANE_ALTITUDE")
            agl = aq.find("PLANE_ALT_ABOVE_GROUND")
            latitude = aq.find("PLANE_LATITUDE")
            longitude = aq.find("PLANE_LONGITUDE")
            altitude_value = altitude.value
            agl_value = agl.value
            latitude_value = latitude.value
            longitude_value = longitude.value
            if None in (altitude_value, agl_value, latitude_value, longitude_value):
                raise ValueError("One or more aircraft values are None")
            return {
                "Altitude_above_ground": agl_value * 0.3048,
                "Latitude": latitude_value,
                "Longitude": longitude_value,
                "Altitude": altitude_value * 0.3048
            }
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            sm = SimConnect()
            aq = AircraftRequests(sm, _time=1)

# Only connect to SimConnect if needed
if gps_origin == 'msfs':
    sm = SimConnect()
    aq = AircraftRequests(sm, _time=1)
else:
    sm = None
    aq = None

main_folder = os.getcwd()
git_clones_folder = os.path.join(main_folder, 'git_clones')

# WandB setup
api = wandb.Api()
sweep_path = f"johanndavidblake-ludwig-maximilianuniversity-of-munich/Heli-Logs/{sweep_id}"
run_path = f"{sweep_path}/{run_id}"
run = api.run(run_path)
config = run.config

def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    else:
        return obj

repo_url = "https://github.com/JohannBlake/Simulation.git"
commit_id_fitting_to_model = config['commit_id']
sweep_base_folder = os.path.join(git_clones_folder, sweep_id)
sweep_base_folder_test_file_to_decide_to_delete = os.path.join(sweep_base_folder, 'Simulation', 'parameters_default.yaml')

if os.path.exists(sweep_base_folder) and not os.path.exists(sweep_base_folder_test_file_to_decide_to_delete):
    shutil.rmtree(sweep_base_folder)

base_folder = os.path.join(sweep_base_folder, 'Simulation')

def clone_repo_if_needed(repo_url, base_folder, commit_id):
    if os.path.exists(base_folder):
        print(f"Destination path '{base_folder}' already exists: Skipping clone.")
        return
    os.makedirs(base_folder, exist_ok=True)
    repo = git.Repo.clone_from(repo_url, base_folder)
    print(f"Repository cloned to {base_folder}")
    repo.git.checkout(commit_id)
    print(f"Checked out to commit {commit_id}")

clone_repo_if_needed(repo_url, base_folder, commit_id_fitting_to_model)

# Update parameters from run config
os.chdir(base_folder)
yaml_file_path = os.path.join(base_folder, 'parameters_default.yaml')
with open(yaml_file_path, 'r') as file:
    parameters = yaml.safe_load(file)
parameters.update(config)
with open(yaml_file_path, 'w') as file:
    yaml.safe_dump(parameters, file)

# Load model from server
remote_user = 'blake'
remote_host = 'madeira.dbs.ifi.lmu.de'
run_name = run.name
model_file_name = f"{run_name}-best_model.zip"
model_path = os.path.join(base_folder, model_file_name)
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
#print("TO BE ENABLED. skip loading model from server")
subprocess.run(scp_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')

os.chdir(base_folder)
sys.path.insert(0, base_folder)

import class_gymenv
import get_parameters
importlib.reload(class_gymenv)
importlib.reload(get_parameters)
from get_parameters import *

def calculate_center_coordinates(pointcloud):
    positions = np.array([point['position'] for point in pointcloud])
    center = positions.mean(axis=0)
    return {
        'centerLongitude': float(center[0]),
        'centerLatitude': float(center[1]),
        'centerHeight': float(center[2])
    }

update_interval = 0.01  # seconds

sio = Client()

def convert_to_decimal(degrees, minutes):
    return degrees + minutes / 60

def format_gpgga_message(lat, lon, alt):
    """
    Format the GPS data into the GPGGA NMEA sentence format.
    Timestamp is fixed to 12:00:00 on 06.06.2024.
    """
    timestamp = "120000.000"  # Fixed timestamp
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60
    lat_dir = "N" if lat >= 0 else "S"
    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60
    lon_dir = "E" if lon >= 0 else "W"
    altitude = f"{alt:.1f}"  # Altitude in meters

    gpgga = f"$GPGGA,{timestamp},{lat_deg:02d}{lat_min:09.6f},{lat_dir}," \
            f"{lon_deg:03d}{lon_min:09.6f},{lon_dir},1,09,2.1,{altitude},M,,M,,0000"
    checksum = calculate_checksum(gpgga)
    return f"{gpgga}*{checksum:02X}"

def calculate_checksum(sentence):
    """
    Calculate the checksum for an NMEA sentence.
    """
    checksum = 0
    for char in sentence[1:]:  # Skip the initial '$'
        checksum ^= ord(char)
    return checksum

def read_serial_data(model, gymenv, obs, sbenv, ser_data, ser_gps):
    init_run = True
    predicted_path = []
    # --- Initialize target_area_geojson_list with first 10 steps ---
    while True:
        try:
            # --- GPS acquisition ---
            if gps_origin == 'serial':
                gps_data = ser_gps.readline().decode('ascii', errors='replace')
                if gps_data.startswith('$GPGGA'):
                    parts = gps_data.split(',')
                    if len(parts) > 9:
                        lat_deg = int(float(parts[2]) / 100)
                        lat_min = float(parts[2]) % 100
                        lat = convert_to_decimal(lat_deg, lat_min)
                        if parts[3] == 'S':
                            lat = -lat
                        lon_deg = int(float(parts[4]) / 100)
                        lon_min = float(parts[4]) % 100
                        lon = convert_to_decimal(lon_deg, lon_min)
                        if parts[5] == 'W':
                            lon = -lon
                        h = float(parts[9])
                    else:
                        print('Wrong format:', gps_data)
                        continue

            elif gps_origin == 'msfs':
                position = get_aircraft_position(sm, aq)
                if not position:
                    print("Debug: Unable to retrieve MSFS position")
                    continue
                lon, lat, h = position["Longitude"], position["Latitude"], position["Altitude"]
                # send GPS to LUCY via COM9 in GPGGA format
                gpgga_msg = format_gpgga_message(lat, lon, 90)
                if ser_gps is not None:
                    try:
                        ser_gps.write(f"{gpgga_msg}\r\n".encode())
                        ser_gps.flush()
                    except Exception as e:
                        print(f"Error writing GPS data to COM9: {e}")
            elif gps_origin == 'input':
                # Use first predicted path point or default if not available
                if 'predicted_path' in locals() and predicted_path and len(predicted_path) > 0:
                    if random.random() < 0: # always false
                        # Pick a random point on earth
                        lon = random.uniform(-180, 180)
                        lat = random.uniform(-90, 90)
                        h = 90
                    else:
                        p = predicted_path[-how_many_steps_to_predict]
                        lon, lat, h = p[0], p[1], 2000
                else:
                    lon, lat, h = gymenv.heli.position_as_geo_coordinate[0], gymenv.heli.position_as_geo_coordinate[1], 2000
                print(f"Go to first predicted path point: {lon},{lat},{h}. Press Enter to continue.")
                _ = input()
                gpgga_msg = format_gpgga_message(lat, lon, 90)
                if ser_gps is not None:
                    try:
                        ser_gps.write(f"{gpgga_msg}\r\n".encode())
                        ser_gps.flush()
                    except Exception as e:
                        print(f"Error writing GPS data to COM9: {e}")

            # --- ODL acquisition ---
            if odl_origin == 'yevhen':
                data = ser_data.read(6)
                if not data:
                    print("Debug: No data received for ODL")
                else:
                    mlo, mhi = data[2], data[3]
                    mant = mhi * 256 + mlo
                    exp = data[4] - 256 if data[4] > 127 else data[4]
                    odl = mant * (2 ** (exp - 15))
            elif odl_origin == 'gymenv':
                odl = gymenv.heli.current_radiation_exposure
            elif odl_origin == 'input':
                # Use first predicted path point or default if not available
                if 'predicted_path' in locals() and predicted_path and len(predicted_path) > 0:
                    if random.random() < 0: # always false
                        # Pick a random point on earth
                        lon = random.uniform(-180, 180)
                        lat = random.uniform(-90, 90)
                        h = 90
                    else:
                        p = predicted_path[-how_many_steps_to_predict]
                        lon, lat, h = p[0], p[1], 2000
                else:
                    lon, lat, h = gymenv.heli.position_as_geo_coordinate[0], gymenv.heli.position_as_geo_coordinate[1], 2000
                print(f"Go to first predicted path point: {lon},{lat},{h}. Press Enter to continue.")
                _ = input()
            # Calculate close_enough based on geo distance to predicted_path[0]
            if init_run:
                close_enough = True
                init_run = False
            else:
                current_pos = (lat, lon)
                predicted_first = (predicted_path[-how_many_steps_to_predict][1], predicted_path[-how_many_steps_to_predict][0])  # (lat, lon)
                distance = geodesic(current_pos, predicted_first).meters
                close_enough = distance <= radius_for_close_enough
            if close_enough:
                # update observed_radiation_map in observation
                if odl > threshold:
                    sbenv.envs[0].unwrapped.heli.observed_radiation_map_points[-1, 2] =  odl
                else:
                    sbenv.envs[0].unwrapped.heli.observed_radiation_map_points[-1, 2] = 0
                obs_gymenv = gymenv._get_obs()
                obs = {k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else v for k, v in obs_gymenv.items()}
                if only_one_distorted_image_in_observation:
                    obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
                else:
                    obs["image_small"] = np.transpose(obs["image_small"], (0, 3, 1, 2))
                    obs["image_medium"] = np.transpose(obs["image_medium"], (0, 3, 1, 2))
                    obs["image_large"] = np.transpose(obs["image_large"], (0, 3, 1, 2))
                # predict next 10 steps assuming odl 0 again -> better: no radiation map points. also make steps with step() (we assume just past seen radiation, so just set current radaition always to 0))
                target_area_geojson_obj = {
                    "type": "Feature",
                    "geometry": mapping(gymenv.heli.target_area),
                    "properties": {}
                }
                sio.emit('target_area_update', {'target_area': target_area_geojson_obj})
                if how_many_steps_to_predict != 1:
                    predicted_path = predicted_path[:-(how_many_steps_to_predict-1)]
                for i in range(how_many_steps_to_predict):
                    action, _states = model.predict(obs, deterministic=True)
                    output_vectorized_env = sbenv.step(action)
                    if i != 0:
                        sbenv.envs[0].unwrapped.heli.observed_radiation_map_points = sbenv.envs[0].unwrapped.heli.observed_radiation_map_points[:-1]
                    obs_gymenv = gymenv._get_obs()
                    obs = {k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else v for k, v in obs_gymenv.items()}
                    if only_one_distorted_image_in_observation:
                        obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
                    else:
                        obs["image_small"] = np.transpose(obs["image_small"], (0, 3, 1, 2))
                        obs["image_medium"] = np.transpose(obs["image_medium"], (0, 3, 1, 2))
                        obs["image_large"] = np.transpose(obs["image_large"], (0, 3, 1, 2))
                    next_pred = [
                        sbenv.envs[0].unwrapped.heli.position_as_geo_coordinate[0],
                        sbenv.envs[0].unwrapped.heli.position_as_geo_coordinate[1],
                    ]
                    predicted_path.append(next_pred)

                    if how_many_steps_to_predict != 1:
                        if i == 0:
                            gymenv_state_after_first_prediction = gymenv.get_state()
                sio.emit('predicted_path_update', {'predicted_path': predicted_path})
                if how_many_steps_to_predict != 1:
                    gymenv.set_state(gymenv_state_after_first_prediction)

            sio.emit('update_data', {'new_point': [lon, lat, h]})
            try:
                time.sleep(update_interval)
            except serial.SerialException as e:
                print(f"Serial exception: {e}")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        except serial.SerialException as e:
            print(f"Serial exception: {e}")
            break

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    sio.connect('http://127.0.0.1:5000')
    os.chdir(base_folder)

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
        model_class = model_classes[parameters['sb3_model_type']]
        model = model_class.load(model_path, env=env, buffer_size=10)
        print(f"Loaded {parameters['sb3_model_type']} model")
        return model

    try:
        # data port for ODL
        if odl_origin == 'yevhen':
            ser_data = serial.Serial("COM14", 4800, timeout=0.2)
        else:
            ser_data = None
        # port for GPS
        if gps_origin == 'serial':
            ser_gps = serial.Serial("COM11", 4800, timeout=0.2)
        elif gps_origin in ('msfs', 'input'):
            ser_gps = serial.Serial("COM9", 4800, timeout=0.2)
        else:
            ser_gps = None
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        if "COM9" in str(e):
            print("Debug: COM9 might be in use or lacks permissions. Check system settings.")
        raise

    model = load_model(model_path, env=VecTransposeImage(DummyVecEnv([lambda: class_gymenv.GymnasiumEnv()])))
    sbenv = model.get_env()
    gymenv = sbenv.envs[0].unwrapped

    import json
    from shapely.geometry import shape

    polygon_path = "C:\\Users\\johan\\programming\\Simulation\\polygon.geojson"
    print("Waiting for user to define polygon in web UI and click Finish Polygon...")
    if os.path.exists(polygon_path):
        os.remove(polygon_path)
    while not os.path.exists(polygon_path):
        time.sleep(1)
    with open(polygon_path, "r", encoding="utf-8") as f:
        polygon_geojson = json.load(f)

    # calc polygon center
    polygon_center_shapely = shape(polygon_geojson['features'][0]['geometry']).centroid
    polygon_center = Point(polygon_center_shapely.y, polygon_center_shapely.x)  # Point(latitude, longitude)
    obs_gymenv, _ = gymenv.reset()
    target_area_polygon_for_reset = shape(polygon_geojson['features'][0]['geometry'])
    start_position_geo_coordinates_scaled_to_grid = compute_position_from_geo_coordinate(
        (list(target_area_polygon_for_reset.exterior.coords)[0][0],list(target_area_polygon_for_reset.exterior.coords)[0][1]),
        anchor_point=gymenv.env_sim.anchor_point,
        granularity=gymenv.env_sim.granularity
    )
    gymenv.heli.update_position(np.array([list(target_area_polygon_for_reset.exterior.coords)[0][0],list(target_area_polygon_for_reset.exterior.coords)[0][1],90]))
    gymenv.heli.target_area = target_area_polygon_for_reset
    gymenv.heli.cone_polygon = Polygon()
    gymenv.heli.polygon_center = polygon_center
    area, border_length = gymenv.heli.geod.geometry_area_perimeter(gymenv.heli.target_area)
    gymenv.heli.target_area_size = abs(area)
    gymenv.heli.target_area_border_length = abs(border_length)
    gymenv.env_sim.max_flight_time = max(timestep*3, episode_length_factor * gymenv.env_sim.timestep * gymenv.heli.target_area_size / area_measurable_per_step)
    obs_gymenv = gymenv._get_obs()
    obs_sbenv = {k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else v for k, v in obs_gymenv.items()}
    if only_one_distorted_image_in_observation:
        obs_sbenv["image"] = np.transpose(obs_sbenv["image"], (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
    else:
        obs_sbenv["image_small"] = np.transpose(obs_sbenv["image_small"], (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
        obs_sbenv["image_medium"] = np.transpose(obs_sbenv["image_medium"], (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
        obs_sbenv["image_large"] = np.transpose(obs_sbenv["image_large"], (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
        
    read_serial_data(model, gymenv, obs_sbenv, sbenv, ser_data, ser_gps)