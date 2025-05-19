import numpy as np
import yaml
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parameters_default.yaml file
yaml_file_path = os.path.join(script_dir, 'parameters_default.yaml')

# Open and read the YAML file
with open(yaml_file_path, 'r') as file:
    parameters = yaml.safe_load(file)

current_dir = os.getcwd()

if current_dir.startswith('/dss'):
    cluster = 'lrz'
elif current_dir.startswith('/home/stud'):
    cluster = 'lmu'
else:
    cluster = 'unknown'
# Overwrite num_scenarios if we are in lrz or lmu cluster
if cluster in ['lrz', 'lmu']:
    parameters['num_scenarios'] = 20000
    print(f"Overwriting num_scenarios to {parameters['num_scenarios']} for cluster {cluster}")

for key, value in parameters.items():
    globals()[key] = value