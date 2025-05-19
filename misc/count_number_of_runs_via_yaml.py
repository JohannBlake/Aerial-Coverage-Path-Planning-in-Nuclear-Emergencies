import importlib
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
from functools import reduce
import operator

def return_num_runs_in_yaml_file():
    with open('parameters_sweep.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Extract the parameter values
    parameters = sweep_config['parameters']

    # Filter parameters to include only those that are dictionaries and have 'values' and are not empty
    filtered_parameters = {k: v for k, v in parameters.items() if isinstance(v, dict) and 'values' in v and v['values']}

    # Calculate the product of the lengths of the value lists
    num_runs = reduce(operator.mul, [len(param['values']) for param in filtered_parameters.values()])
    return num_runs