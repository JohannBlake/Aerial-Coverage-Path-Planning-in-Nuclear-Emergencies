import os
import sys
import importlib

# Add the parent directory of the script to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
from scipy.ndimage import label, sum as ndsum
from collections import deque

class sim:
    def __init__(self, env, heli):
        self.env = env
        self.heli = heli
        self.amount_of_steps_ignored = 0
        self.log_total_amount_of_steps_ignored = 0
        self.amount_of_new_measured_points = 0
        self.amount_of_remeasured_points = 0
    def reset(self):
        self.log_total_amount_of_steps_ignored = 0
        self.amount_of_steps_ignored = 0
        self.amount_of_new_measured_points = 0
        self.amount_of_remeasured_points = 0
    def step(self, action_vector): # Update sim state
        action_vector = self.map_gymenv_action_to_sim_action(action_vector)
        self.time_rel_before_executing_action = self.env.time_rel
        self.heli.step(action_vector)
        self.env.step()
        if self.env.percentage_of_episode_finished_by_time_passed >= 1:
            self.env.end_sim = True
            self.env.end_sim_reason = 'Time limit reached'

    def map_gymenv_action_to_sim_action(self, action_vector):
        """
        Maps the gym env action vector to the sim action vector.
        
        :param action_vector: The gym env action vector.
        """
        if action_type == 'ta':
            action_vector = np.array([action_vector[0]* largest_horizontal_turning_angle_deg, action_vector[1] * largest_vertical_height_change_per_second])
        elif action_type == 'ta_with_time_in_between_actions':
            action_vector = np.array([action_vector[0]* largest_horizontal_turning_angle_deg, action_vector[1] * largest_vertical_height_change_per_second, action_vector[2]])
        elif action_type == 'ta without height change':
            action_vector = np.array([action_vector[0]* largest_horizontal_turning_angle_deg])
        elif action_type == 'ta without height change with termination option':
            action_vector = np.array([action_vector[0]* largest_horizontal_turning_angle_deg])
        return action_vector