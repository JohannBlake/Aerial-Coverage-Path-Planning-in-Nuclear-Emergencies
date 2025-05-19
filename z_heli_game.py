import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
import pygame
import importlib
import visualization
importlib.reload(visualization)
from visualization import append_output_gymenv_values, append_reward_bar_to_image
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
import class_gymenv
importlib.reload(class_gymenv)
from class_gymenv import *
import misc.simulate_and_precalculate_radiation_data

height_data = np.load('height_data.npy')
measuring_area_scenarios = np.load(os.path.join('.', 'misc', 'radiation_data', 'simulated', 'measuring_area_scenarios.npy'), allow_pickle=True) 
scenarios_filename = os.path.join('.', 'misc', 'radiation_data', 'simulated', f'simulated_radiation_scenarios.npy')
radiation_data = np.load(scenarios_filename)
scenarios_geo_coords = np.load(scenarios_filename + '_geo_coords' + '.npy' , allow_pickle=True)

# Control variables
vis_2d_only_termination_state = False
control_interval_ms = 150  # Interval for control actions in milliseconds
action_step_size    = 0.25 # Step size for actions

metric_data = {}
gymenv      = class_gymenv.GymnasiumEnv(height_data, measuring_area_scenarios, radiation_data, scenarios_geo_coords)
metric_data = {}
gymenv.reset()
terminated = False
action = [0, 0]
def ensure_three_channels(image):
    if image.shape[-1] == 2:  # If the image has 2 channels
        return np.stack((image[:, :, 0], image[:, :, 1], np.zeros_like(image[:, :, 0])), axis=-1)
    return image  # If already 3 channels, return as is

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption("Game")
window_width, window_height = screen.get_size()

# Function to render metrics as text on the Pygame window
def render_metrics(screen, metric_data, image_width, window_height):
    font_size = max(12, window_height // 25)  # Adjust font size based on window height
    font = pygame.font.SysFont(None, font_size)
    y_offset = 300
    for key, value in metric_data.items():
        text = f'{key}: {value[-1] if isinstance(value, list) else value}'
        text_surface = font.render(text, True, (30, 10, 255))
        screen.blit(text_surface, (image_width + 10, y_offset))  # Display text to the right of the image
        y_offset += (font_size + 5)/2

print("Starting game.")
running = True
paused = False
last_control_time = pygame.time.get_ticks()
keys_held = {'a': 0, 'd': 0, 'w': 0, 's': 0}

while running:
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                keys_held['a'] = current_time
            elif event.key == pygame.K_d:
                keys_held['d'] = current_time
            elif event.key == pygame.K_w:
                keys_held['w'] = current_time
            elif event.key == pygame.K_s:
                keys_held['s'] = current_time
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:  # Press '+' to increase control interval
                control_interval_ms -= 20
            elif event.key == pygame.K_MINUS:  # Press '-' to decrease control interval
                control_interval_ms += 20
            elif event.key == pygame.K_q:  # Press 'q' to quit the game loop
                running = False
            elif event.key == pygame.K_SPACE:  # Press 'space' to pause/resume
                paused = not paused
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                keys_held['a'] = 0
            elif event.key == pygame.K_d:
                keys_held['d'] = 0
            elif event.key == pygame.K_w:
                keys_held['w'] = 0
            elif event.key == pygame.K_s:
                keys_held['s'] = 0
            elif event.key == pygame.K_q:  # Press 'q' to quit the game loop
                running = False
            elif event.key == pygame.K_r:  # Press 'r' to reset the environment
                gymenv.reset()
        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            window_width, window_height = event.w, event.h

    if not paused and current_time - last_control_time >= control_interval_ms:
        if keys_held['a']:
            duration_held = current_time - keys_held['a']
            action[0] += (duration_held / control_interval_ms) * action_step_size
        if keys_held['d']:
            duration_held = current_time - keys_held['d']
            action[0] -= (duration_held / control_interval_ms) * action_step_size
        if keys_held['w']:
            duration_held = current_time - keys_held['w']
            action[1] += (duration_held / control_interval_ms) * action_step_size
        if keys_held['s']:
            duration_held = current_time - keys_held['s']
            action[1] -= (duration_held / control_interval_ms) * action_step_size

        gymenv.step(action)
        output = gymenv._get_obs(), gymenv._calculate_reward(action), gymenv.env_sim.end_sim, False, gymenv.info

        if only_one_distorted_image_in_observation:
            # Convert the image to 3 channels if it has 2
            image = ensure_three_channels(output[0]['image'])
            image = append_reward_bar_to_image(image=image, reward=output[1])
        else:
            # Convert all images to 3 channels if they have 2
            image_small = ensure_three_channels(output[0]['image_small'])
            image_medium = ensure_three_channels(output[0]['image_medium'])
            image_large = ensure_three_channels(output[0]['image_large'])

            image_small = append_reward_bar_to_image(image=image_small, reward=output[1])
        white_line = np.ones((1, 1, 3), dtype=np.uint8) * 255
        terminated = output[2]
        if terminated:
            gymenv.reset()
        if (vis_2d_only_termination_state and terminated) or not vis_2d_only_termination_state:
            white_line = np.ones((output[0]['image_small' if not only_one_distorted_image_in_observation else 'image'].shape[0], 1, 3), dtype=np.uint8) * 255

            # Append reward bar on the relevant image
            if only_one_distorted_image_in_observation:
                image = append_reward_bar_to_image(image=image, reward=output[1])
            else:
                image_small = append_reward_bar_to_image(image=image_small, reward=output[1])
            metric_data = append_output_gymenv_values(metric_data, gymenv)
            last_elements_of_metric_data = {key: value[-1] for key, value in metric_data.items()}

            screen.fill((0, 0, 0))

            if only_one_distorted_image_in_observation:
                transposed_image = np.transpose(image, (1, 0, 2))
                max_dim = max(transposed_image.shape[:2])
                square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                square_image[:transposed_image.shape[0], :transposed_image.shape[1]] = transposed_image

                aspect_ratio = transposed_image.shape[0] / transposed_image.shape[1]
                if window_width / window_height > aspect_ratio:
                    new_width = int(window_height * aspect_ratio)
                    new_height = window_height
                else:
                    new_width = window_width
                    new_height = int(window_width / aspect_ratio)

                # Scale the image to fit into the window while preserving aspect ratio.
                scaled_image = pygame.transform.scale(
                    pygame.surfarray.make_surface(square_image), (new_width, new_height)
                )

                screen.blit(scaled_image, (0, 0))
            else:
                transposed_image_small = np.transpose(image_small, (1, 0, 2))
                transposed_image_medium = np.transpose(image_medium, (1, 0, 2))
                transposed_image_large = np.transpose(image_large, (1, 0, 2))

                max_dim = max(transposed_image_small.shape[:2])
                square_image_small = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                square_image_small[:transposed_image_small.shape[0], :transposed_image_small.shape[1]] = transposed_image_small

                square_image_medium = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                square_image_medium[:transposed_image_medium.shape[0], :transposed_image_medium.shape[1]] = transposed_image_medium

                square_image_large = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                square_image_large[:transposed_image_large.shape[0], :transposed_image_large.shape[1]] = transposed_image_large

                aspect_ratio = transposed_image_small.shape[0] / transposed_image_small.shape[1]
                if window_width / window_height > aspect_ratio:
                    new_width = int(window_height * aspect_ratio)
                    new_height = window_height
                else:
                    new_width = window_width
                    new_height = int(window_width / aspect_ratio)

                total_width = window_width - 2
                image_width = total_width // 3
                image_height = int(image_width / aspect_ratio)

                scaled_image_small = pygame.transform.scale(
                    pygame.surfarray.make_surface(square_image_small), (image_width, image_height)
                )
                scaled_image_medium = pygame.transform.scale(
                    pygame.surfarray.make_surface(square_image_medium), (image_width, image_height)
                )
                scaled_image_large = pygame.transform.scale(
                    pygame.surfarray.make_surface(square_image_large), (image_width, image_height)
                )

                screen.blit(scaled_image_small, (0, 0))
                screen.blit(scaled_image_medium, (scaled_image_small.get_width(), 0))
                screen.blit(scaled_image_large, (scaled_image_small.get_width() + scaled_image_medium.get_width(), 0))

            render_metrics(screen, last_elements_of_metric_data, 0, window_height)
            pygame.display.flip()

        action = [0, 0]
        last_control_time = current_time
pygame.quit()