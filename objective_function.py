from __future__ import annotations
import wandb
# Imports
import importlib
import get_parameters
importlib.reload(get_parameters)
from get_parameters import *
import class_gymenv
importlib.reload(class_gymenv)
from stable_baselines3 import PPO, A2C, TD3, DDPG, SAC, DQN
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os
import importlib
from stable_baselines3.common.monitor import Monitor
from gymnasium.vector import AsyncVectorEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv
import misc.simulate_and_precalculate_radiation_data
import dill
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#import omnisafe
#import omnisafe
#from omnisafe.typing import DEVICE_CPU

policy_type = 'MultiInputPolicy'

# 1: Define objective/training function
def objective():
    # Create training environments
    height_data = np.load('height_data.npy')
    measuring_area_scenarios = np.load(os.path.join('.', 'misc', 'radiation_data', 'simulated', 'measuring_area_scenarios.npy'), allow_pickle=True) 
    scenarios_filename = os.path.join('.', 'misc', 'radiation_data', 'simulated', f'simulated_radiation_scenarios.npy')
    radiation_data = np.load(scenarios_filename)
    scenarios_geo_coords = np.load(scenarios_filename + '_geo_coords' + '.npy' , allow_pickle=True)

    envs = [lambda: Monitor(class_gymenv.GymnasiumEnv(height_data, measuring_area_scenarios, radiation_data, scenarios_geo_coords, test_env = False)) for _ in range(num_envs_for_training)]
    gymenv = SubprocVecEnv(envs)  # Changed from DummyVecEnv to SubprocVecEnv
    gymenv = VecTransposeImage(gymenv)

    # Create evaluation environments
    envs_eval = [lambda: Monitor(class_gymenv.GymnasiumEnv(height_data, measuring_area_scenarios, radiation_data, scenarios_geo_coords, test_env = True)) for _ in range(num_envs_for_training)]
    gymenv_eval = SubprocVecEnv(envs_eval)  # Changed from DummyVecEnv to SubprocVecEnv
    gymenv_eval = VecTransposeImage(gymenv_eval) 
    directory_path = f'./logs/{wandb.run.name}'
    file_path = os.path.join(directory_path, 'gymenv.pkl')
    os.makedirs(directory_path, exist_ok=True)
    #with open(file_path, 'wb') as file:
    #    dill.dump(gymenv, file)
    if training_library == 'sb3':
        n_actions = gymenv.action_space.shape[-1]
        mean = np.zeros(n_actions)
        sigma = sigma_coef_for_noise * np.ones(n_actions)

        action_noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

        if sb3_model_type == 'PPO':
            model = PPO(policy_type, gymenv, verbose=0,
                        learning_rate=learning_rate,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        gamma=gamma,
                        clip_range=clip_range,
                        max_grad_norm = 0.3,
                        ent_coef=ent_coef)
        elif sb3_model_type == 'A2C':
            model = A2C(policy_type, gymenv, verbose=0,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        n_steps=n_steps,
                        ent_coef=ent_coef)
        elif sb3_model_type == 'TD3':
            model = TD3(policy_type, gymenv, verbose=0,
                        learning_rate=learning_rate_td3,
                        buffer_size=buffer_size,
                        learning_starts=learning_starts,
                        batch_size=batch_size,
                        gamma=gamma,
                        train_freq=(train_freq_td3, "step"),
                        gradient_steps=gradient_steps_td3,
                        action_noise=action_noise if include_action_noise else None
                        )
        elif sb3_model_type == 'DDPG':
            model = DDPG(policy_type, gymenv, verbose=0,
                        learning_rate=learning_rate,
                        learning_starts=learning_starts,
                        buffer_size=buffer_size,
                        gamma=gamma,
                        action_noise=action_noise if include_action_noise else None
                        )
        elif sb3_model_type == 'DQN':
            model = DQN(policy_type, gymenv, verbose=0,
                        learning_rate=learning_rate,
                        buffer_size=buffer_size,
                        learning_starts=learning_starts,
                        gamma=gamma,
                        train_freq=train_freq)
        elif sb3_model_type == 'TRPO':
            model = TRPO(policy_type, gymenv, verbose=0,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        )
        elif sb3_model_type == 'SAC':
            model = SAC(policy_type, gymenv, verbose=0,
                        learning_rate=learning_rate_sac,
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        learning_starts=learning_starts,
                        gamma=gamma,
                        train_freq=train_freq_sac,
                        gradient_steps=gradient_steps_sac,
                        action_noise=action_noise if include_action_noise else None
                        )
        else:
            raise ValueError("Unsupported model type")
        class WandbEvalCallback(EvalCallback):
            def __init__(self, *args, **kwargs):
                super(WandbEvalCallback, self).__init__(*args, **kwargs)

            def _on_step(self) -> bool:
                # Log custom metrics from the environment
                if self.num_timesteps % self.eval_freq == 0:
                    # Log custom metrics from the environment's info dictionary
                    wandb.log(gymenv.get_attr('info', 0)[0]['data_at_end_of_preceding_episode'])
                return super(WandbEvalCallback, self)._on_step()
        eval_callback = WandbEvalCallback(
            eval_env=gymenv_eval,
            best_model_save_path=directory_path,
            eval_freq=logging_frequency,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        class CalculateMeanReward(BaseCallback):
            def __init__(self, verbose=0):
                super(CalculateMeanReward, self).__init__(verbose)
                self.total_rewards = []
                self.mean_total_reward_of_last_10_episodes = 0  # Initialize the mean reward attribute

            def _on_step(self) -> bool:
                if 'episode' in self.locals['infos'][0]:
                    self.total_rewards.append(self.locals['infos'][0]['episode']['r'])
                return True

            def _on_training_end(self) -> None:
                if len(self.total_rewards) >= 10:
                    self.mean_total_reward_of_last_10_episodes = sum(self.total_rewards[-10:]) / 10
                else:
                    self.mean_total_reward_of_last_10_episodes = sum(self.total_rewards) / len(self.total_rewards)
                print("Mean Total Reward of Last 10 Episodes:", self.mean_total_reward_of_last_10_episodes)
                wandb.log({'mean_total_reward_of_last_10_episodes': self.mean_total_reward_of_last_10_episodes})

            def get_mean_total_reward_of_last_10_episodes(self):
                return self.mean_total_reward_of_last_10_episodes
        calculate_mean_reward_callback = CalculateMeanReward()
        model.learn(
            total_timesteps = total_timesteps_for_training
            ,callback = [eval_callback]
            )
        mean_total_reward_of_last_10_episodes = calculate_mean_reward_callback.get_mean_total_reward_of_last_10_episodes()
    if training_library == 'omnisafe':
        custom_cfgs = {
            'train_cfgs': {
                'total_steps': 1000,
            },
            'algo_cfgs': {
                'steps_per_epoch': 400,
                'update_iters': 1,
                'gamma': 0.98,
                'start_learning_steps': 100,
                'exploration_noise': 0.,
            }
        }
        agent = omnisafe.Agent('DDPG', env_id, custom_cfgs=custom_cfgs)
        agent.learn()
    return mean_total_reward_of_last_10_episodes