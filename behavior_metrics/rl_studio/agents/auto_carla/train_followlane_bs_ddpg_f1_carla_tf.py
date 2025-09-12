import logging
from datetime import datetime, timedelta
import glob
import time
import psutil
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import BasePolicy

from gym import spaces
from typing import Callable, Tuple

import torch as th
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
import tensorflow as tf
from tqdm import tqdm

from stable_baselines3.common.callbacks import CallbackList

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
# from wandb.integration.sb3 import WandbCallback
# import wandb



from stable_baselines3 import DDPG
# from stable_baselines3 import DDPG
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure



# Function to update scatter plot with new data
def update_scatter_plot(ax, x, y, z, xlabel, ylabel, zlabel):
    ax.clear()
    ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.draw()
    plt.pause(0.001)


def collect_usage():
    cpu_usage = psutil.cpu_percent(interval=None)  # Get CPU usage percentage
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_usage = gpu_info.gpu
    return cpu_usage, gpu_usage


def combine_attributes(obj1, obj2, obj3):
    combined_dict = {}

    # Extract attributes from obj1
    obj1_dict = obj1.__dict__
    for key, value in obj1_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj2
    obj2_dict = obj2.__dict__
    for key, value in obj2_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj3
    obj3_dict = obj3.__dict__
    for key, value in obj3_dict.items():
        combined_dict[key] = value

    return combined_dict

def new_lr_schedule(progress_remaining):
    return 0.00003

class PeriodicSaveCallback(BaseCallback):
    def __init__(self, env, params, save_path, save_freq=10000, verbose=1):
        super(PeriodicSaveCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.step_count = 0
        self.env = env
        self.params = params

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            model_save_path = os.path.join(self.save_path, f"model_{self.step_count}_steps")
            self.model.save(model_save_path)
            date_time = time.strftime('%Y%m%d-%H%M%S')

            mlflow.set_experiment("followlane_carla")
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_type", "ddpg_bs")
                mlflow.log_metric("avg_speed", self.env.last_avg_speed)
                mlflow.log_metric("max_speed", self.env.last_max_speed)
                mlflow.log_metric("deviation", self.env.episode_last_d_deviation)
                mlflow.log_metric("cum_reward", self.env.last_cum_reward)
                mlflow.set_tag("detection_mode", self.params["detection_mode"])
                mlflow.log_param("actions", self.params["actions"])
                mlflow.log_param("zig_zag_punish", self.params["zig_zag_punish"])
                mlflow.set_tag("running_mode", self.params["running_mode"])
                mlflow.log_param("datetime", date_time)
                mlflow.log_metric("last_episode_steps", self.env.last_steps)
                mlflow.log_metric("steps", self.step_count)
                mlflow.log_artifact(model_save_path + ".zip", artifact_path="saved_models")
            # mlflow.log_metric("gamma", self.env.episode_d_deviation)
            # mlflow.log_metric("tau", self.env.episode_d_deviation)
            # mlflow.log_metric("lr", self.env.episode_d_deviation)
            # mlflow.log_metric("d_importance", self.env.episode_d_deviation)
            # mlflow.log_metric("v_importance", self.env.episode_d_deviation)
            # mlflow.log_metric("high_vel_punish", self.env.episode_d_deviation)
            if self.verbose > 0:
                print(f"Saved model at step {self.step_count}")
        return True


class ExplorationRateCallback(BaseCallback):
    def __init__(self, tensorboard, stage=None, initial_exploration_rate=1.0,
                 exploration_min=0.03, decay_rate=0.005, decay_steps=1000, verbose=1):
        super(ExplorationRateCallback, self).__init__(verbose)
        self.min_exploration_rate = exploration_min
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0
        self.tensorboard = tensorboard
        self.iters_w_on_min = 0
        self.stage = stage
        # Configure noise rates based on stage
        if stage == "w":
            self.w_initial = 0.05
            self.v_initial = 0
        elif stage == "v":
            self.w_initial = 0
            self.v_initial = initial_exploration_rate
        elif stage == "r":
            self.w_initial = 0.1
            self.v_initial = 0.4
        self.w_exploration_rate = self.w_initial
        self.v_exploration_rate = self.v_initial
        self.n_actions = None  # Will be initialized at training start

    def _on_training_start(self):
        # Initialize number of actions and set initial action noise
        self.n_actions = self.model.action_space.shape[0]

        self.model.action_noise = NormalActionNoise(
            mean=np.zeros(self.n_actions),
            sigma=np.array([self.v_exploration_rate] + [self.w_exploration_rate])
        )
        self.tensorboard.update_stats(std_dev_v=self.v_exploration_rate)
        self.tensorboard.update_stats(std_dev_w=self.w_exploration_rate)

        if self.verbose > 0:
            print(f"Training started: Initial exploration rates set to v={self.v_exploration_rate}, w={self.w_exploration_rate}")
        return True

    def _on_step(self) -> bool:
        self.current_step += 1

        if self.current_step % self.decay_steps == 0:
            # Decay exploration rates for first and subsequent actions
            self.v_exploration_rate = max(self.min_exploration_rate,
                                          self.v_exploration_rate - self.decay_rate)
            self.w_exploration_rate = max(self.min_exploration_rate,
                                          self.w_exploration_rate - self.decay_rate)

            if self.w_exploration_rate == self.min_exploration_rate:
                self.iters_w_on_min +=1
            if self.iters_w_on_min == 50 and self.stage == "w":
                self.iters_w_on_min = 0
                self.w_exploration_rate = 0.02

            # Update the action noise with the new exploration rates
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(self.n_actions),
                sigma=np.array([self.v_exploration_rate] + [self.w_exploration_rate])
            )
            self.tensorboard.update_stats(std_dev_v=self.v_exploration_rate)
            self.tensorboard.update_stats(std_dev_w=self.w_exploration_rate)

        return True

from stable_baselines3.common.buffers import ReplayBuffer
class CustomReplayBuffer(ReplayBuffer):
    def add(self, obs,next_obs, action, reward , done, infos=None):
        action = np.array(action, copy=True)  # Ensure no inplace modification issues
        action[0][0] = 0.7  # Force action[0] to always be 0.8
        super().add(obs, next_obs, action, reward, done, infos)

import torch as th
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update

class CustomTD3(TD3):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actions = self.actor(replay_data.observations)

                # 🛑 Freeze learning for action[1] by detaching it
                actions = th.cat([actions[:, :1], actions[:, 1:2].detach(), actions[:, 2:]], dim=1)

                actor_loss = -self.critic.q1_forward(replay_data.observations, actions).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

class TrainerFollowLaneDDPGCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DDPG
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):

        pynvml.nvmlInit()

        self.actor_loss = 0
        self.critic_loss = 0
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.global_params = LoadGlobalParams(config)
        self.environment = LoadEnvVariablesDDPGCarla(config)
        self.environment.environment["debug_waypoints"] = False
        logs_dir = f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard = ModifiedTensorBoard(
            log_dir=logs_dir
        )
        self.environment.environment["tensorboard"] = self.tensorboard

        self.loss = 0

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        agent_log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_baselines_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        # self.log = LoggingHandler(agent_log_file)

        ## Load Carla server
        # CarlaEnv.__init__(self)
        self.environment.environment["actions"] = self.global_params.actions_set

        self.environment.environment["entropy_factor"] = config["settings"]["entropy_factor"]
        self.environment.environment["use_curves_state"] = config["settings"]["use_curves_state"]
        self.env = gym.make(self.env_params.env_name, **self.environment.environment)
        self.all_steps = 0
        self.current_max_reward = 0
        self.best_epoch = 0
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_steer = []
        self.episodes_reward = []
        self.step_fps = []
        self.bad_perceptions = 0
        self.crash = 0

        self.all_steps_reward = []
        self.all_steps_velocity = []
        self.all_steps_steer = []
        self.all_steps_state0 = []
        self.all_steps_state11 = []

        self.cpu_usages = 0
        self.gpu_usages = 0

        self.exploration = self.algoritmhs_params.std_dev if self.global_params.mode != "inference" else 0
        self.n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=self.exploration * np.ones(self.n_actions))


        self.params = {
            "policy": "CustomPolicy",
            "learning_rate": 0.00035,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.90,
            "tau": 0.005,
            "total_timesteps": 5000000
        }

        # Init Agents
        if self.environment.environment["mode"] in ["inference", "retraining"]:
            actor_retrained_model = self.environment.environment['retrain_ddpg_tf_model_name']
            if self.environment.environment.get("stage") == "v":
                self.ddpg_agent = CustomTD3.load(actor_retrained_model)
            else:
                self.ddpg_agent = DDPG.load(actor_retrained_model)
            # Set the environment on the loaded model
            self.ddpg_agent.set_env(self.env)
        else:
            # Assuming `self.params` and `self.global_params` are defined properly
            self.ddpg_agent = DDPG(
                # CustomDDPGPolicy,
                "MlpPolicy",
                self.env,
                policy_kwargs=dict(net_arch=dict(pi=[32, 32, 32], qf=[32, 32, 32])),
                learning_rate=self.params["learning_rate"],
                buffer_size=self.params["buffer_size"],
                batch_size=self.params["batch_size"],
                tau=self.params["tau"],
                gamma=self.params["gamma"],
                verbose=1,
                # tensorboard_log=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
            )

        print(self.ddpg_agent.actor)
        # Set the action noise on the loaded model
        self.ddpg_agent.action_noise = action_noise

        agent_logger = configure(agent_log_file, ["stdout", "csv", "tensorboard"])

        self.ddpg_agent.set_logger(agent_logger)
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

    def main(self):
        # run = wandb.init(
        #     project="rl-follow-lane",
        #     config=self.params,
        #     sync_tensorboard=True,
        # )
        exploration_rate_callback = ExplorationRateCallback(self.tensorboard,
                                                            stage=self.environment.environment.get("stage"),
                                                            initial_exploration_rate=self.exploration,
                                                            decay_rate= self.global_params.decrease_substraction,
                                                            decay_steps=self.global_params.steps_to_decrease,
                                                            exploration_min=self.global_params.decrease_min,
                                                            verbose=1)

        # Assuming `self.env` is your original environment
        self.eval_env = Monitor(self.env)

        # TODO Note that evalCallback is useful, but it slow down training getting stucked. To refine
        # wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            eval_freq=100000,
            deterministic=True,
            render=False
        )

        params = {
            "detection_mode": self.environment.environment["detection_mode"],
            "actions": self.global_params.actions_set,
            "zig_zag_punish": self.environment.environment["punish_zig_zag_value"],
            "running_mode": self.environment.environment["mode"],
        }
        periodic_save_callback = PeriodicSaveCallback(
            env = self.env,
            params = params,
            save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            verbose=1
        )

        callback_list = CallbackList([exploration_rate_callback, eval_callback, periodic_save_callback])
       # callback_list = CallbackList([exploration_rate_callback, periodic_save_callback])

        if self.environment.environment["mode"] == "inference":
            self.evaluate_ddpg_agent(self.env, self.ddpg_agent, 10000)

       # Apply the new learning rate schedule
        self.ddpg_agent.learning_rate = new_lr_schedule

        if self.environment.environment.get("stage") == "w":
            self.ddpg_agent.replay_buffer = CustomReplayBuffer(
                self.ddpg_agent.buffer_size,
                self.ddpg_agent.observation_space,
                self.ddpg_agent.action_space,
                self.ddpg_agent.device,
                self.ddpg_agent.n_envs,
                True,
                False,
            )

        self.ddpg_agent.learn(total_timesteps=self.params["total_timesteps"],
                              callback=callback_list)

    def evaluate_ddpg_agent(self, env, agent, num_episodes):
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Use the agent to predict the action without learning (no training)
                action, _states = agent.predict(obs, deterministic=True)

                # Take the action in the environment
                obs, reward, done, done, info = env.step(action)
                episode_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        # self.env.close()

    def set_stats(self, info, prev_state):
        self.episodes_speed.append(info["velocity"])
        self.episodes_steer.append(info["steering_angle"])
        self.episodes_d_reward.append(info["d_reward"])
        self.episodes_reward.append(info["reward"])

        self.all_steps_reward.append(info["reward"])
        self.all_steps_velocity.append(info["velocity"])
        self.all_steps_steer.append(info["steering_angle"])
        self.all_steps_state0.append(prev_state[0])
        self.all_steps_state11.append(prev_state[6])

        pass