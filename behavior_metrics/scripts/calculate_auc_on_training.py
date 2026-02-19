import numpy as np
import os
import tensorflow as tf  # Required for converting tensor_proto to numpy array
from scipy.integrate import trapz
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import List, Tuple, Union, Optional, Dict


# ----------------------------------------------------------------------
# Core Functions for AUC Calculation
# ----------------------------------------------------------------------

def find_best_log_dir(start_dir: str, required_tag: str) -> Optional[str]:
    """
    Recursively searches for a TensorBoard log directory that contains
    an event file AND contains the required scalar or tensor tag.

    Args:
        start_dir: The path to the starting directory (e.g., '/ppo/overall').
        required_tag: The scalar or tensor tag we must find (e.g., 'cum_rewards').

    Returns:
        The path to the log directory containing the required data, or None.
    """
    checked_dirs = set()

    # If the user passed the full file path, we just use its directory
    if os.path.isfile(start_dir) and 'tfevents' in os.path.basename(start_dir):
        start_dir = os.path.dirname(start_dir)

    for root, _, files in os.walk(start_dir):
        log_dir = root

        # Check if we have already processed this directory
        if log_dir in checked_dirs:
            continue

        # Check if any file in the current directory looks like a TensorBoard event file
        if any('tfevents' in file for file in files):
            checked_dirs.add(log_dir)

            try:
                # Initialize EventAccumulator with the directory containing the tfevents file
                ea = EventAccumulator(log_dir, size_guidance={'scalars': 0, 'tensors': 0})  # Ensure tensors are loaded
                ea.Reload()

                # Check if the required tag is present in SCALARS or TENSORS
                if required_tag in ea.Tags().get('scalars', []) or required_tag in ea.Tags().get('tensors', []):
                    return log_dir  # Found the correct log directory
            except Exception as e:
                # Ignore logs that fail to load
                # print(f"Skipping log directory {log_dir} due to error: {e}")
                continue

    return None


def extract_scalar_data(log_dir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts step numbers (x-axis) and values (y-axis) for a given tag,
    checking first for Scalars and then for Tensors.

    Args:
        log_dir: The path to the TensorBoard log directory (or a parent directory).
        tag: The scalar or tensor tag name (e.g., 'cum_rewards').

    Returns:
        A tuple of numpy arrays: (steps, values).
    """
    # 1. Find the actual log directory that contains the required data
    actual_log_dir = find_best_log_dir(log_dir, tag)

    if not actual_log_dir:
        # If the required tag is not found anywhere in the tree, raise an error
        raise KeyError(f"Key '{tag}' was not found in any valid log file within the path {log_dir}.")

    # 2. Initialize and reload the EventAccumulator using the found directory
    ea = EventAccumulator(actual_log_dir, size_guidance={'scalars': 0, 'tensors': 0})
    ea.Reload()

    all_steps = []
    all_values = []

    # --- TENSOR LOGIC (NEW) ---
    if tag in ea.Tags().get('tensors', []):
        tensors = ea.Tensors(tag)

        for event in tensors:
            all_steps.append(event.step)
            # Use tf.make_ndarray to convert the tensor_proto to a numpy array
            tensor_values = tf.make_ndarray(event.tensor_proto)

            # Assuming 'cum_rewards' is a single value tensor, extract the single float
            # If the tensor is an array, you might need to adjust this indexing (e.g., tensor_values[0])
            all_values.append(tensor_values.item())

            # Separate steps and values into NumPy arrays
        steps = np.array(all_steps)
        values = np.array(all_values)
        return steps, values

    # --- SCALAR LOGIC (ORIGINAL) ---
    elif tag in ea.Tags().get('scalars', []):
        scalars = ea.Scalars(tag)

        # Separate steps and values into NumPy arrays
        steps = np.array([s.step for s in scalars])
        values = np.array([s.value for s in scalars])

        return steps, values

    # --- ERROR HANDLING ---
    else:
        available_scalar_tags = ea.Tags().get('scalars', [])
        available_tensor_tags = ea.Tags().get('tensors', [])
        raise KeyError(
            f"Key '{tag}' not found. Available SCALAR tags: {available_scalar_tags}. Available TENSOR tags: {available_tensor_tags}"
        )


def calculate_auc(log_dir: str, tag: str = 'cum_rewards') -> float:
    """
    Calculates the Area Under the Curve (AUC) for a scalar metric
    (e.g., cumulative rewards) with respect to the training steps.

    Args:
        log_dir: The path to the TensorBoard log directory.
        tag: The scalar tag name (default is 'cum_rewards').

    Returns:
        The calculated AUC score (float).
    """
    try:
        steps, values = extract_scalar_data(log_dir, tag)
    except KeyError as e:
        print(f"Error: {e}")
        return 0.0

    if len(steps) < 2:
        print("Warning: Insufficient data points (less than 2) to calculate AUC.")
        return 0.0

    # Use the trapezoidal rule for numerical integration:
    # AUC = sum((y_i + y_{i+1})/2 * (x_{i+1} - x_i))
    auc_score = trapz(values, steps)

    return auc_score


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------

def launch_calculation(log_path, metric_tag):
    # --- Print available tags (Optional, but helpful for debugging) ---
    # Find the actual directory containing the log file (for tag printing)
    actual_log_dir = find_best_log_dir(log_path, metric_tag)

    try:
        if actual_log_dir:
            # Re-initialize the accumulator for displaying available tags
            ea = EventAccumulator(actual_log_dir, size_guidance={'scalars': 0, 'tensors': 0})
            ea.Reload()

            # Print the directory name where the tfevents file was actually found
            print(f"--- Available Tags in {os.path.basename(actual_log_dir)} ---")
            print(f"SCALAR Tags: {ea.Tags().get('scalars', [])}")
            print(f"TENSOR Tags: {ea.Tags().get('tensors', [])}")
        else:
            print(f"--- No valid log file containing tag '{metric_tag}' found in {log_path} ---")

        print("-" * 40)
    except Exception as e:
        print(f"Could not load accumulator to check tags: {e}")

    final_auc = calculate_auc(log_path, metric_tag)

    # --- Calculate and print AUC ---
    print(f"\nMetric Tag: '{metric_tag}'")
    return final_auc


if __name__ == '__main__':
    # REPLACE THIS WITH THE ACTUAL PATH TO YOUR TENSORBOARD LOG DIRECTORY
    # This path should contain the 'events.out.tfevents.xxx' file.
    # print("calculating td3")
    # log_path = '/home/ruben/Desktop/2020-phd-ruben-lucas/src/RL-Studio/rl_studio/logs/training/follow_lane_carla_td3_auto_carla_baselines/logs/20251112-194934_baselines_training_follow_lane_carla_td3_auto_carla_baselines.log'
    # metric_tag = 'rollout/ep_rew_mean'
    # td3 = launch_calculation(log_path, metric_tag)
    #
    # print("calculating sac")
    # log_path = '/home/ruben/Desktop/2020-phd-ruben-lucas/src/RL-Studio/rl_studio/logs/training/follow_lane_carla_sac_auto_carla_baselines/logs/20251031-092737_baselines_training_follow_lane_carla_sac_auto_carla_baselines.log'
    # metric_tag = 'eval/mean_reward'
    # sac = launch_calculation(log_path, metric_tag)

    print("calculating ppo")
    # Note: Using the directory path is generally better than the full file path for robustness
    log_path = '/home/ruben/Desktop/2020-phd-ruben-lucas/src/RL-Studio/rl_studio/logs/training/follow_lane_carla_ppo_continuous_auto_carla_baselines/TensorBoard/20251103-112559/ppo/overall'
    metric_tag = 'cum_rewards'
    ppo = launch_calculation(log_path, metric_tag)

    # print("calculating ddpg")
    # log_path = '/home/ruben/Desktop/2020-phd-ruben-lucas/src/RL-Studio/rl_studio/logs/training/follow_lane_carla_ddpg_auto_carla_baselines/logs/20250925-090229_baselines_training_follow_lane_carla_ddpg_auto_carla_baselines.log'
    # metric_tag = 'rollout/ep_rew_mean'
    # ddpg = launch_calculation(log_path, metric_tag)

    print("AUC results:")
    # print(f"SAC: {sac:.2f}")
    print(f"PPO: {ppo:.2f}")
    # print(f"TD3: {td3:.2f}")
    # print(f"DDPG: {ddpg:.2f}")