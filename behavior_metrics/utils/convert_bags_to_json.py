
import os
import json
import glob
import pandas as pd
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import metrics_carla
from utils.configuration import Config as Configuration

# A simple mock for carla.Location, Transform and Waypoint to be used in offline mode.
class MockLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.location = MockLocation(x, y, z)

class MockWaypoint:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.transform = MockTransform(x, y, z)

# Hardcoded list of folders from the user
BROKEN_EXPERIMENTS = [
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-134910_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-135503_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-140004_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-140538_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-141145_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-141712_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-142317_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-142847_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-143452_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-144044_brain_carla_sb_sac_01.py_town_06_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-191916_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-192205_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-192407_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-192717_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-193113_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-193433_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-193632_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-194019_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-194217_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
    "/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/2025_10_20_ablation/20251005-194535_brain_carla_sb_sac_01.py_town_10_clockwise_ruben.launch",
]

def get_map_waypoints(map_name):
    """
    Loads waypoints from a CSV file based on the map name.
    """
    map_number_str = ''.join(filter(str.isdigit, map_name))
    if not map_number_str:
        print(f"Could not extract map number from {map_name}")
        return None
    
    map_number = int(map_number_str)
    waypoints_file = f"carla_maps_waypoints/carla_map_{map_number:02d}_waypoints.csv"

    if not os.path.exists(waypoints_file):
        print(f"Waypoints file not found: {waypoints_file}")
        return None

    df = pd.read_csv(waypoints_file, index_col=0)
    map_waypoints = []
    for index, row in df.iterrows():
        # Assuming the CSV has columns '0' for x and '1' for y.
        map_waypoints.append(MockWaypoint(x=row['0'], y=row['1'], z=0))
    return map_waypoints

def convert_bag_to_json(experiment_path):
    """
    Processes a .bag file in the given experiment folder and generates a .json file with metrics.
    """
    print(f"Processing experiment: {experiment_path}")

    bag_files = glob.glob(os.path.join(experiment_path, '*.bag'))
    json_files = glob.glob(os.path.join(experiment_path, '*.json'))

    if not bag_files:
        print(f"  No .bag file found in {experiment_path}")
        return
    # Filter out already converted files
    json_files = [f for f in json_files if '_converted.json' not in f]

    if not bag_files:
        print(f"  No .bag file found in {experiment_path}")
        return

    bag_file = bag_files[0]
    
    if not json_files:
        print(f"  No original .json file found in {experiment_path}, creating a dummy one.")
        # Create a dummy json file if it does not exist
        json_file_path = os.path.splitext(bag_file)[0] + ".json"
        dummy_data = {
            "carla_map": "Town06",  # Default or detected from path
            "timestamp": os.path.basename(bag_file).replace('.bag', ''),
            "experiment_configuration": {}
        }
        with open(json_file_path, 'w') as f:
            json.dump(dummy_data, f)
        json_file = json_file_path
    else:
        json_file = json_files[0]

    with open(json_file, 'r') as f:
        experiment_metrics = json.load(f)

    map_waypoints = get_map_waypoints(experiment_metrics['carla_map'])
    if not map_waypoints:
        return

    # Create a simplified waypoints_info. This is required for some metric calculations.
    # The color is a dummy value.
    waypoints_info = [{'transform': wp.transform, 'color': {'r':255, 'g':0, 'b':0}} for wp in map_waypoints]

    config = Configuration(None)
    if 'experiment_configuration' in experiment_metrics:
        config.__dict__.update(experiment_metrics['experiment_configuration'])
    else:
        print(f"  'experiment_configuration' not found in {json_file}. Using default config.")

    clockwise = 'clockwise' in experiment_path

    experiment_metrics_filename = os.path.join(experiment_path, experiment_metrics['timestamp'])
    
    updated_metrics = metrics_carla.get_metrics(
        experiment_metrics,
        bag_file,
        map_waypoints,
        experiment_metrics_filename,
        config,
        waypoints_info,
        clockwise
    )

    if not updated_metrics:
        print(f"  Failed to process metrics for {bag_file}")
        return

    # NOTE: Some metrics calculated in controller_carla.py's stop_recording_metrics
    # (like collision types, driving score) require a live CARLA world and are skipped here.

    output_json_path = os.path.join(experiment_path, experiment_metrics['timestamp'] + '_converted.json')
    with open(output_json_path, 'w') as f:
        json.dump(updated_metrics, f, indent=4)
    
    print(f"  Successfully converted {bag_file} to {output_json_path}")


if __name__ == '__main__':
    for path in BROKEN_EXPERIMENTS:
        convert_bag_to_json(path)
