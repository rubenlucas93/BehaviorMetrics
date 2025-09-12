import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from pilot_carla import PilotCarla
from utils import environment
from utils import metrics_carla
from scripts import plot_tensorboard_perc_histogram
from utils.colors import Colors
from utils.configuration import Config
from utils.controller_carla import ControllerCarla
from utils.logger import logger
from utils.traffic import TrafficManager


def check_args(argv):
    """Function that handles argument checking and parsing.

    Arguments:
        argv {list} -- list of arguments from command line.

    Returns:
        dict -- dictionary with the detected configuration.
    """
    parser = argparse.ArgumentParser(description='Neural Behaviors Suite',
                                     epilog='Enjoy the program! :)')

    parser.add_argument('-c',
                        '--config',
                        type=str,
                        action='append',
                        required=True,
                        help='{}Path to the configuration file in YML format.{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g',
                       '--gui',
                       action='store_true',
                       help='{}Load the GUI (Graphic User Interface). Requires PyQt5 installed{}'.format(
                           Colors.OKBLUE, Colors.ENDC))

    group.add_argument('-t',
                       '--tui',
                       action='store_true',
                       help='{}Load the TUI (Terminal User Interface). Requires npyscreen installed{}'.format(
                           Colors.OKBLUE, Colors.ENDC))

    group.add_argument('-s',
                       '--script',
                       action='store_true',
                       help='{}Run Behavior Metrics as script{}'.format(
                           Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-r',
                        '--random',
                        action='store_true',
                        help='{}Run Behavior Metrics F1 with random spawning{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    args = parser.parse_args()

    config_data = {'config': None, 'gui': None, 'tui': None, 'script': None, 'random': False}
    if args.config:
        config_data['config'] = []
        for config_file in args.config:
            if not os.path.isfile(config_file):
                parser.error('{}No such file {} {}'.format(Colors.FAIL, config_file, Colors.ENDC))

        config_data['config'] = args.config

    if args.gui:
        config_data['gui'] = args.gui

    if args.tui:
        config_data['tui'] = args.tui

    if args.script:
        config_data['script'] = args.script

    if args.random:
        config_data['random'] = args.random

    return config_data

def main_win(configuration, controller):
    """shows the Qt main window of the application

    Arguments:
        configuration {Config} -- configuration instance for the application
        controller {Controller} -- controller part of the MVC model of the application
    """
    try:
        from PyQt5.QtWidgets import QApplication
        from ui.gui.views_controller import ParentWindow, ViewsController

        app = QApplication(sys.argv)
        main_window = ParentWindow()

        views_controller = ViewsController(main_window, configuration, controller)
        views_controller.show_main_view(True)

        main_window.show()

        app.exec_()
    except Exception as e:
        logger.error(e)

def is_config_correct(app_configuration):
    is_correct = True
    if len(app_configuration.current_world) != len(app_configuration.experiment_timeouts):
        logger.error('Config error: Worlds number is not equal to experiment timeouts')
        is_correct = False
    if len(app_configuration.brain_path) != len(app_configuration.experiment_model):
        logger.error('Config error: Brains number is not equal to experiment models')
        is_correct = False

    return is_correct

def generate_agregated_experiments_metrics(experiments_starting_time, experiments_elapsed_times, app_configuration):
    result = metrics_carla.get_aggregated_experiments_list(experiments_starting_time)

    experiments_starting_time_dt = datetime.fromtimestamp(experiments_starting_time)
    experiments_starting_time_str = str(experiments_starting_time_dt.strftime("%Y%m%d-%H%M%S")) + '_experiments_metrics'

    os.makedirs(experiments_starting_time_str, exist_ok=True)
    experiments_metrics_and_titles = [
        # {
        #     'metric': 'line_speeds_by_waypoint',
        #     'title': 'speed by waypoint'
        # },
        # {
        #     'metric': 'histogram_speeds',
        #     'title': 'Experiment speed histograms'
        # },
        # {
        #     'metric': 'histogram_position_deviations',
        #     'title': 'Experiment position_deviations histograms'
        # },
        {
            'metric': 'experiment_total_simulated_time',
            'title': 'Experiment total simulated time per experiment'
        },
        {
            'metric': 'position_deviation_total_err',
            'title': 'Position deviation total error per experiment'
        },
        {
            'metric': 'effective_completed_distance',
            'title': 'Effective completed distance per experiment'
        },
        {
            'metric': 'experiment_total_real_time',
            'title': 'Experiment total real time per experiment'
        },
        {
            'metric': 'suddenness_distance_control_commands',
            'title': 'Suddennes distance control commands per experiment'
        },
        {
            'metric': 'suddenness_distance_throttle',
            'title': 'Suddennes distance throttle per experiment'
        },
        {
            'metric': 'suddenness_distance_steer',
            'title': 'Suddennes distance steer per experiment'
        },
        {
            'metric': 'suddenness_distance_brake_command',
            'title': 'Suddennes distance brake per experiment'
        },
        {
            'metric': 'suddenness_distance_control_command_per_km',
            'title': 'Suddennes distance control commands per km per experiment'
        },
        {
            'metric': 'suddenness_distance_throttle_per_km',
            'title': 'Suddennes distance throttle per km per experiment'
        },
        {
            'metric': 'suddenness_distance_steer_per_km',
            'title': 'Suddennes distance steer per km per experiment'
        },
        {
            'metric': 'suddenness_distance_brake_command_per_km',
            'title': 'Suddennes distance brake per km per experiment'
        },
        {
            'metric': 'brain_iterations_frequency_simulated_time',
            'title': 'Brain itertions frequency simulated time per experiment'
        },
        {
            'metric': 'mean_brain_iterations_simulated_time',
            'title': 'Mean brain iterations simulated time per experiment'
        }, 
        {
            'metric': 'gpu_mean_inference_time',
            'title': 'GPU mean inference time per experiment'
        }, 
        {
            'metric': 'mean_brain_iterations_real_time',
            'title': 'Mean brain iterations real time per experiment'
        },
        {
            'metric': 'target_brain_iterations_real_time',
            'title': 'Target brain iterations real time per experiment'
        },
        {
            'metric': 'completed_distance',
            'title': 'Total distance per experiment'
        },
        {
            'metric': 'average_speed',
            'title': 'Average speed per experiment'
        },
        {
            'metric': 'max_speed',
            'title': 'Max speed per experiment'
        },
        {
            'metric': 'collisions',
            'title': 'Total collisions per experiment'
        },
        {
            'metric': 'lane_invasions',
            'title': 'Total lane invasions per experiment'
        },
        {
            'metric': 'position_deviation_mean',
            'title': 'Mean position deviation per experiment'
        },
        {
            'metric': 'position_deviation_mean_per_km',
            'title': 'Mean position deviation per km per experiment'
        },
        {
            'metric': 'gpu_inference_frequency',
            'title': 'GPU inference frequency per experiment'
        },
        {
            'metric': 'brain_iterations_frequency_real_time',
            'title': 'Brain frequency per experiment'
        },
        {
            'metric': 'collisions_per_km',
            'title': 'Collisions per km per experiment'
        },
        {
            'metric': 'lane_invasions_per_km',
            'title': 'Lane invasions per experiment'
        },
        {
            'metric': 'suddenness_distance_speed',
            'title': 'Suddenness distance speed per experiment'
        },
        {
            'metric': 'suddenness_distance_speed_per_km',
            'title': 'Suddenness distance speed per km per experiment'
        },
        
        {
            'metric': 'completed_laps',
            'title': 'Completed laps per experiment'
        },
    ]

    if app_configuration.task == 'follow_lane_traffic':
        experiments_metrics_and_titles.append(
            {
                'metric': 'dangerous_distance_pct_km',
                'title': 'Percentage of dangerous distance per km'
            },
            {
                'metric': 'close_distance_pct_km',
                'title': 'Percentage of close distance per km'
            },
            {
                'metric': 'medium_distance_pct_km',
                'title': 'Percentage of medium distance per km'
            },
            {
                'metric': 'great_distance_pct_km',
                'title': 'Percentage of great distance per km'
            },
        )

    metrics_carla.get_all_experiments_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles)
    metrics_carla.get_per_model_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles)
    metrics_carla.get_all_experiments_aggregated_metrics_boxplot(result, experiments_starting_time_str, experiments_metrics_and_titles)

    with open(experiments_starting_time_str + '/' + 'experiment_elapsed_times.json', 'w') as f:
        json.dump(experiments_elapsed_times, f)

    # df = pd.DataFrame(experiments_elapsed_times)
    # fig = plt.figure(figsize=(20,10))
    # df['elapsed_time'].plot.bar()
    # plt.title('Experiments elapsed time || Experiments total time: ' + str(experiments_elapsed_times['total_experiments_elapsed_time']) + ' secs.')
    # fig.tight_layout()
    # plt.xticks(rotation=90)
    # plt.savefig(experiments_starting_time_str + '/' + 'experiment_elapsed_times.png')
    # plt.close()

from types import SimpleNamespace

if __name__ == '__main__':
    timestamp_str = "20250704-214225"
    dt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
    experiments_starting_time = dt.timestamp()
    experiments_elapsed_times = {'experiment_counter': [], 'elapsed_time': []}
    experiments_elapsed_times['total_experiments_elapsed_time'] = time.time() - experiments_starting_time
    app_configuration = SimpleNamespace(**{"task": "none"})

    generate_agregated_experiments_metrics(experiments_starting_time, experiments_elapsed_times, app_configuration)
    # comparisons = metrics_carla.get_tensorboard_comparisons(experiments_starting_time)
    # plot_tensorboard_perc_histogram.save_histograms_comparison_same(comparisons[0], comparisons[1], comparisons[2])

    sys.exit(0)
