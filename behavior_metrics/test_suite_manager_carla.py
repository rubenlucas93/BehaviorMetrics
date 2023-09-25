import argparse
import os
import sys
import threading
import time
import rospy
import importlib
import random

from pilot_carla import PilotCarla
from utils import environment
from utils.colors import Colors
from utils.configuration import Config
from utils.controller_carla import ControllerCarla
from utils.logger import logger
from utils.constants import CARLA_TOWNS_TIMEOUTS
from utils.traffic import TrafficManager

def check_args(argv):
    parser = argparse.ArgumentParser(description='Testing suite runner.')

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

    parser.add_argument('-world_counter',
                        type=str,
                        action='append',
                        help='{}World counter{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-brain_counter',
                        type=str,
                        action='append',
                        help='{}Brain counter{}'.format(
                            Colors.OKBLUE, Colors.ENDC))
    
    parser.add_argument('-route_counter',
                        type=str,
                        action='append',
                        help='{}Route counter{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    args = parser.parse_args()

    config_data = {'config': None, 'gui': None, 'tui': None, 'script': None, 'random': False, 'world_counter': 0, 'brain_counter': 0, 'route_counter': 0}
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

    return config_data

def main():
    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'][0])
    world_counter = int(config_data['world_counter'])
    brain_counter = int(config_data['brain_counter'])
    route_counter = int(config_data['route_counter'])

    logger.info(str(world_counter) + ' ' + str(brain_counter) + ' ' + str(route_counter))

    world = app_configuration.current_world[world_counter]
    brain = app_configuration.brain_path[brain_counter]
    experiment_model = app_configuration.experiment_model[brain_counter]
    

    if not os.path.exists(app_configuration.test_suite):
        logger.info('Test suite file does not exist! Killing program...')
        sys.exit(-1)
    
    module_dir = os.path.dirname(app_configuration.test_suite)
    if module_dir not in sys.path:
        sys.path.append(module_dir)
    module_name = os.path.splitext(os.path.basename(app_configuration.test_suite))[0]
    test_routes_module = importlib.import_module(module_name)
    TEST_ROUTES = getattr(test_routes_module, 'TEST_ROUTES')
    spawn_point = TEST_ROUTES[route_counter]['start']
    town = TEST_ROUTES[route_counter]['map']
    environment.launch_env(world, 
                           random_spawn_point=False, 
                           carla_simulator=True, 
                           config_spawn_point=spawn_point,
                           config_town=town)
    controller = ControllerCarla()

    # generate traffic
    traffic_manager = TrafficManager(app_configuration.number_of_vehicle, 
                                         app_configuration.number_of_walker, 
                                         app_configuration.percentage_walker_running, 
                                         app_configuration.percentage_walker_crossing,
                                         app_configuration.async_mode,
                                         port=random.randint(8000, 9000))
    traffic_manager.generate_traffic()
    
    # Launch control
    pilot = PilotCarla(app_configuration, controller, brain, experiment_model=experiment_model)
    pilot.daemon = True
    pilot.start()
    logger.info('Executing app')
    controller.resume_pilot()
    controller.unpause_carla_simulation()
    controller.record_metrics(app_configuration.stats_out, world_counter=world_counter, brain_counter=brain_counter, repetition_counter=route_counter)
    if app_configuration.use_world_timeouts:
        experiment_timeout = CARLA_TOWNS_TIMEOUTS[controller.carla_map.name]
    else:
        experiment_timeout = app_configuration.experiment_timeouts[world_counter]

    rospy.sleep(experiment_timeout)
    controller.stop_recording_metrics()
    controller.pilot.stop()
    controller.stop_pilot()
    controller.pause_carla_simulation()

    logger.info('closing all processes...')
    controller.pilot.kill()
    environment.close_ros_and_simulators()
    while not controller.pilot.execution_completed:
        time.sleep(1)


if __name__ == '__main__':
    main()
    sys.exit(0)