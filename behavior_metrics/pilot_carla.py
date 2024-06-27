#!/usr/bin/env python
""" This module is responsible for handling the logic of the robot and its current brain.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import threading
import time
import rospy
import subprocess
import os

from datetime import datetime
from brains.brains_handler import Brains
from robot.actuators import Actuators
from robot.sensors import Sensors
from utils.logger import logger
from utils.constants import MIN_EXPERIMENT_PERCENTAGE_COMPLETED, ROOT_PATH
from rosgraph_msgs.msg import Clock
from carla_msgs.msg import CarlaControl

import numpy as np

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class PilotCarla(threading.Thread):
    """This class handles the robot and its brain.

    This class called PilotCarla that handles the initialization of the robot sensors and actuators and the
    brain that will control the robot. The main logic consists of an infinite loop called every 60 milliseconds that
    invoke an action from the brain.

    Attributes:
        controller {utils.controller.Controller} -- Controller instance of the MVC of the application
        configuration {utils.configuration.Config} -- Configuration instance of the application
        sensors {robot.sensors.Sensors} -- Sensors instance of the robot
        actuators {robot.actuators.Actuators} -- Actuators instance of the robot
        brains {brains.brains_handler.Brains} -- Brains controller instance
    """

    def __init__(self, configuration, controller, brain_path, experiment_model=None):
        """Constructor of the pilot class

        Arguments:
            configuration {utils.configuration.Config} -- Configuration instance of the application
            controller {utils.controller.Controller} -- Controller instance of the MVC of the application
        """

        self.controller = controller
        self.controller.set_pilot(self)
        self.configuration = configuration
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        threading.Thread.__init__(self, args=self.stop_event)
        self.brain_path = brain_path
        self.robot_type = self.brain_path.split("/")[-2]
        self.sensors = None
        self.actuators = None
        self.brains = None
        self.experiment_model = experiment_model
        self.initialize_robot()
        self.pose3d = self.sensors.get_pose3d('pose3d_0')
        self.start_pose = np.array([self.pose3d.getPose3d().x, self.pose3d.getPose3d().y])
        self.previous = datetime.now()
        self.checkpoints = []
        self.metrics = {}
        self.checkpoint_save = False
        self.max_distance = 0.5
        self.execution_completed = False
        self.stats_thread = threading.Thread(target=self.track_stats)
        self.stats_thread.start()
        self.ros_clock_time = 0
        self.real_time_factor = 0
        self.brain_iterations_real_time = []
        self.brain_iterations_simulated_time = []
        self.real_time_factors = []
        self.real_time_update_rate = 1000
        self.pilot_start_time = 0
        self.time_cycle = self.configuration.pilot_time_cycle
        self.async_mode = self.configuration.async_mode
        self.waypoint_publisher_path = self.configuration.waypoint_publisher_path

    def __wait_carla(self):
        """Wait for simulator to be initialized"""

        self.stop_event.set() 

    def initialize_robot(self):
        """Initialize robot interfaces (sensors and actuators) and its brain from configuration"""
        self.stop_interfaces()
        self.actuators = Actuators(self.configuration.actuators)
        self.sensors = Sensors(self.configuration.sensors)
        if self.experiment_model:
            self.brains = Brains(self.sensors, self.actuators, self.brain_path, self.controller,
                                 self.experiment_model, self.configuration.brain_kwargs)
        else:
            self.brains = Brains(self.sensors, self.actuators, self.brain_path, self.controller,
                                 config=self.configuration.brain_kwargs)
        self.__wait_carla()

    def stop_interfaces(self):
        """Function that kill the current interfaces of the robot. For reloading purposes."""
        if self.sensors:
            self.sensors.kill()
        if self.actuators:
            self.actuators.kill()
        pass

    def run(self):
        """Main loop of the class. Calls a brain action every self.time_cycle"""
        "TODO: cleanup measure of ips"
        self.brain_iterations_simulated_time = []
        self.real_time_factors = []
        self.sensors.get_camera('camera_0').total_frames = 0
        self.pilot_start_time = time.time()

        control_pub = rospy.Publisher('/carla/control', CarlaControl, queue_size=1)
        control_command = CarlaControl()
        control_command.command = 1 # PAUSE
        control_pub.publish(control_command)

        self.waypoint_publisher = None
        while not self.kill_event.is_set():
            if not self.stop_event.is_set():
                if self.waypoint_publisher is None and self.waypoint_publisher_path is not None:
                    self.waypoint_publisher = subprocess.Popen(["roslaunch", ROOT_PATH + '/' + self.waypoint_publisher_path])
                control_pub = rospy.Publisher('/carla/control', CarlaControl, queue_size=1)
                control_command = CarlaControl()
                if self.async_mode:
                    control_command.command = 2 # STEP_ONCE
                else:
                    control_command.command = 0 # PLAY
                control_pub.publish(control_command)

                start_time = datetime.now()
                start_time_ros = self.ros_clock_time
                self.execution_completed = False
                try:
                    self.brains.active_brain.execute()
                except AttributeError as e:
                    logger.warning('No Brain selected')
                    logger.error(e)
                except Exception as ex:
                    logger.warning(type(ex).__name__)
                    logger.warning(ex)
                    logger.warning('ERROR Pilot Carla!')
                    self.stop()
                    self.kill()
                    os._exit(-1)

                dt = datetime.now() - start_time
                ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
                self.brain_iterations_real_time.append(ms / 1000)
                if ms < self.time_cycle:
                    time.sleep((self.time_cycle - ms) / 1000.0)
                self.real_time_factors.append(self.real_time_factor)
                self.brain_iterations_simulated_time.append(self.ros_clock_time - start_time_ros)
                if not self.async_mode:
                    self.controller.world.tick()
                else:
                    self.controller.world.wait_for_tick()

        self.execution_completed = True
        self.kill()
        logger.info('Pilot: pilot killed.')

    def stop(self):
        """Pause the main loop"""

        self.stop_event.set()

    def play(self):
        """Resume the main loop."""

        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill(self):
        """Destroy the main loop. For exiting"""
        self.stop_interfaces()
        self.actuators.kill()
        self.kill_event.set()

    def reload_brain(self, brain_path, model=None):
        """Reload a brain specified by brain_path

        This function is useful if one wants to change the environment of the robot (simulated world).

        Arguments:
            brain_path {str} -- Path to the brain module to load.
        """
        self.brains.load_brain(brain_path, model=model)

    def finish_line(self):
        pose = self.pose3d.getPose3d()
        current_point = np.array([pose.x, pose.y])

        dist = (self.start_pose - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        if dist < self.max_distance:
            return True
        return False

    def clock_callback(self, clock_data):
        self.ros_clock_time = clock_data.clock.to_sec()

    def track_stats(self):
        self.clock_subscriber = rospy.Subscriber("/clock", Clock, self.clock_callback)
