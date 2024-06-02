#!/usr/bin/env python

"""This module contains the controller of the application.

This application is based on a type of software architecture called Model View Controller. This is the controlling part
of this architecture (controller), which communicates the logical part (model) with the user interface (view).

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

import shlex
import subprocess
import threading
import cv2
import rospy
import os
import time
import rosbag
import json
import math
from utils.logger import logger
try:
    import carla
except ModuleNotFoundError as ex:
    logger.error('CARLA is not supported')
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime
from std_msgs.msg import String, Float32
from utils import metrics_carla
from utils.constants import CARLA_INFRACTION_PENALTIES
try:
    from carla_msgs.msg import CarlaLaneInvasionEvent
    from carla_msgs.msg import CarlaCollisionEvent
except ModuleNotFoundError as ex:
    logger.error('CARLA is not supported')
from PIL import Image
__author__ = 'sergiopaniego'
__contributors__ = []
__license__ = 'GPLv3'


class ControllerCarla:
    """This class defines the controller of the architecture, responsible of the communication between the logic (model)
    and the user interface (view).

    Attributes:
        data {dict} -- Data to be sent to the view. The key is a frame_if of the view and the value is the data to be
        displayed. Depending on the type of data the frame handles (images, laser, etc)
        pose3D_data -- Pose data to be sent to the view
        recording {bool} -- Flag to determine if a rosbag is being recorded
    """

    def __init__(self):
        """ Constructor of the class. """
        pass
        self.__data_loc = threading.Lock()
        self.__pose_loc = threading.Lock()
        self.data = {}
        self.pose3D_data = None
        self.recording = False
        self.cvbridge = CvBridge()

        client = carla.Client('localhost', 2000)
        client.set_timeout(100.0) # seconds
        self.world = client.get_world()
        time.sleep(5) # takes a few second for the correct map to finish loading
        self.carla_map = self.world.get_map()
        while len(self.world.get_actors().filter('vehicle.*')) == 0:
            logger.info("Waiting for vehicles!")
            time.sleep(1)
        ego_vehicle_role_name = "ego_vehicle"
        self.ego_vehicle = None
        while self.ego_vehicle is None:
            for vehicle in self.world.get_actors().filter('vehicle.*'):
                if vehicle.attributes.get('role_name') == ego_vehicle_role_name:
                    self.ego_vehicle = vehicle
                    break
            if self.ego_vehicle is None:
                logger.info("Waiting for vehicle with role_name 'ego_vehicle'")
                time.sleep(1)  # sleep for 1 second before checking again
        self.map_waypoints = self.carla_map.generate_waypoints(0.5)
        self.weather = self.world.get_weather()
        self.pub = rospy.Publisher('/carla/ego_vehicle/steering_angle', Float32, queue_size=1)

        
    # GUI update
    def update_frame(self, frame_id, data):
        """Update the data to be retrieved by the view.

        This function is called by the logic to update the data obtained by the robot to a specific frame in GUI.

        Arguments:
            frame_id {str} -- Identifier of the frame that will show the data
            data {dict} -- Data to be shown
        """
        try:
            steer_angle = self.ego_vehicle.get_control().steer # TODO There must be a better way
            self.pub.publish(steer_angle)
            with self.__data_loc:
                self.data[frame_id] = data
        except Exception as e:
            logger.info(e)

    def get_data(self, frame_id):
        """Function to collect data retrieved by the robot for an specific frame of the GUI

        This function is called by the view to get the last updated data to be shown in the GUI.

        Arguments:
            frame_id {str} -- Identifier of the frame.

        Returns:
            data -- Depending on the caller frame could be image data, laser data, etc.
        """
        try:
            with self.__data_loc:
                data = self.data.get(frame_id, None)
        except Exception:
            pass

        return data

    def update_pose3d(self, data):
        """Update the pose3D data retrieved from the robot

        Arguments:
            data {pose3d} -- 3D position of the robot in the environment
        """
        try:
            with self.__pose_loc:
                self.pose3D_data = data
        except Exception:
            pass

    def get_pose3D(self):
        """Function to collect the pose3D data updated in `update_pose3d` function.

        This method is called from the view to collect the pose data and display it in GUI.

        Returns:
            pose3d -- 3D position of the robot in the environment
        """
        return self.pose3D_data

    # Simulation and dataset
    def reset_carla_simulation(self):
        logger.info("Restarting simulation")

    def pause_carla_simulation(self):
        logger.info("Pausing simulation")
        self.pilot.stop_event.set()

    def unpause_carla_simulation(self):
        logger.info("Resuming simulation")
        self.pilot.stop_event.clear()

    def record_rosbag(self, topics, dataset_name):
        """Start the recording process of the dataset using rosbags

        Arguments:
            topics {list} -- List of topics to be recorde
            dataset_name {str} -- Path of the resulting bag file
        """

        if not self.recording:
            logger.info("Recording bag at: {}".format(dataset_name))
            self.recording = True
            command = "rosbag record -O " + dataset_name + " " + " ".join(topics) + " __name:=behav_bag"
            command = shlex.split(command)
            with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
                self.rosbag_proc = subprocess.Popen(command, stdout=out, stderr=err)
        else:
            logger.info("Rosbag already recording")
            self.stop_record()

    def stop_record(self):
        """Stop the rosbag recording process."""
        if self.rosbag_proc and self.recording:
            logger.info("Stopping bag recording")
            self.recording = False
            command = "rosnode kill /behav_bag"
            command = shlex.split(command)
            with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
                subprocess.Popen(command, stdout=out, stderr=err)
        else:
            logger.info("No bag recording")

    def reload_brain(self, brain, model=None):
        """Helper function to reload the current brain from the GUI.

        Arguments:
            brain {srt} -- Brain to be reloadaed.
        """
        logger.info("Reloading brain... {}".format(brain))

        self.pause_pilot()
        self.pilot.reload_brain(brain, model)

    # Helper functions (connection with logic)

    def set_pilot(self, pilot):
        self.pilot = pilot

    def stop_pilot(self):
        self.pilot.kill_event.set()

    def pause_pilot(self):
        self.pilot.stop_event.set()

    def resume_pilot(self):
        self.start_time = datetime.now()
        self.pilot.start_time = datetime.now()
        self.pilot.stop_event.clear()

    def initialize_robot(self):
        self.pause_pilot()
        self.pilot.initialize_robot()


    def record_metrics(self, metrics_record_dir_path, world_counter=None, brain_counter=None, repetition_counter=None):
        logger.info("Recording metrics bag at: {}".format(metrics_record_dir_path))

        self.pilot.brain_iterations_real_time = []
        self.time_str = time.strftime("%Y%m%d-%H%M%S")       
        if world_counter is not None:
            current_world_head, current_world_tail = os.path.split(self.pilot.configuration.current_world[world_counter])
        else:
            current_world_head, current_world_tail = os.path.split(self.pilot.configuration.current_world)
        if brain_counter is not None:
            current_brain_head, current_brain_tail = os.path.split(self.pilot.configuration.brain_path[brain_counter])
        else:
            current_brain_head, current_brain_tail = os.path.split(self.pilot.configuration.brain_path)
        self.experiment_metrics = {
            'timestamp': self.time_str,
            'experiment_configuration': self.pilot.configuration.__dict__,
            'world_launch_file': current_world_tail,
            'brain_file': current_brain_tail,
            'robot_type': self.pilot.configuration.robot_type,
            'carla_map': self.carla_map.name,
            'ego_vehicle': self.ego_vehicle.type_id,
            'vehicles_number': len(self.world.get_actors().filter('vehicle.*')),
            'async_mode': self.pilot.configuration.async_mode,
            'weather': {
                'cloudiness': self.weather.cloudiness,
                'precipitation': self.weather.precipitation,
                'precipitation_deposits': self.weather.precipitation_deposits,
                'wind_intensity': self.weather.wind_intensity,
                'sun_azimuth_angle': self.weather.sun_azimuth_angle,
                'sun_altitude_angle': self.weather.sun_altitude_angle,
                'fog_density': self.weather.fog_density,
                'fog_distance': self.weather.fog_distance,
                'fog_falloff': self.weather.fog_falloff,
                'wetness': self.weather.wetness,
                'scattering_intensity': self.weather.scattering_intensity,
                'mie_scattering_scale': self.weather.mie_scattering_scale,
                'rayleigh_scattering_scale': self.weather.rayleigh_scattering_scale,
                },
        }
        if hasattr(self.pilot.configuration, 'experiment_model'):
            if brain_counter is not None:
                self.experiment_metrics['experiment_model'] = self.pilot.configuration.experiment_model[brain_counter]
            else:
                self.experiment_metrics['experiment_model'] = self.pilot.configuration.experiment_model

        if hasattr(self.pilot.configuration, 'experiment_name'):
            self.experiment_metrics['experiment_name'] = self.pilot.configuration.experiment_name

        if hasattr(self.pilot.configuration, 'experiment_name'):
            self.experiment_metrics['experiment_name'] = self.pilot.configuration.experiment_name
            self.experiment_metrics['experiment_description'] = self.pilot.configuration.experiment_description
            self.experiment_metrics['experiment_timeout'] = self.pilot.configuration.experiment_timeouts[world_counter]
            self.experiment_metrics['experiment_repetition'] = repetition_counter
        

        self.metrics_record_dir_path = metrics_record_dir_path
        os.mkdir(self.metrics_record_dir_path + self.time_str)
        self.experiment_metrics_bag_filename = self.metrics_record_dir_path + self.time_str + '/' + self.time_str + '.bag'

        topics = [
            '/carla/npc_vehicle_1/odometry',
            '/carla/ego_vehicle/odometry',
            '/carla/ego_vehicle/collision',
            '/carla/ego_vehicle/lane_invasion',
            '/carla/ego_vehicle/speedometer',
            '/carla/ego_vehicle/steering_angle',
            '/carla/ego_vehicle/vehicle_status',
            '/clock',
            ]

        command = "rosbag record -O " + self.experiment_metrics_bag_filename + " " + " ".join(topics) + " __name:=behav_metrics_bag"
        command = shlex.split(command)
        with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
            self.proc = subprocess.Popen(command, stdout=out, stderr=err)

    def stop_recording_metrics(self, termination_code=None, route_length=None):
        logger.info("Stopping metrics bag recording")
        end_time = time.time()

        mean_brain_iterations_real_time = sum(self.pilot.brain_iterations_real_time) / len(self.pilot.brain_iterations_real_time)
        brain_iterations_frequency_real_time = 1 / mean_brain_iterations_real_time

        mean_brain_iterations_simulated_time = sum(self.pilot.brain_iterations_simulated_time) / len(self.pilot.brain_iterations_simulated_time)
        brain_iterations_frequency_simulated_time = 1 / mean_brain_iterations_simulated_time

        target_brain_iterations_real_time = 1 / (self.pilot.time_cycle / 1000)

        if hasattr(self.pilot.brains.active_brain, 'cameras_first_images') and self.pilot.brains.active_brain.cameras_first_images != []:
            first_images =  self.pilot.brains.active_brain.cameras_first_images
            last_images = self.pilot.brains.active_brain.cameras_last_images
        else:
            first_images = []
            last_images = []

        command = "rosnode kill /behav_metrics_bag"
        command = shlex.split(command)
        with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
            subprocess.Popen(command, stdout=out, stderr=err)

        # Wait for rosbag file to be closed. Otherwise it causes error
        while os.path.isfile(self.experiment_metrics_bag_filename + '.active'):
            pass
        
        if hasattr(self.pilot.brains.active_brain, 'inference_times'):
            self.pilot.brains.active_brain.inference_times = self.pilot.brains.active_brain.inference_times[10:-10]
            self.experiment_metrics['gpu_mean_inference_time'] = sum(self.pilot.brains.active_brain.inference_times) / len(self.pilot.brains.active_brain.inference_times)
            self.experiment_metrics['gpu_inference_frequency'] = 1 / self.experiment_metrics['gpu_mean_inference_time']
            self.experiment_metrics['gpu_inference'] = self.pilot.brains.active_brain.gpu_inference
        else:
            self.experiment_metrics['gpu_mean_inference_time'] = 0
            self.experiment_metrics['gpu_inference_frequency'] = 0
            self.experiment_metrics['gpu_inference'] = 0

        if hasattr(self.pilot.brains.active_brain, 'bird_eye_view_images'):
            self.experiment_metrics['bird_eye_view_images'] = self.pilot.brains.active_brain.bird_eye_view_images
            self.experiment_metrics['bird_eye_view_unique_images'] = self.pilot.brains.active_brain.bird_eye_view_unique_images
            self.experiment_metrics['bird_eye_view_unique_images_percentage'] = self.experiment_metrics['bird_eye_view_unique_images'] / self.experiment_metrics['bird_eye_view_images']
        else:
            self.experiment_metrics['bird_eye_view_images'] = 0
            self.experiment_metrics['bird_eye_view_unique_images'] = 0
            self.experiment_metrics['bird_eye_view_unique_images_percentage'] = 0

        self.experiment_metrics['brain_iterations_simulated_time'] = len(self.pilot.brain_iterations_simulated_time)
        self.experiment_metrics['mean_brain_iterations_real_time'] = mean_brain_iterations_real_time
        self.experiment_metrics['brain_iterations_frequency_real_time'] = brain_iterations_frequency_real_time
        self.experiment_metrics['target_brain_iterations_real_time'] = target_brain_iterations_real_time
        self.experiment_metrics['mean_brain_iterations_simulated_time'] = mean_brain_iterations_simulated_time
        self.experiment_metrics['brain_iterations_frequency_simulated_time'] = brain_iterations_frequency_simulated_time
        self.experiment_metrics['experiment_total_real_time'] = end_time - self.pilot.pilot_start_time

        experiment_metrics_filename = self.metrics_record_dir_path + self.time_str + '/' + self.time_str
        self.experiment_metrics = metrics_carla.get_metrics(self.experiment_metrics, self.experiment_metrics_bag_filename, self.map_waypoints, experiment_metrics_filename, self.pilot.configuration)
        self.experiment_metrics['collisions_vehicle'] = 0
        self.experiment_metrics['collisions_walker'] = 0
        self.experiment_metrics['collisions_static'] = 0

        collision_actor_ids = self.experiment_metrics['collision_actor_ids']
        for actor_id in collision_actor_ids:
            actor = self.world.get_actor(actor_id)
            if actor:
                actor_type = actor.type_id.split('.')[0]
                if actor_type == 'vehicle':
                    self.experiment_metrics['collisions_vehicle'] += 1
                elif actor_type == 'walker':
                    self.experiment_metrics['collisions_walker'] += 1
                else:
                    self.experiment_metrics['collisions_static'] += 1
            else:
                print(f"No actor found with ID {actor_id}")

        if hasattr(self.pilot.brains.active_brain, 'red_light_counter'):
            self.experiment_metrics['traffic_light_infractions'] = self.pilot.brains.active_brain.red_light_counter
            self.experiment_metrics['traffic_light_infractions_per_km'] = self.experiment_metrics['traffic_light_infractions'] / (self.experiment_metrics['effective_completed_distance']/1000)

        if route_length is not None:
            self.experiment_metrics['route_completion'] = self.experiment_metrics['effective_completed_distance'] / route_length
        
        wrong_turn_counter = 0
        time_out_counter = 0
        if termination_code is not None:
            if termination_code == 1:
                self.experiment_metrics['termination cause'] = 'success'
            elif termination_code == 2:
                wrong_turn_counter += 1
                self.experiment_metrics['termination cause'] = 'wrong turn'
            elif termination_code == 3:
                time_out_counter += 1
                self.experiment_metrics['termination cause'] = 'time out'
        
        if 'route_completion' in self.experiment_metrics.keys() and 'traffic_light_infractions' in self.experiment_metrics.keys() and termination_code is not None:
            self.experiment_metrics['driving score'] = self.experiment_metrics['route_completion'] * \
                                                    CARLA_INFRACTION_PENALTIES['collision_vehicle']**self.experiment_metrics['collisions_vehicle'] * \
                                                    CARLA_INFRACTION_PENALTIES['collision_static']**self.experiment_metrics['collisions_static'] * \
                                                    CARLA_INFRACTION_PENALTIES['collision_walker']**self.experiment_metrics['collisions_walker'] * \
                                                    CARLA_INFRACTION_PENALTIES['red_light']**self.experiment_metrics['traffic_light_infractions'] * \
                                                    CARLA_INFRACTION_PENALTIES['wrong_turn']**wrong_turn_counter * \
                                                    CARLA_INFRACTION_PENALTIES['time_out']**time_out_counter
        self.save_metrics(first_images, last_images)

        for key, value in self.experiment_metrics.items():
            logger.info('* ' + str(key) + ' ---> ' + str(value))

        logger.info("Stopped metrics bag recording")


    def save_metrics(self, first_images, last_images):        
        with open(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + '.json', 'w') as f:
            json.dump(self.experiment_metrics, f)
        logger.info("Metrics stored in JSON file")

        for counter, image in enumerate(first_images):
            im = Image.fromarray(image)
            im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_first_image_" + str(counter) + ".jpeg")

        for counter, image in enumerate(last_images):
            im = Image.fromarray(image)
            im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_last_image_" + str(counter) + ".jpeg")



