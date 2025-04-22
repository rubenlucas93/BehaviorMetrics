#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import math
import numpy as np
import threading
import time
from PIL import Image
from sklearn.linear_model import LinearRegression
import carla



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from stable_baselines3 import PPO

import torch
from collections import Counter

import random
import yaml
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare
)

from brains.CARLA.utils.lanes_detector import LaneDetector
from utils.constants import DATASETS_DIR, ROOT_PATH

from brains.CARLA.utils.modified_tensorboard import ModifiedTensorBoard
from stable_baselines3.common.noise import NormalActionNoise


GENERATED_DATASETS_DIR = ROOT_PATH + '/' + DATASETS_DIR

NO_DETECTED = 1


from pydantic import BaseModel
class InferenceExecutorValidator(BaseModel):
    settings: dict
    inference: dict

from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        self.action_net = nn.Sequential(
            self.action_net,  # Use the original action network
            nn.Tanh()         # Add the Tanh activation
        )

class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        self.client = carla.Client(
            "localhost",
            2000,
        )
        self.client.set_timeout(10.0)
        print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")

        self.world = self.client.get_world()
        self.map = self.world.get_map()
        all_actors = self.world.get_actors()
        vehicles = all_actors.filter("vehicle.*")
        if len(vehicles) > 0:
            self.car = vehicles[0]
        else:
            print("No vehicles found in the world.")

        self.lidar = all_actors.filter("sensor.lidar.ray_cast")[0]
        self.lidar.listen(self.process_lidar_data)

        # location = self.car.get_transform()
        # spectator = self.world.get_spectator()
        # spectator_location = carla.Transform(
        #     location.location + carla.Location(z=100),
        #     carla.Rotation(-90, location.rotation.yaw, 0))
        # spectator.set_transform(spectator_location)

        self.last_action = [0, 0]
        self.last_state = [0, 0, 0, 0, 0]
        self.detection_mode = "carla_perfect"
        self.camera = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')
        self.speedometer = sensors.get_speed('speedometer_0')
        self.wheel = sensors.get_wheel('wheel')
        self.start_time = time.time()
        self.avg_speed = 0

        self.pose = sensors.get_pose3d('pose3d_0')

        self.previous_time = 0

        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config

        self.threshold_image = np.zeros((640, 360, 3), np.uint8)
        self.color_image = np.zeros((640, 360, 3), np.uint8)
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.cont = 0
        self.iteration = 0
        self.step = 0
        self.previous_action = [0, 0]

        self.sync_mode = True
        self.show_images = False
        # self.detection_mode = 'lane_detector'

        # self.previous_timestamp = 0
        # self.previous_image = 0

        self.previous_v = None
        self.previous_w = None
        self.previous_w_normalized = None

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/Tensorboard/ppo/{time.strftime('%Y%m%d-%H%M%S')}"
        )

        args = {
            'algorithm': 'ppo',
            'environment': 'simple',
            'agent': 'f1',
            'filename': 'brains/CARLA/config/config_inference_followlane_sb_ppo_f1_carla.yaml'
        }


        f = open(args['filename'], "r")
        read_file = f.read()

        config_file = yaml.load(read_file, Loader=yaml.FullLoader)
        inference_params = {
            "settings": self.get_settings(config_file),
            "inference": self.get_inference(config_file, args['algorithm']),
        }

        self.x_row = self.get_states_rows(config_file)
        inference_distances = {
            "Carla/Maps/Town10HD": 2200,
            "Carla/Maps/Town06": 2500,
            "Carla/Maps/Town04": 3000
        }


        params = InferenceExecutorValidator(**inference_params)
        inference_file = params.inference["params"]["inference_tf_model_name"]
        self.inference_distance = inference_distances[self.map.name]

        self.lane_detector = LaneDetector(self.car, self.map, self.x_row)

        self.ppo_agent = PPO.load(inference_file, custom_objects={"CustomActorCriticPolicy": CustomActorCriticPolicy})
        # action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.0 * np.ones(2))
        # self.ppo_agent.action_noise = action_noise

        time.sleep(2)

    def get_states_rows(self, config_file: dict) -> dict:
        return  config_file["states"][config_file["settings"]["states"]][0]

    def get_inference(self, config_file: dict, input_inference: str) -> dict:
        return {
            "name": input_inference,
            "params": config_file["inference"][input_inference],
        }

    def get_settings(self, config_file: dict) -> dict:
        return {
            "name": "settings",
            "params": config_file["settings"],
        }

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def lidar_point_to_world(self, lidar_detection):
        # Get the sensor's transformation
        sensor_transform = self.lidar.get_transform()
        sensor_location = sensor_transform.location
        sensor_rotation = sensor_transform.rotation

        # Extract relative point from lidar detection
        relative_point = lidar_detection.point  # Example: (x, y, z) in sensor's frame

        # Convert sensor rotation to radians
        yaw = math.radians(sensor_rotation.yaw)
        pitch = math.radians(sensor_rotation.pitch)
        roll = math.radians(sensor_rotation.roll)

        # Rotation matrix for the sensor
        rotation_matrix = np.array([
            [
                math.cos(yaw) * math.cos(pitch),
                math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll),
                math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)
            ],
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll),
                math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll)
            ],
            [
                -math.sin(pitch),
                math.cos(pitch) * math.sin(roll),
                math.cos(pitch) * math.cos(roll)
            ]
        ])

        # Transform the relative point to the world frame
        relative_vector = np.array([relative_point.x, relative_point.y, relative_point.z])
        world_vector = np.dot(rotation_matrix, relative_vector)

        # Add the sensor's global location
        world_x = sensor_location.x + world_vector[0]
        world_y = sensor_location.y + world_vector[1]
        world_z = sensor_location.z + world_vector[2]

        return carla.Location(x=world_x, y=world_y, z=world_z)

    def process_lidar_data(self, data):
        car_location = self.car.get_location()
        min_distance = float('inf')

        # Define the range of lateral (Y) distance for "front" filtering (optional)
        lateral_limit = 2.0  # 2 meters to the left and right of the vehicle's center

        # Define the angle range for the "cone" of front-facing points
        front_angle_limit = 30.0  # 30 degrees (left and right) from the center of the vehicle

        for detection in data:
            # Transform LiDAR point to world coordinates
            world_point = self.lidar_point_to_world(detection)

            # Get the angle of the point relative to the vehicle's forward direction
            angle = math.degrees(math.atan2(world_point.y - car_location.y, world_point.x - car_location.x))

            # Only consider points ahead of the vehicle and within the cone
            # if world_point.x > car_location.x and abs(angle) <= front_angle_limit and abs(world_point.y) < lateral_limit:
            if True:
                # Compute distance to the car
                distance = car_location.distance(world_point)
                if distance < min_distance:
                    min_distance = distance

                # Visualize the LiDAR point in front of the vehicle
        self.lidar_front_distance = min_distance if not math.isinf(min_distance) else 100
        # print(f"Closest object in front of the vehicle is {min_distance} meters away.")

    def execute(self):
        if self.step == 0:
            self.start_time = time.time()
        episode_duration = time.time() - self.start_time
        distance_run = episode_duration * self.avg_speed
        if distance_run >= self.inference_distance:
            print("episode finished")
            self.car.apply_control(carla.VehicleControl(throttle=0, brake = 1))
            return True
        if not self.step % 200:
            print(distance_run)
        # TODO integrate with environment
        # observation, reward, done, info = self.env.step(action, self.step)
        self.step += 1

        now = time.time()
        fps = 1 / (now - self.previous_time)
        self.previous_time = now
        self.tensorboard.update_fps(fps)

        image = self.camera.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        sensor_time = time.time()
        self.tensorboard.update_times(sensor_time - self.previous_time, "sensor")

        state, image_processed = self.lane_detector.process_image(image)

        perception_time = time.time()
        self.tensorboard.update_times(perception_time - sensor_time, "perception")

        # speed = self.speedometer.getSpeedometer().data
        # w_angle = self.wheel.getWheelAngle()

        final_curvature = self.calculate_curvature_from(state)

        v = self.car.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        w_angle = self.car.get_control().steer

        state.append(speed/40)
        state.append(w_angle)
        #state.append(self.lidar_front_distance/100)
        # state.append(max(0, 1-(abs(state[0])*5)))
        state.append(final_curvature)
        state.append(self.previous_action[0])
        state.append(self.previous_action[1])

        [action, _] = self.ppo_agent.predict(observation=np.array(state), deterministic=True)

        # if self.step % 20:
        #     self.tensorboard.update_stats(speed = speed)
        #
        self.tensorboard.update_actions(action, self.step)

        if float(action[0]) > 0:
            throttle = float(action[0])
            brake = 0
        else:
            brake = -float(action[0])
            throttle = 0

        # self.car.apply_control(carla.VehicleControl(throttle=throttle,
        #                                             brake=brake,
        #                                             steer=float(action[1])))

        action[1] = action[1] * 0.5
        self.previous_action = action

        # TODO (Ruben) OJO, mandando los comandos con ROS via motors cambia el comportamiento del
        #  Entrenamiento (quizá añada un retardo)
        #self.motors.sendThrottle(throttle)
        #self.motors.sendBrake(brake)
        #self.motors.sendSteer(action[1])
        self.car.apply_control(carla.VehicleControl(throttle=throttle,
                                                    brake=brake,
                                                    steer=float(action[1])))
        # print(str(action))
        # print("----")

        # TODO (Ruben) There must be a better way to know how far car is
        #  from center of lane in inference
        # dists = np.mean(state[:7])  # Take 7 elements, apply abs, then mean
        dists = np.mean(state)
        state = { "speed" : speed, "distances": dists}
        self.tensorboard.update_state(state, self.step)

        self.avg_speed = self.avg_speed + (speed - self.avg_speed) / self.step

        action_time = time.time()
        self.tensorboard.update_times(action_time - perception_time, "action")

        self.update_frame('frame_0', image)
        self.update_frame('frame_1', image_processed)
        # self.update_frame('frame_2', image_2)
        # self.update_frame('frame_3', image_3)
        self.update_pose(self.pose.getPose3d())
        #print(self.pose.getPose3d())

        display_time = time.time()
        self.tensorboard.update_times(display_time - action_time, "display")


    def calculate_curvature_from(self, state):
        x = np.array(self.x_row)
        y = np.array(state)

        coefficients = np.polyfit(x, y, 2)  # Returns [a, b, c]
        a, b, c = coefficients

        x_mid = x[2]  # Use the middle point
        y_prime = 2 * a * x_mid + b
        y_double_prime = 2 * a
        curvature = abs(y_double_prime) / ((1 + y_prime ** 2) ** (3 / 2))
        return curvature