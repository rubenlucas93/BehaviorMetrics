#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import math
import numpy as np
import threading
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from stable_baselines3 import SAC
import carla


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

from pydantic import BaseModel
class InferenceExecutorValidator(BaseModel):
    settings: dict
    inference: dict

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
        # location = self.car.get_transform()
        # spectator = self.world.get_spectator()
        # spectator_location = carla.Transform(
        #     location.location + carla.Location(z=100),
        #     carla.Rotation(-90, location.rotation.yaw, 0))
        # spectator.set_transform(spectator_location)

        self.last_action = [0, 0]
        self.last_state = [0, 0, 0, 0, 0]
        self.camera = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')
        self.speedometer = sensors.get_speed('speedometer_0')
        self.wheel = sensors.get_wheel('wheel')

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
        self.avg_speed = 0
        self.start_time = time.time()

        # self.detection_mode = 'lane_detector'

        # self.previous_timestamp = 0
        # self.previous_image = 0

        self.previous_v = None
        self.previous_w = None
        self.previous_w_normalized = None

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/Tensorboard/sac/{time.strftime('%Y%m%d-%H%M%S')}"
        )

        args = {
            'algorithm': 'sac',
            'environment': 'simple',
            'agent': 'f1',
            'filename': 'brains/CARLA/config/config_inference_followlane_sb_sac_f1_carla.yaml'
        }

        f = open(args['filename'], "r")
        read_file = f.read()

        config_file = yaml.load(read_file, Loader=yaml.FullLoader)

        inference_params = {
            "settings": self.get_settings(config_file),
            "inference": self.get_inference(config_file, args['algorithm']),
        }

        # self.x_row = [350, 380, 410, 460, 500] # TODO Read from config
        self.x_row = self.get_states_rows(config_file)

        self.lane_detector = LaneDetector(self.car, self.map, self.x_row)

        params = InferenceExecutorValidator(**inference_params)
        inference_file = params.inference["params"]["inference_tf_model_name"]
        inference_distances = {
            "Carla/Maps/Town10HD": 2200,
            "Carla/Maps/Town06": 2500,
            "Carla/Maps/Town04": 3000
        }
        self.inference_distance = inference_distances[self.map.name]

        self.sac_agent = SAC.load(inference_file)
        action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.0 * np.ones(2))
        self.sac_agent.action_noise = action_noise


        time.sleep(2)

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

    def get_states_rows(self, config_file: dict) -> dict:
        return  config_file["states"][config_file["settings"]["states"]][0]

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def execute(self):
        if self.step == 0:
            self.start_time = time.time()

        episode_duration = time.time() - self.start_time
        distance_run = episode_duration * self.avg_speed
        if distance_run >= self.inference_distance:
            print("episode finished")
            self.car.apply_control(carla.VehicleControl(throttle=0,
                                                        brake=1,
                                                        steer=0))
            return True
        if not self.step % 200:
            print(distance_run)

        # TODO integrate with environment
        # observation, reward, done, info = self.env.step(action, self.step)
        self.step += 1

        now = time.time()
        # difference = (now - self.previous_time)
        # to_wait = 0.05 - difference
        # if to_wait > 0:
        #     time.sleep(to_wait)
        # now = time.time()

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

        states_above_threshold = sum(1 for state_value in state if state_value > 0.9)

        if states_above_threshold is None:
            states_above_threshold = 0

        missing_line = False
        if states_above_threshold == len(state):
            missing_line = True

        perception_time = time.time()
        self.tensorboard.update_times(perception_time - sensor_time, "perception")

        # speed = self.speedometer.getSpeedometer().data
        # w_angle = self.wheel.getWheelAngle()
        # state.append(speed)
        # state.append(w_angle)
        final_curvature = self.calculate_curvature_from(state)
        v = self.car.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        w_angle = self.car.get_control().steer
        state.append(speed/40)
        state.append(w_angle)
        state.append(final_curvature)
        state.append(self.previous_action[0])
        state.append(self.previous_action[1])

        # print(f"speed {speed}")
        # print(f"angle {w_angle}")

        # if not bad_perception and abs(average_difference) < 0.07:
        # if not missing_line:
        [action, _] = self.sac_agent.predict(np.array(state), deterministic=False)

        # self.motors.sendThrottle(action[0]*0.7) # A REVISAR POR QUE HAY QUE ESCALAR ESTO
        # self.motors.sendSteer(action[1])
        # self.motors.sendBrake(action[2] if action[2] > 0.5 else 0)

        if float(action[0]) > 0:
            throttle = float(action[0])
            brake = 0
        else:
            brake = -float(action[0])
            throttle = 0

        action[1] = action[1] * 0.2

        # # TODO OJO!!! Solo temporal para agente04
        if self.step <= 50:
            brake = 0
            throttle = 1
        self.car.apply_control(carla.VehicleControl(throttle=throttle,
                                                    brake=brake,
                                                    steer=float(action[1])))
        self.previous_action = action

        self.tensorboard.update_actions(action, self.step)

        # To calculate distance to center on inference we use the 5 lowest points to reduce curve noise
        # dists = np.mean(state[:7])  # Take 7 elements, apply abs, then mean
        dists = np.mean(state)
        state = { "speed" : speed, "distances": dists, "curvatures": final_curvature}

        self.tensorboard.update_state(state, self.step)

        self.avg_speed = self.avg_speed + (speed - self.avg_speed) / self.step

        # print(str(action))
        # print("----")

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
