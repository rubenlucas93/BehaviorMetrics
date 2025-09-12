#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import carla
import math
import numpy as np
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from stable_baselines3 import DDPG
import carla

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
        self.detection_mode = "carla_perfect"
        self.camera = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')
        self.speedometer = sensors.get_speed('speedometer_0')
        self.wheel = sensors.get_wheel('wheel')
        self.v_goal_buffer = deque(maxlen=10)

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

        self.sync_mode = True
        self.show_images = False
        # self.detection_mode = 'lane_detector'

        # self.previous_timestamp = 0
        # self.previous_image = 0

        self.previous_states = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/Tensorboard/ddpg/{time.strftime('%Y%m%d-%H%M%S')}"
        )

        args = {
            'algorithm': 'ddpg',
            'environment': 'simple',
            'agent': 'f1',
            'filename': 'brains/CARLA/config/config_inference_followlane_sb_ddpg_3_f1_carla.yaml'
        }

        f = open(args['filename'], "r")
        read_file = f.read()

        config_file = yaml.load(read_file, Loader=yaml.FullLoader)

        inference_params = {
            "settings": self.get_settings(config_file),
            "inference": self.get_inference(config_file, args['algorithm']),
        }

        self.x_row = self.get_states_rows(config_file)

        params = InferenceExecutorValidator(**inference_params)
        inference_file = params.inference["params"]["inference_tf_model_name"]

        camera_transform = carla.Transform(carla.Location(x=-1, y=0.0, z=2),
                        carla.Rotation(pitch=-2.5, yaw=0, roll=0.0))
        self.lane_detector = LaneDetector(self.car, self.map, self.world, self.x_row, camera_transform, 130)
        self.inference_distance = self.lane_detector.inference_distances[self.map.name]

        self.ddpg_agent = DDPG.load(inference_file)
        action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.0 * np.ones(2))
        self.ddpg_agent.action_noise = action_noise
        self.lane_detector.set_init_pose()

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
            target_velocity = carla.Vector3D(
                x= 0,
                y= 0,
                z= 0  # Typically 0 unless you want vertical motion
            )
            self.car.set_target_velocity(target_velocity)
            return True
        # if not self.step % 200:
        #     print(distance_run)

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

        [action, _] = self.ddpg_agent.predict(np.array(self.previous_states), deterministic=True)

        # self.motors.sendThrottle(action[0]*0.7) # A REVISAR POR QUE HAY QUE ESCALAR ESTO
        # self.motors.sendSteer(action[1])
        # self.motors.sendBrake(action[2] if action[2] > 0.5 else 0)

        if float(action[0]) > 0:
            throttle = float(action[0])
            brake = 0
        else:
            brake = -float(action[0])
            throttle = 0

        self.car.apply_control(carla.VehicleControl(throttle=throttle,
                                                    brake=brake,
                                                    steer=float(action[1])))


        image = self.camera.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        sensor_time = time.time()
        self.tensorboard.update_times(sensor_time - self.previous_time, "sensor")

        centers, image_processed, center_distance = self.lane_detector.process_image(image)

        perception_time = time.time()
        self.tensorboard.update_times(perception_time - sensor_time, "perception")

        # speed = self.speedometer.getSpeedometer().data
        # w_angle = self.wheel.getWheelAngle()
        # state.append(speed)
        # state.append(w_angle)
        # final_curvature = self.lane_detector.calculate_max_curveture_from_centers(state)

        mean_curvature = self.lane_detector.average_curvature_from_centers(centers)

        state, x_centers, y_centers = self.lane_detector.normalize_centers(centers)
        # v_goal_now = self.lane_detector.calculate_v_goal(mean_curvature, center_distance, y_centers)
        v_goal_now = self.lane_detector.calculate_v_goal(mean_curvature, center_distance)

        self.v_goal_buffer.append(v_goal_now)
        v_goal = sum(self.v_goal_buffer) / len(self.v_goal_buffer)

        v = self.car.get_velocity()
        speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        w_angle = self.car.get_control().steer


        state.append(speed / 25)
        state.append(w_angle)
        # state.append(final_curvature)
        # state.append(misalignment)
        state.append(action[0])
        state.append(action[1])
        state.append(v_goal / 25)

        self.previous_states = state

        self.tensorboard.update_actions(action, self.step)

        # To calculate distance to center on inference we use the 5 lowest points to reduce curve noise
        # dists = np.mean(state[:6)  # Take 7 elements, apply abs, then mean
        # dists = np.mean(state)
        # print(dists)
        # print(state[:10])
        # state = { "speed" : speed, "distances": center_distance, "curvatures": mean_curvature}
        #
        # self.tensorboard.update_state(state, self.step)

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