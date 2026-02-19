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
from collections import deque

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
        self.controller = handler.controller

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
        self.v_goal_buffer = deque(maxlen=10)

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
        self.previous_states = [0] * 25

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

        if config and 'filename' in config:
            filename = config['filename']
        else:
            filename = 'brains/CARLA/followlane/config/config_inference_followlane_sb_sac_f1_carla.yaml'

        args = {
            'algorithm': 'sac',
            'environment': 'simple',
            'agent': 'f1',
            'filename': filename
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

        camera_transform = carla.Transform(carla.Location(x=-2, y=0.0, z=3),
                        carla.Rotation(pitch=-3, yaw=0, roll=0.0))
        self.fov = 90
        self.n_points = 10

        self.lane_detector = LaneDetector(self.car,
                                          self.map,
                                          self.world,
                                          self.x_row,
                                          camera_transform,
                                          self.fov,
                                          self.n_points)

        params = InferenceExecutorValidator(**inference_params)
        inference_file = params.inference["params"]["inference_tf_model_name"]
        # self.lane_detector.set_init_pose()

        self.inference_distance = self.lane_detector.inference_distances[self.map.name]

        self.sac_agent = SAC.load(inference_file)

        self.min_lidar_point = None
        self.target_speed = 0.0
        self.v_lead = 0.0
        self.add_lidar_to_vehicle(self.world, self.car)

        ## Town04 multiple
        # location = carla.Transform(
        #     carla.Location(
        #         x=389.385938,
        #         y=-179.152158,
        #         z=1.457793,
        #     ),
        #     carla.Rotation(
        #         pitch=0.082516,
        #         yaw=270.889893,
        #         roll=0.078263,
        #     ),
        # )
        #
        # self.car.set_transform(location)

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

    def add_lidar_to_vehicle(
            self,
            world,
            vehicle,
            lidar_range=100.0,
            channels=32,
            rotation_frequency=10.0,
            points_per_second=200000,
    ):
        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute('range', '80')
        lidar_bp.set_attribute('channels', '16')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('points_per_second', '100000')

        lidar_bp.set_attribute('upper_fov', '1.5')
        lidar_bp.set_attribute('lower_fov', '-9.0')
        lidar_bp.set_attribute('horizontal_fov', '40.0')

        lidar_transform = carla.Transform(
            carla.Location(x=1.5, z=1.2),
            carla.Rotation(pitch=-1, yaw=0, roll=0)
        )

        self.lidar_sensor = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=vehicle
        )

        self.lidar_sensor.listen(self.process_lidar_data)

    def lidar_point_to_world(self, lidar_detection):
        # Get the sensor's transformation
        sensor_transform = self.lidar_sensor.get_transform()
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
        min_distance = float('inf')
        min_point = None

        for d in data:
            p = d.point

            # Distance
            r = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
            if r < 2.0:
                continue

            # Front only
            if p.x < 3.0:
                continue

            # Angular cone (≈ 12° total)
            azimuth = math.degrees(math.atan2(p.y, p.x))
            if abs(azimuth) > 6.0:
                continue

            # Vertical filtering
            if p.z < -0.3:
                continue  # ground
            if p.z > 0.8:
                continue  # trees / hood

            if r < min_distance:
                min_distance = r
                min_point = p

        self.lidar_front_distance = min_distance if min_point else 100.0
        if min_point:
            self.min_lidar_point = (min_point.x, min_point.y)
        else:
            self.min_lidar_point = None

        dt = 0.05
        v_lead = self.estimate_v_lead(min_distance, self.avg_speed, dt)
        self.target_speed = self.calculate_rss_speed(min_distance, self.avg_speed, v_lead)

    def calculate_rss_speed(self, d_measured, v_ego, v_lead):
        # RSS Parameters
        rho = 0.5
        a_max_accel = 2.0
        a_min_brake = 3.0
        a_max_brake = 5.0
        speed_limit = 25.0

        # 1. Calculate minimum safe distance (RSS)
        term1 = v_ego * rho
        term2 = 0.5 * a_max_accel * (rho ** 2)
        term3 = (v_ego + rho * a_max_accel) ** 2 / (2 * a_min_brake)
        term4 = (v_lead ** 2) / (2 * a_max_brake)

        d_min = term1 + term2 + term3 - term4
        d_min = max(4.0, d_min)  # hard safety floor

        # 2. Comfort distance
        d_comfort = d_min + 15.0

        if self.step % 10 == 1:
            print(
                f"DEBUG | Dist: {d_measured:.2f} | d_min: {d_min:.2f} | "
                f"EgoV: {v_ego:.2f} | LeadV: {v_lead:.2f}"
            )

        # 3. Decision Logic
        # dt = 0.05
        if d_measured > d_comfort:
            # Zone 1: Clear road
            return speed_limit
            # return min(speed_limit, v_ego + a_max_accel * dt)

        elif d_measured > d_min:
            # Zone 2: Smooth deceleration toward lead vehicle speed
            ratio = (d_measured - d_min) / (d_comfort - d_min)

            # 🔧 FIX: interpolate toward CURRENT speed, not speed limit
            target_speed = v_lead + ratio * (v_ego - v_lead)

            # Never accelerate in this zone
            return min(v_ego, target_speed)

        else:
            # Zone 3: Emergency stop
            return 0.0

    def estimate_v_lead(self, current_dist, v_ego, dt):
        # Guard against invalid distance or zero time
        if current_dist >= 100 or dt <= 0:
            return v_ego  # Assume same speed if no car detected

        if not hasattr(self, 'prev_dist') or self.prev_dist is None:
            self.prev_dist = current_dist
            self.v_lead = v_ego
            return v_ego

        v_rel = (current_dist - self.prev_dist) / dt
        measured_v_lead = v_ego + v_rel

        # NaN Guard: If measured_v_lead is nan, keep previous
        if math.isnan(measured_v_lead):
            return self.v_lead

        self.v_lead = (0.7 * self.v_lead) + (0.3 * measured_v_lead)
        self.prev_dist = current_dist
        return max(0, self.v_lead)


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
        if self.controller.lap_completed(distance_run):
            print("episode finished")
            self.controller.stop_car()
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

        [action, _] = self.sac_agent.predict(np.array(self.previous_states), deterministic=True)

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

        centers, image_processed, center_distance,_ = self.lane_detector.process_image(image)

        perception_time = time.time()
        self.tensorboard.update_times(perception_time - sensor_time, "perception")

        # speed = self.speedometer.getSpeedometer().data
        # w_angle = self.wheel.getWheelAngle()
        # state.append(speed)
        # state.append(w_angle)
        # final_curvature = self.lane_detector.calculate_max_curveture_from_centers(state)

        v = self.car.get_velocity()
        speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        w_angle = self.car.get_control().steer

        state, x_centers_normalized, y_normalized = self.lane_detector.normalize_centers(centers)
        half_image = len(x_centers_normalized)//2
        close_points_dev = abs(x_centers_normalized[0] - x_centers_normalized[half_image])
        x_normalized = np.array(x_centers_normalized)
        deviated_points = np.sum(np.abs(x_normalized - 0.5) > 0.1)

        mean_curvature = self.lane_detector.average_curvature_from_centers(centers)
        v_goal_now = self.lane_detector.calculate_v_goal(mean_curvature, center_distance, deviated_points)
        self.v_goal_buffer.append(v_goal_now)
        v_goal = sum(self.v_goal_buffer) / len(self.v_goal_buffer)
        if self.min_lidar_point:
            v_goal = min(v_goal, self.target_speed)

        state.append(speed / 25)
        state.append(w_angle)
        # state.append(final_curvature)
        # state.append(misalignment)
        state.append(action[0])
        state.append(action[1])
        # state.append(close_points_dev)
        # state.append(deviated_points)
        state.append(v_goal / 25)
        state.append(self.lidar_front_distance/100)

        self.previous_states = state

        self.tensorboard.update_actions(action, self.step)

        # To calculate distance to center on inference we use the 5 lowest points to reduce curve noise
        dists = np.mean(state[:2])  # Take 7 elements, apply abs, then mean
        # dists = np.mean(state)
        # print(dists)
        # print(state[:10])
        state = { "speed" : speed, "distances": center_distance, "curvatures": mean_curvature}
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
        return False