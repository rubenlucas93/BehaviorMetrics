#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import math
import numpy as np
import threading
import time
from datetime import datetime

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

import torch
import cv2

from brains.CARLA.utils.perceptions_benchmark_automatic import  detect_lanes_yolop_v2_hybrid_agent, detect_lanes_yolop_v2_drivable_agent
# from brains.CARLA.utils.lanes_detector_yolopv2 import LaneDetector as LaneDetectorYolop
# from brains.CARLA.utils.lanes_detector import LaneDetector as LaneDetector
# from brains.CARLA.utils.lane_detector_paper_1_backup import LaneDetector

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

        self.last_detected_point = None
        self.prev_left_contour = None
        self.prev_right_contour = None
        self.detection_mode = 'yolop_v2'
        self.visualize = False
        self.NON_DETECTED = -1
        self.appended_states = 10

        self.client.set_timeout(10.0)
        print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")
        self.controller = handler.controller

        self.world = self.client.get_world()
        clear_noon = carla.WeatherParameters(
            sun_azimuth_angle=0.0,
            cloudiness=60.0,  # Clouds scatter light, making shadows softer/weaker
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            fog_density=0.0,  # Keep it clear for lane detection
            wetness=0.0,
            sun_altitude_angle=90.0,  # Directly overhead = minimal shadows
        )

        self.world.set_weather(clear_noon)
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
            filename = 'brains/CARLA/followlane_yolop/config/config_inference_followlane_sb_sac_f1_carla.yaml'

        print(filename)
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

        # self.lane_detector_yolop = LaneDetectorYolop(self.n_points)
        # self.lane_detector = LaneDetector(self.car,
        #                                   self.map,
        #                                   self.world,
        #                                   self.x_row,
        #                                   camera_transform,
        #                                   self.fov,
        #                                   self.n_points)


        params = InferenceExecutorValidator(**inference_params)
        inference_file = params.inference["params"]["inference_tf_model_name"]
        # self.lane_detector.set_init_pose()

        self.sac_agent = SAC.load(inference_file)

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
        # image = self.camera.getImage().data
        # centers, image_processed, center_distance, raw_image, paths_image = self.lane_detector_yolop.process_image(image)

        print("SAC initialized!")
        # time.sleep(2)

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

    def detect_lines(self, raw_image, prev_left_contour=None, prev_right_contour=None):
        # YOLOPv2 gets a dedicated, simpler, more direct pipeline
        if self.detection_mode == 'yolop_v2':
            with torch.no_grad():
                (ll_segment, distance_to_center, center_lanes, ll_seg_out,
                 raw_detection_image, extended_image, paths_image, points_for_extension, candidates_viz_from_tracker,
                 best_left_raw, best_right_raw) = detect_lanes_yolop_v2_hybrid_agent(
                    raw_image,
                    reference_point=self.last_detected_point,
                    prev_left_contour=prev_left_contour,
                    prev_right_contour=prev_right_contour
                )
        elif self.detection_mode == 'yolop_v2_drivable':
            with torch.no_grad():
                (ll_segment, distance_to_center, center_lanes, ll_seg_out,
                 raw_detection_image, extended_image, paths_image, points_for_extension, candidates_viz_from_tracker,
                 best_left_raw, best_right_raw) = detect_lanes_yolop_v2_drivable_agent(
                    raw_image,
                    prev_right_contour=prev_right_contour
                )

        if self.detection_mode in ['yolop_v2', 'yolop_v2_drivable']:
            if center_lanes is not None and len(center_lanes) > 0:
                self.last_detected_point = center_lanes[-1]

            # We DO NOT call show_image here anymore!
            # The environment main loop (apply_step) will handle showing the image
            # after the Kalman Filter has had a chance to draw its predictions on it.

            self.masked_ll_segment = ll_segment.copy()

            if self.visualize:
                overlay_image = raw_image.copy()

                # --- LETTERBOX UN-PADDING FOR MASK ---
                # Replicate the letterbox math to accurately un-pad the square/padded mask back to the original 16:9 image
                h_orig, w_orig = overlay_image.shape[:2]
                h_padded, w_padded = ll_segment.shape[:2]

                scale = min(w_padded / w_orig, h_padded / h_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                x_offset = (w_padded - new_w) // 2
                y_offset = (h_padded - new_h) // 2

                # 1. Crop the padding out of the padded mask
                cropped_mask = ll_segment[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

                # 2. Resize the cropped mask back to the EXACT original raw image dimensions
                visible_mask_resized = cv2.resize(cropped_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                mask = visible_mask_resized > 128
                if np.any(mask):
                    overlay_image[mask] = [0, 255, 255]

                # The points returned from the perception script are already correctly scaled to original image space
                for point in center_lanes:
                    cv2.circle(overlay_image, (int(point[0]), int(point[1])), 5, (255, 0, 255), -1)

                self.show_image('overlayed_image', overlay_image)

            return ll_segment, distance_to_center, center_lanes, paths_image, candidates_viz_from_tracker, ll_seg_out, None, None, best_left_raw, best_right_raw


    def normalize_centers(self, centers, vis_image):
        centers = np.array(centers)

        cam_h, cam_w, channels = vis_image.shape

        if len(centers) == 0:
            x_centers = np.full(self.num_points, self.NON_DETECTED, dtype=float)
            y_centers = np.full(self.num_points, self.NON_DETECTED, dtype=float)
        elif self.detection_mode == "lane_detector_v2":
            flat_centers = [item[0] for item in centers]
            x_centers = np.array(flat_centers)
            y_centers = np.array(self.x_row)
        else:
            x_centers = centers[:, 0].astype(float)
            y_centers = centers[:, 1].astype(float)

        # Protect NON_DETECTED (-1) and pure 0 flags from being mathematically normalized
        # so the RL agent can still recognize them as distinct failure states.
        x_mask = (x_centers != self.NON_DETECTED) & (x_centers != 0)
        y_mask = (y_centers != self.NON_DETECTED) & (y_centers != 0)

        x_centers_normalized = x_centers.copy()
        x_centers_normalized[x_mask] = x_centers_normalized[x_mask] / cam_w

        y_centers_normalized = y_centers.copy()
        y_centers_normalized[y_mask] = y_centers_normalized[y_mask] / cam_h

        x_centers_normalized = x_centers_normalized.tolist()
        y_centers_normalized = y_centers_normalized.tolist()

        states = x_centers_normalized + y_centers_normalized # Returns a list
        return states, x_centers_normalized, y_centers_normalized

    def calculate_v_goal(self, cumulative_angle, center_distance, deviated_points, current_v):

        # --- 1️⃣ Normalize curvature to expected range ---
        # Assume cumulative_angle normally in [0, 1.8]
        angle_norm = np.clip(cumulative_angle / 1.8, 0.0, 1.0)

        # --- 2️⃣ Base speed reduction from curvature ---
        v_base = 25.0 - 17.0 * angle_norm  # gives approx 25 → 8 range

        # --- 3️⃣ Smooth deviation penalty ---
        deviation_ratio = deviated_points / self.appended_states
        deviation_penalty = 6.0 * deviation_ratio ** 2  # quadratic = smooth

        # --- 4️⃣ Combine ---
        v_goal = v_base - deviation_penalty

        v_goal = np.clip(v_goal, 8.0, 25.0)

        # if self.step_count % 20 == 0:
            # print(f"----------------------------------------")
            # print(f"monitoring last modification bro! dist_minus -> {dist_error}")
            # print(f"monitoring last modification bro! deviation_points -> {deviated_points}")
            # print(f"monitoring last modification bro! curv_minus -> {cumulative_angle}")
            # print(f"monitoring last modification bro! v_goal -> {v_goal}")
            # print(f"----------------------------------------")
        return v_goal


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

        steer = float(action[1])

        if self.step < 5:
            throttle = 0
            steer = 0

        self.car.apply_control(carla.VehicleControl(throttle=throttle,
                                                    brake=brake,
                                                    steer=steer))

        image = self.camera.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        sensor_time = time.time()
        self.tensorboard.update_times(sensor_time - self.previous_time, "sensor")

        # detected_center_lanes, vis_image, distance_to_center, ll_segment, _ = self.lane_detector_yolop.process_image(image_2)
        # centers_l, image_processed_l, center_distance_l, _ = self.lane_detector.process_image(image)

        # cam_h, cam_w, channels = image_2.shape
        # raw_image = self.lane_detector_yolop.get_resized_image(image_2, cam_h, cam_w)

        (ll_segment, distance_to_center, detected_center_lanes, vis_image,
         candidates_viz, ll_seg_out, _, _, extended_left, extended_right) = self.detect_lines(
            image_2,
            prev_left_contour=self.prev_left_contour,
            prev_right_contour=self.prev_right_contour
        )
        # Tracking memory
        self.prev_left_contour = extended_left
        self.prev_right_contour = extended_right

        final_curvature = calculate_max_curveture_from_centers(detected_center_lanes)
        mean_curvature = average_curvature_from_centers(detected_center_lanes)
        states, x_centers_normalized, y_normalized = self.normalize_centers(detected_center_lanes, candidates_viz)

        # perception_time = time.time()
        # self.tensorboard.update_times(perception_time - sensor_time, "perception")

        # speed = self.speedometer.getSpeedometer().data
        # w_angle = self.wheel.getWheelAngle()
        # state.append(speed)
        # state.append(w_angle)
        # final_curvature = self.lane_detector.calculate_max_curveture_from_centers(state)
        v = self.car.get_velocity()
        speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        w_angle = self.car.get_control().steer

        x_normalized = np.array(x_centers_normalized)
        deviated_points = np.sum(np.abs(x_normalized - 0.5) > 0.1)

        v_goal_now = self.calculate_v_goal(mean_curvature, abs(distance_to_center), deviated_points, speed)

        self.v_goal_buffer.append(v_goal_now)
        v_goal = sum(self.v_goal_buffer) / len(self.v_goal_buffer)


        # print("points")
        # print(states)

        # print("speed")
        # print(speed)
        #
        # print("v_goal")
        # print(v_goal)

        # states = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5015625, 0.5015625, 0.5015625, 0.5015625, 0.486328125, 0.509765625, 0.53515625, 0.560546875, 0.5859375, 0.609375, 0.634765625, 0.66015625, 0.685546875, 0.7109375]
        # v_goal = 19.68927906822399
        # action = [0, 0]
        # speed = 0
        # w_angle = 0

        states.append(speed / 25)
        states.append(v_goal / 25)
        # state.append(final_curvature)
        # state.append(misalignment)
        states.append(w_angle)
        states.append(action[0])
        states.append(action[1])
        # state.append(close_points_dev)
        # state.append(deviated_points)

        # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} [followlane_yolop] State: {state}")

        self.previous_states = states

        # self.tensorboard.update_actions(action, self.step)

        # To calculate distance to center on inference we use the 5 lowest points to reduce curve noise
        # dists = np.mean(state[:2])  # Take 7 elements, apply abs, then mean
        # dists = np.mean(state)
        # print(dists)
        # print(state[:10])
        # state = { "speed" : speed, "distances": center_distance, "curvatures": mean_curvature}
        # self.tensorboard.update_state(state, self.step)

        self.avg_speed = self.avg_speed + (speed - self.avg_speed) / self.step

        # print(str(action))
        # print("----")

        action_time = time.time()
        # self.tensorboard.update_times(action_time - perception_time, "action")

        # self.update_frame('frame_2', image_processed_y)
        self.update_frame('frame_0', vis_image)
        self.update_frame('frame_1', image_2)
        self.update_frame('frame_2', candidates_viz)
        # self.update_pose(self.pose.getPose3d())
        #print(self.pose.getPose3d())
        # display_time = time.time()
        # self.tensorboard.update_times(display_time - action_time, "display")
        return False

def curvature_from_three_points(p1, p2, p3):
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    if area == 0:
        return 0
    return (4 * area) / (a * b * c)

def calculate_max_curveture_from_centers(center_points):
    curvatures = []
    for i in range(1, len(center_points) - 1):
        k = curvature_from_three_points(
            np.array(center_points[i - 1]),
            np.array(center_points[i]),
            np.array(center_points[i + 1])
        )
        curvatures.append(k)
    return max(curvatures) if curvatures else 0


def average_curvature_from_centers(center_points):
    total_angle = 0.0

    for i in range(1, len(center_points) - 1):
        p1 = np.array(center_points[i - 1])
        p2 = np.array(center_points[i])
        p3 = np.array(center_points[i + 1])

        v1 = p2 - p1
        v2 = p3 - p2

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            continue  # skip degenerate segment

        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cos_theta)  # radians

        total_angle += angle

    return total_angle  # in radians