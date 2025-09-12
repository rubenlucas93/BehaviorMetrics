#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import math
import numpy as np
import threading
import time
import carla
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare
)
from utils.constants import DATASETS_DIR, ROOT_PATH
from brains.CARLA.utils.lanes_detector import LaneDetector
from brains.CARLA.utils.modified_tensorboard import ModifiedTensorBoard


GENERATED_DATASETS_DIR = ROOT_PATH + '/' + DATASETS_DIR


def draw_lane_offset(world, vehicle, map):
    vehicle_transform = vehicle.get_transform()
    vehicle_loc = vehicle_transform.location

    # Get lane center
    waypoint = map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_center = waypoint.transform.location

    # Draw a line from lane center to vehicle
    world.debug.draw_line(lane_center + carla.Location(z=0.3),
                          vehicle_loc + carla.Location(z=0.3),
                          thickness=0.05,
                          color=carla.Color(255, 0, 0),
                          life_time=0.5)

    # Optional: draw vehicle forward direction
    forward_vector = vehicle_transform.get_forward_vector()
    world.debug.draw_arrow(vehicle_loc + carla.Location(z=0.5),
                           vehicle_loc + forward_vector * 2 + carla.Location(z=0.5),
                           thickness=0.02, arrow_size=0.2,
                           color=carla.Color(0, 0, 255), life_time=0.5)


def draw_lane_center(world, map, vehicle, length=50, interval=1.5):
    location = vehicle.get_transform().location
    waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

    for i in range(length):
        world.debug.draw_point(waypoint.transform.location + carla.Location(z=0.3),
                               size=0.1, color=carla.Color(0, 255, 0), life_time=0.5)
        waypoint = waypoint.next(interval)[0]  # move forward


class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        self.camera = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')

        self.pose = sensors.get_pose3d('pose3d_0')
        self.one_time_code = False

#        self.bird_eye_view = sensors.get_bird_eye_view('bird_eye_view_0')

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

        # self.previous_timestamp = 0
        # self.previous_image = 0

        self.previous_v = None
        self.previous_w = None
        self.previous_w_normalized = None
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        self.world = client.get_world()
        self.map = self.world.get_map()

        time.sleep(3)
        self.car = self.world.get_actors().filter('vehicle.*')[0]

        self.counter = 0

        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        traffic_speed_limits = self.world.get_actors().filter('traffic.speed_limit*')
        print(traffic_speed_limits)
        for traffic_light in traffic_lights:
            traffic_light.set_green_time(20000)
            traffic_light.set_state(carla.TrafficLightState.Green)

        for speed_limit in traffic_speed_limits:
            success = speed_limit.destroy()
            print(success)
        self.x_row = [275, 280, 285, 295, 320, 360, 390, 420, 450, 480]


        # route = ["Straight", "Straight", "Straight", "Straight", "Straight",
        # "Straight", "Straight", "Straight", "Straight", "Straight",
        # "Straight", "Straight", "Straight", "Straight", "Straight",
        # "Straight", "Straight", "Straight", "Straight", "Straight",
        # "Straight", "Straight", "Straight", "Straight", "Straight",
        # "Straight", "Straight", "Straight", "Straight", "Straight"]
        # traffic_manager = client.get_trafficmanager()
        # traffic_manager.set_route(self.car, route)
        self.car.set_autopilot(True)
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
        self.step = 0
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/Tensorboard/autopilot/{time.strftime('%Y%m%d-%H%M%S')}"
        )

        

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
        self.step += 1

        draw_lane_center(self.world, self.map, self.car)
        draw_lane_offset(self.world, self.car, self.map)

        image = self.camera.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        # bird_eye_view_1 = self.bird_eye_view.getImage(self.vehicle)
        # bird_eye_view_1 = cv2.cvtColor(bird_eye_view_1, cv2.COLOR_BGR2RGB)

        #print(self.bird_eye_view.getImage(self.vehicle))
        centers, image_processed, center_distance = self.lane_detector.process_image(image)
        state, x_centers_normalized, y_normalized = self.lane_detector.normalize_centers(centers)

        # print(f"{center_distance} - {x_centers_normalized[0]}, {x_centers_normalized[1]}, {x_centers_normalized[2]}")
        # curvature = self.lane_detector.calculate_curvature_from(state)

        # self.tensorboard.update_times(state[0], "state_1")
        # self.tensorboard.update_times(state[1], "state_2")
        # self.tensorboard.update_times(state[2], "state_3")
        # self.tensorboard.update_times(state[3], "state_4")
        # self.tensorboard.update_times(state[4], "state_5")
        # self.tensorboard.update_times(state[5], "state_6")
        # self.tensorboard.update_times(state[6], "state_7")
        # self.tensorboard.update_times(state[7], "state_8")
        # self.tensorboard.update_times(state[8], "state_9")
        # self.tensorboard.update_times(state[9], "state_10")
        # self.tensorboard.update_times(curvature, "final_curvature")
        self.tensorboard.update_times(center_distance, "center_distance")

        if not self.one_time_code:
            self.lane_detector.move_car_to_lane_center(self.car, self.world, self.map)
            self.one_time_code = True
        # if self.step % 100 == 0:
        #     #Cálculos cutres de la velocidad objetivo
        #     curv = (abs(curvature)*120000)
        #     error = abs(center_distance) * 15
        #     print("-----")
        #     print(curv)
        #     print(error)
        #     print(max(5, 25 - (curv + error)))

        self.update_frame('frame_1', image)
        self.update_frame('frame_2', image_1)
        self.update_frame('frame_3', image_3)

        self.update_frame('frame_0', image_processed)
        

        self.update_pose(self.pose.getPose3d())
        #print(self.pose.getPose3d())

