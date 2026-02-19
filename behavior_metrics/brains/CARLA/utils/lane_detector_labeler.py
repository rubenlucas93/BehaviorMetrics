import carla
import torch
import numpy as np
from PIL import Image
import cv2
from sklearn.linear_model import LinearRegression
from brains.CARLA.utils.ground_truth.camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    check_inside_image,
    create_lane_lines,
    get_matrix_global,
    CameraGeometry,
)
from collections import Counter
import os
import json

POINTS_PER_MAP = {
    'Carla/Maps/Town02_Opt': [{'location': {'x': 181.01332092285156, 'y': 302.49774169921875, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 178.749267578125, 'roll': 0.0}}, {'location': {'x': 161.05384826660156, 'y': 302.52716064453125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.92127990722656, 'roll': 0.0}}, {'location': {'x': 141.05386352539062, 'y': 302.55462646484375, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.92127990722656, 'roll': 0.0}}, {'location': {'x': 121.05730438232422, 'y': 302.5545654296875, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 101.05730438232422, 'y': 302.5477600097656, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 81.05730438232422, 'y': 302.5409851074219, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 61.05730438232422, 'y': 302.5342102050781, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 41.055267333984375, 'y': 302.53515625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': -180.03903198242188, 'roll': 0.0}}, {'location': {'x': 21.055269241333008, 'y': 302.5487976074219, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.96095275878906, 'roll': 0.0}}, {'location': {'x': 1.594491720199585, 'y': 302.3045349121094, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 194.7761688232422, 'roll': 0.0}}, {'location': {'x': -3.3816475868225098, 'y': 287.3636474609375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.09098052978516, 'roll': 0.0}}, {'location': {'x': -3.413407802581787, 'y': 267.3636779785156, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.09098052978516, 'roll': 0.0}}, {'location': {'x': -3.440734386444092, 'y': 247.36642456054688, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.01293182373047, 'roll': 0.0}}, {'location': {'x': -3.4452500343322754, 'y': 227.36642456054688, 'z': 1.0}, 'rotation': {'pitch': 0.0}}],
    'Carla/Maps/Town06': [{'location': {'x': 181.01710510253906, 'y': 251.58737182617188, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 201.01710510253906, 'y': 251.5941925048828, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 221.01710510253906, 'y': 251.60101318359375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 241.01710510253906, 'y': 251.6078338623047, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 261.01708984375, 'y': 251.61465454101562, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 281.01708984375, 'y': 251.62147521972656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 301.01708984375, 'y': 251.62831115722656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 321.01708984375, 'y': 251.6351318359375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 341.0170593261719, 'y': 251.64195251464844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 361.0170593261719, 'y': 251.64877319335938, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 381.0170593261719, 'y': 251.6555938720703, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 401.01708984375, 'y': 251.66241455078125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 421.01708984375, 'y': 251.6692352294922, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 441.01708984375, 'y': 251.6760711669922, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}],
    'Carla/Maps/Town10HD': [{'location': {'x': 86.12206268310547, 'y': 135.372802734375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -30.75467300415039, 'roll': 0.0}}, {'location': {'x': 101.69976806640625, 'y': 119.30815887451172, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -59.33835983276367, 'roll': 0.0}}, {'location': {'x': 108.82305908203125, 'y': 98.4915771484375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -82.88043212890625, 'roll': 0.0}}, {'location': {'x': 109.33499145507812, 'y': 77.892578125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -449.6092224121094, 'roll': 0.0}}, {'location': {'x': 109.47138977050781, 'y': 57.893043518066406, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -449.6092224121094, 'roll': 0.0}}, {'location': {'x': 109.6077880859375, 'y': 37.89350891113281, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.6092529296875, 'roll': 0.0}}, {'location': {'x': 109.74417877197266, 'y': 17.893972396850586, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.6092529296875, 'roll': 0.0}}, {'location': {'x': 109.88057708740234, 'y': -2.1055612564086914, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.6092529296875, 'roll': 0.0}}, {'location': {'x': 109.9718017578125, 'y': -22.06999969482422, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 270.779296875, 'roll': 0.0}}, {'location': {'x': 104.95696258544922, 'y': -43.847347259521484, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -117.44700622558594, 'roll': 0.0}}, {'location': {'x': 89.95840454101562, 'y': -60.514007568359375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -146.5219268798828, 'roll': 0.0}}, {'location': {'x': 68.75068664550781, 'y': -67.79183959960938, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -175.59683227539062, 'roll': 0.0}}, {'location': {'x': 48.34845733642578, 'y': -67.9167251586914, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.97657775878906, 'roll': 0.0}}, {'location': {'x': 28.34845542907715, 'y': -67.90855407714844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.976577758}}]
}

class LaneDetectorLabeler:

    def __init__(self, car, map, world, x_row, camera_transform, fov, n_points):
        self.inference_distances = {
            "Carla/Maps/Town10HD": 1000,
            "Carla/Maps/Town05": 1000,
            "Carla/Maps/Town02_Opt": 1000,
            "Carla/Maps/Town01": 1000,
            "Carla/Maps/Town01_Opt": 1000,
            "Carla/Maps/Town04": 2000,
            "Carla/Maps/Town03": 2000,
            "Carla/Maps/Town06": 2000
        } # OJO que no los estás usando ahora. Los buenos están en controllerCarla
        self.n_points = n_points
        self.world = world
        self.last_valid_centers = None
        self.lane_points = None
        self.NON_DETECTED = -1
        self.detection_mode = "carla_perfect"
        self.sync_mode = True
        self.show_images = False
        self.car = car
        self.map = map
        self.x_row = x_row
        self.no_detected = [[0]] * len(x_row)
        self.fov = fov

        # Labeling configuration
        self.output_folder = "/home/ruben/Desktop/lane_detection_labels"
        self.raw_output_folder = os.path.join(self.output_folder, "raw")
        self.with_centers_output_folder = os.path.join(self.output_folder, "with_centers")
        self.labels_file = os.path.join(self.output_folder, "labels.json")
        self.frame_counter = 0
        self.labels = {}
        os.makedirs(self.raw_output_folder, exist_ok=True)
        os.makedirs(self.with_centers_output_folder, exist_ok=True)

        # Load existing labels to continue from where we left off
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                self.labels = json.load(f)
            if self.labels:
                # Get the last frame number from the keys (e.g., "frame_000123")
                last_frame_key = sorted(self.labels.keys())[-1]
                self.frame_counter = int(last_frame_key.split('_')[1]) + 1

        if self.detection_mode == 'yolop':
            from utils.yolop.YOLOP import get_net
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            # INIT YOLOP
            self.yolop_model = get_net()
            checkpoint = torch.load("/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/utils/yolop/weights/End-to-end.pth")
            self.yolop_model.load_state_dict(checkpoint['state_dict'])
        elif self.detection_mode == "lane_detector_v2":
            self.lane_model = torch.load(
                '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/models/CARLA/fastai_torch_lane_detector_model.pth')
            self.lane_model.eval()
        elif self.detection_mode == "lane_detector_v2_poly":
            self.lane_model = torch.load(
                '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/models/CARLA/fastai_torch_lane_detector_model.pth')
            self.lane_model.eval()
        elif self.detection_mode == "lane_detector":
            self.lane_model = torch.load('/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/models/CARLA/best_model_torch.pth')
            self.lane_model.eval()
        else:
            self.trafo_matrix_vehicle_to_cam = np.array(
                camera_transform.get_inverse_matrix()
            )
            self.k = None



    def choose_lane(self, distance_to_center_normalized, center_points):
        close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                              distance_to_center_normalized]
        distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
        centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
        return distances, centers


    def get_resized_image(self, sensor_data, new_width=640):
        sensor_data = np.array(sensor_data, copy=True)
        height = sensor_data.shape[0]
        width = sensor_data.shape[1]
        new_height = int((new_width / width) * height)
        resized_img = Image.fromarray(sensor_data).resize((new_width, new_height))
        resized_img_np = np.array(resized_img)
        return resized_img_np


    def detect_lane_detector(self, raw_image):
        image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = torch.softmax(self.lane_model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output


    def detect_yolop(self, raw_image):
        img = self.transform(raw_image)
        img = img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        det_out, da_seg_out, ll_seg_out = self.yolop_model(img)
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bicubic')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        return ll_seg_mask


    def lane_detection_overlay(self, image, left_mask, right_mask):
        res = np.copy(image)
        res[left_mask > 0.5, :] = [255, 0, 0]
        res[right_mask > 0.5, :] = [0, 0, 255]
        return res

    def set_inait_pose(self):
        import random
        spawn_dict = random.choice(POINTS_PER_MAP[self.map.name])
        location = carla.Location(
            x=spawn_dict['location']['x'],
            y=spawn_dict['location']['y'],
            z=spawn_dict['location']['z']
        )
        yaw_offset = random.uniform(-4.0, 4.0)
        if random.random() < 0.5:
            yaw_offset += 180.0
        new_rotation = carla.Rotation(
            pitch=spawn_dict['rotation']['pitch'],
            yaw=spawn_dict['rotation']['yaw'] + yaw_offset,
            roll=spawn_dict['rotation']['roll']
        )
        new_transform = carla.Transform(location, new_rotation)
        print(new_transform)
        self.car.set_transform(new_transform)

    def post_process(self, ll_segment):
        mask = np.zeros_like(ll_segment)
        pts = np.array([[280, 100], [-150, 600], [730, 600], [440, 100]], np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        ll_segment_masked = cv2.bitwise_and(ll_segment, mask)
        return ll_segment_masked


    def detect_lines(self, raw_image):
        if self.detection_mode == 'yolop':
            with torch.no_grad():
                ll_segment = (self.detect_yolop(raw_image) * 255).astype(np.uint8)
            lines = self.post_process_hough_yolop(ll_segment)
        elif self.detection_mode == 'lane_detector_v2':
            with torch.no_grad():
                ll_segment, left_mask, right_mask = self.detect_lane_detector(raw_image)[0]
            ll_segment = np.zeros_like(raw_image)
            ll_segment = self.lane_detection_overlay(ll_segment, left_mask, right_mask)
            blue_channel = ll_segment[:, :, 0]
            red_channel = ll_segment[:, :, 2]
            lines = []
            left_line = self.post_process_hough_lane_det(blue_channel)
            if left_line is not None:
                lines.append([left_line])
            right_line = self.post_process_hough_lane_det(red_channel)
            if right_line is not None:
                lines.append([right_line])
            ll_segment = 0.5 * blue_channel + 0.5 * red_channel
            ll_segment = cv2.convertScaleAbs(ll_segment)
        elif self.detection_mode == 'lane_detector_v2_poly':
            with torch.no_grad():
                ll_segment, left_mask, right_mask = self.detect_lane_detector(raw_image)[0]
            ll_segment = np.zeros_like(raw_image)
            ll_segment = self.lane_detection_overlay(ll_segment, left_mask, right_mask)
            blue_channel = ll_segment[:, :, 0]
            red_channel = ll_segment[:, :, 2]
            ll_segment_left = self.post_process_hough_lane_det_poly(blue_channel)
            ll_segment_right = self.post_process_hough_lane_det_poly(red_channel)
            ll_segment = 0.5 * ll_segment_left if ll_segment_left is not None else np.zeros_like(blue_channel)
            ll_segment = ll_segment + 0.5 * ll_segment_right if ll_segment_right is not None else ll_segment
            ll_segment = cv2.convertScaleAbs(ll_segment)
            return ll_segment.astype(np.uint8), False
        elif self.detection_mode == 'carla_perfect':
            ll_segment = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            height = ll_segment.shape[0]
            width = ll_segment.shape[1]
            trafo_matrix_global_to_camera = get_matrix_global(self.car, self.trafo_matrix_vehicle_to_cam)
            if self.k is None:
                self.k = get_intrinsic_matrix(self.fov, width, height)
            waypoint = self.map.get_waypoint(
                self.car.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            _, center_distance, alignment = self.get_lane_position(self.car, self.map)
            opposite = alignment < 0.5
            misalignment = (1 - abs(alignment)) * 10
            center_list, left_boundary, right_boundary, type_lane = create_lane_lines(waypoint, self.car, opposite=opposite)
            projected_left_boundary = project_polyline(
                left_boundary, trafo_matrix_global_to_camera, self.k, ll_segment.shape).astype(np.int32)
            projected_right_boundary = project_polyline(
                right_boundary, trafo_matrix_global_to_camera, self.k, ll_segment.shape).astype(np.int32)
            if (not check_inside_image(projected_right_boundary, width, height)
                    or not check_inside_image(projected_right_boundary, width, height)):
                return ll_segment, misalignment, center_distance, np.empty((0, 2)), np.empty((0, 2))
            image = np.zeros_like(ll_segment, dtype=np.uint8)
            self.draw_line_through_points(projected_left_boundary, image)
            self.draw_line_through_points(projected_right_boundary, image)
            return image, misalignment, center_distance, projected_left_boundary, projected_right_boundary
        detected_lines = self.merge_and_extend_lines(lines, ll_segment)
        boundary_y = ll_segment.shape[1] * 2 // 5
        ll_segment[boundary_y:, :] = detected_lines[boundary_y:, :]
        ll_segment = (ll_segment // 255).astype(np.uint8)
        return ll_segment

    def calculate_max_curveture_from_centers(self, center_points):
        curvatures = []
        for i in range(1, len(center_points) - 1):
            k = curvature_from_three_points(
                np.array(center_points[i - 1]),
                np.array(center_points[i]),
                np.array(center_points[i + 1])
            )
            curvatures.append(k)
        return max(curvatures) if curvatures else 0

    def get_lane_position(self, vehicle: carla.Vehicle, map: carla.Map):
        waypoint = map.get_waypoint(
            vehicle.get_transform().location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        vehicle_forward = vehicle.get_transform().get_forward_vector()
        vehicle_forward_np = np.array([vehicle_forward.x, vehicle_forward.y])
        waypoint_forward = waypoint.transform.get_forward_vector()
        waypoint_forward_np = np.array([waypoint_forward.x, waypoint_forward.y])
        vehicle_location = vehicle.get_transform().location
        waypoint_location = waypoint.transform.location
        waypoint_to_vehicle = carla.Location(
            vehicle_location.x - waypoint_location.x,
            vehicle_location.y - waypoint_location.y,
            vehicle_location.z - waypoint_location.z
        )
        waypoint_to_vehicle_np = np.array([waypoint_to_vehicle.x, waypoint_to_vehicle.y])
        lane_right_np = np.array([-waypoint_forward_np[1], waypoint_forward_np[0]])
        lane_offset = np.dot(waypoint_to_vehicle_np, lane_right_np) / np.linalg.norm(lane_right_np)
        lane_alignment = np.dot(vehicle_forward_np, waypoint_forward_np)
        return None, lane_offset, lane_alignment

    def draw_line_through_points(self, points, image):
        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed=False, color=(255, 0, 0), thickness=2)
        return image

    def get_stable_lane_lines(self, opposite: bool = False):
        if self.lane_points is None:
            wp = self.map.get_waypoint(
                self.car.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            center, _, _, last_wp = create_lane_lines(wp, opposite=opposite)
            self.lane_points = {
                "center": center.tolist(),
                "last_wp": last_wp
            }

        while len(self.lane_points["center"]) < 90:
            last_wp = self.lane_points["last_wp"]
            next_wps = last_wp.previous(1.0) if opposite else last_wp.next(1.0)
            if not next_wps:
                break
            new_wp = next_wps[0]
            self.lane_points["last_wp"] = new_wp
            center_np = carla_vec_to_np_array(new_wp.transform.location)
            self.lane_points["center"].append(center_np.tolist())

        center_arr = np.asarray(self.lane_points["center"])
        return center_arr, self.lane_points["last_wp"]

    def normalize_centers(self, centers):
        x_centers = centers[:, 0]
        x_centers_normalized = (x_centers / 640).tolist()
        states = x_centers_normalized
        y_centers = centers[:, 1]
        y_centers_normalized = (y_centers / 512).tolist()
        states = states + y_centers_normalized  # Returns a list
        return states, x_centers_normalized, y_centers_normalized

    def average_curvature_from_centers(self, center_points):
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

    def calculate_v_goal(self, mean_curvature, center_distance, deviated_points):
        dist_error = abs(center_distance) * 10
        close_error = 0

        mean_curv = max(0, mean_curvature - 1) * 10

        if deviated_points >= self.n_points / 2:
            mean_curv = max(0, mean_curvature - 1) * 30
            close_error = 9

        elif deviated_points >= self.n_points / 3:
            mean_curv = max(0, mean_curvature - 1) * 20
            close_error = 6

        elif deviated_points >= self.n_points / 4:
            mean_curv = max(0, mean_curvature - 1) * 10
            close_error = 3

        v_goal = max(9, 25 - (mean_curv + dist_error))
        v_goal = max(2, v_goal - (close_error))

        return v_goal

    def process_image(self, image):
        raw_image = image
        (ll_segment,
         misalignment,
         center_distance,
         center_points) = self.detect_center_line_perfect(image, n_points=self.n_points)

        centers_image = np.zeros(raw_image.shape, dtype=np.uint8)
        for index in range(len(center_points)):
            cv2.circle(centers_image, (center_points[index][0], center_points[index][1]), radius=3,
                       color=(255, 255, 0), thickness=-1)
        gray_overlay = cv2.cvtColor(centers_image, cv2.COLOR_BGR2GRAY)
        mask = gray_overlay > 10
        mask_3ch = np.stack([mask] * 3, axis=-1)
        stacked_image = np.where(mask_3ch, centers_image, raw_image)

        # --- Begin Labeling Logic ---
        if self.frame_counter % 200 == 0:
            base_filename = f"frame_{self.frame_counter:06d}"
            cv2.imwrite(os.path.join(self.raw_output_folder, f"{base_filename}_raw.png"), raw_image)
            cv2.imwrite(os.path.join(self.with_centers_output_folder, f"{base_filename}_overlay.png"), stacked_image)
            self.labels[base_filename] = center_points.tolist()
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=4)
        self.frame_counter += 1
        # --- End Labeling Logic ---

        return center_points, stacked_image, center_distance, misalignment

    def detect_center_line_perfect(self, ll_segment, n_points=20):
        ll_segment = cv2.cvtColor(ll_segment, cv2.COLOR_BGR2GRAY)
        height, width = ll_segment.shape
        trafo_matrix_global_to_camera = get_matrix_global(self.car, self.trafo_matrix_vehicle_to_cam)

        if self.k is None:
            self.k = get_intrinsic_matrix(90, width, height)

        _, center_distance, alignment = self.get_lane_position(self.car, self.map)
        opposite = alignment < 0.5
        misalignment = (1 - abs(alignment)) * 10
        center_list, _ = self.get_stable_lane_lines(opposite=opposite)

        if center_list is None or len(center_list) < 2:
            interpolated_center = np.full((n_points, 2), self.NON_DETECTED)
        else:
            projected_center = project_polyline(
                center_list, trafo_matrix_global_to_camera, self.k, image_shape=ll_segment.shape
            ).astype(np.int32)
            h, w = ll_segment.shape[:2]
            mask = np.array([0 <= pt[0] < w and 0 <= pt[1] < h for pt in projected_center])

            if np.sum(mask) < 2:
                interpolated_center = np.full((n_points, 2), self.NON_DETECTED)
            else:
                visible_center = projected_center[mask]
                for i, keep in enumerate(mask):
                    if keep:
                        first_true_index = i
                        break
                else:
                    first_true_index = len(mask)
                self.lane_points["center"] = self.lane_points["center"][first_true_index:]
                interpolated_center = interpolate_lane_points_with_roi(visible_center, n_points, height)
        return ll_segment, misalignment, center_distance, interpolated_center

def curvature_from_three_points(p1, p2, p3):
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    if area == 0:
        return 0
    return (4 * area) / (a * b * c)

def carla_vec_to_np_array(vec):
    return np.array([vec.x, vec.y, vec.z])

def interpolate_lane_points(lane_points: np.ndarray, num_points: int = 20, start_y: int = 640) -> np.ndarray:
    if lane_points.shape[0] < 2:
        return np.zeros((num_points, 2), dtype=np.float32)
    p0, p1 = lane_points[0], lane_points[1]
    dy, dx = p1[1] - p0[1], p1[0] - p0[0]
    slope = dx / dy if dy != 0 else 0
    delta_y = start_y - p0[1]
    extrapolated_x = p0[0] + slope * delta_y
    extrapolated_point = np.array([extrapolated_x, start_y])
    if start_y > p0[1]:
        lane_points = np.vstack([extrapolated_point, lane_points])
    deltas = np.diff(lane_points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)
    interp_x = np.interp(target_lengths, cumulative_lengths, lane_points[:, 0])
    interp_y = np.interp(target_lengths, cumulative_lengths, lane_points[:, 1])
    interpolated = np.stack((interp_x, interp_y), axis=1)
    return interpolated.astype(np.int32)

def interpolate_lane_points_with_roi(lane_points: np.ndarray, num_points: int, height: int) -> np.ndarray:
    if lane_points.shape[0] < 2:
        return np.full((num_points, 2), -1, dtype=np.int32)

    top_y = int(height // 1.8)
    bottom_y = height - 20
    target_y_coords = np.linspace(bottom_y, top_y, num_points)

    original_y_coords = lane_points[:, 1]
    original_x_coords = lane_points[:, 0]

    if not np.all(np.diff(original_y_coords) <= 0):
        sort_indices = np.argsort(original_y_coords)[::-1]
        original_y_coords = original_y_coords[sort_indices]
        original_x_coords = original_x_coords[sort_indices]

    xp = original_y_coords[::-1]
    fp = original_x_coords[::-1]

    # Ensure xp has at least 2 points for interpolation
    if len(xp) < 2:
         return np.full((num_points, 2), -1, dtype=np.int32)

    interpolated_x_coords = np.interp(target_y_coords, xp, fp)
    interpolated_points = np.stack((interpolated_x_coords, target_y_coords), axis=1)

    return interpolated_points.astype(np.int32)
