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

class LaneDetector():

    def __init__(self, car, map, x_row):
        self.detection_mode = "carla_perfect"
        self.sync_mode = True
        self.show_images = False
        self.car = car
        self.map = map
        self.x_row = x_row
        self.no_detected = [[0]] * len(x_row)

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
            camera_transform = carla.Transform(carla.Location(x=0.14852, y=0.0, z=2.5),
                                               carla.Rotation(pitch=-3.248, yaw=-0.982, roll=0.0))

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
        # Assuming sensor_data is the image obtained from the sensor
        # Convert sensor_data to a numpy array or PIL Image if needed
        # For example, if sensor_data is a numpy array:
        # sensor_data = Image.fromarray(sensor_data)
        sensor_data = np.array(sensor_data, copy=True)

        # Get the current width and height
        height = sensor_data.shape[0]
        width = sensor_data.shape[1]

        # Calculate the new height to maintain the aspect ratio
        new_height = int((new_width / width) * height)

        resized_img = Image.fromarray(sensor_data).resize((new_width, new_height))

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        resized_img_np = np.array(resized_img)

        return resized_img_np


    def detect_lane_detector(self, raw_image):
        image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = torch.softmax(self.lane_model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output


    def detect_yolop(self, raw_image):
        # Run inference
        img = self.transform(raw_image)
        img = img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        det_out, da_seg_out, ll_seg_out = self.yolop_model(img)

        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bicubic')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        return ll_seg_mask


    def lane_detection_overlay(self, image, left_mask, right_mask):
        res = np.copy(image)
        # We show only points with probability higher than 0.5
        res[left_mask > 0.5, :] = [255, 0, 0]
        res[right_mask > 0.5, :] = [0, 0, 255]
        return res


    def post_process(self, ll_segment):
        ''''
        Lane line post-processing
        '''
        # ll_segment = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # ll_segment = morphological_process(ll_segment, kernel_size=20, func_type=cv2.MORPH_CLOSE)
        # return ll_segment
        # ll_segment = morphological_process(ll_segment, kernel_size=4, func_type=cv2.MORPH_OPEN)
        # ll_segment = morphological_process(ll_segment, kernel_size=8, func_type=cv2.MORPH_CLOSE)

        # Step 1: Create a binary mask image representing the trapeze
        mask = np.zeros_like(ll_segment)
        # pts = np.array([[300, 250], [-500, 600], [800, 600], [450, 260]], np.int32)
        pts = np.array([[280, 100], [-150, 600], [730, 600], [440, 100]], np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))  # Fill trapeze region with white (255)

        # Step 2: Apply the mask to the original image
        ll_segment_masked = cv2.bitwise_and(ll_segment, mask)
        # Apply the exclusion mask to ll_segment
        ll_segment_excluding_mask = cv2.bitwise_not(mask)
        ll_segment_excluded = cv2.bitwise_and(ll_segment, ll_segment_excluding_mask)
        self.display_image(ll_segment_excluded) if self.show_images else None
        self.display_image(ll_segment_masked) if self.show_images else None
        self.display_image(mask) if self.show_images else None

        return ll_segment_masked


    def detect_lines(self, raw_image):
        # if self.detection_mode == 'programmatic':
        #     gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        #     # mask_white = cv2.inRange(gray, 200, 255)
        #     # mask_image = cv2.bitWiseAnd(gray, mask_white)
        #     blur = cv2.GaussianBlur(gray, (5, 5), 0)
        #     ll_segment = cv2.Canny(blur, 50, 100)
        #     cv2.imshow("raw", ll_segment)
        #     processed = self.post_process(ll_segment)
        #     lines = self.post_process_hough_programmatic(processed)
        if self.detection_mode == 'yolop':
            with torch.no_grad():
                ll_segment = (self.detect_yolop(raw_image) * 255).astype(np.uint8)
            # processed = self.post_process(ll_segment)
            lines = self.post_process_hough_yolop(ll_segment)
        elif self.detection_mode == 'lane_detector_v2':
            with torch.no_grad():
                ll_segment, left_mask, right_mask = self.detect_lane_detector(raw_image)[0]
            ll_segment = np.zeros_like(raw_image)
            ll_segment = self.lane_detection_overlay(ll_segment, left_mask, right_mask)
            cv2.imshow("raw", ll_segment) if self.sync_mode and self.show_images else None
            # Extract blue and red channels
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

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
            cv2.imshow("raw", ll_segment) if self.sync_mode and self.show_images else None
            # Extract blue and red channels
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            ll_segment_left = self.post_process_hough_lane_det_poly(blue_channel)
            ll_segment_right = self.post_process_hough_lane_det_poly(red_channel)
            ll_segment = 0.5 * ll_segment_left if ll_segment_left is not None else np.zeros_like(blue_channel)
            ll_segment = ll_segment + 0.5 * ll_segment_right if ll_segment_right is not None else ll_segment
            ll_segment = cv2.convertScaleAbs(ll_segment)
            return ll_segment.astype(np.uint8), False
        elif self.detection_mode == 'carla_perfect':
            ll_segment = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

            heigth = ll_segment.shape[0]
            width = ll_segment.shape[1]

            trafo_matrix_global_to_camera = get_matrix_global(self.car, self.trafo_matrix_vehicle_to_cam)
            if self.k is None:
                self.K = get_intrinsic_matrix(110, width, heigth)

            waypoint = self.map.get_waypoint(
                self.car.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            center_list, left_boundary, right_boundary, type_lane = create_lane_lines(waypoint, self.car)

            projected_left_boundary = project_polyline(
                left_boundary, trafo_matrix_global_to_camera, self.K).astype(np.int32)
            projected_right_boundary = project_polyline(
                right_boundary, trafo_matrix_global_to_camera, self.K).astype(np.int32)

            if (not check_inside_image(projected_right_boundary, width, heigth)
                    or not check_inside_image(projected_right_boundary, width, heigth)):
                return ll_segment
            image = np.zeros_like(ll_segment, dtype=np.uint8)
            self.draw_line_through_points(projected_left_boundary, image)
            self.draw_line_through_points(projected_right_boundary, image)
            return image

        detected_lines = self.merge_and_extend_lines(lines, ll_segment)

        # line_mask = morphological_process(line_mask, kernel_size=15, func_type=cv2.MORPH_CLOSE)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = cv2.dilate(line_mask, (15, 15), iterations=15)
        # line_mask = cv2.erode(line_mask, (5, 5), iterations=20)

        # TODO (Ruben) It is quite hardcoded and unrobust. Fix this to enable all lines and more than

        # 1 lane detection and cameras in other positions
        boundary_y = ll_segment.shape[1] * 2 // 5
        # Copy the lower part of the source image into the target image
        ll_segment[boundary_y:, :] = detected_lines[boundary_y:, :]
        ll_segment = (ll_segment // 255).astype(np.uint8)  # Keep the lower one-third of the image

        return ll_segment


    def draw_line_through_points(self, points, image):
        # Convert the points list to a format compatible with OpenCV (a numpy array)
        points = np.array(points, dtype=np.int32)  # Make sure points are integers

        # Draw a polyline through the points
        cv2.polylines(image, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        return image


    def display_image(self, ll_segment):
        # Display the image
        pil_image = Image.fromarray(ll_segment)

        # Display the image using PIL
        pil_image.show()


    def post_process_hough_yolop(self, ll_segment):
        # Step 4: Perform Hough transform to detect lines
        lines = cv2.HoughLinesP(
            ll_segment,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 60,  # Angle resolution in radians
            threshold=8,  # Min number of votes for valid line
            minLineLength=8,  # Min allowed length of line
            maxLineGap=20  # Max allowed gap between line for joining them
        )

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Draw the detected lines on the blank image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines in white (255, 255, 255)

        # Apply dilation to the line image

        edges = cv2.Canny(line_mask, 50, 100)

        # Reapply HoughLines on the dilated image
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 90,  # Angle resolution in radians
            threshold=35,  # Min number of votes for valid line
            minLineLength=15,  # Min allowed length of line
            maxLineGap=20  # Max allowed gap between line for joining them
        )
        # Sort lines by their length
        # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

        # Create a blank image to draw lines
        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Iterate over points
        for points in lines if lines is not None else []:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Postprocess the detected lines
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
        # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
        # eroded_image = cv2.erode(line_mask, kernel, iterations=1)

        return lines


    def post_process_hough_lane_det(self, ll_segment):
        # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
        # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
        cv2.imshow("preprocess", ll_segment) if self.show_images else None
        # edges = cv2.Canny(ll_segment, 50, 100)
        # Extract coordinates of non-zero points
        nonzero_points = np.argwhere(ll_segment == 255)
        if len(nonzero_points) == 0:
            return None

        # Extract x and y coordinates
        x = nonzero_points[:, 1].reshape(-1, 1)  # Reshape for scikit-learn input
        y = nonzero_points[:, 0]

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x, y)

        # Predict y values based on x
        y_pred = model.predict(x)

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Draw the linear regression line
        for i in range(len(x)):
            cv2.circle(line_mask, (x[i][0], int(y_pred[i])), 2, (255, 0, 0), -1)

        cv2.imshow("result", line_mask) if self.show_images else None

        # Find the minimum and maximum x coordinates
        min_x = np.min(x)
        max_x = np.max(x)

        # Find the corresponding predicted y-values for the minimum and maximum x coordinates
        y1 = int(model.predict([[min_x]]))
        y2 = int(model.predict([[max_x]]))

        # Define the line segment
        line_segment = (min_x, y1, max_x, y2)

        return line_segment


    def merge_and_extend_lines(self, lines, ll_segment):
        # Merge parallel lines
        merged_lines = []
        for line in lines if lines is not None else []:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Compute the angle of the line

            # Check if there is a similar line in the merged lines
            found = False
            for merged_line in merged_lines:
                angle_diff = abs(merged_line['angle'] - angle)
                if angle_diff < 20 and abs(angle) > 25:  # Adjust this threshold based on your requirement
                    # Merge the lines by averaging their coordinates
                    merged_line['x1'] = (merged_line['x1'] + x1) // 2
                    merged_line['y1'] = (merged_line['y1'] + y1) // 2
                    merged_line['x2'] = (merged_line['x2'] + x2) // 2
                    merged_line['y2'] = (merged_line['y2'] + y2) // 2
                    found = True
                    break

            if not found and abs(angle) > 25:
                merged_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'angle': angle})

        # Draw the merged lines on the original image
        merged_image = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8
        # if len(merged_lines) < 2 or len(merged_lines) > 2:
        #    print("ii")
        for line in merged_lines:
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            cv2.line(merged_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the original image with merged lines
        # cv2.imshow('Merged Lines', merged_image) if self.sync_mode and self.show_images else None

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Step 5: Perform linear regression on detected lines
        # Iterate over detected lines
        for line in merged_lines if lines is not None else []:
            # Extract endpoints of the line
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']

            # Fit a line to the detected points
            vx, vy, x0, y0 = cv2.fitLine(np.array([[x1, y1], [x2, y2]], dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate the slope and intercept of the line
            slope = vy / vx

            # Extend the line if needed (e.g., to cover the entire image width)
            extended_y1 = ll_segment.shape[0] - 1  # Bottom of the image
            extended_x1 = x0 + (extended_y1 - y0) / slope
            extended_y2 = 0  # Upper part of the image
            extended_x2 = x0 + (extended_y2 - y0) / slope

            if extended_x1 > 2147483647 or extended_x2 > 2147483647 or extended_y1 > 2147483647 or extended_y2 > 2147483647:
                cv2.line(line_mask, (int(x0), 0), (int(x0), ll_segment.shape[0] - 1), (255, 0, 0), 2)
                continue
            # Draw the extended line on the image
            cv2.line(line_mask, (int(extended_x1), extended_y1), (int(extended_x2), extended_y2), (255, 0, 0), 2)
        return line_mask


    def discard_not_confident_centers(self, center_lane_indexes):
        # Count the occurrences of each list size leaving out of the equation the non-detected
        size_counter = Counter(len(inner_list) for inner_list in center_lane_indexes)
        # Check if size_counter is empty, which mean no centers found
        if not size_counter:
            return center_lane_indexes
        # Find the most frequent size
        # most_frequent_size = max(size_counter, key=size_counter.get)

        # Iterate over inner lists and set elements to 1 if the size doesn't match majority
        result = []
        for inner_list in center_lane_indexes:
            # if len(inner_list) != most_frequent_size:
            if len(inner_list) < 1:  # If we don't see the 2 lanes, we discard the row
                inner_list = [0] * len(inner_list)  # Set all elements to 1
            result.append(inner_list)

        return result


    def calculate_center(self, mask):
        width = mask.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        # ## As we drive in the right lane, we get from right to left
        # lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from the right lane to center
        center_lane_indexes = [
            self.find_lane_center(lines[x], x) for x, _ in enumerate(lines)
        ]

        # this part consists of checking the number of lines detected in all rows
        # then discarding the rows (set to 1) in which more or less centers are detected
        # center_lane_indexes = self.discard_not_confident_centers(center_lane_indexes)

        center_lane_distances = [
            [center_image - x for x in inner_array] for inner_array in center_lane_indexes
        ]

        # Calculate the average position of the lane lines
        ## normalized distance
        distance_to_center_normalized = [
            np.array(x) / (width - center_image) for x in center_lane_distances
        ]
        return center_lane_indexes, distance_to_center_normalized


    def draw_dash(self, index, dist, ll_segment):
        ll_segment[index, dist - 1] = 255  # <-- here is the real calculated center
        ll_segment[index, dist - 3] = 255
        ll_segment[index, dist - 2] = 255
        ll_segment[index, dist - 4] = 255
        ll_segment[index, dist - 5] = 255
        ll_segment[index, dist - 6] = 255


    def add_midpoints(self, ll_segment, index, dist):
        # Set the value at the specified index and distance to 1
        self.draw_dash(index, dist, ll_segment)
        self.draw_dash(index + 2, dist, ll_segment)
        self.draw_dash(index + 1, dist, ll_segment)
        self.draw_dash(index - 1, dist, ll_segment)
        self.draw_dash(index - 2, dist, ll_segment)


    def show_ll_seg_image(self, dists, ll_segment, suffix="", name='ll_seg'):
        if self.detection_mode == "carla_perfect":
            ll_segment_int8 = ll_segment
        else:
            ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
        ll_segment_all = [np.copy(ll_segment_int8), np.copy(ll_segment_int8), np.copy(ll_segment_int8)]

        # draw the midpoint used as right center lane
        for index, dist in zip(self.x_row, dists):
            # Set the value at the specified index and distance to 1
            self.add_midpoints(ll_segment_all[0], index, dist)

        # draw a line for the selected perception points
        for index in self.x_row:
            for i in range(630):
                ll_segment_all[0][index][i] = 255

        ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
        # We now show the segmentation and center lane postprocessing
        # self.display_image(ll_segment_stacked) if self.show_images else None
        return ll_segment_stacked


    def find_lane_center(self, mask, i):
        # Find the indices of 1s in the array
        mask_array = np.array(mask)
        indices = np.where(mask_array > 0.8)[0]

        # If there are no 1s or only one set of 1s, return None
        if len(indices) < 2:
            # TODO (Ruben) For the moment we will be dealing with no detection as a fixed number
            return self.no_detected[i]

        # Find the indices where consecutive 1s change to 0
        diff_indices = np.where(np.diff(indices) > 1)[0]
        # If there is only one set of 1s, return None
        if len(diff_indices) == 0:
            return self.no_detected[i]

        interested_line_borders = np.array([], dtype=np.int8)

        for index in diff_indices:
            interested_line_borders = np.append(interested_line_borders, indices[index])
            interested_line_borders = np.append(interested_line_borders, int(indices[index + 1]))

        midpoints = self.calculate_midpoints(interested_line_borders)
        self.no_detected[i] = midpoints

        return midpoints


    def calculate_midpoints(self, input_array):
        midpoints = []
        for i in range(0, len(input_array) - 1, 2):
            midpoint = (input_array[i] + input_array[i + 1]) // 2
            midpoints.append(midpoint)
        return midpoints


    def process_image(self, image):
        raw_image = self.get_resized_image(image)

        ll_segment = self.detect_lines(raw_image)

        (
            center_lanes,
            distance_to_center_normalized,
        ) = self.calculate_center(ll_segment)

        right_lane_normalized_distances, right_center_lane = self.choose_lane(distance_to_center_normalized, center_lanes)
        image_processed = self.show_ll_seg_image(right_center_lane, ll_segment)

        return right_lane_normalized_distances, image_processed