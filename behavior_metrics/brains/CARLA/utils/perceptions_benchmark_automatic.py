import argparse
import math
import os, sys
import shutil
import time
from pathlib import Path
from sklearn.linear_model import LinearRegression
from statsmodels.distributions.empirical_distribution import ECDF
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from collections import Counter

print(sys.path)
import random
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# detection_mode = 'yolop'
# detection_mode = 'lane_detector'
# detection_mode = 'programmatic'

show_images = False
apply_mask = True


def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


device = select_device()
images_high = 380
upper_limit = 200
x_row = [upper_limit + 10, 270, 300, 320, images_high - 10]
NO_DETECTED = 0
THRESHOLDS_PERC = [0.1, 0.3, 0.5, 0.7, 0.9]
PERFECT_THRESHOLD = 0.9

import torchvision.transforms as transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# Ensure you have the YOLOPv2 source files in your utils folder
# 1. Standard Transforms (Keep these, they are the same for v2)
import torchvision.transforms as transforms

yolop_v2_lines_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. INIT YOLOPv2
# Enable CuDNN benchmark for faster convolutions on fixed input sizes
cudnn.benchmark = True
cudnn.deterministic = True

# Load the model directly using torch.jit or the model class
model_path = "/home/ruben/Desktop/unibotics/src/RL-Studio/rl_studio/envs/carla/utils/yolop/weights/yolopv2.pt"
yolop_v2_lines_model = torch.jit.load(
    model_path,
    map_location=device
)

# Convert to FP16 (Half Precision) if running on GPU for a ~2x speedup
half_precision = device.type != 'cpu'
if half_precision:
    yolop_v2_lines_model.half()
else:
    yolop_v2_lines_model.float()

yolop_v2_lines_model.to(device)
yolop_v2_lines_model.eval()  # CRITICAL for inference quality


def post_process(ll_segment):
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
    pts = np.array([[280, 200], [-50, 500], [630, 500], [440, 200]], np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))  # Fill trapeze region with white (255)
    cv2.imshow("applied_mask", mask) if show_images else None

    # Step 2: Apply the mask to the original image
    ll_segment_masked = cv2.bitwise_and(ll_segment, mask)
    ll_segment_excluding_mask = cv2.bitwise_not(mask)
    # Apply the exclusion mask to ll_segment
    ll_segment_excluded = cv2.bitwise_and(ll_segment, ll_segment_excluding_mask)
    cv2.imshow("discarded", ll_segment_excluded) if show_images else None

    return ll_segment_masked


def post_process_hough_lane_det_v1(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
    # edges = cv2.Canny(ll_segment, 50, 100)

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=7,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=60  # Max allowed gap between line for joining them
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
    cv2.imshow("hough", line_mask) if show_images else None

    return lines


# TODO It is not feasible for online. Since it is calling predict thrice. Optimize
def post_process_hough_lane_det(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
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

    cv2.imshow("result", line_mask) if show_images else None

    # Find the minimum and maximum x coordinates
    min_x = np.min(x)
    max_x = np.max(x)

    # Find the corresponding predicted y-values for the minimum and maximum x coordinates
    y1 = int(model.predict([[min_x]]))
    y2 = int(model.predict([[max_x]]))

    # Define the line segment
    line_segment = (min_x, y1, max_x, y2)

    return line_segment


def post_process_hough_yolop_v1(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    ll_segment = cv2.erode(ll_segment, (

    ), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
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

    # edges = cv2.Canny(line_mask, 50, 100)

    # cv2.imshow("intermediate_hough", ll_segment) if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
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
    cv2.imshow("hough", line_mask) if show_images else None

    return lines


def post_process_hough_yolop(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    cv2.imshow("preprocess", ll_segment) if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=40,  # Min number of votes for valid line
        minLineLength=10,  # Min allowed length of line
        maxLineGap=30  # Max allowed gap between line for joining them
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
    cv2.imshow("hough", line_mask) if show_images else None

    return lines


def post_process_hough_programmatic(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 60,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=10,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Draw the detected lines on the blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines in white (255, 255, 255)

    # Apply dilation to the line image

    edges = cv2.Canny(line_mask, 50, 100)

    cv2.imshow("intermediate_hough", edges) if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 60,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=13,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
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
    cv2.imshow("hough", line_mask) if show_images else None

    return lines


# global variable for temporal smoothing
prev_fit = None


def apply_clahe_bgr(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def soften_overbright_lanes(image, white_threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find very bright pixels
    mask = gray > white_threshold

    result = image.copy()

    # Reduce brightness slightly instead of eroding
    result[mask] = result[mask] * 0.6
    # result[mask] = np.power(result[mask] / 255.0, 1.2) * 255.0

    return result.astype(np.uint8)


def preprocess_image_for_yolop(raw_image, target_size=(1280, 720)):
    """
    Combines Letterboxing (Aspect Ratio Preservation) with
    Lighting Preprocessing and Tensor Transformation.
    """
    # 1. Apply Lighting Adjustments (Physics-informed preprocessing)
    # CLAHE heavily boosts local contrast between the road and faint solid lines
    # processed_image = apply_clahe_bgr(raw_image)
    processed_image = soften_overbright_lanes(raw_image)

    h, w = processed_image.shape[:2]
    target_w, target_h = target_size

    # YOLOP (and all YOLO models) requires the input dimensions to be strictly divisible by 32
    # (the maximum stride of the network's convolutions). Otherwise, tensor concatenations fail.
    target_w = ((target_w + 31) // 32) * 32
    target_h = ((target_h + 31) // 32) * 32

    # 2. Calculate Letterbox Dimensions
    # (Forces geometry to stay consistent for better right-lane detection)
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 3. Resize with High-Quality Interpolation
    resized_content = cv2.resize(processed_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 4. Create Padding (Canvas)
    # Using 114 (gray) is the standard YOLO/YOLOP neutral background
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    # Center the image on the canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_content

    # 5. Metadata for Post-Processing (Crucial for un-padding the mask)
    metadata = {
        "scale": scale,
        "offset": (x_offset, y_offset),
        "original_shape": (h, w)
    }

    # 6. Transform to Model Tensor
    # 'yolop_v2_lines_transform' typically handles Normalization and BGR->RGB
    img_tensor = yolop_v2_lines_transform(canvas).to(device)

    if 'half_precision' in globals() and half_precision:
        img_tensor = img_tensor.half()

    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return canvas, img_tensor, metadata


def run_yolop_v2_inference(raw_image):
    resized_image, preprocessed_image, metadata = preprocess_image_for_yolop(raw_image)

    with torch.no_grad():
        outputs = yolop_v2_lines_model(preprocessed_image)

    return outputs[1], outputs[2], resized_image, metadata


def process_yolop_v2_drivable_output(da_seg_out, resized_image):
    # Interpolate back to original resolution
    da_seg_mask = torch.nn.functional.interpolate(
        da_seg_out,
        size=(resized_image.shape[0], resized_image.shape[1]),
        mode='bilinear',
        align_corners=False
    )

    # Apply softmax and select the drivable class (usually class 1)
    da_seg_mask = torch.softmax(da_seg_mask, dim=1)
    da_seg_mask = da_seg_mask[0][1, :, :]  # Select class 1 for drivable area

    # Squeeze and convert to numpy for OpenCV operations
    probs_np = da_seg_mask.squeeze().cpu().numpy()

    # Threshold to create a binary mask
    binary_mask = (probs_np > 0.5).astype(np.uint8) * 255

    # Morphological Closing to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask


def handle_detection_failure(ll_seg_mask):
    """
    Handles cases where lane detection fails.
    For the benchmark, we return empty lists, indicating no detection.
    """
    # Create an empty debug viz of the same size as the mask
    debug_image = np.zeros((ll_seg_mask.shape[0], ll_seg_mask.shape[1], 3), dtype=np.uint8)
    return np.zeros_like(ll_seg_mask), [], [], debug_image


def detect_yolop_v2_lines(raw_image, force_classical_fallback=False, ll_seg_out_from_inference=None):
    if ll_seg_out_from_inference is None:
        # Resize as your model expects
        raw_image = cv2.resize(raw_image, (640, 640))

        kernel = np.ones((4, 3), np.uint8)
        raw_image = cv2.erode(raw_image, kernel, iterations=1)

        img = yolop_v2_lines_transform(raw_image).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        outputs = yolop_v2_lines_model(img)

        ll_seg_out = outputs[2]
    else:
        ll_seg_out = ll_seg_out_from_inference
        # When ll_seg_out_from_inference is provided, we assume raw_image is already resized
        # and is the resized_image from run_yolop_v2_inference

    ll_seg_mask = torch.nn.functional.interpolate(
        ll_seg_out,
        size=(raw_image.shape[0], raw_image.shape[1]),
        mode='bilinear',
        align_corners=False
    )

    probs = torch.sigmoid(ll_seg_mask)

    # Do everything on the GPU!
    probs_min = probs.min()
    probs_max = probs.max()
    # Normalize relative to max activation
    normalized_tensor = (probs - probs_min) / (probs_max - probs_min + 1e-6)

    # Keep strongest activations only and cast to uint8 before transferring to CPU
    threshold_value = 0.6
    yolo_mask_tensor = (normalized_tensor > threshold_value).byte() * 255
    yolo_mask = yolo_mask_tensor.squeeze().cpu().numpy()

    # --- Morphological Operations ---
    # 1. Opening: Remove small scattered noise (e.g. 3x3 kernel)
    # kernel_open = np.ones((3, 3), np.uint8)
    # yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_OPEN, kernel_open)

    # 2. Closing: Join disconnected line segments vertically (e.g. tall 15x3 kernel)
    # kernel_close = np.ones((15, 3), np.uint8)
    # yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel_close)

    # ==========================
    # CLASSICAL CV FALLBACK
    # ==========================

    if not force_classical_fallback:
        return yolo_mask

    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    # Light blur only (avoid destroying lines)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=40
    )

    cv_mask = np.zeros_like(yolo_mask)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cv_mask, (x1, y1), (x2, y2), 255, 2)

    cv_mask_fallback = classical_lane_fallback(raw_image)

    # Apply road ROI
    cv_mask_fallback = apply_road_roi(cv_mask_fallback, keep_ratio=0.35)

    cv_mask = cv2.bitwise_or(cv_mask, cv_mask_fallback)

    hybrid_mask = cv2.bitwise_or(yolo_mask, cv_mask)

    return hybrid_mask


def apply_road_roi(mask, keep_ratio=0.25):
    """
    Keeps only the bottom portion of the image.
    keep_ratio = 0.45 means bottom 45% is kept.
    """
    h = mask.shape[0]
    roi_mask = np.zeros_like(mask)
    roi_mask[int(h * (1 - keep_ratio)):, :] = 255
    return cv2.bitwise_and(mask, roi_mask)


def filter_lane_like_regions(white_mask):
    filtered = np.zeros_like(white_mask)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )

    h, w = white_mask.shape

    for i in range(1, num_labels):  # skip background
        x, y, w_box, h_box, area = stats[i]

        # --- FILTER RULES ---

        # 1. Remove tiny areas
        if area < 80:
            continue

        # 2. Aspect ratio: long and thin
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if aspect < 2.5:
            continue

        # 3. Orientation: mostly vertical
        component = (labels == i).astype(np.uint8) * 255
        ys, xs = np.where(component > 0)
        if len(xs) < 30:
            continue

        vx, vy, _, _ = cv2.fitLine(
            np.column_stack([xs, ys]), cv2.DIST_L2, 0, 0.01, 0.01
        )

        angle = abs(np.arctan2(vy, vx)) * 180 / np.pi
        if angle < 25 or angle > 155:
            continue  # reject near-horizontal

        # 4. Position: bottom half only
        cy = centroids[i][1]
        if cy < h * 0.45:
            continue

        # Passed all filters → keep
        filtered[labels == i] = 255

    return filtered


def classical_lane_fallback(raw_image):
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

    # Detect white-ish pixels (lane paint)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean noise
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # Keep only bottom half (where lanes are)
    h = white_mask.shape[0]
    white_mask[:h // 2, :] = 0

    # Optional thinning
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    return white_mask


from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize


def _extend_line(skeleton, height):
    if skeleton is None:
        return None, None

    points = np.argwhere(skeleton > 0)
    if len(points) < 900:  # Need at least a few points to do anything
        return skeleton, None

    # Create a copy to draw extensions on and a debug visualizer
    extended_mask = skeleton.copy()
    debug_viz = cv2.cvtColor(skeleton.copy(), cv2.COLOR_GRAY2BGR)
    width = skeleton.shape[1]

    # Sort points from top to bottom
    points = points[points[:, 0].argsort()]

    # --- SMOOTHING STEP ---
    try:
        # Window size must be odd and smaller than the number of points
        window_length = min(len(points) // 3, 51)  # Window up to 51, or 1/3 of points
        if window_length < 5: window_length = 5
        if window_length % 2 == 0: window_length += 1

        if len(points) > window_length:
            # Smooth the x-coordinates of the line
            smoothed_x = savgol_filter(points[:, 1], window_length=window_length, polyorder=2)
            # Update points with the smoothed x-values
            points = np.array(list(zip(points[:, 0], smoothed_x))).astype(np.int32)
    except Exception:
        pass  # If smoothing fails for any reason, proceed with the original noisy line
    # --- END SMOOTHING ---

    y_coords = points[:, 0]
    min_y, max_y = y_coords.min(), y_coords.max()

    # --- Downward Extension ---
    BORDER_TOLERANCE = 5  # pixels
    if max_y < height - 5:  # 5-pixel tolerance for bottom
        num_points_to_use = 900
        lower_points = points[-num_points_to_use:]
        if len(lower_points) < 2:
            lower_points = points

        # Draw the points being used for downward extension in RED
        for p in lower_points:
            cv2.circle(debug_viz, (p[1], p[0]), 3, (0, 0, 255), -1)

        if len(lower_points) >= 2:
            y_fit = lower_points[:, 0]
            x_fit = lower_points[:, 1]
            try:
                # Use a more robust linear fit if we don't have enough points for a quadratic one
                degree = 2 if len(np.unique(y_fit)) > 2 else 1
                fit = np.polyfit(y_fit, x_fit, degree)
                line_fn = np.poly1d(fit)

                original_bottom_x = int(np.mean(points[y_coords == max_y, 1]))
                original_bottom_point = (original_bottom_x, max_y)

                # ONLY extend downwards if the line is NOT hitting the left/right border
                if not (original_bottom_x < BORDER_TOLERANCE or original_bottom_x > width - 1 - BORDER_TOLERANCE):
                    target_y = height - 1
                    target_x = int(line_fn(target_y))
                    target_x = np.clip(target_x, 0, width - 1)
                    target_point = (target_x, target_y)

                    cv2.line(extended_mask, original_bottom_point, target_point, 255, 2)
            except (np.linalg.LinAlgError, TypeError):
                pass  # If fit fails, we just don't extend downwards

    # --- Upward Extension ---
    if min_y > height // 3:  # Only extend up if the line starts below the 25% upper part of the image
        num_points_to_use = 900
        upper_points = points[:num_points_to_use]
        if len(upper_points) < 2:
            upper_points = points

        # Draw the points being used for upward extension in BLUE
        for p in upper_points:
            cv2.circle(debug_viz, (p[1], p[0]), 3, (255, 0, 0), -1)

        if len(upper_points) >= 2:
            y_fit = upper_points[:, 0]
            x_fit = upper_points[:, 1]
            try:
                degree = 2 if len(np.unique(y_fit)) > 2 else 1
                fit = np.polyfit(y_fit, x_fit, degree)
                line_fn = np.poly1d(fit)

                original_top_x = int(np.mean(points[y_coords == min_y, 1]))
                original_top_point = (original_top_x, min_y)

                target_y = height // 3
                target_x = int(line_fn(target_y))
                target_x = np.clip(target_x, 0, width - 1)
                target_point = (target_x, target_y)

                cv2.line(extended_mask, original_top_point, target_point, 255, 2)
            except (np.linalg.LinAlgError, TypeError):
                pass  # If fit fails, we just don't extend upwards

    return extended_mask, debug_viz


def _line_crosses_white(p1, p2, binary_mask):
    """Returns True if line between p1 and p2 crosses white pixels"""
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))

    # Create a mask with just the line
    line_mask = np.zeros_like(binary_mask)
    cv2.line(line_mask, p1, p2, 255, 1)

    # Check if any white pixel from original mask overlaps the line
    overlap = cv2.bitwise_and(binary_mask, line_mask)
    return np.any(overlap > 0)


def extend_polyline_from_fit(fit_fn, y_min, y_max, width, num_points=100):
    ys = np.linspace(y_min, y_max, num_points).astype(int)
    xs = np.clip(fit_fn(ys), 0, width - 1).astype(int)
    return np.array(list(zip(xs, ys))).reshape((-1, 1, 2))


import numpy as np
import cv2
import os


def _extend_contour(contour, target_y, direction, num_points_for_fit=100):
    if contour is None or len(contour) < 20:
        return contour

    # Sort points by y-coordinate to ensure correct order
    contour = contour[contour[:, 1].argsort()]

    y_pts, x_pts = contour[:, 1], contour[:, 0]

    if direction == 'down':
        if y_pts[-1] >= target_y: return contour
        # Use more points from the end of the line for a more stable linear trend
        fit_points_y = y_pts[-num_points_for_fit:]
        fit_points_x = x_pts[-num_points_for_fit:]
        y_start = y_pts[-1]
        y_end = target_y
    else:
        if y_pts[0] <= target_y: return contour
        fit_points_y = y_pts[:num_points_for_fit]
        fit_points_x = x_pts[:num_points_for_fit]
        y_start = target_y
        y_end = y_pts[0]

    if len(fit_points_y) < 2: return contour

    try:
        # Strictly Linear Extension: avoids the "swinging" effect of high-degree polynomials
        fit = np.polyfit(fit_points_y, fit_points_x, 1)
        
        # SAFETY CHECK: If the line is quite horizontal (slope dx/dy > 1.2), 
        # do not extend it to avoid wild horizontal projections.
        if abs(fit[0]) > 1.2:
            return contour

        fit_fn = np.poly1d(fit)

        num_new_points = abs(int(y_end - y_start))
        if num_new_points <= 0: return contour

        y_new = np.linspace(y_start, y_end, num_new_points).astype(int)
        x_new = fit_fn(y_new).astype(int)

        extension_points = np.vstack((x_new, y_new)).T

        if direction == 'down':
            return np.vstack((contour, extension_points))
        else:
            return np.vstack((extension_points, contour))

    except (np.linalg.LinAlgError, TypeError):
        return contour


_outlier_tracking = {
    'left': {'count': 0},
    'right': {'count': 0}
}


def _find_and_draw_lane_boundaries(binary_mask, save_path_intermediate_dir=None, base_name=None, reference_point=None,
                                   drivable_mask=None, prev_left_contour=None, prev_right_contour=None):
    global _outlier_tracking
    height, width = binary_mask.shape

    # --- Step 0: Pre-process mask to remove noise ---
    debug_viz = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Smooth the mask to close tiny internal holes that cause "spurs" (fake branches).
    # NOTE: We only use MORPH_CLOSE. Using MORPH_OPEN (erosion) was causing parallel
    # thick lines at the bottom of the image to touch/bleed into each other, creating
    # massive fake U-shape junctions at the bottom of the screen!
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, morph_kernel)

    # --- Step 1: Skeletonize to get thin lines ---
    skeleton = skeletonize(smoothed_mask // 255).astype(np.uint8) * 255

    # --- Step 2: Robust Splitting at Junctions ---
    # OPTIMIZATION: We only care about junctions in the drivable area.
    roi_y_start = int(height * 0.4)
    roi_skeleton = skeleton[roi_y_start:, :]

    raw_junction_points = []
    junction_viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    # 1. Topological Junctions (Branches, T, Y, X intersections)
    kernel3x3 = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(roi_skeleton // 255, -1, kernel3x3)
    
    roi_topological = np.argwhere((neighbor_count >= 4) & (roi_skeleton > 0))
    for y, x in roi_topological:
        raw_junction_points.append({'y': y + roi_y_start, 'x': x, 'type': 'topological'})

    # 2. Inverted U/V Shapes (Peaks of continuous lines that turn around)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_skeleton, connectivity=8)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
            
        x_min = stats[i, cv2.CC_STAT_LEFT]
        y_min = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        
        label_roi = (labels[y_min:y_min+h_box, x_min:x_min+w_box] == i)
        pts_y, pts_x = np.where(label_roi)
        pts_y += y_min
        pts_x += x_min
        
        unique_y = np.unique(pts_y)
        has_multiple_legs = False
        for y_val in unique_y:
            xs_at_y = np.sort(pts_x[pts_y == y_val])
            if len(xs_at_y) > 1 and np.max(np.diff(xs_at_y)) > 15:
                has_multiple_legs = True
                break
                
        if has_multiple_legs:
            peak_y = np.min(pts_y)
            peak_x = int(np.mean(pts_x[pts_y == peak_y]))
            raw_junction_points.append({'y': peak_y + roi_y_start, 'x': peak_x, 'type': 'uv_peak'})

    # BORDER EXCLUSION: Ignore all junctions that are really close to the image borders 
    # to avoid artifacts from lines entering/exiting the camera frame.
    junction_points = []
    border_margin = 25 # Pixels from left/right/bottom
    for jp in raw_junction_points:
        x, y = jp['x'], jp['y']
        if x > border_margin and x < width - border_margin and y < height - border_margin:
            junction_points.append(jp)
            
            # Visualization
            color = (0, 165, 255) if jp['type'] == 'topological' else (255, 0, 255)
            cv2.circle(junction_viz, (x, y), 5, color, 1)

    # Create a copy of the SKELETON to "erase" the junctions from.
    # Because the skeleton is only 1 pixel thick, a tiny 3-pixel radius is enough to cleanly sever the lines!
    split_skeleton = skeleton.copy()

    for jp in junction_points:
        x, y = jp['x'], jp['y']
        # Erase a tiny 3-pixel radius EXACTLY on the skeleton lines to ensure clean severance
        cv2.circle(split_skeleton, (int(x), int(y)), 3, 0, -1)
        
        # Draw red filled circles on the final split points in the visualization
        cv2.circle(junction_viz, (int(x), int(y)), 3, (0, 0, 255), -1)  

    if save_path_intermediate_dir:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_1.5_split_mask.png"), split_skeleton)

    # --- Step 3: Find Contours of Clean Segments ---
    # Now that junctions are erased, we can find contours of the simple, separated skeleton segments.
    contours, _ = cv2.findContours(split_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter out tiny contours that are likely just noise, UNLESS they are connected to a junction.
    # We still want to preserve any branches connected to our tiny 3px cuts.
    min_contour_length = 250
    long_contours = []
    
    for c in contours:
        pts = c.reshape(-1, 2)
        min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
        top_x = np.mean(pts[pts[:, 1] == min_y, 0])
        bottom_x = np.mean(pts[pts[:, 1] == max_y, 0])
        
        # ROAD LENGTH: Euclidean distance from tip to tip.
        # This measures the actual span of the line, ignoring how "thick" the mask is.
        road_length = np.sqrt((top_x - bottom_x)**2 + (max_y - min_y)**2)
        
        if road_length < 20:
            # Discard tiny fragments or horizontal blobs that don't cover enough distance
            continue

        if cv2.arcLength(c, False) > min_contour_length:
            long_contours.append(c)
        else:
            # Check if this small contour is adjacent to a junction
            is_near_junction = False
            for jp in junction_points:
                jx, jy = jp['x'], jp['y']
                # Tightened the search radius to 15px since the cut is only 3px now
                if np.sqrt((top_x - jx)**2 + (min_y - jy)**2) < 15 or \
                   np.sqrt((bottom_x - jx)**2 + (max_y - jy)**2) < 15:
                    is_near_junction = True
                    break
            
            if is_near_junction:
                long_contours.append(c)

    # --- Step 3.5: DAG-based Junction Merging and Tail Discarding ---
    # We build a Directed Acyclic Graph (DAG) of the segments around junctions.
    # If 1 segment is below a junction and 2+ are above, we branch the below segment into multiple paths.
    # If 2+ segments are below and 1 is above, the above is a discarded tail (original behavior).
    contour_info = []
    for idx, c in enumerate(long_contours):
        pts = c.reshape(-1, 2)
        min_y = pts[:, 1].min()
        max_y = pts[:, 1].max()
        top_pts = pts[pts[:, 1] == min_y]
        bottom_pts = pts[pts[:, 1] == max_y]
        top_x = np.mean(top_pts[:, 0])
        bottom_x = np.mean(bottom_pts[:, 0])
        contour_info.append({
            'contour': c,
            'pts': pts,
            'min_y': min_y,
            'max_y': max_y,
            'top_x': top_x,
            'bottom_x': bottom_x,
            'is_tail': False
        })

    for info in contour_info:
        for jp in junction_points:
            jx, jy = jp['x'], jp['y']
            if jy > height / 1.5:
                continue
            # Tightened vertical margin and proximity radius for the 3px cut
            if info['max_y'] <= jy + 10:
                dist = np.sqrt((info['bottom_x'] - jx)**2 + (info['max_y'] - jy)**2)
                if dist < 15: 
                    info['is_tail'] = True
                    # Draw indicator on junction_viz for discarded tail
                    cv2.putText(junction_viz, "Tail Discarded", (int(info['bottom_x']) + 10, int(info['max_y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    break

    edges = {i: [] for i in range(len(contour_info))}
    in_degree = {i: 0 for i in range(len(contour_info))}
    
    for jp in junction_points:
        jx, jy = jp['x'], jp['y']
        
        above_idxs = []
        below_idxs = []
        
        for i, info in enumerate(contour_info):
            # Checking proximity to the junction point
            # Tightened vertical margin and proximity radius for the 3px cut
            if info['max_y'] <= jy + 10:
                dist = np.sqrt((info['bottom_x'] - jx)**2 + (info['max_y'] - jy)**2)
                if dist < 15:
                    above_idxs.append(i)
            elif info['min_y'] >= jy - 10:
                dist = np.sqrt((info['top_x'] - jx)**2 + (info['min_y'] - jy)**2)
                if dist < 15:
                    below_idxs.append(i)
                    
        # If 1 below and 1+ above, branch the below segment into each above segment
        if len(below_idxs) == 1 and len(above_idxs) >= 1:
            below_info = contour_info[below_idxs[0]]
            for a_idx in above_idxs:
                edges[below_idxs[0]].append(a_idx)
                in_degree[a_idx] += 1
                above_info = contour_info[a_idx]
                
                # Draw indicator arrow and text on junction_viz
                pt1 = (int(below_info['top_x']), int(below_info['min_y']))
                pt2 = (int(above_info['bottom_x']), int(above_info['max_y']))
                cv2.arrowedLine(junction_viz, pt1, pt2, (255, 255, 0), 2, tipLength=0.2)
                cv2.putText(junction_viz, "Merged/Split", (int((pt1[0]+pt2[0])/2) + 10, int((pt1[1]+pt2[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if save_path_intermediate_dir:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_1_junctions_erased.png"), junction_viz)

    processed_contours = []
    roots = [i for i, info in enumerate(contour_info) if in_degree[i] == 0 and not info['is_tail']]
    
    def dfs(node_idx, current_path_contours):
        path = current_path_contours + [contour_info[node_idx]['contour']]
        children = edges[node_idx]
        if not children:
            if len(path) == 1:
                processed_contours.append(path[0])
            else:
                processed_contours.append(np.concatenate(path, axis=0))
        else:
            for child_idx in children:
                dfs(child_idx, path)

    for root_idx in roots:
        dfs(root_idx, [])

    long_contours = processed_contours

    # --- Step 4: Classify Contours and Create Candidate Visualization ---
    lane_candidates = []
    # Create a new blank visualization for all potential lane candidates
    candidates_viz = np.zeros((height, width, 3), dtype=np.uint8)

    # DEBUG: Draw previous frame's lines in white (255, 255, 255)
    if prev_left_contour is not None and len(prev_left_contour) > 0:
        cv2.polylines(candidates_viz, [prev_left_contour.reshape((-1, 1, 2))], isClosed=False, color=(255, 255, 255),
                      thickness=1)
    if prev_right_contour is not None and len(prev_right_contour) > 0:
        cv2.polylines(candidates_viz, [prev_right_contour.reshape((-1, 1, 2))], isClosed=False, color=(255, 255, 255),
                      thickness=1)

    for contour in long_contours:
        # Draw all surviving long contours in green for complete visualization
        cv2.polylines(candidates_viz, [contour.reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 0), thickness=2)

        points = contour.reshape(-1, 2)
        x_pts, y_pts = points[:, 0], points[:, 1]

        # DEBUG: Add y_span text to visualize vertical coverage
        y_span_debug = int(y_pts.max() - y_pts.min())
        mid_idx = len(points) // 2
        y_txt = f"Y:{y_span_debug}"
        text_x = int(points[mid_idx, 0]) + 10
        text_y = int(points[mid_idx, 1])
        (tw, th), _ = cv2.getTextSize(y_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if text_x + tw > width:
            text_x = int(points[mid_idx, 0]) - 10 - tw
        if text_y - th < 0:
            text_y = th + 5
        cv2.putText(candidates_viz, y_txt, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(y_pts) < 3: continue

        try:
            # Fit a simple line to get a function representation
            fit = np.polyfit(y_pts, x_pts, 1)
            fit_fn = np.poly1d(fit)

            lane_candidates.append({
                'points': points,
                'fit_fn': fit_fn
            })
        except (np.linalg.LinAlgError, TypeError):
            lane_candidates.append({
                'points': points,
                'fit_fn': None
            })
            continue

    # --- Step 5 & 6: Select Best Left and Right Lanes with Two-Stage Fallback ---
    if not lane_candidates:
        return np.zeros_like(binary_mask), [], [], debug_viz, None, None, np.array(
            []), debug_viz, debug_viz, np.zeros_like(binary_mask), candidates_viz, None, None

    # --- Define Helpers ---
    key_y_points = np.linspace(height // 2, height - 1, 32).astype(int)

    def get_x_at_y(contour, y_vals, side=None):
        if contour is None or len(contour) < 2: return None
        try:
            y_pts = contour[:, 1]
            x_pts = contour[:, 0]

            # Group all x-coordinates by their unique y-coordinate
            y_unique, idx = np.unique(y_pts, return_inverse=True)

            # CRITICAL FIX: If a line has only 1 unique Y coordinate
            if len(y_unique) < 2: return None

            # --- Get the inner-most edge of the thick contour (Optimized) ---
            if side == 'left':
                x_unique = np.full(len(y_unique), -np.inf)
                np.maximum.at(x_unique, idx, x_pts)
            elif side == 'right':
                x_unique = np.full(len(y_unique), np.inf)
                np.minimum.at(x_unique, idx, x_pts)
            else:
                x_unique = np.bincount(idx, weights=x_pts) / np.bincount(idx)

            # --- Perform simple, fast, and robust interpolation/extrapolation ---
            if np.isscalar(y_vals):
                return float(np.interp(y_vals, y_unique, x_unique))
            else:
                return np.interp(y_vals, y_unique, x_unique)

        except Exception:
            return None

    # --- Stage 1 & 2: Unified Global Election (No Buckets) ---
    ego_center_x = width / 2.0
    best_left_seed = None
    best_right_seed = None

    # 1. Spatial Election (Vanishing Point Heuristic)
    if lane_candidates:
        enriched_cands = []
        for lane in lane_candidates:
            pts = lane['points']
            y_pts = pts[:, 1]
            x_pts = pts[:, 0]
            max_y = y_pts.max()
            y_span = max_y - y_pts.min()

            if y_span < 80:
                continue

            x_bottom = get_x_at_y(pts, height - 1)
            if x_bottom is None: x_bottom = np.mean(x_pts)

            # Distance to the center of the car
            dist_to_center = abs(x_bottom - ego_center_x)

            # To avoid fitting a horizontal line to the bottom contour edge of thick masks,
            # we group the points by Y and average the X coordinates to create a 1-pixel thin centerline.
            y_unique, idx = np.unique(y_pts, return_inverse=True)
            x_unique = np.bincount(idx, weights=x_pts) / np.bincount(idx)

            # Local slope (bottom section) for tracking projection
            y_thresh = max_y - min(40, y_span * 0.3)
            bottom_mask = y_unique >= y_thresh
            if np.sum(bottom_mask) > 3:
                fit = np.polyfit(y_unique[bottom_mask], x_unique[bottom_mask], 1)
            else:
                fit = np.polyfit(y_unique, x_unique, 1)
            slope = fit[0]  # dx/dy

            # PROJECTED ANCHOR: Use the fit to find where this line WOULD hit the bottom
            # This is much more stable for tracking dashed lines than raw points.
            projected_x_bottom = fit[0] * (height - 1) + fit[1]
            
            cand_mean_dist = np.mean(np.abs(x_unique - ego_center_x))

            enriched_cands.append({
                'lane': lane,
                'dist_to_center': dist_to_center,
                'x_bottom': projected_x_bottom,  # Use projected X for tracking matching
                'slope': slope,
                'pts': pts,
                'centerline': (x_unique, y_unique),
                'mean_dist': cand_mean_dist,
                'length': len(pts)
            })

        if enriched_cands:
            best_left_cand = None
            best_right_cand = None

            # --- COMBINED SCORE ELECTION LOGIC (SIDE-AGNOSTIC) ---
            # Extract previous anchors for tracking continuity
            prev_anchors = []
            if prev_left_contour is not None and len(prev_left_contour) > 0:
                # Get the bottom-most X coordinate of the previous left line
                pts = prev_left_contour.reshape(-1, 2)
                prev_anchors.append(pts[np.argmax(pts[:, 1]), 0])
            if prev_right_contour is not None and len(prev_right_contour) > 0:
                # Get the bottom-most X coordinate of the previous right line
                pts = prev_right_contour.reshape(-1, 2)
                prev_anchors.append(pts[np.argmax(pts[:, 1]), 0])

            # DYNAMIC CENTER TRACKING: Use previous lines to define the L/R boundary
            if len(prev_anchors) == 2:
                dynamic_center_x = sum(prev_anchors) / 2.0
            elif len(prev_anchors) == 1:
                # If only one line tracked, assume lane width is roughly 400px at bottom
                if prev_anchors[0] < width / 2.0:
                    dynamic_center_x = prev_anchors[0] + 200
                else:
                    dynamic_center_x = prev_anchors[0] - 200
            else:
                dynamic_center_x = width / 2.0

            # 1. Classify all candidates as OK or REJ based on strict criteria
            valid_candidates = []
            relaxed_candidates = []

            for cand in enriched_cands:
                # Same criteria as drivable agent: Length >= 90
                is_long_enough = cand['length'] >= 90
                
                # Assign status for debug drawing (OK/REJ)
                cand['status'] = "OK" if is_long_enough else "REJ"

                # Calculate the Combined Score (50% Slope, 50% P-Distance)
                norm_slope = min(abs(cand['slope']), 1.0)
                
                # TRACKING-AWARE P DISTANCE:
                if len(prev_anchors) > 0:
                    dists = [abs(cand['x_bottom'] - pa) for pa in prev_anchors]
                    proj_dist_abs = min(dists)
                else:
                    proj_dist_abs = abs(cand['x_bottom'] - width / 2.0)
                    
                norm_p = min(proj_dist_abs / max(1, width / 2), 1.0)
                
                cand['combined_score'] = 0.5 * norm_slope + 0.5 * norm_p

                if is_long_enough:
                    valid_candidates.append(cand)
                else:
                    relaxed_candidates.append(cand)

                # DEBUG: Draw the straight line and its score on candidates_viz
                top_pt = (cand['centerline'][0][0], cand['centerline'][1][0])
                bottom_pt = (cand['centerline'][0][-1], cand['centerline'][1][-1])
                cv2.line(candidates_viz, (int(top_pt[0]), int(top_pt[1])), (int(bottom_pt[0]), int(bottom_pt[1])),
                         (0, 165, 255), 2)  # Orange for the straight line
                         
                # Draw the Status, Combined Score (C), L, and P text
                debug_txt = f"{cand['status']} | C:{cand['combined_score']:.2f} | L:{cand['length']} | P:{proj_dist_abs:.0f}"
                
                mid_pt_x = int((top_pt[0] + bottom_pt[0]) / 2) + 10
                mid_pt_y = int((top_pt[1] + bottom_pt[1]) / 2)
                
                (tw, th), _ = cv2.getTextSize(debug_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                if mid_pt_x + tw > width: mid_pt_x = int((top_pt[0] + bottom_pt[0]) / 2) - 10 - tw
                if mid_pt_y - th - 5 < 0: mid_pt_y = th + 5
                    
                cv2.rectangle(candidates_viz, (mid_pt_x - 2, mid_pt_y - th - 5), (mid_pt_x + tw + 2, mid_pt_y + 5), (0, 0, 0), -1)
                cv2.putText(candidates_viz, debug_txt, (mid_pt_x, mid_pt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # 2. TWO-STAGE ELECTION: Search across both Valid and Relaxed to ensure we get a Left + Right pair
            best_left_cand = None
            best_right_cand = None

            left_valid = [c for c in valid_candidates if c['x_bottom'] < dynamic_center_x]
            right_valid = [c for c in valid_candidates if c['x_bottom'] >= dynamic_center_x]
            left_relaxed = [c for c in relaxed_candidates if c['x_bottom'] < dynamic_center_x]
            right_relaxed = [c for c in relaxed_candidates if c['x_bottom'] >= dynamic_center_x]
            
            left_valid.sort(key=lambda c: c['combined_score'])
            right_valid.sort(key=lambda c: c['combined_score'])
            left_relaxed.sort(key=lambda c: c['combined_score'])
            right_relaxed.sort(key=lambda c: c['combined_score'])

            if left_valid: best_left_cand = left_valid[0]
            elif left_relaxed: best_left_cand = left_relaxed[0]

            if right_valid: best_right_cand = right_valid[0]
            elif right_relaxed: best_right_cand = right_relaxed[0]

            elected_cands = []
            if best_left_cand: elected_cands.append(best_left_cand)
            if best_right_cand: elected_cands.append(best_right_cand)

            # Fallback if we still don't have 2 lines (i.e., one side was completely empty)
            if len(elected_cands) < 2:
                # Sort ALL candidates: prioritize Valid over Relaxed, then by Score
                pool = valid_candidates + relaxed_candidates
                pool.sort(key=lambda c: (0 if c['length'] >= 90 else 1, c['combined_score']))
                
                for cand in pool:
                    is_distinct = True
                    for e_cand in elected_cands:
                        if abs(cand['x_bottom'] - e_cand['x_bottom']) < 50:
                            is_distinct = False
                            break
                    if is_distinct:
                        elected_cands.append(cand)
                    if len(elected_cands) == 2:
                        break

            best_left_seed = best_left_cand['lane'] if best_left_cand else None
            best_right_seed = best_right_cand['lane'] if best_right_cand else None
            
            # If the fallback triggered, re-assign seeds dynamically based on their relative positions
            if len(elected_cands) == 2 and (not best_left_seed or not best_right_seed):
                cand1, cand2 = elected_cands[0], elected_cands[1]
                if cand1['x_bottom'] < cand2['x_bottom']:
                    best_left_seed, best_right_seed = cand1['lane'], cand2['lane']
                else:
                    best_left_seed, best_right_seed = cand2['lane'], cand1['lane']
            elif len(elected_cands) == 1 and not best_left_seed and not best_right_seed:
                # Only 1 line found total via fallback
                if elected_cands[0]['x_bottom'] < dynamic_center_x:
                    best_left_seed = elected_cands[0]['lane']
                else:
                    best_right_seed = elected_cands[0]['lane']

    # 4. Assign elected candidates
    chosen_left_candidate = best_left_seed['points'] if best_left_seed is not None else None
    chosen_right_candidate = best_right_seed['points'] if best_right_seed is not None else None

    # --- Step 6: Extension ---
    best_left_contour = chosen_left_candidate
    best_right_contour = chosen_right_candidate

    # Pass exactly what was detected to the next frame
    best_left_raw = best_left_contour.copy() if best_left_contour is not None else None
    best_right_raw = best_right_contour.copy() if best_right_contour is not None else None

    # Calculate dynamic top target based on the highest point (minimum Y) of both lines
    top_y_target = height // 2 + 50  # Fallback
    if best_left_contour is not None and best_right_contour is not None:
        top_y_target = min(best_left_contour[:, 1].min(), best_right_contour[:, 1].min())
    elif best_left_contour is not None:
        top_y_target = best_left_contour[:, 1].min()
    elif best_right_contour is not None:
        top_y_target = best_right_contour[:, 1].min()

    # Extend to the bottom and top (dynamic top target)
    if best_left_contour is not None:
        best_left_contour = _extend_contour(best_left_contour, target_y=height - 1, direction='down')
        best_left_contour = _extend_contour(best_left_contour, target_y=top_y_target, direction='up')
    if best_right_contour is not None:
        best_right_contour = _extend_contour(best_right_contour, target_y=height - 1, direction='down')
        best_right_contour = _extend_contour(best_right_contour, target_y=top_y_target, direction='up')

    # Highlight the chosen lanes on the visualization image
    if best_left_contour is not None:
        cv2.polylines(candidates_viz, [best_left_contour.reshape((-1, 1, 2))], isClosed=False, color=(255, 255, 0),
                      thickness=2)
    if best_right_contour is not None:
        cv2.polylines(candidates_viz, [best_right_contour.reshape((-1, 1, 2))], isClosed=False, color=(255, 0, 255),
                      thickness=2)

    # --- Step 7: No Virtual Line Fallback ---
    # We no longer invent virtual lines if one is missing. We only rely on actual detections.
    # The hybrid path calculation will handle cases where only one line is present.

    # --- Step 8: Accurate Midpoint Calculation from Contours ---
    hybrid_path_pts = []
    if best_left_contour is not None and best_right_contour is not None and len(best_left_contour) >= 2 and len(
            best_right_contour) >= 2:
        # Both lines detected: compute mathematical average
        y_min = min(best_left_contour[:, 1].min(), best_right_contour[:, 1].min())
        y_max = max(best_left_contour[:, 1].max(), best_right_contour[:, 1].max())

        sampled_ys = np.linspace(y_min, y_max, 10).astype(int)

        for y in sampled_ys:
            l_x = get_x_at_y(best_left_contour, y, side='left')
            r_x = get_x_at_y(best_right_contour, y, side='right')

            if l_x is None:
                idx = np.argmin(np.abs(best_left_contour[:, 1] - y))
                l_x = best_left_contour[idx, 0]
            if r_x is None:
                idx = np.argmin(np.abs(best_right_contour[:, 1] - y))
                r_x = best_right_contour[idx, 0]

            mid_x = int((l_x + r_x) / 2)
            hybrid_path_pts.append([mid_x, int(y)])
    elif best_left_contour is not None and len(best_left_contour) >= 2:
        # Only left line detected: shift it to roughly represent the center
        # Assuming a default lane width of 200 pixels at the bottom
        y_min, y_max = best_left_contour[:, 1].min(), best_left_contour[:, 1].max()
        sampled_ys = np.linspace(y_min, y_max, 10).astype(int)
        for y in sampled_ys:
            l_x = get_x_at_y(best_left_contour, y, side='left')
            if l_x is None:
                idx = np.argmin(np.abs(best_left_contour[:, 1] - y))
                l_x = best_left_contour[idx, 0]

            # Rough perspective shift (100 pixels at bottom, 0 at top)
            shift = 100 * (y / height)
            hybrid_path_pts.append([int(l_x + shift), int(y)])
    elif best_right_contour is not None and len(best_right_contour) >= 2:
        # Only right line detected
        y_min, y_max = best_right_contour[:, 1].min(), best_right_contour[:, 1].max()
        sampled_ys = np.linspace(y_min, y_max, 10).astype(int)
        for y in sampled_ys:
            r_x = get_x_at_y(best_right_contour, y, side='right')
            if r_x is None:
                idx = np.argmin(np.abs(best_right_contour[:, 1] - y))
                r_x = best_right_contour[idx, 0]

            shift = 100 * (y / height)
            hybrid_path_pts.append([int(r_x - shift), int(y)])
    t_path_end = time.time()

    # --- Final Mask Generation & Visualization ---
    t_viz_start = time.time()
    final_mask = np.zeros_like(binary_mask)
    lines_debug_viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    if best_left_contour is not None:
        cv2.polylines(lines_debug_viz, [best_left_contour], isClosed=False, color=(255, 0, 0), thickness=2)
    if best_right_contour is not None:
        cv2.polylines(lines_debug_viz, [best_right_contour], isClosed=False, color=(0, 0, 255), thickness=2)

    all_paths_viz = lines_debug_viz.copy()

    if hybrid_path_pts:
        path_pts_np = np.array([hybrid_path_pts], dtype=np.int32)
        cv2.polylines(final_mask, path_pts_np, isClosed=False, color=255, thickness=2)
        cv2.polylines(all_paths_viz, path_pts_np, isClosed=False, color=(0, 255, 255), thickness=2)

    final_hybrid_path_points = np.array(hybrid_path_pts, dtype=np.int32)
    all_extended_lines_mask = final_mask.copy()

    if save_path_intermediate_dir:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_1.5_split_debug.png"), debug_viz)
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_2_lines_debug.png"), lines_debug_viz)
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_3_final_path.png"), all_paths_viz)

    t_viz_end = time.time()

    return final_mask, [], [], debug_viz, None, None, final_hybrid_path_points, lines_debug_viz, all_paths_viz, all_extended_lines_mask, candidates_viz, best_left_raw, best_right_raw



# GLOBAL TRACKING STATE
g_drivable_consecutive_failures = 0
g_hybrid_consecutive_failures = 0

def detect_lanes_yolop_v2_drivable_agent(image, save_path_intermediate_dir=None, img_path=None, prev_right_contour=None):
    global g_drivable_consecutive_failures
    t_start = time.time()
    base_name = Path(img_path).stem if img_path else "debug"

    orig_height, orig_width = image.shape[:2]

    # --- NO MORE PRE-INFERENCE MASKING ---
    # We pass the full raw image to YOLOP to give the network full context
    t_roi = time.time()

    with torch.no_grad():
        da_seg_out, ll_seg_out, resized_image, metadata = run_yolop_v2_inference(image)
        t_inference = time.time()
        
        drivable_mask = process_yolop_v2_drivable_output(da_seg_out, resized_image)
        t_da_post = time.time()

        ll_segment = detect_yolop_v2_lines(resized_image, force_classical_fallback=False, ll_seg_out_from_inference=ll_seg_out)
        t_lines_post = time.time()

    height, width = drivable_mask.shape

    # --- EXPLICIT EGO-LANE SEVERING ---
    kernel_cut = np.ones((9, 9), np.uint8)
    ll_segment_thick = cv2.dilate(ll_segment, kernel_cut, iterations=2)
    drivable_mask[ll_segment_thick > 0] = 0

    if save_path_intermediate_dir and img_path:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_raw_drivable.png"), drivable_mask)

    # --- NEW ALGORITHM START ---
    ROI_TOP_RATIO = 0.40
    MORPH_KERNEL = 7
    MIN_DRIVABLE_PIXELS_PER_ROW = 30
    LANE_WIDTH_BOTTOM = 430
    LANE_WIDTH_TOP = 0
    MAX_EDGE_JUMP = 80
    POLY_DEGREE = 2
    MIN_LENGTH_DRIVABLE_SIDE = 90

    # Step 1: Morphological cleanup
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    drivable_mask_clean = cv2.morphologyEx(drivable_mask, cv2.MORPH_OPEN, kernel)
    drivable_mask_clean = cv2.morphologyEx(drivable_mask_clean, cv2.MORPH_CLOSE, kernel)

    # Step 1.5: Identify all limits in the image for the drivable area
    mask_uint8 = (drivable_mask_clean > 0).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    candidate_limits = []
    roi_top = int(height * ROI_TOP_RATIO)
    center_x_img = width // 2

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 35000:
            continue
            
        x_min = stats[label, cv2.CC_STAT_LEFT]
        y_min = stats[label, cv2.CC_STAT_TOP]
        w_box = stats[label, cv2.CC_STAT_WIDTH]
        h_box = stats[label, cv2.CC_STAT_HEIGHT]
        
        y_start = max(roi_top, y_min)
        y_end = min(height, y_min + h_box)
        if y_end <= y_start:
            continue
            
        label_roi = (labels[y_start:y_end, x_min:x_min+w_box] == label)
        padded_roi = np.pad(label_roi, ((0,0), (0,1)), mode='constant', constant_values=False)
        transitions = padded_roi[:, :-1] & ~padded_roi[:, 1:]
        ys, xs = np.nonzero(transitions)
        
        if len(ys) == 0:
            continue
            
        y_unique, idx = np.unique(ys, return_index=True)
        xs_per_y = np.split(xs, idx[1:])
        
        y_unique = y_unique + y_start
        row_to_xs = {y: arr + x_min for y, arr in zip(y_unique, xs_per_y)}
        
        right_edge_points = []
        prev_x = None
        
        for y_img in range(y_end - 1, y_start - 1, -1):
            if y_img not in row_to_xs:
                continue
                
            available_xs = row_to_xs[y_img]
            x = available_xs[-1] if prev_x is None else available_xs[np.argmin(np.abs(available_xs - prev_x))]
                
            if prev_x is None or abs(x - prev_x) < MAX_EDGE_JUMP:
                right_edge_points.append([int(x), int(y_img)])
                prev_x = x
            else:
                if len(right_edge_points) < 20:
                    right_edge_points = [[int(x), int(y_img)]]
                    prev_x = x
                else:
                    break
                    
        if len(right_edge_points) == 0:
            continue
            
        bottom_point = right_edge_points[0]
        
        similarity = float('inf')
        if prev_right_contour is not None and len(prev_right_contour) > 0:
            prev_y_to_x = {pt[1]: pt[0] for pt in prev_right_contour}
            diffs = [abs(pt[0] - prev_y_to_x[pt[1]]) for pt in right_edge_points if pt[1] in prev_y_to_x]
            if len(diffs) > 0:
                similarity = np.mean(diffs)
                
        candidate_limits.append({
            'label': label,
            'right_edge_points': right_edge_points,
            'bottom_point': bottom_point,
            'similarity': similarity,
            'area': area
        })

    ego_label = 0
    best_right_raw = []
    right_edges_raw = []
    tracking_failed = False
    
    # --- DEBUG VISUALIZATION: Tracking Area ---
    tracking_debug_viz = cv2.cvtColor(drivable_mask_clean, cv2.COLOR_GRAY2BGR)
    cv2.circle(tracking_debug_viz, (center_x_img, roi_top), 5, (0, 255, 255), -1) # Yellow: center point

    # Draw the area of all connected components to help the user tune the filter
    for l in range(1, num_labels):
        area = stats[l, cv2.CC_STAT_AREA]
        cx, cy = int(centroids[l][0]), int(centroids[l][1])
        cv2.putText(tracking_debug_viz, f"Area: {area}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(tracking_debug_viz, f"Area: {area}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for cand in candidate_limits:
        # Draw all candidate right edges in gray
        for pt in cand['right_edge_points']:
            cv2.circle(tracking_debug_viz, (pt[0], pt[1]), 1, (100, 100, 100), -1)
        # Draw their bottom anchor points in orange
        bottom = cand['bottom_point']
        cv2.circle(tracking_debug_viz, (bottom[0], bottom[1]), 4, (0, 165, 255), -1)


    if len(candidate_limits) > 0:
        # Evaluate each candidate strictly according to user rules
        valid_candidates = []

        for cand in candidate_limits:
            pts = np.array(cand['right_edge_points'])
            x_pts = pts[:, 0]
            y_pts = pts[:, 1]
            cand['length'] = len(x_pts)
            
            if len(y_pts) > 2:
                fit = np.polyfit(y_pts, x_pts, 1)
                proj_x = fit[0] * (height - 1) + fit[1]
                
                # DRAW PROJECTED LINE FOR DEBUGGING
                # Start of line: Topmost point of the candidate
                top_y = int(np.min(y_pts))
                top_x = int(fit[0] * top_y + fit[1])
                # End of line: Projected point at the bottom of the image
                bottom_y = height - 1
                bottom_x = int(proj_x)
                cv2.line(tracking_debug_viz, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 255), 1)
            else:
                proj_x = x_pts[0]
                
            cand['proj_x'] = proj_x
            cand['is_right_side'] = proj_x >= center_x_img
            cand['mean_dist'] = np.mean(np.abs(x_pts - center_x_img))

            # Rule 1 & 2: Minimum criteria
            if cand['length'] >= MIN_LENGTH_DRIVABLE_SIDE and cand['is_right_side']:
                valid_candidates.append(cand)
            else:
                reason = []
                if cand['length'] < MIN_LENGTH_DRIVABLE_SIDE: reason.append(f"Length {cand['length']} < {MIN_LENGTH_DRIVABLE_SIDE}")
                if not cand['is_right_side']: reason.append(f"ProjX {cand['proj_x']:.0f} is Left Side")
                # print(f"[REJECT] Candidate Label {cand['label']} (Area {cand['area']}): {', '.join(reason)}")
            
            # Update debug viz with new filtering details
            # Display all three criteria for transparency: Length, Projected X distance, and Mean Distance
            bottom = cand['bottom_point']
            status = "OK" if cand['length'] >= MIN_LENGTH_DRIVABLE_SIDE and cand['is_right_side'] else "REJ"
            proj_dist = cand['proj_x'] - center_x_img
            debug_txt = f"{status} | L:{cand['length']} | P:{proj_dist:.0f} | M:{cand['mean_dist']:.1f}"
            
            # Draw the text and a small box so it's readable, keeping it on screen
            (tw, th), _ = cv2.getTextSize(debug_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = bottom[0] + 10
            if text_x + tw > width:
                text_x = bottom[0] - 10 - tw
            text_y = bottom[1] - 25
            if text_y < 20:
                text_y = bottom[1] + 25
                
            cv2.rectangle(tracking_debug_viz, (text_x - 2, text_y - th - 5), (text_x + tw + 2, text_y + 5), (0, 0, 0), -1)
            cv2.putText(tracking_debug_viz, debug_txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)


        # TWO-STAGE ELECTION: Prioritize right side, but accept left side over nothing.
        if len(valid_candidates) > 0:
            best_candidate = min(valid_candidates, key=lambda c: c['mean_dist'])
            # print(f"[ELECT] Strict Candidate Label {best_candidate['label']} selected (Side: {'Right' if best_candidate['is_right_side'] else 'Left'})")
        else:
            # RELAXED FALLBACK: If no strict candidates exist, accept ANY candidate with sufficient length
            relaxed_candidates = [c for c in candidate_limits if c['length'] >= MIN_LENGTH_DRIVABLE_SIDE]
            if len(relaxed_candidates) > 0:
                best_candidate = min(relaxed_candidates, key=lambda c: c['mean_dist'])
                # print(f"[ELECT-RELAXED] Relaxed Candidate Label {best_candidate['label']} selected (Side: {'Right' if best_candidate['is_right_side'] else 'Left'})")
            else:
                best_candidate = None

        if best_candidate is not None:
            ego_label = best_candidate['label']
            right_edges_raw = best_candidate['right_edge_points']
            
            # Draw the SELECTED right edge in green
            for pt in right_edges_raw:
                cv2.circle(tracking_debug_viz, (pt[0], pt[1]), 2, (0, 255, 0), -1)
            bottom = right_edges_raw[0]
            cv2.circle(tracking_debug_viz, (bottom[0], bottom[1]), 6, (0, 255, 0), -1)
        else:
            # Fallback if strict criteria fail
            # print(f"[FAIL] No valid candidates found (Candidates tested: {len(candidate_limits)})")
            tracking_failed = True
            
        # Draw previous contour in Red to show where we fallback to, if tracking failed
        if tracking_failed and prev_right_contour is not None and len(prev_right_contour) > 0:
            # print(f"[TRACKING] Falling back to previous frame's contour (Length {len(prev_right_contour)})")
            for pt in prev_right_contour:
                cv2.circle(tracking_debug_viz, (pt[0], pt[1]), 2, (0, 0, 255), -1)
            bottom = prev_right_contour[0]
            cv2.circle(tracking_debug_viz, (bottom[0], bottom[1]), 6, (0, 0, 255), -1)
            
            # Calculate stats for the fallback line to display transparency
            pts = np.array(prev_right_contour)
            x_pts = pts[:, 0]
            y_pts = pts[:, 1]
            f_len = len(x_pts)
            if len(y_pts) > 2:
                fit = np.polyfit(y_pts, x_pts, 1)
                f_proj = fit[0] * (height - 1) + fit[1]
            else:
                f_proj = x_pts[0]
            f_mean = np.mean(np.abs(x_pts - center_x_img))
            f_proj_dist = f_proj - center_x_img
            
            f_txt = f"FALLBACK | L:{f_len} | P:{f_proj_dist:.0f} | M:{f_mean:.1f}"
            (tw, th), _ = cv2.getTextSize(f_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = bottom[0] + 10
            if text_x + tw > width:
                text_x = bottom[0] - 10 - tw
            text_y = bottom[1] - 25
            if text_y < 20:
                text_y = bottom[1] + 25
            
            cv2.rectangle(tracking_debug_viz, (text_x - 2, text_y - th - 5), (text_x + tw + 2, text_y + 5), (0, 0, 0), -1)
            cv2.putText(tracking_debug_viz, f_txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)




        if not tracking_failed and len(right_edges_raw) > 0:
            for pt in right_edges_raw:
                cv2.circle(tracking_debug_viz, (pt[0], pt[1]), 2, (0, 255, 0), -1)
            bottom = right_edges_raw[0]
            cv2.circle(tracking_debug_viz, (bottom[0], bottom[1]), 6, (0, 255, 0), -1)
    else:
        # print(f"[FAIL] No candidate_limits found (Connected Components: {num_labels-1})")
        tracking_failed = True
        
    if save_path_intermediate_dir and img_path:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_3.5_edge_tracking_debug.png"), tracking_debug_viz)

    if tracking_failed:
        # Pass the failure directly to the Environment so its Kalman Filter can handle the prediction
        return (np.zeros_like(drivable_mask), 1.0, [], [], np.zeros_like(drivable_mask), np.zeros_like(drivable_mask), image, image, [], [], [])

    if ego_label > 0:
        drivable_mask_clean = (labels == ego_label).astype(np.uint8) * 255
    
    if save_path_intermediate_dir and img_path:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_drivable_clean.png"), drivable_mask_clean)

    # Step 2: Extract Region of Interest
    roi = drivable_mask_clean[roi_top:height, :]
    
    # Store tracked state
    best_right_raw = right_edges_raw

    # Step 4: Temporal Edge Filtering & Step 5: Center Estimation
    center_points = []
    prev_right_edge = None
    right_edges_for_viz = []

    # REVERSE edges so it goes bottom-up as the original expected
    # Since we appended from bottom to top, it's already bottom-up, wait: 
    # we appended from height-1 to roi_top-1. So the list is bottom-to-top.
    # The original loop reversed it: right_edges_raw.reverse() so it goes top-to-bottom? No, original loop iterated bottom to top, then reversed it to make it top-to-bottom.
    # We will just reverse it.
    right_edges_raw_reversed = list(reversed(right_edges_raw))

    if right_edges_raw_reversed:
        # We anchor the top of the perspective (LANE_WIDTH_TOP) to a fixed visual horizon
        horizon_y = (height // 2) + 70

        for right_edge, y_img in right_edges_raw_reversed:
            if prev_right_edge is not None:
                if abs(right_edge - prev_right_edge) > MAX_EDGE_JUMP:
                    continue

            prev_right_edge = right_edge
            right_edges_for_viz.append([right_edge, y_img])

            # Interpolate lane width based on vertical position (perspective geometry)
            progress_down_image = (y_img - horizon_y) / max(1, (height - horizon_y))
            progress_down_image = max(0.0, min(1.0, progress_down_image))

            current_lane_width = LANE_WIDTH_TOP + (LANE_WIDTH_BOTTOM - LANE_WIDTH_TOP) * progress_down_image

            center_x = right_edge - int(current_lane_width / 2)
            center_x = np.clip(center_x, 0, width - 1)
            center_points.append([center_x, y_img])

    # Create a visualization of the edges and centers on the clean mask
    edges_viz = cv2.cvtColor(drivable_mask_clean, cv2.COLOR_GRAY2BGR)
    for pt in right_edges_for_viz:
        cv2.circle(edges_viz, (pt[0], pt[1]), 2, (0, 0, 255), -1)  # Red for right edge
    for pt in center_points:
        cv2.circle(edges_viz, (pt[0], pt[1]), 2, (255, 0, 0), -1)  # Blue for center points

    if save_path_intermediate_dir and img_path:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_edges_and_centers.png"), edges_viz)

    output_mask = np.zeros_like(drivable_mask)

    # Polyfit to get a smooth curve
    poly_points = []
    path_points_for_sampling = []
    if len(center_points) > 10:
        try:
            c_pts = np.array(center_points)
            weights = (c_pts[:, 1] / height) ** 2
            fit = np.polyfit(c_pts[:, 1], c_pts[:, 0], POLY_DEGREE, w=weights)
            curve_fn = np.poly1d(fit)
            y_min, y_max = c_pts[:, 1].min(), height - 1
            plot_y = np.linspace(y_min, y_max, 50).astype(int)
            fit_x = np.clip(curve_fn(plot_y).astype(int), 0, width - 1)

            pts = np.array([np.transpose(np.vstack([fit_x, plot_y]))], np.int32)
            cv2.polylines(output_mask, pts, False, 255, 2)
            poly_points = pts.squeeze() if pts.shape[0] > 0 else []
            path_points_for_sampling = np.transpose(np.vstack([fit_x, plot_y]))
        except Exception as e:
            print(f"[{base_name}] Drivable area polyfit Error: {e}")
            pass

    # --- UNIFIED STEP 4: Calculate 10 points and distance directly from the PATH ---
    center_lanes = []
    distance_to_center = 1.0  # Default to max error

    if len(path_points_for_sampling) > 1:
        try:
            # We sample directly from it via interpolation.
            path_ys = path_points_for_sampling[:, 1]
            path_xs = path_points_for_sampling[:, 0]

            y_min, y_max = path_ys.min(), path_ys.max()

            # Generate 10 evenly spaced y-coordinates to sample
            target_ys = np.linspace(y_min, y_max, 10).astype(int)

            # For each target y, find the corresponding x by linearly interpolating from the path
            interp_xs = np.interp(target_ys, path_ys, path_xs)

            center_lanes = np.column_stack((interp_xs.astype(np.int32), target_ys)).astype(np.int32)

            # Calculate weighted error for distance_to_center
            raw_errors = (center_lanes[:, 0] - (width // 2))
            weights = np.linspace(0.2, 1.0, len(raw_errors))
            weighted_error = np.sum(raw_errors * weights) / np.sum(weights)
            distance_to_center = np.clip(weighted_error / (width // 4), -1.0, 1.0)
        except Exception as e:
            print(f"[{base_name}] ERROR during direct sampling from path: {e}")

    if len(center_lanes) == 0:
        center_lanes = np.array([[0, 0] for _ in range(10)])

    t_boundaries = time.time()

    # Visualization
    final_output_viz = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- SCALING FIX ---
    # Scale the coordinates from the letterboxed space back to the original image space
    h_orig, w_orig = image.shape[:2]
    h_resized, w_resized = drivable_mask.shape[:2]

    scale = min(w_resized / w_orig, h_resized / h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)
    x_offset = (w_resized - new_w) // 2
    y_offset = (h_resized - new_h) // 2

    # Scale the final path points and draw as a green polyline
    if len(path_points_for_sampling) > 1:
        # Vectorized Scaling
        scaled_path_points = (path_points_for_sampling - np.array([x_offset, y_offset])) / scale
        scaled_path_points = scaled_path_points.astype(np.int32)
        cv2.polylines(final_output_viz, [scaled_path_points.reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 0),
                      thickness=2)

    # Draw the 10 evenly spaced center points after scaling
    scaled_center_lanes = []
    if len(center_lanes) > 0:
        scaled_centers_arr = (center_lanes - np.array([x_offset, y_offset])) / scale
        scaled_centers_arr = scaled_centers_arr.astype(np.int32)
        for p in scaled_centers_arr:
            scaled_center_lanes.append((int(p[0]), int(p[1])))
            cv2.circle(final_output_viz, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)  # Red dots for the 10 points

    if save_path_intermediate_dir and img_path:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_drivable_centers.png"), final_output_viz)

    center_lanes = np.array(scaled_center_lanes )

    t_end = time.time()
    print(
        f"[{base_name}] PIPELINE TIMING | ROI: {(t_roi - t_start) * 1000:.1f}ms | Inference: {(t_inference - t_roi) * 1000:.1f}ms | LinesPost: {(t_lines_post - t_inference) * 1000:.1f}ms | DrivablePost: {(t_da_post - t_lines_post) * 1000:.1f}ms | Boundaries: {(t_boundaries - t_da_post) * 1000:.1f}ms | Scaling/Misc: {(t_end - t_boundaries) * 1000:.1f}ms | TOTAL: {(t_end - t_start) * 1000:.1f}ms")

    # To match the hybrid agent's 11-item return signature for easy swapping:
    return (output_mask / 255).astype(np.uint8), distance_to_center, center_lanes, center_lanes,            np.zeros_like(output_mask), np.zeros_like(output_mask), final_output_viz,            edges_viz, np.zeros_like(output_mask), best_right_raw, best_right_raw



def detect_lanes_yolop_v2_hybrid_agent(image, save_path_intermediate_dir=None, img_path=None,
                                       force_drivable_fallback=False, force_classical_fallback=False,
                                       reference_point=None, prev_left_contour=None, prev_right_contour=None):
    t_start = time.time()
    base_name = Path(img_path).stem if img_path else "debug"

    orig_height, orig_width = image.shape[:2]

    # --- NO MORE PRE-INFERENCE MASKING ---
    # We pass the full raw image to YOLOP to give the network full context
    t_roi = time.time()

    # Step 1 & 2: Calculate both drivable area and lines first
    with torch.no_grad():
        da_seg_out, ll_seg_out, resized_image, metadata = run_yolop_v2_inference(image)
        t_inference = time.time()

        ll_segment = detect_yolop_v2_lines(resized_image, force_classical_fallback,
                                           ll_seg_out_from_inference=ll_seg_out)
        t_lines_post = time.time()

        drivable_mask = process_yolop_v2_drivable_output(da_seg_out, resized_image)
        t_da_post = time.time()

    height, width = ll_segment.shape

    # --- APPLY TRAPEZOIDAL ROI TO THE MASKS ---
    roi_vertices = np.array([
        [0, height - 20],  # Bottom-left (adjusted to exclude hood)
        [width, height - 20],  # Bottom-right
        [width, height // 2.5],  # Top-right
        [0, height // 2.5]  # Top-left
    ], dtype=np.int32)
    roi_mask = np.zeros_like(ll_segment)
    cv2.fillPoly(roi_mask, [roi_vertices], 255)

    # Apply ROI to masks to isolate relevant areas
    ll_segment = cv2.bitwise_and(ll_segment, ll_segment, mask=roi_mask)
    drivable_mask = cv2.bitwise_and(drivable_mask, drivable_mask, mask=roi_mask)

    if save_path_intermediate_dir:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_0_raw_line_mask.png"), ll_segment)
        # Visualization of ROI on original image for debug
        roi_viz = image.copy()
        v_scaled = roi_vertices.copy()
        v_scaled[:, 0] = v_scaled[:, 0] * orig_width / width
        v_scaled[:, 1] = v_scaled[:, 1] * orig_height / height
        cv2.polylines(roi_viz, [v_scaled.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_roi_visualization.png"), roi_viz)

    # Use the simplified line-finding logic which now only extends lines
    final_mask_image, _, _, _, _, _, final_hybrid_path_points_from_tracker, lines_debug_viz_from_tracker, all_paths_viz_from_tracker, all_extended_lines_mask_from_tracker, candidates_viz_from_tracker, best_left_raw, best_right_raw = _find_and_draw_lane_boundaries(
        ll_segment, save_path_intermediate_dir, base_name, reference_point=reference_point, drivable_mask=drivable_mask,
        prev_left_contour=prev_left_contour, prev_right_contour=prev_right_contour)
    t_boundaries = time.time()

    # cv2.imshow("candidates", candidates_viz_from_tracker)
    if save_path_intermediate_dir:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_candidates_viz.png"),
                    candidates_viz_from_tracker)

    # Step 3: Check if any lines were extended or if fallback is forced.
    if np.any(all_extended_lines_mask_from_tracker) and not force_drivable_fallback:
        # --- PRIMARY LOGIC: Lines were detected. Use the extended lines mask directly. ---
        if save_path_intermediate_dir:
            print(f"[{base_name}] SUCCESS: Lines detected. Returning all extended lines.")
        final_mask = final_mask_image
    else:
        # --- FALLBACK LOGIC: No lines detected, use drivable area ---
        if save_path_intermediate_dir:
            print(f"[{base_name}] FALLBACK: No lines detected. Using drivable-area logic.")
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_1_drivable_mask.png"),
                        drivable_mask)

        # The mask is already isolated via the trapezoidal ROI applied above
        ego_lane_mask = drivable_mask

        if save_path_intermediate_dir:
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_2_roi_applied.png"),
                        ego_lane_mask)

        center_points = []
        for y in range(height // 2, height, 4):
            row = ego_lane_mask[y, :]
            white_pixels = np.where(row > 0)[0]
            if len(white_pixels) > 1:
                center_points.append([(white_pixels[0] + white_pixels[-1]) // 2, y])

        final_mask = np.zeros_like(drivable_mask)
        final_hybrid_path_points_from_tracker = []
        if len(center_points) > 10:
            c_pts = np.array(center_points)
            try:
                weights = (c_pts[:, 1] / height) ** 2
                fit = np.polyfit(c_pts[:, 1], c_pts[:, 0], 2, w=weights)
                curve_fn = np.poly1d(fit)
                y_min, y_max = c_pts[:, 1].min(), height - 1
                plot_y = np.linspace(y_min, y_max, 50).astype(int)
                fit_x = np.clip(curve_fn(plot_y).astype(int), 0, width - 1)

                pts = np.array([np.transpose(np.vstack([fit_x, plot_y]))], np.int32)
                cv2.polylines(final_mask, pts, False, 255, 2)
                final_hybrid_path_points_from_tracker = np.transpose(np.vstack([fit_x, plot_y]))
            except Exception as e:
                print(f"[{base_name}] ERROR in drivable fallback polyfit: {e}")

    # --- UNIFIED STEP 4: Calculate 10 points and distance directly from the HYBRID PATH ---
    center_lanes = []
    distance_to_center = 1.0  # Default to max error

    path_points_for_sampling = final_hybrid_path_points_from_tracker
    if len(path_points_for_sampling) > 1:
        try:
            # The hybrid path is our source of truth. We sample directly from it via interpolation.
            path_ys = path_points_for_sampling[:, 1]
            path_xs = path_points_for_sampling[:, 0]

            y_min, y_max = path_ys.min(), path_ys.max()

            # Generate 10 evenly spaced y-coordinates to sample
            target_ys = np.linspace(y_min, y_max, 10).astype(int)

            # For each target y, find the corresponding x by linearly interpolating from the hybrid path
            # np.interp is perfect for this as it handles 1D interpolation efficiently.
            interp_xs = np.interp(target_ys, path_ys, path_xs)

            center_lanes = np.column_stack((interp_xs.astype(np.int32), target_ys)).astype(np.int32)

            # Calculate weighted error for distance_to_center
            raw_errors = (center_lanes[:, 0] - (width // 2))
            weights = np.linspace(0.2, 1.0, len(raw_errors))
            weighted_error = np.sum(raw_errors * weights) / np.sum(weights)
            distance_to_center = np.clip(weighted_error / (width // 4), -1.0, 1.0)
        except Exception as e:
            print(f"[{base_name}] ERROR during direct sampling from hybrid path: {e}")

    if len(center_lanes) == 0:
        center_lanes = np.array([[0, 0] for _ in range(10)])

    # Visualization for both paths
    final_output_viz = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- SCALING FIX ---
    # Replicate the letterboxing math used in preprocess_image_for_yolop to invert it
    # This correctly scales coordinates from the padded 640x640 space back to the original image space
    h_orig, w_orig = image.shape[:2]
    h_resized, w_resized = final_mask.shape[:2]

    scale = min(w_resized / w_orig, h_resized / h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)
    x_offset = (w_resized - new_w) // 2
    y_offset = (h_resized - new_h) // 2

    # Scale the final path points and draw as a green polyline
    if len(path_points_for_sampling) > 1:
        # Vectorized Scaling
        scaled_path_points = (path_points_for_sampling - np.array([x_offset, y_offset])) / scale
        scaled_path_points = scaled_path_points.astype(np.int32)
        cv2.polylines(final_output_viz, [scaled_path_points.reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 0),
                      thickness=2)

    # Draw the 10 evenly spaced center points after scaling
    scaled_center_lanes = []
    if len(center_lanes) > 0:
        scaled_centers_arr = (center_lanes - np.array([x_offset, y_offset])) / scale
        scaled_centers_arr = scaled_centers_arr.astype(np.int32)
        for p in scaled_centers_arr:
            scaled_center_lanes.append((int(p[0]), int(p[1])))
            cv2.circle(final_output_viz, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)  # Red dots for the 10 points

    # Crucially update the returned array so the RL environment gets the correctly scaled points
    center_lanes = np.array(scaled_center_lanes)

    if save_path_intermediate_dir:
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_3_final_output.png"),
                    final_output_viz)

    t_end = time.time()
    print(
        f"[{base_name}] PIPELINE TIMING | ROI: {(t_roi - t_start) * 1000:.1f}ms | Inference: {(t_inference - t_roi) * 1000:.1f}ms | LinesPost: {(t_lines_post - t_inference) * 1000:.1f}ms | DrivablePost: {(t_da_post - t_lines_post) * 1000:.1f}ms | Boundaries: {(t_boundaries - t_da_post) * 1000:.1f}ms | Scaling/Misc: {(t_end - t_boundaries) * 1000:.1f}ms | TOTAL: {(t_end - t_start) * 1000:.1f}ms")


    # --- TRACKING LOGIC (DEPRECATED) ---
    # The Kalman filter in followlane_carla_sb_3 now handles temporal smoothing and jumps
    # We no longer force a tracking failure here based on 50px distance.
    tracking_failed = False
    if center_lanes is None or len(center_lanes) == 0:
        tracking_failed = True

    if tracking_failed:
        # Pass the failure directly to the Environment so its Kalman Filter can handle the prediction.
        # CRITICAL FIX: Return final_output_viz instead of raw image so the user can see what the perception DID manage to find (e.g. candidates)
        return (np.zeros_like(final_mask), 1.0, [], [], np.zeros_like(final_mask), np.zeros_like(final_mask), final_output_viz, final_output_viz, candidates_viz_from_tracker if 'candidates_viz_from_tracker' in locals() else np.zeros_like(final_mask), [], [])

    return (final_mask / 255).astype(
        np.uint8), distance_to_center, center_lanes, ll_seg_out, ll_segment, all_extended_lines_mask_from_tracker if 'all_extended_lines_mask_from_tracker' in locals() else np.zeros_like(final_mask), final_output_viz, lines_debug_viz_from_tracker if 'lines_debug_viz_from_tracker' in locals() else np.zeros_like(final_mask), candidates_viz_from_tracker if 'candidates_viz_from_tracker' in locals() else np.zeros_like(final_mask), best_left_raw, best_right_raw

def detect_lanes_yolop_v2_lines_agent(image, save_path_intermediate_dir=None, img_path=None,
                                      force_classical_fallback=False):
    with torch.no_grad():
        ll_segment = detect_yolop_v2_lines(image, force_classical_fallback)
    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_raw.png"), ll_segment)

    # Use the simplified line-finding logic which now only extends lines
    output_mask, _, _, debug_viz, _, _, all_extended_lines_mask, _, _ = _find_and_draw_lane_boundaries(ll_segment,
                                                                                                       save_path_intermediate_dir,
                                                                                                       base_name)

    if save_path_intermediate_dir and img_path:
        base_name = Path(img_path).stem
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_Edge-Optimized_Consensus_Debug.png"),
                    debug_viz)

    # Return the combined extended line mask directly and default values
    final_mask = all_extended_lines_mask if all_extended_lines_mask is not None else np.zeros_like(ll_segment)
    distance_to_center = 0.0  # Placeholder
    center_lanes = np.array(
        [[image.shape[1] // 2, y] for y in range(image.shape[0] // 2, image.shape[0], image.shape[0] // 20)])

    return (final_mask / 255).astype(np.uint8), distance_to_center, center_lanes, center_lanes


def normalize_centers(centers):
    x_centers = centers[:, 0]
    x_centers_normalized = (x_centers / 640).tolist()
    states = x_centers_normalized
    y_centers = centers[:, 1]
    y_centers_normalized = (y_centers / 640).tolist()  # Adjusted for 384 height
    states = states + y_centers_normalized  # Returns a list
    return states, x_centers_normalized, y_centers_normalized


def merge_and_extend_lines(lines, ll_segment):
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
    for line in merged_lines:
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        cv2.line(merged_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the merged lines
    cv2.imshow('Merged Lines', merged_image) if show_images else None

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


def lane_detection_overlay(image, left_mask, right_mask):
    res = np.copy(image)
    # We show only points with probability higher than 0.5
    res[left_mask > 0.5, :] = [255, 0, 0]
    res[right_mask > 0.5, :] = [0, 0, 255]
    return res


def extract_green_lines(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the green color in HSV space
    lower_green = np.array([35, 100, 200])
    upper_green = np.array([85, 255, 255])

    # Create a mask for the green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    return green_mask


def detect_lines(raw_image, detection_mode, processing_mode, img_path, save_path_intermediate_dir=None,
                 force_drivable_fallback=False, force_classical_fallback=False, prev_left_contour=None,
                 prev_right_contour=None):
    if detection_mode == 'yolop_v2_lines':
        detected_lines, distance_to_center, distance_to_center_normalized, centers = detect_lanes_yolop_v2_lines_agent(
            raw_image, save_path_intermediate_dir, img_path, force_classical_fallback)
        x_centers = [point[0] for point in centers]
        return detected_lines, distance_to_center, distance_to_center_normalized, np.array(x_centers)
    elif detection_mode == 'yolop_v2_drivable':
        results = detect_lanes_yolop_v2_drivable_agent(
            raw_image, save_path_intermediate_dir, img_path)
        
        detected_lines = results[0]
        distance_to_center = results[1]
        center_lanes = results[2]
        
        x_centers = [point[0] for point in center_lanes]
        return detected_lines, distance_to_center, center_lanes, np.array(x_centers)
    elif detection_mode == 'yolop_v2_hybrid':
        # detect_lanes_yolop_v2_hybrid_agent returns 11 values:
        # 0:mask, 1:dist, 2:lanes1, 3:lanes2, 4:ll_seg, 5:ext_mask, 6:path_pts, 7:debug_viz, 8:cand_viz, 9:left_raw, 10:right_raw
        results = detect_lanes_yolop_v2_hybrid_agent(
            raw_image, save_path_intermediate_dir, img_path, force_drivable_fallback, force_classical_fallback,
            reference_point=None, prev_left_contour=prev_left_contour, prev_right_contour=prev_right_contour)

        detected_lines = results[0]
        distance_to_center = results[1]
        center_lanes = results[2]
        final_hybrid_path_points_from_tracker = results[6]
        lines_debug_viz_output = results[7]
        best_left_raw = results[9]
        best_right_raw = results[10]

        x_centers = [point[0] for point in center_lanes]
        return detected_lines, distance_to_center, center_lanes, np.array(
            x_centers), None, None, final_hybrid_path_points_from_tracker, lines_debug_viz_output, best_left_raw, best_right_raw


def choose_lane(distance_to_center_normalized, center_points):
    last_row = len(x_row) - 1
    closest_lane_index = min(enumerate(distance_to_center_normalized[last_row]), key=lambda x: abs(x[1]))[0]
    distances = [array[closest_lane_index] if len(array) > closest_lane_index else min(array) for array in
                 distance_to_center_normalized]
    centers = [array[closest_lane_index] if len(array) > closest_lane_index else min(array) for array in center_points]
    return distances, centers


def choose_lane_v1(distance_to_center_normalized, center_points):
    close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                          distance_to_center_normalized]
    distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
    centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
    return distances, centers


def find_lane_center(mask):
    # Find the indices of 1s in the array
    mask_array = np.array(mask)
    indices = np.where(mask_array > 0.8)[0]

    # If there are no 1s or only one set of 1s, return None
    if len(indices) < 2:
        # TODO (Ruben) For the moment we will be dealing with no detection as a fixed number
        return [NO_DETECTED]

    # Find the indices where consecutive 1s change to 0
    diff_indices = np.where(np.diff(indices) > 1)[0]
    # If there is only one set of 1s, return None
    if len(diff_indices) == 0:
        return [NO_DETECTED]

    interested_line_borders = np.array([], dtype=np.int8)
    for index in diff_indices:
        interested_line_borders = np.append(interested_line_borders, indices[index])
        interested_line_borders = np.append(interested_line_borders, int(indices[index + 1]))

    midpoints = calculate_midpoints(interested_line_borders)
    return midpoints


def calculate_midpoints(input_array):
    midpoints = []
    for i in range(0, len(input_array) - 1, 2):
        midpoint = (input_array[i] + input_array[i + 1]) // 2
        midpoints.append(midpoint)
    return midpoints


def detect_missing_points(lines):
    num_points = len(lines)
    max_line_points = max(len(line) for line in lines)
    missing_line_count = sum(1 for line in lines if len(line) < max_line_points)

    return missing_line_count > 0 and missing_line_count <= num_points // 2


def interpolate_missing_points(input_lists, x_row):
    # Find the index of the list with the maximum length
    max_length_index = max(range(len(input_lists)), key=lambda i: len(input_lists[i]))

    # Determine the length of the complete lists
    complete_length = len(input_lists[max_length_index])

    # Initialize a list to store the inferred list
    inferred_list = []

    # Iterate over each index in x_row
    for i, x_value in enumerate(x_row):
        # If the current index is in the list with incomplete points
        if len(input_lists[i]) < complete_length:
            interpolated_list = []
            for points_i in range(complete_length):
                # TODO calculates interpolated point of missing line and then build the interpolated_list
                # Since it is not trivial, we just discard this point from the moment
                interpolated_y = NO_DETECTED
                interpolated_list.append(interpolated_y)
            inferred_list.append(interpolated_list)
        else:
            # If the current list is complete, simply append the corresponding y value
            inferred_list.append(input_lists[i])

    return inferred_list


def calculate_lines_percent(mask):
    width = mask.shape[1]
    center_image = width / 2

    line_size = images_high - upper_limit
    line_thresholds = np.array(mask[upper_limit: images_high]) > 0.8

    # Initialize counters
    positive_count = 0
    negative_count = 0

    # Iterate through each set of indices
    for thresholds in line_thresholds:
        indices = np.where(thresholds)[0]

        # Check if any index is positive
        if any(index - center_image > 0 for index in indices):
            positive_count += 1
        # Check if any index is negative
        if any(index - center_image < 0 for index in indices):
            negative_count += 1

    left_lane_perc = negative_count / line_size
    right_lane_perc = positive_count / line_size

    return left_lane_perc, right_lane_perc


def calculate_center_v1(mask):
    width = mask.shape[1]
    center_image = width / 2
    lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]
    center_lane_indexes = [
        find_lane_center(lines[x]) for x, _ in enumerate(lines)
    ]

    # this part consists of checking the number of lines detected in all rows
    # then discarding the rows (set to 1) in which more or less centers are detected
    center_lane_indexes = discard_not_confident_centers(center_lane_indexes)

    center_lane_distances = [
        [center_image - x for x in inner_array] for inner_array in center_lane_indexes
    ]

    # Calculate the average position of the lane lines
    ## normalized distance
    distance_to_center_normalized = [
        np.array(x) / (width - center_image) for x in center_lane_distances
    ]
    return center_lane_indexes, distance_to_center_normalized


def calculate_center(mask):
    width = mask.shape[1]
    center_image = width / 2
    lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]
    center_lane_indexes = [
        find_lane_center(lines[x]) for x, _ in enumerate(lines)
    ]

    if detect_missing_points(center_lane_indexes):
        center_lane_indexes = interpolate_missing_points(center_lane_indexes, x_row)
    # this part consists of checking the number of lines detected in all rows
    # then discarding the rows (set to 1) in which more or less centers are detected
    center_lane_indexes = discard_not_confident_centers(center_lane_indexes)

    center_lane_distances = [
        [center_image - x for x in inner_array] for inner_array in center_lane_indexes
    ]

    # Calculate the average position of the lane lines
    ## normalized distance
    distance_to_center_normalized = [
        np.array(x) / (width - center_image) for x in center_lane_distances
    ]
    return center_lane_indexes, distance_to_center_normalized


def discard_not_confident_centers(center_lane_indexes):
    # Count the occurrences of each list size leaving out of the equation the non-detected
    size_counter = Counter(len(inner_list) for inner_list in center_lane_indexes if NO_DETECTED not in inner_list)
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
            inner_list = [NO_DETECTED] * len(inner_list)  # Set all elements to 1
        result.append(inner_list)

    return result


def get_ll_seg_image(dists, ll_segment, suffix="", name='ll_seg'):
    ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
    ll_segment_all = [np.copy(ll_segment_int8), np.copy(ll_segment_int8), np.copy(ll_segment_int8)]

    # draw the midpoint used as right center lane
    for index, dist in zip(x_row, dists):
        # Set the value at the specified index and distance to 1
        add_midpoints(ll_segment_all[0], index, dist)

    # draw a line for the selected perception points
    for index in x_row:
        for i in range(630):
            ll_segment_all[0][index][i] = 255
    ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
    # We now show the segmentation and center lane postprocessing
    cv2.imshow(name + suffix, ll_segment_stacked) if show_images else None
    return ll_segment_stacked


def add_midpoints(ll_segment, index, dist):
    # Set the value at the specified index and distance to 1
    draw_dash(index, dist, ll_segment)
    draw_dash(index + 2, dist, ll_segment)
    draw_dash(index + 1, dist, ll_segment)
    draw_dash(index - 1, dist, ll_segment)
    draw_dash(index - 2, dist, ll_segment)


def draw_dash(index, dist, ll_segment):
    ll_segment[index, dist - 1] = 255  # <-- here is the real calculated center
    ll_segment[index, dist - 3] = 255
    ll_segment[index, dist - 2] = 255
    ll_segment[index, dist - 4] = 255
    ll_segment[index, dist - 5] = 255
    ll_segment[index, dist - 6] = 255


def calculate_and_plot_lines_counts_above(save_results_benchmark_path, percs, yolop_left_perc, yolop_right_perc,
                                          yolop_v2_lines_left_perc, yolop_v2_lines_right_perc,
                                          lane_detector_left_perc, lane_detector_right_perc,
                                          lane_detector_v3_left_perc, lane_detector_v3_right_perc,
                                          programmatic_left_perc, programmatic_right_perc,
                                          perfect_left_perc, perfect_right_perc, processing_mode):
    for perc in percs:
        yolop_counts = calculate_lines_counts_above(perc, yolop_left_perc, yolop_right_perc)
        yolop_v2_lines_counts = calculate_lines_counts_above(perc, yolop_v2_lines_left_perc, yolop_v2_lines_right_perc)
        lane_det_counts = calculate_lines_counts_above(perc, lane_detector_left_perc,
                                                       lane_detector_right_perc)
        lane_det_v3_counts = calculate_lines_counts_above(perc, lane_detector_v3_left_perc,
                                                          lane_detector_v3_right_perc)
        prog_counts = calculate_lines_counts_above(perc, programmatic_left_perc, programmatic_right_perc)

        perfect_counts = calculate_lines_counts_above(perc, perfect_left_perc, perfect_right_perc)

        subplots = 6 if processing_mode != "none" else 5

        fig2, axs2 = plt.subplots(1, subplots, figsize=(18, 6))

        axs2[0].bar(['Just left', 'Just right', 'both', 'none'], yolop_counts, color=['blue', 'green'])
        axs2[0].set_title('YOLOP')
        axs2[0].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[0].set_ylim(0, 100)

        axs2[1].bar(['Just left', 'Just right', 'both', 'none'], yolop_v2_lines_counts, color=['blue', 'green'])
        axs2[1].set_title('yolop_v2_lines')
        axs2[1].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[1].set_ylim(0, 100)

        axs2[2].bar(['Just left', 'Just right', 'both', 'none'], lane_det_counts, color=['blue', 'green'])
        axs2[2].set_title('Lane Detector')
        axs2[2].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[2].set_ylim(0, 100)

        axs2[3].bar(['Just left', 'Just right', 'both', 'none'], lane_det_v3_counts, color=['blue', 'green'])
        axs2[3].set_title('Lane Detector V3')
        axs2[3].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[3].set_ylim(0, 100)

        axs2[4].bar(['Just left', 'Just right', 'both', 'none'], perfect_counts, color=['blue', 'green'])
        axs2[4].set_title('Perfect')
        axs2[4].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[4].set_ylim(0, 100)

        if processing_mode != "none":
            axs2[5].bar(['Just left', 'Just right', 'both', 'none'], prog_counts, color=['blue', 'green'])
            axs2[5].set_title('Programmatic')
            axs2[5].set_ylabel(f'percentage of images above {perc * 100}% detected')
            axs2[5].set_ylim(0, 100)

        plt.savefig(save_results_benchmark_path / f'plot_{perc * 100}_above.png')
        plt.close()
    pass


def calculate_lines_counts_above(threshold, left_perc, right_perc):
    return [
        (np.sum((np.array(left_perc) >= threshold) & (np.array(right_perc) < threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) < threshold) & (np.array(right_perc) >= threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) >= threshold) & (np.array(right_perc) >= threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) < threshold) & (np.array(right_perc) < threshold)) / len(left_perc)) * 100,
    ]


def perform_all_benchmarking(dataset, processing_mode, detection_modes=None, save_intermediate=False,
                             force_drivable_fallback=False, force_classical_fallback=False, consecutive=False):
    if detection_modes is None:
        detection_modes = []
    # Create the save directory if it doesn't exist
    save_results_benchmark_dir = str(opt.save_dir + "/benchmark/" + processing_mode + "/")
    save_results_benchmark_path = Path(save_results_benchmark_dir)
    save_results_benchmark_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(save_results_benchmark_dir):
        for file_name in os.listdir(save_results_benchmark_dir):
            file_path = os.path.join(save_results_benchmark_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {save_results_benchmark_dir}: {e}")

    all_modes = {
        "yolop_v2_lines": "yolop_v2_lines",
        "yolop_v2_drivable": "yolop_v2_drivable",
        "yolop_v2_hybrid": "yolop_v2_hybrid",
    }

    modes_to_run = detection_modes if detection_modes else all_modes.keys()

    results = {}

    for mode in modes_to_run:
        if mode in all_modes:
            if processing_mode == 'none' and 'yolop_v2' in mode:
                print(f"Skipping {mode} for processing_mode {processing_mode}.")
                continue
            print(f"Running benchmark on {mode} with processing_mode: {processing_mode}")
            times, errors, left_perc, right_perc, percentage = benchmark_one(dataset, mode, processing_mode,
                                                                             save_intermediate, force_drivable_fallback,
                                                                             force_classical_fallback, consecutive)
            results[mode] = {
                "times": times, "errors": errors, "left_perc": left_perc,
                "right_perc": right_perc, "percentage": percentage, "label": all_modes[mode]
            }

    if not results:
        print("No benchmarks were run. Exiting.")
        return

    # Dynamically build lists for plotting
    labels = [res["label"] for res in results.values()]
    percentages = [res["percentage"] for res in results.values()]
    times = [np.mean(res["times"]) for res in results.values()]
    errors = [np.mean(res["errors"]) if res["errors"] else 0 for res in results.values()]

    # Generate colors for the plots
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    # Plot 1: Percentage of Detected Images
    plt.figure(figsize=(10, 10))
    plt.bar(labels, percentages, color=colors)
    plt.title('Percentage of Detected Images')
    plt.ylabel('Percentage of images perfectly detected')
    plt.ylim(0, 100)
    plt.savefig(save_results_benchmark_path / 'plot0.png')
    plt.close()

    # Plot 2: Average Times
    plt.figure(figsize=(10, 10))
    plt.bar(labels, times, color=colors)
    plt.title('Average Times')
    plt.ylabel('Time (s)')
    plt.savefig(save_results_benchmark_path / 'plottimes.png')
    plt.close()

    # Plot 3: Average errors
    plt.figure(figsize=(10, 6))
    pixel_errors = np.array(errors)  # NOT Assuming 320 is the scaling factor
    plt.bar(labels, pixel_errors, color=colors)
    plt.xlabel('Perception Mode')
    plt.ylabel('Average Error (pixels)')
    plt.title('Average error when both lines detected')
    plt.savefig(save_results_benchmark_path / 'ploterrolane.png')
    plt.close()

    print(f"All plots saved to {save_results_benchmark_path}")



def benchmark_one(dataset, detection_mode, processing_mode, save_intermediate=False, force_drivable_fallback=False,
                  force_classical_fallback=False, consecutive=False):
    global prev_fit
    prev_fit = None
    print("running benchmarking on " + detection_mode)
    # Run inference
    t0 = time.time()

    save_path_bad_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/bad")
    save_path_good_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/good")
    save_path_bad_raw_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/bad_raw")
    save_path_out_raw_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/good_raw")
    save_results_metrics_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/metrics")
    save_path_intermediate_dir = None
    if save_intermediate:
        save_path_intermediate_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/intermediate")

    # TODO avoid this duplicated code
    if not os.path.exists(save_path_bad_dir):
        os.makedirs(save_path_bad_dir)

    if not os.path.exists(save_path_good_dir):
        os.makedirs(save_path_good_dir)

    if not os.path.exists(save_path_bad_raw_dir):
        os.makedirs(save_path_bad_raw_dir)

    if not os.path.exists(save_path_out_raw_dir):
        os.makedirs(save_path_out_raw_dir)

    if not os.path.exists(save_results_metrics_dir):
        os.makedirs(save_results_metrics_dir)

    if save_intermediate and not os.path.exists(save_path_intermediate_dir):
        os.makedirs(save_path_intermediate_dir)

    directories = [save_path_bad_dir, save_path_good_dir, save_path_bad_raw_dir, save_path_out_raw_dir,
                   save_results_metrics_dir]
    if save_intermediate:
        directories.append(save_path_intermediate_dir)
    for directory_path in directories:
        if os.path.exists(directory_path):
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {directory_path}: {e}")

    detected = 0
    all_images = 0
    all_avg_errors_when_detected = []
    all_left_perc = []
    all_right_perc = []
    times = []
    prev_left_contour = None
    prev_right_contour = None
    for i, (path, img, img_det, vid_cap, shapes) in enumerate(dataset):
        all_images += 1

        save_path_bad = str(save_path_bad_dir + '/' + Path(path).name)
        save_path_good = str(save_path_good_dir + '/' + Path(path).name)
        save_path_bad_raw = str(save_path_bad_raw_dir + '/' + Path(path).name)
        save_path_out_raw = str(save_path_out_raw_dir + '/' + Path(path).name)

        height = img_det.shape[0]
        width = img_det.shape[1]

        # MIMIC ONLINE QUALITY: Do not pre-resize. Use original 1280x720 quality.
        # The detect_lines/inference functions will handle their own internal resizing if needed.
        resized_img_np = img_det.copy()

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        # resized_img_np = np.array(resized_img)
        start = time.time()

        detect_lines_results = detect_lines(resized_img_np, detection_mode, processing_mode, path,
                                            save_path_intermediate_dir, force_drivable_fallback,
                                            force_classical_fallback, prev_left_contour, prev_right_contour)

        # Ensure we have at least 10 elements
        padded_results = list(detect_lines_results) + [None] * max(0, 10 - len(detect_lines_results))
        ll_seg_out_list = padded_results[0]
        distance_to_center = padded_results[1]
        center_lanes_normalized = padded_results[2]
        x_centers = padded_results[3]

        if consecutive and detection_mode == 'yolop_v2_hybrid':
            prev_left_contour = padded_results[8]
            prev_right_contour = padded_results[9]

        ll_seg_out = ll_seg_out_list

        # Original 'good'/'bad' logic for other models
        if processing_mode == "none":
            left_percent, right_percent = calculate_lines_percent(ll_seg_out)
            if left_percent > PERFECT_THRESHOLD and right_percent > PERFECT_THRESHOLD:
                detected += 1
                cv2.imwrite(save_path_good, ll_seg_out)
                cv2.imwrite(save_path_out_raw, img)
            else:
                cv2.imwrite(save_path_bad, ll_seg_out)
                cv2.imwrite(save_path_bad_raw, img)
        else:
            overlay = resized_img_np.copy()
            # Create a color mask for the overlay
            color_mask = np.zeros_like(resized_img_np)
            # ll_seg_out is a single-channel mask of detected lines
            ll_seg_out_resized = cv2.resize(ll_seg_out, (resized_img_np.shape[1], resized_img_np.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
            color_mask[ll_seg_out_resized > 0] = [0, 0, 255]  # Red for detected lines
            # Blend the original image with the color mask
            overlayed_image = cv2.addWeighted(overlay, 1, color_mask, 0.5, 0)

            # Draw the 10 center points onto the overlayed image using the correct variable
            if center_lanes_normalized is not None and len(center_lanes_normalized) > 0:
                for point in center_lanes_normalized:
                    center_point = (int(point[0]), int(point[1]))
                    cv2.circle(overlayed_image, center_point, 5, (0, 255, 255), -1)  # Draw yellow dots

            if wasDetected(x_centers.tolist()):
                detected += 1
                cv2.imwrite(save_path_good, overlayed_image)
                cv2.imwrite(save_path_out_raw, img)
            else:
                cv2.imwrite(save_path_bad, overlayed_image)
                cv2.imwrite(save_path_bad_raw, img)

        processing_time = time.time() - start
        times.append(processing_time)

        # This part below is now redundant for yolop_v2_lines but needed for other models' metrics
        # ll_seg_out_raw = ll_seg_out
        # (
        #     center_lanes,
        #     distance_to_center_normalized,
        # ) = calculate_center_v1(ll_seg_out)
        # right_lane_normalized_distances, right_center_lane = choose_lane_v1(distance_to_center_normalized, center_lanes)

        # ll_segment_stacked = get_ll_seg_image(right_center_lane, ll_seg_out)

        left_percent, right_percent = calculate_lines_percent(ll_seg_out)
        all_left_perc.append(left_percent)
        all_right_perc.append(right_percent)

        if dataset.mode == 'images':
            total_error = 0
            detected_points = 0
            # for x in right_lane_normalized_distances:
            #     if abs(x) != 1:
            #         detected_points += 1
            #         total_error += abs(x)

            if detected_points > 0:
                average_abs = total_error / detected_points
                all_avg_errors_when_detected.append(average_abs)

        else:
            cv2.imshow('image', ll_seg_out)
            cv2.waitKey(1)  # 1 millisecond

        cv2.waitKey(0) if show_images else None

    cv2.waitKey(10000) if show_images else None
    print('Done. (%.3fs)' % (time.time() - t0))
    print(f"total good -> {detected} images of {all_images} were detected = {(detected / all_images) * 100}%.")

    file_name = processing_mode + "_errors.pkl"
    save_results_metrics_errors_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_errors_file, 'wb') as file:
        pickle.dump(all_avg_errors_when_detected, file)

    file_name = processing_mode + "_left.pkl"
    save_results_metrics_left_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_left_file, 'wb') as file:
        pickle.dump(all_left_perc, file)

    file_name = processing_mode + "_right.pkl"
    save_results_metrics_right_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_right_file, 'wb') as file:
        pickle.dump(all_right_perc, file)

    return times, all_avg_errors_when_detected, all_left_perc, all_right_perc, (detected / all_images) * 100


def wasDetected(labels: list):
    if not labels:
        return False  # Empty list is a failure

    # Check for yolop_v2_lines failure case (all zeros)
    is_all_zeros = all(x == 0 for x in labels)
    if is_all_zeros:
        return False

    # Check for other modes' failure case (all abs(1))
    is_all_ones = all(abs(x) == 1 for x in labels)
    if is_all_ones:
        return False

    # If neither failure case is met, it's a success
    return True


def anyDetected(labels: list):
    for i in range(len(labels)):
        if abs(labels[i]) < 0.8:
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/home/ruben/Desktop/broken_perc',
                        help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='/home/ruben/Desktop/broken_perc_output/',
                        help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--detection-modes', nargs='+', type=str, help='List of detection modes to benchmark')
    parser.add_argument('--save-intermediate-images', action='store_true',
                        help='save all intermediate images for debugging')
    parser.add_argument('--benchmark', action='store_true',
                        help='run without saving any images to measure pure performance')
    parser.add_argument('--force-drivable-fallback', action='store_true',
                        help='force the hybrid agent to use drivable area fallback logic')
    parser.add_argument('--force-classical-fallback', action='store_true',
                        help='force to only use classical lane fallback.')
    parser.add_argument('--consecutive', action='store_true',
                        help='treat offline images as consecutive frames and use tracking.')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)