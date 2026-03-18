import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

def _fit_weighted_robust(pts, height):
    y_pts, x_pts = pts[:, 1], pts[:, 0]
    weights = np.exp((y_pts - height) / (height / 2))
    best_fit, max_inliers = None, -1
    for _ in range(10):
        idx = np.random.choice(len(pts), min(len(pts), 5), replace=False)
        try:
            current_degree = 3 if len(np.unique(y_pts[idx])) > 3 else 1
            sample_fit = np.polyfit(y_pts[idx], x_pts[idx], current_degree)
            inliers = np.abs(np.polyval(sample_fit, y_pts) - x_pts) < 25
            if np.sum(inliers) > max_inliers:
                max_inliers = np.sum(inliers)
                final_degree = 3 if len(np.unique(y_pts[inliers])) > 3 else 1
                best_fit = np.polyfit(y_pts[inliers], x_pts[inliers], final_degree, w=weights[inliers])
        except (np.linalg.LinAlgError, TypeError): continue
    return np.poly1d(best_fit) if best_fit is not None else None


def _extend_contour(contour, target_y, direction, num_points_for_fit=30):
    """
    Extends a contour to a target y-coordinate using a localized polynomial fit.
    - contour: The numpy array of [x, y] points.
    - target_y: The y-coordinate to extend to.
    - direction: 'down' or 'up'.
    - num_points_for_fit: How many points from the end of the contour to use for fitting.
    """
    if contour is None or len(contour) < 2:
        return contour

    # Sort points by y-coordinate to ensure correct order
    contour = contour[contour[:, 1].argsort()]

    y_pts, x_pts = contour[:, 1], contour[:, 0]

    # Decide which points to use for fitting the extension curve
    if direction == 'down':
        if y_pts[-1] >= target_y: return contour  # Already past the target
        fit_points_y = y_pts[-num_points_for_fit:]
        fit_points_x = x_pts[-num_points_for_fit:]
        y_start = y_pts[-1]
        y_end = target_y
    else:  # 'up'
        if y_pts[0] <= target_y: return contour  # Already past the target
        fit_points_y = y_pts[:num_points_for_fit]
        fit_points_x = x_pts[:num_points_for_fit]
        y_start = target_y
        y_end = y_pts[0]

    if len(fit_points_y) < 2: return contour  # Not enough points to fit

    try:
        # Fit a curve to the end segment of the line
        # Require a minimum number of points (e.g., 10) to confidently fit a quadratic curve.
        # Overfitting a 2nd-degree polynomial to very few points creates wild, unnecessary curves.
        degree = 2 if len(fit_points_y) > 30 else 1
        fit = np.polyfit(fit_points_y, fit_points_x, degree)

        # --- ROBUSTNESS CHECK ---
        # If the quadratic term is too large, the curve is unstable.
        # Fall back to a simple, robust linear fit to prevent "weird" curves.
        if degree == 2 and abs(fit[0]) > 0.005:
            fit = np.polyfit(fit_points_y, fit_points_x, 1)

        fit_fn = np.poly1d(fit)

        # Generate new points for the extension
        num_new_points = abs(int(y_end - y_start))
        if num_new_points <= 0: return contour

        y_new = np.linspace(y_start, y_end, num_new_points).astype(int)
        x_new = fit_fn(y_new).astype(int)

        extension_points = np.vstack((x_new, y_new)).T

        # Append or prepend the new points to the original contour
        if direction == 'down':
            return np.vstack((contour, extension_points))
        else:
            return np.vstack((extension_points, contour))

    except (np.linalg.LinAlgError, TypeError):
        # If fitting fails, just return the original contour
        return contour

def _find_and_draw_lane_boundaries(binary_mask, save_path_intermediate_dir=None, base_name=None, reference_point=None):
    height, width = binary_mask.shape
    debug_viz = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # --- Step 1: Skeletonize to get thin lines ---
    skeleton = skeletonize(binary_mask // 255).astype(np.uint8) * 255

    # --- Step 2: Robust Splitting at Junctions ---
    # To find junctions in smooth curves (common with neural network outputs),
    # we look at a wider neighborhood (5x5) instead of just immediate neighbors.
    # A point on a simple line will have few neighbors in this larger area,
    # but a merge point will be "busier" and have more.
    kernel = np.ones((5, 5), dtype=np.uint8)  # Use a larger 5x5 kernel
    neighbor_count = cv2.filter2D(skeleton // 255, -1, kernel)

    # A junction is a point on the original skeleton that is "busy"
    # (has more than 5 neighbors in its 5x5 radius).
    # This threshold is loose enough to catch smooth merges.
    junction_points = np.argwhere((neighbor_count > 5) & (skeleton > 0)).tolist()

    # --- NEW: Detect lines that go "up and down" (inverted U-shapes) ---
    # These are continuous curves (like merged lanes at the horizon) that don't form
    # a busy junction, so the kernel misses them. We find connected components and
    # split them at their highest point (minimum y) if they have parts in the same y coordinate.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    for i in range(1, num_labels):
        pts_y, pts_x = np.where(labels == i)

        if len(pts_y) < 20:  # Ignore tiny components
            continue

        peak_y = np.min(pts_y)
        peak_x = int(np.mean(pts_x[pts_y == peak_y]))

        # User's logic: check if the line has parts in the same y coordinate
        is_u_shape = False
        unique_y = np.unique(pts_y)
        for y_val in unique_y:
            xs_at_y = np.sort(pts_x[pts_y == y_val])
            if len(xs_at_y) > 1 and np.max(np.diff(xs_at_y)) > 10:
                # Gap of more than 10 pixels at the same Y coordinate means distinct legs
                is_u_shape = True
                break

        if is_u_shape:
            junction_points.append([peak_y, peak_x])

    # Create a copy of the skeleton to "erase" the junctions from.
    split_skeleton = skeleton.copy()
    for y, x in junction_points:
        cv2.circle(split_skeleton, (int(x), int(y)), 5, 0, -1)  # Erase a 5-pixel radius around each junction.

    if save_path_intermediate_dir:
        # Create a visualization showing where the junctions were detected.
        junction_viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        for y, x in junction_points:
            cv2.circle(junction_viz, (int(x), int(y)), 5, (0, 0, 255), 1)  # Draw red circles on the original skeleton.
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_1_junctions_erased.png"), junction_viz)
        cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_1.5_split_skeleton.png"), split_skeleton)

    # --- Step 3: Find Contours of Clean Segments ---
    # Now that junctions are erased, we can find contours of the simple, separated line segments.
    contours, _ = cv2.findContours(split_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter out tiny contours that are likely just noise.
    min_contour_length = 50
    long_contours = [c for c in contours if cv2.arcLength(c, False) > min_contour_length]

    # The complex logic for splitting merged contours is no longer needed.
    # We now have a clean list of line segments.
    processed_contours = long_contours

    # --- Step 4: Classify Contours and Create Candidate Visualization ---
    lane_candidates = []
    # Create a new visualization for all potential lane candidates, starting from the skeleton
    candidates_viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    for contour in long_contours:
        points = contour.reshape(-1, 2)
        x_pts, y_pts = points[:, 0], points[:, 1]

        if len(y_pts) < 3: continue  # Need at least 3 points for a line

        try:
            # Fit a simple line just to get its position at the bottom of the image
            fit = np.polyfit(y_pts, x_pts, 1)  # Linear fit is sufficient
            fit_fn = np.poly1d(fit)

            lane_candidates.append({
                'points': points,
                'fit_fn': fit_fn
            })
            # Draw this candidate contour in green on the visualization image
            cv2.polylines(candidates_viz, [points.reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 0), thickness=1)
        except (np.linalg.LinAlgError, TypeError):
            continue

    # --- Step 5: Select Best Left and Right Lanes ---
    if not lane_candidates:
        # Return empty/default values if no lanes are found, including the new viz image
        return np.zeros_like(binary_mask), [], [], debug_viz, None, None, np.array(
            []), debug_viz, debug_viz, np.zeros_like(binary_mask), candidates_viz

    y_bottom = height - 1
    LANE_WIDTH_PIXELS = 100  # Approximate lane width in pixels
    MATCHING_THRESHOLD = 300  # Max pixels a lane can drift between frames

    # If we have a reference point, use it for tracking. Otherwise, fall back to image center.
    expected_center_x = reference_point[0] if reference_point is not None else width / 2

    expected_left_x = expected_center_x - (LANE_WIDTH_PIXELS / 2)
    expected_right_x = expected_center_x + (LANE_WIDTH_PIXELS / 2)

    # Draw vertical lines for expected positions on the candidates visualization
    cv2.line(candidates_viz, (int(expected_left_x), 0), (int(expected_left_x), height - 1), (255, 100, 0),
             1)  # Blueish for left
    cv2.line(candidates_viz, (int(expected_right_x), 0), (int(expected_right_x), height - 1), (0, 100, 255),
             1)  # Orangish for right
    cv2.line(candidates_viz, (int(expected_center_x), 0), (int(expected_center_x), height - 1), (255, 255, 255),
             1)  # White for center ref

    left_lanes, right_lanes = [], []

    # Classify candidates and find their distance to the *expected* positions
    for lane in lane_candidates:
        x_bottom = lane['fit_fn'](y_bottom)
        if x_bottom < expected_center_x:
            dist = abs(x_bottom - expected_left_x)
            left_lanes.append({'lane': lane, 'dist': dist})
        else:
            dist = abs(x_bottom - expected_right_x)
            right_lanes.append({'lane': lane, 'dist': dist})

    # Select the best lanes, but only if they are within the matching threshold
    best_left, best_right = None, None
    if left_lanes:
        best_left_candidate = min(left_lanes, key=lambda x: x['dist'])
        if best_left_candidate['dist'] < MATCHING_THRESHOLD:
            best_left = best_left_candidate['lane']['points']

    if right_lanes:
        best_right_candidate = min(right_lanes, key=lambda x: x['dist'])
        if best_right_candidate['dist'] < MATCHING_THRESHOLD:
            best_right = best_right_candidate['lane']['points']

    # Highlight the chosen lanes on the visualization image
    if best_left is not None:
        cv2.polylines(candidates_viz, [best_left.reshape((-1, 1, 2))], isClosed=False, color=(255, 0, 0),
                      thickness=2)  # Blue for chosen left
    if best_right is not None:
        cv2.polylines(candidates_viz, [best_right.reshape((-1, 1, 2))], isClosed=False, color=(0, 0, 255),
                      thickness=2)  # Red for chosen right

    best_left_contour = best_left
    best_right_contour = best_right

    # --- Step 6: Line Extension ---
    # Ensure both detected lanes stretch from the middle to the bottom of the image.
    if best_left_contour is not None:
        best_left_contour = _extend_contour(best_left_contour, target_y=height - 1, direction='down')
        best_left_contour = _extend_contour(best_left_contour, target_y=int(height * 0.6), direction='up')

    if best_right_contour is not None:
        best_right_contour = _extend_contour(best_right_contour, target_y=height - 1, direction='down')
        best_right_contour = _extend_contour(best_right_contour, target_y=int(height * 0.6), direction='up')

    # --- Step 7: Path Generation with Fallbacks (using original points) ---
    # If one lane is missing, we create a virtual contour by shifting the existing one.
    if best_left_contour is not None and best_right_contour is None:
        best_right_contour = best_left_contour.copy()
        best_right_contour[:, 0] += LANE_WIDTH_PIXELS
    elif best_right_contour is not None and best_left_contour is None:
        best_left_contour = best_right_contour.copy()
        best_left_contour[:, 0] -= LANE_WIDTH_PIXELS

    # --- Step 8: Accurate Midpoint Calculation from Contours ---
    hybrid_path_pts = []
    if best_left_contour is not None and best_right_contour is not None:
        # Create lookup dictionaries for fast access to x-coordinates for each y
        left_lookup = {pt[1]: pt[0] for pt in best_left_contour}
        right_lookup = {pt[1]: pt[0] for pt in best_right_contour}

        # Find the common range of y-values, now guaranteed to be from mid to bottom
        y_points = np.linspace(int(height * 0.65), height - 20, 10).astype(int)

        for y in y_points:
            # Find the closest available y in each lookup in case of discrete points
            actual_y_left = min(left_lookup.keys(), key=lambda k: abs(k - y))
            actual_y_right = min(right_lookup.keys(), key=lambda k: abs(k - y))

            x_left = left_lookup[actual_y_left]
            x_right = right_lookup[actual_y_right]

            x_mid = (x_left + x_right) / 2
            hybrid_path_pts.append([int(x_mid), y])

    # --- Final Mask Generation & Visualization ---
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

    return final_mask, [], [], debug_viz, None, None, final_hybrid_path_points, lines_debug_viz, all_paths_viz, all_extended_lines_mask, candidates_viz

def apply_clahe_bgr(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def soften_overbright_lanes(image, white_threshold=130):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find very bright pixels
    mask = gray > white_threshold

    result = image.copy()

    # Reduce brightness slightly instead of eroding
    result[mask] = result[mask] * 0.85

    return result.astype(np.uint8)

def apply_road_roi(mask, keep_ratio=0.25):
    """
    Keeps only the bottom portion of the image.
    keep_ratio = 0.45 means bottom 45% is kept.
    """
    h = mask.shape[0]
    roi_mask = np.zeros_like(mask)
    roi_mask[int(h * (1 - keep_ratio)):, :] = 255
    return cv2.bitwise_and(mask, roi_mask)

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

class LaneDetector:
    def __init__(self, car=None, x_row=None, camera_transform=None, fov=None, n_points=None):
        self.device = self._select_device()
        self.yolop_v2_lines_model = self._load_model()
        self.yolop_v2_lines_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.last_detected_point = None
        self.n_points = n_points

    def _select_device(self, logger=None, device='', batch_size=None):
        cpu_request = device.lower() == 'cpu'
        if device and not cpu_request:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
            assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device

        cuda = False if cpu_request else torch.cuda.is_available()
        return torch.device('cuda:0' if cuda else 'cpu')

    def _load_model(self):
        model_path = "/home/ruben/Desktop/unibotics/src/RL-Studio/rl_studio/envs/carla/utils/yolop/weights/yolopv2.pt"
        model = torch.jit.load(model_path, map_location=self.device).float()
        model.to(self.device)
        model.eval()
        return model

    def _resize_image_for_model(self, image):
        return cv2.resize(image, (640, 384))

    def _scale_and_draw_visualization(self, original_image, resized_image, final_mask, center_lanes, save_path_intermediate_dir=None, base_name=None):
        # 1. Create the 'overlayed_image'
        overlay_image = original_image.copy()
        
        h_orig, w_orig, _ = overlay_image.shape
        h_resized, w_resized = final_mask.shape if final_mask is not None else resized_image.shape[:2]

        if final_mask is not None:
            # Ensure overlay_image is 3-channel before color assignment
            visible_mask_resized = cv2.resize(final_mask, (w_orig, h_orig),
                                            interpolation=cv2.INTER_NEAREST)
            mask = visible_mask_resized > 128
            if np.any(mask):
                overlay_image[mask] = [0, 255, 255] # BGR for yellow, thresholded

        # --- FIX: Scale points instead of resizing ---
        w_scale = w_orig / w_resized
        h_scale = h_orig / h_resized
        
        valid_points_resized = [(int(p[0] * w_scale), int(p[1] * h_scale)) for p in center_lanes]
        # --- END FIX ---

        # Draw center points
        for point in valid_points_resized:
            cv2.circle(overlay_image, (int(point[0]), int(point[1])), 5, (255, 0, 255), -1) # Magenta

        if save_path_intermediate_dir:
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_3_final_output.png"), overlay_image)

        center_lanes_scaled = np.array(valid_points_resized, dtype=np.int32)
        center_lanes_scaled = center_lanes_scaled[::-1]

        return center_lanes_scaled, overlay_image

    def get_resized_image(self, sensor_data, target_w=640, target_h=384):
        # Check if it's a CARLA raw sensor object or already a numpy array
        if hasattr(sensor_data, 'raw_data'):
            # Case: Raw CARLA sensor data
            array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
            rgb_image = array[:, :, :3]
        else:
            # Case: It's already a numpy array (e.g. from a previous cv2 step)
            rgb_image = sensor_data
            # If it has 4 channels (BGRA), drop the 4th
            if rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[:, :, :3]

        # Optimized Resize using OpenCV
        resized_img = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return resized_img

    def process_image(self, image):
        if hasattr(image, "shape"):
            print(f"DEBUG: process_image input shape: {image.shape}")
        image = self.get_resized_image(image)
        print(f"DEBUG: process_image after get_resized_image shape: {image.shape}")
        ll_segment, distance_to_center, raw_center_lanes_normalized, _, raw_detection_image, extended_image, paths_image, points_for_extension, _ = self.detect_lanes_yolop_v2_hybrid_agent(
            image, reference_point=self.last_detected_point
        )

        if raw_center_lanes_normalized is not None and len(raw_center_lanes_normalized) > 0:
            self.last_detected_point = raw_center_lanes_normalized[-1]

        _, processed_image = self._scale_and_draw_visualization(image, image, ll_segment, raw_center_lanes_normalized)

        if len(raw_detection_image.shape) == 2:
            raw_detection_image = cv2.cvtColor(raw_detection_image, cv2.COLOR_GRAY2RGB)

        return raw_center_lanes_normalized, processed_image, distance_to_center, raw_detection_image, paths_image

    def run_yolop_v2_inference(self, raw_image, stretch_factor=1.0):
        print(f"DEBUG: run_yolop_v2_inference input shape: {raw_image.shape}")

        # --- STEP 1: CLAHE only ---
        raw_image = apply_clahe_bgr(raw_image)
        raw_image = soften_overbright_lanes(raw_image)

        h, w, _ = raw_image.shape

        new_h = int(h * stretch_factor)
        new_h = ((new_h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32

        resized_image = cv2.resize(raw_image, (new_w, new_h))
        print(f"DEBUG: run_yolop_v2_inference resized_image shape: {resized_image.shape}")

        img = self.yolop_v2_lines_transform(resized_image).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = self.yolop_v2_lines_model(img)

        return outputs[1], outputs[2], resized_image

    def detect_yolop_v2_lines(self, raw_image, force_classical_fallback=False, ll_seg_out_from_inference=None):
        if ll_seg_out_from_inference is None:
            # Resize as your model expects
            raw_image = cv2.resize(raw_image, (640, 384))

            img = self.yolop_v2_lines_transform(raw_image).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            outputs = self.yolop_v2_lines_model(img)

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
        probs_np = probs.squeeze().cpu().numpy()

        # Normalize relative to max activation
        normalized = (probs_np - probs_np.min()) / (probs_np.max() - probs_np.min() + 1e-6)

        # Keep strongest activations only
        threshold_value = 0.6  # try 0.5 – 0.7
        yolo_mask = (normalized > threshold_value).astype(np.uint8) * 255

        # Reconnect dashed lines post-unmerging with a conservative vertical kernel
        kernel = np.ones((5, 1), np.uint8)
        yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel)

        # ==========================
        # CLASSICAL CV FALLBACK
        # ==========================

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

        if not force_classical_fallback:
            return yolo_mask

        cv_mask = classical_lane_fallback(raw_image)

        # Apply road ROI
        cv_mask = apply_road_roi(cv_mask, keep_ratio=0.35)

        # Now filter lane-like regions
        # cv_mask_filtered = filter_lane_like_regions(cv_mask)

        hybrid_mask = cv2.bitwise_or(yolo_mask, cv_mask)

        return hybrid_mask

    def detect_lanes_yolop_v2_hybrid_agent(self, image, save_path_intermediate_dir=None, img_path=None,
                                           force_drivable_fallback=False, force_classical_fallback=False,
                                           reference_point=None):
        print(f"DEBUG: detect_lanes_yolop_v2_hybrid_agent input image shape: {image.shape}")
        base_name = Path(img_path).stem if img_path else "debug"
        # Step 1 & 2: Calculate both drivable area and lines first
        with torch.no_grad():
            da_seg_out, ll_seg_out, resized_image = self.run_yolop_v2_inference(image)
            ll_segment = self.detect_yolop_v2_lines(resized_image, force_classical_fallback,
                                               ll_seg_out_from_inference=ll_seg_out)
            print(f"DEBUG: detect_lanes_yolop_v2_hybrid_agent ll_segment shape: {ll_segment.shape}")
            # drivable_mask = process_yolop_v2_drivable_output(da_seg_out, resized_image)

        height, width = ll_segment.shape

        # --- EGO LANE ISOLATION with STATIC TRAPEZOIDAL ROI ---
        # 1. Define the vertices of the trapezoid that represents the ego lane.
        # These points are fine-tuned for a 640x384 image with a standard forward-facing camera.
        roi_vertices = np.array([
            [0, height - 10],  # Bottom-left
            [width, height - 10],  # Bottom-right
            [width, height // 2.5],  # Top-right
            [0, height // 2.5]  # Top-left
        ], dtype=np.int32)

        # 2. Create a black mask and draw the filled trapezoid onto it.
        roi_mask = np.zeros_like(ll_segment)
        cv2.fillPoly(roi_mask, [roi_vertices], 255)
        ll_segment = cv2.bitwise_and(ll_segment, ll_segment, mask=roi_mask)

        if save_path_intermediate_dir:
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_0_raw_line_mask.png"), ll_segment)

        # Use the simplified line-finding logic which now only extends lines
        final_mask_image, _, _, _, _, _, final_hybrid_path_points_from_tracker, lines_debug_viz_from_tracker, all_paths_viz_from_tracker, all_extended_lines_mask_from_tracker, candidates_viz_from_tracker = _find_and_draw_lane_boundaries(
            ll_segment, save_path_intermediate_dir, base_name, reference_point=reference_point)

        if save_path_intermediate_dir:
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_candidates_viz.png"),
                        candidates_viz_from_tracker)

        # Step 3: Check if any lines were extended or if fallback is forced.
        # if np.any(all_extended_lines_mask) and not force_drivable_fallback: # OJO!! FALBACK when town03 added
        if True:  # Here we should add the condition to check if lines were detected
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

            roi_vertices = np.array([
                [0, height], [width, height],
                [width // 2 + 80, height // 2], [width // 2 - 80, height // 2]
            ], dtype=np.int32)
            roi_mask = np.zeros_like(drivable_mask)
            cv2.fillPoly(roi_mask, [roi_vertices], 255)
            ego_lane_mask = cv2.bitwise_and(drivable_mask, drivable_mask, mask=roi_mask)

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
            if len(center_points) > 10:
                c_pts = np.array(center_points)
                try:
                    fit = np.polyfit(c_pts[:, 1], c_pts[:, 0], 2)
                    curve_fn = np.poly1d(fit)
                    y_min, y_max = c_pts[:, 1].min(), height - 1
                    plot_y = np.linspace(y_min, y_max, 50).astype(int)
                    fit_x = np.clip(curve_fn(plot_y).astype(int), 0, width - 1)

                    pts = np.array([np.transpose(np.vstack([fit_x, plot_y]))], np.int32)
                    cv2.polylines(final_mask, pts, False, 255, 2)
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
        # Scale the coordinates from the resized space back to the original image space
        h_orig, w_orig, _ = final_output_viz.shape
        h_resized, w_resized = final_mask.shape
        w_scale = w_orig / w_resized
        h_scale = h_orig / h_resized

        # Scale the final path points and draw as a green polyline
        if len(path_points_for_sampling) > 1:
            scaled_path_points = (path_points_for_sampling * [w_scale, h_scale]).astype(np.int32)
            cv2.polylines(final_output_viz, [scaled_path_points.reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 0),
                          thickness=2)

        # Draw the 10 evenly spaced center points after scaling
        for p in center_lanes:
            scaled_p = (int(p[0] * w_scale), int(p[1] * h_scale))
            cv2.circle(final_output_viz, scaled_p, 5, (255, 0, 0), -1)  # Red dots for the 10 points

        if save_path_intermediate_dir:
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_3_final_output.png"),
                        final_output_viz)

        return (final_mask / 255).astype(
            np.uint8), distance_to_center, center_lanes, center_lanes, ll_segment, all_extended_lines_mask_from_tracker, final_output_viz, lines_debug_viz_from_tracker, candidates_viz_from_tracker

    def normalize_centers(self, centers):
        x_centers = centers[:, 0]
        x_centers_normalized = (x_centers / 640).tolist()
        states = x_centers_normalized
        y_centers = centers[:, 1]
        y_centers_normalized = (y_centers / 512).tolist()
        states = states + y_centers_normalized
        return states, x_centers_normalized, y_centers_normalized

    def calculate_v_goal(self, mean_curvature, center_distance, deviated_points):
        dist_error = abs(center_distance) * 10
        close_error = 0
        mean_curv = max(0, mean_curvature - 1) * 10
        if deviated_points >= self.n_points / 2:
            mean_curv = max(0, mean_curvature - 1) * 30; close_error = 9
        elif deviated_points >= self.n_points / 3:
            mean_curv = max(0, mean_curvature - 1) * 20; close_error = 6
        elif deviated_points >= self.n_points / 4:
            mean_curv = max(0, mean_curvature - 1) * 10; close_error = 3
        v_goal = max(9, 25 - (mean_curv + dist_error))
        return max(2, v_goal - close_error)

    def average_curvature_from_centers(self, center_points):
        total_angle = 0.0
        for i in range(1, len(center_points) - 1):
            p1, p2, p3 = np.array(center_points[i-1]), np.array(center_points[i]), np.array(center_points[i+1])
            v1, v2 = p2 - p1, p3 - p2
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0: continue
            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            total_angle += np.arccos(cos_theta)
        return total_angle