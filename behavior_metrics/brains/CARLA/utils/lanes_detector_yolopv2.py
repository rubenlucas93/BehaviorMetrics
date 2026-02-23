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

def _find_and_draw_lane_boundaries(binary_mask, save_path_intermediate_dir=None, base_name=None, reference_point=None):
    height, width = binary_mask.shape
    debug_viz = np.zeros((height, width, 3), dtype=np.uint8)
    Y_GAP_THRESHOLD, MAX_PATH_JUMP_DISTANCE, MAX_LOST_FRAMES = 8, 30, 8
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    active_paths, terminated_paths = [], []

    for y in range(height // 2, height, 4):
        row = closed_mask[y, :]
        padded_row = np.pad(row, (1, 1), 'constant')
        diffs = np.diff(padded_row.astype(np.int32))
        starts, ends = np.where(diffs > 0)[0], np.where(diffs < 0)[0]
        current_midpoints = []
        if len(starts) > 0 and len(starts) == len(ends):
            marking_centers = [(s + e) // 2 for s, e in zip(starts, ends)]
            if len(marking_centers) >= 2:
                current_midpoints = [[(marking_centers[i] + marking_centers[i+1]) // 2, y] for i in range(len(marking_centers) - 1)]
        if not active_paths and current_midpoints:
            active_paths = [{'pts': [midpt], 'lost_count': 0} for midpt in current_midpoints]
            continue
        if not current_midpoints:
            for path_data in active_paths: path_data['lost_count'] += 1
        elif active_paths:
            cost_matrix = np.full((len(active_paths), len(current_midpoints)), fill_value=1e5)
            for i, path_data in enumerate(active_paths):
                last_pt = np.array(path_data['pts'][-1])
                prediction = last_pt + (last_pt - np.array(path_data['pts'][-2])) * (path_data['lost_count'] + 1) if len(path_data['pts']) >= 2 else last_pt
                for j, midpt in enumerate(current_midpoints): cost_matrix[i, j] = np.linalg.norm(prediction - np.array(midpt))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_paths, assigned_midpoints = set(), set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < MAX_PATH_JUMP_DISTANCE:
                    active_paths[r]['pts'].append(current_midpoints[c]); active_paths[r]['lost_count'] = 0
                    assigned_paths.add(r); assigned_midpoints.add(c)
            for i in range(len(active_paths)):
                if i not in assigned_paths: active_paths[i]['lost_count'] += 1
            for i in range(len(current_midpoints)):
                if i not in assigned_midpoints: active_paths.append({'pts': [current_midpoints[i]], 'lost_count': 0})
        active_paths = [p for p in active_paths if p['lost_count'] < MAX_LOST_FRAMES or (len(p['pts']) > 10 and terminated_paths.append(p['pts']))]

    final_paths = terminated_paths + [p['pts'] for p in active_paths if len(p['pts']) > 10]
    
    all_paths_viz = np.zeros((height, width, 3), dtype=np.uint8)
    for path in final_paths:
        color = tuple(np.random.randint(60, 256, 3).tolist())
        pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(all_paths_viz, [pts], False, color, 1)

    best_path, best_score, fitted_paths = None, float('inf'), []
    long_paths = [p for p in final_paths if len(p) > 10]

    if not long_paths: return np.zeros_like(binary_mask), [], [], debug_viz, None, None, np.array([], dtype=np.int32), None, all_paths_viz, None
    for path in long_paths:
        fn = _fit_weighted_robust(np.array(path), height)
        if fn: fitted_paths.append({'path': path, 'fit': fn})
    if not fitted_paths: return np.zeros_like(binary_mask), [], [], debug_viz, None, None, np.array([], dtype=np.int32), None, all_paths_viz, None
    
    for fitted in fitted_paths:
        path_points = np.array(fitted['path']); shape_fn = fitted['fit']
        last_points = path_points[-min(len(path_points), 20):]
        final_bottom_x = shape_fn(height-1)
        if len(last_points) >= 2:
            try: final_bottom_x = np.poly1d(np.polyfit(last_points[:, 1], last_points[:, 0], 1))(height - 1)
            except (np.linalg.LinAlgError, TypeError): pass
        score = abs(final_bottom_x - (reference_point[0] if reference_point is not None else width / 2))
        if score < best_score: best_score, best_path = score, path_points

    final_mask, hybrid_path_pts = np.zeros_like(binary_mask), []
    if best_path is not None:
        try:
            shape_fn = _fit_weighted_robust(best_path, height)
            if shape_fn:
                if len(best_path) > 0:
                    first_point = best_path[0]
                    if first_point[1] > height // 2:
                        extension_ys_up = np.arange(height // 2, int(first_point[1]))
                        if len(extension_ys_up) > 0:
                            extension_xs_up = np.clip(shape_fn(extension_ys_up), 0, width - 1)
                            hybrid_path_pts.extend([[ex, ey] for ex, ey in zip(reversed(extension_xs_up), reversed(extension_ys_up))])
                    hybrid_path_pts.append(first_point.tolist())
                for i in range(len(best_path) - 1):
                    p1, p2 = best_path[i], best_path[i+1]
                    if p2[1] - p1[1] > Y_GAP_THRESHOLD:
                        gap_ys = np.arange(p1[1] + 1, p2[1])
                        if len(gap_ys) > 0:
                            gap_xs = np.clip(shape_fn(gap_ys), 0, width - 1)
                            hybrid_path_pts.extend([[gx, gy] for gx, gy in zip(gap_xs, gap_ys)])
                    hybrid_path_pts.append(p2.tolist())
                if hybrid_path_pts:
                    last_point = hybrid_path_pts[-1]
                    if last_point[1] < height - 1:
                        last_points = best_path[-min(len(best_path), 20):]
                        extension_fn = shape_fn
                        if len(last_points) >= 2:
                            try: extension_fn = np.poly1d(np.polyfit(last_points[:, 1], last_points[:, 0], 1))
                            except (np.linalg.LinAlgError, TypeError): pass
                        extension_ys = np.arange(int(last_point[1]) + 1, height)
                        if len(extension_ys) > 0:
                            extension_xs = np.clip(extension_fn(extension_ys), 0, width - 1)
                            hybrid_path_pts.extend([[ex, ey] for ex, ey in zip(extension_xs, extension_ys)])
                path_pts = np.array(hybrid_path_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(final_mask, [path_pts], False, 255, 2)
        except: cv2.polylines(final_mask, [best_path.reshape((-1, 1, 2))], False, 255, 2)
    return final_mask, [], [], debug_viz, None, None, np.array(hybrid_path_pts, dtype=np.int32), None, all_paths_viz, None

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

def detect_yolop_v2_lines(raw_image, force_classical_fallback=False, ll_seg_out_from_inference=None):
    ll_seg_out = ll_seg_out_from_inference
    ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, size=(raw_image.shape[0], raw_image.shape[1]), mode='bilinear', align_corners=False)
    probs = torch.sigmoid(ll_seg_mask)
    probs_np = probs.squeeze().cpu().numpy()
    _, yolo_mask = cv2.threshold((probs_np * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel)
    return yolo_mask

class LaneDetector:
    def __init__(self, car=None, x_row=None, camera_transform=None, fov=None, n_points=None):
        self.device = self._select_device()
        self.yolop_v2_lines_model = self._load_model()
        self.yolop_v2_lines_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.n_points = n_points

    def _select_device(self, logger=None, device='', batch_size=None):
        cpu_request = device.lower() == 'cpu'
        if device and not cpu_request:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
            assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device

        cuda = False if cpu_request else torch.cuda.is_available()
        return torch.device('cuda:0' if cuda else 'cpu')

    def _load_model(self):
        model_path = "/home/ruben/Desktop/2020-phd-ruben-lucas/src/RL-Studio/rl_studio/envs/carla/utils/yolop/weights/yolopv2.pt"
        model = torch.jit.load(model_path, map_location=self.device).float()
        model.to(self.device)
        model.eval()
        return model

    def _resize_image_for_model(self, image):
        return cv2.resize(image, (640, 384))

    def _scale_and_draw_visualization(self, original_image, resized_image, final_mask, center_lanes, save_path_intermediate_dir=None, base_name=None):
        center_lanes_for_viz = center_lanes.copy().astype(np.int32)
        center_lanes_for_viz = center_lanes_for_viz[::-1]

        final_output_viz = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        if final_mask is not None and final_mask.shape[0] > 0 and final_mask.shape[1] > 0:
            final_mask_resized = cv2.resize(final_mask, (resized_image.shape[1], resized_image.shape[0]))
            final_output_viz[final_mask_resized > 0] = [0, 255, 0]

        for p in center_lanes_for_viz:
            cv2.circle(final_output_viz, (p[0], p[1]), 5, (0, 0, 255), -1)

        if save_path_intermediate_dir:
            cv2.imwrite(os.path.join(save_path_intermediate_dir, f"{base_name}_hybrid_3_final_output.png"), final_output_viz)

        x_scale = original_image.shape[1] / resized_image.shape[1]
        y_scale = original_image.shape[0] / resized_image.shape[0]
        
        center_lanes_scaled = center_lanes.copy().astype(np.float32)
        center_lanes_scaled[:, 0] *= x_scale
        center_lanes_scaled[:, 1] *= y_scale
        
        center_lanes_scaled = center_lanes_scaled.astype(np.int32)
        center_lanes_scaled = center_lanes_scaled[::-1]

        return center_lanes_scaled, final_output_viz

    def process_image(self, image):
        resized_image = self._resize_image_for_model(image)
        final_mask, distance_to_center, center_lanes, _, raw_detection_image, extended_image, paths_image, points_for_extension = self.detect_lanes_yolop_v2_hybrid_agent(
            resized_image, img_path=None
        )
        center_lanes_scaled, processed_image = self._scale_and_draw_visualization(image, resized_image, final_mask, center_lanes)

        return center_lanes_scaled, processed_image, distance_to_center, raw_detection_image, paths_image

    def run_yolop_v2_inference(self, raw_image, stretch_factor=1.0):

        # --- STEP 1: CLAHE only ---
        raw_image = apply_clahe_bgr(raw_image)
        raw_image = soften_overbright_lanes(raw_image)

        h, w, _ = raw_image.shape

        new_h = int(h * stretch_factor)
        new_h = ((new_h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32

        resized_image = cv2.resize(raw_image, (new_w, new_h))

        img = self.yolop_v2_lines_transform(resized_image).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = self.yolop_v2_lines_model(img)

        return outputs[1], outputs[2], resized_image

    def detect_lanes_yolop_v2_hybrid_agent(self, image, save_path_intermediate_dir=None, img_path=None,
                                           force_drivable_fallback=False, force_classical_fallback=False,
                                           reference_point=None):
        base_name = Path(img_path).stem if img_path else "debug"
        # Step 1 & 2: Calculate both drivable area and lines first
        with torch.no_grad():
            da_seg_out, ll_seg_out, resized_image = self.run_yolop_v2_inference(image)
            ll_segment = detect_yolop_v2_lines(resized_image, force_classical_fallback,
                                               ll_seg_out_from_inference=ll_seg_out)
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