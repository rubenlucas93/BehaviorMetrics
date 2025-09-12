#
# Refactor and reused code from https://github.com/thomasfermi/Algorithms-for-Automated-Driving/
#


import numpy as np

def get_intrinsic_matrix(field_of_view_deg, image_width, image_height):
    # For our Carla camera alpha_u = alpha_v = alpha
    # alpha can be computed given the cameras field of view via
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu],
                     [0, alpha, Cv],
                     [0, 0, 1.0]])


def project_polyline(points_3d, trafo, K, image_shape=None):
    # Step 1: Homogeneous coordinates
    points_3d_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)  # Nx4

    # Step 2: Transform to camera space
    points_cam = (trafo @ points_3d_h.T).T  # Nx4

    # Step 3: Filter points behind the camera
    valid_z = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_z]

    # Step 4: Project to image plane
    points_2d = (K @ points_cam[:, :3].T).T
    points_2d = points_2d[:, :2] / points_cam[:, 2:3]  # Normalize x/z and y/z

    # Step 5 (NEW): Remove out-of-bounds pixels
    # if image_shape is not None:
    #     h, w = image_shape
    #     x, y = points_2d[:, 0], points_2d[:, 1]
    #     in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    #     points_2d = points_2d[in_bounds]

    return points_2d


def check_inside_image(pixel_array, width, height):
    ok = (0 < pixel_array[:, 0]) & (pixel_array[:, 0] < width)
    ok = ok & (0 < pixel_array[:, 1]) & (pixel_array[:, 1] < height)
    ratio = np.sum(ok) / len(pixel_array)
    return ratio > 0.5


def carla_vec_to_np_array(vec):
    return np.array([vec.x,
                     vec.y,
                     vec.z])


def get_matrix_global(vehicle, trafo_matrix_vehicle_to_cam, opposite=False):
    # draw lane boundaries as augmented reality
    trafo_matrix_world_to_vehicle = np.array(
        vehicle.get_transform().get_inverse_matrix()
    )
    trafo_matrix_global_to_camera = (
        trafo_matrix_vehicle_to_cam @ trafo_matrix_world_to_vehicle
    )
    if opposite:
        mat_swap_axes = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
    else:
        mat_swap_axes = np.array(
            [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )

    trafo_matrix_global_to_camera = (
        mat_swap_axes @ trafo_matrix_global_to_camera
    )

    return trafo_matrix_global_to_camera


def get_next_waypoint(opposite, waypoint, meters):
    if opposite:
        next_waypoints = waypoint.previous(meters)
    else:
        next_waypoints = waypoint.next(meters)
    if len(next_waypoints)>1:
        return [next_waypoints[0]]
    else:
        return next_waypoints


def create_lane_lines(waypoint, opposite=False, exclude_junctions=True, only_turns=True):

    center_list, left_boundary, right_boundary = [], [], []

    # This number represents how high we go in the line detection
    N = 100

    for i in range(N):

        if (
            str(waypoint.right_lane_marking.type)
            + str(waypoint.left_lane_marking.type)
        ).find("NONE") != -1:
            pass
            #print("None waypoints")
            #return None, None, None, None
        # if there is a junction on the path, return None
        if exclude_junctions and waypoint.is_junction:
            pass
            #print("junction on the path")
            #return None, None, None, None
        if opposite:
            next_waypoints = waypoint.previous(1.0)
        else:
            next_waypoints = waypoint.next(1.0)
            # if there is a branch on the path, return None

        if len(next_waypoints) != 1:
            pass
            #print("Branch on the path")
            #return None, None, None, None

        waypoint = next_waypoints[0]
        center = carla_vec_to_np_array(waypoint.transform.location)
        center_list.append(center)
        # Hack to avoid lines on top of the front part (uc3m-atlas)
        #if (i<2):
        #    continue
        offset = (
            carla_vec_to_np_array(waypoint.transform.get_right_vector())
            * waypoint.lane_width
            / 2.0
        )
        left_boundary.append(center - offset)
        right_boundary.append(center + offset)

    return (
        np.array(center_list),
        np.array(left_boundary),
        np.array(right_boundary),
        waypoint
    )

class CameraGeometry(object):
    def __init__(self, height=1.3, yaw_deg=0, pitch_deg=-5, roll_deg=0, image_width=1024, image_height=512, field_of_view_deg=45):
        # scalar constants
        self.height = height
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.yaw_deg = yaw_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg
        # camera intriniscs and extrinsics
        self.intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, image_width, image_height)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                            [cp*sr, cp*cr, sp],
                                            [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
        self.rotation_cam_to_road = rotation_road_to_cam.T # for rotation matrices, taking the transpose is the same as inversion
        self.translation_cam_to_road = np.array([0,-self.height,0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3,0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3,3] = self.translation_cam_to_road
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])


    def camframe_to_roadframe(self,vec_in_cam_frame):
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_camframe(self,u,v):
        # NOTE: The results depend very much on the pitch angle (0.5 degree error yields bad result)
        # Here is a paper on vehicle pitch estimation:
        # https://refubium.fu-berlin.de/handle/fub188/26792
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height*Kinv_uv_hom/denominator
    
    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y]) # read book section on coordinate systems to understand this

    def precompute_grid(self,dist=60):
        cut_v = int(self.compute_minimum_v(dist=dist)+1)
        xy = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                X,Y,Z= self.uv_to_roadXYZ_roadframe_iso8855(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):
        """
        Find cut_v such that pixels with v<cut_v are irrelevant for polynomial fitting.
        Everything that is further than `dist` along the road is considered irrelevant.
        """        
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v