import carla
import pygame
import numpy as np
import time

from camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    check_inside_image,
    create_lane_lines,
    get_matrix_global,
    CameraGeometry,
)

def carla_image_to_pygame(carla_image):
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    return pygame.surfarray.make_surface(array[:, :, :3].swapaxes(0, 1))


def initialize_pygame(display_width, display_height):
    pygame.init()
    display = pygame.display.set_mode((display_width, display_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    return display



# Main
client = carla.Client('localhost', 3005)
client.set_timeout(10.0)
client.load_world('Town04')
world = client.get_world()


# TM
tm = client.get_trafficmanager(3010)
port_tm = tm.get_port()
print(port_tm)
tm.set_global_distance_to_leading_vehicle(2.0)
tm.global_percentage_speed_difference(-30.0)      

# Sync mode
settings = world.get_settings()
settings.synchronous_mode = True  
settings.fixed_delta_seconds = 0.1
world.apply_settings(settings)


blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle')[0]
spawn_point = world.get_map().get_spawn_points()[0]

# This is specific config for Town4 to start in center lane
spawn_point.location.y = -371

vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True, tm.get_port())

# Set camera and config
width=800
heigth=600
fov=110

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(width))
camera_bp.set_attribute('image_size_y', str(heigth))
camera_bp.set_attribute('fov', str(fov))

camera_transform = carla.Transform(carla.Location(x=0.14852, y=0.0, z=1.7292), 
                                   carla.Rotation(pitch=-3.248, yaw=-0.982, roll=0.0))  

# Translation matrix, convert vehicle reference system to camera reference system

trafo_matrix_vehicle_to_cam = np.array(
    camera_transform.get_inverse_matrix()
)

# Intrinsic matrix, transform camera space (3D) to image space (2D) 
K = get_intrinsic_matrix(fov, width, heigth)


## Config needed to set the TM properly for this scenario
# Spawn secundary vehicle (forward)
waypoint_ahead = spawn_point
waypoint_ahead.location.x = waypoint_ahead.location.x + 5
vehicle_ahead_bp = blueprint_library.filter('vehicle')[0]
vehicle_ahead = world.spawn_actor(vehicle_ahead_bp, waypoint_ahead)
vehicle_ahead.set_autopilot(True, tm.get_port())


# TM config for main vehicle
tm.vehicle_percentage_speed_difference(vehicle_ahead, 10) 
tm.vehicle_percentage_speed_difference(vehicle, 0) 
tm.ignore_lights_percentage(vehicle, 0)  
tm.ignore_signs_percentage(vehicle, 0)   
tm.auto_lane_change(vehicle, False)      

# This does not work properly!
tm.distance_to_leading_vehicle(vehicle, 0.2) 

tm.ignore_lights_percentage(vehicle, 100)
tm.ignore_signs_percentage(vehicle, 100)

# camera
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

display = initialize_pygame(width, heigth)


camera_image = None
def process_image(image):
    global camera_image
    camera_image = carla_image_to_pygame(image)


camera.listen(process_image)

# Sync mode
try:
    while True:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        
        world.tick()
        
        if camera_image is not None:

            display.blit(camera_image, (0, 0))

            trafo_matrix_global_to_camera = get_matrix_global (vehicle, trafo_matrix_vehicle_to_cam)

            waypoint = world.get_map().get_waypoint(
                vehicle.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,                
            )
            
            center_list, left_boundary, right_boundary, type_lane = create_lane_lines(waypoint, vehicle)

            projected_left_boundary = project_polyline(
                left_boundary, trafo_matrix_global_to_camera, K).astype(np.int32)
            projected_right_boundary = project_polyline(
                right_boundary, trafo_matrix_global_to_camera, K).astype(np.int32)
           
            if (not check_inside_image(projected_right_boundary, width, heigth)
                or  not check_inside_image( projected_right_boundary, width, heigth )):
                continue
           
            if len(projected_left_boundary) > 1:
                pygame.draw.lines(
                    display, (255, 0, 0), False, projected_left_boundary, 4
                )
            if len(projected_right_boundary) > 1:
                pygame.draw.lines(
                    display,
                    (0, 255, 0),
                    False,
                    projected_right_boundary,
                    4,
                )                   

            pygame.display.flip()
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Saliendo...")

finally:
    
    camera.stop()
    vehicle.destroy()
    vehicle_ahead.destroy()
    pygame.quit()
        
    world.apply_settings(settings)
