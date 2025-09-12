import carla
import math

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.load_world("Town06")
map = world.get_map()

# Starting location
location = carla.Transform(
    carla.Location(x=180.999569, y=303.000885, z=1),
    carla.Rotation(pitch=0.06, yaw=0.999954, roll=-0.006836)
)

# Get closest waypoint
start_wp = map.get_waypoint(location.location, project_to_road=True, lane_type=carla.LaneType.Driving)

# Threshold distance to detect loops
DIST_THRESHOLD = 0.5  # meters

# Keep track of visited positions
visited_positions = []

def is_visited(new_wp, visited, threshold=DIST_THRESHOLD):
    for loc in visited:
        dx = loc.x - new_wp.transform.location.x
        dy = loc.y - new_wp.transform.location.y
        dz = loc.z - new_wp.transform.location.z
        if math.sqrt(dx*dx + dy*dy + dz*dz) < threshold:
            return True
    return False

# Traverse along the lane
ordered_waypoints = [start_wp]
visited_positions.append(start_wp.transform.location)
current_wp = start_wp

num_visited = 0
while True:
    next_wps = current_wp.next(1.0)  # 1 meter ahead
    if not next_wps:
        break

    next_wp = next_wps[0]

    if is_visited(next_wp, visited_positions):
        num_visited += 1
        if num_visited > 3:
            break

    ordered_waypoints.append(next_wp)
    visited_positions.append(next_wp.transform.location)
    current_wp = next_wp

print(f"Total waypoints collected along the lane: {len(ordered_waypoints)}")

# Compute cumulative distance along waypoints
cum_dist = [0.0]
for i in range(1, len(ordered_waypoints)):
    loc_prev = ordered_waypoints[i-1].transform.location
    loc_curr = ordered_waypoints[i].transform.location
    dx = loc_curr.x - loc_prev.x
    dy = loc_curr.y - loc_prev.y
    dz = loc_curr.z - loc_prev.z
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    cum_dist.append(cum_dist[-1] + dist)

total_distance = cum_dist[-1]
num_sectors = 4
sector_length = total_distance / num_sectors

# Define 4 distinct colors
sector_colors = [
    carla.Color(255, 0, 0),    # Red
    carla.Color(0, 255, 0),    # Green
    carla.Color(0, 0, 255),    # Blue
    carla.Color(255, 255, 0),  # Yellow
]

# Draw waypoints colored by sector and create log list
waypoints_log = []

for i, wp in enumerate(ordered_waypoints):
    distance = cum_dist[i]
    sector_id = min(int(distance // sector_length), num_sectors - 1)
    color = sector_colors[sector_id]

    world.debug.draw_point(
        wp.transform.location + carla.Location(z=0.5),
        size=0.05,
        color=color,
        life_time=0.0,
        persistent_lines=True
    )

    # Log as dict with location + rotation
    t = wp.transform
    waypoints_log.append({
        'location': {'x': t.location.x, 'y': t.location.y, 'z': t.location.z},
        'rotation': {'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw, 'roll': t.rotation.roll}
    })

print("Waypoints list:")
print(waypoints_log[::20])

# Prepare index-to-transform mapping
idx_to_transform = {i: wp.transform for i, wp in enumerate(ordered_waypoints)}

spectator = world.get_spectator()

while True:
    try:
        user_input = input("Enter a waypoint index (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

        idx = int(user_input)
        if idx not in idx_to_transform:
            print(f"Index {idx} is out of range! Max index is {len(ordered_waypoints)-1}")
            continue

        target_transform = idx_to_transform[idx]

        # Move spectator above the waypoint
        spectator_location = carla.Transform(
            target_transform.location + carla.Location(z=100),
            carla.Rotation(pitch=-90)
        )
        spectator.set_transform(spectator_location)

        # Determine sector color
        distance = cum_dist[idx]
        sector_id = min(int(distance // sector_length), num_sectors - 1)
        color = sector_colors[sector_id]

        print(f"Moved spectator to index {idx} at {target_transform.location}")
        print(f"Waypoint belongs to sector {sector_id} with color {color}")

    except ValueError:
        print("Please enter a valid integer index or 'q' to quit.")
    except Exception as e:
        print(f"Error: {e}")