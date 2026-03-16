import glob
import os
import sys
import numpy as np
import open3d as o3d
import cv2
import time
import math
import threading

from matplotlib import cm
from numpy import random
from torchvision import transforms
from PIL import Image

from model import DDPG_Model
from utils import ModifiedTensorBoard
from evaluation import evaluate_agent

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

sys.path.append('C:/Users/sdhm1/CARLA_0.9.15/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner # type: ignore

# Ensure the best_models folder exists
best_model_folder = 'best_models'
os.makedirs(best_model_folder, exist_ok=True)

# Initialize best_reward variable
best_reward = float('-inf')

# Initialize the Modified TensorBoard (change step count if it's not a fresh start)
os.makedirs('./tb_logs', exist_ok=True)
tensorboard = ModifiedTensorBoard(log_dir='./tb_logs', start_step=0)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
SECONDS_PER_EPISODE = 15

# Define color maps
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:, :3]

class CarlaEnv():

    def __init__(self, show_preview=False):
        super(CarlaEnv, self).__init__()
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        print('Connected to Carla Sim.')

        self.world = self.client.get_world()

        self.map = self.world.get_map() 
        self.bp_lib = self.world.get_blueprint_library()

        self.ego_bp = self.bp_lib.filter('vehicle.tesla.model3')[0]

        self.actor_list = []

        self.show_preview = show_preview
        
        self.weather_presets = {
            0: 'Default',
            1: 'ClearNoon',
            2: 'CloudyNoon',
            3: 'WetNoon',
            4: 'WetCloudyNoon',
            5: 'MidRainyNoon',
            6: 'HardRainNoon',
            7: 'SoftRainNoon',
            8: 'ClearSunset',
            9: 'CloudySunset',
            10: 'WetSunset',
            11: 'WetCloudySunset',
            12: 'MidRainSunset',
            13: 'HardRainSunset',
            14: 'SoftRainSunset'
        }

        self.weather_ids = list(self.weather_presets.keys())
        self.current_episode = 0

        self.camera_data = {
            'rgb_image': np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.uint8),
            'processed_image': np.zeros((1, 128, 128), dtype=np.float32)    
        }

        self.point_list = o3d.geometry.PointCloud()
        self.radar_list = o3d.geometry.PointCloud()
        self.lidar_data = None
        self.radar_data = None

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print('Initialized Carla Sim Enviroment.')

    def run_preview(self, env):
        try:
            while True:
                env.preview()
        except Exception as e:
            print(f"Preview window closed: {e}")
            return

    # Function to change the weather in CARLA every x episodes
    def change_weather(self, episode_interval=50):
        if self.current_episode % episode_interval == 0:
            next_weather_index = (self.current_episode // episode_interval) % len(self.weather_ids)
            new_weather_id = self.weather_ids[next_weather_index]
            new_weather_name = self.weather_presets[new_weather_id]

            # Get the WeatherParameters attribute dynamically
            new_weather_param = getattr(carla.WeatherParameters, new_weather_name)

            # Set the weather in CARLA
            self.world.set_weather(new_weather_param)

            print(f'Weather set to {new_weather_name} for Episode {self.current_episode}')

    # Function to generate a route 100 meters away and visualize it
    def generate_route(self, distance_away=100, visualize=True):
        curr_wp = self.start_wp
        distance_covered = 0

        while curr_wp and (distance_covered < distance_away):
            if visualize:
               self.world.debug.draw_string(curr_wp.transform.location, "*", 
                                            draw_shadow=False, 
                                            color=carla.Color(r=0, g=255, b=0), 
                                            life_time=15.0, 
                                            persistent_lines=True)
            
            # Get the new waypoint 2 meters ahead
            next_wps = curr_wp.next(2.0)

            if not next_wps:
                break

            curr_wp = next_wps[0]

            # Append it to the list
            self.gen_route_wps_list.append(curr_wp)

            distance_covered += 2.0

        return curr_wp

    # Function to define a 3D coordinate axis for Open3D visualizations
    def add_open3d_axis(self, vis):
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]))

        axis.lines = o3d.utility.Vector2iVector(np.array([
            [0, 1],
            [0, 2],
            [0, 3]
        ]))

        axis.colors = o3d.utility.Vector3dVector(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]))

        vis.add_geometry(axis)

    def rgb_callback(self, image, camera_data):
        i = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        # Remove alpha channel
        camera_data['rgb_image'] = i[:, :, :3]

        p_i = self.preprocess(camera_data['rgb_image'])
        camera_data['processed_image'] = p_i

    # For LIDAR Sensor
    def lidar_callback(self, point_cloud, point_list):
        # Reshape the array according to the numbver of points
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4)) # 4 columns => x, y, z, i

        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])
        ]

        # Extract coordinates and form a Point Cloud
        points = data[:, :-1]

        # Reverse y-axis (different definition of it in UE4 and Carla Sim)
        points[:, :1] = -points[:, :1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        # Update self.lidar_data with the latest processed LIDAR points
        self.lidar_data = np.array(point_list.points)

    # For RADAR Sensor
    def radar_callback(self, data, point_list):
        radar_data = np.zeros((len(data), 4))
        
        for i, detection in enumerate(data):
            x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
            y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
            z = detection.depth * math.sin(detection.altitude)
            
            radar_data[i, :] = [x, y, z, detection.velocity]

        # Scale velocity into a colormap 
        intensity = np.abs(radar_data[:, -1])
        intensity_col = 1.0 - np.log(intensity)/np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 2])
        ]
        
        points = radar_data[:, :-1]
        points[:, :1] = -points[:, :1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        # Update self.radar_data with the latest processed RADAR points
        self.radar_data = np.array(point_list.points)

    # For Collision Sensor
    def collision_data(self, event):
        self.collision_list.append(event)

    # For Obstacle Detector
    def obstacle_detector_callback(self, event):
        self.obstacle_distance = event.distance 

    def reset(self):
        self.actor_list = []
        self.collision_list = []
        self.obstacle_distance = 100.0
        self.total_distance_travelled = 0.0

        self.preview_running = False

        # Get all spawn points from the map
        spawn_points = self.map.get_spawn_points()
        # Spawn points ids to use
        spawn_points_ids = [117, 90, 106 , 86, 2, 3, 141, 20, 103, 106, 78, 
                            15, 16, 24, 25, 129, 130, 124, 37, 58, 59, 62, 63,
                            92, 127, 72, 74, 143, 144, 139, 138, 13, 14, 75,
                            125, 0, 1, 113] 

        # Loop until the vehicle is successfully spawned
        while True:
            selected_wp = random.choice(spawn_points_ids)
            self.transform = spawn_points[selected_wp]

            try:
                # Try spawning the ego vehicle
                self.vehicle = self.world.spawn_actor(self.ego_bp, self.transform)
                self.actor_list.append(self.vehicle)
                print(f'Vehicle successfully spawned at: {self.transform.location}')
                # Initilize spawn point
                self.spawn_point = self.transform.location
                # Initialize start waypoint
                self.start_wp = self.map.get_waypoint(self.spawn_point, 
                                                 project_to_road=True, 
                                                 lane_type=carla.LaneType.Driving)
                # Initialize last waypoint location as the spawn point location
                self.last_wp_location = self.transform.location
                # Initialize list with waypoints with the start waypoint
                self.gen_route_wps_list = [self.start_wp]
                break
            except RuntimeError:
                print('Spawn failed due to collision. Searching for a new spawn point...')

        if self.show_preview:
            # Start the preview thread
            preview_thread = threading.Thread(target=self.run_preview, args=(self,), daemon=True)
            preview_thread.start()

        # Get the destination waypoint
        destination_wp = self.generate_route()

        if destination_wp:
            self.destination_location = destination_wp.transform.location
            print(f'Destination Location: {self.destination_location}')
        else:
            print('No valid destination found!')

        # Transforms
        sensor_init_trans = carla.Transform(carla.Location(z=1.6, x=0.4))

        # RGB Camera Sensor
        rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')

        rgb_camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
        rgb_camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        rgb_camera_bp.set_attribute('fov', '110')

        self.rgb_camera = self.world.spawn_actor(rgb_camera_bp, sensor_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_camera)
        self.rgb_camera.listen(lambda image: self.rgb_callback(image, self.camera_data))

        # LIDAR Sensor
        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('noise_stddev', '0.1')
        lidar_bp.set_attribute('upper_fov', '15')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('points_per_second', '500000')

        self.lidar_sensor = self.world.spawn_actor(lidar_bp, sensor_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.lidar_sensor)
        self.lidar_sensor.listen(lambda data: self.lidar_callback(data, self.point_list))

        # RADAR Sensor
        radar_bp = self.bp_lib.find('sensor.other.radar')

        radar_bp.set_attribute('horizontal_fov', '30.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '10000')

        self.radar_sensor = self.world.spawn_actor(radar_bp, sensor_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.radar_sensor)
        self.radar_sensor.listen(lambda data: self.radar_callback(data, self.radar_list))

        # Set throttle and break to 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # In some cases the sensor detects a collision when the vehicle spawns
        time.sleep(5)

        # Collision Sensor
        collision_sensor_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, sensor_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # Obstacle Detector
        obstacle_detector_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')

        obstacle_detector_bp.set_attribute('distance', '10.0')

        self.obstacle_detector = self.world.spawn_actor(obstacle_detector_bp, sensor_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle_detector)
        self.obstacle_detector.listen(lambda event: self.obstacle_detector_callback(event))

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        # Return the initial observations
        rgb_obs = self.camera_data['processed_image']  # Initial RGB image
        lidar_obs = self.lidar_data  # Initial LIDAR points
        radar_obs = self.radar_data  # Initial RADAR points

        return rgb_obs, lidar_obs, radar_obs

    def step(self, action, training=True):
        # Initialize base reward and done flag
        reward = 0
        done = False

        # Extract throttle, brake, and steering values from the action
        throttle_value = float(action[0])  
        brake_value = float(action[1])     
        steer_value = float(action[2])    

        # Ignore very small brake values
        if brake_value < 0.1:
            brake_value = 0

        # Check if throttle or brake value is bigger and do that as an action
        if throttle_value > brake_value:
            brake_value = 0.0
        else:
            throttle_value = 0.0

        # Get vehicle velocity and compute speed in km/h
        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        kmh = int(3.6 * speed)

        print(f'Vehicle Speed (km/h): {kmh}')
        print(f'Throttle Value: {throttle_value} | Brake Value: {brake_value} | Steer Value: {steer_value}')

        # Apply the control to the vehicle
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_value, brake=brake_value, steer=steer_value))
        
        if training:
            # Get the closest waypoint from the generated route
            waypoint = min(self.gen_route_wps_list, key=lambda wp: wp.transform.location.distance(self.vehicle.get_location()))

            # Get the traffic light object
            traffic_light = self.vehicle.get_traffic_light()

            # Fetch distance
            obstacle_distance = getattr(self, "obstacle_distance", 100.0)

            # Calculate angle
            vehicle_yaw = self.vehicle.get_transform().rotation.yaw
            waypoint_yaw = waypoint.transform.rotation.yaw

            # Normalize yaw difference to [-180, 180]
            yaw_diff = vehicle_yaw - waypoint_yaw
            yaw_diff = (yaw_diff + 180) % 360 - 180

            # Calculate position on the road
            vehicle_location = self.vehicle.get_transform().location
            waypoint_location = waypoint.transform.location

            distance_to_wp = vehicle_location.distance(waypoint_location)

            # Calculate the distance between the last waypoint and the current waypoint
            segment_distance = self.last_wp_location.distance(waypoint_location)

            if segment_distance > 0.1:  # Threshold to avoid small floating point errors
                # Add the segment distance to the total travelled distance if the vehicle has moved
                self.total_distance_travelled += segment_distance
                # Update the location of the previous waypoint with the current one
                self.last_wp_location = waypoint_location

            # Calculate factors
            angle_factor = max(1.0 - (abs(yaw_diff) / 60), 0.0)
            position_factor = max(1.0 - (distance_to_wp / 8), 0.0)

            print(f'Total Distance Travelled: {self.total_distance_travelled}')
            print(f'Angle factor: {angle_factor} | Pos factor: {position_factor}')

            # End Conditions
            # Check if the vehicle has reached the destination
            if self.total_distance_travelled >= 100.0:
                done = True
                reward = 20
                print(f'Destination Reached! Ending Episode!')
            else:
                # Collision penalty
                if len(self.collision_list) != 0:
                    done = True  # End the episode
                    reward = -1
                    print(f'Collision Detected! Ending Episode. Reward: {reward}')
                
                if abs(yaw_diff) > 90:
                    done = True
                    reward = -1
                    print(f'Wrong Direction. Yaw Diff: {yaw_diff}. Ending Episode. Reward: {reward}')
                
                if distance_to_wp > 10:
                    done = True
                    reward = -1
                    print(f'WayPoint distance: {distance_to_wp}. Ending Episode. Reward: {reward}') 

                if kmh > 50:
                    done = True
                    reward = -1
                    print(f'Speeding! Ending Episode. Reward: {reward}')          

            if not done:
                # Traffic light handling
                if traffic_light and self.vehicle.is_at_traffic_light() and traffic_light.get_state() == carla.TrafficLightState.Red:
                    if brake_value > 0:
                        reward += brake_value
                        print(f'Braking at Red Light. Reward Applied. Reward: {reward}')
                        if kmh == 0:
                            print('Stopped at Red Light!')
                    else:
                        reward -= min((kmh / 50), 1) # Max penalty: -1 | Penalty scaled based on vehicle's speed
                        print(f'Red Light Violation! Penalty Applied. Reward: {reward}')
                else:
                    if obstacle_distance > 10:
                        if (kmh == 0 and throttle_value == 0) or (throttle_value == 0 and brake_value == 0 and kmh == 0):
                            reward -= 1
                            print(f'Stationary Vehicle. Penalty Applied. Reward: {reward}')
                        elif throttle_value > 0 and kmh < 1:
                            reward += angle_factor * 0.1
                            # print(f'Angle Reward Applied. Reward: {reward}')

                            reward += position_factor * 0.1
                            # print(f'Position Reward Applied. Reward: {reward}')

                            reward += position_factor * angle_factor * throttle_value * 0.2
                            # print(f'Throttle Reward Applied. Reward: {reward}')
                        elif kmh >= 1:
                            reward += angle_factor * 0.25
                            # print(f'Angle Reward Applied. Reward: {reward}')

                            reward += position_factor * 0.25
                            # print(f'Position Reward Applied. Reward: {reward}')
                        
                            if kmh < 20:
                                reward += position_factor * angle_factor * (kmh / 20) * 0.5
                                # print(f'Trajectory Reward Applied. Reward: {reward}')
                            elif kmh <= 30:
                                reward += position_factor * angle_factor * 0.5
                                # print(f'Max Trajectory Reward Applied. Reward: {reward}')

        # Check if the episode has ended based on time
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            print(f'Time Limit Reached. Ending Episode.')

        # Get the latest observations
        rgb_obs = self.camera_data['processed_image']  # RGB image from the camera
        lidar_obs = self.lidar_data  # Processed LiDAR points from callback
        radar_obs = self.radar_data  # Processed radar points from callback

        # Validate observations
        if rgb_obs is None or lidar_obs is None or radar_obs is None:
            print('Warning: Missing Observations!')
            done = True  # End the episode if observations are invalid

        print(f'\nStep Results - Reward: {reward}, Done: {done}\n')

        return rgb_obs, lidar_obs, radar_obs, reward, done, None

    def preview(self, duration=SECONDS_PER_EPISODE):
        # Check if a preview window is already open
        if hasattr(self, 'preview_running') and self.preview_running:
            return

        self.preview_running = True  # Flag to indicate the preview is active

        # Initialize visualization for lidar and radar sensors
        vis = o3d.visualization.Visualizer()
        end_time = time.time() + duration
        frame = 0

        try:
            while time.time() < end_time:
                # Update the environment
                self.world.tick()

                # Initialize visualization for lidar and radar sensors
                if frame == 0:
                    vis.create_window(
                        window_name='Lidar and Radar',
                        width=960,
                        height=540,
                        left=480,
                        top=270
                    )
                    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
                    vis.get_render_option().point_size = 1
                    vis.get_render_option().show_coordinate_frame = True
                    self.add_open3d_axis(vis)
                
                # Capture RGB camera image
                rgb_image = self.camera_data['rgb_image']

                cv2.imshow('RGB Camera', rgb_image)

                # For lidar and radar point clouds
                if frame == 2:
                    vis.add_geometry(self.point_list)
                    vis.add_geometry(self.radar_list)
                vis.update_geometry(self.point_list)
                vis.update_geometry(self.radar_list)

                # Render Open3D scene
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.005)
                frame += 1

                # Close window by pressing 'q'
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            # Stop cameras and destroy all windows
            cv2.destroyAllWindows()
            vis.destroy_window()
            print('Preview Windows Destroyed.')

            self.preview_running = False  # Reset the flag

if __name__ == "__main__":
    try:
        random.seed(1)
        np.random.seed(1)

        scores = []

        # Create an environment instance
        env = CarlaEnv(show_preview=False)

        agent = DDPG_Model(
            actor_lr=0.0001,
            critic_lr=0.001,
            rgb_dims=(3, 128, 128),
            lidar_dims=(256, 3),
            radar_dims=(256, 3),
            tau=0.001,
            gamma=0.99,
            num_of_actions=3,
            max_size=70000,
            pixel_layer_size=200,
            layer1_size=400,
            layer2_size=300,
            batch_size=64
        )

        # Load model
        # loaded_episode = X
        # agent.load_models(episode=loaded_episode, is_best=False)

        for episode in range(1, 50001):

            done = False
            episode_reward = 0
            env.current_episode = episode
            env.change_weather()  # Change weather every 50 episodes
            env.collision_list = []
            env.gen_route_wps_list = []
            rgb_obs, lidar_obs, radar_obs = env.reset()

            while not done:
                action = agent.choose_action((rgb_obs, lidar_obs, radar_obs))

                # Execute the action in the CARLA Enviroment
                new_rgb_obs, new_lidar_obs, new_radar_obs, reward, done, _ = env.step(action)

                # Store transition in the Replay Buffer
                agent.memory.store(rgb_obs, lidar_obs, radar_obs, action, reward, done, new_rgb_obs, new_lidar_obs, new_radar_obs)

                # Train the agent
                agent.train()

                # Update Score 
                episode_reward += reward

                # Update current state
                rgb_obs, lidar_obs, radar_obs = new_rgb_obs, new_lidar_obs, new_radar_obs

            # Store episode score to scores
            scores.append(episode_reward)

            # Log metrics after the episode
            tensorboard.log_scalar('Training Episode Reward', episode_reward, step=episode)
            tensorboard.log_scalar('Actor Loss', agent.actor_loss, step=episode)
            tensorboard.log_scalar('Critic Loss', agent.critic_loss, step=episode)

            print(f'Episode {episode} Reward: {episode_reward}\n')

            #Save model every x episodes
            if episode % 50 == 0:
                agent.save_models(episode)
                print(f'Model saved at episode {episode}')

            # Clear actors at the end of each episode
            for actor in env.actor_list:
                actor.destroy()

            env.actor_list.clear()

            # Evaluation every x episodes
            if episode % 10 == 0:
                print('Evaluating Agent...')
                avg_reward = evaluate_agent(agent, env)

                tensorboard.log_scalar('Average Evaluation Reward', avg_reward, step=episode)

                # Check if the current evaluation reward is the best
                if avg_reward > best_reward:
                    best_reward = avg_reward

                    # Save the new best model
                    agent.save_models(episode=episode, is_best=True)

                    print(f'New best model saved!\n')             

        # Close TensorBoard
        tensorboard.close()

    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')