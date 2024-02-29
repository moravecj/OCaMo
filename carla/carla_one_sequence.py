import carla
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
import sys

random.seed(42)

outp_dir = sys.argv[2]

# initialize the world in synchronous mode and 10Hz
client = carla.Client('localhost', 2000)
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
world.apply_settings(settings)
# initialize trafic manager for car autopilots
traffic_manager = client.get_trafficmanager()
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
traffic_manager.set_respawn_dormant_vehicles(True)
traffic_manager.set_synchronous_mode(True)
# get the selected spawn point
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
spawn_point = int(sys.argv[1])

os.makedirs('{}/data_{}'.format(outp_dir, str(spawn_point).zfill(3)))
# spawn the vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[spawn_point])
# spawn another 100 random agents in the world
for i in range(100):
    vehicle_bp =random.choice(bp_lib.filter('vehicle'))
    npc_veh = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if npc_veh != None:
        npc_veh.set_autopilot(True)

time.sleep(1)
# Add 1080p RGB camera to the vehicle
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x",str(1920))
camera_bp.set_attribute("image_size_y",str(1080))
camera_bp.set_attribute("fov",str(60))
camera_init_trans = carla.Transform(carla.Location(z = 2,x = 0, y=-0.1))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to = vehicle)
# Add 3D rotational LiDAR with 64 channels to the vehicle
lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
lidar_bp.set_attribute('dropoff_general_rate', '0.0')
lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
lidar_bp.set_attribute('upper_fov', str(2.0))
lidar_bp.set_attribute('lower_fov', str(-24.9))
lidar_bp.set_attribute('channels', str(64))
lidar_bp.set_attribute('range', str(120.0))
lidar_bp.set_attribute('points_per_second', str(2880000))
lidar_init_trans = carla.Transform(carla.Location(z = 2,x = 0, y = 0.1))
lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)
# camera new frame callback
number_of_frames_camera = 0
def camera_callback(image):
    global number_of_frames_camera
    number_of_frames_camera += 1
    if number_of_frames_camera > 10:
        plt.imsave('{}/data_{}/{}.jpg'.format(outp_dir, str(spawn_point).zfill(3), str(number_of_frames_camera - 10).zfill(3)), np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[:,:,:3][:,:,::-1])

camera.listen(lambda image: camera_callback(image))
# LiDAR new frame callback
number_of_frames_lidar = 0
def lidar_callback(point_cloud):
    global number_of_frames_lidar
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    data[:, 1] = -data[:, 1]
    number_of_frames_lidar += 1
    if number_of_frames_lidar > 10:
        savemat('{}/data_{}/{}.mat'.format(outp_dir, str(spawn_point).zfill(3), str(number_of_frames_lidar - 10).zfill(3)), {'points': data})

lidar.listen(lambda data: lidar_callback(data))
# Main loop to acquire 210 frames. The first 10 frames are there only for the initialization, not saved.
while True:
    world.tick()
    vehicle.set_autopilot(True)
    transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
    if number_of_frames_camera >= 209 and number_of_frames_lidar >= 209:
        time.sleep(10) # sensor callbacks might be still running
        break