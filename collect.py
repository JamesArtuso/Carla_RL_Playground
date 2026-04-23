#!/usr/bin/env python
import gym
import gym_carla
import carla
import numpy as np
import cv2
import os

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 200,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  obs = env.reset()
  
  
  os.makedirs("figures/cam", exist_ok=True)
  os.makedirs("figures/bird", exist_ok=True)
  num_pics = 10000
  idx = 107
  
  
  while idx < num_pics:
    action = [2.0, 1*(np.random.rand(1))-0.5]
    obs,r,done,info = env.step(action)
    cam = obs['camera']
    bird = obs['birdeye']
    
    cam_bgr = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
    bird_bgr = cv2.cvtColor(bird, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f"figures/cam/{idx:06d}.png", cam_bgr)
    cv2.imwrite(f"figures/bird/{idx:06d}.png", bird_bgr)
    
    idx += 1
    
    if done:
      obs = env.reset()


if __name__ == '__main__':
  main()
