import glob
import os
import sys
import numpy as np
import random
from model import DDPG_Model
from carla_environment import CarlaEnv

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

if __name__ == "__main__":
    try:
        random.seed(1)
        np.random.seed(1)

        # Create an environment instance
        env = CarlaEnv(show_preview=True)

        # Change weather manually
        # env.world.set_weather(carla.WeatherParameters.ClearNoon)

        agent = DDPG_Model(
            actor_lr=0.0001,
            critic_lr=0.001,
            rgb_dims=(3, 128, 128),
            lidar_dims=(256, 3),
            radar_dims=(256, 3),
            tau=0.001,
            gamma=0.99,
            num_of_actions=3,
            max_size=60000,
            pixel_layer_size=200,
            layer1_size=400,
            layer2_size=300,
            batch_size=64
        )

        # Load the saved model
        saved_episode = 35690
        agent.load_models(episode=saved_episode, is_best=True)
        # print(f"Model loaded from episode {saved_episode}\n")

        done = False
        rgb_obs, lidar_obs, radar_obs = env.reset()

        while not done:
            action = agent.choose_action((rgb_obs, lidar_obs, radar_obs), evaluate=True)

            new_rgb_obs, new_lidar_obs, new_radar_obs, reward, done, _ = env.step(action, training=False)

            rgb_obs, lidar_obs, radar_obs = new_rgb_obs, new_lidar_obs, new_radar_obs

            # print(f'Action taken: {action}')

        print('Simulation completed.')

        for actor in env.actor_list:
            actor.destroy()
            print('Actors Destroyed.')

        env.actor_list.clear()

    except KeyboardInterrupt:
        pass
    finally:
        print("\nDone.")
