import numpy as np

def evaluate_agent(agent, env, eval_episodes = 3):
    total_rewards = []

    for episode in range(1, eval_episodes + 1):
        rgb_obs, lidar_obs, radar_obs = env.reset()
        done = False
        episode_reward = 0

        while not done:

            action = agent.choose_action((rgb_obs, lidar_obs, radar_obs), evaluate=True)

            new_rgb_obs, new_lidar_obs, new_radar_obs, reward, done, _ = env.step(action)

            episode_reward += reward

            rgb_obs, lidar_obs, radar_obs = new_rgb_obs, new_lidar_obs, new_radar_obs

        total_rewards.append(episode_reward)

        print(f'Evaluation Episode {episode} Reward: {episode_reward}\n')

        # Clear actors at the end of each episode
        for actor in env.actor_list:
            actor.destroy()

        env.actor_list.clear()

    avg_reward = np.mean(total_rewards)

    print(f'Total rewards from eval episodes: {total_rewards}')
    print(f"Evaluation: Average Reward over {eval_episodes} episodes: {avg_reward:.4f}\n")

    return avg_reward