import numpy as np
import torch
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, max_size, rgb_shape, lidar_shape, radar_shape, num_of_actions) -> None:
        self.memory_size = max_size
        self.memory_pos = 0  # Most recent saved position in memory
        self.rgb_memory = np.zeros((self.memory_size, *rgb_shape), dtype=np.float32)
        self.lidar_memory = np.zeros((self.memory_size, *lidar_shape), dtype=np.float32)
        self.radar_memory = np.zeros((self.memory_size, *radar_shape), dtype=np.float32)
        self.new_rgb_memory = np.zeros((self.memory_size, *rgb_shape), dtype=np.float32)
        self.new_lidar_memory = np.zeros((self.memory_size, *lidar_shape), dtype=np.float32)
        self.new_radar_memory = np.zeros((self.memory_size, *radar_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, num_of_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store(self, rgb_state, lidar_state, radar_state, action, reward, done, new_rgb_state, new_lidar_state, new_radar_state):
        index = self.memory_pos % self.memory_size

        lidar_state = self.process_state_points(lidar_state)
        new_lidar_state = self.process_state_points(new_lidar_state)

        if radar_state.size > 0:
            radar_state = self.process_state_points(radar_state) 
        else:
            radar_state = np.zeros((256, 3))
        
        if new_radar_state.size > 0:
            new_radar_state = self.process_state_points(new_radar_state)
        else:
            new_radar_state = np.zeros((256, 3)) 


        # Store each type of state data in its respective memory array
        self.rgb_memory[index] = rgb_state
        self.lidar_memory[index] = lidar_state
        self.radar_memory[index] = radar_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)  # done is True or False
        self.new_rgb_memory[index] = new_rgb_state
        self.new_lidar_memory[index] = new_lidar_state
        self.new_radar_memory[index] = new_radar_state
        self.memory_pos += 1

    def process_state_points(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        num_points = state.shape[0]

        if num_points > 256:
            # Ensure divisible size for pooling
            if num_points % (256 * 3) != 0:
                state = state[:256 * (num_points // (256 * 3)) * 3]

            state = state.view(256, -1, 3)  # Reshape to (256, N, 3) | N = total points / 256
            state = torch.mean(state, dim=1)  # Mean Pooling
        elif num_points < 256:
            # Randomly duplicate points to reach target size
            if num_points > 0:
                indices = torch.randint(0, num_points, (256 - num_points,))
            else:
                indices = torch.randint(0, 1, (256,))  # Default fallback
            padding = state[indices]
            state = torch.cat((state, padding), dim=0)

        return state

    def sample(self, batch_size):
        max_memory = min(self.memory_pos, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        # Retrieve samples from each memory array
        rgb_states = self.rgb_memory[batch]
        lidar_states = self.lidar_memory[batch]
        radar_states = self.radar_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        new_rgb_states = self.new_rgb_memory[batch]
        new_lidar_states = self.new_lidar_memory[batch]
        new_radar_states = self.new_radar_memory[batch]

        return rgb_states, lidar_states, radar_states, new_rgb_states, new_lidar_states, new_radar_states, actions, rewards, terminal
