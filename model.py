import numpy as np
import torch
import torch.nn.functional as F

from noise import OUActionNoise
from memory import ReplayBuffer
from networks import ActorNetwork, CriticNetWork

class DDPG_Model:
    def __init__(self, actor_lr, critic_lr, rgb_dims, lidar_dims, radar_dims, tau, gamma=0.99, num_of_actions=3, max_size=1000000, pixel_layer_size=200, layer1_size=400, layer2_size=300, batch_size=64) -> None:
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, rgb_dims, lidar_dims, radar_dims, num_of_actions)
        self.batch_size = batch_size
        self.min_action = np.array([0.0, 0.0, -1.0])  # min [throttle, brake, steering]
        self.max_action = np.array([1.0, 1.0, 1.0])   # max [throttle, brake, steering]
        self.actor_loss = 0.0
        self.critic_loss = 0.0
        
        self.actor = ActorNetwork(actor_lr, rgb_dims, lidar_dims, radar_dims, pixel_layer_size, layer1_size, layer2_size, num_of_actions, 'actor')
        self.target_actor = ActorNetwork(actor_lr, rgb_dims, lidar_dims, radar_dims, pixel_layer_size, layer1_size, layer2_size, num_of_actions, 'target_actor')

        self.critic = CriticNetWork(critic_lr, rgb_dims, lidar_dims, radar_dims, pixel_layer_size, layer1_size, layer2_size, num_of_actions, 'critic')
        self.target_critic = CriticNetWork(critic_lr, rgb_dims, lidar_dims, radar_dims, pixel_layer_size, layer1_size, layer2_size, num_of_actions, 'target_critic')

        self.noise = OUActionNoise(mu=np.zeros(num_of_actions))

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation, evaluate=False):
        self.actor.eval()

        # Unpack observation
        rgb_obs, lidar_obs, radar_obs = observation

        # Convert each observation component to tensors
        rgb_obs = torch.tensor(rgb_obs, dtype=torch.float).unsqueeze(0)
        lidar_obs = torch.tensor(lidar_obs, dtype=torch.float).unsqueeze(0)
        radar_obs = torch.tensor(radar_obs, dtype=torch.float).unsqueeze(0)
        
        # Get the raw action output (mu) from the actor network
        mu = self.actor(rgb_obs, lidar_obs, radar_obs)

        # Scale throttle and brake from [-1, 1] to [0, 1]
        mu[:, 0] = (mu[:, 0] + 1) / 2  # Scale throttle
        mu[:, 1] = (mu[:, 1] + 1) / 2  # Scale brake

        # Only add noise during training
        if not evaluate:
            # Add noise for exploration
            noise = torch.tensor(self.noise(), dtype=torch.float)
            mu_prime = mu + noise
        else:
            mu_prime = mu

        # Clip the action within the valid range [min_action, max_action]
        min_action = torch.tensor(self.min_action, dtype=torch.float32)
        max_action = torch.tensor(self.max_action, dtype=torch.float32)
        mu_prime = torch.max(torch.min(mu_prime, max_action), min_action)

        # Set actor back to train mode
        self.actor.train()

        # Return final action as a numpy array
        return mu_prime.cpu().detach().numpy().squeeze()

    # Function to store state transitions
    def store_state(self, rgb_state, lidar_state, radar_state, new_rgb_state, new_lidar_state, new_radar_state, action, reward, done):
        self.memory.store(rgb_state, lidar_state, radar_state, new_rgb_state, new_lidar_state, new_radar_state, action, reward, done)

    def train(self):
        # Don't start learning until memory is filled
        if self.memory.memory_pos < self.batch_size:
            return
        
        rgb_states, lidar_states, radar_states, new_rgb_states, new_lidar_states, new_radar_states, actions, rewards,  done = self.memory.sample(self.batch_size)
        
        # Convert them to tensors
        rgb_states = torch.tensor(rgb_states, dtype=torch.float)
        lidar_states = torch.tensor(lidar_states, dtype=torch.float)
        radar_states = torch.tensor(radar_states, dtype=torch.float)
        new_rgb_states = torch.tensor(new_rgb_states, dtype=torch.float)
        new_lidar_states = torch.tensor(new_lidar_states, dtype=torch.float)
        new_radar_states = torch.tensor(new_radar_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        done = torch.tensor(done)

        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

        target_actions = self.target_actor.forward(new_rgb_states, new_lidar_states, new_radar_states)
        critic_new_state_value = self.target_critic.forward(new_rgb_states, new_lidar_states, new_radar_states, target_actions)
        critic_value = self.critic.forward(rgb_states, lidar_states, radar_states, actions)

        target = []

        for i in range(self.batch_size):
            target.append(rewards[i] + self.gamma * critic_new_state_value[i] * (1 - done[i]))

        # Convert to tensor and reshape it
        target = torch.tensor(target)
        target = target.view(self.batch_size, 1)

        # Calculate the loss for critic
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        # Backpropagation
        critic_loss.backward()

        # Set optimizer
        self.critic.optimizer.step()    
        # Store critic loss
        self.critic_loss = critic_loss.item()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        # Calculate the loss for actor
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(rgb_states, lidar_states, radar_states)

        self.actor.train()
        actor_loss = -self.critic.forward(rgb_states, lidar_states, radar_states, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()

        self.actor.optimizer.step()

        # Store actor loss
        self.actor_loss = actor_loss.item()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        # Update parameters for the networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Get the parameters
        actor_parameters = self.actor.named_parameters()
        target_actor_parameters = self.target_actor.named_parameters()
        critic_parameters = self.critic.named_parameters()
        target_critic_parameters = self.target_critic.named_parameters()

        # Convert them into dictionaries
        actor_state_dict = dict(actor_parameters)
        target_actor_state_dict = dict(target_actor_parameters)
        critic_state_dict = dict(critic_parameters)
        target_critic_state_dict = dict(target_critic_parameters)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1-tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict, strict=False)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1-tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict, strict=False) 

    def save_models(self, episode, is_best=False):
        print('Saving models...')
        self.actor.save_checkpoint(episode, is_best)
        self.target_actor.save_checkpoint(episode, is_best)
        self.critic.save_checkpoint(episode, is_best)
        self.target_critic.save_checkpoint(episode, is_best)

    def load_models(self, episode, is_best=False):
        print('Loading models...')
        self.actor.load_checkpoint(episode, is_best)
        self.target_actor.load_checkpoint(episode, is_best)
        self.critic.load_checkpoint(episode, is_best)
        self.target_critic.load_checkpoint(episode, is_best)