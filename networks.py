import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, lr, rgb_dims, lidar_dims, radar_dims, fcp_dims, fc1_dims, fc2_dims, num_of_actions, type, chkpt_dir='./saved_models') -> None:
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.rgb_dims = rgb_dims
        self.lidar_dims = lidar_dims
        self.radar_dims = radar_dims
        self.combined_input_size = fcp_dims + fc2_dims + fc2_dims
        self.fcp_dims = fcp_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.num_of_actions = num_of_actions
        self.best_chkpt_file = os.path.join('./best_models', type + '_ddpg')
        self.chkpt_file  = os.path.join(chkpt_dir, type + '_ddpg')
        os.makedirs(chkpt_dir, exist_ok=True)

        # Pixel Case for RGB Images
        self.conv1 = nn.Conv2d(in_channels=self.rgb_dims[0], out_channels=32, kernel_size=5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.conv3_bn = nn.BatchNorm2d(32)

        # Calculate the output size after convolutions
        conv_output_size = self.calculate_conv_output_size(rgb_dims[1], rgb_dims[2]) * 32  # Times 32 for final conv layer's out_channels

        # FC1 for pixel case
        self.pixel_fc1 = nn.Linear(conv_output_size, self.fcp_dims)
        p_f1 = 1 / np.sqrt(self.pixel_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.pixel_fc1.weight.data, -p_f1, p_f1)
        torch.nn.init.uniform_(self.pixel_fc1.bias.data, -p_f1, p_f1)

        self.pixel_bn1 = nn.BatchNorm1d(self.fcp_dims)

        # FC2 for pixel case
        self.pixel_fc2 = nn.Linear(self.fcp_dims, self.fcp_dims)
        p_f2 = 1 / np.sqrt(self.pixel_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.pixel_fc2.weight.data, -p_f2, p_f2)
        torch.nn.init.uniform_(self.pixel_fc2.bias.data, -p_f2, p_f2)

        self.pixel_bn2 = nn.BatchNorm1d(self.fcp_dims)

        # Low dim case for LIDAR 3D points
        # FC1
        self.lidar_fc1 = nn.Linear(3, self.fc1_dims)
        l_f1 = 1 / np.sqrt(self.lidar_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.lidar_fc1.weight.data, -l_f1, l_f1)
        torch.nn.init.uniform_(self.lidar_fc1.bias.data, -l_f1, l_f1)

        # Batch Normalization
        self.lidar_bn1 = nn.BatchNorm1d(self.fc1_dims)

        # FC2
        self.lidar_fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        lidar_f2 = 1 / np.sqrt(self.lidar_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.lidar_fc2.weight.data, -lidar_f2, lidar_f2)
        torch.nn.init.uniform_(self.lidar_fc2.bias.data, -lidar_f2, lidar_f2)

        self.lidar_bn2 = nn.BatchNorm1d(self.fc2_dims)

        # Low dim case for RADAR 3D points
        # FC1
        self.radar_fc1 = nn.Linear(3, self.fc1_dims)
        radar_f1 = 1 / np.sqrt(self.radar_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.radar_fc1.weight.data, -radar_f1, radar_f1)
        torch.nn.init.uniform_(self.radar_fc1.bias.data, -radar_f1, radar_f1)

        # Batch Normalization
        self.radar_bn1 = nn.BatchNorm1d(self.fc1_dims)

        # FC2
        self.radar_fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        radar_f2 = 1 / np.sqrt(self.radar_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.radar_fc2.weight.data, -radar_f2, radar_f2)
        torch.nn.init.uniform_(self.radar_fc2.bias.data, -radar_f2, radar_f2)

        self.radar_bn2 = nn.BatchNorm1d(self.fc2_dims)

        # Low dim case for Combined Features of RGB, LIDAR and RADAR Layers
        # Fully Connected Layer 1
        self.combined_fc1 = nn.Linear(self.combined_input_size, self.fc1_dims)
        combined_f1 = 1 / np.sqrt(self.combined_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.combined_fc1.weight.data, -combined_f1, combined_f1)
        torch.nn.init.uniform_(self.combined_fc1.bias.data, -combined_f1, combined_f1)

        # Batch Normalization for FC1
        self.combined_bn1 = nn.BatchNorm1d(self.fc1_dims)

        # Fully Connected Layer 2
        self.combined_fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        combined_f2 = 1 / np.sqrt(self.combined_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.combined_fc2.weight.data, -combined_f2, combined_f2)
        torch.nn.init.uniform_(self.combined_fc2.bias.data, -combined_f2, combined_f2)

        # Batch Normalization for FC2
        self.combined_bn2 = nn.BatchNorm1d(self.fc2_dims)

        # Final Layer
        fl = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.num_of_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -fl, fl)
        torch.nn.init.uniform_(self.mu.bias.data, -fl, fl)

        # Adam Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def calculate_conv_output_size(self, height, width):
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        
        return conv_height * conv_width  # Total size after flattening

    def forward(self, rgb_input, lidar_input, radar_input):
        # RGB
        x_rgb = self.conv1(rgb_input)
        x_rgb = self.conv1_bn(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = self.conv2(x_rgb)
        x_rgb = self.conv2_bn(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = self.conv3(x_rgb)
        x_rgb = self.conv3_bn(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = x_rgb.view(x_rgb.size(0), -1)  # Flatten
        x_rgb = self.pixel_fc1(x_rgb)
        x_rgb = self.pixel_bn1(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = self.pixel_fc2(x_rgb)
        x_rgb = self.pixel_bn2(x_rgb)
        x_rgb = F.relu(x_rgb)

        # LIDAR
        x_lidar = torch.mean(lidar_input, dim=1)  # Global Pooling with mean
        x_lidar = self.lidar_fc1(x_lidar)
        x_lidar = self.lidar_bn1(x_lidar)
        x_lidar = F.relu(x_lidar)
        x_lidar = self.lidar_fc2(x_lidar)
        x_lidar = self.lidar_bn2(x_lidar)
        x_lidar = F.relu(x_lidar)

        # RADAR

        # Preprocess radar input
        radar_input = torch.nan_to_num(radar_input, nan=0.0)  # Replace NaN values with 0.0

        # Validate radar input for NaN and Inf values
        if torch.isnan(radar_input).any():
            print('Radar input contains NaNs, replacing with zeros.')
            radar_input = torch.zeros_like(radar_input)

        if torch.isinf(radar_input).any():
            print('Radar input contains infinities, replacing with zeros.')
            radar_input = torch.zeros_like(radar_input)

        x_radar = torch.mean(radar_input, dim=1)  # Global Pooling with mean
        x_radar = self.radar_fc1(x_radar)
        x_radar = self.radar_bn1(x_radar)
        x_radar = F.relu(x_radar)
        x_radar = self.radar_fc2(x_radar)
        x_radar = self.radar_bn2(x_radar)
        x_radar = F.relu(x_radar)

        # Combined Features
        x_combined = torch.cat([x_rgb, x_lidar, x_radar], dim=-1)
        x_combined = self.combined_fc1(x_combined)
        x_combined = self.combined_bn1(x_combined)
        x_combined = F.relu(x_combined)
        x_combined = self.combined_fc2(x_combined)
        x_combined = self.combined_bn2(x_combined)
        x_combined = F.relu(x_combined)

        # Final Output (mu)
        mu = torch.tanh(self.mu(x_combined))  # Action space (-1, 1)

        return mu

    def save_checkpoint(self, episode, is_best=False):
        print('Saving checkpoint...')

        if is_best:
            checkpoint_path = f'{self.best_chkpt_file}_best_{episode}.pth'
        else:
            checkpoint_path = f'{self.chkpt_file}_{episode}.pth'
        
        torch.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, episode, is_best=False):
        print('Loading checkpoint...')
        
        if is_best:
            checkpoint_path = f'{self.best_chkpt_file}_best_{episode}.pth'
        else:
            checkpoint_path = f'{self.chkpt_file}_{episode}.pth'

        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint)


class CriticNetWork(nn.Module):
    def __init__(self, lr, rgb_dims, lidar_dims, radar_dims, fcp_dims, fc1_dims, fc2_dims, num_of_actions, type, chkpt_dir='./saved_models') -> None:
        super(CriticNetWork, self).__init__()
        self.lr = lr
        self.rgb_dims = rgb_dims
        self.lidar_dims = lidar_dims
        self.radar_dims = radar_dims
        self.combined_input_size = fcp_dims + fc2_dims + fc2_dims
        self.fcp_dims = fcp_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.num_of_actions = num_of_actions
        self.best_chkpt_file = os.path.join('./best_models', type + '_ddpg')
        self.chkpt_file  = os.path.join(chkpt_dir, type + '_ddpg')
        os.makedirs(chkpt_dir, exist_ok=True)

        # Pixel Case for RGB Images
        self.conv1 = nn.Conv2d(in_channels=rgb_dims[0], out_channels=32, kernel_size=5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.conv3_bn = nn.BatchNorm2d(32)

        # Calculate the output size after convolutions
        conv_output_size = self.calculate_conv_output_size(rgb_dims[1], rgb_dims[2]) * 32  # Times 32 for final conv layer's out_channels

        # FC1 for pixel case
        self.pixel_fc1 = nn.Linear(conv_output_size, self.fcp_dims)
        p_f1 = 1 / np.sqrt(self.pixel_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.pixel_fc1.weight.data, -p_f1, p_f1)
        torch.nn.init.uniform_(self.pixel_fc1.bias.data, -p_f1, p_f1)

        self.pixel_bn1 = nn.BatchNorm1d(self.fcp_dims)

        # FC2 for pixel case
        self.pixel_fc2 = nn.Linear(self.fcp_dims, self.fcp_dims)
        p_f2 = 1 / np.sqrt(self.pixel_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.pixel_fc2.weight.data, -p_f2, p_f2)
        torch.nn.init.uniform_(self.pixel_fc2.bias.data, -p_f2, p_f2)

        self.pixel_bn2 = nn.BatchNorm1d(self.fcp_dims)

        # Low dim case for LIDAR 3D points
        # FC1
        self.lidar_fc1 = nn.Linear(3, self.fc1_dims)
        l_f1 = 1 / np.sqrt(self.lidar_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.lidar_fc1.weight.data, -l_f1, l_f1)
        torch.nn.init.uniform_(self.lidar_fc1.bias.data, -l_f1, l_f1)

        # Batch Normalization
        self.lidar_bn1 = nn.BatchNorm1d(self.fc1_dims)

        # FC2
        self.lidar_fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        lidar_f2 = 1 / np.sqrt(self.lidar_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.lidar_fc2.weight.data, -lidar_f2, lidar_f2)
        torch.nn.init.uniform_(self.lidar_fc2.bias.data, -lidar_f2, lidar_f2)

        self.lidar_bn2 = nn.BatchNorm1d(self.fc2_dims)

        # Low dim case for RADAR 3D points
        # FC1
        self.radar_fc1 = nn.Linear(3, self.fc1_dims)
        radar_f1 = 1 / np.sqrt(self.radar_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.radar_fc1.weight.data, -radar_f1, radar_f1)
        torch.nn.init.uniform_(self.radar_fc1.bias.data, -radar_f1, radar_f1)

        # Batch Normalization
        self.radar_bn1 = nn.BatchNorm1d(self.fc1_dims)

        # FC2
        self.radar_fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        radar_f2 = 1 / np.sqrt(self.radar_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.radar_fc2.weight.data, -radar_f2, radar_f2)
        torch.nn.init.uniform_(self.radar_fc2.bias.data, -radar_f2, radar_f2)

        self.radar_bn2 = nn.BatchNorm1d(self.fc2_dims)

        # Low dim case for Combined Features of RGB, LIDAR and RADAR Layers
        # Fully Connected Layer 1
        self.combined_fc1 = nn.Linear(self.combined_input_size, self.fc1_dims)
        combined_f1 = 1 / np.sqrt(self.combined_fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.combined_fc1.weight.data, -combined_f1, combined_f1)
        torch.nn.init.uniform_(self.combined_fc1.bias.data, -combined_f1, combined_f1)

        # Batch Normalization for FC1
        self.combined_bn1 = nn.BatchNorm1d(self.fc1_dims)

        # Fully Connected Layer 2
        self.combined_fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        combined_f2 = 1 / np.sqrt(self.combined_fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.combined_fc2.weight.data, -combined_f2, combined_f2)
        torch.nn.init.uniform_(self.combined_fc2.bias.data, -combined_f2, combined_f2)

        # Batch Normalization for FC2
        self.combined_bn2 = nn.BatchNorm1d(self.fc2_dims)

        # Final Layer
        self.action_value = nn.Linear(self.num_of_actions, self.fc2_dims)
        fl = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -fl, fl)
        torch.nn.init.uniform_(self.q.bias.data, -fl, fl)

        # Adam Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def calculate_conv_output_size(self, height, width):
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        
        return conv_height * conv_width  # Total size after flattening

    def forward(self, rgb_input, lidar_input, radar_input, action):
        # For the RGB input
        x_rgb = self.conv1(rgb_input)
        x_rgb = self.conv1_bn(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = self.conv2(x_rgb)
        x_rgb = self.conv2_bn(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = self.conv3(x_rgb)
        x_rgb = self.conv3_bn(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = x_rgb.view(x_rgb.size(0), -1)  # Flatten
        x_rgb = self.pixel_fc1(x_rgb)
        x_rgb = self.pixel_bn1(x_rgb)
        x_rgb = F.relu(x_rgb)
        x_rgb = self.pixel_fc2(x_rgb)
        x_rgb = self.pixel_bn2(x_rgb)
        x_rgb = F.relu(x_rgb)

        # For the LIDAR input
        x_lidar = torch.mean(lidar_input, dim=1)  # Global Pooling with mean
        x_lidar = self.lidar_fc1(x_lidar)
        x_lidar = self.lidar_bn1(x_lidar)
        x_lidar = F.relu(x_lidar)
        x_lidar = self.lidar_fc2(x_lidar)
        x_lidar = self.lidar_bn2(x_lidar)
        x_lidar = F.relu(x_lidar)


        # For the RADAR input

        # Preprocess radar input
        radar_input = torch.nan_to_num(radar_input, nan=0.0)  # Replace NaN values with 0.0

        # Validate radar input for NaN and Inf Values
        if torch.isnan(radar_input).any():
            print('Radar input contains NaNs, replacing with zeros.')
            radar_input = torch.zeros_like(radar_input)

        if torch.isinf(radar_input).any():
            print('Radar input contains infinities, replacing with zeros.')
            radar_input = torch.zeros_like(radar_input)

        x_radar = torch.mean(radar_input, dim=1)  # Global Pooling with mean
        x_radar = self.radar_fc1(x_radar)
        x_radar = self.radar_bn1(x_radar)
        x_radar = F.relu(x_radar)
        x_radar = self.radar_fc2(x_radar)
        x_radar = self.radar_bn2(x_radar)
        x_radar = F.relu(x_radar)

        # Combine the results
        combined_features = torch.cat([x_rgb, x_lidar, x_radar], dim=1)

        # For the Combined input
        state_value = self.combined_fc1(combined_features)
        state_value = self.combined_bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.combined_fc2(state_value)
        state_value = self.combined_bn2(state_value)

        action_value = F.relu(self.action_value(action))
        
        state_action_value = torch.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, episode, is_best=False):
        print('Saving checkpoint...')

        if is_best:
            checkpoint_path = f'{self.best_chkpt_file}_best_{episode}.pth'
        else:
            checkpoint_path = f'{self.chkpt_file}_{episode}.pth'
        
        torch.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, episode, is_best=False):
        print('Loading checkpoint...')
        
        if is_best:
            checkpoint_path = f'{self.best_chkpt_file}_best_{episode}.pth'
        else:
            checkpoint_path = f'{self.chkpt_file}_{episode}.pth'

        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint)