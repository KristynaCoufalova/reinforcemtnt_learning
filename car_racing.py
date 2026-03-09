#!/usr/bin/env python3
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

import npfl139
npfl139.require_version("2425.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--continuous", default=0, type=int, help="Use continuous actions.")
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--train", default=True, type=bool, help="Train the model.")
parser.add_argument("--evaluate", default=False, type=bool, help="Evaluate the model.")

# Hyperparameters
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100000
TARGET_UPDATE = 1000
NUM_EPISODES = 1000
START_TRAINING = 5000
SAVE_INTERVAL = 50
STACK_FRAMES = 4
UPDATE_FREQ = 4
PRIORITIZED_REPLAY = True
ALPHA = 0.6  # Priority exponent
BETA_START = 0.4  # Importance sampling start value
BETA_END = 1.0    # Importance sampling end value
BETA_FRAMES = 200000  # Frames over which to anneal beta


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        # Create convolutional layers without flattening yet
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened convolutional output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = self.features(dummy_input)
            conv_out_size = conv_out.view(1, -1).size(1)
            print(f"Conv output size: {conv_out_size}")
        
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        # Normalize input
        x = x / 255.0
        
        # Pass through convolutional layers
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        
        # Calculate advantage and value
        advantage = self.advantage(flattened)
        value = self.value(flattened)
        
        # Combine them to get Q-values (Dueling DQN architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        else:
            priorities = self.priorities[:len(self.buffer)]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states), 
            np.array(dones, dtype=np.bool_),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class FrameStacker:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def reset(self, frame):
        self.frames.clear()
        frame = self.preprocess(frame)
        for _ in range(self.n_frames):
            self.frames.append(frame)
        return self.get_stacked_frames()
    
    def preprocess(self, frame):
        # Downsample to 42x42 grayscale
        # Convert to grayscale using a weighted sum
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        
        # Crop and resize to focus on the track area
        # The original image is 96x96, so we can crop out some of the edges
        h, w = gray.shape
        crop_h, crop_w = int(h * 0.8), int(w * 0.8)
        start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
        
        # Crop the center region
        cropped = gray[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Downsample to 42x42 (scaled to maintain aspect ratio)
        downsampled = cropped[::2, ::2]
        return downsampled.astype(np.uint8)
    
    def add_frame(self, frame):
        self.frames.append(self.preprocess(frame))
        return self.get_stacked_frames()
    
    def get_stacked_frames(self):
        # Stack frames along channel dimension (NCHW format for PyTorch)
        return np.stack(self.frames, axis=0)


def discrete_to_continuous(action, continuous=False):
    """Convert discrete actions to continuous actions."""
    if continuous:
        # Map 5 discrete actions to continuous space
        if action == 0:    # do nothing
            return np.array([0.0, 0.0, 0.0])
        elif action == 1:  # left
            return np.array([-0.8, 0.0, 0.0])
        elif action == 2:  # right
            return np.array([0.8, 0.0, 0.0])
        elif action == 3:  # gas
            return np.array([0.0, 0.5, 0.0])
        elif action == 4:  # brake
            return np.array([0.0, 0.0, 0.5])
    else:
        return action


def train(env, q_net, target_net, optimizer, buffer, device, args, save_dir="models"):
    """Train the DQN agent."""
    frame_idx = 0
    beta_schedule = np.linspace(BETA_START, BETA_END, BETA_FRAMES)
    best_reward = -float('inf')
    frame_stacker = FrameStacker(STACK_FRAMES)
    rewards_history = []
    
    # Create directory for saving models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        state = frame_stacker.reset(obs)
        done = False
        episode_reward = 0
        
        # Keep track of time spent on road vs off-road for additional reward shaping
        on_road_time = 0
        off_road_time = 0
        
        while not done:
            # Select action with epsilon-greedy policy
            epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * frame_idx / EPS_DECAY)
            if random.random() < epsilon:
                action = random.randrange(5)  # 5 discrete actions
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = q_values.max(1)[1].item()
            
            # Convert discrete action to continuous if needed
            env_action = discrete_to_continuous(action, bool(args.continuous))
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            next_state = frame_stacker.add_frame(next_obs)
            
            # Reward shaping: encourage staying on the track
            if reward > -0.1:  # Assume this means we're on the track
                on_road_time += 1
                off_road_time = 0
            else:
                off_road_time += 1
                on_road_time = 0
            
            # Additional penalty for being off-road too long
            if off_road_time > 20:
                reward -= 0.5
            
            # Bonus for staying on track consistently
            if on_road_time > 10:
                reward += 0.1
                
            # Store transition in replay buffer
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            frame_idx += 1
            
            # Start training after collecting enough samples
            if len(buffer) > START_TRAINING and frame_idx % UPDATE_FREQ == 0:
                # Compute beta for importance sampling
                beta = beta_schedule[min(frame_idx, BETA_FRAMES-1)]
                
                # Sample from replay buffer
                states, actions, rewards, next_states, dones, indices, weights = buffer.sample(BATCH_SIZE, beta)
                
                # Convert to tensors
                states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
                
                # Compute current Q values
                q_values = q_net(states_tensor)
                q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Compute next Q values with Double DQN
                with torch.no_grad():
                    # Get actions from policy network
                    next_actions = q_net(next_states_tensor).max(1)[1]
                    # Get Q-values from target network
                    next_q_values = target_net(next_states_tensor)
                    next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    
                    # Compute target Q values
                    target_q_values = rewards_tensor + GAMMA * next_q_values * (1 - dones_tensor)
                
                # Compute TD error for prioritized replay
                td_errors = torch.abs(target_q_values - q_values).detach().cpu().numpy()
                
                # Update priorities
                buffer.update_priorities(indices, td_errors + 1e-6)  # Small epsilon to avoid zero priority
                
                # Compute loss with importance sampling weights
                loss = (weights_tensor * nn.functional.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10)
                
                optimizer.step()
                
                # Update target network
                if frame_idx % TARGET_UPDATE == 0:
                    target_net.load_state_dict(q_net.state_dict())
            
            if done:
                print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}")
                rewards_history.append(episode_reward)
                
                # Calculate moving average reward
                if len(rewards_history) >= 100:
                    avg_reward = sum(rewards_history[-100:]) / 100
                    print(f"Last 100 episodes average reward: {avg_reward:.2f}")
                    
                    # Save best model
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        torch.save(q_net.state_dict(), os.path.join(save_dir, "best_model.pt"))
                        print(f"New best model saved with average reward: {best_reward:.2f}")
                
                # Regular saving
                if (episode + 1) % SAVE_INTERVAL == 0:
                    torch.save(q_net.state_dict(), os.path.join(save_dir, f"model_ep{episode+1}.pt"))
                    
                break
    
    # Save final model
    torch.save(q_net.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print("Training completed and final model saved.")


def evaluate(env, q_net, device, args, episodes=15):
    """Evaluate the trained DQN agent."""
    frame_stacker = FrameStacker(STACK_FRAMES)
    total_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset(start_evaluation=True)
        state = frame_stacker.reset(obs)
        done = False
        episode_reward = 0
        
        while not done:
            # Select greedy action
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = q_values.max(1)[1].item()
            
            # Convert discrete action to continuous if needed
            env_action = discrete_to_continuous(action, bool(args.continuous))
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            next_state = frame_stacker.add_frame(next_obs)
            
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average evaluation reward over {episodes} episodes: {avg_reward:.2f}")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize networks
    # For frame stacking, input shape is (STACK_FRAMES, 42, 42)
    q_net = DQN((STACK_FRAMES, 42, 42), 5).to(device)  # 5 discrete actions
    target_net = DQN((STACK_FRAMES, 42, 42), 5).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # Initialize optimizer
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    # Initialize replay buffer
    buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE, alpha=ALPHA)

    # ReCodEx evaluation mode
    if args.recodex:
        # Load the pre-trained model
        q_net.load_state_dict(torch.load("model.pt", map_location=device))
        q_net.eval()

        while True:
            evaluate(env, q_net, device, args)
    else:
        # Training mode
        if args.train:
            train(env, q_net, target_net, optimizer, buffer, device, args)
        
        # Evaluation mode
        if args.evaluate:
            # Load the best model
            q_net.load_state_dict(torch.load("models/best_model.pt", map_location=device))
            q_net.eval()
            evaluate(env, q_net, device, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("CarRacingFS-v3", frame_skip=main_args.frame_skip, continuous=main_args.continuous),
        main_args.seed, main_args.render_each, evaluate_for=15, report_each=1)

    main(main_env, main_args)