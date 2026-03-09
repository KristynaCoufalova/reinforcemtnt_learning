#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import npfl139
npfl139.require_version("2425.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# Custom arguments
parser.add_argument("--episodes", default=1000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--epsilon_start", default=1.0, type=float)
parser.add_argument("--epsilon_end", default=0.05, type=float)
parser.add_argument("--epsilon_decay", default=0.995, type=float)
parser.add_argument("--target_update_freq", default=100, type=int)
parser.add_argument("--replay_buffer_size", default=50000, type=int)
parser.add_argument("--model_path", default="cartpole_pixels_model.pt", type=str)

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        def conv2d_out(size, k, s): return (size - k) // s + 1
        convw = conv2d_out(conv2d_out(conv2d_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_out(conv2d_out(conv2d_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.head(self.conv(x))

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    if args.recodex:
        model = torch.load(args.model_path, map_location=torch.device("cpu"))
        model.eval()
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            while not done:
                with torch.no_grad():
                    q_values = model(state)
                    action = int(torch.argmax(q_values))
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3, 80, 80)
    n_actions = env.action_space.n
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = deque(maxlen=args.replay_buffer_size)

    epsilon = args.epsilon_start
    episode_returns = []

    def select_action(state):
        if random.random() < epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            return int(torch.argmax(policy_net(state_tensor)))

    def optimize():
        if len(replay_buffer) < args.batch_size:
            return
        batch = random.sample(replay_buffer, args.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states)).permute(0, 3, 1, 2).float().to(device) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.from_numpy(np.array(next_states)).permute(0, 3, 1, 2).float().to(device) / 255.0
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + args.gamma * max_next_q_values * (1 - dones)

        loss = loss_fn(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    for episode in range(args.episodes):
        state, done = env.reset()[0], False
        total_reward = 0
        while not done:
            action = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            loss = optimize()

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        episode_returns.append(total_reward)

        if episode % args.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 10 == 0:
            recent = episode_returns[-100:]
            mean = np.mean(recent)
            std = np.std(recent)
            print(f"Episode {episode}, Return: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Mean100: {mean:.2f} ± {std:.2f}, Loss: {loss:.4f}" if loss else "")

    print("Saving model...")
    torch.save(policy_net, args.model_path)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(gym.make("CartPolePixels-v1", frames=3), main_args.seed, main_args.render_each)
    main(main_env, main_args)
