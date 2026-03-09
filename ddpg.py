#!/usr/bin/env python3
import argparse
import collections
import copy

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.8")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--replay_buffer_size", default=1_000_000, type=int, help="Replay buffer size")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--max_episodes", default=1000, type=int, help="Maximum number of training episodes.")
parser.add_argument("--train_steps_per_update", default=5, type=int, help="Number of training steps per environment step.")


class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # Get action and observation dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Create the actor network
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, action_dim),
            torch.nn.Tanh()
        ).to(self.device)
        
        # Scale the tanh output to match the action space
        self.action_scale = torch.tensor(
            (env.action_space.high - env.action_space.low) / 2,
            device=self.device
        )
        self.action_bias = torch.tensor(
            (env.action_space.high + env.action_space.low) / 2,
            device=self.device
        )
        
        # Create the target actor as a copy of the actor
        self.target_actor = copy.deepcopy(self.actor)
        
        # Create the critic network
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)
        
        # Create the target critic as a copy of the critic
        self.target_critic = copy.deepcopy(self.critic)
        
        # Set up optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.learning_rate)
        
        # Store arguments
        self.args = args
        
        # For stability
        self.update_counter = 0

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # Train the critic using MSE loss
        states_actions = torch.cat([states, actions], dim=1)
        q_values = self.critic(states_actions)
        critic_loss = torch.nn.functional.mse_loss(q_values, returns.unsqueeze(1))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Gradient clipping for stability
        self.critic_optimizer.step()
        
        # Update the actor less frequently for stability
        self.update_counter += 1
        if self.update_counter % 2 == 0:
            # Train the actor using the DPG loss
            actor_actions = self.actor(states)
            actor_actions = actor_actions * self.action_scale + self.action_bias
            actor_loss = -self.critic(torch.cat([states, actor_actions], dim=1)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Gradient clipping for stability
            self.actor_optimizer.step()
            
            # Update target networks with EMA
            npfl139.update_params_by_ema(self.target_actor, self.actor, self.args.target_tau)
            npfl139.update_params_by_ema(self.target_critic, self.critic, self.args.target_tau)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # Get actions from the actor network
        actions = self.actor(states)
        # Scale from [-1, 1] to the action space range
        actions = actions * self.action_scale + self.action_bias
        return actions

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # Get actions from the target actor
        actions = self.target_actor(states)
        # Scale from [-1, 1] to the action space range
        actions = actions * self.action_scale + self.action_bias
        # Evaluate actions using the target critic
        states_actions = torch.cat([states, actions], dim=1)
        return self.target_critic(states_actions)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    # Replay memory of a specified maximum size.
    replay_buffer = npfl139.MonolithicReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # Predict actions without exploration noise during evaluation
            action = agent.predict_actions(np.expand_dims(state, axis=0))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Set target return based on environment
    target_return = -150 if args.env == "Pendulum-v1" else 9000
    best_eval_return = float('-inf')
    
    # Noise process for action exploration
    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    
    # Adaptive noise scaling - will reduce over time
    noise_scale = 1.0
    
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset()[0], False
            noise.reset()
            episode_steps = 0
            
            while not done:
                # Predict actions with exploration noise
                action = agent.predict_actions(np.expand_dims(state, axis=0))[0]
                # Add noise and scale it (decrease over time)
                action = action + noise.sample() * noise_scale
                # Clip actions to valid range
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state
                episode_steps += 1

                if len(replay_buffer) < 4 * args.batch_size:
                    continue
                
                # Perform multiple training steps per environment step for more efficient learning
                for _ in range(args.train_steps_per_update):
                    # Sample batch from replay buffer
                    batch = replay_buffer.sample(args.batch_size)
                    states, actions, rewards, dones, next_states = batch
                    
                    # Compute target Q-values
                    next_values = agent.predict_values(next_states)
                    
                    # Calculate returns using Bellman equation
                    target_returns = rewards + args.gamma * (1 - dones) * next_values.flatten()
                    
                    # Train the agent
                    agent.train(states, actions, target_returns)
            
            # Gradually reduce noise for better exploitation as training progresses
            noise_scale = max(0.1, noise_scale * 0.995)

        # Periodic evaluation
        returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        mean_return = np.mean(returns)
        print("Evaluation after episode {}: {:.2f}".format(env.episode, mean_return))
        
        # Update best return and check if we've reached the target
        best_eval_return = max(best_eval_return, mean_return)
        if mean_return >= target_return:
            print(f"Target return {target_return} achieved with {mean_return:.2f}!")
            training = False
        
        # Stop after max episodes
        if env.episode >= args.max_episodes:
            print(f"Reached maximum episodes. Best evaluation return: {best_eval_return:.2f}")
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)