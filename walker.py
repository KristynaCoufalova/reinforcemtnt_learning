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
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--envs", default=8, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=10000, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0003, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="walker.pt", type=str, help="Model path")
parser.add_argument("--replay_buffer_size", default=1_000_000, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")


class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # Create an actor
        class Actor(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                # Create two hidden layers with `hidden_layer_size` and ReLU activation
                self.input_layer = torch.nn.Linear(env.observation_space.shape[0], hidden_layer_size)
                self.hidden_layer = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
                # Layer for generating means with `env.action_space.shape[0]` units and no activation
                self.mean_layer = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])
                # Layer for generating sds with `env.action_space.shape[0]` units and `torch.exp` activation
                self.sd_layer = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])

                # Create a variable representing a logarithm of alpha
                self._log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))

                # Create two tensors representing the action scale and offset
                self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2))
                self.register_buffer("action_offset", torch.tensor((env.action_space.high + env.action_space.low) / 2))
                
                # Create the composed transform for numerical stability
                self.transform = torch.distributions.transforms.ComposeTransform([
                    torch.distributions.transforms.TanhTransform(),
                    torch.distributions.transforms.AffineTransform(self.action_offset, self.action_scale)
                ], cache_size=1)

            def forward(self, inputs: torch.Tensor, sample: bool):
                # Pass the inputs through the hidden layers
                x = torch.relu(self.input_layer(inputs))
                x = torch.relu(self.hidden_layer(x))
                
                # Compute the means and standard deviations
                mus = self.mean_layer(x)
                sds = torch.exp(self.sd_layer(x))
                
                if sample:
                    # Create the action distribution using Normal with mus and sds
                    normal_dist = torch.distributions.Normal(mus, sds)
                    
                    # Transform the distribution to be in the action range
                    action_dist = torch.distributions.TransformedDistribution(normal_dist, self.transform)
                    
                    # Sample actions
                    actions = action_dist.rsample()
                    
                    # Compute log probabilities
                    log_prob = action_dist.log_prob(actions)
                    log_prob = log_prob.mean(dim=1, keepdim=True)
                    
                    # Compute alpha
                    alpha = torch.exp(self._log_alpha)
                    
                    return actions, log_prob, alpha
                else:
                    # Without sampling, use deterministic actions
                    actions = torch.tanh(mus) * self.action_scale + self.action_offset
                    return actions, None, None

        # Instantiate the actor
        self._actor = Actor(args.hidden_layer_size).to(self.device)

        # Create a critic
        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Input layer takes observation and action
                self.input_layer = torch.nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], args.hidden_layer_size)
                self.hidden_layer = torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
                self.output_layer = torch.nn.Linear(args.hidden_layer_size, 1)

            def forward(self, state, action):
                x = torch.cat([state, action], dim=1)
                x = torch.relu(self.input_layer(x))
                x = torch.relu(self.hidden_layer(x))
                return self.output_layer(x)

        # Create two critics and their target networks
        self._critic1 = Critic().to(self.device)
        self._critic2 = Critic().to(self.device)
        self._target_critic1 = copy.deepcopy(self._critic1)
        self._target_critic2 = copy.deepcopy(self._critic2)

        # Define optimizers
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        self._critic1_optimizer = torch.optim.Adam(self._critic1.parameters(), lr=args.learning_rate)
        self._critic2_optimizer = torch.optim.Adam(self._critic2.parameters(), lr=args.learning_rate)
        self._alpha_optimizer = torch.optim.Adam([self._actor._log_alpha], lr=args.learning_rate)

        # Store arguments
        self.args = args
        
        # Create MSE loss
        self._mse_loss = torch.nn.MSELoss()

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
              dones: torch.Tensor, next_states: torch.Tensor) -> None:
        # Calculate target Q-values
        with torch.no_grad():
            next_actions, next_log_probs, alpha = self._actor(next_states, sample=True)
            
            # Compute target Q values
            target_q1 = self._target_critic1(next_states, next_actions)
            target_q2 = self._target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            
            # Calculate target using Bellman equation
            target = rewards + (1 - dones) * self.args.gamma * target_q
        
        # Update critics
        # Critic 1
        current_q1 = self._critic1(states, actions)
        critic1_loss = self._mse_loss(current_q1, target)
        self._critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self._critic1_optimizer.step()
        
        # Critic 2
        current_q2 = self._critic2(states, actions)
        critic2_loss = self._mse_loss(current_q2, target)
        self._critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self._critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs, alpha = self._actor(states, sample=True)
        
        # Compute Q values for new actions
        q1 = self._critic1(states, new_actions)
        q2 = self._critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor loss: maximize Q - alpha * log_prob
        actor_loss = (alpha.detach() * log_probs - q).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -self._actor._log_alpha * (log_probs.detach() + self.args.target_entropy)
        alpha_loss = alpha_loss.mean()
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()
        
        # Update target networks
        npfl139.update_params_by_ema(self._target_critic1, self._critic1, self.args.target_tau)
        npfl139.update_params_by_ema(self._target_critic2, self._critic2, self.args.target_tau)

    # Predict actions without sampling.
    @npfl139.typed_torch_function(device, torch.float32)
    def predict_mean_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return predicted actions.
        with torch.no_grad():
            return self._actor(states, sample=False)[0]

    # Predict actions with sampling.
    @npfl139.typed_torch_function(device, torch.float32)
    def predict_sampled_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return sampled actions from the predicted distribution
        with torch.no_grad():
            return self._actor(states, sample=True)[0]

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # Produce the predicted returns
        with torch.no_grad():
            actions, log_probs, alpha = self._actor(states, sample=True)
            
            # Compute Q values from both target critics
            q1 = self._target_critic1(states, actions)
            q2 = self._target_critic2(states, actions)
            
            # Return the minimum Q value minus the entropy term
            return torch.min(q1, q2) - alpha * log_probs

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # Predict an action by using a greedy policy
            action = agent.predict_mean_actions(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        agent.load_actor(args.model_path)
        while True:
            evaluate_episode(True)

    # Create the asynchroneous vector environment for training.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})

    # Replay memory of a specified maximum size.
    replay_buffer = npfl139.MonolithicReplayBuffer(args.replay_buffer_size, args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state = vector_env.reset(seed=args.seed)[0]
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # Predict actions by calling `agent.predict_sampled_actions`.
            action = agent.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated
            
            # Override reward on crash to 0 to speed up training
            reward = np.where(reward < -10, 0, reward)
            
            replay_buffer.append_batch(Transition(state, action, reward, done, next_state))
            state = next_state

            # Training
            if len(replay_buffer) >= 10 * args.batch_size:
                # Randomly uniformly sample transitions from the replay buffer.
                batch = replay_buffer.sample(args.batch_size)
                states, actions, rewards, dones, next_states = batch
                # Perform the training
                agent.train(states, actions, rewards, dones, next_states)

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        print(f"Evaluation: mean return {np.mean(returns):.2f} after {len(replay_buffer)} transitions")
        
        # Save model if reaching good performance
        if np.mean(returns) >= 250:
            agent.save_actor(args.model_path)
            print(f"Model saved with average return {np.mean(returns):.2f}")
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)