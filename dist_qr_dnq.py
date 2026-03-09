#!/usr/bin/env python3
import argparse
import collections
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.6")

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--render_each", default=0, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--verify", default=False, action="store_true")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epsilon", default=1.0, type=float)
parser.add_argument("--epsilon_final", default=0.05, type=float)
parser.add_argument("--epsilon_final_at", default=1000, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--hidden_layer_size", default=128, type=int)
parser.add_argument("--kappa", default=1.0, type=float)
parser.add_argument("--learning_rate", default=0.0005, type=float)
parser.add_argument("--quantiles", default=51, type=int)
parser.add_argument("--target_update_freq", default=100, type=int)


class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._model = torch.nn.Sequential(
            torch.nn.Linear(np.prod(env.observation_space.shape), args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, int(env.action_space.n) * int(args.quantiles)),
            torch.nn.Unflatten(-1, (int(env.action_space.n), int(args.quantiles)))
        ).to(self.device)

        self.gamma = args.gamma
        self.kappa = args.kappa
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    @staticmethod
    def compute_loss(states_quantiles, actions, rewards, dones, next_states_quantiles, gamma, kappa):
        batch_size, num_actions, num_quantiles = states_quantiles.shape
        tau = torch.arange(1, num_quantiles + 1, dtype=torch.float32, device=states_quantiles.device) / num_quantiles
        tau = tau.view(1, 1, -1)

        actions = actions.view(-1, 1, 1).expand(-1, 1, num_quantiles)
        predicted = states_quantiles.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = next_states_quantiles.mean(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            next_actions = next_actions.view(-1, 1, 1).expand(-1, 1, num_quantiles)
            target_quantiles = next_states_quantiles.gather(1, next_actions).squeeze(1)
            targets = rewards.view(-1, 1) + gamma * (1 - dones.view(-1, 1)) * target_quantiles

        diff = targets.unsqueeze(1) - predicted.unsqueeze(2)
        abs_diff = torch.abs(diff)

        if kappa == 0.0:
            loss = torch.abs(tau - (diff < 0).float()) * abs_diff
        else:
            huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
            loss = torch.abs(tau - (diff < 0).float()) * huber

        return loss.mean()

    def train(self, states, actions, rewards, dones, next_states, target_model):
        self._model.train()
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        loss = self.compute_loss(
            self._model(states),
            actions,
            rewards,
            dones,
            target_model(next_states),
            self.gamma,
            self.kappa
        )
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states):
        self._model.eval()
        with torch.no_grad():
            return self._model(states).mean(dim=2).cpu().numpy()

    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    if args.verify:
        return Network.compute_loss

    network = Network(env, args)
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    replay_buffer = npfl139.ReplayBuffer()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    max_episodes = 5000
    while env.episode < max_episodes:
        state, done = env.reset()[0], False
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            if np.random.rand() < epsilon:
                action = np.random.randint(len(q_values))
            else:
                action = int(np.argmax(q_values))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.append(Transition(state, action, reward, done, next_state))

            if len(replay_buffer) >= 1000:
                sample = replay_buffer.sample(args.batch_size)
                states = np.array([t.state for t in sample])
                actions = np.array([t.action for t in sample])
                rewards = np.array([t.reward for t in sample])
                dones = np.array([t.done for t in sample])
                next_states = np.array([t.next_state for t in sample])
                network.train(states, actions, rewards, dones, next_states, target_network._model)

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if env.episode % args.target_update_freq == 0:
            target_network.copy_weights_from(network)

    # Evaluation phase
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            action = int(np.argmax(q_values))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)
    result = main(main_env, main_args)

    if main_args.verify:
        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-1.4, 0.1, 0.8], [-1.2, 0.1, 1.1]]]),
            actions=torch.tensor([1]), rewards=torch.tensor([-1.5]), dones=torch.tensor([0.]),
            next_states_quantiles=torch.tensor([[[-0.4, 0.1, 0.4], [-0.5, 1.0, 1.6]]]),
            gamma=0.2, kappa=1.5).numpy(force=True), 0.3294963, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.0, 0.1, 1.2], [-1.8, -0.2, -0.1]],
                                           [[-0.3, 0.5, 1.3], [-1.4, -0.7, -0.1]],
                                           [[-0.3, -0.0, 1.9], [-1.1, -0.2, -0.1]]]),
            actions=torch.tensor([1, 0, 1]), rewards=torch.tensor([0.5, 1.4, 0.1]), dones=torch.tensor([0., 0., 1.]),
            next_states_quantiles=torch.tensor([[[-1.1, 0.2, 0.3], [-0.4, 1.1, 1.3]],
                                                [[-0.6, -0.5, 2.0], [-0.3, 0.2, 0.4]],
                                                [[-0.9, 0.7, 2.3], [-0.3, 0.7, 0.7]]]),
            gamma=0.8, kappa=0.0).numpy(force=True), 0.4392593, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.8, -0.5, -0.0, 0.3], [-0.7, -0.2, -0.2, 1.6]],
                                           [[-1.5, -1.4, -0.6, 0.1], [-2.1, -1.5, -0.3, 0.3]]]),
            actions=torch.tensor([1, 0]), rewards=torch.tensor([-0.0, 0.7]), dones=torch.tensor([1., 0.]),
            next_states_quantiles=torch.tensor([[[-1.2, 0.3, 0.4, 0.7], [-1.2, -0.1, 0.4, 2.2]],
                                                [[-1.5, 0.2, 0.2, 0.5], [-0.9, 0.4, 0.5, 1.3]]]),
            gamma=0.3, kappa=3.5).numpy(force=True), 0.2906375, atol=1e-5)
