#!/usr/bin/env python3
import argparse
import collections
import copy
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.5")

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--render_each", default=0, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--verify", default=False, action="store_true")
parser.add_argument("--atoms", default=51, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epsilon", default=1.0, type=float)
parser.add_argument("--epsilon_final", default=0.01, type=float)
parser.add_argument("--epsilon_final_at", default=2000, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--hidden_layer_size", default=128, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--target_update_freq", default=100, type=int)


class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        input_size = np.prod(env.observation_space.shape)
        self._model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n * args.atoms),
            torch.nn.Unflatten(1, (int(env.action_space.n), int(args.atoms))),
        )

        self._model.register_buffer("atoms", torch.linspace(0, 200, args.atoms))
        self._model.to(self.device)
        self._target_model = copy.deepcopy(self._model)
        self._target_model.to(self.device)

        self.gamma = args.gamma
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    @staticmethod
    def compute_loss(states_logits, actions, rewards, dones, next_states_logits, atoms, gamma):
        batch_size, num_actions, num_atoms = states_logits.shape
        device = states_logits.device
        delta_z = atoms[1] - atoms[0]
        v_min, v_max = atoms[0], atoms[-1]

        with torch.no_grad():
            next_qs = torch.sum(torch.nn.functional.softmax(next_states_logits, dim=-1) * atoms, dim=-1)
            next_actions = torch.argmax(next_qs, dim=-1)
            next_logits = next_states_logits[torch.arange(batch_size), next_actions]
            next_dist = torch.nn.functional.softmax(next_logits, dim=-1)

            Tz = rewards.unsqueeze(1) + gamma * (1 - dones).unsqueeze(1) * atoms.unsqueeze(0)
            Tz = Tz.clamp(v_min.item(), v_max.item())
            b = (Tz - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            proj_dist = torch.zeros(batch_size, num_atoms, device=device)
            for j in range(num_atoms):
                lj = l[:, j].clamp(0, num_atoms - 1)
                uj = u[:, j].clamp(0, num_atoms - 1)
                pj = next_dist[:, j]

                proj_dist.scatter_add_(1, lj.unsqueeze(1), pj * (uj.float() - b[:, j]).unsqueeze(1))
                proj_dist.scatter_add_(1, uj.unsqueeze(1), pj * (b[:, j] - lj.float()).unsqueeze(1))

        logits = states_logits[torch.arange(batch_size), actions]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return -torch.sum(proj_dist * log_probs, dim=-1).mean()

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states, actions, rewards, dones, next_states):
        self._model.train()
        self._target_model.eval()
        with torch.no_grad():
            next_states_logits = self._target_model(next_states)
        loss = self.compute_loss(
            self._model(states), actions, rewards, dones,
            next_states_logits, self._model.atoms, self.gamma
        )
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(states)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            q_values = torch.sum(probs * self._model.atoms, dim=-1)
            return q_values.cpu().numpy()

    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    if args.verify:
        return Network.compute_loss

    network = Network(env, args)
    replay_buffer = npfl139.ReplayBuffer()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True
    while training:
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

            if len(replay_buffer) >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                states, actions, rewards, dones, next_states = zip(*batch)
                network.train(np.array(states), np.array(actions), np.array(rewards),
                              np.array(dones, dtype=np.float32), np.array(next_states))

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if env.episode % args.target_update_freq == 0:
            network._target_model.load_state_dict(network._model.state_dict())

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
        import numpy as np
        np.testing.assert_allclose(result(
            states_logits=torch.tensor([[[-1.5, 1.2, -1.2], [-0.0, -1.8, -0.1]],
                                        [[-0.2, -0.3, 1.3], [0.5, -1.1, -0.7]],
                                        [[-0.1, 1.9, -0.0], [-0.3, -1.1, -0.1]]]),
            actions=torch.tensor([0, 1, 0]),
            rewards=torch.tensor([0.5, -0.2, 0.7]), dones=torch.tensor([1., 0., 0.]),
            next_states_logits=torch.tensor([[[1.1, 0.2, 0.3], [0.3, 1.1, 1.3]],
                                             [[-0.4, -0.5, -0.6], [2.0, 1.2, 0.4]],
                                             [[-0.3, -0.9, 2.3], [0.7, 0.7, -0.3]]]),
            atoms=torch.tensor([-2., -1., 0.]),
            gamma=0.3).numpy(force=True), 2.170941, atol=1e-5)

        np.testing.assert_allclose(result(
            states_logits=torch.tensor([[[0.1, 1.4, -0.5, -0.8], [0.3, -0.0, -0.2, -0.2]],
                                        [[1.2, -0.8, -1.4, -1.5], [0.1, -0.6, -2.1, -0.3]]]),
            actions=torch.tensor([0, 1]),
            rewards=torch.tensor([0.5, 0.6]), dones=torch.tensor([0., 0.]),
            next_states_logits=torch.tensor([[[0.8, 1.2, -1.2, 0.7], [0.3, 0.4, -1.2, 0.4]],
                                             [[-0.2, 1.0, -1.5, 0.2], [0.2, 0.5, 0.4, -0.9]]]),
            atoms=torch.tensor([-3., 0., 3., 6.]),
            gamma=0.2).numpy(force=True), 1.43398, atol=1e-5)
