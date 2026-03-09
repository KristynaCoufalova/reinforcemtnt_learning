import argparse
import collections

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.4")

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=10, type=int, help="Target update frequency.")
parser.add_argument("--replay_buffer_size", default=100000, type=int, help="Maximum replay buffer size.")
parser.add_argument("--min_buffer_size", default=1000, type=int, help="Minimum buffer size to start training.")
parser.add_argument("--evaluate_only", default=False, action="store_true", help="Only evaluate the policy without training")


class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self._model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, action_dim)
        ).to(self.device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.SmoothL1Loss()

    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states)

    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    network = Network(env, args)
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    if not args.evaluate_only:
        epsilon = args.epsilon
        training = True
        evaluation_window = collections.deque(maxlen=100)

        while training:
            state, done = env.reset()[0], False
            episode_return = 0

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.randint(env.action_space.n)
                else:
                    q_values = network.predict(state[np.newaxis])[0]
                    action = np.argmax(q_values)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward

                replay_buffer.append(Transition(state, action, reward, done, next_state))

                if len(replay_buffer) >= args.min_buffer_size:
                    transitions = replay_buffer.sample(args.batch_size)
                    states = np.array([t.state for t in transitions])
                    actions = np.array([t.action for t in transitions])
                    rewards = np.array([t.reward for t in transitions])
                    dones = np.array([t.done for t in transitions])
                    next_states = np.array([t.next_state for t in transitions])

                    next_q_values = target_network.predict(next_states)
                    max_next_q_values = np.max(next_q_values, axis=1)
                    current_q_values = network.predict(states)

                    for i in range(args.batch_size):
                        if dones[i]:
                            current_q_values[i, actions[i]] = rewards[i]
                        else:
                            current_q_values[i, actions[i]] = rewards[i] + args.gamma * max_next_q_values[i]

                    network.train(states, current_q_values)

                state = next_state

            evaluation_window.append(episode_return)

            if args.epsilon_final_at:
                epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

            if env.episode % args.target_update_freq == 0:
                target_network.copy_weights_from(network)

            if env.episode % 10 == 0:
                print(f"Episode {env.episode}, epsilon: {epsilon:.3f}, mean return: {np.mean(evaluation_window):.2f}")

            if len(evaluation_window) >= 100 and np.mean(evaluation_window) >= 450:
                print(f"Solved in {env.episode} episodes! Mean return: {np.mean(evaluation_window):.2f}")
                break

            if env.episode >= 3000:
                print(f"Stopping after 3000 episodes. Best mean return: {np.mean(evaluation_window):.2f}")
                break

    # Final evaluation
    total_returns = []
    for _ in range(100):
        state, done = env.reset(start_evaluation=True)[0], False
        episode_return = 0
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        total_returns.append(episode_return)

    print(f"The mean 100-episode return after evaluation {np.mean(total_returns):.2f} ± {np.std(total_returns):.2f}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)
    main(main_env, main_args)
