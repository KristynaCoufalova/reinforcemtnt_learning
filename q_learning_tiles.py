#!/usr/bin/env python3
import argparse
import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2425.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Initial learning rate.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=30000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=18, type=int, help="Number of tiles.")
parser.add_argument("--n_step", default=5, type=int, help="Number of steps for n-step updates.")

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # Initialize weights for the action-value function approximation
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon
    alpha = args.alpha
    n = args.n_step  # n-step updates

    training = True
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        trajectory = []  # Stores (state, action, reward) tuples
        while not done:
            # Choose an action using epsilon-greedy policy with noise to prevent overfitting
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                q_values = np.sum(W[state], axis=0)
                action = np.argmax(q_values + np.random.randn() * 0.01)  # Add small noise to break ties

            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            trajectory.append((state, action, reward))
            
            # n-step update
            if len(trajectory) >= n or done:
                G = sum([args.gamma ** i * trajectory[i][2] for i in range(len(trajectory))])
                if not done:
                    best_next_action = np.argmax(np.sum(W[next_state], axis=0))
                    G += args.gamma ** n * W[next_state, best_next_action]
                
                # Update weights using importance sampling correction
                state_t, action_t, _ = trajectory[0]
                W[state_t, action_t] += alpha * (G - W[state_t, action_t])
                trajectory.pop(0)

            state = next_state
        
        # Decay epsilon and adaptive learning rate
        if args.epsilon_final_at:
            epsilon = max(args.epsilon_final, args.epsilon * 0.995)
        alpha = max(0.01, alpha * 0.999)  # Decay learning rate for stability

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Choose greedy action
            action = np.argmax(np.sum(W[state], axis=0))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
 