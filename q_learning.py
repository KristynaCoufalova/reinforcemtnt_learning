import argparse
import gymnasium as gym
import numpy as np
import npfl139

npfl139.require_version("2425.2")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--epsilon_decay", default=0.99, type=float, help="Decay factor for epsilon.")
parser.add_argument("--num_episodes", default=50000, type=int, help="Number of training episodes.")

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # Initialize Q-table
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    epsilon = args.epsilon

    training = True
    episode_count = 0
    
    while training and episode_count < args.num_episodes:
        state, done = env.reset()[0], False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)  # Exploration
            else:
                action = np.argmax(Q[state])  # Exploitation
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update rule
            Q[state, action] += args.alpha * (reward + args.gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
        
        # Decay epsilon
        epsilon *= args.epsilon_decay
        episode_count += 1
    
    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Greedy action selection for evaluation
            action = np.argmax(Q[state])
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
