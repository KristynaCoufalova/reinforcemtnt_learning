#!/usr/bin/env python3
import argparse
import gymnasium as gym
import numpy as np
import pickle
import random
from collections import deque, defaultdict
import npfl139
npfl139.require_version("2425.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.9, type=float, help="Initial exploration factor.")
parser.add_argument("--gamma", default=0.995, type=float, help="Discounting factor.")
parser.add_argument("--expert_demos", default=200, type=int, help="Number of expert demonstrations to use.")
parser.add_argument("--episodes", default=15000, type=int, help="Number of training episodes.")

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)
    
    # Get action space size
    n_actions = env.action_space.n
    print(f"Action space size: {n_actions}")
    
    # Initialize Q-table using defaultdict to handle any possible state
    q_table = defaultdict(lambda: np.zeros(n_actions))
    
    if args.recodex:
        # Load the pre-trained Q-table for evaluation
        try:
            with open("q_table.pkl", "rb") as f:
                loaded_q_table = pickle.load(f)
                
                # Convert the loaded Q-table to a defaultdict if it's a regular dict
                if isinstance(loaded_q_table, dict):
                    for state, actions in loaded_q_table.items():
                        q_table[state] = actions
                else:
                    # If it's not a dict, it might be a different format, try to adapt
                    q_table = loaded_q_table
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            # Fallback to an empty Q-table
            q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # Handle different Q-table formats
                if isinstance(q_table, defaultdict) or isinstance(q_table, dict):
                    # Dictionary-based Q-table
                    if state in q_table:
                        action = np.argmax(q_table[state])
                    else:
                        # If state not in Q-table, take a random action
                        action = env.action_space.sample()
                else:
                    # Array-based Q-table (legacy format)
                    try:
                        action = np.argmax(q_table[state])
                    except IndexError:
                        # If state out of bounds, take a random action
                        action = env.action_space.sample()
                
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    else:
        # Initialize expert table
        expert_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Experience replay buffer
        replay_buffer = deque(maxlen=10000)
        
        # 1. Learn from a large number of expert demonstrations
        print("Learning from expert demonstrations...")
        expert_states = set()  # Track unique states seen in expert demos
        expert_returns = []
        
        # Collect expert demonstrations
        for i in range(args.expert_demos):
            if i % 10 == 0:
                print(f"Processing expert demo {i}/{args.expert_demos}")
            
            # Get expert trajectory with random seeds for diversity
            trajectory = env.expert_trajectory(seed=np.random.randint(0, 10000))
            
            # Calculate the total return for this trajectory
            total_return = sum(reward for _, _, reward in trajectory[:-1])  # Exclude the terminal state
            expert_returns.append(total_return)
            
            # Add to replay buffer and update expert table
            for j in range(len(trajectory) - 1):  # Exclude the terminal state
                state, action, _ = trajectory[j]
                next_state, next_action, reward = trajectory[j+1] if j+1 < len(trajectory)-1 else (trajectory[j+1][0], None, 0)
                
                # Add to expert states set
                expert_states.add(state)
                
                # Add experience to replay buffer
                done = j == len(trajectory) - 2
                replay_buffer.append((state, action, reward, next_state, done))
                
                # Direct update to expert Q-table (imitation learning)
                if not done:
                    # Higher learning rate for expert demonstrations
                    expert_table[state][action] += 0.2 * (reward + args.gamma * np.max(expert_table[next_state]) - expert_table[state][action])
                else:
                    expert_table[state][action] += 0.2 * (reward - expert_table[state][action])
        
        print(f"Expert demonstrations complete. Average expert return: {np.mean(expert_returns):.2f}")
        print(f"Number of unique states observed: {len(expert_states)}")
        
        # Initialize Q-table with expert knowledge
        for state in expert_states:
            q_table[state] = expert_table[state].copy()
        
        # 2. Q-learning with prioritized experience replay and exploration bonus
        print("Starting enhanced Q-learning training...")
        epsilon = args.epsilon
        epsilon_min = 0.01
        epsilon_decay = 0.9998  # Slower decay
        
        # Learning rate decay
        alpha = args.alpha
        alpha_min = 0.01
        alpha_decay = 0.9999
        
        # Performance tracking
        returns_history = []
        best_avg_return = -float('inf')
        patience = 0
        
        for episode in range(args.episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            episode_steps = 0
            
            # Collect experience from this episode
            episode_buffer = []
            
            while not done:
                episode_steps += 1
                
                # Epsilon-greedy action selection with expert guidance
                if np.random.rand() < epsilon:
                    # Exploration with bias towards expert actions when available
                    if state in expert_states and np.random.rand() < 0.7:
                        # 70% chance to follow expert policy during exploration
                        action = np.argmax(expert_table[state])
                    else:
                        action = env.action_space.sample()
                else:
                    # Exploitation - use our learned Q-values
                    action = np.argmax(q_table[state])
                
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store in episode buffer
                episode_buffer.append((state, action, reward, next_state, done))
                
                # Add exploration bonus for rarely visited states
                if episode_steps % 5 == 0:  # Every 5 steps, sample from replay buffer
                    if replay_buffer:
                        for _ in range(min(10, len(replay_buffer))):
                            s, a, r, ns, d = random.sample(replay_buffer, 1)[0]
                            
                            # Q-learning update with double learning
                            if not d:
                                best_action = np.argmax(q_table[ns])
                                target = r + args.gamma * q_table[ns][best_action]
                            else:
                                target = r
                            
                            q_table[s][a] += alpha * (target - q_table[s][a])
                
                # Q-learning update for current step
                if not done:
                    best_next_action = np.argmax(q_table[next_state])
                    q_table[state][action] += alpha * (reward + args.gamma * q_table[next_state][best_next_action] - q_table[state][action])
                else:
                    q_table[state][action] += alpha * (reward - q_table[state][action])
                
                state = next_state
                total_reward += reward
            
            # Add episode experience to replay buffer
            replay_buffer.extend(episode_buffer)
            
            # Decay parameters
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            alpha = max(alpha_min, alpha * alpha_decay)
            
            # Track performance
            returns_history.append(total_reward)
            recent_mean = np.mean(returns_history[-100:]) if len(returns_history) >= 100 else np.mean(returns_history)
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}, mean 100-episode return {recent_mean:.2f}, epsilon: {epsilon:.4f}, alpha: {alpha:.4f}")
                
                # Early stopping with patience
                if recent_mean > best_avg_return:
                    best_avg_return = recent_mean
                    patience = 0
                    # Save best model
                    with open("q_table_best.pkl", "wb") as f:
                        pickle.dump(dict(q_table), f)
                else:
                    patience += 1
                
                # If we've reached our target performance, stop early
                if recent_mean > 200 and np.mean(returns_history[-300:]) > 195:
                    print(f"Reached target performance of 200+ - stopping training early")
                    # Load the best model
                    with open("q_table_best.pkl", "rb") as f:
                        saved_q_table = pickle.load(f)
                        q_table = defaultdict(lambda: np.zeros(n_actions))
                        for k, v in saved_q_table.items():
                            q_table[k] = v
                    break
                
                # If no improvement for a long time, reload best model and continue
                if patience >= 20:
                    print("No improvement for 2000 episodes, reloading best model")
                    try:
                        with open("q_table_best.pkl", "rb") as f:
                            saved_q_table = pickle.load(f)
                            q_table = defaultdict(lambda: np.zeros(n_actions))
                            for k, v in saved_q_table.items():
                                q_table[k] = v
                        patience = 0
                    except:
                        pass
        
        print("Training complete!")
        
        # Save final Q-table as a regular dictionary for maximum compatibility
        with open("q_table.pkl", "wb") as f:
            pickle.dump(dict(q_table), f)
        
        # Final evaluation with multiple runs
        print("Running thorough evaluation...")
        eval_episodes = 200
        eval_returns = []
        
        for _ in range(eval_episodes):
            state, done = env.reset()[0], False
            episode_return = 0
            
            while not done:
                action = np.argmax(q_table[state])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
            
            eval_returns.append(episode_return)
        
        print(f"Final evaluation over {eval_episodes} episodes: {np.mean(eval_returns):.2f} +-{np.std(eval_returns):.2f}")
        print(f"Percentage of episodes with return > 200: {sum(r > 200 for r in eval_returns) / len(eval_returns) * 100:.1f}%")

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    
    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteLunarLanderWrapper(gym.make("LunarLander-v3")), main_args.seed, main_args.render_each)
    
    main(main_env, main_args)