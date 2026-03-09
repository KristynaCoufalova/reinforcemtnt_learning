#!/usr/bin/env python3
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
# If you add more arguments, ReCodEx will keep them with your default values.


class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 10 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states: int = 11
    actions: int = 4
    action_labels: list[str] = ["↑", "→", "↓", "←"]

    @staticmethod
    def step(state: int, action: int) -> list[tuple[float, float, int]]:
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability: float, state: int, action: int) -> tuple[float, float, int]:
        state += (state >= 5)
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not (new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        state -= (state >= 5)
        return (probability, +1 if state == 10 else -100 if state == 6 else 0, state)


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> tuple[list[float] | np.ndarray, list[int] | np.ndarray]:
    # Start with zero value function and "go North" policy
    value_function = np.zeros(GridWorld.states, dtype=np.float64)  # Using 64-bit floats as required
    policy = np.zeros(GridWorld.states, dtype=int)

    # Cache transitions to avoid recalculation
    transitions = {}
    for state in range(GridWorld.states):
        transitions[state] = {}
        for action in range(GridWorld.actions):
            transitions[state][action] = GridWorld.step(state, action)

    # Policy iteration algorithm
    for _ in range(args.steps):
        # Policy Evaluation - solve system of linear equations
        # For each state s, we have: v(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv(s')]
        # With a deterministic policy, this becomes: v(s) = Σ_{s',r} p(s',r|s,π(s))[r + γv(s')]
        # We can rewrite as: v(s) = R(s) + γ Σ_{s'} P(s'|s,π(s))v(s')
        # Which gives us a system of linear equations: v = R + γPv, or (I - γP)v = R

        # Build coefficient matrix A (I - γP) and right-hand side b (rewards)
        A = np.eye(GridWorld.states, dtype=np.float64)  # Identity matrix
        b = np.zeros(GridWorld.states, dtype=np.float64)  # Reward vector

        for state in range(GridWorld.states):
            action = policy[state]
            for prob, reward, next_state in transitions[state][action]:
                # Add reward contribution to b
                b[state] += prob * reward
                
                # Add transition probability contribution to A
                # We're building (I - γP), so we subtract from the identity matrix
                A[state, next_state] -= args.gamma * prob

        # Solve the linear system Av = b for value function v
        value_function = np.linalg.solve(A, b)

        # Policy Improvement
        policy_stable = True
        for state in range(GridWorld.states):
            old_action = policy[state]
            
            # Calculate value for each action
            action_values = np.zeros(GridWorld.actions, dtype=np.float64)
            for action in range(GridWorld.actions):
                for prob, reward, next_state in transitions[state][action]:
                    action_values[action] += prob * (reward + args.gamma * value_function[next_state])
            
            # Choose best action using tolerance-based argmax
            best_action = argmax_with_tolerance(action_values)
            
            # Update policy
            if old_action != best_action:
                policy_stable = False
                policy[state] = best_action
        
        # Early termination if policy is stable
        if policy_stable:
            break

    # Convert numpy arrays to lists for return
    return value_function.tolist(), policy.tolist()


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(main_args)

    # Print results
    for r in range(3):
        for c in range(4):
            state = 4 * r + c
            state -= (state >= 5)
            print("        " if r == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if r == 1 and c == 1 else GridWorld.action_labels[policy[state]], end="")
        print()