# Reinforcement Learning Implementations

A collection of reinforcement learning algorithms implemented in Python, progressing from classical tabular methods to modern deep RL. Built as part of coursework using [Gymnasium](https://gymnasium.farama.org/) environments and PyTorch.

---

## Contents

### Classical / Tabular RL

| File | Algorithm | Environment |
|------|-----------|-------------|
| `bandits.py` | Multi-Armed Bandits (ε-greedy, averaging, fixed α) | Custom |
| `exact_policy_iteration.py` | Exact Policy Iteration (linear system solve) | GridWorld |
| `q_learning.py` | Q-Learning (tabular, ε-greedy with decay) | MountainCar |
| `q_learning_tiles.py` | Q-Learning with tile coding | MountainCar |

### Deep RL — Value-Based

| File | Algorithm | Environment |
|------|-----------|-------------|
| `q_network.py` | DQN (experience replay, target network) | CartPole-v1 |
| `dist_c51.py` | C51 — Distributional RL (categorical) | CartPole-v1 |
| `dist_51.py` | C51 variant | CartPole-v1 |
| `dist_qr_dnq.py` | QR-DQN — Quantile Regression DQN | CartPole-v1 |
| `cart_pole_pixels.py` | DQN from raw pixel observations | CartPole (pixels) |
| `gym_cartpole.py` | CartPole baseline | CartPole-v1 |

### Deep RL — Policy Gradient

| File | Algorithm | Environment |
|------|-----------|-------------|
| `reinforce.py` | REINFORCE (Monte Carlo policy gradient) | CartPole-v1 |
| `ddpg.py` | DDPG (actor-critic, continuous actions, OU noise) | Pendulum-v1 |
| `walker.py` | SAC (Soft Actor-Critic, twin critics, entropy tuning) | BipedalWalker-v3 |
| `cheetah.py` | SAC variant | HalfCheetah |
| `car_racing.py` | Policy gradient for continuous control | CarRacing-v2 |
| `lunar_lander.py` | Deep RL | LunarLander-v2 |

### Supporting Utilities

| File | Description |
|------|-------------|
| `sgd_backpropagation.py` | Manual backpropagation implementation |
| `sgd_manual.py` | Manual SGD from scratch |
| `bboxes_utils.py` | Bounding box utilities for vision tasks |
| `ccn_manual.py` | Manual CNN implementation |
| `uppercase.py` | Sequence modelling baseline |

---

## Algorithms Covered

- **Exploration**: ε-greedy, optimistic initialization, Ornstein-Uhlenbeck noise
- **Tabular**: Policy iteration, Q-learning, tile coding
- **Value-based deep RL**: DQN, C51, QR-DQN, experience replay, target networks
- **Policy gradient**: REINFORCE, DDPG, SAC
- **Distributional RL**: Categorical (C51), quantile regression (QR-DQN)

---

## Requirements

```bash
pip install gymnasium torch numpy npfl139
```

> Some environments require additional packages:
> - `pip install gymnasium[box2d]` for BipedalWalker, LunarLander, CarRacing
> - `pip install gymnasium[mujoco]` for HalfCheetah

---

## Usage

Each script is self-contained and accepts command-line arguments. Example:

```bash
# Q-Learning on MountainCar
python q_learning.py --alpha 0.1 --epsilon 0.5 --gamma 0.99

# DQN on CartPole
python q_network.py --batch_size 128 --epsilon 0.3 --hidden_layer_size 128

# DDPG on Pendulum
python ddpg.py --env Pendulum-v1 --noise_sigma 0.2

# SAC on BipedalWalker
python walker.py --env BipedalWalker-v3 --hidden_layer_size 256
```

Run any script with `--help` to see all configurable hyperparameters.

---

## Notes

- Scripts use the `npfl139` evaluation wrapper from the [NPFL139 course](https://ufal.mff.cuni.cz/courses/npfl139) at MFF UK
- GPU is used automatically when available (CUDA)
- All experiments use seeded environments for reproducibility
