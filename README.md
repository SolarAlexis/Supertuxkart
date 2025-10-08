# 🏎️ SuperTuxKart Reinforcement Learning (RL Project)

This project aims to train an **agent** capable of playing *SuperTuxKart* using **Reinforcement Learning (RL)** techniques.  
It was developed as part of an academic project where students had to design, implement, and evaluate an RL policy to control a kart in the **[SuperTuxKart Gymnasium environment](https://github.com/bpiwowar/pystk2-gymnasium)**  a custom OpenAI Gym-compatible interface built specifically for this challenge.

The training framework is based on **[BBRL (Brain and Behavior Reinforcement Learning)](https://github.com/osigaud/bbrl)**, a modular and PyTorch-compatible library for designing RL agents and experiments.

The objective of this project is to teach an agent to navigate and complete laps in SuperTuxKart efficiently using **model-free RL methods** (e.g., DQN, PPO, SAC, etc.).  
The agent interacts with the `pystk2-gymnasium` environment by receiving visual or numerical observations and returning actions (e.g., steering, acceleration, braking).
