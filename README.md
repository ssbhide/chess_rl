# ♟️ Chess RL Agent

A reinforcement learning–based chess engine trained via **self-play** using **Deep Q-Learning (DQN)** and **convolutional neural networks**.

This project implements a full RL pipeline for chess, including:

* A Gym-style chess environment
* A CNN-based Q-network
* Experience replay and target networks
* Self-play training
* Legal-move masking

The agent learns purely from game outcomes, without human data.

---

## 📌 Features

* ♞ **Custom Chess Environment**

  * Built on `python-chess`
  * Gymnasium-compatible API
  * Tensor-based board representation

* 🧠 **Deep Q-Network (DQN)**

  * CNN processes board state
  * Outputs Q-values for all possible moves
  * Target network for stable training

* 🔁 **Self-Play Training**

  * Agent plays against itself
  * Experience replay buffer
  * ε-greedy exploration

* 🚫 **Illegal Move Masking**

  * Ensures the agent only selects legal chess moves

---

## 🗂️ Project Structure

```
chess_rl/
├── env/
│   └── chess_env.py        # Gym-style chess environment
├── agents/
│   └── dqn_agent.py        # DQN agent + replay buffer
├── models/
│   └── cnn.py              # CNN Q-network
├── train.py                # Self-play training loop
├── eval.py                 # (Optional) Evaluation script
├── requirements.txt
└── README.md
```

---

## 🧠 State & Action Representation

### State (Observation)

* Shape: **(12, 8, 8)**
* 12 feature planes:

  * 6 white piece types
  * 6 black piece types
* Each plane is a binary 8×8 grid

### Action Space

* **4096 discrete actions**
* Encoded as `(from_square × 64 + to_square)`
* Illegal actions are masked during action selection

---

## 🏗️ Setup Instructions

### 1️⃣ Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Verify installation

```bash
python - <<EOF
import chess, torch
print("Setup OK")
EOF
```

---

## 🚀 Training the Agent

Run:

```bash
python train.py
```

Training details:

* Default: **50,000 episodes**
* Uses self-play
* Periodically saves model checkpoints
* Prints average reward and exploration rate

Example output:

```
Episode    500 | Avg Reward (last 100): -0.210 | Epsilon: 0.94
```

> ⚠️ Early rewards will be negative — this is expected.

---

## ⚡ GPU Acceleration (Recommended)

If you have an NVIDIA GPU:

1. Install CUDA-enabled PyTorch from:
   [https://pytorch.org](https://pytorch.org)

2. Verify CUDA is active:

```bash
python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF
```

The agent will automatically use the GPU if available.

---

## 📈 Expected Learning Progress

Over time, the agent learns to:

* Avoid illegal moves
* Survive longer games
* Capture hanging pieces
* Execute simple checkmates

This project is designed for **learning and experimentation**, not to compete with Stockfish.

---

## 🔬 Future Improvements

* AlphaZero-style **MCTS + policy/value network**
* PPO or Actor-Critic methods
* Endgame-only curriculum learning
* Parallel self-play environments
* Web-based UI to play against the agent
* Evaluation vs Stockfish

---

## 🎓 Why This Project Matters

This project demonstrates:

* Reinforcement learning fundamentals
* Neural network design
* Complex action-space handling
* Clean system architecture
* Applied AI beyond toy problems

It is suitable as:

* A resume project
* A research starter
* A foundation for stronger chess engines

---

## 📜 License

MIT License
