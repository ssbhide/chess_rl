import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import ChessCNN


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500_000,
        batch_size=64,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = ChessCNN().to(self.device)
        self.target_net = ChessCNN().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0

    def select_action(self, state, legal_moves_mask):
        """
        state: numpy array (12, 8, 8)
        legal_moves_mask: numpy array (4096,), 1 if legal else 0
        """
        self.steps += 1

        # Epsilon decay
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 / self.epsilon_decay),
        )

        if random.random() < self.epsilon:
            legal_actions = np.where(legal_moves_mask == 1)[0]
            if len(legal_actions) == 0:
                return None

            return int(np.random.choice(legal_actions))

        state_t = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.q_net(state_t)[0].cpu().numpy()

        q_values[legal_moves_mask == 0] = -1e9
        return int(np.argmax(q_values))

    def store(self, *transition):
        self.replay.push(*transition)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
