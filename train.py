import numpy as np
import chess
import torch
import os

from env.chess_env import ChessEnv
from agents.dqn_agent import DQNAgent


def get_legal_moves_mask(board: chess.Board):
    mask = np.zeros(64 * 64, dtype=np.int8)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        mask[action] = 1
    return mask


def select_random_move(board):
    return np.random.choice(list(board.legal_moves))


def train(
    num_episodes=50_000,
    start_episode=0,
    resume_from=None,
    target_update_freq=1_000,
    max_moves_per_game=200,
):
    env = ChessEnv()
    agent = DQNAgent()

    # 🔹 Resume from checkpoint
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint: {resume_from}")
        agent.q_net.load_state_dict(
            torch.load(resume_from, map_location=agent.device)
        )
        agent.target_net.load_state_dict(agent.q_net.state_dict())

        # 🔧 Fix epsilon resume
        agent.epsilon = max(
            agent.epsilon_end,
            agent.epsilon * (agent.epsilon_decay ** start_episode),
        )

    episode_rewards = []

    for episode in range(start_episode + 1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_moves_per_game):
            board = env.board

            # ---- Agent (White) ----
            legal_mask = get_legal_moves_mask(board)

            if legal_mask.sum() == 0:
                break  # game over, no legal moves

            action = agent.select_action(state, legal_mask)
            if action is None:
                break

            next_state, reward, done, _, _ = env.step(action)

            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

            # ---- Random Opponent (Black) ----
            if not board.is_game_over():
                board.push(select_random_move(board))
                state = env._get_obs()

        episode_rewards.append(total_reward)

        if episode % target_update_freq == 0:
            agent.update_target()

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode {episode:6d} | "
                f"Avg Reward (last 100): {avg_reward:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        if episode % 5_000 == 0:
            path = f"chess_dqn_ep{episode}.pt"
            print(f"Saving model to {path}")
            torch.save(agent.q_net.state_dict(), path)


if __name__ == "__main__":
    train(
        num_episodes=50_000,
        start_episode=0,
        resume_from="chess_dqn_ep5000.pt",
    )
