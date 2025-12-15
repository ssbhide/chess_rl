import numpy as np
import chess

from env.chess_env import ChessEnv
from agents.dqn_agent import DQNAgent


def get_legal_moves_mask(board: chess.Board):
    """
    Returns a (4096,) mask where legal moves are 1, illegal are 0
    """
    mask = np.zeros(64 * 64, dtype=np.int8)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        mask[action] = 1
    return mask


def train(
    num_episodes=50_000,
    target_update_freq=1_000,
    max_moves_per_game=200,
):
    env = ChessEnv()
    agent = DQNAgent()

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_moves_per_game):
            board = env.board
            legal_mask = get_legal_moves_mask(board)

            action = agent.select_action(state, legal_mask)
            next_state, reward, done, _, _ = env.step(action)

            agent.store(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target()

        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode {episode:6d} | "
                f"Avg Reward (last 100): {avg_reward:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Save model periodically
        if episode % 5_000 == 0:
            agent_path = f"chess_dqn_ep{episode}.pt"
            print(f"Saving model to {agent_path}")
            import torch

            torch.save(agent.q_net.state_dict(), agent_path)


if __name__ == "__main__":
    train()
