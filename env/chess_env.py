import chess
import gymnasium as gym
import numpy as np

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()

        # Observation: 12x8x8 tensor
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(12, 8, 8), dtype=np.int8
        )

        # Action space: from_square * 64 + to_square
        self.action_space = gym.spaces.Discrete(64 * 64)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.board.reset()
        return self._get_obs(), {}

    def step(self, action):
        from_sq = action // 64
        to_sq = action % 64
        move = chess.Move(from_sq, to_sq)

        reward = 0
        done = False

        if move not in self.board.legal_moves:
            reward = -0.05
            done = True
        else:
            self.board.push(move)

            if self.board.is_checkmate():
                reward = 1.0
                done = True
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                reward = 0.0
                done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((12, 8, 8), dtype=np.int8)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                plane = self._piece_to_plane(piece)
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                obs[plane][row][col] = 1

        return obs

    def _piece_to_plane(self, piece):
        offset = 0 if piece.color == chess.WHITE else 6
        return offset + piece.piece_type - 1

