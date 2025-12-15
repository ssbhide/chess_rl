import chess
import gymnasium as gym
import numpy as np

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}


class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()

        # Observation: 12 x 8 x 8 (piece planes)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(12, 8, 8), dtype=np.int8
        )

        # Action: from_square * 64 + to_square
        self.action_space = gym.spaces.Discrete(64 * 64)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.board.reset()
        return self._get_obs(), {}

    def step(self, action):
        from_sq = action // 64
        to_sq = action % 64
        move = chess.Move(from_sq, to_sq)

        reward = 0.0
        done = False

        # Illegal move penalty
        if move not in self.board.legal_moves:
            reward = -0.05
            done = True
            return self._get_obs(), reward, done, False, {}

        # ---- Reward shaping ----
        material_before = self._material_count()

        self.board.push(move)

        material_after = self._material_count()
        material_delta = material_after - material_before
        reward += 0.01 * material_delta

        # Bonus for giving check
        if self.board.is_check():
            reward += 0.02

        # ---- Terminal conditions ----
        if self.board.is_checkmate():
            reward += 1.0
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

    def _material_count(self):
        """
        Returns material advantage from White's perspective.
        Positive = White is ahead.
        """
        total = 0
        for piece_type, value in PIECE_VALUES.items():
            total += len(self.board.pieces(piece_type, chess.WHITE)) * value
            total -= len(self.board.pieces(piece_type, chess.BLACK)) * value
        return total
