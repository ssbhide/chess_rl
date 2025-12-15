import numpy as np
import chess
import torch
from tqdm import trange

from env.chess_env import ChessEnv
from models.cnn import ChessCNN


def get_legal_moves_mask(board):
    mask = np.zeros(64 * 64, dtype=np.int8)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        mask[action] = 1
    return mask


def load_model(path, device="cpu"):
    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def select_agent_move(model, state, board, device):
    mask = get_legal_moves_mask(board)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_vals = model(state_t)[0].cpu().numpy()

    q_vals[mask == 0] = -1e9
    action = int(np.argmax(q_vals))
    return chess.Move(action // 64, action % 64)


def select_random_move(board):
    return np.random.choice(list(board.legal_moves))


def play_game(model, device, max_moves=200):
    env = ChessEnv()
    state, _ = env.reset()
    board = env.board

    for _ in range(max_moves):
        if board.is_game_over():
            break

        # Agent move (White)
        move = select_agent_move(model, state, board, device)
        board.push(move)
        state = env._get_obs()

        if board.is_game_over():
            break

        # Random move (Black)
        board.push(select_random_move(board))
        state = env._get_obs()

    result = board.result()
    if result == "1-0":
        return 1.0
    elif result == "1/2-1/2":
        return 0.5
    else:
        return 0.0


def evaluate(model_path, num_games=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)

    scores = []
    for _ in trange(num_games):
        score = play_game(model, device)
        scores.append(score)

    scores = np.array(scores)

    print("\nEvaluation results")
    print("------------------")
    print(f"Games played : {num_games}")
    print(f"Wins         : {np.sum(scores == 1.0)}")
    print(f"Draws        : {np.sum(scores == 0.5)}")
    print(f"Losses       : {np.sum(scores == 0.0)}")
    print(f"Avg score    : {scores.mean():.3f}")


if __name__ == "__main__":
    evaluate("chess_rl_chess_dqn_ep10000.pt", num_games=100)
