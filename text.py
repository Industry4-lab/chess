
import chess
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Neural Network for DQN
class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 128)  # Input: 64 (board squares), hidden layer 128
        self.fc2 = nn.Linear(128, 128)  # Hidden layer
        self.fc3 = nn.Linear(128, 4672)  # Output: Over-approximation of all possible chess moves

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# DQN Algorithm
class DQN:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.model = DQNNetwork()
        self.target_model = DQNNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, board):
        if random.random() < self.epsilon:
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves).uci()  # Select a random legal move
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.model(state)
                return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_state).max(1)[0]
        target_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

# Function to get a valid move from the user in UCI format
def get_user_move_uci(board):
    while True:
        try:
            user_input = input("Enter your move in UCI format (e.g., e2e4, g1f3): ").strip()
            move = chess.Move.from_uci(user_input)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move, please try again.")
        except ValueError:
            print("Invalid move format, please try again.")

# Function to display the board
def display_board(board):
    print(board)

# Main function to handle the game loop
def play_game():
    board = chess.Board()
    dqn = DQN(state_dim=64, action_dim=4672)

    while not board.is_game_over():
        display_board(board)

        # Get move from user for White
        print("White's turn:")
        user_move = get_user_move_uci(board)
        board.push(user_move)

        if board.is_game_over():
            break

        if board.is_check():
            print("Black is in CHECK.")

        display_board(board)

        # Generate and make a DQN move for Black
        print("Black's turn:")
        state = np.array(board.board_fen()).reshape(-1)  # Simplified representation
        black_move = dqn.select_action(state, board)
        print(f"Black plays: {black_move}")
        board.push(chess.Move.from_uci(black_move))

        if board.is_game_over():
            break

        if board.is_check():
            print("White is in CHECK.")

        # DQN Training Step
        next_state = np.array(board.board_fen()).reshape(-1)
        reward = 0  # Define a reward function here
        done = board.is_game_over()
        dqn.replay_buffer.push(state, black_move, reward, next_state, done)
        dqn.train()

        if done:
            dqn.update_target_network()

    # Display the final board and result
    display_board(board)
    if board.is_checkmate():
        print("CHECKMATE!")
    elif board.is_stalemate():
        print("Stalemate.")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw due to seventy-five-move rule.")
    elif board.is_fivefold_repetition():
        print("Draw due to fivefold repetition.")
    else:
        print("Game over!")
        print(f"Result: {board.result()}")

# Run the game
if __name__ == "__main__":
    play_game()
