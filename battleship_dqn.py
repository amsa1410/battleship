import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class BattleshipEnv:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.ship_sizes = [5, 4, 3, 3, 2]  # Carrier, Battleship, Submarine, Cruiser, Destroyer
        self.reset()

    def reset(self):
        self.target_board = np.zeros((self.board_size, self.board_size))  # -1: miss, 0: unknown, 1: hit
        self.ship_board = np.zeros((self.board_size, self.board_size))     # 1: ship, 0: water
        self._place_ships()
        self.hits = 0
        self.misses = 0
        self.total_shots = 0
        self.ships_remaining = sum(self.ship_sizes)
        return self._get_state()

    def _place_ships(self):
        for size in self.ship_sizes:
            placed = False
            while not placed:
                orientation = random.choice(['horizontal', 'vertical'])
                if orientation == 'horizontal':
                    x = random.randint(0, self.board_size - size)
                    y = random.randint(0, self.board_size - 1)
                    if np.all(self.ship_board[y, x:x+size] == 0):
                        self.ship_board[y, x:x+size] = 1
                        placed = True
                else:
                    x = random.randint(0, self.board_size - 1)
                    y = random.randint(0, self.board_size - size)
                    if np.all(self.ship_board[y:y+size, x] == 0):
                        self.ship_board[y:y+size, x] = 1
                        placed = True

    def _get_state(self):
        return self.target_board.flatten()

    def step(self, action):
        x = action % self.board_size
        y = action // self.board_size

        if self.target_board[y, x] != 0:
            return self._get_state(), -1, False, {}  # Penalize repeated moves

        if self.ship_board[y, x] == 1:
            self.target_board[y, x] = 1
            self.hits += 1
            self.ships_remaining -= 1
            reward = 10
        else:
            self.target_board[y, x] = -1
            self.misses += 1
            reward = -2

        self.total_shots += 1
        done = self.ships_remaining == 0
        if done:
            reward = 100  # Win reward

        return self._get_state(), reward, done, {}


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        return torch.argmax(self.model(state)).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def visualize_board(env):
    plt.clf()
    cmap = mcolors.ListedColormap(['blue', 'lightblue', 'red']) 
    plt.imshow(env.target_board, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 0, 1], label='Miss (Blue) / Water (Light Blue) / Hit (Red)')
    plt.title(f"Shots: {env.total_shots} | Ships Left: {env.ships_remaining}")
    plt.grid(True, color='black', linewidth=0.5)
    plt.pause(0.1)  

def train_and_play(episodes=100, board_size=10):
    env = BattleshipEnv(board_size)
    agent = DQNAgent(board_size * board_size, board_size * board_size)
    
    plt.ion()  
    for e in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, _, done, _ = env.step(action)
            visualize_board(env)  # Update the plot
            if done:
                print(f"Episode {e + 1}/{episodes} | Shots: {env.total_shots}")
                break
            agent.replay(32)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train_and_play(episodes=50) 