import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class BattleshipEnv:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.ship_sizes = [5, 4, 3, 3, 2]  # Carrier, Battleship, Submarine, Cruiser, Destroyer
        self.ships = []  # Track ship positions and hits
        self.reset()

    def reset(self):
        self.target_board = np.zeros((self.board_size, self.board_size))
        self.ship_board = np.zeros((self.board_size, self.board_size))
        self.ships = []  # List of [size, hits, positions] for each ship
        self._place_ships()
        self.hits = 0
        self.misses = 0
        self.total_shots = 0
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
                        positions = [(y, x+i) for i in range(size)]
                        self.ships.append([size, 0, positions])
                        placed = True
                else:
                    x = random.randint(0, self.board_size - 1)
                    y = random.randint(0, self.board_size - size)
                    if np.all(self.ship_board[y:y+size, x] == 0):
                        self.ship_board[y:y+size, x] = 1
                        positions = [(y+i, x) for i in range(size)]
                        self.ships.append([size, 0, positions])
                        placed = True

    def _get_state(self):
        # Combine target board with proximity hints
        state = np.zeros((self.board_size, self.board_size, 2))
        state[:,:,0] = self.target_board
        # Add proximity channel (1 if adjacent to a hit)
        hits_y, hits_x = np.where(self.target_board == 1)
        for y, x in zip(hits_y, hits_x):
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.board_size and 0 <= nx < self.board_size:
                    state[ny,nx,1] = 1
        return state.flatten()

    def step(self, action):
        x = action % self.board_size
        y = action // self.board_size

        if self.target_board[y, x] != 0:
            return self._get_state(), -10, False, {}  # Keep small penalty for repeated moves

        reward = -100  # Increased penalty for missing (hitting water)
        ship_hit = None

        if self.ship_board[y, x] == 1:
            self.target_board[y, x] = 1
            self.hits += 1
            reward = 500  # Keep same reward for hitting a ship part
            
            # Check which ship was hit and update its status
            for ship in self.ships:
                if (y, x) in ship[2]:
                    ship[1] += 1  # Increment hits on this ship
                    if ship[1] == ship[0]:  # Ship sunk
                        reward = 1000  # Keep same reward for sinking a ship
                    break

        else:
            self.target_board[y, x] = -1
            self.misses += 1

        self.total_shots += 1
        done = all(ship[1] == ship[0] for ship in self.ships)  # All ships sunk
        

        return self._get_state(), reward, done, {}

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Increased memory size
        self.gamma = 0.99  # Increased discount factor
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.hit_memory = deque(maxlen=1000)  # Special memory for successful hits

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        if reward >= 500:  # Store successful hits in special memory (changed from 50)
            self.hit_memory.append((state, action, reward, next_state, done))
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
            # Mask unavailable actions
            mask = torch.ones(self.action_size) * float('-inf')
            mask[list(available_actions)] = 0
            action_values = action_values + mask
            
            # Get top 3 actions
            top_actions = torch.topk(action_values, min(3, len(available_actions))).indices[0]
            # Choose randomly from top actions
            return random.choice(top_actions.tolist())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Mix regular experiences with successful hits
        hit_size = min(batch_size // 4, len(self.hit_memory))
        reg_size = batch_size - hit_size
        
        reg_batch = random.sample(self.memory, reg_size)
        hit_batch = random.sample(self.hit_memory, hit_size) if hit_size > 0 else []
        minibatch = reg_batch + hit_batch

        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def visualize_board(env, episode, total_reward, hits):
    plt.clf()
    # Create a 2x1 subplot
    plt.subplot(1, 2, 1)
    # Show target board (shots)
    cmap = mcolors.ListedColormap(['blue', 'lightblue', 'red'])
    plt.imshow(env.target_board, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'Target Board (Episode {episode})')
    plt.colorbar(ticks=[-1, 0, 1], label='Miss (Blue) / Unknown (Light Blue) / Hit (Red)')
    
    # Show actual ship positions
    plt.subplot(1, 2, 2)
    ship_cmap = mcolors.ListedColormap(['white', 'grey'])
    plt.imshow(env.ship_board, cmap=ship_cmap)
    plt.title('Ship Positions')
    
    # Add stats as text
    plt.suptitle(f'Episode: {episode} | Score: {total_reward} | Hits: {hits}/{env.hits + env.misses}')
    plt.pause(0.1)  # Pause to show the board

def train_and_play(episodes=1000, board_size=10):
    plt.ion()  # Enable interactive mode
    env = BattleshipEnv(board_size)
    state_size = board_size * board_size * 2
    action_size = board_size * board_size
    agent = DQNAgent(state_size, action_size)
    batch_size = 64  # Increased batch size
    
    scores = []
    hits_per_episode = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        hits_this_episode = 0
        available_actions = set(range(action_size))
        
        while len(available_actions) > 0:
            action = agent.act(state, available_actions)
            available_actions.remove(action)
            
            next_state, reward, done, _ = env.step(action)
            if reward >= 500:
                hits_this_episode += 1
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Visualize the current state
            visualize_board(env, e+1, total_reward, hits_this_episode)
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            if done:
                break
        
        if e % 5 == 0:
            agent.update_target_model()
            
        scores.append(total_reward)
        hits_per_episode.append(hits_this_episode)
        
        # Modified print statement to show total shots taken
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Total Shots: {env.total_shots}")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train_and_play(episodes=1000)