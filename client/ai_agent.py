import torch
from ai.model import DQN

class AIClient:
    def __init__(self):
        self.model = DQN(100, 100)  
        self.model.load_state_dict(torch.load('ai/pretrained/battleship_dqn.pth'))

    def predict_move(self, board_state):
        state = torch.FloatTensor(board_state.flatten())
        return self.model(state).argmax().item()  