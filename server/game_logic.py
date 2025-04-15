import numpy as np

class BattleshipGame:
    def __init__(self, size=10):
        self.size = size
        self.ships = [5, 4, 3, 3, 2]
        self.reset()
        
    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self._place_ships()
        
    def _place_ships(self):
        # Same placement logic as before
        pass
        
    def make_move(self, x, y):
        # Return hit/miss/sink status
        pass
