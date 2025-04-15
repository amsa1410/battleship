import numpy as np

class BattleshipGame:
    def __init__(self, size=10):
        self.board = np.zeros((self.size, self.size))
        self.ships = [5, 4, 3, 3, 2]
        self._place_ships()
        
    def _place_ships(self):
        for ship in self.ships:
            placed = False
            while not placed:
                x, y = np.random.randint(0, 10), np.random.randint(0, 10)
                orientation = np.random.choice(['H', 'V'])
                if self._is_valid_placement(x, y, ship, orientation):
                    if orientation == 'H':
                        self.board[y, x:x+ship] = 1
                    else:
                        self.board[y:y+ship, x] = 1
                    placed = True

        
    def make_move(self, x, y):
        if self.board[y, x] == 1:
            self.board[y, x] = 2  # Hit
            return "HIT"
        return "MISS"
