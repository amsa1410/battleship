import tkinter as tk
from tkinter import messagebox
import random


# Constants
GRID_SIZE = 10
CELL_SIZE = 40
SHIP_SIZES = [5, 4, 3, 3, 2]

# Colors
COLOR_EMPTY = "white"
COLOR_SHIP = "gray"
COLOR_HIT = "red"
COLOR_MISS = "blue"

class BattleshipGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Battleship Game")

        # Initialize player and computer grids
        self.player1_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.player2_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Place ships randomly
        self.place_ships(self.player1_grid)
        self.place_ships(self.player2_grid)

        # Ask the user if they want to play against the computer or another player
        self.mode = self.ask_mode()

        # Create player canvases
        self.player1_canvas = self.create_grid(0, 0, self.player1_grid, clickable=False)
        self.player2_canvas = self.create_grid(GRID_SIZE * CELL_SIZE + 50, 0, self.player2_grid, clickable=True)

        # Initialize turn (Player 1 starts)
        self.current_player = 1

    def ask_mode(self):
        response = messagebox.askyesno("Game Mode", "Do you want to play against the computer?")
        return "computer" if response else "player"

    def create_grid(self, x, y, grid, clickable):
        canvas = tk.Canvas(self.root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg=COLOR_EMPTY)
        canvas.place(x=x, y=y)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1, y1 = j * CELL_SIZE, i * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_EMPTY, outline="black")

                if grid[i][j] == 1:
                    canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_SHIP, outline="black")

        if clickable:
            canvas.bind("<Button-1>", lambda event, g=grid: self.on_click(event, g))

        return canvas

    def place_ships(self, grid):
        for size in SHIP_SIZES:
            placed = False
            while not placed:
                orientation = random.choice(["horizontal", "vertical"])
                if orientation == "horizontal":
                    row = random.randint(0, GRID_SIZE - 1)
                    col = random.randint(0, GRID_SIZE - size)
                    if all(grid[row][col + i] == 0 for i in range(size)):
                        for i in range(size):
                            grid[row][col + i] = 1
                        placed = True
                else:
                    row = random.randint(0, GRID_SIZE - size)
                    col = random.randint(0, GRID_SIZE - 1)
                    if all(grid[row + i][col] == 0 for i in range(size)):
                        for i in range(size):
                            grid[row + i][col] = 1
                        placed = True

    def on_click(self, event, grid):
        if self.current_player == 2 and self.mode == "computer":
            return  # Computer's turn, ignore player clicks

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if grid[row][col] == 1:
            self.player2_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_HIT, outline="black")
            grid[row][col] = -1  # Mark as hit
            if self.check_win(grid):
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.root.quit()
        elif grid[row][col] == 0:
            self.player2_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_MISS, outline="black")
            grid[row][col] = -2  # Mark as miss

        # Switch turns
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        if self.mode == "computer" and self.current_player == 2:
            self.root.after(1000, self.computer_turn)

    def computer_turn(self):
        while True:
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            if self.player1_grid[row][col] >= 0:  # Cell not already hit or missed
                break

        if self.player1_grid[row][col] == 1:
            self.player1_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_HIT, outline="black")
            self.player1_grid[row][col] = -1  # Mark as hit
            if self.check_win(self.player1_grid):
                messagebox.showinfo("Game Over", "Computer wins!")
                self.root.quit()
        else:
            self.player1_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_MISS, outline="black")
            self.player1_grid[row][col] = -2  # Mark as miss

        # Switch back to player's turn
        self.current_player = 1

    def check_win(self, grid):
        for row in grid:
            if 1 in row:
                return False
        return True

if __name__ == "__main__":
    root = tk.Tk()
    game = BattleshipGame(root)
    root.mainloop()

