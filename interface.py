import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random

# Constants
GRID_SIZE = 10
CELL_SIZE = 40
SHIP_SIZES = [5, 4, 3, 3, 2]

# Colors
COLOR_EMPTY = "white"
COLOR_SHIP = "gray"
COLOR_HIT = "#bc1823"
COLOR_MISS = "#4344a0"
BUTTON_COLOR = "#4344a0" 

class StartMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Battleship - Main Menu")
        
        # Get the window size
        window_width = self.root.winfo_screenwidth()
        window_height = self.root.winfo_screenheight()
        
        # Load and resize background image to match window size
        bg_image = Image.open("image de fond.png")
        bg_image = bg_image.resize((window_width, window_height))
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        
        # Create and place background label
        bg_label = tk.Label(root, image=self.bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Create frame for buttons
        button_frame = tk.Frame(root)
        button_frame.configure(bg='SystemButtonFace')
        button_frame.place(relx=0.2, rely=0.8, anchor='center')
        
        # Create custom buttons using the create_rounded_button method
        vs_ai_button = self.create_rounded_button(button_frame, "Play vs AI", self.start_ai_game)
        vs_ai_button.pack(pady=10)
        
        vs_player_button = self.create_rounded_button(button_frame, "Play vs Player", self.start_player_game)
        vs_player_button.pack(pady=10)

    def create_rounded_button(self, parent, text, command):
        frame = tk.Frame(parent, bg='SystemButtonFace')
        
        # Create a canvas for the rounded button
        canvas = tk.Canvas(frame, height=50, width=200,
                          bg='SystemButtonFace', highlightthickness=0)
        canvas.pack()

        # Create rounded rectangle using arcs and lines
        def create_rounded_rectangle(x1, y1, x2, y2, radius):
            points = [x1+radius, y1,
                    x2-radius, y1,
                    x2, y1,
                    x2, y1+radius,
                    x2, y2-radius,
                    x2, y2,
                    x2-radius, y2,
                    x1+radius, y2,
                    x1, y2,
                    x1, y2-radius,
                    x1, y1+radius,
                    x1, y1]
            return canvas.create_polygon(points, smooth=True, 
                                       fill='#4344a0', 
                                       outline='#4344a0',
                                       stipple='gray50')

        # Create the button shape
        create_rounded_rectangle(2, 2, 198, 48, 15)

        # Add text
        canvas.create_text(100, 25, text=text,
                          fill='white',
                          font=('Arial', 12, 'bold'))
        
        # Bind click event
        canvas.bind('<Button-1>', lambda e: command())
        canvas.bind('<Enter>', lambda e: canvas.configure(cursor='hand2'))
        
        return frame

    def start_ai_game(self):
        self.root.withdraw()  # Hide the main menu
        game_window = tk.Toplevel()
        BattleshipGame(game_window, "computer")
        
    def start_player_game(self):
        self.root.withdraw()  # Hide the main menu
        game_window = tk.Toplevel()
        BattleshipGame(game_window, "player")

class BattleshipGame:
    def __init__(self, root, mode):
        self.root = root
        self.root.title("Battleship Game")
        self.mode = mode
        self.placing_ships = True
        self.current_ship_index = 0
        self.current_orientation = "horizontal"
        self.current_player = 1  # Add this line to initialize current_player

        # Initialize grids
        self.player1_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.player2_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Place AI/Player 2 ships randomly but don't show them
        self.place_ships(self.player2_grid)

        # Create canvases
        self.player1_canvas = self.create_grid(0, 0, self.player1_grid, clickable=True)
        self.player2_canvas = self.create_grid(GRID_SIZE * CELL_SIZE + 50, 0, self.player2_grid, clickable=False)

        # Add ship placement instructions
        self.instruction_label = tk.Label(
            root,
            text=f"Place your {SHIP_SIZES[0]}-cell ship. Press 'R' to rotate",
            font=('Arial', 12)
        )
        self.instruction_label.place(x=10, y=GRID_SIZE * CELL_SIZE + 10)

        # Bind rotation key
        root.bind('r', self.toggle_orientation)
        
        # Start with ship placement phase
        self.player1_canvas.bind("<Button-1>", self.place_player_ship)

    def toggle_orientation(self, event):
        if self.placing_ships:
            self.current_orientation = "vertical" if self.current_orientation == "horizontal" else "horizontal"

    def place_player_ship(self, event):
        if not self.placing_ships:
            return

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        ship_size = SHIP_SIZES[self.current_ship_index]

        # Check if placement is valid
        if self.is_valid_placement(row, col, ship_size, self.current_orientation):
            # Place the ship
            self.place_ship(row, col, ship_size, self.current_orientation)
            self.current_ship_index += 1

            if self.current_ship_index < len(SHIP_SIZES):
                # Update instruction for next ship
                self.instruction_label.config(
                    text=f"Place your {SHIP_SIZES[self.current_ship_index]}-cell ship. Press 'R' to rotate"
                )
            else:
                # All ships placed, start the game
                self.placing_ships = False
                self.instruction_label.config(text="Game Started - Your Turn")
                self.player1_canvas.unbind("<Button-1>")
                self.player2_canvas.bind("<Button-1>", lambda e: self.on_click(e, self.player2_grid))
                self.root.unbind('r')

    def is_valid_placement(self, row, col, size, orientation):
        if orientation == "horizontal":
            if col + size > GRID_SIZE:
                return False
            return all(self.player1_grid[row][col + i] == 0 for i in range(size))
        else:
            if row + size > GRID_SIZE:
                return False
            return all(self.player1_grid[row + i][col] == 0 for i in range(size))

    def place_ship(self, row, col, size, orientation):
        if orientation == "horizontal":
            for i in range(size):
                self.player1_grid[row][col + i] = 1
                self.player1_canvas.create_rectangle(
                    (col + i) * CELL_SIZE, row * CELL_SIZE,
                    (col + i + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                    fill=COLOR_SHIP, outline="black"
                )
        else:
            for i in range(size):
                self.player1_grid[row + i][col] = 1
                self.player1_canvas.create_rectangle(
                    col * CELL_SIZE, (row + i) * CELL_SIZE,
                    (col + 1) * CELL_SIZE, (row + i + 1) * CELL_SIZE,
                    fill=COLOR_SHIP, outline="black"
                )

    def create_grid(self, x, y, grid, clickable):
        canvas = tk.Canvas(self.root, width=GRID_SIZE * CELL_SIZE, 
                         height=GRID_SIZE * CELL_SIZE, bg=COLOR_EMPTY)
        canvas.place(x=x, y=y)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1, y1 = j * CELL_SIZE, i * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                canvas.create_rectangle(x1, y1, x2, y2, 
                                     fill=COLOR_EMPTY, outline="black")

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

        # Check if cell was already hit
        if grid[row][col] < 0:
            return

        hit = False
        if grid[row][col] == 1:
            self.player2_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_HIT, outline="black")
            grid[row][col] = -1  # Mark as hit
            hit = True
            
            if self.check_win(grid):
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.root.quit()
        else:
            self.player2_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_MISS, outline="black")
            grid[row][col] = -2  # Mark as miss
            # Switch turns only on miss
            self.current_player = 3 - self.current_player  # Toggle between 1 and 2

        # If it's a miss and next player is computer, do computer turn
        if not hit and self.mode == "computer" and self.current_player == 2:
            self.root.after(1000, self.computer_turn)

    def computer_turn(self):
        while True:
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            if self.player1_grid[row][col] >= 0:  # Cell not already hit or missed
                break

        hit = False
        if self.player1_grid[row][col] == 1:
            self.player1_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_HIT, outline="black")
            self.player1_grid[row][col] = -1  # Mark as hit
            hit = True
            
            if self.check_win(self.player1_grid):
                messagebox.showinfo("Game Over", "Computer wins!")
                self.root.quit()
        else:
            self.player1_canvas.create_rectangle(col * CELL_SIZE, row * CELL_SIZE,
                                                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                                                fill=COLOR_MISS, outline="black")
            self.player1_grid[row][col] = -2  # Mark as miss
            # Switch back to player's turn only on miss
            self.current_player = 1

        # If computer hits, play again after a delay
        if hit:
            self.root.after(1000, self.computer_turn)

    def check_win(self, grid):
        for row in grid:
            if 1 in row:
                return False
        return True

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1920x1080")  
    menu = StartMenu(root)
    root.mainloop()

