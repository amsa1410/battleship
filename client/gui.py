import sys
import pygame

class BattleshipGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.font = pygame.font.SysFont('Arial', 24)
        self.board = [[0 for _ in range(10)] for _ in range(10)]
        
    def draw_board(self, board):
        for y in range(10):
            for x in range(10):
                rect = pygame.Rect(x*50+100, y*50+100, 50, 50)
                color = (0, 0, 255) if self.board[y][x] == 0 else (255, 0, 0)
                pygame.draw.rect(self.screen,color, rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = (event.pos[0]-100)//50, (event.pos[1]-100)//50
                    if 0 <= x < 10 and 0 <= y < 10:
                        return x, y
                    
            self.screen.fill((0,0,0))
            self.draw_board()
            pygame.display.flip()
        pygame.quit()
        return None, None
    
    def update(self, message):
    print(f"Updating GUI with message: {message}")