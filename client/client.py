import socket
import pygame
from ai_agent import DQNAgent

class BattleshipClient:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gui = BattleshipGUI()
        self.ai_agent = DQNAgent(100, 100)  # 10x10 board
        
    def connect(self, host='localhost', port=5555):
        self.client.connect((host, port))
        # Start game thread
        threading.Thread(target=self.receive).start()
        self.gui.run()

    def receive(self):
        while True:
            try:
                message = self.client.recv(1024).decode('utf-8')
                # Update GUI based on server messages
            except:
                break