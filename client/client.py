import socket
import pygame
import threading
from gui import BattleshipGUI
from ai_agent import DQNAgent

class BattleshipClient:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gui = BattleshipGUI()
        self.ai_agent = DQNAgent(100, 100)  # 10x10 board
        
    def connect(self, host='localhost', port=5555):
        try:
            self.client.connect((host, port))
            threading.Thread(target=self.receive).start()
            self.gui.run()
        except Exception as e:
            print(f"Error connecting to server: {e}")

    def receive(self):
        while True:
            try:
                message = self.client.recv(1024).decode('utf-8')
            except Exception as e:
                print(f"Error receiving message: {e}")
                break
    
    def receive(self):
    while True:
        try:
            message = self.client.recv(1024).decode('utf-8')
            self.gui.update(message)
        except Exception as e:
            print(f"Error receiving message: {e}")
            break