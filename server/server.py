import socket
import threading
from game_logic import BattleshipGame

class BattleshipServer:
    def __init__(self, host='localhost', port=5555):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen()
        self.clients = []
        self.game = BattleshipGame()
        
    def broadcast(self, message):
        for client in self.clients:
            client.send(message.encode('utf-8'))

    def handle_client(self, client):
        while True:
            try:
                message = client.recv(1024).decode('utf-8')
                # Process game moves here
                self.broadcast(message)
            except:
                self.clients.remove(client)
                client.close()
                break

    def start(self):
        print("Server started. Waiting for connections...")
        while len(self.clients) < 2:
            client, address = self.server.accept()
            self.clients.append(client)
            thread = threading.Thread(target=self.handle_client, args=(client,))
            thread.start()
