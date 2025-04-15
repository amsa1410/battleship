import socket
import threading
from game_logic import BattleshipGame

class BattleshipServer:
    def __init__(self, host='localhost', port=5555):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients = []
        self.game = BattleshipGame()
        
    def broadcast(self, message):
        for client in self.clients:
            client.send(message.encode('utf-8'))

    def handle_client(self, client):
        while True:
            try:
                message = client.recv(1024).decode('utf-8')
                if msg.startswith("MOVE:"):
                    x, y = map(int, msg.split(":")[1].split(","))
                    result = self.game.make_move(x, y)
                    self.broadcast(f"RESULT:{x},{y},{result}")
            except:
                self.clients.remove(client)
                client.close()
                break

    def start(self, host='localhost', port=5555):
        print("Server started. Waiting for connections...")
        self.server.bind((host, port))
        self.server.listen(2)
        print(f"Server running on {host}:{port}")
        while len(self.clients) < 2:
            client, address = self.server.accept()
            self.clients.append(client)
            threading.Thread(target=self.handle_client, args=(client,)).start()


if __name__ == "__main__":
    server = Server()
    server.start()            
