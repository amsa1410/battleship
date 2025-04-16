import socket
import threading

HOST = '192.168.207.30'  
PORT = 12345         

def gerer_client(conn, addr):
    print(f"Connecté à {addr}")

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break  

            reponse = data.decode('utf-8').upper().encode('utf-8')
            conn.sendall(reponse)

    finally:
        conn.close()
        print(f"Déconnexion de {addr}")

def main():
    # Créer le socket du serveur
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))  # Associer le socket à l'adresse et au port
        s.listen()             # Écouter les connexions entrantes

        print(f"Serveur en écoute sur {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()  # Accepter une nouvelle connexion
            # Créer un nouveau thread pour gérer le client
            thread = threading.Thread(target=gerer_client, args=(conn, addr))
            thread.start()

if __name__ == "__main__":
    main()