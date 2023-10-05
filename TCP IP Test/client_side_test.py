import socket
import time
import numpy as np  # If you generate data using numpy

def send_full_message(sock, message):
    msg = message.encode('utf-8')
    header = len(msg).to_bytes(4, 'big')
    sock.sendall(header + msg)

IP = '169.254.103.247'
PORT = 10000
BUFFER_SIZE = 1024
DATA_SEND_COUNT_BEFORE_PREDICTION = 80

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))

send_counter = 0

try:
    while True:
        for _ in range(DATA_SEND_COUNT_BEFORE_PREDICTION):
            # Generate or acquire your 8 channels of data here
            data = np.random.randn(1, 8)  # Generating random data for testing of shape 1x8
            flat_data = data.flatten()  # Convert the 1x8 data to a flat array
            message = ",".join(map(str, flat_data))
            send_full_message(client_socket, message)
            send_counter += 1
            print(f"Sent data {send_counter}")
            time.sleep(0.005)
        
        # After sending data 80 times, expect to receive a prediction
        header = client_socket.recv(4)
        message_length = int.from_bytes(header, 'big')
        predictions = client_socket.recv(message_length).decode('utf-8')
        pred1, pred2 = map(float, predictions.split(","))
        
        # Print or do something with the predictions
        print("Received predictions:", pred1, pred2)

except KeyboardInterrupt:
    print("Shutting down...")
finally:
    client_socket.close()

