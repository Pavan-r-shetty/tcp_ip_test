import socket
import numpy as np
from collections import deque
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def receive_full_message(sock):
    # First, read the 4-byte length header
    header = sock.recv(4)
    if not header:
        return None  # Connection closed

    message_length = int.from_bytes(header, 'big')

    # Now, read the message
    chunks = []
    bytes_recd = 0
    while bytes_recd < message_length:
        chunk = sock.recv(min(message_length - bytes_recd, 2048))
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd += len(chunk)

    return b''.join(chunks).decode('utf-8')

def start_server(model):
    # Initialize server
    IP = '169.254.103.247'
    PORT = 10000
    BUFFER_SIZE = 1024

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((IP, PORT))
    server_socket.listen(1)
    print(f"Server started, listening on {IP}:{PORT}")

    client_socket, addr = server_socket.accept()
    print(f"Accepted connection from {addr}")

    data_queue = deque(maxlen=80)

    try:
        while True:
            # Receive data from Raspberry Pi
            message = receive_full_message(client_socket)
            if not message:
                print("Connection closed by client.")
                break
            data = np.array(list(map(float, message.split(","))))

            data_queue.append(data)
            
            if len(data_queue) == 80:
                # Reshape to match model input shape
                model_input = np.array(data_queue).reshape(1, 80, 8).astype(np.float32)
                
                # Predict using ModelRT
              

                pred1, pred2 = model.predict(model_input)
                # print(pred1, pred2)
                
                # Send predictions back to Raspberry Pi
                prediction_message = f"{pred1[0][0]},{pred2[0][0]}"
                header = len(prediction_message).to_bytes(4, 'big')
                client_socket.send(header + prediction_message.encode('utf-8'))


    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        client_socket.close()
        server_socket.close()


class ModelRT:
    def __init__(self, m_file):
        self.m_file = m_file
        self.input_shape = DATA_SHAPE
        self.init_model()

    def init_model(self):
        # Load model and set up engine
        f = open(self.m_file, "rb")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device memory
        model_input = np.ones((1,) + self.input_shape, dtype=np.float32)
        self.output = np.empty([1, self.input_shape[0]], dtype=np.float32)
        self.output2 = np.empty([1, self.input_shape[0]], dtype=np.float32)
        self.d_input = cuda.mem_alloc(1 * model_input.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.d_output2 = cuda.mem_alloc(1 * self.output2.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output), int(self.d_output2)]

        # Create stream to transfer data between CPU and GPU
        self.stream = cuda.Stream()

    def predict(self, model_input):
        cuda.memcpy_htod_async(self.d_input, model_input, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        cuda.memcpy_dtoh_async(self.output2, self.d_output2, self.stream)
        self.stream.synchronize()
        return self.output, self.output2

if __name__ == "__main__":
    # Initialize ModelRT
    # Constants
    BUFFER_SIZE = 1024
    DATA_SHAPE = (80, 8)
    MODEL_FILE_PATH = "model.trt"
    model = ModelRT(MODEL_FILE_PATH)
    
    # Start the server
    start_server(model)
