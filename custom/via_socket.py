import socket
import time
import cv2
import pickle
import struct
from yolo import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image


model = YOLO(model_path="model_data/my_yolo.h5", classes_path="model_data/voc_classes.txt")
graph = tf.get_default_graph()

HOST = "127.0.0.1"
PORT = 8889

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(10)

while True:
    try:
        conn, address = server_socket.accept()
        with conn:
            print("Connected by", address)
            conn.send("connected".encode("UTF-8"))

            while True:
                data = b""
                payload_size = struct.calcsize(">L")
                while len(data) < payload_size:
                    data += conn.recv(4096)
                    if not data:
                        break
                if not data:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]
                while len(data) < msg_size:
                    data += conn.recv(4096)
                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                with graph.as_default():
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = model.detect_image(img)
                    img = np.asarray(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    success, img = cv2.imencode(".jpg", img)
                    data = pickle.dumps(img, 0)
                    size = len(data)
                    conn.sendall(struct.pack(">L", size) + data)
                    print("JPEG image sent.")
    except Exception as e:
        print(e)
    finally:
        print("Closing server socket...")
        server_socket.close()
print("END")
