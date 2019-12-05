import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import tensorflow as tf


model = YOLO(model_path="model_data/my_yolo.h5", classes_path="model_data/voc_classes.txt")
graph = tf.get_default_graph()


while True:
    frame = cv2.imread("../MyTelloPy/frame1.jpg", cv2.IMREAD_COLOR)
    if frame is None:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    with graph.as_default():
        img = model.detect_image(img)
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("../MyTelloPy/frame2.jpg", img)

    if cv2.waitKey(200) == 27:
        break
