import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from flask import Flask
import tensorflow as tf


app = Flask(__name__)


@app.before_first_request
def load_model():
    app.model = YOLO(model_path="model_data/my_yolo.h5", classes_path="model_data/voc_classes.txt")
    app.graph = tf.get_default_graph()


@app.route("/")
def index():
    model = app.model
    graph = app.graph

    frame = cv2.imread("../MyTelloPy/frame1.jpg", cv2.IMREAD_COLOR)
    img = Image.fromarray(frame)
    with graph.as_default():
        img = model.detect_image(img)
        img = np.asarray(img)

        cv2.imwrite("../MyTelloPy/frame2.jpg", img)
        return "ok"
    return "error"




if __name__ == "__main__":
    app.run(debug=True)
