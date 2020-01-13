import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


model = YOLO(
    model_path="model_data/yolo.h5",
    anchors_path="model_data/yolo_anchors.txt",
    classes_path="model_data/coco_classes.txt"
)

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    img = cv2.resize(frame, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = model.detect_image(img)
    img = np.asarray(img)

    cv2.imshow("frame", frame)
    cv2.imshow("image", img)
    if cv2.waitKey(1) in [ord('q'), 27]:
        break
camera.release()
cv2.destroyAllWindows()
