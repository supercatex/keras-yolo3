import cv2
import numpy as np
from PIL import Image
from yolo import YOLO


model = YOLO(
    model_path="model_data/my_yolo.h5",
    anchors_path="model_data/yolo_anchors.txt",
    classes_path="model_data/voc_classes.txt"
)

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    img = Image.fromarray(frame)
    img = model.detect_image(img)
    img = np.asarray(img)

    cv2.imshow("frame", frame)
    cv2.imshow("image", img)
    if cv2.waitKey(1) in [ord('q'), 27]:
        break
camera.release()
cv2.destroyAllWindows()
